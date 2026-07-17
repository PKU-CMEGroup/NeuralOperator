"""Train fixed-step 1D Euler target heads on collaborator data.

This script is the light pilot harness for the solver-facing target ladder:

* conservative_state: direct next conservative state
* residual: conservative state residual over one fixed coarse step
* projected_residual: residual projected to a learned net boundary exchange
* flux: predicted face flux, followed by a fixed conservative FV update
* physical_flux_correction: base Rusanov flux plus bounded learned correction
* interface: predicted face states, followed by Rusanov flux and FV update
* relative_interface: directed relative traces followed by a shared face flux

The model is not conditioned on ``dt``. ``--step-stride`` selects a fixed
coarse-step operator from saved trajectory frames. Face-flux heads can be
supervised through the next state, the solver-exported time-averaged face flux,
or both.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utility.time_dependent_no.euler1d import (
    Euler1DBatch,
    conservative_to_primitive,
    make_euler1d_batch,
    primitive_to_conservative,
)
from utility.time_dependent_no.euler1d_data import (
    Euler1DNPZ,
    Euler1DRolloutWindowDataset,
    Euler1DTimePairDataset,
    collate_euler1d_pairs,
    collate_euler1d_rollout_windows,
    load_euler1d_npz,
)
from utility.time_dependent_no.euler1d_models import (
    CPGNetEuler1D,
    Euler1DCoordinates,
    Euler1DTarget,
    FNOEuler1DHead,
)
from utility.time_dependent_no.euler1d_targets import (
    canonicalize_owner_oriented_face_flux,
    make_target_adapter,
)


EPS = 1.0e-12
NEAR_FLOOR = 1.0e-6
LIMITER_ACTIVE_TOL = 1.0e-6
CORRECTION_SATURATION_TOL = 0.95
MODEL_CHOICES = ("cpgnet", "fno")
TARGET_CHOICES = (
    "conservative_state",
    "residual",
    "projected_residual",
    "primitive_residual",
    "limited_residual",
    "flux",
    "limited_flux",
    "physical_flux_correction",
    "interface",
    "relative_interface",
    "positive_limited_interface",
)
CPG_TARGET = "cpg_interface"
ARG_TARGET_CHOICES = ("state", *TARGET_CHOICES, CPG_TARGET)
COORDINATE_CHOICES = ("primitive", "conservative")
INPUT_NORMALIZATION_CHOICES = ("none", "fixed_physical", "empirical")
LOSS_NORMALIZATION_CHOICES = ("fixed_physical", "empirical")
TARGET_SUPERVISION_CHOICES = ("state", "direct_flux", "joint")
FLUX_GAUGE_MODE_CHOICES = ("raw", "canonical")
INTERFACE_FLUX_MODE_CHOICES = ("rusanov", "central")
UNROLL_BURN_IN_MODE_CHOICES = ("generated", "teacher")


@dataclass(frozen=True)
class PrimitiveNormalizer:
    mean: torch.Tensor
    std: torch.Tensor
    coordinates: str = "primitive"
    normalization: str = "empirical"

    @classmethod
    def from_states(
        cls,
        primitive: np.ndarray,
        *,
        gamma: float,
        coordinates: str,
        normalization: str,
    ) -> "PrimitiveNormalizer":
        if coordinates not in COORDINATE_CHOICES:
            raise ValueError(f"unsupported coordinates: {coordinates}")
        if normalization not in INPUT_NORMALIZATION_CHOICES:
            raise ValueError(f"unsupported normalization: {normalization}")
        states = np.asarray(primitive)
        if coordinates == "conservative":
            states = primitive_to_conservative_np(states, gamma)
        if normalization == "empirical":
            mean_np = states.mean(axis=(0, 1, 2))
            std_np = np.maximum(states.std(axis=(0, 1, 2)), 1.0e-6)
        else:
            mean_np = np.zeros(3, dtype=np.float64)
            std_np = np.ones(3, dtype=np.float64)
            if normalization == "fixed_physical" and coordinates == "conservative":
                std_np[2] = 1.0 / (gamma - 1.0)
        return cls(
            mean=torch.as_tensor(mean_np, dtype=torch.float32).reshape(1, 1, 3),
            std=torch.as_tensor(std_np, dtype=torch.float32).reshape(1, 1, 3),
            coordinates=coordinates,
            normalization=normalization,
        )

    @classmethod
    def from_source(
        cls,
        source: Euler1DNPZ,
        case_indices: np.ndarray,
        *,
        step_stride: int = 1,
        coordinates: str = "primitive",
        normalization: str = "empirical",
    ) -> "PrimitiveNormalizer":
        if step_stride < 1 or step_stride >= source.num_frames:
            raise ValueError("step_stride must be in [1, num_frames - 1]")
        targets = source.data[case_indices, step_stride:]
        return cls.from_states(
            targets,
            gamma=source.gamma,
            coordinates=coordinates,
            normalization=normalization,
        )

    @classmethod
    def from_source_inputs(
        cls,
        source: Euler1DNPZ,
        case_indices: np.ndarray,
        *,
        step_stride: int = 1,
        coordinates: str = "primitive",
        normalization: str = "empirical",
    ) -> "PrimitiveNormalizer":
        if step_stride < 1 or step_stride >= source.num_frames:
            raise ValueError("step_stride must be in [1, num_frames - 1]")
        inputs = source.data[case_indices, :-step_stride]
        return cls.from_states(
            inputs,
            gamma=source.gamma,
            coordinates=coordinates,
            normalization=normalization,
        )

    @classmethod
    def from_source_face_flux(
        cls,
        source: Euler1DNPZ,
        case_indices: np.ndarray,
        *,
        step_stride: int,
        normalization: str,
    ) -> "PrimitiveNormalizer":
        """Fit component scales for solver-exported macro-step face fluxes."""

        if source.face_flux_integral is None:
            raise ValueError(
                "direct face-flux supervision requires face_flux_integral in the dataset"
            )
        if step_stride < 1 or step_stride >= source.num_frames:
            raise ValueError("step_stride must be in [1, num_frames - 1]")
        if normalization not in LOSS_NORMALIZATION_CHOICES:
            raise ValueError(f"unsupported normalization: {normalization}")
        if normalization == "fixed_physical":
            return cls(
                mean=torch.zeros(1, 1, 3),
                std=torch.tensor([[[1.0, 1.0, 1.0 / (source.gamma - 1.0)]]]),
                coordinates="face_flux",
                normalization=normalization,
            )

        channel_sum = np.zeros(3, dtype=np.float64)
        channel_square_sum = np.zeros(3, dtype=np.float64)
        count = 0
        for case in np.asarray(case_indices, dtype=np.int64):
            impulses = source.face_flux_integral[case].astype(np.float64, copy=False)
            prefix = np.concatenate(
                (
                    np.zeros((1, impulses.shape[1], 3), dtype=np.float64),
                    np.cumsum(impulses, axis=0),
                ),
                axis=0,
            )
            macro_impulse = prefix[step_stride:] - prefix[:-step_stride]
            macro_dt = (
                source.t[case, step_stride:] - source.t[case, :-step_stride]
            ).astype(np.float64)
            macro_flux = macro_impulse / macro_dt[:, None, None]
            channel_sum += macro_flux.sum(axis=(0, 1))
            channel_square_sum += np.square(macro_flux).sum(axis=(0, 1))
            count += int(np.prod(macro_flux.shape[:-1]))
        if count == 0:
            raise ValueError("cannot fit face-flux normalizer from an empty case split")
        mean = channel_sum / count
        variance = np.maximum(channel_square_sum / count - np.square(mean), 0.0)
        std = np.maximum(np.sqrt(variance), 1.0e-6)
        return cls(
            mean=torch.as_tensor(mean, dtype=torch.float32).reshape(1, 1, 3),
            std=torch.as_tensor(std, dtype=torch.float32).reshape(1, 1, 3),
            coordinates="face_flux",
            normalization=normalization,
        )

    def to(self, device: torch.device) -> "PrimitiveNormalizer":
        return PrimitiveNormalizer(
            mean=self.mean.to(device),
            std=self.std.to(device),
            coordinates=self.coordinates,
            normalization=self.normalization,
        )

    @classmethod
    def from_source_boundary_exchange(
        cls,
        source: Euler1DNPZ,
        case_indices: np.ndarray,
        *,
        step_stride: int,
    ) -> "PrimitiveNormalizer":
        """Fit zero-centered RMS scales for net solver boundary impulses."""

        if source.face_flux_integral is None:
            raise ValueError(
                "boundary-exchange supervision requires face_flux_integral "
                "in the dataset"
            )
        if step_stride < 1 or step_stride >= source.num_frames:
            raise ValueError("step_stride must be in [1, num_frames - 1]")

        channel_square_sum = np.zeros(3, dtype=np.float64)
        count = 0
        for case in np.asarray(case_indices, dtype=np.int64):
            impulses = source.face_flux_integral[case].astype(np.float64, copy=False)
            prefix = np.concatenate(
                (
                    np.zeros((1, impulses.shape[1], 3), dtype=np.float64),
                    np.cumsum(impulses, axis=0),
                ),
                axis=0,
            )
            macro_impulse = prefix[step_stride:] - prefix[:-step_stride]
            boundary_exchange = -(macro_impulse[:, 0] + macro_impulse[:, -1])
            channel_square_sum += np.square(boundary_exchange).sum(axis=0)
            count += int(boundary_exchange.shape[0])
        if count == 0:
            raise ValueError(
                "cannot fit boundary-exchange normalizer from an empty case split"
            )
        rms = np.maximum(np.sqrt(channel_square_sum / count), 1.0e-8)
        return cls(
            mean=torch.zeros(1, 1, 3),
            std=torch.as_tensor(rms, dtype=torch.float32).reshape(1, 1, 3),
            coordinates="boundary_exchange",
            normalization="empirical",
        )

    def mse(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction_norm = (prediction - self.mean) / self.std
        target_norm = (target - self.mean) / self.std
        return torch.mean((prediction_norm - target_norm).square())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    date = datetime.now().strftime("%Y%m%d")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--boundary-exchange-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Weight on RMS-normalized net boundary-impulse supervision for "
            "the projected-residual target."
        ),
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "artifacts"
        / "time_dependent_no"
        / f"euler1d_target_ladder_{date}",
    )
    parser.add_argument("--model", choices=(*MODEL_CHOICES, "all"), default="all")
    parser.add_argument("--target", choices=(*ARG_TARGET_CHOICES, "all"), default="all")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument(
        "--unroll-epochs",
        type=int,
        default=0,
        help="Autoregressive fine-tuning epochs after one-step training.",
    )
    parser.add_argument(
        "--unroll-steps",
        type=int,
        default=3,
        help="Number of differentiable raw rollout steps during fine-tuning.",
    )
    parser.add_argument(
        "--unroll-burn-in-steps",
        type=int,
        default=0,
        help=(
            "No-gradient model-generated steps before each differentiable "
            "unroll; the resulting recurrent state is detached."
        ),
    )
    parser.add_argument(
        "--unroll-burn-in-mode",
        choices=UNROLL_BURN_IN_MODE_CHOICES,
        default="generated",
        help=(
            "Prefix-state source before the differentiable unroll. 'generated' "
            "uses detached model rollout; 'teacher' starts from the exact state "
            "at the same time offset as a matched sampling control."
        ),
    )
    parser.add_argument(
        "--unroll-lr-factor",
        type=float,
        default=0.1,
        help="Learning-rate multiplier for autoregressive fine-tuning.",
    )
    parser.add_argument(
        "--unroll-noise-factor",
        type=float,
        default=0.1,
        help="Input-noise multiplier for the first unrolled state.",
    )
    parser.add_argument(
        "--unroll-admissibility-weight",
        type=float,
        default=0.0,
        help=(
            "Training-only weight on a smooth density/internal-energy barrier "
            "during autoregressive fine-tuning; inference remains unchanged."
        ),
    )
    parser.add_argument(
        "--unroll-admissibility-margin-fraction",
        type=float,
        default=0.5,
        help=(
            "Density and pressure safety margins as a fraction of the minimum "
            "truth value in the training trajectories."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Default: 1e-4 for CPGNet and 1e-3 for FNO.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Default: 0 for CPGNet and 1e-5 for FNO.",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--flux-correction-scale",
        type=float,
        default=1.0,
        help="Multiplier for bounded physical_flux_correction outputs.",
    )
    parser.add_argument(
        "--flux-correction-scale-floor",
        type=float,
        default=1.0e-6,
        help="Minimum componentwise physical flux scale for correction bounds.",
    )
    parser.add_argument(
        "--input-noise-std",
        type=float,
        default=0.0,
        help=(
            "Training-only primitive input noise. Density and pressure receive "
            "multiplicative log-normal noise; velocity receives additive Gaussian noise."
        ),
    )
    parser.add_argument(
        "--initial-frame-weight",
        type=float,
        default=1.0,
        help=(
            "Sampling weight for trajectory windows beginning at frame zero. "
            "The default 1 preserves uniform temporal sampling."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Train/validation/test split seed. Defaults to --seed.",
    )
    parser.add_argument("--train-cases", type=int, default=384)
    parser.add_argument("--val-cases", type=int, default=64)
    parser.add_argument("--test-cases", type=int, default=64)
    parser.add_argument("--rollout-steps", type=int, default=20)
    parser.add_argument(
        "--rollout-final-frame",
        type=int,
        default=80,
        help=(
            "Evaluate all strides to the same saved-frame horizon. "
            "Overrides --rollout-steps and must be divisible by --step-stride."
        ),
    )
    parser.add_argument(
        "--step-stride",
        type=int,
        default=4,
        help="Saved-frame stride for the fixed coarse-step operator.",
    )
    parser.add_argument(
        "--input-coordinates",
        choices=COORDINATE_CHOICES,
        default="primitive",
        help="State coordinates presented to the FNO backbone.",
    )
    parser.add_argument(
        "--loss-coordinates",
        choices=COORDINATE_CHOICES,
        default="primitive",
        help="Coordinates used by the standardized next-state training loss.",
    )
    parser.add_argument(
        "--recurrent-coordinates",
        choices=COORDINATE_CHOICES,
        default="primitive",
        help="Coordinates retained between raw autoregressive rollout calls.",
    )
    parser.add_argument(
        "--input-normalization",
        choices=INPUT_NORMALIZATION_CHOICES,
        default="none",
        help="Normalization applied to the selected FNO state coordinates.",
    )
    parser.add_argument(
        "--loss-normalization",
        choices=LOSS_NORMALIZATION_CHOICES,
        default="empirical",
        help="Normalization used to weight the selected next-state loss coordinates.",
    )
    parser.add_argument(
        "--target-supervision",
        choices=TARGET_SUPERVISION_CHOICES,
        default="state",
        help=(
            "Training loss for a face-flux head: decoded next state, direct "
            "solver-exported time-averaged flux, or their joint loss."
        ),
    )
    parser.add_argument(
        "--flux-loss-normalization",
        choices=LOSS_NORMALIZATION_CHOICES,
        default="empirical",
        help="Component normalization for direct face-flux loss.",
    )
    parser.add_argument(
        "--flux-loss-weight",
        type=float,
        default=1.0,
        help="Multiplier on normalized direct flux MSE in joint supervision.",
    )
    parser.add_argument(
        "--flux-gauge-mode",
        choices=FLUX_GAUGE_MODE_CHOICES,
        default="raw",
        help=(
            "Raw solver flux or its decoder-equivalent canonical representative "
            "with the constant-physical-flux null mode removed."
        ),
    )
    parser.add_argument(
        "--interface-flux-mode",
        choices=INTERFACE_FLUX_MODE_CHOICES,
        default="rusanov",
        help="Shared face-flux decoder for the relative-interface target.",
    )
    parser.add_argument(
        "--positive-transform",
        choices=("none", "softplus", "exp"),
        default="softplus",
        help="Primitive positivity transform for state and interface target heads.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--cpg-hidden-dim", type=int, default=128)
    parser.add_argument("--cpg-message-passing-steps", type=int, default=12)
    parser.add_argument("--cpg-mlp-layers", type=int, default=3)
    parser.add_argument("--fno-width", type=int, default=64)
    parser.add_argument("--fno-modes", type=int, default=16)
    parser.add_argument("--fno-layers", type=int, default=4)
    parser.add_argument("--fno-fc-dim", type=int, default=128)
    parser.add_argument("--fno-pad-ratio", type=float, default=0.0)
    parser.add_argument(
        "--save-checkpoints",
        action="store_true",
        help="Save final model checkpoints under the run output directory.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--continuation-checkpoint",
        type=Path,
        default=None,
        help=(
            "Initialize model weights from a compatible smaller-stride "
            "checkpoint; optimizer and scheduler state are not loaded."
        ),
    )
    args = parser.parse_args(argv)
    if args.split_seed is None:
        args.split_seed = args.seed
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(args: argparse.Namespace) -> torch.device:
    if args.device == "cpu":
        return torch.device("cpu")
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(args.gpu)
        return torch.device("cuda")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        return torch.device("cuda")
    return torch.device("cpu")


def split_cases(
    source: Euler1DNPZ, args: argparse.Namespace
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if args.train_cases < 1:
        raise ValueError("--train-cases must be >= 1")
    if args.val_cases < 1:
        raise ValueError("--val-cases must be >= 1")
    if args.test_cases < 1:
        raise ValueError("--test-cases must be >= 1")
    if source.num_cases < 3:
        raise ValueError("at least three cases are required for train/val/test split")

    requested = args.train_cases + args.val_cases + args.test_cases
    if requested > source.num_cases:
        raise ValueError(
            f"requested {requested} cases but dataset has only {source.num_cases}; "
            "regenerate more cases or lower --train-cases/--val-cases/--test-cases"
        )
    rng = np.random.default_rng(args.split_seed)
    permutation = rng.permutation(source.num_cases)
    train_end = args.train_cases
    val_end = train_end + args.val_cases
    train = np.sort(permutation[:train_end]).astype(np.int64)
    val = np.sort(permutation[train_end:val_end]).astype(np.int64)
    test = np.sort(permutation[val_end:requested]).astype(np.int64)
    return train, val, test


def make_initial_frame_sampler(
    items: Sequence[tuple[int, int]],
    *,
    initial_frame_weight: float,
    generator: torch.Generator,
) -> WeightedRandomSampler | None:
    """Optionally upweight the rollout initial condition without changing epoch size."""

    if initial_frame_weight == 1.0:
        return None
    weights = torch.ones(len(items), dtype=torch.float64)
    for index, (_, start_frame) in enumerate(items):
        if start_frame == 0:
            weights[index] = initial_frame_weight
    return WeightedRandomSampler(
        weights,
        num_samples=len(items),
        replacement=True,
        generator=generator,
    )


def build_model(
    model_name: str,
    target: Euler1DTarget,
    args: argparse.Namespace,
    normalizer: PrimitiveNormalizer | None = None,
) -> nn.Module:
    if model_name == "cpgnet":
        if target != CPG_TARGET:
            raise ValueError(
                "--model cpgnet requires --target cpg_interface; "
                "the residual target head is a deprecated pilot, not CPGNet"
            )
        if normalizer is None:
            raise ValueError("CPGNet requires training-set primitive statistics")
        model = CPGNetEuler1D(
            hidden_dim=args.cpg_hidden_dim,
            message_passing_steps=args.cpg_message_passing_steps,
            mlp_layers=args.cpg_mlp_layers,
            primitive_mean=normalizer.mean,
            primitive_std=normalizer.std,
        )
    elif model_name == "fno":
        if args.fno_layers < 2:
            raise ValueError("--fno-layers must be >= 2")
        model = FNOEuler1DHead(
            target,
            modes=[args.fno_modes] * (args.fno_layers - 1),
            width=args.fno_width,
            layers=[args.fno_width] * args.fno_layers,
            fc_dim=args.fno_fc_dim,
            pad_ratio=args.fno_pad_ratio,
            input_coordinates=cast(
                Euler1DCoordinates,
                getattr(args, "input_coordinates", "primitive"),
            ),
            input_mean=None if normalizer is None else normalizer.mean,
            input_std=None if normalizer is None else normalizer.std,
        )
    else:
        raise ValueError(f"unsupported model: {model_name}")

    if target in (
        "residual",
        "projected_residual",
        "primitive_residual",
        "limited_residual",
        "flux",
        "limited_flux",
        "physical_flux_correction",
        "relative_interface",
    ):
        zero_initialize_identity_update_output(model, target)
    return model


def zero_initialize_identity_update_output(
    model: nn.Module, target: Euler1DTarget
) -> None:
    """Start an FNO update head at the identity operator."""

    if not isinstance(model, FNOEuler1DHead):
        raise TypeError(
            f"zero initialization for {target} is unsupported for "
            f"{type(model).__name__}"
        )
    nn.init.zeros_(model.model.fc2.weight)
    if model.model.fc2.bias is not None:
        nn.init.zeros_(model.model.fc2.bias)


def load_continuation_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    *,
    model_name: str,
    target_name: str,
    args: argparse.Namespace,
    train_cases: np.ndarray,
    val_cases: np.ndarray,
    test_cases: np.ndarray,
) -> dict[str, Any]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("continuation checkpoint must contain a mapping")
    if checkpoint.get("model") != model_name:
        raise ValueError("continuation checkpoint model does not match")
    if checkpoint.get("target") != target_name:
        raise ValueError("continuation checkpoint target does not match")

    source_args = checkpoint.get("args")
    if not isinstance(source_args, dict):
        raise ValueError("continuation checkpoint is missing training arguments")
    contract_fields = (
        "fno_width",
        "fno_modes",
        "fno_layers",
        "fno_fc_dim",
        "fno_pad_ratio",
        "input_coordinates",
        "input_normalization",
        "loss_coordinates",
        "loss_normalization",
        "recurrent_coordinates",
        "positive_transform",
        "target_supervision",
        "flux_gauge_mode",
        "interface_flux_mode",
        "input_noise_std",
        "initial_frame_weight",
        "boundary_exchange_loss_weight",
        "unroll_admissibility_weight",
    )
    legacy_defaults = {
        "interface_flux_mode": "rusanov",
        "initial_frame_weight": 1.0,
        "boundary_exchange_loss_weight": 0.0,
    }
    for field in contract_fields:
        source_value = source_args.get(field, legacy_defaults.get(field))
        current_value = getattr(args, field)
        if source_value != current_value:
            raise ValueError(
                f"continuation checkpoint {field}={source_value!r} "
                f"does not match current value {current_value!r}"
            )

    source_stride = int(source_args.get("step_stride", 0))
    if source_stride < 1 or source_stride >= args.step_stride:
        raise ValueError(
            "continuation checkpoint must use a smaller positive step_stride"
        )
    for name, expected in (
        ("train_cases", train_cases),
        ("val_cases", val_cases),
        ("test_cases", test_cases),
    ):
        actual = np.asarray(checkpoint.get(name), dtype=np.int64)
        if not np.array_equal(actual, expected):
            raise ValueError(f"continuation checkpoint {name} do not match")

    source_state = checkpoint.get("model_state_dict")
    if not isinstance(source_state, dict):
        raise ValueError("continuation checkpoint is missing model_state_dict")
    current_state = model.state_dict()
    for name in ("input_mean", "input_std"):
        if name in source_state and not torch.equal(
            source_state[name].detach().cpu(),
            current_state[name].detach().cpu(),
        ):
            raise ValueError(
                "continuation checkpoint input normalization does not match"
            )
    model.load_state_dict(source_state, strict=True)
    return {
        "checkpoint": checkpoint_path,
        "source_step_stride": source_stride,
        "source_best_epoch": checkpoint.get("best_epoch"),
        "source_seed": source_args.get("seed"),
        "source_split_seed": source_args.get("split_seed"),
        "optimizer_state_loaded": False,
    }


def relative_l2_torch(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    batch_size = prediction.shape[0]
    diff = (prediction - target).reshape(batch_size, -1)
    denom = target.reshape(batch_size, -1).norm(dim=1).clamp_min(EPS)
    return diff.norm(dim=1) / denom


def finite_scalar(value: torch.Tensor) -> float:
    if value.numel() != 1:
        raise ValueError("expected a scalar tensor")
    item = float(value.detach().cpu().item())
    return item if math.isfinite(item) else float("nan")


def new_limiter_accumulator() -> dict[str, float]:
    return {
        "theta_sum": 0.0,
        "theta_count": 0.0,
        "theta_min": float("inf"),
        "activation_count": 0.0,
    }


def update_limiter_accumulator(
    accumulator: dict[str, float], aux: dict[str, Any]
) -> None:
    theta = aux.get("limiter_theta")
    if theta is None:
        return
    theta_cpu = theta.detach().cpu()
    accumulator["theta_sum"] += float(theta_cpu.sum().item())
    accumulator["theta_count"] += float(theta_cpu.numel())
    accumulator["theta_min"] = min(
        accumulator["theta_min"], float(theta_cpu.min().item())
    )
    accumulator["activation_count"] += float(
        theta_cpu.lt(1.0 - LIMITER_ACTIVE_TOL).sum().item()
    )


def finalize_limiter_accumulator(accumulator: dict[str, float]) -> dict[str, float]:
    count = accumulator["theta_count"]
    if count <= 0.0:
        return {
            "limiter_theta_mean": float("nan"),
            "limiter_theta_min": float("nan"),
            "limiter_activation_fraction": float("nan"),
        }
    return {
        "limiter_theta_mean": accumulator["theta_sum"] / count,
        "limiter_theta_min": accumulator["theta_min"],
        "limiter_activation_fraction": accumulator["activation_count"] / count,
    }


def new_flux_correction_accumulator() -> dict[str, float]:
    return {
        "ratio_sum": 0.0,
        "ratio_count": 0.0,
        "ratio_max": 0.0,
        "saturation_count": 0.0,
    }


def update_flux_correction_accumulator(
    accumulator: dict[str, float], aux: dict[str, Any]
) -> None:
    correction = aux.get("flux_correction")
    bound = aux.get("flux_correction_bound")
    if correction is None or bound is None:
        return
    ratio = correction.detach().abs().cpu() / bound.detach().abs().cpu().clamp_min(EPS)
    accumulator["ratio_sum"] += float(ratio.sum().item())
    accumulator["ratio_count"] += float(ratio.numel())
    accumulator["ratio_max"] = max(accumulator["ratio_max"], float(ratio.max().item()))
    accumulator["saturation_count"] += float(
        ratio.ge(CORRECTION_SATURATION_TOL).sum().item()
    )


def finalize_flux_correction_accumulator(
    accumulator: dict[str, float],
) -> dict[str, float]:
    count = accumulator["ratio_count"]
    if count <= 0.0:
        return {
            "flux_correction_abs_over_bound_mean": float("nan"),
            "flux_correction_abs_over_bound_max": float("nan"),
            "flux_correction_saturation_fraction": float("nan"),
        }
    return {
        "flux_correction_abs_over_bound_mean": accumulator["ratio_sum"] / count,
        "flux_correction_abs_over_bound_max": accumulator["ratio_max"],
        "flux_correction_saturation_fraction": accumulator["saturation_count"] / count,
    }


def proposed_conservative(prediction: Any) -> torch.Tensor:
    """Return the pre-limiter proposed conservative state when available."""

    proposed = prediction.aux.get("proposed_conservative")
    return prediction.conservative if proposed is None else cast(torch.Tensor, proposed)


def state_pair_for_loss(
    prediction: Any,
    batch: Euler1DBatch,
    coordinates: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return predicted and clean target states in the requested coordinates."""

    if coordinates == "primitive":
        if batch.target_primitive is None:
            raise RuntimeError("batch is missing target_primitive")
        return prediction.primitive, batch.target_primitive
    if coordinates == "conservative":
        target = batch.target_conservative
        if target is None:
            raise RuntimeError("batch is missing target_conservative")
        return prediction.conservative, target
    raise ValueError(f"unsupported loss coordinates: {coordinates}")


def decoded_face_flux(raw: torch.Tensor, prediction: Any) -> torch.Tensor:
    """Return the face flux actually passed through the conservative decoder."""

    aux = getattr(prediction, "aux", {})
    face_flux = aux.get("face_flux")
    return face_flux if isinstance(face_flux, torch.Tensor) else raw


def face_flux_supervision_target(
    prediction: Any,
    batch: Euler1DBatch,
) -> torch.Tensor:
    """Return the raw or gauge-canonical solver flux selected by the adapter."""

    target = batch.target_face_flux
    if target is None:
        raise RuntimeError(
            "direct face-flux supervision requires target_face_flux in the batch"
        )
    aux = getattr(prediction, "aux", {})
    gauge_mode = aux.get("flux_gauge_mode", "raw")
    if gauge_mode == "raw":
        return target
    if gauge_mode == "canonical":
        return canonicalize_owner_oriented_face_flux(
            target,
            batch.geometry.face_normal,
        )
    raise ValueError(f"unsupported flux gauge mode: {gauge_mode}")


def solver_boundary_exchange_target(batch: Euler1DBatch) -> torch.Tensor:
    """Return the net conservative boundary exchange from solver face impulses."""

    target = batch.target_face_flux
    if target is None:
        raise RuntimeError(
            "boundary-exchange supervision requires target_face_flux in the batch"
        )
    boundary = batch.geometry.face_neighbor.lt(0).unsqueeze(-1)
    area = batch.geometry.face_area.to(dtype=target.dtype, device=target.device)
    dt = batch.dt.to(dtype=target.dtype, device=target.device).reshape(-1, 1, 1)
    return -(dt * area.unsqueeze(-1) * target * boundary).sum(dim=1)


def state_pair_boundary_exchange_target(
    current_primitive: torch.Tensor,
    target_primitive: torch.Tensor,
    batch: Euler1DBatch,
) -> torch.Tensor:
    """Return the endpoint-equivalent net conservative exchange."""

    current = primitive_to_conservative(current_primitive, gamma=batch.gamma)
    target = primitive_to_conservative(target_primitive, gamma=batch.gamma)
    volume = batch.geometry.cell_volume.to(dtype=target.dtype, device=target.device)
    return (volume.unsqueeze(-1) * (target - current)).sum(dim=1)


def predicted_boundary_exchange(prediction: Any) -> torch.Tensor:
    """Return the projected-residual boundary token or fail explicitly."""

    boundary_exchange = prediction.aux.get("boundary_exchange")
    if not isinstance(boundary_exchange, torch.Tensor):
        raise RuntimeError(
            "boundary-exchange supervision requires a projected-residual decoder"
        )
    return boundary_exchange


def supervised_loss(
    raw: torch.Tensor,
    prediction: Any,
    batch: Euler1DBatch,
    state_normalizer: PrimitiveNormalizer,
    *,
    loss_coordinates: str,
    target_supervision: str,
    flux_normalizer: PrimitiveNormalizer | None,
    flux_loss_weight: float,
    boundary_exchange_normalizer: PrimitiveNormalizer | None = None,
    boundary_exchange_loss_weight: float = 0.0,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Return total, decoded-state, and optional direct face-flux losses."""

    loss_prediction, loss_target = state_pair_for_loss(
        prediction,
        batch,
        loss_coordinates,
    )
    state_loss = state_normalizer.mse(loss_prediction, loss_target)
    boundary_exchange_loss = None
    if boundary_exchange_normalizer is not None:
        exchange_prediction = predicted_boundary_exchange(prediction)
        exchange_target = solver_boundary_exchange_target(batch)
        boundary_exchange_loss = boundary_exchange_normalizer.mse(
            exchange_prediction.unsqueeze(1),
            exchange_target.unsqueeze(1),
        )
    if boundary_exchange_loss_weight > 0.0 and boundary_exchange_loss is None:
        raise RuntimeError("boundary-exchange loss requires its target normalizer")
    boundary_penalty = (
        state_loss.new_zeros(())
        if boundary_exchange_loss is None or boundary_exchange_loss_weight == 0.0
        else boundary_exchange_loss_weight * boundary_exchange_loss
    )
    if target_supervision == "state":
        return state_loss + boundary_penalty, state_loss, None, boundary_exchange_loss
    if target_supervision not in ("direct_flux", "joint"):
        raise ValueError(f"unsupported target supervision: {target_supervision}")
    if flux_normalizer is None:
        raise RuntimeError("direct face-flux supervision requires a flux normalizer")
    flux_prediction = decoded_face_flux(raw, prediction)
    flux_target = face_flux_supervision_target(prediction, batch)
    if flux_prediction.shape != flux_target.shape:
        raise ValueError(
            "decoded face flux and supervision target shapes disagree: "
            f"{tuple(flux_prediction.shape)} != {tuple(flux_target.shape)}"
        )
    flux_loss = flux_normalizer.mse(flux_prediction, flux_target)
    if target_supervision == "direct_flux":
        return flux_loss, state_loss, flux_loss, boundary_exchange_loss
    return (
        state_loss + flux_loss_weight * flux_loss + boundary_penalty,
        state_loss,
        flux_loss,
        boundary_exchange_loss,
    )


def normalized_face_flux_error_decomposition(
    raw: torch.Tensor,
    batch: Euler1DBatch,
    normalizer: PrimitiveNormalizer,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split normalized face-flux error into decoder-active and null modes.

    The stored 1D face flux is owner-oriented. Its constant-physical-flux
    null mode is therefore proportional to the face normal
    ``[-1, +1, ..., +1]``, rather than to an all-ones stored face vector.
    """

    target = batch.target_face_flux
    if target is None:
        raise RuntimeError("face-flux diagnostics require target_face_flux")
    if raw.shape != target.shape:
        raise ValueError(
            "raw face flux and target_face_flux shapes disagree: "
            f"{tuple(raw.shape)} != {tuple(target.shape)}"
        )
    normal = batch.geometry.face_normal.to(dtype=raw.dtype, device=raw.device)
    if normal.shape != (*raw.shape[:-1], 1):
        raise ValueError(
            "face normals must have shape "
            f"{(*raw.shape[:-1], 1)}, got {tuple(normal.shape)}"
        )
    scale = normalizer.std.to(dtype=raw.dtype, device=raw.device)
    normalized_error = (raw - target) / scale
    null_norm_sq = normal.square().sum(dim=1, keepdim=True).clamp_min(EPS)
    null_coefficient = (normalized_error * normal).sum(
        dim=1, keepdim=True
    ) / null_norm_sq
    gauge_error = normal * null_coefficient
    active_error = normalized_error - gauge_error
    return (
        normalized_error.square().mean(),
        active_error.square().mean(),
        gauge_error.square().mean(),
    )


def new_conservative_safety_accumulator() -> dict[str, float]:
    return {
        "min_density": float("inf"),
        "min_pressure": float("inf"),
        "nonpositive_density": 0.0,
        "nonpositive_pressure": 0.0,
        "near_floor_density": 0.0,
        "near_floor_pressure": 0.0,
        "count": 0.0,
    }


def update_conservative_safety_accumulator(
    accumulator: dict[str, float],
    conservative: torch.Tensor,
    *,
    gamma: float,
) -> None:
    primitive = conservative_to_primitive(conservative, gamma=gamma)
    accumulator["min_density"] = min(
        accumulator["min_density"], float(primitive[..., 0].min().detach().cpu().item())
    )
    accumulator["min_pressure"] = min(
        accumulator["min_pressure"],
        float(primitive[..., 2].min().detach().cpu().item()),
    )
    accumulator["nonpositive_density"] += float(
        primitive[..., 0].le(0.0).sum().detach().cpu().item()
    )
    accumulator["nonpositive_pressure"] += float(
        primitive[..., 2].le(0.0).sum().detach().cpu().item()
    )
    accumulator["near_floor_density"] += float(
        primitive[..., 0].le(NEAR_FLOOR).sum().detach().cpu().item()
    )
    accumulator["near_floor_pressure"] += float(
        primitive[..., 2].le(NEAR_FLOOR).sum().detach().cpu().item()
    )
    accumulator["count"] += float(primitive[..., 0].numel())


def finalize_conservative_safety_accumulator(
    accumulator: dict[str, float],
    *,
    prefix: str,
) -> dict[str, float | int]:
    if accumulator["count"] <= 0.0:
        return {
            f"{prefix}_min_density": float("nan"),
            f"{prefix}_min_pressure": float("nan"),
            f"num_nonpositive_{prefix}_density": 0,
            f"num_nonpositive_{prefix}_pressure": 0,
            f"num_{prefix}_density_near_floor": 0,
            f"num_{prefix}_pressure_near_floor": 0,
        }
    return {
        f"{prefix}_min_density": accumulator["min_density"],
        f"{prefix}_min_pressure": accumulator["min_pressure"],
        f"num_nonpositive_{prefix}_density": int(accumulator["nonpositive_density"]),
        f"num_nonpositive_{prefix}_pressure": int(accumulator["nonpositive_pressure"]),
        f"num_{prefix}_density_near_floor": int(accumulator["near_floor_density"]),
        f"num_{prefix}_pressure_near_floor": int(accumulator["near_floor_pressure"]),
    }


def proposed_safety_metrics(accumulator: dict[str, float]) -> dict[str, float | int]:
    metrics = finalize_conservative_safety_accumulator(
        accumulator,
        prefix="proposed",
    )
    metrics.update(
        {
            "raw_min_density": metrics["proposed_min_density"],
            "raw_min_pressure": metrics["proposed_min_pressure"],
            "num_nonpositive_raw_density": metrics["num_nonpositive_proposed_density"],
            "num_nonpositive_raw_pressure": metrics[
                "num_nonpositive_proposed_pressure"
            ],
            "num_raw_density_near_floor": metrics["num_proposed_density_near_floor"],
            "num_raw_pressure_near_floor": metrics["num_proposed_pressure_near_floor"],
        }
    )
    return metrics


def apply_primitive_input_noise(
    batch: Euler1DBatch,
    noise_std: float,
    *,
    mode: str = "admissible_log_normal",
    rho_floor: float = 1.0e-6,
    pressure_floor: float = 1.0e-6,
) -> Euler1DBatch:
    """Return a denoising-training batch with noisy current primitives only."""

    if noise_std <= 0.0:
        return batch
    current = batch.current_primitive
    noise = torch.randn_like(current) * float(noise_std)
    if mode == "additive":
        return replace(
            batch,
            current_primitive=current + noise,
            current_conservative_state=None,
        )
    if mode != "admissible_log_normal":
        raise ValueError(f"unsupported primitive input noise mode: {mode}")
    noisy = current.clone()
    noisy[..., 0] = (current[..., 0] * torch.exp(noise[..., 0])).clamp_min(rho_floor)
    noisy[..., 1] = current[..., 1] + noise[..., 1]
    noisy[..., 2] = (current[..., 2] * torch.exp(noise[..., 2])).clamp_min(
        pressure_floor
    )
    return replace(
        batch,
        current_primitive=noisy,
        current_conservative_state=None,
    )


def evaluate_one_step(
    model: nn.Module,
    adapter: nn.Module,
    loader: DataLoader,
    normalizer: PrimitiveNormalizer,
    device: torch.device,
    loss_coordinates: str = "primitive",
    target_supervision: str = "state",
    flux_normalizer: PrimitiveNormalizer | None = None,
    flux_loss_weight: float = 1.0,
    boundary_exchange_normalizer: PrimitiveNormalizer | None = None,
    boundary_exchange_loss_weight: float = 0.0,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_state_loss = 0.0
    total_flux_loss = 0.0
    flux_loss_samples = 0
    total_boundary_exchange_loss = 0.0
    boundary_exchange_loss_samples = 0
    total_flux_reference_mse = 0.0
    total_flux_divergence_active_mse = 0.0
    total_flux_gauge_mse = 0.0
    flux_reference_samples = 0
    total_boundary_exchange_relative_l2 = 0.0
    boundary_exchange_samples = 0
    projection_closure_max_abs = 0.0
    total_rel_l2 = 0.0
    total_samples = 0
    min_density = float("inf")
    min_pressure = float("inf")
    proposed_accumulator = new_conservative_safety_accumulator()
    limiter_accumulator = new_limiter_accumulator()
    flux_correction_accumulator = new_flux_correction_accumulator()

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            raw = model(batch)
            prediction = adapter(raw, batch)
            if batch.target_primitive is None:
                raise RuntimeError("evaluation batch is missing target_primitive")
            loss, state_loss, flux_loss, boundary_exchange_loss = supervised_loss(
                raw,
                prediction,
                batch,
                normalizer,
                loss_coordinates=loss_coordinates,
                target_supervision=target_supervision,
                flux_normalizer=flux_normalizer,
                flux_loss_weight=flux_loss_weight,
                boundary_exchange_normalizer=boundary_exchange_normalizer,
                boundary_exchange_loss_weight=boundary_exchange_loss_weight,
            )
            rel_l2 = relative_l2_torch(prediction.primitive, batch.target_primitive)
            batch_size = batch.current_primitive.shape[0]
            total_loss += finite_scalar(loss) * batch_size
            total_state_loss += finite_scalar(state_loss) * batch_size
            if flux_loss is not None:
                total_flux_loss += finite_scalar(flux_loss) * batch_size
                flux_loss_samples += batch_size
            if boundary_exchange_loss is not None:
                total_boundary_exchange_loss += (
                    finite_scalar(boundary_exchange_loss) * batch_size
                )
                boundary_exchange_loss_samples += batch_size
            face_flux = decoded_face_flux(raw, prediction)
            if (
                flux_normalizer is not None
                and batch.target_face_flux is not None
                and face_flux.shape == batch.target_face_flux.shape
            ):
                reference_mse, active_mse, gauge_mse = (
                    normalized_face_flux_error_decomposition(
                        face_flux,
                        batch,
                        flux_normalizer,
                    )
                )
                total_flux_reference_mse += finite_scalar(reference_mse) * batch_size
                total_flux_divergence_active_mse += (
                    finite_scalar(active_mse) * batch_size
                )
                total_flux_gauge_mse += finite_scalar(gauge_mse) * batch_size
                flux_reference_samples += batch_size
            boundary_exchange = prediction.aux.get("boundary_exchange")
            if isinstance(boundary_exchange, torch.Tensor):
                volume = batch.geometry.cell_volume.to(
                    dtype=prediction.conservative.dtype,
                    device=prediction.conservative.device,
                ).unsqueeze(-1)
                predicted_exchange = (
                    volume * (prediction.conservative - batch.current_conservative)
                ).sum(dim=1)
                projection_closure_max_abs = max(
                    projection_closure_max_abs,
                    float(
                        (predicted_exchange - boundary_exchange)
                        .abs()
                        .max()
                        .cpu()
                        .item()
                    ),
                )
                target_conservative = batch.target_conservative
                if target_conservative is not None:
                    target_exchange = (
                        volume * (target_conservative - batch.current_conservative)
                    ).sum(dim=1)
                    exchange_relative_l2 = relative_l2_torch(
                        boundary_exchange,
                        target_exchange,
                    )
                    total_boundary_exchange_relative_l2 += float(
                        exchange_relative_l2.cpu().sum().item()
                    )
                    boundary_exchange_samples += batch_size
            total_rel_l2 += float(rel_l2.detach().cpu().sum().item())
            total_samples += batch_size
            min_density = min(
                min_density, float(prediction.primitive[..., 0].min().cpu().item())
            )
            min_pressure = min(
                min_pressure, float(prediction.primitive[..., 2].min().cpu().item())
            )
            update_conservative_safety_accumulator(
                proposed_accumulator,
                proposed_conservative(prediction),
                gamma=batch.gamma,
            )
            update_limiter_accumulator(limiter_accumulator, prediction.aux)
            update_flux_correction_accumulator(
                flux_correction_accumulator, prediction.aux
            )

    proposed_metrics = proposed_safety_metrics(proposed_accumulator)
    limiter_metrics = finalize_limiter_accumulator(limiter_accumulator)
    flux_correction_metrics = finalize_flux_correction_accumulator(
        flux_correction_accumulator
    )
    flux_reference_mse = (
        total_flux_reference_mse / flux_reference_samples
        if flux_reference_samples > 0
        else float("nan")
    )
    flux_gauge_mse = (
        total_flux_gauge_mse / flux_reference_samples
        if flux_reference_samples > 0
        else float("nan")
    )
    return {
        "loss": total_loss / max(total_samples, 1),
        "state_loss": total_state_loss / max(total_samples, 1),
        "flux_loss": (
            total_flux_loss / flux_loss_samples
            if flux_loss_samples > 0
            else float("nan")
        ),
        "flux_reference_mse": flux_reference_mse,
        "flux_divergence_active_mse": (
            total_flux_divergence_active_mse / flux_reference_samples
            if flux_reference_samples > 0
            else float("nan")
        ),
        "flux_gauge_mse": flux_gauge_mse,
        "flux_gauge_fraction": (
            flux_gauge_mse / max(flux_reference_mse, EPS)
            if flux_reference_samples > 0
            else float("nan")
        ),
        "boundary_exchange_relative_l2": (
            total_boundary_exchange_relative_l2 / boundary_exchange_samples
            if boundary_exchange_samples > 0
            else float("nan")
        ),
        "projection_closure_max_abs": (
            projection_closure_max_abs
            if boundary_exchange_samples > 0
            else float("nan")
        ),
        "relative_l2": total_rel_l2 / max(total_samples, 1),
        "min_density": min_density,
        "min_pressure": min_pressure,
        "boundary_exchange_loss": (
            total_boundary_exchange_loss / boundary_exchange_loss_samples
            if boundary_exchange_loss_samples > 0
            else float("nan")
        ),
        **proposed_metrics,
        **limiter_metrics,
        **flux_correction_metrics,
    }


def train_one_epoch(
    model: nn.Module,
    adapter: nn.Module,
    loader: DataLoader,
    normalizer: PrimitiveNormalizer,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    input_noise_std: float = 0.0,
    input_noise_mode: str = "admissible_log_normal",
    loss_coordinates: str = "primitive",
    target_supervision: str = "state",
    flux_normalizer: PrimitiveNormalizer | None = None,
    flux_loss_weight: float = 1.0,
    boundary_exchange_normalizer: PrimitiveNormalizer | None = None,
    boundary_exchange_loss_weight: float = 0.0,
    defer_metric_sync: bool = False,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_state_loss = 0.0
    total_flux_loss = 0.0
    flux_loss_samples = 0
    total_flux_reference_mse = 0.0
    total_flux_divergence_active_mse = 0.0
    total_flux_gauge_mse = 0.0
    flux_reference_samples = 0
    total_rel_l2 = 0.0
    total_samples = 0
    deferred_loss: list[torch.Tensor] = []
    deferred_state_loss: list[torch.Tensor] = []
    deferred_relative_l2: list[torch.Tensor] = []

    for batch in loader:
        batch = batch.to(device)
        if batch.target_primitive is None:
            raise RuntimeError("training batch is missing target_primitive")
        batch = apply_primitive_input_noise(
            batch, input_noise_std, mode=input_noise_mode
        )
        raw = model(batch)
        prediction = adapter(raw, batch)
        loss, state_loss, flux_loss, _boundary_exchange_loss = supervised_loss(
            raw,
            prediction,
            batch,
            normalizer,
            loss_coordinates=loss_coordinates,
            target_supervision=target_supervision,
            flux_normalizer=flux_normalizer,
            flux_loss_weight=flux_loss_weight,
            boundary_exchange_normalizer=boundary_exchange_normalizer,
            boundary_exchange_loss_weight=boundary_exchange_loss_weight,
        )
        if defer_metric_sync:
            torch._assert_async(torch.isfinite(loss), "non-finite training loss")
        elif not torch.isfinite(loss):
            raise FloatingPointError("non-finite training loss")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_size = batch.current_primitive.shape[0]
        with torch.no_grad():
            rel_l2 = relative_l2_torch(prediction.primitive, batch.target_primitive)
            face_flux = decoded_face_flux(raw, prediction)
            if (
                flux_normalizer is not None
                and batch.target_face_flux is not None
                and face_flux.shape == batch.target_face_flux.shape
            ):
                reference_mse, active_mse, gauge_mse = (
                    normalized_face_flux_error_decomposition(
                        face_flux,
                        batch,
                        flux_normalizer,
                    )
                )
                total_flux_reference_mse += finite_scalar(reference_mse) * batch_size
                total_flux_divergence_active_mse += (
                    finite_scalar(active_mse) * batch_size
                )
                total_flux_gauge_mse += finite_scalar(gauge_mse) * batch_size
                flux_reference_samples += batch_size
        if defer_metric_sync:
            deferred_loss.append(loss.detach() * batch_size)
            deferred_state_loss.append(state_loss.detach() * batch_size)
            deferred_relative_l2.append(rel_l2.detach().sum())
        else:
            total_loss += finite_scalar(loss) * batch_size
            total_state_loss += finite_scalar(state_loss) * batch_size
            if flux_loss is not None:
                total_flux_loss += finite_scalar(flux_loss) * batch_size
                flux_loss_samples += batch_size
            total_rel_l2 += float(rel_l2.detach().cpu().sum().item())
        total_samples += batch_size

    if defer_metric_sync:
        total_loss = float(torch.stack(deferred_loss).sum().cpu().item())
        total_state_loss = float(torch.stack(deferred_state_loss).sum().cpu().item())
        total_rel_l2 = float(torch.stack(deferred_relative_l2).sum().cpu().item())
    flux_reference_mse = (
        total_flux_reference_mse / flux_reference_samples
        if flux_reference_samples > 0
        else float("nan")
    )
    flux_gauge_mse = (
        total_flux_gauge_mse / flux_reference_samples
        if flux_reference_samples > 0
        else float("nan")
    )
    return {
        "loss": total_loss / max(total_samples, 1),
        "state_loss": total_state_loss / max(total_samples, 1),
        "flux_loss": (
            total_flux_loss / flux_loss_samples
            if flux_loss_samples > 0
            else float("nan")
        ),
        "flux_reference_mse": flux_reference_mse,
        "flux_divergence_active_mse": (
            total_flux_divergence_active_mse / flux_reference_samples
            if flux_reference_samples > 0
            else float("nan")
        ),
        "flux_gauge_mse": flux_gauge_mse,
        "flux_gauge_fraction": (
            flux_gauge_mse / max(flux_reference_mse, EPS)
            if flux_reference_samples > 0
            else float("nan")
        ),
        "relative_l2": total_rel_l2 / max(total_samples, 1),
    }


def conservative_admissibility_barrier(
    conservative: torch.Tensor,
    *,
    gamma: float,
    density_margin: float,
    pressure_margin: float,
) -> torch.Tensor:
    """Smoothly penalize states approaching nonpositive density or pressure."""

    if density_margin <= 0.0 or pressure_margin <= 0.0:
        raise ValueError("admissibility margins must be positive")
    density, momentum, energy = conservative.unbind(dim=-1)
    density_scale = torch.as_tensor(
        density_margin, dtype=conservative.dtype, device=conservative.device
    )
    pressure_scale = torch.as_tensor(
        pressure_margin, dtype=conservative.dtype, device=conservative.device
    )
    density_violation = F.softplus(
        (density_scale - density) / density_scale,
        beta=10.0,
    )
    safe_density = density.clamp_min(density_scale)
    internal_energy = energy - 0.5 * momentum.square() / safe_density
    pressure = (gamma - 1.0) * internal_energy
    pressure_violation = F.softplus(
        (pressure_scale - pressure) / pressure_scale,
        beta=10.0,
    )
    return density_violation.square().mean() + pressure_violation.square().mean()


def train_unrolled_epoch(
    model: nn.Module,
    adapter: nn.Module,
    loader: DataLoader,
    normalizer: PrimitiveNormalizer,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    input_noise_std: float = 0.0,
    input_noise_mode: str = "admissible_log_normal",
    loss_coordinates: str = "primitive",
    burn_in_steps: int = 0,
    burn_in_mode: str = "generated",
    admissibility_weight: float = 0.0,
    density_margin: float = 1.0e-3,
    pressure_margin: float = 1.0e-3,
    boundary_exchange_normalizer: PrimitiveNormalizer | None = None,
    boundary_exchange_loss_weight: float = 0.0,
    defer_metric_sync: bool = False,
) -> dict[str, float]:
    """Train through a fixed window after a generated or teacher prefix."""

    if burn_in_steps < 0:
        raise ValueError("burn_in_steps must be >= 0")
    if burn_in_mode not in UNROLL_BURN_IN_MODE_CHOICES:
        raise ValueError(f"unsupported burn-in mode: {burn_in_mode}")
    if burn_in_mode == "teacher" and burn_in_steps == 0:
        raise ValueError("teacher burn-in mode requires burn_in_steps > 0")
    if boundary_exchange_loss_weight > 0.0 and boundary_exchange_normalizer is None:
        raise ValueError("boundary-exchange loss requires its target normalizer")
    model.train()
    total_loss = 0.0
    total_state_loss = 0.0
    total_admissibility_loss = 0.0
    total_rel_l2 = 0.0
    total_samples = 0
    total_burn_in_relative_l2 = 0.0
    total_burn_in_samples = 0
    total_burn_in_nonpositive_samples = 0
    burn_in_min_density = float("inf")
    burn_in_min_pressure = float("inf")
    deferred_loss: list[torch.Tensor] = []
    deferred_state_loss: list[torch.Tensor] = []
    deferred_admissibility_loss: list[torch.Tensor] = []
    deferred_relative_l2: list[torch.Tensor] = []

    for initial_batch, target_sequence, dt_sequence in loader:
        initial_batch = initial_batch.to(device)
        target_sequence = target_sequence.to(device)
        dt_sequence = dt_sequence.to(device)
        sequence_steps = int(target_sequence.shape[1])
        if burn_in_steps >= sequence_steps:
            raise ValueError(
                "burn_in_steps must leave at least one supervised rollout step"
            )
        noisy_batch = apply_primitive_input_noise(
            initial_batch, input_noise_std, mode=input_noise_mode
        )
        current = noisy_batch.current_primitive
        current_conservative = noisy_batch.current_conservative
        batch_size = initial_batch.current_primitive.shape[0]
        step_losses: list[torch.Tensor] = []
        step_state_losses: list[torch.Tensor] = []
        step_admissibility_losses: list[torch.Tensor] = []
        step_relative_l2: list[torch.Tensor] = []

        if burn_in_steps > 0 and burn_in_mode == "generated":
            burn_in_nonpositive = torch.zeros(
                batch_size,
                dtype=torch.bool,
                device=device,
            )
            with torch.no_grad():
                for step in range(burn_in_steps):
                    step_batch = replace(
                        initial_batch,
                        current_primitive=current,
                        current_conservative_state=current_conservative,
                        target_primitive=target_sequence[:, step],
                        target_face_flux=None,
                        dt=dt_sequence[:, step],
                    )
                    prediction = adapter(model(step_batch), step_batch)
                    current = prediction.primitive
                    current_conservative = prediction.conservative
                    density = current[..., 0]
                    pressure = current[..., 2]
                    burn_in_nonpositive |= (density <= 0.0).any(dim=1)
                    burn_in_nonpositive |= (pressure <= 0.0).any(dim=1)
                    burn_in_min_density = min(
                        burn_in_min_density,
                        float(density.detach().min().cpu().item()),
                    )
                    burn_in_min_pressure = min(
                        burn_in_min_pressure,
                        float(pressure.detach().min().cpu().item()),
                    )
            burn_in_relative_l2 = relative_l2_torch(
                current,
                target_sequence[:, burn_in_steps - 1],
            )
            total_burn_in_relative_l2 += float(
                burn_in_relative_l2.detach().cpu().sum().item()
            )
            total_burn_in_nonpositive_samples += int(
                burn_in_nonpositive.detach().cpu().sum().item()
            )
            total_burn_in_samples += batch_size
            current = current.detach()
            current_conservative = current_conservative.detach()
        elif burn_in_steps > 0:
            current = target_sequence[:, burn_in_steps - 1].detach()
            current_conservative = primitive_to_conservative(
                current,
                gamma=initial_batch.gamma,
            ).detach()
            density = current[..., 0]
            pressure = current[..., 2]
            burn_in_nonpositive = (density <= 0.0).any(dim=1)
            burn_in_nonpositive |= (pressure <= 0.0).any(dim=1)
            total_burn_in_relative_l2 += 0.0
            total_burn_in_nonpositive_samples += int(
                burn_in_nonpositive.detach().cpu().sum().item()
            )
            total_burn_in_samples += batch_size
            burn_in_min_density = min(
                burn_in_min_density,
                float(density.detach().min().cpu().item()),
            )
            burn_in_min_pressure = min(
                burn_in_min_pressure,
                float(pressure.detach().min().cpu().item()),
            )

        optimizer.zero_grad(set_to_none=True)
        for step in range(burn_in_steps, sequence_steps):
            step_batch = replace(
                initial_batch,
                current_primitive=current,
                current_conservative_state=current_conservative,
                target_primitive=target_sequence[:, step],
                target_face_flux=None,
                dt=dt_sequence[:, step],
            )
            raw = model(step_batch)
            prediction = adapter(raw, step_batch)
            loss_prediction, loss_target = state_pair_for_loss(
                prediction,
                step_batch,
                loss_coordinates,
            )
            state_loss = normalizer.mse(loss_prediction, loss_target)
            if not defer_metric_sync and not torch.isfinite(state_loss):
                diagnostic = {
                    "sequence_step": step,
                    "raw_all_finite": bool(torch.isfinite(raw).all()),
                    "raw_abs_max": float(raw.detach().abs().max().cpu().item()),
                    "current_all_finite": bool(torch.isfinite(current).all()),
                    "current_min_density": float(
                        current[..., 0].detach().min().cpu().item()
                    ),
                    "current_min_pressure": float(
                        current[..., 2].detach().min().cpu().item()
                    ),
                    "prediction_all_finite": bool(
                        torch.isfinite(prediction.primitive).all()
                    ),
                    "prediction_min_density": float(
                        prediction.primitive[..., 0].detach().min().cpu().item()
                    ),
                    "prediction_min_pressure": float(
                        prediction.primitive[..., 2].detach().min().cpu().item()
                    ),
                    "prediction_abs_max": float(
                        prediction.primitive.detach().abs().max().cpu().item()
                    ),
                }
                raise FloatingPointError(
                    "non-finite unrolled state loss: "
                    + json.dumps(diagnostic, sort_keys=True)
                )
            if admissibility_weight > 0.0:
                admissibility_loss = conservative_admissibility_barrier(
                    prediction.conservative,
                    gamma=step_batch.gamma,
                    density_margin=density_margin,
                    pressure_margin=pressure_margin,
                )
                step_loss = state_loss + admissibility_weight * admissibility_loss
            else:
                admissibility_loss = state_loss.new_zeros(())
                step_loss = state_loss
            if boundary_exchange_loss_weight > 0.0:
                truth_current = (
                    initial_batch.current_primitive
                    if step == 0
                    else target_sequence[:, step - 1]
                )
                exchange_target = state_pair_boundary_exchange_target(
                    truth_current,
                    target_sequence[:, step],
                    initial_batch,
                )
                exchange_prediction = predicted_boundary_exchange(prediction)
                boundary_exchange_loss = boundary_exchange_normalizer.mse(
                    exchange_prediction.unsqueeze(1),
                    exchange_target.unsqueeze(1),
                )
                step_loss = (
                    step_loss + boundary_exchange_loss_weight * boundary_exchange_loss
                )
            step_losses.append(step_loss)
            step_state_losses.append(state_loss)
            step_admissibility_losses.append(admissibility_loss)
            step_relative_l2.append(
                relative_l2_torch(prediction.primitive, target_sequence[:, step])
            )
            current = prediction.primitive
            current_conservative = prediction.conservative

        loss = torch.stack(step_losses).mean()
        state_loss = torch.stack(step_state_losses).mean()
        admissibility_loss = torch.stack(step_admissibility_losses).mean()
        if defer_metric_sync:
            torch._assert_async(
                torch.isfinite(loss),
                "non-finite unrolled training loss",
            )
        elif not torch.isfinite(loss):
            raise FloatingPointError("non-finite unrolled training loss")
        loss.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        rel_l2 = torch.stack(step_relative_l2, dim=1).mean(dim=1)
        if defer_metric_sync:
            deferred_loss.append(loss.detach() * batch_size)
            deferred_state_loss.append(state_loss.detach() * batch_size)
            deferred_admissibility_loss.append(admissibility_loss.detach() * batch_size)
            deferred_relative_l2.append(rel_l2.detach().sum())
        else:
            total_loss += finite_scalar(loss) * batch_size
            total_state_loss += finite_scalar(state_loss) * batch_size
            total_admissibility_loss += finite_scalar(admissibility_loss) * batch_size
            total_rel_l2 += float(rel_l2.detach().cpu().sum().item())
        total_samples += batch_size

    if defer_metric_sync:
        total_loss = float(torch.stack(deferred_loss).sum().cpu().item())
        total_state_loss = float(torch.stack(deferred_state_loss).sum().cpu().item())
        total_admissibility_loss = float(
            torch.stack(deferred_admissibility_loss).sum().cpu().item()
        )
        total_rel_l2 = float(torch.stack(deferred_relative_l2).sum().cpu().item())
    return {
        "loss": total_loss / max(total_samples, 1),
        "state_loss": total_state_loss / max(total_samples, 1),
        "admissibility_loss": total_admissibility_loss / max(total_samples, 1),
        "relative_l2": total_rel_l2 / max(total_samples, 1),
        "burn_in_relative_l2": (
            total_burn_in_relative_l2 / total_burn_in_samples
            if total_burn_in_samples
            else float("nan")
        ),
        "burn_in_nonpositive_sample_fraction": (
            total_burn_in_nonpositive_samples / total_burn_in_samples
            if total_burn_in_samples
            else float("nan")
        ),
        "burn_in_min_density": (
            burn_in_min_density if total_burn_in_samples else float("nan")
        ),
        "burn_in_min_pressure": (
            burn_in_min_pressure if total_burn_in_samples else float("nan")
        ),
    }


def primitive_to_conservative_np(primitive: np.ndarray, gamma: float) -> np.ndarray:
    rho = primitive[..., 0]
    velocity = primitive[..., 1]
    pressure = primitive[..., 2]
    momentum = rho * velocity
    energy = pressure / (gamma - 1.0) + 0.5 * rho * velocity**2
    return np.stack((rho, momentum, energy), axis=-1)


def cell_widths_np(x: np.ndarray) -> np.ndarray:
    interior_faces = 0.5 * (x[:-1] + x[1:])
    left_face = x[0] - 0.5 * (x[1] - x[0])
    right_face = x[-1] + 0.5 * (x[-1] - x[-2])
    faces = np.concatenate(([left_face], interior_faces, [right_face]))
    return np.diff(faces)


def effective_cfl_profile_np(
    primitive: np.ndarray,
    dt: np.ndarray,
    x: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Return the Euler domain-of-dependence radius in grid cells per step."""

    primitive = np.asarray(primitive, dtype=np.float64)
    dt = np.asarray(dt, dtype=np.float64).reshape(-1)
    if primitive.ndim != 3 or primitive.shape[0] != dt.size:
        raise ValueError("primitive must have shape [steps, cells, 3] matching dt")
    rho = primitive[..., 0]
    velocity = primitive[..., 1]
    pressure = primitive[..., 2]
    sound_speed = np.sqrt(gamma * pressure / rho)
    max_wave_speed = np.max(np.abs(velocity) + sound_speed, axis=1)
    dx_min = float(np.min(cell_widths_np(np.asarray(x, dtype=np.float64))))
    return max_wave_speed * dt / dx_min


def conservative_total_np(
    primitive: np.ndarray, x: np.ndarray, gamma: float
) -> np.ndarray:
    widths = cell_widths_np(x).astype(np.float64)
    conservative = primitive_to_conservative_np(primitive, gamma).astype(np.float64)
    return np.sum(conservative * widths[..., None], axis=-2)


def pressure_front_position_np(primitive: np.ndarray, x: np.ndarray) -> np.ndarray:
    if primitive.shape[-2] < 2:
        return np.full(primitive.shape[:-2], np.nan, dtype=np.float64)
    pressure = primitive[..., 2]
    grad = np.abs(np.diff(pressure, axis=-1))
    idx = np.argmax(grad, axis=-1)
    face_x = 0.5 * (x[:-1] + x[1:])
    return face_x[idx]


def separated_pressure_fronts_np(
    primitive: np.ndarray,
    x: np.ndarray,
    *,
    num_fronts: int = 2,
    min_separation_cells: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Return strength-ranked pressure-gradient fronts after 1D nonmax suppression."""

    if num_fronts < 1:
        raise ValueError("num_fronts must be positive")
    if min_separation_cells < 0:
        raise ValueError("min_separation_cells must be nonnegative")
    if primitive.shape[-1] != 3 or primitive.shape[-2] != x.shape[0]:
        raise ValueError("primitive and x shapes are inconsistent")
    num_faces = primitive.shape[-2] - 1
    if num_faces < num_fronts:
        shape = (*primitive.shape[:-2], num_fronts)
        return (
            np.full(shape, np.nan, dtype=np.float64),
            np.full(shape, np.nan, dtype=np.float64),
        )

    pressure_jump = np.abs(np.diff(primitive[..., 2], axis=-1))
    flat_jump = pressure_jump.reshape(-1, num_faces)
    face_x = 0.5 * (np.asarray(x, dtype=np.float64)[:-1] + x[1:])
    positions = np.full((flat_jump.shape[0], num_fronts), np.nan, dtype=np.float64)
    strengths = np.full_like(positions, np.nan)
    for row_id, row in enumerate(flat_jump):
        selected: list[int] = []
        for face_id in np.argsort(-row, kind="stable"):
            if all(
                abs(int(face_id) - previous) > min_separation_cells
                for previous in selected
            ):
                selected.append(int(face_id))
                if len(selected) == num_fronts:
                    break
        if len(selected) != num_fronts:
            continue
        positions[row_id] = face_x[selected]
        strengths[row_id] = row[selected]
    shape = (*primitive.shape[:-2], num_fronts)
    return positions.reshape(shape), strengths.reshape(shape)


def pressure_front_top2_metrics_np(
    prediction: np.ndarray,
    target: np.ndarray,
    x: np.ndarray,
    *,
    min_separation_cells: int = 4,
) -> dict[str, np.ndarray]:
    """Match two separated fronts by position and retain their strength error."""

    pred_x, pred_strength = separated_pressure_fronts_np(
        prediction,
        x,
        num_fronts=2,
        min_separation_cells=min_separation_cells,
    )
    truth_x, truth_strength = separated_pressure_fronts_np(
        target,
        x,
        num_fronts=2,
        min_separation_cells=min_separation_cells,
    )
    direct = np.abs(pred_x[..., 0] - truth_x[..., 0]) + np.abs(
        pred_x[..., 1] - truth_x[..., 1]
    )
    crossed = np.abs(pred_x[..., 0] - truth_x[..., 1]) + np.abs(
        pred_x[..., 1] - truth_x[..., 0]
    )
    use_crossed = crossed < direct
    matched_strength = np.where(
        use_crossed[..., None],
        pred_strength[..., ::-1],
        pred_strength,
    )
    strength_relative_l1 = np.sum(
        np.abs(matched_strength - truth_strength),
        axis=-1,
    ) / np.maximum(np.sum(np.abs(truth_strength), axis=-1), EPS)
    primary_to_truth_top2 = np.minimum(
        np.abs(pred_x[..., 0] - truth_x[..., 0]),
        np.abs(pred_x[..., 0] - truth_x[..., 1]),
    )
    return {
        "position_assignment_mae": 0.5 * np.minimum(direct, crossed),
        "strength_relative_l1": strength_relative_l1,
        "primary_to_truth_top2_mae": primary_to_truth_top2,
    }


def relative_l2_np(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    diff = prediction.reshape(prediction.shape[0], -1) - target.reshape(
        target.shape[0], -1
    )
    denom = np.linalg.norm(target.reshape(target.shape[0], -1), axis=1)
    return np.linalg.norm(diff, axis=1) / np.maximum(denom, EPS)


def rollout_case(
    model: nn.Module,
    adapter: nn.Module,
    source: Euler1DNPZ,
    case_id: int,
    steps: int,
    step_stride: int,
    device: torch.device,
    final_frame: int | None = None,
    recurrent_coordinates: str = "primitive",
) -> dict[str, Any]:
    model.eval()
    if recurrent_coordinates not in COORDINATE_CHOICES:
        raise ValueError(f"unsupported recurrent coordinates: {recurrent_coordinates}")
    if step_stride < 1 or step_stride >= source.num_frames:
        raise ValueError("step_stride must be in [1, num_frames - 1]")
    if final_frame is not None:
        if final_frame < step_stride or final_frame >= source.num_frames:
            raise ValueError(
                "rollout_final_frame must be in [step_stride, num_frames - 1]"
            )
        if final_frame % step_stride != 0:
            raise ValueError("rollout_final_frame must be divisible by step_stride")
        max_steps = final_frame // step_stride
    else:
        max_steps = min(steps, (source.num_frames - 1) // step_stride)
        final_frame = max_steps * step_stride
    x_np = source.x[case_id]
    truth_input_ids = np.arange(max_steps, dtype=np.int64) * step_stride
    truth_next_ids = truth_input_ids + step_stride
    truth_dt = source.t[case_id, truth_next_ids] - source.t[case_id, truth_input_ids]
    truth_effective_cfl = effective_cfl_profile_np(
        source.data[case_id, truth_input_ids],
        truth_dt,
        x_np,
        source.gamma,
    )
    current_primitive = (
        torch.from_numpy(np.asarray(source.data[case_id, 0], dtype=np.float32))
        .unsqueeze(0)
        .to(device)
    )
    current_conservative = primitive_to_conservative(
        current_primitive,
        gamma=source.gamma,
    )
    x = torch.from_numpy(x_np).unsqueeze(0).to(device)
    left = torch.from_numpy(source.left_states[case_id]).unsqueeze(0).to(device)
    right_initial = (
        torch.from_numpy(source.right_states[case_id]).unsqueeze(0).to(device)
    )
    predictions: list[np.ndarray] = []
    proposed_accumulator = new_conservative_safety_accumulator()
    limiter_accumulator = new_limiter_accumulator()
    flux_correction_accumulator = new_flux_correction_accumulator()
    termination_reason: str | None = None
    first_invalid_step: int | None = None

    with torch.no_grad():
        for step in range(max_steps):
            current_frame = step * step_stride
            next_frame = current_frame + step_stride
            dt = torch.tensor(
                [source.t[case_id, next_frame] - source.t[case_id, current_frame]],
                dtype=current_primitive.dtype,
                device=device,
            )
            if recurrent_coordinates == "conservative":
                model_primitive = conservative_to_primitive(
                    current_conservative,
                    gamma=source.gamma,
                )
            else:
                model_primitive = current_primitive
            batch = make_euler1d_batch(
                model_primitive,
                x,
                dt,
                current_conservative_state=(
                    current_conservative
                    if recurrent_coordinates == "conservative"
                    else None
                ),
                gamma=source.gamma,
                left_boundary_primitive=left,
                right_initial_primitive=right_initial,
            )
            raw = model(batch)
            decoded = adapter(raw, batch)
            next_state = decoded.primitive
            update_conservative_safety_accumulator(
                proposed_accumulator,
                proposed_conservative(decoded),
                gamma=source.gamma,
            )
            update_limiter_accumulator(limiter_accumulator, decoded.aux)
            update_flux_correction_accumulator(flux_correction_accumulator, decoded.aux)
            if not torch.isfinite(next_state).all():
                termination_reason = "nonfinite_state"
                first_invalid_step = step + 1
                break
            raw_recurrence = bool(decoded.aux.get("raw_recurrence", False))
            admissible = bool(
                torch.all(next_state[..., 0] > 0.0)
                and torch.all(next_state[..., 2] > 0.0)
            )
            if raw_recurrence and not admissible:
                termination_reason = "nonpositive_raw_state"
                first_invalid_step = step + 1
                break
            predictions.append(
                next_state.squeeze(0).detach().cpu().numpy().astype(np.float64)
            )
            current_primitive = next_state.detach()
            current_conservative = decoded.conservative.detach()

    if not predictions:
        proposed_metrics = proposed_safety_metrics(proposed_accumulator)
        limiter_metrics = finalize_limiter_accumulator(limiter_accumulator)
        flux_correction_metrics = finalize_flux_correction_accumulator(
            flux_correction_accumulator
        )
        return {
            "case_id": int(case_id),
            "finite": termination_reason != "nonfinite_state",
            "admissible": False,
            "termination_reason": termination_reason,
            "first_invalid_step": first_invalid_step,
            "num_steps": 0,
            "requested_steps": int(max_steps),
            "survival_fraction": 0.0,
            "rollout_relative_l2_mean": float("nan"),
            "rollout_relative_l2_final": float("nan"),
            "min_density": float("nan"),
            "min_pressure": float("nan"),
            "max_abs_primitive": float("nan"),
            **proposed_metrics,
            **limiter_metrics,
            **flux_correction_metrics,
            "shock_position_mae": float("nan"),
            "shock_top2_position_mae": float("nan"),
            "shock_top2_strength_relative_l1": float("nan"),
            "shock_primary_to_truth_top2_mae": float("nan"),
            "conservative_total_error_final": float("nan"),
            "step_stride": int(step_stride),
            "final_frame": int(final_frame),
            "requested_final_frame": int(final_frame),
            "completed_horizon": False,
            "initial_effective_cfl": float(truth_effective_cfl[0]),
            "truth_effective_cfl_max": float(np.max(truth_effective_cfl)),
            "recurrent_coordinates": recurrent_coordinates,
        }

    pred = np.stack(predictions, axis=0)
    truth_ids = np.arange(1, pred.shape[0] + 1, dtype=np.int64) * step_stride
    truth = source.data[case_id, truth_ids].astype(np.float64)
    rel_l2 = relative_l2_np(pred, truth)
    pred_front = pressure_front_position_np(pred, x_np)
    truth_front = pressure_front_position_np(truth, x_np)
    top2_front = pressure_front_top2_metrics_np(pred, truth, x_np)
    final_total_pred = conservative_total_np(pred[-1], x_np, source.gamma)
    final_total_truth = conservative_total_np(truth[-1], x_np, source.gamma)
    total_error = np.linalg.norm(final_total_pred - final_total_truth) / max(
        np.linalg.norm(final_total_truth),
        EPS,
    )
    proposed_metrics = proposed_safety_metrics(proposed_accumulator)
    limiter_metrics = finalize_limiter_accumulator(limiter_accumulator)
    flux_correction_metrics = finalize_flux_correction_accumulator(
        flux_correction_accumulator
    )
    return {
        "case_id": int(case_id),
        "finite": bool(
            np.isfinite(pred).all() and termination_reason != "nonfinite_state"
        ),
        "admissible": termination_reason is None,
        "termination_reason": termination_reason,
        "first_invalid_step": first_invalid_step,
        "num_steps": int(pred.shape[0]),
        "requested_steps": int(max_steps),
        "survival_fraction": float(pred.shape[0] / max_steps),
        "rollout_relative_l2_mean": float(np.mean(rel_l2)),
        "rollout_relative_l2_final": float(rel_l2[-1]),
        "min_density": float(np.min(pred[..., 0])),
        "min_pressure": float(np.min(pred[..., 2])),
        "max_abs_primitive": float(np.max(np.abs(pred))),
        **proposed_metrics,
        **limiter_metrics,
        **flux_correction_metrics,
        "shock_position_mae": float(np.mean(np.abs(pred_front - truth_front))),
        "shock_top2_position_mae": float(
            np.mean(top2_front["position_assignment_mae"])
        ),
        "shock_top2_strength_relative_l1": float(
            np.mean(top2_front["strength_relative_l1"])
        ),
        "shock_primary_to_truth_top2_mae": float(
            np.mean(top2_front["primary_to_truth_top2_mae"])
        ),
        "conservative_total_error_final": float(total_error),
        "step_stride": int(step_stride),
        "final_frame": int(truth_ids[-1]),
        "requested_final_frame": int(final_frame),
        "completed_horizon": bool(
            pred.shape[0] == max_steps and truth_ids[-1] == final_frame
        ),
        "initial_effective_cfl": float(truth_effective_cfl[0]),
        "truth_effective_cfl_max": float(np.max(truth_effective_cfl)),
        "recurrent_coordinates": recurrent_coordinates,
        "truth_frame_ids": truth_ids.tolist(),
        "relative_l2_by_step": rel_l2.tolist(),
    }


def summarize_rollouts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "finite": False,
            "num_cases": 0,
            "num_completed_cases": 0,
            "completion_fraction": 0.0,
            "survival_fraction_mean": 0.0,
            "survival_fraction_min": 0.0,
            "rollout_relative_l2_mean": float("nan"),
            "rollout_relative_l2_final": float("nan"),
        }
    numeric_keys = (
        "rollout_relative_l2_mean",
        "rollout_relative_l2_final",
        "min_density",
        "min_pressure",
        "proposed_min_density",
        "proposed_min_pressure",
        "raw_min_density",
        "raw_min_pressure",
        "max_abs_primitive",
        "shock_position_mae",
        "shock_top2_position_mae",
        "shock_top2_strength_relative_l1",
        "shock_primary_to_truth_top2_mae",
        "conservative_total_error_final",
        "limiter_theta_mean",
        "limiter_theta_min",
        "limiter_activation_fraction",
        "flux_correction_abs_over_bound_mean",
        "flux_correction_abs_over_bound_max",
        "flux_correction_saturation_fraction",
    )
    completed = np.asarray(
        [bool(row.get("completed_horizon")) for row in rows], dtype=np.float64
    )
    survival = np.asarray(
        [float(row.get("survival_fraction", 0.0)) for row in rows],
        dtype=np.float64,
    )
    initial_cfl = np.asarray(
        [float(row.get("initial_effective_cfl", float("nan"))) for row in rows],
        dtype=np.float64,
    )
    truth_cfl_max = np.asarray(
        [float(row.get("truth_effective_cfl_max", float("nan"))) for row in rows],
        dtype=np.float64,
    )
    summary: dict[str, Any] = {
        "finite": all(bool(row["finite"]) for row in rows),
        "admissible": all(bool(row.get("admissible", True)) for row in rows),
        "num_cases": len(rows),
        "num_completed_cases": int(np.sum(completed)),
        "completion_fraction": float(np.mean(completed)),
        "survival_fraction_mean": float(np.mean(survival)),
        "survival_fraction_min": float(np.min(survival)),
        "num_steps_min": min(int(row["num_steps"]) for row in rows),
        "num_steps_max": max(int(row["num_steps"]) for row in rows),
        "final_frame_min": min(int(row["final_frame"]) for row in rows),
        "final_frame_max": max(int(row["final_frame"]) for row in rows),
        "completed_horizon": all(bool(row.get("completed_horizon")) for row in rows),
        "num_nonpositive_terminations": sum(
            row.get("termination_reason") == "nonpositive_raw_state" for row in rows
        ),
        "num_nonfinite_terminations": sum(
            row.get("termination_reason") == "nonfinite_state" for row in rows
        ),
    }
    for prefix, values in (
        ("initial_effective_cfl", initial_cfl),
        ("truth_effective_cfl_max", truth_cfl_max),
    ):
        finite_values = values[np.isfinite(values)]
        summary[f"{prefix}_mean"] = (
            float(np.mean(finite_values)) if finite_values.size else float("nan")
        )
        summary[f"{prefix}_median"] = (
            float(np.median(finite_values)) if finite_values.size else float("nan")
        )
        summary[f"{prefix}_max"] = (
            float(np.max(finite_values)) if finite_values.size else float("nan")
        )
    min_keys = {
        "min_density",
        "min_pressure",
        "proposed_min_density",
        "proposed_min_pressure",
        "raw_min_density",
        "raw_min_pressure",
        "limiter_theta_min",
    }
    for key in numeric_keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            summary[key] = float("nan")
        elif key in min_keys:
            summary[key] = float(np.min(finite_values))
        else:
            summary[key] = float(np.mean(finite_values))
    summary["num_nonpositive_proposed_density"] = int(
        sum(int(row.get("num_nonpositive_proposed_density", 0)) for row in rows)
    )
    summary["num_nonpositive_raw_density"] = summary["num_nonpositive_proposed_density"]
    summary["num_nonpositive_proposed_pressure"] = int(
        sum(int(row.get("num_nonpositive_proposed_pressure", 0)) for row in rows)
    )
    summary["num_nonpositive_raw_pressure"] = summary[
        "num_nonpositive_proposed_pressure"
    ]
    summary["num_proposed_density_near_floor"] = int(
        sum(int(row.get("num_proposed_density_near_floor", 0)) for row in rows)
    )
    summary["num_raw_density_near_floor"] = summary["num_proposed_density_near_floor"]
    summary["num_proposed_pressure_near_floor"] = int(
        sum(int(row.get("num_proposed_pressure_near_floor", 0)) for row in rows)
    )
    summary["num_raw_pressure_near_floor"] = summary["num_proposed_pressure_near_floor"]
    return summary


def write_history(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    preferred = [
        "model",
        "model_implementation",
        "parameter_count",
        "target",
        "target_type",
        "status",
        "seed",
        "split_seed",
        "seed_count",
        "train_cases_count",
        "val_cases_count",
        "test_cases_count",
        "best_epoch",
        "selection_metric",
        "selection_score",
        "step_stride",
        "rollout_final_frame",
        "input_coordinates",
        "recurrent_coordinates",
        "predicted_quantity",
        "target_supervision",
        "flux_gauge_mode",
        "interface_flux_mode",
        "loss_coordinates",
        "input_normalization",
        "loss_normalization",
        "flux_loss_normalization",
        "flux_loss_weight",
        "unroll_epochs",
        "unroll_steps",
        "unroll_burn_in_steps",
        "unroll_burn_in_mode",
        "unroll_lr_factor",
        "unroll_optimizer_reset",
        "unroll_admissibility_weight",
        "unroll_admissibility_margin_fraction",
        "unroll_density_margin",
        "unroll_pressure_margin",
        "unroll_checkpoint_selected",
        "completion_fraction",
        "survival_fraction_mean",
        "survival_fraction_min",
        "initial_effective_cfl_median",
        "truth_effective_cfl_max_median",
        "truth_effective_cfl_max_max",
        "input_noise_std",
        "input_noise_mode",
        "one_step_loss",
        "one_step_state_loss",
        "one_step_flux_loss",
        "one_step_relative_l2",
        "one_step_boundary_exchange_relative_l2",
        "one_step_projection_closure_max_abs",
        "test_loss",
        "test_relative_l2",
        "rollout_relative_l2_mean",
        "rollout_relative_l2_final",
        "min_density",
        "min_pressure",
        "proposed_min_density",
        "proposed_min_pressure",
        "raw_min_density",
        "raw_min_pressure",
        "num_nonpositive_proposed_density",
        "num_nonpositive_proposed_pressure",
        "num_proposed_density_near_floor",
        "num_proposed_pressure_near_floor",
        "num_nonpositive_raw_density",
        "num_nonpositive_raw_pressure",
        "num_raw_density_near_floor",
        "num_raw_pressure_near_floor",
        "limiter_theta_mean",
        "limiter_theta_min",
        "limiter_activation_fraction",
        "flux_correction_abs_over_bound_mean",
        "flux_correction_abs_over_bound_max",
        "flux_correction_saturation_fraction",
        "max_abs_primitive",
        "shock_position_mae",
        "shock_top2_position_mae",
        "shock_top2_strength_relative_l1",
        "shock_primary_to_truth_top2_mae",
        "conservative_total_error_final",
        "runtime_seconds",
    ]
    ordered = [key for key in preferred if key in fieldnames] + [
        key for key in fieldnames if key not in preferred
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return json_ready(value.detach().cpu().tolist())
    if isinstance(value, Path):
        return str(value)
    return value


def model_implementation_name(model_name: str, args: argparse.Namespace) -> str:
    if model_name == "fno":
        return (
            "FNOEuler1DHead"
            f"(width={args.fno_width},modes={args.fno_modes},"
            f"layers={args.fno_layers},fc_dim={args.fno_fc_dim})"
        )
    if model_name == "cpgnet":
        return (
            "CPGNetEuler1D"
            f"(hidden={args.cpg_hidden_dim},message_layers={args.cpg_message_passing_steps},"
            "edge_encoder_layers=4,directed_positive_interfaces=True,"
            "shared_rusanov_flux=True,geometry=exact,no_cell_limiter=True)"
        )
    return model_name


def predicted_quantity_name(target_name: str) -> str:
    if target_name == "conservative_state":
        return "next_conservative_state"
    if target_name in ("residual", "limited_residual"):
        return "conservative_increment"
    if target_name == "projected_residual":
        return "projected_conservative_increment_and_boundary_exchange"
    if target_name == "primitive_residual":
        return "primitive_log_increment"
    if target_name in ("flux", "limited_flux"):
        return "face_flux"
    if target_name == "physical_flux_correction":
        return "face_flux_correction"
    if target_name in (
        "interface",
        "relative_interface",
        "positive_limited_interface",
        CPG_TARGET,
    ):
        return "interface_states"
    if target_name == "state":
        return "next_primitive_state"
    return target_name


def clone_state_dict_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def load_state_dict_cpu(
    model: nn.Module, state: dict[str, torch.Tensor], device: torch.device
) -> None:
    model.load_state_dict({key: value.to(device) for key, value in state.items()})


def rollout_cases(
    model: nn.Module,
    adapter: nn.Module,
    source: Euler1DNPZ,
    case_ids: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    return [
        rollout_case(
            model,
            adapter,
            source,
            int(case_id),
            args.rollout_steps,
            args.step_stride,
            device,
            final_frame=args.rollout_final_frame,
            recurrent_coordinates=args.recurrent_coordinates,
        )
        for case_id in case_ids.tolist()
    ]


def rollout_selection_score(summary: dict[str, Any], one_step_loss: float) -> float:
    """Prioritize valid rollouts, then survival, then bounded one-step fit."""

    value = float(summary.get("rollout_relative_l2_final", float("nan")))
    finite = bool(summary.get("finite", False))
    complete = bool(summary.get("completed_horizon", False))
    admissible = bool(summary.get("admissible", True))
    if finite and complete and admissible and math.isfinite(value):
        return value
    fallback = float(one_step_loss)
    if not math.isfinite(fallback):
        return float("inf")
    survival = float(summary.get("survival_fraction_mean", 0.0))
    survival = min(max(survival, 0.0), 1.0)
    fit_rank = math.atan(max(fallback, 0.0)) / (0.5 * math.pi)
    failure_tier = 1.0e6 if finite else 1.0e12
    return failure_tier + 100.0 * (1.0 - survival) + fit_rank


def run_single(
    source: Euler1DNPZ,
    train_cases: np.ndarray,
    val_cases: np.ndarray,
    test_cases: np.ndarray,
    model_name: str,
    target_name: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    start = time.perf_counter()
    target = cast(Euler1DTarget, target_name)
    implementation = model_implementation_name(model_name, args)
    run_name = f"{model_name}_{target_name}"
    if args.target_supervision != "state":
        run_name += f"_{args.target_supervision}"
    if target_name == "flux" and args.flux_gauge_mode != "raw":
        run_name += f"_{args.flux_gauge_mode}"
    if target_name == "relative_interface":
        run_name += f"_{args.interface_flux_mode}"
    if args.boundary_exchange_loss_weight > 0.0:
        run_name += f"_boundaryw{args.boundary_exchange_loss_weight:g}"
    if args.continuation_checkpoint is not None:
        run_name += "_continuation"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = Euler1DTimePairDataset(
        source,
        case_indices=train_cases,
        step_stride=args.step_stride,
    )
    val_dataset = Euler1DTimePairDataset(
        source,
        case_indices=val_cases,
        step_stride=args.step_stride,
    )
    test_dataset = Euler1DTimePairDataset(
        source,
        case_indices=test_cases,
        step_stride=args.step_stride,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_sampler = make_initial_frame_sampler(
        train_dataset.pairs,
        initial_frame_weight=args.initial_frame_weight,
        generator=generator,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=collate_euler1d_pairs,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_euler1d_pairs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_euler1d_pairs,
    )
    unroll_loader: DataLoader | None = None
    unroll_density_margin = float("nan")
    unroll_pressure_margin = float("nan")
    if args.unroll_epochs > 0:
        unroll_dataset = Euler1DRolloutWindowDataset(
            source,
            case_indices=train_cases,
            step_stride=args.step_stride,
            rollout_steps=args.unroll_steps,
            burn_in_steps=args.unroll_burn_in_steps,
        )
        unroll_generator = torch.Generator().manual_seed(args.seed + 1)
        unroll_sampler = make_initial_frame_sampler(
            unroll_dataset.windows,
            initial_frame_weight=args.initial_frame_weight,
            generator=unroll_generator,
        )
        unroll_loader = DataLoader(
            unroll_dataset,
            batch_size=args.batch_size,
            shuffle=unroll_sampler is None,
            sampler=unroll_sampler,
            collate_fn=collate_euler1d_rollout_windows,
            generator=unroll_generator,
        )
        train_primitive = source.data[train_cases]
        margin_fraction = float(args.unroll_admissibility_margin_fraction)
        unroll_density_margin = margin_fraction * float(np.min(train_primitive[..., 0]))
        unroll_pressure_margin = margin_fraction * float(
            np.min(train_primitive[..., 2])
        )

    normalizer = PrimitiveNormalizer.from_source(
        source,
        train_cases,
        step_stride=args.step_stride,
        coordinates=args.loss_coordinates,
        normalization=args.loss_normalization,
    ).to(device)
    flux_normalizer = None
    if (
        target_name in ("flux", "projected_residual")
        and source.face_flux_integral is not None
    ):
        flux_normalizer = PrimitiveNormalizer.from_source_face_flux(
            source,
            train_cases,
            step_stride=args.step_stride,
            normalization=args.flux_loss_normalization,
        ).to(device)
    training_flux_normalizer = (
        None if target_name == "projected_residual" else flux_normalizer
    )
    boundary_exchange_normalizer = None
    if target_name == "projected_residual" and source.face_flux_integral is not None:
        boundary_exchange_normalizer = (
            PrimitiveNormalizer.from_source_boundary_exchange(
                source,
                train_cases,
                step_stride=args.step_stride,
            ).to(device)
        )
    training_boundary_exchange_normalizer = (
        boundary_exchange_normalizer
        if args.boundary_exchange_loss_weight > 0.0
        else None
    )
    input_coordinates = (
        "primitive" if model_name == "cpgnet" else args.input_coordinates
    )
    input_normalization = (
        "empirical" if model_name == "cpgnet" else args.input_normalization
    )
    input_normalizer = PrimitiveNormalizer.from_source_inputs(
        source,
        train_cases,
        step_stride=args.step_stride,
        coordinates=input_coordinates,
        normalization=input_normalization,
    ).to(device)
    model = build_model(model_name, target, args, input_normalizer).to(device)
    continuation_initialization = None
    if args.continuation_checkpoint is not None:
        continuation_initialization = load_continuation_checkpoint(
            model,
            args.continuation_checkpoint,
            model_name=model_name,
            target_name=target_name,
            args=args,
            train_cases=train_cases,
            val_cases=val_cases,
            test_cases=test_cases,
        )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    adapter = make_target_adapter(
        target_name,
        positive_transform=args.positive_transform,
        flux_correction_scale=args.flux_correction_scale,
        flux_correction_scale_floor=args.flux_correction_scale_floor,
        flux_gauge_mode=args.flux_gauge_mode,
        interface_flux_mode=args.interface_flux_mode,
    ).to(device)
    optimizer_class: type[torch.optim.Optimizer]
    optimizer_class = torch.optim.Adam if model_name == "cpgnet" else torch.optim.AdamW
    learning_rate = args.lr
    if learning_rate is None:
        learning_rate = 1.0e-4 if model_name == "cpgnet" else 1.0e-3
    weight_decay = args.weight_decay
    if weight_decay is None:
        weight_decay = 0.0 if model_name == "cpgnet" else 1.0e-5
    optimizer = optimizer_class(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    input_noise_mode = "additive" if model_name == "cpgnet" else "admissible_log_normal"

    history: list[dict[str, Any]] = []
    best_state = clone_state_dict_cpu(model)
    best_epoch = 0
    best_score = float("inf")
    best_val_eval: dict[str, float] | None = None
    best_val_rollout: dict[str, Any] | None = None

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            adapter,
            train_loader,
            normalizer,
            optimizer,
            device,
            args.grad_clip,
            input_noise_std=args.input_noise_std,
            input_noise_mode=input_noise_mode,
            loss_coordinates=args.loss_coordinates,
            target_supervision=args.target_supervision,
            flux_normalizer=training_flux_normalizer,
            flux_loss_weight=args.flux_loss_weight,
            boundary_exchange_normalizer=training_boundary_exchange_normalizer,
            boundary_exchange_loss_weight=args.boundary_exchange_loss_weight,
        )
        val_metrics = evaluate_one_step(
            model,
            adapter,
            val_loader,
            normalizer,
            device,
            loss_coordinates=args.loss_coordinates,
            target_supervision=args.target_supervision,
            flux_normalizer=flux_normalizer,
            flux_loss_weight=args.flux_loss_weight,
            boundary_exchange_normalizer=boundary_exchange_normalizer,
            boundary_exchange_loss_weight=args.boundary_exchange_loss_weight,
        )
        val_rollout_rows = rollout_cases(
            model, adapter, source, val_cases, args, device
        )
        val_rollout_summary = summarize_rollouts(val_rollout_rows)
        selection_score = rollout_selection_score(
            val_rollout_summary, val_metrics["loss"]
        )
        if selection_score < best_score or best_epoch == 0:
            best_score = selection_score
            best_epoch = epoch
            best_state = clone_state_dict_cpu(model)
            best_val_eval = dict(val_metrics)
            best_val_rollout = dict(val_rollout_summary)

        row = {
            "epoch": epoch,
            "stage": "one_step",
            "stage_epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_state_loss": train_metrics["state_loss"],
            "train_admissibility_loss": float("nan"),
            "train_burn_in_relative_l2": float("nan"),
            "train_burn_in_nonpositive_sample_fraction": float("nan"),
            "train_burn_in_min_density": float("nan"),
            "train_burn_in_min_pressure": float("nan"),
            "train_flux_loss": train_metrics["flux_loss"],
            "train_flux_reference_mse": train_metrics["flux_reference_mse"],
            "train_flux_divergence_active_mse": train_metrics[
                "flux_divergence_active_mse"
            ],
            "train_flux_gauge_mse": train_metrics["flux_gauge_mse"],
            "train_flux_gauge_fraction": train_metrics["flux_gauge_fraction"],
            "train_relative_l2": train_metrics["relative_l2"],
            "val_loss": val_metrics["loss"],
            "val_state_loss": val_metrics["state_loss"],
            "val_flux_loss": val_metrics["flux_loss"],
            "val_flux_reference_mse": val_metrics["flux_reference_mse"],
            "val_flux_divergence_active_mse": val_metrics["flux_divergence_active_mse"],
            "val_flux_gauge_mse": val_metrics["flux_gauge_mse"],
            "val_flux_gauge_fraction": val_metrics["flux_gauge_fraction"],
            "val_relative_l2": val_metrics["relative_l2"],
            "val_rollout_relative_l2_mean": val_rollout_summary[
                "rollout_relative_l2_mean"
            ],
            "val_rollout_relative_l2_final": val_rollout_summary[
                "rollout_relative_l2_final"
            ],
            "val_completion_fraction": val_rollout_summary["completion_fraction"],
            "val_survival_fraction_mean": val_rollout_summary["survival_fraction_mean"],
            "val_shock_position_mae": val_rollout_summary["shock_position_mae"],
            "val_conservative_total_error_final": val_rollout_summary[
                "conservative_total_error_final"
            ],
            "val_min_density": val_rollout_summary["min_density"],
            "val_min_pressure": val_rollout_summary["min_pressure"],
            "val_limiter_activation_fraction": val_rollout_summary[
                "limiter_activation_fraction"
            ],
            "selection_score": selection_score,
            "is_best_checkpoint": epoch == best_epoch,
        }
        history.append(row)
        write_history(run_dir / "history.csv", history)
        print(
            f"{model_name:15s} {target_name:27s} epoch {epoch:03d}/{args.epochs:03d} "
            f"train_loss={row['train_loss']:.4e} val_loss={row['val_loss']:.4e} "
            f"val_rollout_final={row['val_rollout_relative_l2_final']:.4e} "
            f"val_survival={row['val_survival_fraction_mean']:.3f} "
            f"best_epoch={best_epoch:03d}",
            flush=True,
        )

    if unroll_loader is not None:
        load_state_dict_cpu(model, best_state, device)
        optimizer = optimizer_class(
            model.parameters(),
            lr=learning_rate * args.unroll_lr_factor,
            weight_decay=weight_decay,
        )
        for stage_epoch in range(1, args.unroll_epochs + 1):
            epoch = args.epochs + stage_epoch
            train_metrics = train_unrolled_epoch(
                model,
                adapter,
                unroll_loader,
                normalizer,
                optimizer,
                device,
                args.grad_clip,
                input_noise_std=args.input_noise_std * args.unroll_noise_factor,
                input_noise_mode=input_noise_mode,
                loss_coordinates=args.loss_coordinates,
                burn_in_steps=args.unroll_burn_in_steps,
                burn_in_mode=args.unroll_burn_in_mode,
                admissibility_weight=args.unroll_admissibility_weight,
                density_margin=unroll_density_margin,
                pressure_margin=unroll_pressure_margin,
                boundary_exchange_normalizer=training_boundary_exchange_normalizer,
                boundary_exchange_loss_weight=args.boundary_exchange_loss_weight,
            )
            val_metrics = evaluate_one_step(
                model,
                adapter,
                val_loader,
                normalizer,
                device,
                loss_coordinates=args.loss_coordinates,
                target_supervision=args.target_supervision,
                flux_normalizer=flux_normalizer,
                flux_loss_weight=args.flux_loss_weight,
                boundary_exchange_normalizer=boundary_exchange_normalizer,
                boundary_exchange_loss_weight=args.boundary_exchange_loss_weight,
            )
            val_rollout_rows = rollout_cases(
                model, adapter, source, val_cases, args, device
            )
            val_rollout_summary = summarize_rollouts(val_rollout_rows)
            selection_score = rollout_selection_score(
                val_rollout_summary, val_metrics["loss"]
            )
            if selection_score < best_score:
                best_score = selection_score
                best_epoch = epoch
                best_state = clone_state_dict_cpu(model)
                best_val_eval = dict(val_metrics)
                best_val_rollout = dict(val_rollout_summary)

            row = {
                "epoch": epoch,
                "stage": "autoregressive",
                "stage_epoch": stage_epoch,
                "train_loss": train_metrics["loss"],
                "train_state_loss": train_metrics["state_loss"],
                "train_admissibility_loss": train_metrics["admissibility_loss"],
                "train_burn_in_relative_l2": train_metrics["burn_in_relative_l2"],
                "train_burn_in_nonpositive_sample_fraction": train_metrics[
                    "burn_in_nonpositive_sample_fraction"
                ],
                "train_burn_in_min_density": train_metrics["burn_in_min_density"],
                "train_burn_in_min_pressure": train_metrics["burn_in_min_pressure"],
                "train_relative_l2": train_metrics["relative_l2"],
                "val_loss": val_metrics["loss"],
                "val_relative_l2": val_metrics["relative_l2"],
                "val_rollout_relative_l2_mean": val_rollout_summary[
                    "rollout_relative_l2_mean"
                ],
                "val_rollout_relative_l2_final": val_rollout_summary[
                    "rollout_relative_l2_final"
                ],
                "val_completion_fraction": val_rollout_summary["completion_fraction"],
                "val_survival_fraction_mean": val_rollout_summary[
                    "survival_fraction_mean"
                ],
                "val_shock_position_mae": val_rollout_summary["shock_position_mae"],
                "val_conservative_total_error_final": val_rollout_summary[
                    "conservative_total_error_final"
                ],
                "val_min_density": val_rollout_summary["min_density"],
                "val_min_pressure": val_rollout_summary["min_pressure"],
                "val_limiter_activation_fraction": val_rollout_summary[
                    "limiter_activation_fraction"
                ],
                "selection_score": selection_score,
                "is_best_checkpoint": epoch == best_epoch,
            }
            history.append(row)
            write_history(run_dir / "history.csv", history)
            train_loss_value = row["train_loss"]
            val_loss_value = row["val_loss"]
            val_rollout_value = row["val_rollout_relative_l2_final"]
            print(
                f"{model_name:15s} {target_name:27s} "
                f"autoregressive {stage_epoch:03d}/{args.unroll_epochs:03d} "
                f"train_loss={train_loss_value:.4e} "
                f"val_loss={val_loss_value:.4e} "
                f"val_rollout_final={val_rollout_value:.4e} "
                f"val_survival={row['val_survival_fraction_mean']:.3f} "
                f"best_epoch={best_epoch:03d}",
                flush=True,
            )

    load_state_dict_cpu(model, best_state, device)
    rollout_rows = rollout_cases(model, adapter, source, test_cases, args, device)
    rollout_summary = summarize_rollouts(rollout_rows)
    final_eval = evaluate_one_step(
        model,
        adapter,
        test_loader,
        normalizer,
        device,
        loss_coordinates=args.loss_coordinates,
        target_supervision=args.target_supervision,
        flux_normalizer=flux_normalizer,
        flux_loss_weight=args.flux_loss_weight,
        boundary_exchange_normalizer=boundary_exchange_normalizer,
        boundary_exchange_loss_weight=args.boundary_exchange_loss_weight,
    )
    runtime = time.perf_counter() - start
    if best_val_eval is None:
        best_val_eval = evaluate_one_step(
            model,
            adapter,
            val_loader,
            normalizer,
            device,
            loss_coordinates=args.loss_coordinates,
            target_supervision=args.target_supervision,
            flux_normalizer=flux_normalizer,
            flux_loss_weight=args.flux_loss_weight,
            boundary_exchange_normalizer=boundary_exchange_normalizer,
            boundary_exchange_loss_weight=args.boundary_exchange_loss_weight,
        )
    if best_val_rollout is None:
        best_val_rollout = summarize_rollouts(
            rollout_cases(model, adapter, source, val_cases, args, device)
        )

    payload = {
        "continuation_initialization": continuation_initialization,
        "boundary_exchange_loss_weight": args.boundary_exchange_loss_weight,
        "boundary_exchange_normalizer_std": (
            None
            if boundary_exchange_normalizer is None
            else boundary_exchange_normalizer.std.detach().cpu().reshape(3)
        ),
        "model": model_name,
        "model_implementation": implementation,
        "parameter_count": parameter_count,
        "target": target_name,
        "target_type": target_name,
        "input_coordinates": args.input_coordinates,
        "recurrent_coordinates": args.recurrent_coordinates,
        "predicted_quantity": predicted_quantity_name(target_name),
        "target_supervision": args.target_supervision,
        "flux_gauge_mode": args.flux_gauge_mode,
        "interface_flux_mode": args.interface_flux_mode,
        "loss_coordinates": args.loss_coordinates,
        "input_normalization": input_normalization,
        "loss_normalization": args.loss_normalization,
        "flux_loss_normalization": args.flux_loss_normalization,
        "flux_loss_weight": args.flux_loss_weight,
        "status": "ok",
        "data_path": args.data_path,
        "train_cases": train_cases,
        "val_cases": val_cases,
        "test_cases": test_cases,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "seed_count": 1,
        "epochs": args.epochs,
        "unroll_epochs": args.unroll_epochs,
        "unroll_steps": args.unroll_steps if args.unroll_epochs > 0 else 0,
        "unroll_burn_in_steps": (
            args.unroll_burn_in_steps if args.unroll_epochs > 0 else 0
        ),
        "unroll_burn_in_mode": (
            args.unroll_burn_in_mode if args.unroll_burn_in_steps > 0 else "none"
        ),
        "unroll_lr_factor": args.unroll_lr_factor,
        "unroll_noise_factor": args.unroll_noise_factor,
        "unroll_optimizer_reset": args.unroll_epochs > 0,
        "unroll_admissibility_weight": args.unroll_admissibility_weight,
        "unroll_admissibility_margin_fraction": (
            args.unroll_admissibility_margin_fraction
        ),
        "unroll_density_margin": unroll_density_margin,
        "unroll_pressure_margin": unroll_pressure_margin,
        "batch_size": args.batch_size,
        "optimizer": optimizer_class.__name__,
        "lr": learning_rate,
        "weight_decay": weight_decay,
        "positive_transform": (
            None if target_name == "relative_interface" else args.positive_transform
        ),
        "interface_trace_parameterization": (
            "multiplicative_exp_density_pressure_sound_speed_velocity"
            if target_name == "relative_interface"
            else None
        ),
        "input_noise_std": args.input_noise_std,
        "input_noise_mode": input_noise_mode,
        "initial_frame_weight": args.initial_frame_weight,
        "flux_correction_scale": args.flux_correction_scale,
        "flux_correction_scale_floor": args.flux_correction_scale_floor,
        "step_stride": args.step_stride,
        "rollout_final_frame": args.rollout_final_frame,
        "device": str(device),
        "checkpoint_selection": {
            "metric": (
                "completed_admissible_val_rollout_then_mean_survival_then_one_step"
            ),
            "best_epoch": best_epoch,
            "best_score": best_score,
            "best_val_one_step": best_val_eval,
            "best_val_rollout": best_val_rollout,
        },
        "normalizer_mean": normalizer.mean.detach().cpu().reshape(3),
        "normalizer_std": normalizer.std.detach().cpu().reshape(3),
        "flux_normalizer_mean": (
            None
            if flux_normalizer is None
            else flux_normalizer.mean.detach().cpu().reshape(3)
        ),
        "flux_normalizer_std": (
            None
            if flux_normalizer is None
            else flux_normalizer.std.detach().cpu().reshape(3)
        ),
        "input_normalizer_mean": input_normalizer.mean.detach().cpu().reshape(3),
        "input_normalizer_std": input_normalizer.std.detach().cpu().reshape(3),
        "history": history,
        "one_step": final_eval,
        "rollout": rollout_summary,
        "rollout_cases": rollout_rows,
        "runtime_seconds": runtime,
    }

    write_history(run_dir / "history.csv", history)
    (run_dir / "metrics.json").write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if args.save_checkpoints:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model": model_name,
                "model_implementation": implementation,
                "parameter_count": parameter_count,
                "target": target_name,
                "args": vars(args),
                "train_cases": train_cases,
                "val_cases": val_cases,
                "test_cases": test_cases,
                "best_epoch": best_epoch,
                "selection_metric": (
                    "completed_admissible_val_rollout_then_mean_survival_then_one_step"
                ),
                "input_noise_mode": input_noise_mode,
                "input_coordinates": args.input_coordinates,
                "recurrent_coordinates": args.recurrent_coordinates,
                "predicted_quantity": predicted_quantity_name(target_name),
                "target_supervision": args.target_supervision,
                "flux_gauge_mode": args.flux_gauge_mode,
                "interface_flux_mode": args.interface_flux_mode,
                "loss_coordinates": args.loss_coordinates,
                "input_normalization": input_normalization,
                "loss_normalization": args.loss_normalization,
                "flux_loss_normalization": args.flux_loss_normalization,
                "flux_loss_weight": args.flux_loss_weight,
                "normalizer_mean": normalizer.mean.detach().cpu(),
                "normalizer_std": normalizer.std.detach().cpu(),
                "flux_normalizer_mean": (
                    None
                    if flux_normalizer is None
                    else flux_normalizer.mean.detach().cpu()
                ),
                "flux_normalizer_std": (
                    None
                    if flux_normalizer is None
                    else flux_normalizer.std.detach().cpu()
                ),
                "input_normalizer_mean": input_normalizer.mean.detach().cpu(),
                "input_normalizer_std": input_normalizer.std.detach().cpu(),
            },
            run_dir / "checkpoint.pt",
        )

    row = {
        "continuation_initialized": continuation_initialization is not None,
        "continuation_source_step_stride": (
            None
            if continuation_initialization is None
            else continuation_initialization["source_step_stride"]
        ),
        "boundary_exchange_loss_weight": args.boundary_exchange_loss_weight,
        "one_step_boundary_exchange_loss": final_eval["boundary_exchange_loss"],
        "model": model_name,
        "model_implementation": implementation,
        "parameter_count": parameter_count,
        "target": target_name,
        "target_type": target_name,
        "input_coordinates": args.input_coordinates,
        "recurrent_coordinates": args.recurrent_coordinates,
        "predicted_quantity": predicted_quantity_name(target_name),
        "target_supervision": args.target_supervision,
        "flux_gauge_mode": args.flux_gauge_mode,
        "interface_flux_mode": args.interface_flux_mode,
        "loss_coordinates": args.loss_coordinates,
        "input_normalization": input_normalization,
        "loss_normalization": args.loss_normalization,
        "flux_loss_normalization": args.flux_loss_normalization,
        "flux_loss_weight": args.flux_loss_weight,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "seed_count": 1,
        "train_cases_count": int(train_cases.size),
        "val_cases_count": int(val_cases.size),
        "test_cases_count": int(test_cases.size),
        "best_epoch": best_epoch,
        "epochs": args.epochs,
        "unroll_epochs": args.unroll_epochs,
        "unroll_steps": args.unroll_steps if args.unroll_epochs > 0 else 0,
        "unroll_burn_in_steps": (
            args.unroll_burn_in_steps if args.unroll_epochs > 0 else 0
        ),
        "unroll_burn_in_mode": (
            args.unroll_burn_in_mode if args.unroll_burn_in_steps > 0 else "none"
        ),
        "unroll_lr_factor": args.unroll_lr_factor,
        "unroll_noise_factor": args.unroll_noise_factor,
        "unroll_optimizer_reset": args.unroll_epochs > 0,
        "unroll_admissibility_weight": args.unroll_admissibility_weight,
        "unroll_admissibility_margin_fraction": (
            args.unroll_admissibility_margin_fraction
        ),
        "unroll_density_margin": unroll_density_margin,
        "unroll_pressure_margin": unroll_pressure_margin,
        "unroll_checkpoint_selected": best_epoch > args.epochs,
        "optimizer": optimizer_class.__name__,
        "lr": learning_rate,
        "weight_decay": weight_decay,
        "selection_metric": (
            "completed_admissible_val_rollout_then_mean_survival_then_one_step"
        ),
        "selection_score": best_score,
        "step_stride": args.step_stride,
        "rollout_final_frame": args.rollout_final_frame,
        "input_noise_std": args.input_noise_std,
        "input_noise_mode": input_noise_mode,
        "initial_frame_weight": args.initial_frame_weight,
        "flux_correction_scale": args.flux_correction_scale,
        "flux_correction_scale_floor": args.flux_correction_scale_floor,
        "status": "ok",
        "one_step_loss": final_eval["loss"],
        "one_step_state_loss": final_eval["state_loss"],
        "one_step_flux_loss": final_eval["flux_loss"],
        "one_step_flux_reference_mse": final_eval["flux_reference_mse"],
        "one_step_flux_divergence_active_mse": final_eval["flux_divergence_active_mse"],
        "one_step_flux_gauge_mse": final_eval["flux_gauge_mse"],
        "one_step_flux_gauge_fraction": final_eval["flux_gauge_fraction"],
        "one_step_boundary_exchange_relative_l2": final_eval[
            "boundary_exchange_relative_l2"
        ],
        "one_step_projection_closure_max_abs": final_eval["projection_closure_max_abs"],
        "one_step_relative_l2": final_eval["relative_l2"],
        "test_loss": final_eval["loss"],
        "test_relative_l2": final_eval["relative_l2"],
        "one_step_min_density": final_eval["min_density"],
        "one_step_min_pressure": final_eval["min_pressure"],
        "one_step_proposed_min_density": final_eval["proposed_min_density"],
        "one_step_proposed_min_pressure": final_eval["proposed_min_pressure"],
        "one_step_raw_min_density": final_eval["raw_min_density"],
        "one_step_raw_min_pressure": final_eval["raw_min_pressure"],
        "one_step_num_nonpositive_proposed_density": final_eval[
            "num_nonpositive_proposed_density"
        ],
        "one_step_num_nonpositive_proposed_pressure": final_eval[
            "num_nonpositive_proposed_pressure"
        ],
        "one_step_num_proposed_density_near_floor": final_eval[
            "num_proposed_density_near_floor"
        ],
        "one_step_num_proposed_pressure_near_floor": final_eval[
            "num_proposed_pressure_near_floor"
        ],
        "one_step_num_nonpositive_raw_density": final_eval[
            "num_nonpositive_raw_density"
        ],
        "one_step_num_nonpositive_raw_pressure": final_eval[
            "num_nonpositive_raw_pressure"
        ],
        "one_step_num_raw_density_near_floor": final_eval["num_raw_density_near_floor"],
        "one_step_num_raw_pressure_near_floor": final_eval[
            "num_raw_pressure_near_floor"
        ],
        "one_step_limiter_theta_mean": final_eval["limiter_theta_mean"],
        "one_step_limiter_theta_min": final_eval["limiter_theta_min"],
        "one_step_limiter_activation_fraction": final_eval[
            "limiter_activation_fraction"
        ],
        "one_step_flux_correction_abs_over_bound_mean": final_eval[
            "flux_correction_abs_over_bound_mean"
        ],
        "one_step_flux_correction_abs_over_bound_max": final_eval[
            "flux_correction_abs_over_bound_max"
        ],
        "one_step_flux_correction_saturation_fraction": final_eval[
            "flux_correction_saturation_fraction"
        ],
        "runtime_seconds": runtime,
    }
    row.update(rollout_summary)
    return row


def requested_values(value: str, choices: tuple[str, ...]) -> list[str]:
    if value == "all":
        return list(choices)
    return [value]


def requested_experiment_pairs(
    model_value: str, target_value: str
) -> list[tuple[str, str]]:
    models = requested_values(model_value, MODEL_CHOICES)
    pairs: list[tuple[str, str]] = []
    for model_name in models:
        if model_name == "cpgnet":
            if target_value not in ("all", CPG_TARGET):
                raise ValueError("--model cpgnet only supports --target cpg_interface")
            pairs.append((model_name, CPG_TARGET))
            continue
        if target_value == CPG_TARGET:
            raise ValueError("--target cpg_interface is reserved for --model cpgnet")
        for target_name in requested_values(target_value, TARGET_CHOICES):
            pairs.append((model_name, target_name))
    return pairs


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.continuation_checkpoint is not None and (
        args.model != "fno" or args.target != "residual"
    ):
        raise ValueError(
            "--continuation-checkpoint currently requires --model fno --target residual"
        )
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.unroll_epochs < 0:
        raise ValueError("--unroll-epochs must be >= 0")
    if args.unroll_steps < 2:
        raise ValueError("--unroll-steps must be >= 2")
    if args.unroll_burn_in_steps < 0:
        raise ValueError("--unroll-burn-in-steps must be >= 0")
    if args.unroll_lr_factor <= 0.0:
        raise ValueError("--unroll-lr-factor must be positive")
    if args.unroll_noise_factor < 0.0:
        raise ValueError("--unroll-noise-factor must be nonnegative")
    if args.unroll_admissibility_weight < 0.0:
        raise ValueError("--unroll-admissibility-weight must be nonnegative")
    if args.unroll_admissibility_margin_fraction <= 0.0:
        raise ValueError("--unroll-admissibility-margin-fraction must be positive")
    if args.initial_frame_weight <= 0.0:
        raise ValueError("--initial-frame-weight must be positive")
    if args.unroll_admissibility_weight > 0.0 and args.unroll_epochs == 0:
        raise ValueError("--unroll-admissibility-weight requires --unroll-epochs > 0")
    if args.unroll_burn_in_steps > 0 and args.unroll_epochs == 0:
        raise ValueError("--unroll-burn-in-steps requires --unroll-epochs > 0")
    if args.unroll_burn_in_mode == "teacher" and args.unroll_burn_in_steps == 0:
        raise ValueError(
            "--unroll-burn-in-mode teacher requires --unroll-burn-in-steps > 0"
        )
    if args.unroll_epochs > 0 and args.target_supervision != "state":
        raise ValueError(
            "autoregressive fine-tuning currently requires --target-supervision state"
        )
    if args.model in ("cpgnet", "all") and (
        args.input_coordinates != "primitive"
        or args.loss_coordinates != "primitive"
        or args.recurrent_coordinates != "primitive"
        or args.input_normalization != "none"
        or args.loss_normalization != "empirical"
    ):
        raise ValueError(
            "non-default coordinate diagnostics require --model fno; "
            "the corrected CPGNet contract remains primitive/primitive"
        )
    if args.input_noise_std < 0.0:
        raise ValueError("--input-noise-std must be nonnegative")
    if args.boundary_exchange_loss_weight < 0.0:
        raise ValueError("--boundary-exchange-loss-weight must be nonnegative")
    if args.boundary_exchange_loss_weight > 0.0:
        if args.model != "fno" or args.target != "projected_residual":
            raise ValueError(
                "boundary-exchange supervision requires "
                "--model fno --target projected_residual"
            )
        if args.target_supervision != "state":
            raise ValueError(
                "boundary-exchange supervision requires --target-supervision state"
            )
        if args.input_noise_std != 0.0:
            raise ValueError("boundary-exchange labels require --input-noise-std 0")
    if args.flux_loss_weight <= 0.0:
        raise ValueError("--flux-loss-weight must be positive")
    if args.target_supervision != "state":
        if args.model != "fno" or args.target != "flux":
            raise ValueError(
                "direct/joint flux supervision requires --model fno --target flux"
            )
        if args.input_noise_std != 0.0:
            raise ValueError(
                "direct solver-flux labels require --input-noise-std 0; "
                "noise changes the decoded update without changing the clean flux label"
            )
    if args.flux_gauge_mode != "raw" and args.target != "flux":
        raise ValueError("--flux-gauge-mode canonical requires --target flux")
    if args.interface_flux_mode != "rusanov" and args.target != "relative_interface":
        raise ValueError(
            "--interface-flux-mode central requires --target relative_interface"
        )
    if args.flux_correction_scale < 0.0:
        raise ValueError("--flux-correction-scale must be nonnegative")
    if args.flux_correction_scale_floor <= 0.0:
        raise ValueError("--flux-correction-scale-floor must be positive")
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    set_seed(args.seed)
    device = select_device(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source = load_euler1d_npz(args.data_path)
    if args.boundary_exchange_loss_weight > 0.0 and source.face_flux_integral is None:
        raise ValueError(
            "boundary-exchange supervision requires face_flux_integral in the dataset"
        )
    if args.target_supervision != "state" and source.face_flux_integral is None:
        raise ValueError(
            "direct/joint flux supervision requires face_flux_integral in the dataset"
        )
    train_cases, val_cases, test_cases = split_cases(source, args)
    experiment_pairs = requested_experiment_pairs(args.model, args.target)

    print(
        json.dumps(
            {
                "data_path": str(args.data_path),
                "data_shape": list(source.data.shape),
                "device": str(device),
                "train_cases": train_cases.tolist(),
                "val_cases": val_cases.tolist(),
                "test_cases": test_cases.tolist(),
                "seed": args.seed,
                "split_seed": args.split_seed,
                "experiment_pairs": experiment_pairs,
                "step_stride": args.step_stride,
                "rollout_final_frame": args.rollout_final_frame,
                "input_coordinates": args.input_coordinates,
                "recurrent_coordinates": args.recurrent_coordinates,
                "target_supervision": args.target_supervision,
                "flux_gauge_mode": args.flux_gauge_mode,
                "interface_flux_mode": args.interface_flux_mode,
                "loss_coordinates": args.loss_coordinates,
                "input_normalization": args.input_normalization,
                "loss_normalization": args.loss_normalization,
                "flux_loss_normalization": args.flux_loss_normalization,
                "flux_loss_weight": args.flux_loss_weight,
                "input_noise_std": args.input_noise_std,
                "initial_frame_weight": args.initial_frame_weight,
                "flux_correction_scale": args.flux_correction_scale,
                "flux_correction_scale_floor": args.flux_correction_scale_floor,
                "output_dir": str(args.output_dir),
            },
            indent=2,
        ),
        flush=True,
    )

    summary_rows: list[dict[str, Any]] = []
    for model_name, target_name in experiment_pairs:
        try:
            summary_rows.append(
                run_single(
                    source,
                    train_cases,
                    val_cases,
                    test_cases,
                    model_name,
                    target_name,
                    args,
                    device,
                )
            )
        except Exception as exc:
            failure = {
                "model": model_name,
                "model_implementation": model_implementation_name(model_name, args),
                "target": target_name,
                "target_type": target_name,
                "input_coordinates": args.input_coordinates,
                "recurrent_coordinates": args.recurrent_coordinates,
                "predicted_quantity": predicted_quantity_name(target_name),
                "target_supervision": args.target_supervision,
                "flux_gauge_mode": args.flux_gauge_mode,
                "interface_flux_mode": args.interface_flux_mode,
                "loss_coordinates": args.loss_coordinates,
                "input_normalization": args.input_normalization,
                "loss_normalization": args.loss_normalization,
                "flux_loss_normalization": args.flux_loss_normalization,
                "flux_loss_weight": args.flux_loss_weight,
                "seed": args.seed,
                "split_seed": args.split_seed,
                "seed_count": 1,
                "status": "failed",
                "error": repr(exc),
            }
            summary_rows.append(failure)
            print(json.dumps(failure, indent=2), flush=True)
            if args.fail_fast:
                raise

    write_summary_csv(args.output_dir / "summary.csv", summary_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(json_ready(summary_rows), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "summary_csv": str(args.output_dir / "summary.csv"),
                "summary_json": str(args.output_dir / "summary.json"),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
