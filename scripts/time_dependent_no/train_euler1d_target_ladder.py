"""Train fixed-step 1D Euler target heads on collaborator data.

This script is the light pilot harness for the solver-facing target ladder:

* residual: conservative state residual over one fixed coarse step
* flux: predicted face flux, followed by a fixed conservative FV update
* physical_flux_correction: base Rusanov flux plus bounded learned correction
* interface: predicted face states, followed by Rusanov flux and FV update

The model is not conditioned on ``dt``. ``--step-stride`` selects a fixed
coarse-step operator from saved trajectory frames, and all targets are
supervised only through the next primitive state at that stride.
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
from torch.utils.data import DataLoader

import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utility.time_dependent_no.euler1d import (
    Euler1DBatch,
    conservative_to_primitive,
    make_euler1d_batch,
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
    CPGStylePilotEuler1DHead,
    Euler1DTarget,
    FNOEuler1DHead,
)
from utility.time_dependent_no.euler1d_targets import make_target_adapter


EPS = 1.0e-12
NEAR_FLOOR = 1.0e-6
LIMITER_ACTIVE_TOL = 1.0e-6
CORRECTION_SATURATION_TOL = 0.95
MODEL_CHOICES = ("cpgnet", "fno")
ARG_MODEL_CHOICES = (*MODEL_CHOICES, "cpg_style_pilot")
TARGET_CHOICES = (
    "residual",
    "primitive_residual",
    "limited_residual",
    "flux",
    "limited_flux",
    "physical_flux_correction",
    "interface",
    "positive_limited_interface",
)
CPG_TARGET = "cpg_interface"
ARG_TARGET_CHOICES = ("state", *TARGET_CHOICES, CPG_TARGET)


@dataclass(frozen=True)
class PrimitiveNormalizer:
    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def from_source(
        cls,
        source: Euler1DNPZ,
        case_indices: np.ndarray,
        *,
        step_stride: int = 1,
    ) -> "PrimitiveNormalizer":
        if step_stride < 1 or step_stride >= source.num_frames:
            raise ValueError("step_stride must be in [1, num_frames - 1]")
        targets = source.data[case_indices, step_stride:]
        mean = torch.as_tensor(targets.mean(axis=(0, 1, 2)), dtype=torch.float32)
        std_np = targets.std(axis=(0, 1, 2))
        std = torch.as_tensor(np.maximum(std_np, 1.0e-6), dtype=torch.float32)
        return cls(mean=mean.reshape(1, 1, 3), std=std.reshape(1, 1, 3))

    @classmethod
    def from_source_inputs(
        cls,
        source: Euler1DNPZ,
        case_indices: np.ndarray,
        *,
        step_stride: int = 1,
    ) -> "PrimitiveNormalizer":
        if step_stride < 1 or step_stride >= source.num_frames:
            raise ValueError("step_stride must be in [1, num_frames - 1]")
        inputs = source.data[case_indices, :-step_stride]
        mean = torch.as_tensor(inputs.mean(axis=(0, 1, 2)), dtype=torch.float32)
        std_np = inputs.std(axis=(0, 1, 2))
        std = torch.as_tensor(np.maximum(std_np, 1.0e-6), dtype=torch.float32)
        return cls(mean=mean.reshape(1, 1, 3), std=std.reshape(1, 1, 3))

    def to(self, device: torch.device) -> "PrimitiveNormalizer":
        return PrimitiveNormalizer(mean=self.mean.to(device), std=self.std.to(device))

    def mse(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction_norm = (prediction - self.mean) / self.std
        target_norm = (target - self.mean) / self.std
        return torch.mean((prediction_norm - target_norm).square())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    date = datetime.now().strftime("%Y%m%d")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "artifacts"
        / "time_dependent_no"
        / f"euler1d_target_ladder_{date}",
    )
    parser.add_argument("--model", choices=(*ARG_MODEL_CHOICES, "all"), default="all")
    parser.add_argument("--target", choices=(*ARG_TARGET_CHOICES, "all"), default="all")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument(
        "--unroll-epochs",
        type=int,
        default=0,
        help="CPGNet-only autoregressive fine-tuning epochs after one-step training.",
    )
    parser.add_argument(
        "--unroll-steps",
        type=int,
        default=3,
        help="Number of differentiable CPGNet rollout steps during fine-tuning.",
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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Default: 1e-4 for CPGNet and 1e-3 for FNO/pilot heads.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Default: 0 for CPGNet and 1e-5 for FNO/pilot heads.",
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
    parser.add_argument("--seed", type=int, default=20260707)
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
    return parser.parse_args(argv)


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
    rng = np.random.default_rng(args.seed)
    permutation = rng.permutation(source.num_cases)
    train_end = args.train_cases
    val_end = train_end + args.val_cases
    train = np.sort(permutation[:train_end]).astype(np.int64)
    val = np.sort(permutation[train_end:val_end]).astype(np.int64)
    test = np.sort(permutation[val_end:requested]).astype(np.int64)
    return train, val, test


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
    elif model_name == "cpg_style_pilot":
        model = CPGStylePilotEuler1DHead(
            target,
            hidden_dim=args.cpg_hidden_dim,
            message_passing_steps=args.cpg_message_passing_steps,
            mlp_layers=args.cpg_mlp_layers,
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
        )
    else:
        raise ValueError(f"unsupported model: {model_name}")

    if target in (
        "residual",
        "primitive_residual",
        "limited_residual",
        "flux",
        "limited_flux",
        "physical_flux_correction",
    ):
        zero_initialize_identity_update_output(model, target)
    return model


def zero_initialize_identity_update_output(
    model: nn.Module, target: Euler1DTarget
) -> None:
    """Start increment and flux heads from the no-update map."""

    if isinstance(model, CPGStylePilotEuler1DHead):
        module = (
            model.node_decoder
            if target in ("residual", "primitive_residual", "limited_residual")
            else model.face_decoder
        )
        zero_initialize_last_linear(module)
        return
    if isinstance(model, FNOEuler1DHead):
        nn.init.zeros_(model.model.fc2.weight)
        if model.model.fc2.bias is not None:
            nn.init.zeros_(model.model.fc2.bias)
        return
    zero_initialize_last_linear(model)


def zero_initialize_last_linear(module: nn.Module) -> None:
    last_linear = None
    for child in module.modules():
        if isinstance(child, nn.Linear):
            last_linear = child
    if last_linear is None:
        raise ValueError("residual model has no Linear output layer to initialize")
    nn.init.zeros_(last_linear.weight)
    if last_linear.bias is not None:
        nn.init.zeros_(last_linear.bias)


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
    rho_floor: float = 1.0e-6,
    pressure_floor: float = 1.0e-6,
) -> Euler1DBatch:
    """Return a denoising-training batch with noisy current primitives only."""

    if noise_std <= 0.0:
        return batch
    current = batch.current_primitive
    noise = torch.randn_like(current) * float(noise_std)
    noisy = current.clone()
    noisy[..., 0] = (current[..., 0] * torch.exp(noise[..., 0])).clamp_min(rho_floor)
    noisy[..., 1] = current[..., 1] + noise[..., 1]
    noisy[..., 2] = (current[..., 2] * torch.exp(noise[..., 2])).clamp_min(
        pressure_floor
    )
    return replace(batch, current_primitive=noisy)


def evaluate_one_step(
    model: nn.Module,
    adapter: nn.Module,
    loader: DataLoader,
    normalizer: PrimitiveNormalizer,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
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
            loss = normalizer.mse(prediction.primitive, batch.target_primitive)
            rel_l2 = relative_l2_torch(prediction.primitive, batch.target_primitive)
            batch_size = batch.current_primitive.shape[0]
            total_loss += finite_scalar(loss) * batch_size
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
    return {
        "loss": total_loss / max(total_samples, 1),
        "relative_l2": total_rel_l2 / max(total_samples, 1),
        "min_density": min_density,
        "min_pressure": min_pressure,
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
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_rel_l2 = 0.0
    total_samples = 0

    for batch in loader:
        batch = batch.to(device)
        if batch.target_primitive is None:
            raise RuntimeError("training batch is missing target_primitive")
        batch = apply_primitive_input_noise(batch, input_noise_std)
        raw = model(batch)
        prediction = adapter(raw, batch)
        loss = normalizer.mse(prediction.primitive, batch.target_primitive)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite training loss")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        with torch.no_grad():
            rel_l2 = relative_l2_torch(prediction.primitive, batch.target_primitive)
        batch_size = batch.current_primitive.shape[0]
        total_loss += finite_scalar(loss) * batch_size
        total_rel_l2 += float(rel_l2.detach().cpu().sum().item())
        total_samples += batch_size

    return {
        "loss": total_loss / max(total_samples, 1),
        "relative_l2": total_rel_l2 / max(total_samples, 1),
    }


def train_unrolled_epoch(
    model: nn.Module,
    adapter: nn.Module,
    loader: DataLoader,
    normalizer: PrimitiveNormalizer,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    input_noise_std: float = 0.0,
) -> dict[str, float]:
    """Train through a fixed autoregressive window without state detachment."""

    model.train()
    total_loss = 0.0
    total_rel_l2 = 0.0
    total_samples = 0

    for initial_batch, target_sequence, dt_sequence in loader:
        initial_batch = initial_batch.to(device)
        target_sequence = target_sequence.to(device)
        dt_sequence = dt_sequence.to(device)
        noisy_batch = apply_primitive_input_noise(initial_batch, input_noise_std)
        current = noisy_batch.current_primitive
        step_losses: list[torch.Tensor] = []
        step_relative_l2: list[torch.Tensor] = []

        optimizer.zero_grad(set_to_none=True)
        for step in range(target_sequence.shape[1]):
            step_batch = replace(
                initial_batch,
                current_primitive=current,
                target_primitive=target_sequence[:, step],
                dt=dt_sequence[:, step],
            )
            prediction = adapter(model(step_batch), step_batch)
            step_loss = normalizer.mse(prediction.primitive, target_sequence[:, step])
            step_losses.append(step_loss)
            step_relative_l2.append(
                relative_l2_torch(prediction.primitive, target_sequence[:, step])
            )
            current = prediction.primitive

        loss = torch.stack(step_losses).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite unrolled training loss")
        loss.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_size = initial_batch.current_primitive.shape[0]
        rel_l2 = torch.stack(step_relative_l2, dim=1).mean(dim=1)
        total_loss += finite_scalar(loss) * batch_size
        total_rel_l2 += float(rel_l2.detach().cpu().sum().item())
        total_samples += batch_size

    return {
        "loss": total_loss / max(total_samples, 1),
        "relative_l2": total_rel_l2 / max(total_samples, 1),
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
) -> dict[str, Any]:
    model.eval()
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
    current = torch.from_numpy(source.data[case_id, 0]).unsqueeze(0).to(device)
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
                dtype=current.dtype,
                device=device,
            )
            batch = make_euler1d_batch(
                current,
                x,
                dt,
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
            current = next_state.detach()

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
            "rollout_relative_l2_mean": float("nan"),
            "rollout_relative_l2_final": float("nan"),
            "min_density": float("nan"),
            "min_pressure": float("nan"),
            "max_abs_primitive": float("nan"),
            **proposed_metrics,
            **limiter_metrics,
            **flux_correction_metrics,
            "shock_position_mae": float("nan"),
            "conservative_total_error_final": float("nan"),
            "step_stride": int(step_stride),
            "final_frame": int(final_frame),
            "completed_horizon": False,
        }

    pred = np.stack(predictions, axis=0)
    truth_ids = np.arange(1, pred.shape[0] + 1, dtype=np.int64) * step_stride
    truth = source.data[case_id, truth_ids].astype(np.float64)
    rel_l2 = relative_l2_np(pred, truth)
    pred_front = pressure_front_position_np(pred, x_np)
    truth_front = pressure_front_position_np(truth, x_np)
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
        "rollout_relative_l2_mean": float(np.mean(rel_l2)),
        "rollout_relative_l2_final": float(rel_l2[-1]),
        "min_density": float(np.min(pred[..., 0])),
        "min_pressure": float(np.min(pred[..., 2])),
        "max_abs_primitive": float(np.max(np.abs(pred))),
        **proposed_metrics,
        **limiter_metrics,
        **flux_correction_metrics,
        "shock_position_mae": float(np.mean(np.abs(pred_front - truth_front))),
        "conservative_total_error_final": float(total_error),
        "step_stride": int(step_stride),
        "final_frame": int(truth_ids[-1]),
        "completed_horizon": bool(
            pred.shape[0] == max_steps and truth_ids[-1] == final_frame
        ),
        "truth_frame_ids": truth_ids.tolist(),
        "relative_l2_by_step": rel_l2.tolist(),
    }


def summarize_rollouts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "finite": False,
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
        "conservative_total_error_final",
        "limiter_theta_mean",
        "limiter_theta_min",
        "limiter_activation_fraction",
        "flux_correction_abs_over_bound_mean",
        "flux_correction_abs_over_bound_max",
        "flux_correction_saturation_fraction",
    )
    summary: dict[str, Any] = {
        "finite": all(bool(row["finite"]) for row in rows),
        "admissible": all(bool(row.get("admissible", True)) for row in rows),
        "num_cases": len(rows),
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
        "target",
        "target_type",
        "status",
        "seed",
        "seed_count",
        "train_cases_count",
        "val_cases_count",
        "test_cases_count",
        "best_epoch",
        "selection_metric",
        "selection_score",
        "step_stride",
        "rollout_final_frame",
        "input_noise_std",
        "one_step_loss",
        "one_step_relative_l2",
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
    if model_name == "cpg_style_pilot":
        return (
            "CPGStylePilotEuler1DHead"
            f"(hidden={args.cpg_hidden_dim},message_steps={args.cpg_message_passing_steps},"
            "deprecated=True)"
        )
    return model_name


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
        )
        for case_id in case_ids.tolist()
    ]


def rollout_selection_score(summary: dict[str, Any], one_step_loss: float) -> float:
    """Prioritize valid rollouts, using one-step fit only as a failure fallback."""

    value = float(summary.get("rollout_relative_l2_final", float("nan")))
    finite = bool(summary.get("finite", False))
    complete = bool(summary.get("completed_horizon", False))
    admissible = bool(summary.get("admissible", True))
    if finite and complete and admissible and math.isfinite(value):
        return value
    fallback = float(one_step_loss)
    if not math.isfinite(fallback):
        return float("inf")
    return (1.0e6 if finite else 1.0e12) + fallback


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
    run_dir = args.output_dir / f"{model_name}_{target_name}"
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
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
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
    if model_name == "cpgnet" and args.unroll_epochs > 0:
        unroll_dataset = Euler1DRolloutWindowDataset(
            source,
            case_indices=train_cases,
            step_stride=args.step_stride,
            rollout_steps=args.unroll_steps,
        )
        unroll_loader = DataLoader(
            unroll_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_euler1d_rollout_windows,
            generator=torch.Generator().manual_seed(args.seed + 1),
        )

    normalizer = PrimitiveNormalizer.from_source(
        source,
        train_cases,
        step_stride=args.step_stride,
    ).to(device)
    input_normalizer = PrimitiveNormalizer.from_source_inputs(
        source,
        train_cases,
        step_stride=args.step_stride,
    ).to(device)
    model = build_model(model_name, target, args, input_normalizer).to(device)
    adapter = make_target_adapter(
        target_name,
        positive_transform=args.positive_transform,
        flux_correction_scale=args.flux_correction_scale,
        flux_correction_scale_floor=args.flux_correction_scale_floor,
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
            args.input_noise_std,
        )
        val_metrics = evaluate_one_step(model, adapter, val_loader, normalizer, device)
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
            "train_relative_l2": train_metrics["relative_l2"],
            "val_loss": val_metrics["loss"],
            "val_relative_l2": val_metrics["relative_l2"],
            "val_rollout_relative_l2_mean": val_rollout_summary[
                "rollout_relative_l2_mean"
            ],
            "val_rollout_relative_l2_final": val_rollout_summary[
                "rollout_relative_l2_final"
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
        print(
            f"{model_name:15s} {target_name:27s} epoch {epoch:03d}/{args.epochs:03d} "
            f"train_loss={row['train_loss']:.4e} val_loss={row['val_loss']:.4e} "
            f"val_rollout_final={row['val_rollout_relative_l2_final']:.4e} "
            f"best_epoch={best_epoch:03d}",
            flush=True,
        )

    if unroll_loader is not None:
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] *= args.unroll_lr_factor
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
                args.input_noise_std * args.unroll_noise_factor,
            )
            val_metrics = evaluate_one_step(
                model, adapter, val_loader, normalizer, device
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
                "train_relative_l2": train_metrics["relative_l2"],
                "val_loss": val_metrics["loss"],
                "val_relative_l2": val_metrics["relative_l2"],
                "val_rollout_relative_l2_mean": val_rollout_summary[
                    "rollout_relative_l2_mean"
                ],
                "val_rollout_relative_l2_final": val_rollout_summary[
                    "rollout_relative_l2_final"
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
                f"best_epoch={best_epoch:03d}",
                flush=True,
            )

    load_state_dict_cpu(model, best_state, device)
    rollout_rows = rollout_cases(model, adapter, source, test_cases, args, device)
    rollout_summary = summarize_rollouts(rollout_rows)
    final_eval = evaluate_one_step(model, adapter, test_loader, normalizer, device)
    runtime = time.perf_counter() - start
    if best_val_eval is None:
        best_val_eval = evaluate_one_step(
            model, adapter, val_loader, normalizer, device
        )
    if best_val_rollout is None:
        best_val_rollout = summarize_rollouts(
            rollout_cases(model, adapter, source, val_cases, args, device)
        )

    payload = {
        "model": model_name,
        "model_implementation": implementation,
        "target": target_name,
        "target_type": target_name,
        "status": "ok",
        "data_path": args.data_path,
        "train_cases": train_cases,
        "val_cases": val_cases,
        "test_cases": test_cases,
        "seed": args.seed,
        "seed_count": 1,
        "epochs": args.epochs,
        "unroll_epochs": args.unroll_epochs if model_name == "cpgnet" else 0,
        "unroll_steps": args.unroll_steps if model_name == "cpgnet" else 0,
        "unroll_lr_factor": args.unroll_lr_factor,
        "unroll_noise_factor": args.unroll_noise_factor,
        "batch_size": args.batch_size,
        "optimizer": optimizer_class.__name__,
        "lr": learning_rate,
        "weight_decay": weight_decay,
        "positive_transform": args.positive_transform,
        "input_noise_std": args.input_noise_std,
        "flux_correction_scale": args.flux_correction_scale,
        "flux_correction_scale_floor": args.flux_correction_scale_floor,
        "step_stride": args.step_stride,
        "rollout_final_frame": args.rollout_final_frame,
        "device": str(device),
        "checkpoint_selection": {
            "metric": "completed_admissible_val_rollout_then_one_step_fallback",
            "best_epoch": best_epoch,
            "best_score": best_score,
            "best_val_one_step": best_val_eval,
            "best_val_rollout": best_val_rollout,
        },
        "normalizer_mean": normalizer.mean.detach().cpu().reshape(3),
        "normalizer_std": normalizer.std.detach().cpu().reshape(3),
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
                "target": target_name,
                "args": vars(args),
                "train_cases": train_cases,
                "val_cases": val_cases,
                "test_cases": test_cases,
                "best_epoch": best_epoch,
                "selection_metric": (
                    "completed_admissible_val_rollout_then_one_step_fallback"
                ),
                "normalizer_mean": normalizer.mean.detach().cpu(),
                "normalizer_std": normalizer.std.detach().cpu(),
                "input_normalizer_mean": input_normalizer.mean.detach().cpu(),
                "input_normalizer_std": input_normalizer.std.detach().cpu(),
            },
            run_dir / "checkpoint.pt",
        )

    row = {
        "model": model_name,
        "model_implementation": implementation,
        "target": target_name,
        "target_type": target_name,
        "seed": args.seed,
        "seed_count": 1,
        "train_cases_count": int(train_cases.size),
        "val_cases_count": int(val_cases.size),
        "test_cases_count": int(test_cases.size),
        "best_epoch": best_epoch,
        "epochs": args.epochs,
        "unroll_epochs": args.unroll_epochs if model_name == "cpgnet" else 0,
        "unroll_steps": args.unroll_steps if model_name == "cpgnet" else 0,
        "optimizer": optimizer_class.__name__,
        "lr": learning_rate,
        "weight_decay": weight_decay,
        "selection_metric": ("completed_admissible_val_rollout_then_one_step_fallback"),
        "selection_score": best_score,
        "step_stride": args.step_stride,
        "rollout_final_frame": args.rollout_final_frame,
        "input_noise_std": args.input_noise_std,
        "flux_correction_scale": args.flux_correction_scale,
        "flux_correction_scale_floor": args.flux_correction_scale_floor,
        "status": "ok",
        "one_step_loss": final_eval["loss"],
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
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.unroll_epochs < 0:
        raise ValueError("--unroll-epochs must be >= 0")
    if args.unroll_steps < 2:
        raise ValueError("--unroll-steps must be >= 2")
    if args.unroll_lr_factor <= 0.0:
        raise ValueError("--unroll-lr-factor must be positive")
    if args.unroll_noise_factor < 0.0:
        raise ValueError("--unroll-noise-factor must be nonnegative")
    if args.unroll_epochs > 0 and args.model not in ("cpgnet", "all"):
        raise ValueError("--unroll-epochs is only supported for CPGNet")
    if args.input_noise_std < 0.0:
        raise ValueError("--input-noise-std must be nonnegative")
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
                "experiment_pairs": experiment_pairs,
                "step_stride": args.step_stride,
                "rollout_final_frame": args.rollout_final_frame,
                "input_noise_std": args.input_noise_std,
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
                "seed": args.seed,
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
