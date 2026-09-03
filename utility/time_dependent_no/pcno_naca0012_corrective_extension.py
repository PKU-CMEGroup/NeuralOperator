"""Model-side primitives for ``B3B4_NACA_CM_EXT_20260902A``.

The closed five-arm successor remains untouched.  This module contains only
the new paired-target, pushforward, EMA, and PCNO--PDE-Refiner mechanisms.  It
does not load trajectory values, call SU2, or authorize protected populations.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from numbers import Integral
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from pcno.pcno import PCNO, compute_Fourier_modes
from utility.time_dependent_no.pcno_naca0012 import (
    NUM_DYNAMIC_FIELDS,
    NUM_STATIC_FEATURES,
    NACANormalization,
    NACAPCNOResidual,
    VerifiedNACAGeometry,
    recurrent_step,
)

EXTENSION_CONTRACT_SCHEMA = "time_dependent_no.naca_corrective_extension_contract.v1"
EXTENSION_EXPERIMENT_ID = "B3B4_NACA_CM_EXT_20260902A"
EXTENSION_CONTRACT_SHA256 = (
    "3d786bafff23cc08bbb4434a1f89253af3edde92e6637bcdcad8c263e97ae02c"
)
EXTENSION_CONTRACT_CANONICAL_SHA256 = (
    "417212298524ae9d309868ce38510ce6a3882c65a045c6c0917ff3a579732324"
)
EXTENSION_PREREGISTRATION_SHA256 = (
    "1001e5f35b5d0aef811ec97cd5323c83ea67ca0a7eeed35220405117e47f61d1"
)
EXTENSION_LEARNED_ARMS = (
    "CLEAN_EMA",
    "MP_PDE_PUSHFORWARD_M01",
    "CURRICULUM_EMA_PUSHFORWARD_K13",
    "PAIRED_RECOVERY",
    "DYNAMICS_RELABEL",
    "PCNO_PDEREFINER_K3_VPRED",
)
REFINER_TIMESTEPS = (0, 1, 2, 3)
REFINER_INFERENCE_TIMESTEPS = (3, 2, 1, 0)
REFINER_INPUT_CHANNELS = 25
REFINER_EXTRA_LIFTING_WEIGHTS = 9 * 128
REFINER_MINIMUM_CUMULATIVE_NOISE_STD = 0.05748670866723229
REFINER_BETAS = (
    0.00330472167339124,
    0.02218655771089649,
    0.14895152805827971,
    1.0,
)
REFINER_ALPHAS_CUMPROD = (
    0.9966952783266088,
    0.9745820410138374,
    0.8294165567866693,
    0.0,
)

_VERIFIED_EXTENSION_CONTRACT_TOKEN = object()


class VerifiedExtensionContract:
    """Exact immutable extension contract loaded with its preregistration."""

    __slots__ = ("__payload_bytes",)

    def __init__(self, payload: Mapping[str, Any], *, _token: object) -> None:
        if _token is not _VERIFIED_EXTENSION_CONTRACT_TOKEN:
            raise TypeError("use load_extension_contract")
        self.__payload_bytes = json.dumps(
            dict(payload), separators=(",", ":"), allow_nan=False
        ).encode("utf-8")

    def payload(self) -> dict[str, Any]:
        return json.loads(self.__payload_bytes)


def _stable_file_bytes(path: str | Path, label: str) -> bytes:
    candidate = Path(path)
    if not candidate.is_file() or candidate.is_symlink():
        raise ValueError(f"{label} is absent or aliased")
    before = (candidate.stat().st_size, candidate.stat().st_mtime_ns)
    value = candidate.read_bytes()
    after = (candidate.stat().st_size, candidate.stat().st_mtime_ns)
    if before != after or len(value) != after[0]:
        raise RuntimeError(f"{label} changed while being read")
    return value


def load_extension_contract(
    contract_path: str | Path, preregistration_path: str | Path
) -> VerifiedExtensionContract:
    """Bind production code to the exact frozen JSON and prose contract."""

    preregistration = _stable_file_bytes(
        preregistration_path, "extension preregistration"
    )
    if sha256(preregistration).hexdigest() != EXTENSION_PREREGISTRATION_SHA256:
        raise ValueError("extension preregistration SHA256 differs")
    contract_bytes = _stable_file_bytes(contract_path, "extension contract")
    if sha256(contract_bytes).hexdigest() != EXTENSION_CONTRACT_SHA256:
        raise ValueError("extension contract SHA256 differs")
    payload = json.loads(contract_bytes)
    validate_extension_math_contract(payload)
    if payload.get("preregistration_sha256") != EXTENSION_PREREGISTRATION_SHA256:
        raise ValueError("extension contract does not bind the preregistration")
    return VerifiedExtensionContract(payload, _token=_VERIFIED_EXTENSION_CONTRACT_TOKEN)


def validate_extension_math_contract(payload: Mapping[str, Any]) -> None:
    """Reject drift in the values consumed by this module."""

    if not isinstance(payload, Mapping):
        raise TypeError("extension contract must be an object")
    if payload.get("schema") != EXTENSION_CONTRACT_SCHEMA:
        raise ValueError("extension contract schema differs")
    if payload.get("experiment_id") != EXTENSION_EXPERIMENT_ID:
        raise ValueError("extension experiment identity differs")
    if payload.get("status") != "owner_directed_train_development_only":
        raise ValueError("extension authorization status differs")
    arms = payload.get("arms")
    if not isinstance(arms, Mapping) or tuple(arms) != EXTENSION_LEARNED_ARMS:
        raise ValueError("extension learned-arm order differs")
    refiner = arms.get("PCNO_PDEREFINER_K3_VPRED")
    if not isinstance(refiner, Mapping):
        raise TypeError("PDE-Refiner contract must be an object")
    exact = {
        "algorithm": "released_four_timestep_ddpm_v_prediction",
        "candidate": "normalized_next_state_residual",
        "scheduler_timesteps": list(REFINER_TIMESTEPS),
        "input_channels": REFINER_INPUT_CHANNELS,
        "trained_betas": list(REFINER_BETAS),
        "alphas_cumprod": list(REFINER_ALPHAS_CUMPROD),
        "prediction_type": "v_prediction",
        "clip_sample": False,
        "inference_timestep_order": list(REFINER_INFERENCE_TIMESTEPS),
        "model_calls_per_physical_step": 4,
        "final_candidate_only_enters_recurrence": True,
    }
    for key, expected in exact.items():
        if refiner.get(key) != expected:
            raise ValueError(f"PDE-Refiner contract {key} differs")
    protection = payload.get("protection")
    if not isinstance(protection, Mapping):
        raise TypeError("extension protection contract must be an object")
    if protection.get("offline_training_label_solver_calls") is not True:
        raise ValueError("offline label-generation authorization differs")
    for key in ("prospective_opened", "sealed_opened", "online_solver_calls"):
        if protection.get(key) is not False:
            raise ValueError(f"protected boundary {key} differs")
    try:
        canonical = json.dumps(
            dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise TypeError("extension contract is not canonical JSON") from error
    if sha256(canonical).hexdigest() != EXTENSION_CONTRACT_CANONICAL_SHA256:
        raise ValueError("extension contract canonical payload differs")


def _require_float32_state(value: torch.Tensor, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be a torch.Tensor")
    if value.dtype != torch.float32 or value.ndim != 3 or value.shape[-1] != 5:
        raise ValueError(f"{label} must have float32 shape [B,N,5]")
    if not torch.isfinite(value).all():
        raise ValueError(f"{label} contains nonfinite values")
    return value


def _require_same_states(
    values: Sequence[tuple[str, torch.Tensor]],
) -> tuple[torch.Tensor, ...]:
    checked = tuple(_require_float32_state(value, label) for label, value in values)
    reference = checked[0]
    if any(value.shape != reference.shape for value in checked[1:]):
        raise ValueError("state tensors must share shape")
    if any(value.device != reference.device for value in checked[1:]):
        raise ValueError("state tensors must share device")
    return checked


class ExponentialMovingAverage(nn.Module):
    """Frozen evaluation copy updated after each optimizer step."""

    def __init__(self, model: nn.Module, decay: float = 0.995) -> None:
        super().__init__()
        if not isinstance(model, nn.Module):
            raise TypeError("EMA requires a torch module")
        if not math.isfinite(decay) or not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must lie in [0,1)")
        self.model = copy.deepcopy(model).eval()
        self.model.requires_grad_(False)
        self.register_buffer("_decay", torch.tensor(decay, dtype=torch.float64))
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.int64))

    @property
    def decay(self) -> float:
        return float(self._decay.item())

    def train(self, mode: bool = True) -> ExponentialMovingAverage:
        """Change wrapper mode without ever enabling training behavior in EMA."""

        super().train(mode)
        self.model.eval()
        return self

    @torch.no_grad()
    def update(self, online_model: nn.Module) -> None:
        decay = self.decay
        if not math.isfinite(decay) or not 0.0 <= decay < 1.0:
            raise ValueError("loaded EMA decay is invalid")
        online = online_model.state_dict()
        averaged = self.model.state_dict()
        if tuple(online) != tuple(averaged):
            raise ValueError("EMA and online model state layouts differ")
        for name, target in averaged.items():
            source = online[name].detach().to(device=target.device)
            if source.shape != target.shape or source.dtype != target.dtype:
                raise ValueError(f"EMA tensor {name} differs from the online model")
            if target.is_floating_point() or target.is_complex():
                target.mul_(decay).add_(source, alpha=1.0 - decay)
            else:
                target.copy_(source)
        self.num_updates.add_(1)
        self.model.eval()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)


def maximum_available_prefix_depth(
    center_index: int,
    *,
    first_train_frame: int = 955,
    requested_maximum: int = 3,
) -> int:
    """Cap depth so the clean prefix seed never leaves train frames."""

    for value, label in (
        (center_index, "center index"),
        (first_train_frame, "first train frame"),
        (requested_maximum, "requested maximum"),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{label} must be an integer")
    if requested_maximum < 0:
        raise ValueError("requested maximum must be nonnegative")
    available = int(center_index) - int(first_train_frame) - 1
    if available < 0:
        raise ValueError("center lacks a complete clean BDF2 seed pair")
    return min(int(requested_maximum), available)


def sample_mp_pde_prefix_depth(epoch: int, generator: torch.Generator) -> int:
    """Literal MP-PDE default: clean first epoch, then uniform m in {0,1}."""

    if isinstance(epoch, bool) or not isinstance(epoch, Integral) or epoch < 1:
        raise ValueError("epoch must be a positive one-based integer")
    if not isinstance(generator, torch.Generator):
        raise TypeError("a torch.Generator is required")
    if epoch == 1:
        return 0
    return int(torch.randint(0, 2, (), generator=generator).item())


def curriculum_exposure_probability(epoch: int) -> float:
    """Frozen ten-epoch ramp for the stronger EMA pushforward comparator."""

    if isinstance(epoch, bool) or not isinstance(epoch, Integral) or epoch < 1:
        raise ValueError("epoch must be a positive one-based integer")
    return 0.5 * min(1.0, float(epoch) / 10.0)


def sample_curriculum_requested_prefix_depth(
    epoch: int, generator: torch.Generator
) -> int:
    """Draw one shared requested depth without changing its registered law."""

    if not isinstance(generator, torch.Generator):
        raise TypeError("a torch.Generator is required")
    probability = curriculum_exposure_probability(epoch)
    if float(torch.rand((), generator=generator).item()) >= probability:
        return 0
    return int(torch.randint(1, 4, (), generator=generator).item())


def realize_prefix_depths(
    requested_depth: int, center_indices: Sequence[int]
) -> tuple[int, ...]:
    """Apply the train-boundary cap per example after one shared batch draw."""

    if (
        isinstance(requested_depth, bool)
        or not isinstance(requested_depth, Integral)
        or requested_depth < 0
        or requested_depth > 3
    ):
        raise ValueError("requested depth must be an integer in [0,3]")
    if not isinstance(center_indices, Sequence) or not center_indices:
        raise ValueError("center indices must be a nonempty sequence")
    return tuple(
        min(
            int(requested_depth),
            maximum_available_prefix_depth(center, requested_maximum=3),
        )
        for center in center_indices
    )


def group_examples_by_prefix_depth(
    realized_depths: Sequence[int],
) -> dict[int, tuple[int, ...]]:
    """Group a mixed capped minibatch for scalar-depth prefix construction."""

    if not isinstance(realized_depths, Sequence) or not realized_depths:
        raise ValueError("realized depths must be a nonempty sequence")
    groups: dict[int, list[int]] = {}
    for index, depth in enumerate(realized_depths):
        if (
            isinstance(depth, bool)
            or not isinstance(depth, Integral)
            or not 0 <= depth <= 3
        ):
            raise ValueError("every realized depth must be an integer in [0,3]")
        groups.setdefault(int(depth), []).append(index)
    return {depth: tuple(indices) for depth, indices in sorted(groups.items())}


@dataclass(frozen=True)
class DetachedPrefixPresentation:
    previous: torch.Tensor
    current: torch.Tensor
    target_normalized_residual: torch.Tensor
    aligned_clean_current: torch.Tensor
    prefix_depth: int
    prefix_model_calls: int


def make_detached_prefix_presentation(
    prefix_model: nn.Module,
    clean_sequence: torch.Tensor,
    prefix_depth: int,
    geometry_batch: Mapping[str, torch.Tensor],
    normalization: NACANormalization,
    *,
    fourier_tensors: tuple[torch.Tensor, ...] | None = None,
) -> DetachedPrefixPresentation:
    """Build an arbitrary-depth detached prefix and aligned terminal target.

    ``clean_sequence`` is ``[u[n-m-1], ..., u[n+1]]`` and therefore has
    exactly ``m+3`` frames.
    """

    if isinstance(prefix_depth, bool) or not isinstance(prefix_depth, Integral):
        raise TypeError("prefix depth must be an integer")
    depth = int(prefix_depth)
    if depth < 0 or depth > 3:
        raise ValueError("prefix depth must lie in [0,3]")
    if not isinstance(clean_sequence, torch.Tensor):
        raise TypeError("clean sequence must be a torch.Tensor")
    if (
        clean_sequence.dtype != torch.float32
        or clean_sequence.ndim != 4
        or clean_sequence.shape[1] != depth + 3
        or clean_sequence.shape[-1] != NUM_DYNAMIC_FIELDS
    ):
        raise ValueError("clean sequence must have float32 shape [B,m+3,N,5]")
    if not torch.isfinite(clean_sequence).all():
        raise ValueError("clean sequence contains nonfinite values")
    previous = clean_sequence[:, 0]
    current = clean_sequence[:, 1]
    with torch.no_grad():
        for _ in range(depth):
            previous, current = recurrent_step(
                prefix_model,  # type: ignore[arg-type]
                previous,
                current,
                geometry_batch,
                normalization,
                fourier_tensors=fourier_tensors,
            )
    previous = previous.detach()
    current = current.detach()
    clean_current = clean_sequence[:, depth + 1].detach()
    clean_future = clean_sequence[:, depth + 2].detach()
    target = normalization.normalize_residual(clean_future - current).detach()
    return DetachedPrefixPresentation(
        previous=previous,
        current=current,
        target_normalized_residual=target,
        aligned_clean_current=clean_current,
        prefix_depth=depth,
        prefix_model_calls=depth,
    )


@dataclass(frozen=True)
class PairedDisplacedPresentation:
    previous: torch.Tensor
    current: torch.Tensor
    target_normalized_residual: torch.Tensor
    target_kind: str


def make_paired_displaced_presentation(
    displaced_previous: torch.Tensor,
    displaced_current: torch.Tensor,
    clean_future: torch.Tensor,
    solver_future: torch.Tensor,
    normalization: NACANormalization,
    *,
    target_kind: str,
) -> PairedDisplacedPresentation:
    """Use one displaced pair with either return-to-path or SU2 target."""

    previous, current, clean, solver = _require_same_states(
        (
            ("displaced previous", displaced_previous),
            ("displaced current", displaced_current),
            ("clean future", clean_future),
            ("solver future", solver_future),
        )
    )
    if target_kind == "recovery":
        target_state = clean
    elif target_kind == "dynamics_relabel":
        target_state = solver
    else:
        raise ValueError("target kind must be recovery or dynamics_relabel")
    return PairedDisplacedPresentation(
        previous=previous,
        current=current,
        target_normalized_residual=normalization.normalize_residual(
            target_state - current
        ),
        target_kind=target_kind,
    )


def paired_bank_sign_for_epoch(epoch: int) -> int:
    """Use the negative bank on odd epochs and the positive bank on even ones."""

    if isinstance(epoch, bool) or not isinstance(epoch, Integral) or epoch < 1:
        raise ValueError("epoch must be a positive one-based integer")
    return -1 if int(epoch) % 2 == 1 else 1


def paired_clean_displaced_objective(
    clean_loss: torch.Tensor, displaced_loss: torch.Tensor
) -> torch.Tensor:
    """Apply the frozen 0.5 clean / 0.5 displaced weighting."""

    if (
        not isinstance(clean_loss, torch.Tensor)
        or not isinstance(displaced_loss, torch.Tensor)
        or clean_loss.ndim != 0
        or displaced_loss.ndim != 0
    ):
        raise ValueError("paired component losses must be scalar tensors")
    if clean_loss.device != displaced_loss.device:
        raise ValueError("paired component losses must share device")
    return 0.5 * clean_loss + 0.5 * displaced_loss


def build_refiner_input(
    previous: torch.Tensor,
    current: torch.Tensor,
    candidate_normalized_residual: torch.Tensor,
    timesteps: torch.Tensor,
    geometry_batch: Mapping[str, torch.Tensor],
    normalization: NACANormalization,
) -> torch.Tensor:
    """Assemble static, BDF2, candidate, and four one-hot scheduler channels."""

    previous, current, candidate = _require_same_states(
        (
            ("previous state", previous),
            ("current state", current),
            ("candidate residual", candidate_normalized_residual),
        )
    )
    if not isinstance(timesteps, torch.Tensor):
        raise TypeError("timesteps must be a torch.Tensor")
    if timesteps.dtype != torch.int64 or timesteps.shape != (previous.shape[0],):
        raise ValueError("timesteps must have int64 shape [B]")
    if timesteps.device != previous.device:
        raise ValueError("timesteps and states must share device")
    if torch.any((timesteps < 0) | (timesteps > 3)):
        raise ValueError("refiner timesteps must lie in [0,3]")
    static = geometry_batch.get("static_features")
    if static is None or static.shape != previous.shape[:2] + (NUM_STATIC_FEATURES,):
        raise ValueError("static geometry features must have shape [B,N,6]")
    if static.device != previous.device:
        raise ValueError("geometry and states must share device")
    one_hot = F.one_hot(timesteps, num_classes=4).to(dtype=previous.dtype)
    one_hot = one_hot[:, None, :].expand(-1, previous.shape[1], -1)
    features = torch.cat(
        (
            static.to(dtype=previous.dtype),
            normalization.normalize_state(previous),
            normalization.normalize_state(current),
            candidate,
            one_hot,
        ),
        dim=-1,
    )
    if features.shape[-1] != REFINER_INPUT_CHANNELS:
        raise AssertionError("refiner input layout is inconsistent")
    return features


class NACAPCNORefiner(nn.Module):
    """Shared 25-input PCNO velocity predictor for four DDPM calls."""

    def __init__(
        self,
        *,
        fourier_lengths: Sequence[float],
        zero_initialize: bool = True,
    ) -> None:
        super().__init__()
        if zero_initialize is not True:
            raise ValueError("the frozen refiner requires exact zero initialization")
        lengths = tuple(float(value) for value in fourier_lengths)
        if len(lengths) != 2 or any(
            not math.isfinite(value) or value <= 0.0 for value in lengths
        ):
            raise ValueError("fourier_lengths must contain two positive finite values")
        modes = compute_Fourier_modes(2, [8, 8], list(lengths))
        self.backbone = PCNO(
            2,
            torch.as_tensor(modes, dtype=torch.float32),
            nmeasures=1,
            layers=[128, 128, 128, 128, 128],
            fc_dim=128,
            in_dim=REFINER_INPUT_CHANNELS,
            out_dim=NUM_DYNAMIC_FIELDS,
            act="gelu",
        )
        self.fourier_lengths = lengths
        nn.init.zeros_(self.backbone.fc2.weight)
        nn.init.zeros_(self.backbone.fc2.bias)

    def prepare_fourier_tensors(
        self, geometry_batch: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, ...]:
        return self.backbone.prepare_fourier_tensors(
            geometry_batch["nodes"], geometry_batch["node_weights"]
        )

    def forward(
        self,
        features: torch.Tensor,
        geometry_batch: Mapping[str, torch.Tensor],
        *,
        fourier_tensors: tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != REFINER_INPUT_CHANNELS:
            raise ValueError("NACA refiner features must have shape [B,N,25]")
        if geometry_batch["nodes"].shape[:2] != features.shape[:2]:
            raise ValueError("fixed geometry does not align with refiner features")
        auxiliary = (
            geometry_batch["node_mask"],
            geometry_batch["nodes"],
            geometry_batch["node_weights"],
            geometry_batch["directed_edges"],
            geometry_batch["edge_gradient_weights"],
        )
        return self.backbone(features, auxiliary, fourier_tensors=fourier_tensors)

    def model_config(self) -> dict[str, Any]:
        return {
            "model": "NACAPCNORefiner",
            "backbone": "pcno.PCNO",
            "history": "complete BDF2 pair",
            "candidate": "normalized next-state residual",
            "scheduler_condition": "four-channel one-hot",
            "input_channels": REFINER_INPUT_CHANNELS,
            "output_channels": NUM_DYNAMIC_FIELDS,
            "fourier_modes": [8, 8],
            "fourier_lengths": list(self.fourier_lengths),
            "nmeasures": 1,
            "layers": [128, 128, 128, 128, 128],
            "projection_width": 128,
            "activation": "gelu",
            "output": "DDPM velocity in normalized-residual coordinates",
            "zero_output_head": True,
        }


def build_naca_pcno_refiner(
    extension: VerifiedExtensionContract,
    geometry: VerifiedNACAGeometry,
) -> NACAPCNORefiner:
    """Build the exact refiner only from verified extension and geometry types."""

    if not isinstance(extension, VerifiedExtensionContract):
        raise TypeError("production refiner requires VerifiedExtensionContract")
    if not isinstance(geometry, VerifiedNACAGeometry):
        raise TypeError("production refiner requires VerifiedNACAGeometry")
    validate_extension_math_contract(extension.payload())
    model = NACAPCNORefiner(
        fourier_lengths=geometry.fourier_lengths,
        zero_initialize=True,
    )
    expected = {
        "model": "NACAPCNORefiner",
        "backbone": "pcno.PCNO",
        "history": "complete BDF2 pair",
        "candidate": "normalized next-state residual",
        "scheduler_condition": "four-channel one-hot",
        "input_channels": REFINER_INPUT_CHANNELS,
        "output_channels": NUM_DYNAMIC_FIELDS,
        "fourier_modes": [8, 8],
        "fourier_lengths": [float(value) for value in geometry.fourier_lengths],
        "nmeasures": 1,
        "layers": [128, 128, 128, 128, 128],
        "projection_width": 128,
        "activation": "gelu",
        "output": "DDPM velocity in normalized-residual coordinates",
        "zero_output_head": True,
    }
    if model.model_config() != expected:
        raise ValueError("refiner model configuration differs from the contract")
    return model


def refiner_parameter_difference_from_baseline(
    refiner: NACAPCNORefiner, baseline: NACAPCNOResidual
) -> int:
    """Return and validate the exact lifting-layer-only parameter increase."""

    if not isinstance(refiner, NACAPCNORefiner) or not isinstance(
        baseline, NACAPCNOResidual
    ):
        raise TypeError("expected NACA baseline and refiner models")
    difference = sum(p.numel() for p in refiner.parameters()) - sum(
        p.numel() for p in baseline.parameters()
    )
    if difference != REFINER_EXTRA_LIFTING_WEIGHTS:
        raise ValueError("refiner parameter difference is not lifting-layer-only")
    return difference


class FourStepVPredictionScheduler:
    """Dependency-free four-step DDPM scheduler matching the pinned baseline."""

    def __init__(self, betas: Sequence[float] = REFINER_BETAS) -> None:
        self.betas = tuple(float(value) for value in betas)
        if self.betas != REFINER_BETAS:
            raise ValueError("scheduler betas differ from the frozen contract")
        alpha_product = 1.0
        products: list[float] = []
        for beta in self.betas:
            if not math.isfinite(beta) or not 0.0 < beta <= 1.0:
                raise ValueError("scheduler betas must lie in (0,1]")
            alpha_product *= 1.0 - beta
            products.append(alpha_product)
        self.alphas_cumprod = tuple(products)
        if any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=2.0e-16)
            for actual, expected in zip(self.alphas_cumprod, REFINER_ALPHAS_CUMPROD)
        ):
            raise ValueError("scheduler cumulative alphas differ from the contract")

    def _factors(
        self, timesteps: torch.Tensor, reference: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if timesteps.dtype != torch.int64 or timesteps.shape != (reference.shape[0],):
            raise ValueError("timesteps must have int64 shape [B]")
        if timesteps.device != reference.device:
            raise ValueError("timesteps and sample must share device")
        if torch.any((timesteps < 0) | (timesteps > 3)):
            raise ValueError("scheduler timesteps must lie in [0,3]")
        products = torch.tensor(
            self.alphas_cumprod, dtype=reference.dtype, device=reference.device
        )
        alpha_bar = products.index_select(0, timesteps)
        shape = (reference.shape[0],) + (1,) * (reference.ndim - 1)
        alpha_bar = alpha_bar.reshape(shape)
        return torch.sqrt(alpha_bar), torch.sqrt(torch.clamp_min(1.0 - alpha_bar, 0.0))

    def add_noise(
        self, clean: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        if (
            clean.shape != noise.shape
            or clean.dtype != noise.dtype
            or clean.device != noise.device
        ):
            raise ValueError(
                "clean candidate and noise must share shape, dtype, and device"
            )
        signal, noise_factor = self._factors(timesteps, clean)
        return signal * clean + noise_factor * noise

    def velocity_target(
        self, clean: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        if (
            clean.shape != noise.shape
            or clean.dtype != noise.dtype
            or clean.device != noise.device
        ):
            raise ValueError(
                "clean candidate and noise must share shape, dtype, and device"
            )
        signal, noise_factor = self._factors(timesteps, clean)
        return signal * noise - noise_factor * clean

    def reconstruct_clean(
        self, sample: torch.Tensor, velocity: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        if (
            sample.shape != velocity.shape
            or sample.dtype != velocity.dtype
            or sample.device != velocity.device
        ):
            raise ValueError("sample and velocity must share shape, dtype, and device")
        signal, noise_factor = self._factors(timesteps, sample)
        return signal * sample - noise_factor * velocity

    def step(
        self,
        velocity: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        *,
        noise: torch.Tensor | None,
    ) -> torch.Tensor:
        """One fixed-small-variance DDPM reverse step without clipping."""

        if isinstance(timestep, bool) or not isinstance(timestep, Integral):
            raise TypeError("timestep must be an integer")
        t = int(timestep)
        if t not in REFINER_TIMESTEPS:
            raise ValueError("timestep must lie in [0,3]")
        if (
            velocity.shape != sample.shape
            or velocity.dtype != sample.dtype
            or velocity.device != sample.device
        ):
            raise ValueError("velocity and sample must share shape, dtype, and device")
        if t == 0:
            if noise is not None:
                raise ValueError(
                    "the final deterministic scheduler step takes no noise"
                )
        else:
            if (
                noise is None
                or noise.shape != sample.shape
                or noise.dtype != sample.dtype
                or noise.device != sample.device
            ):
                raise ValueError("stochastic scheduler steps require aligned noise")
            if not torch.isfinite(noise).all():
                raise ValueError("scheduler noise contains nonfinite values")
        if not torch.isfinite(sample).all() or not torch.isfinite(velocity).all():
            raise ValueError("scheduler sample or velocity contains nonfinite values")

        alpha_product_t = self.alphas_cumprod[t]
        alpha_product_previous = self.alphas_cumprod[t - 1] if t > 0 else 1.0
        beta_product_t = 1.0 - alpha_product_t
        beta_product_previous = 1.0 - alpha_product_previous
        current_alpha = alpha_product_t / alpha_product_previous
        current_beta = 1.0 - current_alpha
        alpha_t = torch.tensor(
            alpha_product_t, dtype=sample.dtype, device=sample.device
        )
        beta_t = torch.tensor(beta_product_t, dtype=sample.dtype, device=sample.device)
        clean = torch.sqrt(alpha_t) * sample - torch.sqrt(beta_t) * velocity
        clean_coefficient = (
            math.sqrt(alpha_product_previous) * current_beta / beta_product_t
        )
        sample_coefficient = (
            math.sqrt(current_alpha) * beta_product_previous / beta_product_t
        )
        previous = clean_coefficient * clean + sample_coefficient * sample
        if t > 0:
            variance = beta_product_previous / beta_product_t * current_beta
            previous = previous + math.sqrt(max(variance, 0.0)) * noise  # type: ignore[operator]
        return previous


def sample_refiner_timesteps(
    batch_size: int,
    *,
    generator: torch.Generator,
    device: str | torch.device,
) -> torch.Tensor:
    """Sample the registered uniform per-example training timesteps."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral):
        raise TypeError("batch size must be an integer")
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    if not isinstance(generator, torch.Generator):
        raise TypeError("a torch.Generator is required")
    return torch.randint(
        0,
        len(REFINER_TIMESTEPS),
        (int(batch_size),),
        dtype=torch.int64,
        device=torch.device(device),
        generator=generator,
    )


@dataclass(frozen=True)
class RefinerTrainingPresentation:
    noised_candidate: torch.Tensor
    velocity_target: torch.Tensor
    timesteps: torch.Tensor


def make_refiner_training_presentation(
    clean_normalized_residual: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler: FourStepVPredictionScheduler,
) -> RefinerTrainingPresentation:
    clean, aligned_noise = _require_same_states(
        (("clean residual", clean_normalized_residual), ("refiner noise", noise))
    )
    return RefinerTrainingPresentation(
        noised_candidate=scheduler.add_noise(clean, aligned_noise, timesteps),
        velocity_target=scheduler.velocity_target(clean, aligned_noise, timesteps),
        timesteps=timesteps,
    )


@dataclass(frozen=True)
class RefinerNoiseTape:
    initial: torch.Tensor
    reverse_t3: torch.Tensor
    reverse_t2: torch.Tensor
    reverse_t1: torch.Tensor

    def reverse_noise(self, timestep: int) -> torch.Tensor | None:
        if timestep == 3:
            return self.reverse_t3
        if timestep == 2:
            return self.reverse_t2
        if timestep == 1:
            return self.reverse_t1
        if timestep == 0:
            return None
        raise ValueError("timestep must lie in [0,3]")


def sample_refiner_noise_tape(
    reference: torch.Tensor, *, generator: torch.Generator
) -> RefinerNoiseTape:
    _require_float32_state(reference, "refiner noise reference")
    if not isinstance(generator, torch.Generator):
        raise TypeError("a torch.Generator is required")
    draws = tuple(
        torch.randn(
            reference.shape,
            dtype=reference.dtype,
            device=reference.device,
            generator=generator,
        )
        for _ in range(4)
    )
    return RefinerNoiseTape(*draws)


def predict_refiner_velocity(
    model: nn.Module,
    previous: torch.Tensor,
    current: torch.Tensor,
    candidate: torch.Tensor,
    timestep: int,
    geometry_batch: Mapping[str, torch.Tensor],
    normalization: NACANormalization,
    *,
    fourier_tensors: tuple[torch.Tensor, ...] | None = None,
) -> torch.Tensor:
    timesteps = torch.full(
        (previous.shape[0],), timestep, dtype=torch.int64, device=previous.device
    )
    features = build_refiner_input(
        previous, current, candidate, timesteps, geometry_batch, normalization
    )
    velocity = model(features, geometry_batch, fourier_tensors=fourier_tensors)
    if velocity.shape != candidate.shape:
        raise ValueError("refiner velocity shape differs from the candidate")
    return velocity


@dataclass(frozen=True)
class RefinerRecurrentStep:
    recurrent_previous: torch.Tensor
    next_state: torch.Tensor
    final_normalized_residual: torch.Tensor
    intermediate_candidates: tuple[torch.Tensor, ...]
    model_calls: int


def refined_recurrent_step(
    model: nn.Module,
    previous: torch.Tensor,
    current: torch.Tensor,
    geometry_batch: Mapping[str, torch.Tensor],
    normalization: NACANormalization,
    scheduler: FourStepVPredictionScheduler,
    noise_tape: RefinerNoiseTape,
    *,
    fourier_tensors: tuple[torch.Tensor, ...] | None = None,
) -> RefinerRecurrentStep:
    """Generate once, refine three times, and feed back only the final state."""

    previous, current = _require_same_states(
        (("previous state", previous), ("current state", current))
    )
    tapes = (
        noise_tape.initial,
        noise_tape.reverse_t3,
        noise_tape.reverse_t2,
        noise_tape.reverse_t1,
    )
    if any(
        tape.shape != current.shape
        or tape.dtype != current.dtype
        or tape.device != current.device
        or not torch.isfinite(tape).all()
        for tape in tapes
    ):
        raise ValueError("refiner noise tape does not align with the BDF2 state")
    candidate = noise_tape.initial
    intermediates = [candidate]
    for timestep in REFINER_INFERENCE_TIMESTEPS:
        velocity = predict_refiner_velocity(
            model,
            previous,
            current,
            candidate,
            timestep,
            geometry_batch,
            normalization,
            fourier_tensors=fourier_tensors,
        )
        candidate = scheduler.step(
            velocity,
            timestep,
            candidate,
            noise=noise_tape.reverse_noise(timestep),
        )
        intermediates.append(candidate)
    next_state = current + normalization.decode_residual(candidate)
    return RefinerRecurrentStep(
        recurrent_previous=current,
        next_state=next_state,
        final_normalized_residual=candidate,
        intermediate_candidates=tuple(intermediates),
        model_calls=4,
    )
