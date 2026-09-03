"""Frozen corrective-mechanism primitives for the NACA PCNO successor.

This module contains only model-side transformations registered by
``B3B4_NACA_CM_20260901A``.  It never loads trajectory values, evaluates a
solver, or opens a protected population.  Production callers remain
responsible for hash-binding parent checkpoints and calibration artifacts.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np
import torch

from utility.time_dependent_no.pcno_naca0012 import (
    BASELINE_CONTRACT_SHA256,
    NACA_DYNAMIC_FIELDS,
    NACANormalization,
    NACAPCNOResidual,
    VerifiedNACAContract,
    recurrent_step,
    validate_role_access,
)

SUCCESSOR_CONTRACT_SCHEMA = "time_dependent_no.naca_corrective_successor_contract.v1"
SUCCESSOR_EXPERIMENT_ID = "B3B4_NACA_CM_20260901A"
SUCCESSOR_ARMS = (
    "CLEAN",
    "IID_RECOVERY",
    "ERROR_SUBSPACE_RECOVERY",
    "DETACHED_PUSHFORWARD",
    "PATH_PROJECTION",
)
SUCCESSOR_SEEDS = (17, 29, 43)
ERROR_SUBSPACE_RANK = 16
PATH_PCA_VARIANCE_TARGET = 0.999
PATH_PCA_RANK_CAP = 32
TRAIN_CENTER_INDICES = tuple(range(956, 1194))
DEVELOPMENT_CENTER_INDICES = tuple(range(1234, 1472))
STRUCTURED_PAIR_CENTER_INDICES = tuple(range(957, 1194))
RECOVERY_CALIBRATION_SCHEMA = "time_dependent_no.naca_recovery_calibration.v1"
TRAIN_PATH_PROJECTOR_SCHEMA = "time_dependent_no.naca_train_path_projector.v1"


def _require_exact_contract_value(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise ValueError(f"successor mathematical contract {label} differs")


def validate_successor_math_contract(
    payload: Mapping[str, Any], parent_contract: VerifiedNACAContract
) -> None:
    """Validate only the frozen values consumed by these mathematical primitives."""

    if not isinstance(payload, Mapping):
        raise TypeError("successor contract payload must be an object")
    if not isinstance(parent_contract, VerifiedNACAContract):
        raise TypeError("a VerifiedNACAContract parent is required")
    if parent_contract.file_sha256 != BASELINE_CONTRACT_SHA256:
        raise ValueError("successor parent contract identity differs")
    _require_exact_contract_value(
        payload.get("schema"), SUCCESSOR_CONTRACT_SCHEMA, "schema"
    )
    _require_exact_contract_value(
        payload.get("experiment_id"), SUCCESSOR_EXPERIMENT_ID, "experiment ID"
    )
    _require_exact_contract_value(
        payload.get("inherited_r0_contract_sha256"),
        BASELINE_CONTRACT_SHA256,
        "parent R0 contract",
    )
    state = payload.get("state")
    if not isinstance(state, Mapping):
        raise TypeError("successor contract state must be an object")
    _require_exact_contract_value(
        dict(state),
        {
            "history": "complete_bdf2_pair",
            "dynamic_fields": list(NACA_DYNAMIC_FIELDS),
            "model_output": "normalized_next_state_residual",
        },
        "state",
    )
    calibration = payload.get("calibration")
    if not isinstance(calibration, Mapping):
        raise TypeError("successor contract calibration must be an object")
    _require_exact_contract_value(
        dict(calibration),
        {
            "fit_role": "train_only",
            "parent_seeds": list(SUCCESSOR_SEEDS),
            "teacher_forced_error_coordinates": "state_normalized",
            "iid_scale": "per_field_rms_parent_teacher_forced_next_state_error",
            "iid_multiplier": 1.0,
            "structured_pair_centers_inclusive": [957, 1193],
            "structured_pair_count_per_seed": 237,
            "structured_rank": ERROR_SUBSPACE_RANK,
            "structured_distribution": "centered_gaussian_in_parent_error_pair_pca",
            "structured_energy_match": "iid_expected_history_pair_squared_norm",
            "path_pca_variance_target": PATH_PCA_VARIANCE_TARGET,
            "path_pca_rank_cap": PATH_PCA_RANK_CAP,
            "path_fit_states": "train_current_states_only",
            "path_fit_frames_inclusive": [956, 1193],
            "path_fit_state_count": 238,
        },
        "calibration",
    )
    arms = payload.get("arms")
    if not isinstance(arms, Mapping) or tuple(arms) != SUCCESSOR_ARMS:
        raise ValueError("successor contract arms differ from the frozen order")
    _require_exact_contract_value(
        dict(arms),
        {
            "CLEAN": {"clean_loss_weight": 1.0, "intervention_loss_weight": 0.0},
            "IID_RECOVERY": {
                "clean_loss_weight": 0.5,
                "intervention_loss_weight": 0.5,
                "corrupt_previous": True,
                "corrupt_current": True,
                "slot_noise_correlation": "independent",
                "node_support": "all_nodes",
                "field_support": "all_dynamic_fields",
                "fresh_draws": "each_epoch",
            },
            "ERROR_SUBSPACE_RECOVERY": {
                "clean_loss_weight": 0.5,
                "intervention_loss_weight": 0.5,
                "corrupt_previous": True,
                "corrupt_current": True,
                "fresh_draws": "each_epoch",
            },
            "DETACHED_PUSHFORWARD": {
                "clean_loss_weight": 0.5,
                "intervention_loss_weight": 0.5,
                "exposed_centers_inclusive": [957, 1193],
                "exposed_center_count": 237,
                "prefix_gradient": "detached",
                "target": "stored_clean_future",
            },
            "PATH_PROJECTION": {
                "parent_predictor": "CLEAN",
                "schedule": "every_step",
                "candidate_geometry": "train_pca_piecewise_linear_path",
                "corrected_state": "full_state_segment_interpolant",
                "recurrent_feedback": "corrected_state",
            },
        },
        "arms",
    )


def _strict_indices(values: Iterable[Any], label: str) -> tuple[int, ...]:
    raw = tuple(values)
    if not raw:
        raise ValueError(f"{label} must be nonempty")
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in raw):
        raise TypeError(f"{label} must contain only integer indices")
    converted = tuple(int(value) for value in raw)
    if len(set(converted)) != len(converted):
        raise ValueError(f"{label} must not contain duplicates")
    return converted


def _mapping_scalar(value: Any, label: str) -> Any:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"{label} must be a scalar")
    return array.item()


def _mapping_integer_scalar(value: Any, label: str) -> int:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "iu":
        raise TypeError(f"{label} must be an integer scalar")
    return int(array.item())


def _mapping_float_scalar(value: Any, label: str) -> float:
    array = np.asarray(value)
    if array.shape != () or array.dtype != np.float64:
        raise TypeError(f"{label} must be a float64 scalar")
    return float(array.item())


def validate_successor_training_centers(
    parent_contract: VerifiedNACAContract,
    center_indices: Iterable[int],
) -> tuple[int, ...]:
    """Validate a nonempty train-only batch without exposing a role knob."""

    centers = _strict_indices(center_indices, "successor training centers")
    validate_role_access(parent_contract, "train", centers, horizon_steps=1)
    if any(center not in TRAIN_CENTER_INDICES for center in centers):
        raise ValueError("training center differs from the successor train block")
    return centers


def validate_successor_development_centers(
    parent_contract: VerifiedNACAContract,
    center_indices: Iterable[int],
) -> tuple[int, ...]:
    """Validate a nonempty open development batch for diagnostics only."""

    centers = _strict_indices(center_indices, "successor development centers")
    validate_role_access(parent_contract, "development", centers, horizon_steps=1)
    if any(center not in DEVELOPMENT_CENTER_INDICES for center in centers):
        raise ValueError("development center differs from the successor block")
    return centers


def validate_pushforward_training_centers(
    parent_contract: VerifiedNACAContract,
    center_indices: Iterable[int],
) -> tuple[int, ...]:
    """Require only the 237 centers with a complete in-role prefix pair."""

    centers = validate_successor_training_centers(parent_contract, center_indices)
    if any(center not in STRUCTURED_PAIR_CENTER_INDICES for center in centers):
        raise ValueError("pushforward center lacks its complete train-only prefix")
    return centers


def validate_train_path_indices(
    parent_contract: VerifiedNACAContract,
    frame_indices: Iterable[int],
) -> tuple[int, ...]:
    """Require the exact 238 ordered train-current states before reading them."""

    frames = _strict_indices(frame_indices, "successor train-current frames")
    if frames != TRAIN_CENTER_INDICES:
        raise ValueError(
            "path calibration requires the exact ordered train-current frames"
        )
    validate_role_access(
        parent_contract,
        "train",
        TRAIN_CENTER_INDICES,
        horizon_steps=1,
    )
    return frames


def _require_normalization(normalization: NACANormalization) -> None:
    if not isinstance(normalization, NACANormalization):
        raise TypeError("a NACANormalization is required")


def _require_state(value: Any, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be a torch.Tensor")
    if value.dtype != torch.float32 or value.ndim != 3 or value.shape[-1] != 5:
        raise ValueError(f"{label} must retain float32 shape [B,N,5]")
    if not torch.isfinite(value).all():
        raise ValueError(f"{label} contains nonfinite values")
    return value


def _require_same_states(values: Sequence[tuple[str, Any]]) -> list[torch.Tensor]:
    tensors = [_require_state(value, label) for label, value in values]
    reference = tensors[0]
    for tensor in tensors[1:]:
        if tensor.shape != reference.shape:
            raise ValueError("BDF2 states and perturbations must share one shape")
        if tensor.device != reference.device:
            raise ValueError("BDF2 states and perturbations must share one device")
    return tensors


def _state_scale_tensor(
    normalization: NACANormalization, reference: torch.Tensor
) -> torch.Tensor:
    return torch.tensor(
        normalization.state_scale,
        dtype=reference.dtype,
        device=reference.device,
    )


@dataclass(frozen=True)
class HistoryNoise:
    """Two complete-BDF2 perturbations in state-normalized coordinates."""

    previous_normalized: torch.Tensor
    current_normalized: torch.Tensor


@dataclass(frozen=True)
class RecoveryPresentation:
    """Displaced BDF2 pair and exact stored-clean-future residual target."""

    previous: torch.Tensor
    current: torch.Tensor
    target_normalized_residual: torch.Tensor


def make_recovery_presentation(
    previous: torch.Tensor,
    current: torch.Tensor,
    clean_next: torch.Tensor,
    noise: HistoryNoise,
    normalization: NACANormalization,
) -> RecoveryPresentation:
    """Apply both slot perturbations and target the clean future exactly."""

    _require_normalization(normalization)
    if not isinstance(noise, HistoryNoise):
        raise TypeError("recovery noise must be a HistoryNoise")
    previous, current, clean_next, previous_noise, current_noise = _require_same_states(
        (
            ("previous state", previous),
            ("current state", current),
            ("clean next state", clean_next),
            ("previous normalized noise", noise.previous_normalized),
            ("current normalized noise", noise.current_normalized),
        )
    )
    state_scale = _state_scale_tensor(normalization, current)
    displaced_previous = previous + state_scale * previous_noise
    displaced_current = current + state_scale * current_noise
    target = normalization.normalize_residual(clean_next - displaced_current)
    return RecoveryPresentation(
        previous=displaced_previous,
        current=displaced_current,
        target_normalized_residual=target,
    )


def _require_generator(generator: torch.Generator, device: torch.device) -> None:
    if not isinstance(generator, torch.Generator):
        raise TypeError("an explicit torch.Generator is required")
    if torch.device(generator.device) != device:
        raise ValueError("noise generator and state must use the same device")


def sample_iid_history_noise(
    reference: torch.Tensor,
    field_rms: Sequence[float] | np.ndarray | torch.Tensor,
    *,
    generator: torch.Generator,
) -> HistoryNoise:
    """Draw independent all-node/all-field noise for the two BDF2 slots."""

    reference = _require_state(reference, "IID noise reference")
    _require_generator(generator, reference.device)
    scale = torch.as_tensor(field_rms, dtype=reference.dtype, device=reference.device)
    if scale.shape != (5,) or not torch.isfinite(scale).all() or torch.any(scale < 0.0):
        raise ValueError("IID field RMS must be finite, nonnegative, and shape [5]")
    previous = (
        torch.randn(
            reference.shape,
            dtype=reference.dtype,
            device=reference.device,
            generator=generator,
        )
        * scale
    )
    current = (
        torch.randn(
            reference.shape,
            dtype=reference.dtype,
            device=reference.device,
            generator=generator,
        )
        * scale
    )
    return HistoryNoise(previous, current)


@dataclass(frozen=True)
class RecoveryCalibration:
    """Train-only IID scale and centered rank-16 error-pair PCA law."""

    iid_field_rms: torch.Tensor
    structured_basis: torch.Tensor
    structured_coefficient_std: torch.Tensor
    structured_eigenvalues: torch.Tensor
    structured_captured_variance: float
    structured_energy_rescale: float
    iid_expected_history_pair_energy: float
    pair_sample_count: int
    num_nodes: int

    def __post_init__(self) -> None:
        for name in (
            "iid_field_rms",
            "structured_basis",
            "structured_coefficient_std",
            "structured_eigenvalues",
        ):
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            canonical = (
                value.detach().to(device="cpu", dtype=torch.float64).contiguous()
            )
            if not torch.isfinite(canonical).all():
                raise ValueError(f"{name} contains nonfinite values")
            object.__setattr__(self, name, canonical)
        if self.iid_field_rms.shape != (5,) or torch.any(self.iid_field_rms < 0.0):
            raise ValueError("IID field RMS must be nonnegative shape [5]")
        expected_basis = (ERROR_SUBSPACE_RANK, 2, int(self.num_nodes), 5)
        if self.structured_basis.shape != expected_basis:
            raise ValueError("structured basis has the wrong rank or state shape")
        if self.structured_coefficient_std.shape != (ERROR_SUBSPACE_RANK,):
            raise ValueError("structured coefficient standard deviations differ")
        if self.structured_eigenvalues.shape != (ERROR_SUBSPACE_RANK,):
            raise ValueError("structured eigenvalues differ")
        if torch.any(self.structured_coefficient_std <= 0.0):
            raise ValueError("structured coefficient scales must be positive")
        if torch.any(self.structured_eigenvalues <= 0.0):
            raise ValueError("structured eigenvalues must be positive")
        for name in (
            "structured_captured_variance",
            "structured_energy_rescale",
            "iid_expected_history_pair_energy",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if self.structured_captured_variance > 1.0 + 1.0e-12:
            raise ValueError("structured captured variance exceeds one")
        if (
            isinstance(self.pair_sample_count, bool)
            or not isinstance(self.pair_sample_count, Integral)
            or self.pair_sample_count != 3 * len(STRUCTURED_PAIR_CENTER_INDICES)
        ):
            raise ValueError("structured calibration sample count differs")
        if (
            isinstance(self.num_nodes, bool)
            or not isinstance(self.num_nodes, Integral)
            or self.num_nodes < 1
        ):
            raise ValueError("structured calibration must contain nodes")
        basis = self.structured_basis.reshape(ERROR_SUBSPACE_RANK, -1)
        if not torch.allclose(
            basis @ basis.T,
            torch.eye(ERROR_SUBSPACE_RANK, dtype=torch.float64),
            rtol=1.0e-8,
            atol=1.0e-10,
        ):
            raise ValueError("structured calibration basis is not orthonormal")
        expected_std = (
            torch.sqrt(self.structured_eigenvalues) * self.structured_energy_rescale
        )
        if not torch.allclose(
            self.structured_coefficient_std,
            expected_std,
            rtol=1.0e-12,
            atol=0.0,
        ):
            raise ValueError("structured calibration coefficient law differs")
        if not math.isclose(
            float(torch.sum(torch.square(self.structured_coefficient_std)).item()),
            self.iid_expected_history_pair_energy,
            rel_tol=1.0e-12,
            abs_tol=0.0,
        ):
            raise ValueError("structured and IID expected energies differ")
        object.__setattr__(self, "num_nodes", int(self.num_nodes))

    def to_mapping(self) -> dict[str, np.ndarray]:
        """Return the exact NPZ-compatible calibration representation."""

        return {
            "calibration_schema": np.asarray(RECOVERY_CALIBRATION_SCHEMA),
            "coordinate_system": np.asarray("state_normalized"),
            "structured_distribution": np.asarray(
                "centered_gaussian_in_parent_error_pair_pca"
            ),
            "parent_seeds": np.asarray(SUCCESSOR_SEEDS, dtype=np.int64),
            "train_centers": np.asarray(TRAIN_CENTER_INDICES, dtype=np.int64),
            "structured_pair_centers": np.asarray(
                STRUCTURED_PAIR_CENTER_INDICES, dtype=np.int64
            ),
            "structured_rank": np.asarray(ERROR_SUBSPACE_RANK, dtype=np.int64),
            "iid_field_rms": self.iid_field_rms.numpy().copy(),
            "structured_basis": self.structured_basis.numpy().copy(),
            "structured_coefficient_std": (
                self.structured_coefficient_std.numpy().copy()
            ),
            "structured_eigenvalues": self.structured_eigenvalues.numpy().copy(),
            "structured_captured_variance": np.asarray(
                self.structured_captured_variance, dtype=np.float64
            ),
            "structured_energy_rescale": np.asarray(
                self.structured_energy_rescale, dtype=np.float64
            ),
            "iid_expected_history_pair_energy": np.asarray(
                self.iid_expected_history_pair_energy, dtype=np.float64
            ),
            "pair_sample_count": np.asarray(self.pair_sample_count, dtype=np.int64),
            "num_nodes": np.asarray(self.num_nodes, dtype=np.int64),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> RecoveryCalibration:
        """Reconstruct only the exact frozen calibration representation."""

        expected_keys = {
            "calibration_schema",
            "coordinate_system",
            "structured_distribution",
            "parent_seeds",
            "train_centers",
            "structured_pair_centers",
            "structured_rank",
            "iid_field_rms",
            "structured_basis",
            "structured_coefficient_std",
            "structured_eigenvalues",
            "structured_captured_variance",
            "structured_energy_rescale",
            "iid_expected_history_pair_energy",
            "pair_sample_count",
            "num_nodes",
        }
        if set(value) != expected_keys:
            raise ValueError("recovery calibration mapping keys differ")
        for key, expected in (
            ("calibration_schema", RECOVERY_CALIBRATION_SCHEMA),
            ("coordinate_system", "state_normalized"),
            (
                "structured_distribution",
                "centered_gaussian_in_parent_error_pair_pca",
            ),
            ("structured_rank", ERROR_SUBSPACE_RANK),
        ):
            observed = (
                _mapping_integer_scalar(value[key], key)
                if key == "structured_rank"
                else _mapping_scalar(value[key], key)
            )
            if observed != expected:
                raise ValueError(f"recovery calibration {key} differs")
        for key, expected in (
            ("parent_seeds", SUCCESSOR_SEEDS),
            ("train_centers", TRAIN_CENTER_INDICES),
            ("structured_pair_centers", STRUCTURED_PAIR_CENTER_INDICES),
        ):
            observed = np.asarray(value[key])
            if observed.dtype.kind not in "iu" or tuple(observed.tolist()) != expected:
                raise ValueError(f"recovery calibration {key} differs")
        for key in (
            "iid_field_rms",
            "structured_basis",
            "structured_coefficient_std",
            "structured_eigenvalues",
        ):
            if np.asarray(value[key]).dtype != np.float64:
                raise TypeError(f"recovery calibration {key} must retain float64")
        return cls(
            iid_field_rms=torch.from_numpy(
                np.array(value["iid_field_rms"], dtype=np.float64, copy=True)
            ),
            structured_basis=torch.from_numpy(
                np.array(value["structured_basis"], dtype=np.float64, copy=True)
            ),
            structured_coefficient_std=torch.from_numpy(
                np.array(
                    value["structured_coefficient_std"],
                    dtype=np.float64,
                    copy=True,
                )
            ),
            structured_eigenvalues=torch.from_numpy(
                np.array(value["structured_eigenvalues"], dtype=np.float64, copy=True)
            ),
            structured_captured_variance=_mapping_float_scalar(
                value["structured_captured_variance"],
                "structured_captured_variance",
            ),
            structured_energy_rescale=_mapping_float_scalar(
                value["structured_energy_rescale"],
                "structured_energy_rescale",
            ),
            iid_expected_history_pair_energy=_mapping_float_scalar(
                value["iid_expected_history_pair_energy"],
                "iid_expected_history_pair_energy",
            ),
            pair_sample_count=_mapping_integer_scalar(
                value["pair_sample_count"], "pair_sample_count"
            ),
            num_nodes=_mapping_integer_scalar(value["num_nodes"], "num_nodes"),
        )


def _canonicalize_basis_signs(basis: torch.Tensor) -> torch.Tensor:
    result = basis.clone()
    for row in range(result.shape[0]):
        pivot = int(torch.argmax(torch.abs(result[row])).item())
        if result[row, pivot] < 0.0:
            result[row].neg_()
    return result


def fit_recovery_calibration(
    parent_teacher_forced_state_normalized_errors: torch.Tensor | np.ndarray,
    transition_center_indices: Iterable[int],
    parent_seeds: Sequence[int],
    parent_contract: VerifiedNACAContract,
) -> RecoveryCalibration:
    """Fit the frozen train-only IID scale and exact rank-16 PCA covariance.

    Input errors already have the contract's state-normalized coordinates and
    shape ``[3,238,N,5]``.  This function deliberately performs no second
    normalization.
    """

    centers = _strict_indices(transition_center_indices, "parent error centers")
    if centers != TRAIN_CENTER_INDICES:
        raise ValueError("recovery calibration requires all ordered train centers")
    validate_role_access(parent_contract, "train", centers, horizon_steps=1)
    if tuple(parent_seeds) != SUCCESSOR_SEEDS:
        raise ValueError("recovery calibration parent seeds differ")
    errors = torch.as_tensor(parent_teacher_forced_state_normalized_errors)
    if errors.requires_grad:
        raise ValueError("recovery calibration errors must be detached")
    expected_prefix = (len(SUCCESSOR_SEEDS), len(TRAIN_CENTER_INDICES))
    if (
        errors.ndim != 4
        or tuple(errors.shape[:2]) != expected_prefix
        or errors.shape[-1] != 5
        or errors.shape[2] < 1
        or not errors.is_floating_point()
    ):
        raise ValueError("parent errors must have shape [3,238,N,5]")
    if not torch.isfinite(errors).all():
        raise ValueError("parent errors contain nonfinite values")
    normalized = errors.detach().to(device="cpu", dtype=torch.float64)
    iid_field_rms = torch.sqrt(torch.mean(torch.square(normalized), dim=(0, 1, 2)))
    if torch.any(iid_field_rms <= 0.0):
        raise ValueError("every IID recovery field must have positive parent error RMS")
    if tuple(centers[1:]) != STRUCTURED_PAIR_CENTER_INDICES:
        raise AssertionError("structured pair indexing differs from the contract")
    pair_values = torch.stack(
        (normalized[:, :-1], normalized[:, 1:]),
        dim=2,
    ).reshape(-1, 2, errors.shape[2], 5)
    flat = pair_values.reshape(pair_values.shape[0], -1)
    centered = flat - torch.mean(flat, dim=0, keepdim=True)
    covariance_gram = centered @ centered.T / float(centered.shape[0] - 1)
    covariance_gram = 0.5 * (covariance_gram + covariance_gram.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_gram)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = torch.clamp(eigenvalues[order], min=0.0)
    eigenvectors = eigenvectors[:, order]
    total_variance = float(torch.sum(eigenvalues).item())
    if not math.isfinite(total_variance) or total_variance <= 0.0:
        raise ValueError("centered parent error pairs have no finite variance")
    top_values = eigenvalues[:ERROR_SUBSPACE_RANK]
    tolerance = (
        torch.finfo(torch.float64).eps
        * max(centered.shape)
        * float(eigenvalues[0].item())
    )
    if top_values.shape[0] != ERROR_SUBSPACE_RANK or torch.any(top_values <= tolerance):
        raise ValueError("parent error pairs do not support the frozen rank 16")
    denominators = torch.sqrt(float(centered.shape[0] - 1) * top_values).unsqueeze(1)
    basis_flat = eigenvectors[:, :ERROR_SUBSPACE_RANK].T @ centered
    basis_flat = _canonicalize_basis_signs(basis_flat / denominators)
    closure = basis_flat @ basis_flat.T
    if not torch.allclose(
        closure,
        torch.eye(ERROR_SUBSPACE_RANK, dtype=torch.float64),
        rtol=1.0e-8,
        atol=1.0e-10,
    ):
        raise RuntimeError("structured PCA basis is not numerically orthonormal")
    num_nodes = int(errors.shape[2])
    iid_expected_energy = float(
        2.0 * num_nodes * torch.sum(torch.square(iid_field_rms)).item()
    )
    retained_energy = float(torch.sum(top_values).item())
    energy_rescale = math.sqrt(iid_expected_energy / retained_energy)
    coefficient_std = torch.sqrt(top_values) * energy_rescale
    return RecoveryCalibration(
        iid_field_rms=iid_field_rms,
        structured_basis=basis_flat.reshape(ERROR_SUBSPACE_RANK, 2, num_nodes, 5),
        structured_coefficient_std=coefficient_std,
        structured_eigenvalues=top_values,
        structured_captured_variance=retained_energy / total_variance,
        structured_energy_rescale=energy_rescale,
        iid_expected_history_pair_energy=iid_expected_energy,
        pair_sample_count=int(pair_values.shape[0]),
        num_nodes=num_nodes,
    )


def sample_structured_history_noise(
    calibration: RecoveryCalibration,
    batch_size: int,
    *,
    generator: torch.Generator,
    device: str | torch.device,
    dtype: torch.dtype = torch.float32,
) -> HistoryNoise:
    """Sample the frozen zero-mean matched-energy rank-16 Gaussian law."""

    if not isinstance(calibration, RecoveryCalibration):
        raise TypeError("a RecoveryCalibration is required")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, Integral)
        or batch_size < 1
    ):
        raise ValueError("structured-noise batch size must be a positive integer")
    target_device = torch.device(device)
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("structured noise supports only float32 or float64")
    _require_generator(generator, target_device)
    coefficients = torch.randn(
        (int(batch_size), ERROR_SUBSPACE_RANK),
        dtype=dtype,
        device=target_device,
        generator=generator,
    )
    coefficients = coefficients * calibration.structured_coefficient_std.to(
        device=target_device, dtype=dtype
    )
    basis = calibration.structured_basis.to(device=target_device, dtype=dtype)
    noise = coefficients @ basis.reshape(ERROR_SUBSPACE_RANK, -1)
    noise = noise.reshape(int(batch_size), 2, calibration.num_nodes, 5)
    return HistoryNoise(noise[:, 0], noise[:, 1])


@dataclass(frozen=True)
class DetachedPushforwardPresentation:
    """One detached model-prefix input and its stored-clean-future target."""

    previous: torch.Tensor
    generated_current: torch.Tensor
    target_normalized_residual: torch.Tensor
    center_indices: tuple[int, ...]


def make_detached_pushforward_presentation(
    model: NACAPCNOResidual,
    state_n_minus_2: torch.Tensor,
    state_n_minus_1: torch.Tensor,
    clean_state_n_plus_1: torch.Tensor,
    center_indices: Iterable[int],
    geometry_batch: Mapping[str, torch.Tensor],
    normalization: NACANormalization,
    parent_contract: VerifiedNACAContract,
    *,
    fourier_tensors: tuple[torch.Tensor, ...] | None = None,
) -> DetachedPushforwardPresentation:
    """Construct ``(u[n-1], stopgrad(Psi(u[n-2],u[n-1])))`` exactly."""

    _require_normalization(normalization)
    centers = validate_pushforward_training_centers(parent_contract, center_indices)
    state_n_minus_2, state_n_minus_1, clean_state_n_plus_1 = _require_same_states(
        (
            ("state n-2", state_n_minus_2),
            ("state n-1", state_n_minus_1),
            ("clean state n+1", clean_state_n_plus_1),
        )
    )
    if state_n_minus_2.shape[0] != len(centers):
        raise ValueError("pushforward centers must align one-to-one with the batch")
    with torch.no_grad():
        _, generated_current = recurrent_step(
            model,
            state_n_minus_2,
            state_n_minus_1,
            geometry_batch,
            normalization,
            fourier_tensors=fourier_tensors,
        )
    generated_current = generated_current.detach()
    target = normalization.normalize_residual(clean_state_n_plus_1 - generated_current)
    return DetachedPushforwardPresentation(
        previous=state_n_minus_1,
        generated_current=generated_current,
        target_normalized_residual=target,
        center_indices=centers,
    )


@dataclass(frozen=True)
class PathProjectionResult:
    """Full-state segment interpolant and its deterministic path coordinates."""

    corrected_state: torch.Tensor
    segment_indices: torch.Tensor
    segment_fractions: torch.Tensor
    embedding_distance: torch.Tensor


@dataclass(frozen=True)
class TrainPathProjector:
    """Ordered piecewise-linear path fitted only from normalized train states."""

    normalized_path: torch.Tensor
    pca_mean: torch.Tensor
    pca_basis: torch.Tensor
    path_embedding: torch.Tensor
    normalization: NACANormalization
    frame_indices: tuple[int, ...]
    variance_rank_without_cap: int
    captured_variance: float
    rank_cap_active: bool

    def __post_init__(self) -> None:
        _require_normalization(self.normalization)
        path = self.normalized_path.detach().contiguous()
        mean = self.pca_mean.detach().contiguous()
        basis = self.pca_basis.detach().contiguous()
        embedding = self.path_embedding.detach().contiguous()
        if path.ndim != 3 or path.shape[-1] != 5 or path.shape[0] != 238:
            raise ValueError(
                "path projector requires 238 normalized train-current states"
            )
        flat_dimension = path.shape[1] * path.shape[2]
        if mean.shape != (flat_dimension,):
            raise ValueError("path PCA mean shape differs")
        if basis.ndim != 2 or basis.shape[1] != flat_dimension:
            raise ValueError("path PCA basis shape differs")
        if basis.shape[0] < 1 or basis.shape[0] > PATH_PCA_RANK_CAP:
            raise ValueError("path PCA rank differs from the frozen cap")
        if embedding.shape != (238, basis.shape[0]):
            raise ValueError("path embedding shape differs")
        tensors = (path, mean, basis, embedding)
        if any(not tensor.is_floating_point() for tensor in tensors):
            raise TypeError("path projector arrays must be floating point")
        if any(not torch.isfinite(tensor).all() for tensor in tensors):
            raise ValueError("path projector arrays contain nonfinite values")
        if any(
            tensor.device != path.device or tensor.dtype != path.dtype
            for tensor in tensors[1:]
        ):
            raise ValueError("path projector arrays must share device and dtype")
        if tuple(self.frame_indices) != TRAIN_CENTER_INDICES:
            raise ValueError("path projector frame identity differs")
        if isinstance(self.variance_rank_without_cap, bool) or not isinstance(
            self.variance_rank_without_cap, Integral
        ):
            raise TypeError("path PCA uncapped rank must be an integer")
        if self.variance_rank_without_cap < basis.shape[0]:
            raise ValueError("path PCA uncapped rank is inconsistent")
        if not isinstance(self.rank_cap_active, (bool, np.bool_)):
            raise TypeError("path PCA cap status must be boolean")
        if bool(self.rank_cap_active) != (
            self.variance_rank_without_cap > PATH_PCA_RANK_CAP
        ):
            raise ValueError("path PCA cap status is inconsistent")
        captured = float(self.captured_variance)
        if not math.isfinite(captured) or not 0.0 < captured <= 1.0 + 1.0e-12:
            raise ValueError("path PCA captured variance is invalid")
        object.__setattr__(self, "normalized_path", path)
        object.__setattr__(self, "pca_mean", mean)
        object.__setattr__(self, "pca_basis", basis)
        object.__setattr__(self, "path_embedding", embedding)
        object.__setattr__(self, "frame_indices", tuple(self.frame_indices))
        object.__setattr__(
            self, "variance_rank_without_cap", int(self.variance_rank_without_cap)
        )
        object.__setattr__(self, "captured_variance", captured)
        object.__setattr__(self, "rank_cap_active", bool(self.rank_cap_active))

    @property
    def rank(self) -> int:
        return int(self.pca_basis.shape[0])

    @property
    def num_nodes(self) -> int:
        return int(self.normalized_path.shape[1])

    def to(
        self,
        device: str | torch.device,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> TrainPathProjector:
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("path projector supports only float32 or float64")
        target = torch.device(device)
        return TrainPathProjector(
            normalized_path=self.normalized_path.to(device=target, dtype=dtype),
            pca_mean=self.pca_mean.to(device=target, dtype=dtype),
            pca_basis=self.pca_basis.to(device=target, dtype=dtype),
            path_embedding=self.path_embedding.to(device=target, dtype=dtype),
            normalization=self.normalization,
            frame_indices=self.frame_indices,
            variance_rank_without_cap=self.variance_rank_without_cap,
            captured_variance=self.captured_variance,
            rank_cap_active=self.rank_cap_active,
        )

    def project(self, state: torch.Tensor) -> PathProjectionResult:
        if not isinstance(state, torch.Tensor):
            raise TypeError("path-projection query must be a torch.Tensor")
        if (
            state.dtype not in (torch.float32, torch.float64)
            or state.ndim != 3
            or state.shape[-1] != 5
        ):
            raise ValueError(
                "path-projection query must have float32/float64 shape [B,N,5]"
            )
        if not torch.isfinite(state).all():
            raise ValueError("path-projection query contains nonfinite values")
        if state.shape[1:] != (self.num_nodes, 5):
            raise ValueError("path-projection query differs from the train mesh")
        if (
            state.device != self.normalized_path.device
            or state.dtype != self.normalized_path.dtype
        ):
            raise ValueError(
                "move the path projector to the query device and dtype first"
            )
        normalized = self.normalization.normalize_state(state)
        query_embedding = (
            normalized.reshape(state.shape[0], -1) - self.pca_mean
        ) @ self.pca_basis.T
        segment_start = self.path_embedding[:-1]
        segment_delta = self.path_embedding[1:] - segment_start
        denominator = torch.sum(torch.square(segment_delta), dim=1)
        relative = query_embedding[:, None, :] - segment_start[None, :, :]
        numerator = torch.sum(relative * segment_delta[None, :, :], dim=2)
        fractions = torch.where(
            denominator[None, :] > 0.0,
            numerator
            / torch.clamp_min(denominator[None, :], torch.finfo(state.dtype).tiny),
            torch.zeros_like(numerator),
        ).clamp(0.0, 1.0)
        projected_embedding = (
            segment_start[None, :, :]
            + fractions[:, :, None] * segment_delta[None, :, :]
        )
        distances_squared = torch.sum(
            torch.square(query_embedding[:, None, :] - projected_embedding), dim=2
        )
        segment_indices = torch.argmin(distances_squared, dim=1)
        batch_indices = torch.arange(state.shape[0], device=state.device)
        selected_fractions = fractions[batch_indices, segment_indices]
        left = self.normalized_path.index_select(0, segment_indices)
        right = self.normalized_path.index_select(0, segment_indices + 1)
        interpolated = left + selected_fractions[:, None, None] * (right - left)
        state_scale = torch.tensor(
            self.normalization.state_scale, dtype=state.dtype, device=state.device
        )
        state_mean = torch.tensor(
            self.normalization.state_mean, dtype=state.dtype, device=state.device
        )
        corrected = interpolated * state_scale + state_mean
        distances = torch.sqrt(
            torch.clamp_min(distances_squared[batch_indices, segment_indices], 0.0)
        )
        return PathProjectionResult(
            corrected_state=corrected,
            segment_indices=segment_indices,
            segment_fractions=selected_fractions,
            embedding_distance=distances,
        )

    def correct(self, state: torch.Tensor) -> torch.Tensor:
        return self.project(state).corrected_state

    def to_mapping(self) -> dict[str, np.ndarray]:
        """Return the exact NPZ-compatible path-projector representation."""

        if (
            self.normalized_path.device.type != "cpu"
            or self.normalized_path.dtype != torch.float64
        ):
            raise ValueError(
                "serialize only the canonical CPU-float64 fitted path projector"
            )

        return {
            "projector_schema": np.asarray(TRAIN_PATH_PROJECTOR_SCHEMA),
            "fit_scope": np.asarray("train_current_states_only"),
            "candidate_geometry": np.asarray("train_pca_piecewise_linear_path"),
            "tie_break": np.asarray("lowest_segment_index"),
            "variance_target": np.asarray(PATH_PCA_VARIANCE_TARGET, dtype=np.float64),
            "rank_cap": np.asarray(PATH_PCA_RANK_CAP, dtype=np.int64),
            "normalized_path": self.normalized_path.detach().cpu().numpy().copy(),
            "pca_mean": self.pca_mean.detach().cpu().numpy().copy(),
            "pca_basis": self.pca_basis.detach().cpu().numpy().copy(),
            "path_embedding": self.path_embedding.detach().cpu().numpy().copy(),
            "frame_indices": np.asarray(self.frame_indices, dtype=np.int64),
            "normalization_state_mean": self.normalization.state_mean.copy(),
            "normalization_state_scale": self.normalization.state_scale.copy(),
            "normalization_residual_scale": self.normalization.residual_scale.copy(),
            "normalization_state_rms": self.normalization.state_rms.copy(),
            "variance_rank_without_cap": np.asarray(
                self.variance_rank_without_cap, dtype=np.int64
            ),
            "captured_variance": np.asarray(self.captured_variance, dtype=np.float64),
            "rank_cap_active": np.asarray(self.rank_cap_active, dtype=np.bool_),
        }

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        normalization: NACANormalization,
    ) -> TrainPathProjector:
        """Reconstruct only a path projector bound to the supplied normalizer."""

        _require_normalization(normalization)
        expected_keys = {
            "projector_schema",
            "fit_scope",
            "candidate_geometry",
            "tie_break",
            "variance_target",
            "rank_cap",
            "normalized_path",
            "pca_mean",
            "pca_basis",
            "path_embedding",
            "frame_indices",
            "normalization_state_mean",
            "normalization_state_scale",
            "normalization_residual_scale",
            "normalization_state_rms",
            "variance_rank_without_cap",
            "captured_variance",
            "rank_cap_active",
        }
        if set(value) != expected_keys:
            raise ValueError("train path projector mapping keys differ")
        for key, expected in (
            ("projector_schema", TRAIN_PATH_PROJECTOR_SCHEMA),
            ("fit_scope", "train_current_states_only"),
            ("candidate_geometry", "train_pca_piecewise_linear_path"),
            ("tie_break", "lowest_segment_index"),
            ("variance_target", PATH_PCA_VARIANCE_TARGET),
            ("rank_cap", PATH_PCA_RANK_CAP),
        ):
            observed = (
                _mapping_integer_scalar(value[key], key)
                if key == "rank_cap"
                else _mapping_scalar(value[key], key)
            )
            if observed != expected:
                raise ValueError(f"train path projector {key} differs")
        frames = np.asarray(value["frame_indices"])
        if (
            frames.dtype.kind not in "iu"
            or tuple(frames.tolist()) != TRAIN_CENTER_INDICES
        ):
            raise ValueError("train path projector frame indices differ")
        for key, expected in (
            ("normalization_state_mean", normalization.state_mean),
            ("normalization_state_scale", normalization.state_scale),
            ("normalization_residual_scale", normalization.residual_scale),
            ("normalization_state_rms", normalization.state_rms),
        ):
            raw = np.asarray(value[key])
            if raw.dtype != np.float64:
                raise TypeError(f"train path projector {key} must retain float64")
            observed = np.asarray(raw, dtype=np.float64)
            if not np.array_equal(observed, expected):
                raise ValueError(f"train path projector {key} differs")
        for key in ("normalized_path", "pca_mean", "pca_basis", "path_embedding"):
            if np.asarray(value[key]).dtype != np.float64:
                raise TypeError(f"train path projector {key} must retain float64")
        cap_active = _mapping_scalar(value["rank_cap_active"], "rank_cap_active")
        if not isinstance(cap_active, (bool, np.bool_)):
            raise TypeError("train path projector cap flag must be boolean")
        projector = cls(
            normalized_path=torch.from_numpy(
                np.array(value["normalized_path"], copy=True)
            ),
            pca_mean=torch.from_numpy(np.array(value["pca_mean"], copy=True)),
            pca_basis=torch.from_numpy(np.array(value["pca_basis"], copy=True)),
            path_embedding=torch.from_numpy(
                np.array(value["path_embedding"], copy=True)
            ),
            normalization=normalization,
            frame_indices=tuple(int(item) for item in frames),
            variance_rank_without_cap=_mapping_integer_scalar(
                value["variance_rank_without_cap"],
                "variance_rank_without_cap",
            ),
            captured_variance=_mapping_float_scalar(
                value["captured_variance"], "captured_variance"
            ),
            rank_cap_active=bool(cap_active),
        )
        validate_train_path_projector_closure(projector)
        return projector


def validate_train_path_projector_closure(projector: TrainPathProjector) -> None:
    """Verify the stored PCA mean, basis, and embedding against the path."""

    if not isinstance(projector, TrainPathProjector):
        raise TypeError("a TrainPathProjector is required")
    flat = projector.normalized_path.reshape(238, -1)
    dtype = projector.normalized_path.dtype
    if dtype == torch.float64:
        rtol, atol = 1.0e-10, 1.0e-12
    else:
        rtol, atol = 2.0e-5, 2.0e-6
    expected_mean = torch.mean(flat, dim=0)
    if not torch.allclose(projector.pca_mean, expected_mean, rtol=rtol, atol=atol):
        raise ValueError("train path projector PCA mean does not close")
    basis_closure = projector.pca_basis @ projector.pca_basis.T
    if not torch.allclose(
        basis_closure,
        torch.eye(projector.rank, dtype=dtype, device=basis_closure.device),
        rtol=rtol,
        atol=atol,
    ):
        raise ValueError("train path projector PCA basis is not orthonormal")
    expected_embedding = (flat - projector.pca_mean) @ projector.pca_basis.T
    if not torch.allclose(
        projector.path_embedding,
        expected_embedding,
        rtol=rtol,
        atol=atol,
    ):
        raise ValueError("train path projector embedding does not close")


def fit_train_path_projector(
    train_current_states: np.ndarray,
    frame_indices: Iterable[int],
    normalization: NACANormalization,
    parent_contract: VerifiedNACAContract,
) -> TrainPathProjector:
    """Fit the registered path PCA after rejecting any nontrain frame identity."""

    _require_normalization(normalization)
    frames = validate_train_path_indices(parent_contract, frame_indices)
    states = np.asarray(train_current_states)
    if (
        states.dtype != np.float64
        or states.ndim != 3
        or states.shape[0] != 238
        or states.shape[-1] != 5
    ):
        raise ValueError("path calibration states must retain float64 [238,N,5]")
    if states.shape[1] < 1 or not np.all(np.isfinite(states)):
        raise ValueError("path calibration states must be finite and contain nodes")
    normalized = np.asarray(normalization.normalize_state(states), dtype=np.float64)
    path = torch.from_numpy(np.array(normalized, copy=True))
    flat = path.reshape(path.shape[0], -1)
    mean = torch.mean(flat, dim=0)
    centered = flat - mean
    _, singular_values, right_vectors = torch.linalg.svd(centered, full_matrices=False)
    variance = torch.square(singular_values)
    total_variance = float(torch.sum(variance).item())
    if not math.isfinite(total_variance) or total_variance <= 0.0:
        raise ValueError("train path has no finite PCA variance")
    cumulative = torch.cumsum(variance, dim=0) / total_variance
    threshold_index = torch.nonzero(
        cumulative >= PATH_PCA_VARIANCE_TARGET, as_tuple=False
    )
    if threshold_index.numel() == 0:
        variance_rank_without_cap = int(cumulative.shape[0])
    else:
        variance_rank_without_cap = int(threshold_index[0, 0].item()) + 1
    rank = min(variance_rank_without_cap, PATH_PCA_RANK_CAP)
    basis = _canonicalize_basis_signs(right_vectors[:rank])
    embedding = centered @ basis.T
    captured = float(torch.sum(variance[:rank]).item() / total_variance)
    return TrainPathProjector(
        normalized_path=path,
        pca_mean=mean,
        pca_basis=basis,
        path_embedding=embedding,
        normalization=normalization,
        frame_indices=frames,
        variance_rank_without_cap=variance_rank_without_cap,
        captured_variance=captured,
        rank_cap_active=variance_rank_without_cap > PATH_PCA_RANK_CAP,
    )


@dataclass(frozen=True)
class CorrectedRecurrentStep:
    """Raw learned prediction and corrected state fed to the next step."""

    recurrent_previous: torch.Tensor
    raw_next_state: torch.Tensor
    corrected_next_state: torch.Tensor


def identity_corrector(state: torch.Tensor) -> torch.Tensor:
    """The exact no-op corrector used for recurrence parity checks."""

    return state


def corrected_recurrent_step(
    model: NACAPCNOResidual,
    previous: torch.Tensor,
    current: torch.Tensor,
    geometry_batch: Mapping[str, torch.Tensor],
    normalization: NACANormalization,
    corrector: Callable[[torch.Tensor], torch.Tensor],
    *,
    fourier_tensors: tuple[torch.Tensor, ...] | None = None,
) -> CorrectedRecurrentStep:
    """Predict raw, correct after prediction, and feed back the corrected state."""

    if not callable(corrector):
        raise TypeError("corrector must be callable")
    recurrent_previous, raw_next = recurrent_step(
        model,
        previous,
        current,
        geometry_batch,
        normalization,
        fourier_tensors=fourier_tensors,
    )
    corrected_next = corrector(raw_next)
    if not isinstance(corrected_next, torch.Tensor):
        raise TypeError("corrector must return a torch.Tensor")
    if (
        corrected_next.shape != raw_next.shape
        or corrected_next.dtype != raw_next.dtype
        or corrected_next.device != raw_next.device
    ):
        raise ValueError("corrected state must preserve raw prediction metadata")
    return CorrectedRecurrentStep(
        recurrent_previous=recurrent_previous,
        raw_next_state=raw_next,
        corrected_next_state=corrected_next,
    )
