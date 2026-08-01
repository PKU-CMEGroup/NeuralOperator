"""Dataset helpers for collaborator-generated 1D Euler ``.npz`` files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from utility.time_dependent_no.euler1d import Euler1DBatch, make_euler1d_batch


@dataclass(frozen=True)
class Euler1DNPZ:
    """In-memory view of the collaborator 1D Euler dataset contract."""

    data: np.ndarray
    x: np.ndarray
    t: np.ndarray
    left_states: np.ndarray
    right_states: np.ndarray
    gamma: float
    metadata: dict[str, Any]
    face_flux_integral: np.ndarray | None = None

    @property
    def num_cases(self) -> int:
        return int(self.data.shape[0])

    @property
    def num_frames(self) -> int:
        return int(self.data.shape[1])

    @property
    def num_cells(self) -> int:
        return int(self.data.shape[2])

    def validate(self) -> None:
        if self.data.ndim != 4 or self.data.shape[-1] != 3:
            raise ValueError("data must have shape [cases, frames, cells, 3]")
        if any(size < 1 for size in self.data.shape[:3]):
            raise ValueError("data case, frame, and cell axes must be nonempty")
        if self.x.shape != (self.num_cases, self.num_cells):
            raise ValueError("x must have shape [cases, cells]")
        if self.t.shape != (self.num_cases, self.num_frames):
            raise ValueError("t must have shape [cases, frames]")
        if self.left_states.shape != (self.num_cases, 3):
            raise ValueError("left_states must have shape [cases, 3]")
        if self.right_states.shape != (self.num_cases, 3):
            raise ValueError("right_states must have shape [cases, 3]")
        if self.face_flux_integral is not None and self.face_flux_integral.shape != (
            self.num_cases,
            self.num_frames - 1,
            self.num_cells + 1,
            3,
        ):
            raise ValueError(
                "face_flux_integral must have shape [cases, frames - 1, cells + 1, 3]"
            )
        arrays = {
            "data": self.data,
            "x": self.x,
            "t": self.t,
            "left_states": self.left_states,
            "right_states": self.right_states,
        }
        if self.face_flux_integral is not None:
            arrays["face_flux_integral"] = self.face_flux_integral
        for name, values in arrays.items():
            if not np.isfinite(values).all():
                raise ValueError(f"{name} must contain only finite values")
        if not np.isfinite(self.gamma) or self.gamma <= 1.0:
            raise ValueError("gamma must be finite and greater than one")
        if self.num_cells > 1 and np.any(np.diff(self.x, axis=1) <= 0.0):
            raise ValueError("x must be strictly increasing within every case")
        if self.num_frames > 1 and np.any(np.diff(self.t, axis=1) <= 0.0):
            raise ValueError("t must be strictly increasing within every case")
        if np.any(self.data[..., 0] <= 0.0) or np.any(self.data[..., 2] <= 0.0):
            raise ValueError("serialized density and pressure must be positive")
        for name, states in (
            ("left_states", self.left_states),
            ("right_states", self.right_states),
        ):
            if np.any(states[:, 0] <= 0.0) or np.any(states[:, 2] <= 0.0):
                raise ValueError(f"{name} density and pressure must be positive")


def load_euler1d_npz(path: str | Path) -> Euler1DNPZ:
    """Load and validate a collaborator 1D Euler dataset file."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as arrays:
        required = ("data", "x", "t", "left_states", "right_states", "gamma")
        optional_arrays = ("face_flux_integral",)
        missing = [key for key in required if key not in arrays]
        if missing:
            raise KeyError(f"missing required Euler1D keys: {missing}")

        metadata = {
            key: arrays[key].item() if arrays[key].shape == () else arrays[key]
            for key in arrays.files
            if key not in (*required, *optional_arrays)
        }
        dataset = Euler1DNPZ(
            data=np.asarray(arrays["data"]),
            x=np.asarray(arrays["x"], dtype=np.float32),
            t=np.asarray(arrays["t"], dtype=np.float32),
            left_states=np.asarray(arrays["left_states"], dtype=np.float32),
            right_states=np.asarray(arrays["right_states"], dtype=np.float32),
            gamma=float(np.asarray(arrays["gamma"]).item()),
            metadata=metadata,
            face_flux_integral=(
                None
                if "face_flux_integral" not in arrays
                else np.asarray(arrays["face_flux_integral"], dtype=np.float32)
            ),
        )
    dataset.validate()
    return dataset


def primitive_to_conservative_np(
    primitive: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Convert NumPy primitive states [rho, u, p] to conserved states."""

    rho = primitive[..., 0]
    velocity = primitive[..., 1]
    pressure = primitive[..., 2]
    energy = pressure / (gamma - 1.0) + 0.5 * rho * velocity**2
    return np.stack((rho, rho * velocity, energy), axis=-1)


def conservative_to_primitive_np(
    conservative: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Convert NumPy conserved states to primitive states [rho, u, p]."""

    rho = conservative[..., 0]
    velocity = conservative[..., 1] / rho
    pressure = (gamma - 1.0) * (conservative[..., 2] - 0.5 * rho * velocity**2)
    return np.stack((rho, velocity, pressure), axis=-1)


class Euler1DTimePairDataset(Dataset):
    """Torch dataset of one-step pairs from a loaded 1D Euler trajectory file."""

    def __init__(
        self,
        source: str | Path | Euler1DNPZ,
        *,
        case_indices: list[int] | np.ndarray | None = None,
        step_stride: int = 1,
    ) -> None:
        if step_stride < 1:
            raise ValueError("step_stride must be >= 1")
        self.source = (
            load_euler1d_npz(source) if not isinstance(source, Euler1DNPZ) else source
        )
        self.source.validate()

        if case_indices is None:
            cases = np.arange(self.source.num_cases, dtype=np.int64)
        else:
            cases = np.asarray(case_indices, dtype=np.int64)
        if np.any(cases < 0) or np.any(cases >= self.source.num_cases):
            raise ValueError("case_indices contains out-of-range entries")

        pairs: list[tuple[int, int]] = []
        for case in cases.tolist():
            for step in range(0, self.source.num_frames - step_stride):
                pairs.append((case, step))
        self.pairs = pairs
        self.step_stride = int(step_stride)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | float]:
        case, step = self.pairs[index]
        next_step = step + self.step_stride
        current = np.asarray(self.source.data[case, step], dtype=np.float32)
        target = np.asarray(self.source.data[case, next_step], dtype=np.float32)
        dt = self.source.t[case, next_step] - self.source.t[case, step]
        sample: dict[str, torch.Tensor | float] = {
            "current_primitive": torch.from_numpy(current),
            "target_primitive": torch.from_numpy(target),
            "x": torch.from_numpy(self.source.x[case]),
            "dt": torch.tensor(dt, dtype=torch.float32),
            "left_boundary_primitive": torch.from_numpy(self.source.left_states[case]),
            "right_initial_primitive": torch.from_numpy(self.source.right_states[case]),
            "gamma": self.source.gamma,
        }
        if self.source.face_flux_integral is not None:
            flux_integral = self.source.face_flux_integral[case, step:next_step].sum(
                axis=0
            )
            sample["target_face_flux"] = torch.from_numpy(
                np.asarray(flux_integral / dt, dtype=np.float32)
            )
        return sample


class Euler1DRolloutWindowDataset(Dataset):
    """Fixed-stride trajectory windows for burn-in and recurrent training."""

    def __init__(
        self,
        source: str | Path | Euler1DNPZ,
        *,
        case_indices: list[int] | np.ndarray | None = None,
        step_stride: int = 1,
        rollout_steps: int = 3,
        burn_in_steps: int = 0,
    ) -> None:
        if step_stride < 1:
            raise ValueError("step_stride must be >= 1")
        if rollout_steps < 2:
            raise ValueError("rollout_steps must be >= 2")
        if burn_in_steps < 0:
            raise ValueError("burn_in_steps must be >= 0")
        self.source = (
            load_euler1d_npz(source) if not isinstance(source, Euler1DNPZ) else source
        )
        self.source.validate()
        if case_indices is None:
            cases = np.arange(self.source.num_cases, dtype=np.int64)
        else:
            cases = np.asarray(case_indices, dtype=np.int64)
        if np.any(cases < 0) or np.any(cases >= self.source.num_cases):
            raise ValueError("case_indices contains out-of-range entries")

        sequence_steps = burn_in_steps + rollout_steps
        horizon = step_stride * sequence_steps
        if horizon >= self.source.num_frames:
            raise ValueError("rollout window exceeds the available trajectory")
        self.windows = [
            (case, start)
            for case in cases.tolist()
            for start in range(self.source.num_frames - horizon)
        ]
        self.step_stride = int(step_stride)
        self.rollout_steps = int(rollout_steps)
        self.burn_in_steps = int(burn_in_steps)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | float]:
        case, start = self.windows[index]
        sequence_steps = self.burn_in_steps + self.rollout_steps
        frame_ids = start + self.step_stride * np.arange(
            1, sequence_steps + 1, dtype=np.int64
        )
        previous_ids = np.concatenate(([start], frame_ids[:-1]))
        target_sequence = np.asarray(
            self.source.data[case, frame_ids],
            dtype=np.float32,
        )
        dt_sequence = self.source.t[case, frame_ids] - self.source.t[case, previous_ids]
        return {
            "current_primitive": torch.from_numpy(
                np.asarray(self.source.data[case, start], dtype=np.float32)
            ),
            "target_sequence": torch.from_numpy(target_sequence),
            "dt_sequence": torch.from_numpy(dt_sequence.astype(np.float32)),
            "x": torch.from_numpy(self.source.x[case]),
            "left_boundary_primitive": torch.from_numpy(self.source.left_states[case]),
            "right_initial_primitive": torch.from_numpy(self.source.right_states[case]),
            "gamma": self.source.gamma,
        }


def collate_euler1d_rollout_windows(
    samples: list[dict[str, torch.Tensor | float]],
) -> tuple[Euler1DBatch, torch.Tensor, torch.Tensor]:
    """Collate autoregressive windows into an initial batch and sequences."""

    if not samples:
        raise ValueError("cannot collate an empty sample list")
    gamma = float(samples[0]["gamma"])
    if any(float(sample["gamma"]) != gamma for sample in samples):
        raise ValueError("mixed gamma values in one batch are not supported")
    current = torch.stack([sample["current_primitive"] for sample in samples])
    targets = torch.stack([sample["target_sequence"] for sample in samples])
    dt_sequence = torch.stack([sample["dt_sequence"] for sample in samples])
    x = torch.stack([sample["x"] for sample in samples])
    left = torch.stack([sample["left_boundary_primitive"] for sample in samples])
    right_initial = torch.stack(
        [sample["right_initial_primitive"] for sample in samples]
    )
    initial_batch = make_euler1d_batch(
        current,
        x,
        dt_sequence[:, 0],
        target_primitive=targets[:, 0],
        gamma=gamma,
        left_boundary_primitive=left,
        right_initial_primitive=right_initial,
    )
    return initial_batch, targets, dt_sequence


def collate_euler1d_pairs(samples: list[dict[str, torch.Tensor | float]]):
    """Collate one-step samples into an ``Euler1DBatch``."""

    if not samples:
        raise ValueError("cannot collate an empty sample list")
    gamma = float(samples[0]["gamma"])
    if any(float(sample["gamma"]) != gamma for sample in samples):
        raise ValueError("mixed gamma values in one batch are not supported")

    current = torch.stack([sample["current_primitive"] for sample in samples])
    target = torch.stack([sample["target_primitive"] for sample in samples])
    x = torch.stack([sample["x"] for sample in samples])
    dt = torch.stack([sample["dt"] for sample in samples])
    left = torch.stack([sample["left_boundary_primitive"] for sample in samples])
    right_initial = torch.stack(
        [sample["right_initial_primitive"] for sample in samples]
    )
    has_target_flux = ["target_face_flux" in sample for sample in samples]
    if any(has_target_flux) and not all(has_target_flux):
        raise ValueError("mixed face-flux supervision in one batch is not supported")
    target_face_flux = (
        torch.stack([sample["target_face_flux"] for sample in samples])
        if all(has_target_flux)
        else None
    )
    return make_euler1d_batch(
        current,
        x,
        dt,
        target_primitive=target,
        target_face_flux=target_face_flux,
        gamma=gamma,
        left_boundary_primitive=left,
        right_initial_primitive=right_initial,
    )
