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
            face_flux_integral=None
            if "face_flux_integral" not in arrays
            else np.asarray(arrays["face_flux_integral"], dtype=np.float32),
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


def conservative_restrict(
    primitive: np.ndarray,
    coarse_cells: int,
    gamma: float,
) -> np.ndarray:
    """Restrict uniform fine-grid cell states by conservative cell averaging."""

    if primitive.ndim < 2 or primitive.shape[-1] != 3:
        raise ValueError("primitive must end in [cells, 3]")
    fine_cells = int(primitive.shape[-2])
    if coarse_cells < 1 or fine_cells % coarse_cells != 0:
        raise ValueError("fine resolution must be an integer multiple of coarse")
    factor = fine_cells // coarse_cells
    conservative = primitive_to_conservative_np(primitive, gamma)
    restricted = conservative.reshape(
        *conservative.shape[:-2], coarse_cells, factor, 3
    ).mean(axis=-2)
    return conservative_to_primitive_np(restricted, gamma)


def restrict_euler1d_source(
    source: Euler1DNPZ,
    coarse_cells: int,
) -> Euler1DNPZ:
    """Build a uniform coarse source by restricting every saved fine state."""

    source.validate()
    if source.num_cells < 2:
        raise ValueError("fine source must contain at least two cells")
    if coarse_cells >= source.num_cells:
        raise ValueError("coarse resolution must be smaller than the source")
    if source.num_cells % coarse_cells != 0:
        raise ValueError("source resolution must be divisible by coarse resolution")

    fine_x = source.x.astype(np.float64)
    fine_dx = np.diff(fine_x, axis=1)
    mean_dx = fine_dx.mean(axis=1)
    scale = max(1.0, float(np.max(np.abs(fine_x))))
    if float(np.max(np.abs(fine_dx - mean_dx[:, None]))) > 2.0e-6 * scale:
        raise ValueError("restriction requires a uniform fine grid")
    left = fine_x[:, 0] - 0.5 * mean_dx
    right = fine_x[:, -1] + 0.5 * mean_dx
    coarse_dx = (right - left) / coarse_cells
    offsets = np.arange(coarse_cells, dtype=np.float64) + 0.5
    coarse_x = left[:, None] + coarse_dx[:, None] * offsets[None, :]

    metadata = {
        key: value
        for key, value in source.metadata.items()
        if not key.startswith("face_flux")
    }
    metadata.update(
        {
            "has_face_flux_integral": np.array(False, dtype=np.bool_),
            "nx": np.array(coarse_cells, dtype=np.int32),
            "restriction_source_nx": np.array(source.num_cells, dtype=np.int32),
            "target_contract": np.array(
                "conservative restriction of common fine-grid trajectory"
            ),
        }
    )
    restricted = Euler1DNPZ(
        data=conservative_restrict(
            source.data.astype(np.float64),
            coarse_cells,
            source.gamma,
        ),
        x=coarse_x.astype(np.float32),
        t=source.t.copy(),
        left_states=source.left_states.copy(),
        right_states=source.right_states.copy(),
        gamma=source.gamma,
        metadata=metadata,
        face_flux_integral=None,
    )
    restricted.validate()
    return restricted


def restriction_commutation_metrics(
    coarse: Euler1DNPZ,
    fine: Euler1DNPZ,
    *,
    case_indices: np.ndarray | list[int] | None = None,
    strides: tuple[int, ...] = (1, 2),
) -> dict[str, Any]:
    """Measure state and increment closure for an adjacent restriction pair."""

    coarse.validate()
    fine.validate()
    if fine.num_cells % coarse.num_cells != 0:
        raise ValueError("fine resolution must be divisible by coarse resolution")
    if coarse.num_cases != fine.num_cases or coarse.num_frames != fine.num_frames:
        raise ValueError("coarse and fine trajectory counts must match")
    if case_indices is None:
        cases = np.arange(coarse.num_cases, dtype=np.int64)
    else:
        cases = np.asarray(case_indices, dtype=np.int64)
    if np.any(cases < 0) or np.any(cases >= coarse.num_cases):
        raise ValueError("case_indices contains out-of-range entries")

    coarse_conservative = primitive_to_conservative_np(
        coarse.data[cases].astype(np.float64), coarse.gamma
    )
    restricted_conservative = primitive_to_conservative_np(
        conservative_restrict(
            fine.data[cases].astype(np.float64),
            coarse.num_cells,
            coarse.gamma,
        ),
        coarse.gamma,
    )
    state_difference = (restricted_conservative - coarse_conservative).reshape(
        cases.size, coarse.num_frames, -1
    )
    state_norm = np.linalg.norm(
        coarse_conservative.reshape(cases.size, coarse.num_frames, -1), axis=-1
    )
    state_relative = np.linalg.norm(state_difference, axis=-1) / np.maximum(
        state_norm, 1.0e-12
    )
    state_global_relative = float(
        np.linalg.norm(state_difference.reshape(-1))
        / max(np.linalg.norm(coarse_conservative.reshape(-1)), 1.0e-12)
    )
    state_max_abs = float(np.max(np.abs(state_difference)))

    update_rows: list[dict[str, float | int]] = []
    for stride in sorted(set(strides)):
        if stride < 1 or stride >= coarse.num_frames:
            raise ValueError("strides must lie in [1, num_frames - 1]")
        coarse_delta = (
            coarse_conservative[:, stride:] - coarse_conservative[:, :-stride]
        )
        restricted_delta = (
            restricted_conservative[:, stride:] - restricted_conservative[:, :-stride]
        )
        difference = (restricted_delta - coarse_delta).reshape(
            cases.size, coarse.num_frames - stride, -1
        )
        update_norm = np.linalg.norm(
            coarse_delta.reshape(cases.size, coarse.num_frames - stride, -1),
            axis=-1,
        )
        update_relative = np.linalg.norm(difference, axis=-1) / np.maximum(
            update_norm, 1.0e-12
        )
        update_global_relative = float(
            np.linalg.norm(difference.reshape(-1))
            / max(np.linalg.norm(coarse_delta.reshape(-1)), 1.0e-12)
        )
        update_rows.append(
            {
                "stride": stride,
                "global_relative_l2": update_global_relative,
                "max_abs": float(np.max(np.abs(difference))),
                "relative_l2_mean": float(update_relative.mean()),
                "relative_l2_max": float(update_relative.max()),
            }
        )

    return {
        "coarse_cells": coarse.num_cells,
        "fine_cells": fine.num_cells,
        "case_count": int(cases.size),
        "state_global_relative_l2": state_global_relative,
        "state_max_abs": state_max_abs,
        "state_relative_l2_mean": float(state_relative.mean()),
        "state_relative_l2_max": float(state_relative.max()),
        "updates": update_rows,
    }


def save_euler1d_npz(path: str | Path, source: Euler1DNPZ) -> None:
    """Serialize an Euler1DNPZ without introducing object arrays."""

    source.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "data": source.data,
        "x": source.x,
        "t": source.t,
        "left_states": source.left_states,
        "right_states": source.right_states,
        "gamma": np.array(source.gamma, dtype=np.float32),
    }
    reserved = {*payload, "face_flux_integral"}
    overlap = sorted(reserved.intersection(source.metadata))
    if overlap:
        raise ValueError(f"metadata collides with dataset fields: {overlap}")
    payload.update(source.metadata)
    if source.face_flux_integral is not None:
        payload["face_flux_integral"] = source.face_flux_integral
    np.savez_compressed(path, **payload)


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
