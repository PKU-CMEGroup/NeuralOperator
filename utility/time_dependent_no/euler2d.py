"""2D Euler helpers and CPG-style dataset inspection.

The Track 1 benchmark data is expected to arrive as HDF5 trajectory groups
matching the structure-preserving GNN reference code. This module keeps the
contract NumPy-only: it can inspect the schema, load primitive-variable
sequences, and materialize a single graph frame without depending on PyTorch
Geometric or h5py at import time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ArrayLike = Any
GroupLike = Mapping[str, Any]

PRIMITIVE_KEYS: tuple[str, ...] = ("rho", "v1", "v2", "pres")
PRIMITIVE_NAMES: tuple[str, ...] = ("rho", "v1", "v2", "pres")
CONSERVATIVE_NAMES: tuple[str, ...] = ("rho", "rho_v1", "rho_v2", "energy")
CPG_REQUIRED_KEYS: tuple[str, ...] = (
    "pos",
    "edges",
    "node_type",
    "pres",
    "rho",
    "v1",
    "v2",
    "Mach",
)


class EulerNodeType(IntEnum):
    """Node-type convention used by the cpgGNSpdes reference code."""

    NORMAL = 0
    WALL = 1
    OUTFLOW = 2
    INFLOW = 3


@dataclass(frozen=True)
class ArraySummary:
    """Small JSON-serializable summary for one dataset array."""

    shape: list[int]
    dtype: str
    numeric: bool
    sample_limited: bool = False
    finite_fraction: float | None = None
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    num_nan: int | None = None
    num_inf: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrajectorySummary:
    """Schema and range summary for one trajectory group."""

    name: str
    keys: list[str]
    missing_keys: list[str]
    arrays: dict[str, ArraySummary]
    num_time_steps: int | None
    num_nodes: int | None
    num_edges: int | None
    node_type_counts: dict[int, int]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["arrays"] = {key: value.to_dict() for key, value in self.arrays.items()}
        return data


@dataclass(frozen=True)
class EulerDatasetSummary:
    """Summary for a CPG-style HDF5 file or mapping."""

    path: str | None
    num_trajectories: int
    inspected_trajectories: int
    trajectory_names: list[str]
    trajectories: list[TrajectorySummary]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["trajectories"] = [item.to_dict() for item in self.trajectories]
        return data


def stack_primitive(
    rho: ArrayLike,
    v1: ArrayLike,
    v2: ArrayLike,
    pressure: ArrayLike,
    *,
    axis: int = -1,
) -> np.ndarray:
    """Stack primitive variables as ``[rho, v1, v2, pres]``."""

    arrays = [np.asarray(value, dtype=np.float64) for value in (rho, v1, v2, pressure)]
    first_shape = arrays[0].shape
    if any(array.shape != first_shape for array in arrays):
        shapes = [array.shape for array in arrays]
        raise ValueError(f"primitive variables must share shape, got {shapes}")
    return np.stack(arrays, axis=axis)


def split_primitive(
    primitive: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``rho, v1, v2, pres`` from an array with last dimension 4."""

    prim = np.asarray(primitive, dtype=np.float64)
    if prim.shape[-1] != 4:
        raise ValueError("primitive array must have last dimension 4")
    return prim[..., 0], prim[..., 1], prim[..., 2], prim[..., 3]


def primitive_to_conservative(
    primitive: ArrayLike,
    *,
    gamma: float = 1.4,
) -> np.ndarray:
    """Convert primitive ``[rho, v1, v2, pres]`` to conservative variables."""

    if gamma <= 1.0:
        raise ValueError("gamma must be greater than 1")
    rho, v1, v2, pressure = split_primitive(primitive)
    energy = pressure / (gamma - 1.0) + 0.5 * rho * (v1 * v1 + v2 * v2)
    return np.stack((rho, rho * v1, rho * v2, energy), axis=-1)


def conservative_to_primitive(
    conservative: ArrayLike,
    *,
    gamma: float = 1.4,
    eps: float = 1e-12,
) -> np.ndarray:
    """Convert conservative ``[rho, rho*v1, rho*v2, E]`` to primitives."""

    if gamma <= 1.0:
        raise ValueError("gamma must be greater than 1")
    cons = np.asarray(conservative, dtype=np.float64)
    if cons.shape[-1] != 4:
        raise ValueError("conservative array must have last dimension 4")
    rho = cons[..., 0]
    v1 = cons[..., 1] / np.maximum(rho, eps)
    v2 = cons[..., 2] / np.maximum(rho, eps)
    kinetic = 0.5 * rho * (v1 * v1 + v2 * v2)
    pressure = (gamma - 1.0) * (cons[..., 3] - kinetic)
    return np.stack((rho, v1, v2, pressure), axis=-1)


def node_type_masks(node_type: ArrayLike) -> dict[str, np.ndarray]:
    """Return boolean masks for CPG node types."""

    values = np.asarray(node_type)
    if values.ndim > 0 and values.shape[-1] == 1:
        values = np.squeeze(values, axis=-1)
    masks = {
        "normal": values == int(EulerNodeType.NORMAL),
        "wall": values == int(EulerNodeType.WALL),
        "outflow": values == int(EulerNodeType.OUTFLOW),
        "inflow": values == int(EulerNodeType.INFLOW),
    }
    masks["boundary"] = masks["wall"] | masks["outflow"] | masks["inflow"]
    return masks


def load_cpg_primitive_sequence(group: GroupLike) -> np.ndarray:
    """Load ``[rho, v1, v2, pres]`` as an array of shape ``(T, N, 4)``."""

    missing = [key for key in PRIMITIVE_KEYS if key not in group]
    if missing:
        raise KeyError(f"missing primitive keys: {missing}")
    arrays = [_as_time_node_scalar(group[key], key) for key in PRIMITIVE_KEYS]
    shapes = [array.shape for array in arrays]
    if any(shape != shapes[0] for shape in shapes):
        raise ValueError(f"primitive arrays must share shape, got {shapes}")
    return np.concatenate(arrays, axis=-1)


def make_cpg_graph_frame(
    group: GroupLike,
    frame: int,
    *,
    num_steps: int = 1,
) -> dict[str, np.ndarray]:
    """Materialize a NumPy graph frame using the reference reader convention."""

    if frame < 0:
        raise ValueError("frame must be nonnegative")
    if num_steps < 1:
        raise ValueError("num_steps must be positive")

    primitive = load_cpg_primitive_sequence(group)
    if frame + num_steps >= primitive.shape[0]:
        raise IndexError(
            f"frame + num_steps must be < {primitive.shape[0]}, got "
            f"{frame} + {num_steps}"
        )

    time_steps = primitive.shape[0]
    pos = _frame_indexed_array(group["pos"], frame, "pos", time_steps=time_steps)
    edges = _frame_indexed_array(
        group["edges"], frame, "edges", time_steps=time_steps
    ).astype(np.int64)
    node_type = _frame_indexed_array(
        group["node_type"], frame, "node_type", time_steps=time_steps
    )
    mach = _frame_indexed_array(group["Mach"], frame, "Mach", time_steps=time_steps)
    node_type = _as_node_column(node_type, "node_type").astype(np.int64)
    mach = _as_node_column(mach, "Mach")

    current = primitive[frame]
    target = primitive[frame + 1]
    future = primitive[frame + 1 : frame + num_steps + 1]

    if current.shape[0] != pos.shape[0]:
        raise ValueError("pos and primitive variables disagree on node count")
    if node_type.shape[0] != current.shape[0] or mach.shape[0] != current.shape[0]:
        raise ValueError(
            "node_type/Mach and primitive variables disagree on node count"
        )
    if edges.ndim != 2 or edges.shape[-1] != 2:
        raise ValueError("edges must have shape (num_edges, 2) after frame indexing")

    x = np.concatenate((node_type.astype(np.float64), current, mach), axis=-1)
    return {
        "x": x,
        "y": target,
        "pos": np.asarray(pos, dtype=np.float64),
        "edges": edges,
        "node_type": node_type[:, 0],
        "mach": mach[:, 0],
        "current_primitives": current,
        "target_primitives": target,
        "future_primitives": future,
    }


def inspect_cpg_hdf5_file(
    path: str | Path,
    *,
    max_trajectories: int | None = 3,
    sample_values: bool = True,
    max_values_per_array: int = 500_000,
) -> EulerDatasetSummary:
    """Inspect a CPG-style HDF5 file.

    ``h5py`` is intentionally imported lazily so the core package stays
    NumPy-only until a real HDF5 dataset needs inspection.
    """

    try:
        import h5py  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "h5py is required to inspect .h5 files; install it in the active "
            "environment or pass an already-open mapping to "
            "inspect_cpg_hdf5_mapping."
        ) from exc

    dataset_path = Path(path)
    with h5py.File(dataset_path, "r") as handle:
        return inspect_cpg_hdf5_mapping(
            handle,
            path=str(dataset_path),
            max_trajectories=max_trajectories,
            sample_values=sample_values,
            max_values_per_array=max_values_per_array,
        )


def inspect_cpg_hdf5_mapping(
    mapping: GroupLike,
    *,
    path: str | None = None,
    max_trajectories: int | None = 3,
    sample_values: bool = True,
    max_values_per_array: int = 500_000,
) -> EulerDatasetSummary:
    """Inspect an already-open HDF5-like mapping of trajectory groups."""

    names = sorted(str(name) for name in mapping.keys())
    if max_trajectories is None:
        selected = names
    else:
        if max_trajectories < 0:
            raise ValueError("max_trajectories must be nonnegative or None")
        selected = names[:max_trajectories]

    trajectories = [
        inspect_cpg_trajectory(
            mapping[name],
            name=name,
            sample_values=sample_values,
            max_values_per_array=max_values_per_array,
        )
        for name in selected
    ]

    warnings: list[str] = []
    if not names:
        warnings.append("file contains no trajectory groups")
    if max_trajectories is not None and len(names) > len(selected):
        warnings.append(f"inspected first {len(selected)} of {len(names)} trajectories")
    for trajectory in trajectories:
        warnings.extend(f"{trajectory.name}: {item}" for item in trajectory.warnings)

    return EulerDatasetSummary(
        path=path,
        num_trajectories=len(names),
        inspected_trajectories=len(trajectories),
        trajectory_names=names,
        trajectories=trajectories,
        warnings=warnings,
    )


def inspect_cpg_trajectory(
    group: GroupLike,
    *,
    name: str = "",
    sample_values: bool = True,
    max_values_per_array: int = 500_000,
) -> TrajectorySummary:
    """Inspect one CPG-style trajectory group."""

    keys = sorted(str(key) for key in group.keys())
    missing = [key for key in CPG_REQUIRED_KEYS if key not in group]
    arrays = {
        key: summarize_array(
            group[key],
            sample_values=sample_values,
            max_values=max_values_per_array,
        )
        for key in keys
    }

    num_time_steps = _infer_time_steps(arrays)
    num_nodes = _infer_num_nodes(arrays)
    num_edges = _infer_num_edges(arrays)
    node_counts = (
        _node_type_counts(group.get("node_type")) if "node_type" in group else {}
    )
    warnings = _trajectory_warnings(missing, arrays, node_counts)

    return TrajectorySummary(
        name=name,
        keys=keys,
        missing_keys=missing,
        arrays=arrays,
        num_time_steps=num_time_steps,
        num_nodes=num_nodes,
        num_edges=num_edges,
        node_type_counts=node_counts,
        warnings=warnings,
    )


def summarize_array(
    value: ArrayLike,
    *,
    sample_values: bool = True,
    max_values: int = 500_000,
) -> ArraySummary:
    """Return shape, dtype, and optional numeric range information."""

    shape = _array_shape(value)
    dtype = _array_dtype(value)
    numeric = np.issubdtype(dtype, np.number)
    if not sample_values or not numeric:
        return ArraySummary(
            shape=list(shape),
            dtype=str(dtype),
            numeric=bool(numeric),
            sample_limited=False,
        )

    sample, limited = _sample_array(value, shape, max_values=max_values)
    if sample.size == 0:
        return ArraySummary(
            shape=list(shape),
            dtype=str(dtype),
            numeric=True,
            sample_limited=limited,
            finite_fraction=1.0,
            num_nan=0,
            num_inf=0,
        )

    sample = np.asarray(sample, dtype=np.float64)
    finite = np.isfinite(sample)
    finite_values = sample[finite]
    return ArraySummary(
        shape=list(shape),
        dtype=str(dtype),
        numeric=True,
        sample_limited=limited,
        finite_fraction=float(np.mean(finite)),
        min=float(np.min(finite_values)) if finite_values.size else None,
        max=float(np.max(finite_values)) if finite_values.size else None,
        mean=float(np.mean(finite_values)) if finite_values.size else None,
        num_nan=int(np.isnan(sample).sum()),
        num_inf=int(np.isinf(sample).sum()),
    )


def _as_time_node_scalar(value: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 2:
        array = array[..., None]
    if array.ndim != 3 or array.shape[-1] != 1:
        raise ValueError(f"{name} must have shape (T, N) or (T, N, 1)")
    return array


def _as_node_column(value: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[-1] != 1:
        raise ValueError(f"{name} must have shape (N,) or (N, 1)")
    return array


def _frame_indexed_array(
    value: ArrayLike,
    frame: int,
    name: str,
    *,
    time_steps: int,
) -> np.ndarray:
    shape = _array_shape(value)
    is_temporal_matrix = name in {"node_type", "Mach"} and len(shape) == 2
    if shape and shape[0] == time_steps and (len(shape) >= 3 or is_temporal_matrix):
        array = np.asarray(value[frame])
    else:
        array = np.asarray(value)
    if array.size == 0:
        raise ValueError(f"{name} frame is empty")
    return array


def _array_shape(value: ArrayLike) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(int(dim) for dim in shape)
    return tuple(int(dim) for dim in np.asarray(value).shape)


def _array_dtype(value: ArrayLike) -> np.dtype:
    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        return np.dtype(dtype)
    return np.asarray(value).dtype


def _sample_array(
    value: ArrayLike,
    shape: tuple[int, ...],
    *,
    max_values: int,
) -> tuple[np.ndarray, bool]:
    if max_values <= 0:
        raise ValueError("max_values must be positive")
    total = int(np.prod(shape, dtype=np.int64)) if shape else 1
    if total <= max_values:
        return np.asarray(value), False
    if not shape:
        return np.asarray(value), False

    trailing = int(np.prod(shape[1:], dtype=np.int64)) if len(shape) > 1 else 1
    leading = max(1, min(shape[0], max_values // max(trailing, 1)))
    try:
        sample = np.asarray(value[:leading])
    except (TypeError, ValueError, IndexError):
        sample = np.asarray(value)
    if sample.size > max_values:
        sample = sample.reshape(-1)[:max_values]
    return sample, True


def _infer_time_steps(arrays: dict[str, ArraySummary]) -> int | None:
    candidates = [
        arrays[key].shape[0]
        for key in PRIMITIVE_KEYS
        if key in arrays and len(arrays[key].shape) >= 2
    ]
    if not candidates:
        return None
    return int(candidates[0]) if len(set(candidates)) == 1 else None


def _infer_num_nodes(arrays: dict[str, ArraySummary]) -> int | None:
    candidates: list[int] = []
    for key in PRIMITIVE_KEYS:
        if key in arrays and len(arrays[key].shape) >= 2:
            candidates.append(arrays[key].shape[1])
    for key in ("node_type", "Mach", "pos"):
        if key in arrays:
            shape = arrays[key].shape
            if len(shape) >= 2:
                candidates.append(shape[-2])
    if not candidates:
        return None
    counts = {value: candidates.count(value) for value in set(candidates)}
    return int(max(counts, key=counts.get))


def _infer_num_edges(arrays: dict[str, ArraySummary]) -> int | None:
    if "edges" not in arrays:
        return None
    shape = arrays["edges"].shape
    if len(shape) < 2:
        return None
    return int(shape[-2])


def _node_type_counts(value: ArrayLike | None) -> dict[int, int]:
    if value is None:
        return {}
    array = np.asarray(value)
    if array.ndim >= 3:
        array = array[0]
    elif array.ndim == 2 and array.shape[-1] != 1:
        array = array[0]
    if array.ndim > 1 and array.shape[-1] == 1:
        array = np.squeeze(array, axis=-1)
    unique, counts = np.unique(array.astype(np.int64), return_counts=True)
    return {int(key): int(count) for key, count in zip(unique, counts, strict=True)}


def _trajectory_warnings(
    missing: list[str],
    arrays: dict[str, ArraySummary],
    node_type_counts_: dict[int, int],
) -> list[str]:
    warnings: list[str] = []
    if missing:
        warnings.append(f"missing required keys: {missing}")

    primitive_shapes = [
        tuple(arrays[key].shape) for key in PRIMITIVE_KEYS if key in arrays
    ]
    if primitive_shapes and len(set(primitive_shapes)) != 1:
        warnings.append(f"primitive shape mismatch: {primitive_shapes}")

    for key in ("rho", "pres"):
        summary = arrays.get(key)
        if summary is not None and summary.min is not None and summary.min <= 0.0:
            warnings.append(f"{key} has nonpositive sampled values")

    edges = arrays.get("edges")
    if edges is not None and (len(edges.shape) < 2 or edges.shape[-1] != 2):
        warnings.append("edges should have last dimension 2")

    pos = arrays.get("pos")
    if pos is not None and (len(pos.shape) < 2 or pos.shape[-1] not in (2, 3)):
        warnings.append("pos should end with coordinate dimension 2 or 3")

    unknown_node_types = [key for key in node_type_counts_ if key not in {0, 1, 2, 3}]
    if unknown_node_types:
        warnings.append(f"unknown node_type codes: {unknown_node_types}")
    if node_type_counts_ and int(EulerNodeType.NORMAL) not in node_type_counts_:
        warnings.append("node_type has no normal nodes in sampled frame")
    return warnings

