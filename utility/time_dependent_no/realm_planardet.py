"""Pinned open-data contracts for the REALM PlanarDet benchmark.

The released test trajectory is deliberately named only as sealed metadata.  All
filesystem and manifest helpers in this module accept train/validation objects
only and reject any path component named ``test``.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch

from utility.time_dependent_no.realm_benchmark import (
    ChannelGroups,
    ManifestEntry,
    RealmNormalizer,
    decoded_spatial_pearson,
    grouped_normalized_prediction_error,
    validate_open_manifest_contract,
)

SCHEMA = "w26_l4_planardet_open_contract_v1"

PLANARDET_REPOSITORY = "realm-bench/realm-bench-PlanarDet"
PLANARDET_REVISION = "b084b6fc2e624e4ee5e44b88dd87d628bcbb9a4b"
PLANARDET_OPEN_MANIFEST_SHA256 = (
    "b53f8195f931ac46b112a76adc9c303b0c8b312cd8b054fb3fb0dbaaebbcc31e"
)
PLANARDET_OPEN_BYTES = 2_940_759_467

PLANARDET_FIELDS = (
    "H",
    "H2",
    "H2O",
    "H2O2",
    "HO2",
    "O",
    "O2",
    "OH",
    "T",
    "rho",
    "Ux",
    "Uy",
    "pMax",
)
PLANARDET_GROUPS = ChannelGroups(
    chem=8,
    temperature=1,
    density=1,
    velocity=2,
    pressure=1,
)

PLANARDET_TRAIN_GROUPS = (
    "sampling_phi8e-1-300",
    "sampling_phi1-330",
    "sampling_phi1-300",
    "sampling_phi12e-1-300",
    "sampling_phi8e-1-330",
    "sampling_phi8e-1-290",
    "sampling_phi12e-1-290",
)
PLANARDET_VAL_GROUPS = ("sampling_phi1-290",)
PLANARDET_TEST_GROUPS_METADATA_ONLY = ("sampling_phi12e-1-330",)

PLANARDET_METADATA_PATHS = frozenset(
    {".gitattributes", "data/data.npz", "data/planardet_stats.yaml"}
)

METADATA_KEYS = frozenset(
    {
        "coords",
        "times",
        "variables",
        "train_groups",
        "val_groups",
        "test_groups",
        "spatial_size",
        "num_chemical",
        "num_temperature",
        "num_density",
        "num_velocity",
        "num_pressure",
    }
)
TRAJECTORY_KEY = "data"
RELEASED_SPATIAL_SHAPE_XY = (832, 384)
CANONICAL_SPATIAL_SHAPE_YX = (384, 832)
TRAJECTORY_RELEASED_SHAPE = (50, 13, *RELEASED_SPATIAL_SHAPE_XY)
TRAJECTORY_CANONICAL_SHAPE = (50, 13, *CANONICAL_SPATIAL_SHAPE_YX)
TRAJECTORY_DTYPE = np.dtype("float32")
RELEASED_COORDINATE_ORDER = ("x", "y")
CANONICAL_COORDINATE_ORDER = ("y", "x")
DOMAIN_LENGTHS_XY = (0.0208, 0.0096)
CELL_SIZE_NATIVE = 2.5e-5
TIME_FIRST = 6.0e-5
TIME_CADENCE = 2.0e-7

PRIMARY_BOX_COX_EPSILON = 1.0e-8
SOURCE_SENSITIVITY_BOX_COX_EPSILON = 1.0e-40
BOX_COX_LAMBDA = 0.1
STD_CORRECTION = 1
SCALE_STABILIZER = 1.0e-10
COORDINATE_UNIFORMITY_RTOL = 1.0e-3
FLOAT64_REENCODE_TOLERANCE = 1.0e-10
FLOAT32_REENCODE_TOLERANCE = 1.0e-4
SELF_CORRELATION_TOLERANCE = 1.0e-6
BOUNDEDNESS_EXPANSION_FACTOR = 10.0


def _tuple_of_strings(value: np.ndarray, name: str) -> tuple[str, ...]:
    if value.ndim != 1 or value.dtype.kind not in {"U", "S"}:
        raise ValueError(f"metadata {name} must be a one-dimensional string array")
    result = tuple(str(item) for item in value.tolist())
    if len(result) != len(set(result)):
        raise ValueError(f"metadata {name} entries must be unique")
    return result


@dataclass(frozen=True)
class PlanarDetMetadata:
    released_coords_xy: np.ndarray
    canonical_coords_yx: np.ndarray
    x: np.ndarray
    y: np.ndarray
    dx: float
    dy: float
    times: tuple[float, ...]
    train_groups: tuple[str, ...]
    val_groups: tuple[str, ...]
    test_groups: tuple[str, ...]

    def summary(self) -> dict[str, Any]:
        return {
            "released_spatial_shape_xy": list(RELEASED_SPATIAL_SHAPE_XY),
            "canonical_model_shape_yx": list(CANONICAL_SPATIAL_SHAPE_YX),
            "released_coordinate_order": list(RELEASED_COORDINATE_ORDER),
            "canonical_coordinate_order": list(CANONICAL_COORDINATE_ORDER),
            "canonicalization": "transpose_spatial_axes_and_swap_coordinate_channels",
            "coordinate_dtype": str(self.released_coords_xy.dtype),
            "x_descending": bool(np.all(np.diff(self.x) < 0.0)),
            "y_descending": bool(np.all(np.diff(self.y) < 0.0)),
            "x_center_min": float(self.x.min()),
            "x_center_max": float(self.x.max()),
            "y_center_min": float(self.y.min()),
            "y_center_max": float(self.y.max()),
            "dx_native": self.dx,
            "dy_native": self.dy,
            "domain_lengths_xy": list(DOMAIN_LENGTHS_XY),
            "domain_length_policy": "released_cell_center_span_plus_one_native_cell",
            "time_count": len(self.times),
            "time_first": self.times[0],
            "time_last": self.times[-1],
            "time_cadence": self.times[1] - self.times[0],
            "field_order": list(PLANARDET_FIELDS),
            "train_groups": list(self.train_groups),
            "val_groups": list(self.val_groups),
            "test_group_names_metadata_only": list(self.test_groups),
            "coordinate_units": "m_from_official_case_description",
            "time_units": "s_from_official_case_description",
            "pMax_units": "Pa_from_official_case_description",
            "centering": "cell_centers_inferred_from_half_cell_offsets",
        }


def load_planardet_metadata(path: Path) -> PlanarDetMetadata:
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != METADATA_KEYS:
            raise ValueError("PlanarDet metadata keys do not match the pinned schema")
        counts = (
            int(payload["num_chemical"]),
            int(payload["num_temperature"]),
            int(payload["num_density"]),
            int(payload["num_velocity"]),
            int(payload["num_pressure"]),
        )
        if counts != (8, 1, 1, 2, 1):
            raise ValueError("PlanarDet channel-group counts changed")
        if tuple(int(value) for value in payload["spatial_size"].tolist()) != (
            RELEASED_SPATIAL_SHAPE_XY
        ):
            raise ValueError("PlanarDet spatial size changed")
        variables = tuple(
            value.removesuffix(".npy")
            for value in _tuple_of_strings(payload["variables"], "variables")
        )
        if variables != PLANARDET_FIELDS:
            raise ValueError("PlanarDet field order changed")
        train_groups = _tuple_of_strings(payload["train_groups"], "train_groups")
        val_groups = _tuple_of_strings(payload["val_groups"], "val_groups")
        test_groups = _tuple_of_strings(payload["test_groups"], "test_groups")
        if train_groups != PLANARDET_TRAIN_GROUPS or val_groups != PLANARDET_VAL_GROUPS:
            raise ValueError("PlanarDet open split membership or order changed")
        if test_groups != PLANARDET_TEST_GROUPS_METADATA_ONLY:
            raise ValueError("PlanarDet sealed split metadata changed")
        if set(train_groups) & set(val_groups) or (
            set(train_groups) | set(val_groups)
        ) & set(test_groups):
            raise ValueError("PlanarDet split groups overlap")
        raw_times = _tuple_of_strings(payload["times"], "times")
        times = tuple(float(value) for value in raw_times)
        expected_times = TIME_FIRST + np.arange(50, dtype=np.float64) * TIME_CADENCE
        if not np.allclose(times, expected_times, rtol=0.0, atol=1.0e-15):
            raise ValueError("PlanarDet time cadence changed")
        coords = np.asarray(payload["coords"])

    if coords.dtype != np.dtype("float32") or coords.shape != (
        2,
        *RELEASED_SPATIAL_SHAPE_XY,
    ):
        raise ValueError("PlanarDet coordinate array contract changed")
    if not np.isfinite(coords).all():
        raise ValueError("PlanarDet coordinates must be finite")
    if not np.all(coords[0] == coords[0, :, :1]):
        raise ValueError("released Cx must vary only along the first spatial axis")
    if not np.all(coords[1] == coords[1, :1, :]):
        raise ValueError("released Cy must vary only along the second spatial axis")
    x = np.asarray(coords[0, :, 0], dtype=np.float64)
    y = np.asarray(coords[1, 0, :], dtype=np.float64)
    difference_x = np.diff(x)
    difference_y = np.diff(y)
    if not np.all(difference_x < 0.0) or not np.all(difference_y < 0.0):
        raise ValueError("pinned PlanarDet coordinate axes must be descending")
    dx = float(abs(np.median(difference_x)))
    dy = float(abs(np.median(difference_y)))
    if not np.allclose(
        abs(difference_x), dx, rtol=COORDINATE_UNIFORMITY_RTOL, atol=1.0e-12
    ) or not np.allclose(
        abs(difference_y), dy, rtol=COORDINATE_UNIFORMITY_RTOL, atol=1.0e-12
    ):
        raise ValueError("PlanarDet coordinate spacing is not uniform")
    if not np.allclose(
        (dx, dy), (CELL_SIZE_NATIVE, CELL_SIZE_NATIVE), rtol=COORDINATE_UNIFORMITY_RTOL
    ):
        raise ValueError("PlanarDet native cell size changed")
    inferred_lengths = (
        float(np.ptp(x) + dx),
        float(np.ptp(y) + dy),
    )
    if not np.allclose(
        inferred_lengths, DOMAIN_LENGTHS_XY, rtol=COORDINATE_UNIFORMITY_RTOL
    ):
        raise ValueError("PlanarDet released-window domain lengths changed")
    canonical = np.stack((coords[1].T, coords[0].T), axis=0)
    return PlanarDetMetadata(
        released_coords_xy=coords.copy(),
        canonical_coords_yx=np.ascontiguousarray(canonical),
        x=x,
        y=y,
        dx=dx,
        dy=dy,
        times=times,
        train_groups=train_groups,
        val_groups=val_groups,
        test_groups=test_groups,
    )


def load_planardet_trajectory(path: Path, *, canonical: bool = True) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        if payload.files != [TRAJECTORY_KEY]:
            raise ValueError(f"trajectory must contain only {TRAJECTORY_KEY!r}")
        state = np.asarray(payload[TRAJECTORY_KEY])
    if state.dtype != TRAJECTORY_DTYPE:
        raise ValueError("PlanarDet trajectory dtype must be float32")
    if state.shape != TRAJECTORY_RELEASED_SHAPE:
        raise ValueError(f"trajectory shape must be {TRAJECTORY_RELEASED_SHAPE}")
    if not np.isfinite(state).all():
        raise ValueError("PlanarDet trajectory contains a native-nonfinite value")
    return state.transpose(0, 1, 3, 2) if canonical else state


def trajectory_diagnostics(state: np.ndarray) -> dict[str, Any]:
    if state.shape not in {TRAJECTORY_RELEASED_SHAPE, TRAJECTORY_CANONICAL_SHAPE}:
        raise ValueError("trajectory diagnostics received the wrong shape")
    minima = state.min(axis=(0, 2, 3)).astype(np.float64)
    maxima = state.max(axis=(0, 2, 3)).astype(np.float64)
    negative_species = (state[:, :8] < 0.0).sum(axis=(0, 2, 3))
    pmax_difference = np.diff(state[:, 12].astype(np.float64), axis=0)
    return {
        "finite": bool(np.isfinite(state).all()),
        "admissible_released_state": bool(
            np.all(negative_species == 0)
            and np.all(state[:, 8] > 0.0)
            and np.all(state[:, 9] > 0.0)
            and np.all(state[:, 12] > 0.0)
        ),
        "min_by_channel": minima.tolist(),
        "max_by_channel": maxima.tolist(),
        "negative_species_count_by_channel": negative_species.astype(int).tolist(),
        "nonpositive_temperature_count": int(np.count_nonzero(state[:, 8] <= 0.0)),
        "nonpositive_density_count": int(np.count_nonzero(state[:, 9] <= 0.0)),
        "nonpositive_pMax_count": int(np.count_nonzero(state[:, 12] <= 0.0)),
        "pMax_decrease_count": int(np.count_nonzero(pmax_difference < 0.0)),
        "pMax_increase_count": int(np.count_nonzero(pmax_difference > 0.0)),
        "pMax_equal_count": int(np.count_nonzero(pmax_difference == 0.0)),
        "pMax_min_increment": float(pmax_difference.min()),
        "pMax_max_increment": float(pmax_difference.max()),
        "released_species_sum_min": float(state[:, :8].sum(axis=1).min()),
        "released_species_sum_max": float(state[:, :8].sum(axis=1).max()),
    }


def validate_planardet_open_manifest(
    repository: str,
    revision: str,
    entries: Sequence[ManifestEntry],
) -> dict[str, Any]:
    """Validate the exact pinned train/validation-only release manifest."""

    result = validate_open_manifest_contract(
        repository=repository,
        revision=revision,
        entries=entries,
        expected_repository=PLANARDET_REPOSITORY,
        expected_revision=PLANARDET_REVISION,
        expected_sha256=PLANARDET_OPEN_MANIFEST_SHA256,
        expected_bytes=PLANARDET_OPEN_BYTES,
    )
    expected_trajectories = {
        trajectory_relative_path(split, group)
        for split, groups in (
            ("train", PLANARDET_TRAIN_GROUPS),
            ("val", PLANARDET_VAL_GROUPS),
        )
        for group in groups
    }
    actual_trajectories = {
        entry.path
        for entry in entries
        if entry.path.startswith("data/train/") or entry.path.startswith("data/val/")
    }
    if actual_trajectories != expected_trajectories:
        raise ValueError("PlanarDet open trajectory membership changed")
    if not PLANARDET_METADATA_PATHS.issubset({entry.path for entry in entries}):
        raise ValueError("PlanarDet metadata objects are missing")
    return result


def metadata_manifest_entries(
    entries: Sequence[ManifestEntry],
) -> tuple[ManifestEntry, ...]:
    """Select the three pinned non-trajectory objects for metadata-first audit."""

    selected = tuple(
        entry for entry in entries if entry.path in PLANARDET_METADATA_PATHS
    )
    if {entry.path for entry in selected} != PLANARDET_METADATA_PATHS:
        raise ValueError("PlanarDet metadata subset is incomplete")
    return selected


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_oid(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {size}\0".encode())
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest_file(root: Path, entry: ManifestEntry) -> dict[str, Any]:
    path = root.joinpath(*PurePosixPath(entry.path).parts)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or non-regular manifest file: {entry.path}")
    size = path.stat().st_size
    if size != entry.size:
        raise ValueError(f"manifest file size mismatch: {entry.path}")
    object_oid = sha256_file(path) if entry.lfs else git_blob_oid(path)
    if object_oid != entry.oid:
        raise ValueError(f"manifest object mismatch: {entry.path}")
    return {
        "path": entry.path,
        "size": size,
        "oid": entry.oid,
        "lfs": entry.lfs,
        "content_sha256": object_oid if entry.lfs else sha256_file(path),
    }


def validate_local_open_tree(
    root: Path,
    entries: Sequence[ManifestEntry],
) -> list[dict[str, Any]]:
    """Require exactly ``entries`` and reject sealed paths and symlinks."""

    if root.is_symlink() or not root.is_dir():
        raise ValueError("data root must be an existing non-symlink directory")
    expected = {entry.path for entry in entries}
    if len(expected) != len(entries):
        raise ValueError("manifest paths must be unique")
    expected_directories = {
        PurePosixPath(*path.parts[:index]).as_posix()
        for entry in entries
        for path in (PurePosixPath(entry.path),)
        for index in range(1, len(path.parts))
    }
    actual: set[str] = set()
    actual_directories: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(part.casefold() == "test" for part in relative.parts):
            raise ValueError(f"sealed test path exists under data root: {relative}")
        if path.is_symlink():
            raise ValueError(f"symlinks are forbidden under data root: {relative}")
        canonical = PurePosixPath(*relative.parts).as_posix()
        if path.is_file():
            actual.add(canonical)
        elif path.is_dir():
            actual_directories.add(canonical)
        else:
            raise ValueError(f"unsupported object under data root: {relative}")
    if actual_directories != expected_directories:
        missing = sorted(expected_directories - actual_directories)
        unexpected = sorted(actual_directories - expected_directories)
        raise ValueError(
            "local open directory tree mismatch; "
            f"missing={missing}, unexpected={unexpected}"
        )
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"local open tree mismatch; missing={missing}, unexpected={unexpected}"
        )
    return [verify_manifest_file(root, entry) for entry in sorted(entries)]


def trajectory_relative_path(split: str, group: str) -> str:
    if split not in {"train", "val"}:
        raise ValueError("trajectory split must be train or val")
    allowed = PLANARDET_TRAIN_GROUPS if split == "train" else PLANARDET_VAL_GROUPS
    if group not in allowed:
        raise ValueError(f"unregistered {split} PlanarDet group: {group}")
    if "/" in group or "\\" in group or group in {"", ".", ".."}:
        raise ValueError("trajectory group is not a canonical file stem")
    return f"data/{split}/{group}.npz"


class StreamingChannelMoments:
    """Stable train-only moments over one released trajectory at a time."""

    def __init__(self, *, box_cox_epsilon: float | None) -> None:
        if box_cox_epsilon is not None and (
            not math.isfinite(box_cox_epsilon) or box_cox_epsilon <= 0.0
        ):
            raise ValueError("box_cox_epsilon must be positive when supplied")
        self.box_cox_epsilon = box_cox_epsilon
        self.count = 0
        self.mean = np.zeros(len(PLANARDET_FIELDS), dtype=np.float64)
        self.m2 = np.zeros(len(PLANARDET_FIELDS), dtype=np.float64)
        self.minimum = np.full(len(PLANARDET_FIELDS), np.inf, dtype=np.float64)
        self.maximum = np.full(len(PLANARDET_FIELDS), -np.inf, dtype=np.float64)

    def update(self, state: np.ndarray) -> None:
        if state.shape not in {TRAJECTORY_RELEASED_SHAPE, TRAJECTORY_CANONICAL_SHAPE}:
            raise ValueError("streaming moments require a pinned trajectory tensor")
        if state.dtype != TRAJECTORY_DTYPE:
            raise ValueError("streaming moments require float32 release values")
        batch_count = state.shape[0] * state.shape[2] * state.shape[3]
        batch_mean = np.empty(len(PLANARDET_FIELDS), dtype=np.float64)
        batch_m2 = np.empty(len(PLANARDET_FIELDS), dtype=np.float64)
        batch_min = np.empty(len(PLANARDET_FIELDS), dtype=np.float64)
        batch_max = np.empty(len(PLANARDET_FIELDS), dtype=np.float64)
        for channel in range(len(PLANARDET_FIELDS)):
            values = np.asarray(state[:, channel], dtype=np.float64)
            if self.box_cox_epsilon is not None and channel < 8:
                values = (
                    np.power(np.maximum(values, self.box_cox_epsilon), BOX_COX_LAMBDA)
                    - 1.0
                ) / BOX_COX_LAMBDA
            batch_mean[channel] = values.mean()
            centered = values - batch_mean[channel]
            batch_m2[channel] = np.square(centered).sum()
            batch_min[channel] = values.min()
            batch_max[channel] = values.max()
        if self.count == 0:
            self.mean[:] = batch_mean
            self.m2[:] = batch_m2
        else:
            delta = batch_mean - self.mean
            combined = self.count + batch_count
            self.mean += delta * (batch_count / combined)
            self.m2 += batch_m2 + np.square(delta) * (
                self.count * batch_count / combined
            )
        self.count += batch_count
        self.minimum = np.minimum(self.minimum, batch_min)
        self.maximum = np.maximum(self.maximum, batch_max)

    def finalize(self) -> dict[str, Any]:
        if self.count <= STD_CORRECTION:
            raise ValueError("insufficient samples for train statistics")
        variance = np.maximum(self.m2 / (self.count - STD_CORRECTION), 0.0)
        std = np.sqrt(variance)
        scale = np.where(
            std < SCALE_STABILIZER,
            np.ones_like(std),
            std + SCALE_STABILIZER,
        )
        return {
            "sample_count_per_channel": self.count,
            "mean": self.mean.copy(),
            "std": std,
            "scale": scale,
            "minimum": self.minimum.copy(),
            "maximum": self.maximum.copy(),
            "box_cox_epsilon": self.box_cox_epsilon,
            "box_cox_lambda": (
                BOX_COX_LAMBDA if self.box_cox_epsilon is not None else None
            ),
            "std_correction": STD_CORRECTION,
            "scale_stabilizer": SCALE_STABILIZER,
        }


def planardet_normalizer(
    mean: np.ndarray | torch.Tensor,
    scale: np.ndarray | torch.Tensor,
    *,
    channel_axis: int,
    dtype: torch.dtype,
) -> RealmNormalizer:
    return RealmNormalizer(
        mean=torch.as_tensor(mean, dtype=dtype),
        scale=torch.as_tensor(scale, dtype=dtype),
        transformed_channels=tuple(range(8)),
        channel_axis=channel_axis,
        box_cox_lambda=BOX_COX_LAMBDA,
        box_cox_epsilon=PRIMARY_BOX_COX_EPSILON,
        std_correction=STD_CORRECTION,
        scale_stabilizer=SCALE_STABILIZER,
    )


def _json_statistics(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in stats.items()
    }


def analyze_planardet_open_dataset(
    root: Path,
    entries: Sequence[ManifestEntry],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Audit all open arrays and fit the launch normalizer from train only."""

    inventory = validate_local_open_tree(root, entries)
    metadata = load_planardet_metadata(root / "data" / "data.npz")
    expected_trajectories = {
        trajectory_relative_path(split, group)
        for split, groups in (
            ("train", PLANARDET_TRAIN_GROUPS),
            ("val", PLANARDET_VAL_GROUPS),
        )
        for group in groups
    }
    manifest_trajectories = {
        entry.path
        for entry in entries
        if entry.path.startswith("data/train/") or entry.path.startswith("data/val/")
    }
    if manifest_trajectories != expected_trajectories:
        raise ValueError("manifest trajectory set disagrees with pinned metadata")

    primary_moments = StreamingChannelMoments(box_cox_epsilon=PRIMARY_BOX_COX_EPSILON)
    sensitivity_moments = StreamingChannelMoments(
        box_cox_epsilon=SOURCE_SENSITIVITY_BOX_COX_EPSILON
    )
    raw_moments = StreamingChannelMoments(box_cox_epsilon=None)
    train_max_abs = np.zeros(len(PLANARDET_FIELDS), dtype=np.float64)
    clamp_counts = np.zeros(8, dtype=np.int64)
    trajectory_rows: list[dict[str, Any]] = []

    for split, groups in (
        ("train", PLANARDET_TRAIN_GROUPS),
        ("val", PLANARDET_VAL_GROUPS),
    ):
        for group in groups:
            relative = trajectory_relative_path(split, group)
            state = load_planardet_trajectory(
                root.joinpath(*PurePosixPath(relative).parts), canonical=False
            )
            trajectory_rows.append(
                {
                    "split": split,
                    "group": group,
                    "relative_path": relative,
                    **trajectory_diagnostics(state),
                }
            )
            if split == "train":
                primary_moments.update(state)
                sensitivity_moments.update(state)
                raw_moments.update(state)
                train_max_abs = np.maximum(
                    train_max_abs,
                    np.abs(state).max(axis=(0, 2, 3)).astype(np.float64),
                )
                clamp_counts += np.count_nonzero(
                    state[:, :8] < PRIMARY_BOX_COX_EPSILON,
                    axis=(0, 2, 3),
                )

    primary = primary_moments.finalize()
    sensitivity = sensitivity_moments.finalize()
    raw = raw_moments.finalize()
    normalizers = {
        "float64": planardet_normalizer(
            primary["mean"], primary["scale"], channel_axis=2, dtype=torch.float64
        ),
        "float32": planardet_normalizer(
            primary["mean"], primary["scale"], channel_axis=2, dtype=torch.float32
        ),
    }
    roundtrip_max = {"float64": 0.0, "float32": 0.0}
    identity_npe_mean = 0.0
    identity_npe_sum = 0.0
    minimum_self_correlation = 1.0
    observable_correlation_count = 0
    pearson_status_counts: Counter[str] = Counter()
    maximum_primary_decode_abs_error = np.zeros(len(PLANARDET_FIELDS), dtype=np.float64)

    for split, groups in (
        ("train", PLANARDET_TRAIN_GROUPS),
        ("val", PLANARDET_VAL_GROUPS),
    ):
        for group in groups:
            relative = trajectory_relative_path(split, group)
            state = load_planardet_trajectory(
                root.joinpath(*PurePosixPath(relative).parts), canonical=False
            )
            for frame in range(state.shape[0]):
                for dtype_name, normalizer in normalizers.items():
                    dtype = torch.float64 if dtype_name == "float64" else torch.float32
                    native = (
                        torch.from_numpy(state[frame : frame + 1])
                        .to(dtype=dtype)
                        .unsqueeze(0)
                    )
                    encoded = normalizer.encode(native)
                    decoded = normalizer.decode(encoded)
                    reencoded = normalizer.encode(decoded)
                    if not bool(torch.isfinite(decoded).all()):
                        raise ValueError(
                            "normalizer decode produced a nonfinite reference"
                        )
                    error = float((reencoded - encoded).abs().max().item())
                    roundtrip_max[dtype_name] = max(roundtrip_max[dtype_name], error)
                    if dtype_name == "float64":
                        decode_error = (
                            (decoded - native)
                            .abs()
                            .amax(dim=(0, 1, 3, 4))
                            .cpu()
                            .numpy()
                        )
                        maximum_primary_decode_abs_error = np.maximum(
                            maximum_primary_decode_abs_error, decode_error
                        )
                    if split == "val" and dtype_name == "float64":
                        identity = grouped_normalized_prediction_error(
                            encoded,
                            encoded,
                            groups=PLANARDET_GROUPS,
                        )
                        identity_npe_mean = max(
                            identity_npe_mean, identity.realm_npe_mean
                        )
                        identity_npe_sum = max(
                            identity_npe_sum, identity.realm_npe_sum_source
                        )
                        correlation = decoded_spatial_pearson(decoded, decoded)
                        for case in correlation.statuses:
                            for call in case:
                                pearson_status_counts.update(call)
                        observable = correlation.values[
                            torch.isfinite(correlation.values)
                        ]
                        if observable.numel():
                            observable_correlation_count += observable.numel()
                            minimum_self_correlation = min(
                                minimum_self_correlation,
                                float(observable.min().item()),
                            )

    all_admissible = all(
        bool(row["finite"]) and bool(row["admissible_released_state"])
        for row in trajectory_rows
    )
    all_pmax_monotone = all(row["pMax_decrease_count"] == 0 for row in trajectory_rows)
    gates = {
        "exact_file_inventory": True,
        "metadata_schema": True,
        "canonical_coordinate_layout": True,
        "all_open_trajectories_finite_and_admissible": all_admissible,
        "all_open_pMax_cell_histories_nondecreasing": all_pmax_monotone,
        "float64_reencode": roundtrip_max["float64"] <= FLOAT64_REENCODE_TOLERANCE,
        "float32_reencode": roundtrip_max["float32"] <= FLOAT32_REENCODE_TOLERANCE,
        "identity_grouped_mse": identity_npe_mean == 0.0 and identity_npe_sum == 0.0,
        "observable_identity_correlation": minimum_self_correlation
        >= 1.0 - SELF_CORRELATION_TOLERANCE
        and observable_correlation_count > 0,
        "sealed_test_objects_absent": True,
    }
    report = {
        "schema": SCHEMA,
        "scope": "open_train_validation_reference_audit_no_model",
        "metadata": metadata.summary(),
        "inventory": inventory,
        "trajectory_rows": trajectory_rows,
        "train_statistics_primary": _json_statistics(primary),
        "train_statistics_source_sensitivity": _json_statistics(sensitivity),
        "train_statistics_raw": _json_statistics(raw),
        "primary_clamp_counts_by_species": clamp_counts.tolist(),
        "primary_decode_max_abs_error_by_channel": (
            maximum_primary_decode_abs_error.tolist()
        ),
        "boundedness": {
            "expansion_factor": BOUNDEDNESS_EXPANSION_FACTOR,
            "train_max_abs_by_channel": train_max_abs.tolist(),
            "inclusive_limits_by_channel": (
                train_max_abs * BOUNDEDNESS_EXPANSION_FACTOR
            ).tolist(),
        },
        "pMax_operational_semantics": {
            "release_group": "pressure",
            "official_label": "maximum pressure",
            "empirical_open_contract": "cellwise_nondecreasing_history_field",
            "instantaneous_pressure_available": False,
            "conservative_state_claim": False,
        },
        "normalizer_replay": {
            "max_normalized_reencode_error": roundtrip_max,
            "float64_tolerance": FLOAT64_REENCODE_TOLERANCE,
            "float32_tolerance": FLOAT32_REENCODE_TOLERANCE,
        },
        "validation_identity_metric_replay": {
            "realm_npe_mean_max": identity_npe_mean,
            "realm_npe_sum_source_max": identity_npe_sum,
            "minimum_observable_self_correlation": minimum_self_correlation,
            "observable_correlation_count": observable_correlation_count,
            "self_correlation_tolerance": SELF_CORRELATION_TOLERANCE,
            "pearson_status_counts": dict(sorted(pearson_status_counts.items())),
        },
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "anti_claims": [
            "no model or checkpoint was executed",
            "no test trajectory was present or opened",
            "eight released species are not the complete hydrogen-air composition",
            "pMax is a released cumulative diagnostic, not instantaneous pressure",
            "uniform cell weights do not establish conservation without complete fields and boundary fluxes",
        ],
    }
    arrays = {
        "primary_mean": np.asarray(primary["mean"]),
        "primary_std": np.asarray(primary["std"]),
        "primary_scale": np.asarray(primary["scale"]),
        "source_sensitivity_mean": np.asarray(sensitivity["mean"]),
        "source_sensitivity_std": np.asarray(sensitivity["std"]),
        "source_sensitivity_scale": np.asarray(sensitivity["scale"]),
        "raw_train_mean": np.asarray(raw["mean"]),
        "raw_train_std": np.asarray(raw["std"]),
        "train_max_abs": train_max_abs,
        "canonical_coordinates_yx": metadata.canonical_coords_yx,
    }
    return report, arrays


__all__ = [
    "BOUNDEDNESS_EXPANSION_FACTOR",
    "BOX_COX_LAMBDA",
    "CANONICAL_COORDINATE_ORDER",
    "CANONICAL_SPATIAL_SHAPE_YX",
    "CELL_SIZE_NATIVE",
    "DOMAIN_LENGTHS_XY",
    "METADATA_KEYS",
    "PLANARDET_FIELDS",
    "PLANARDET_GROUPS",
    "PLANARDET_METADATA_PATHS",
    "PLANARDET_OPEN_BYTES",
    "PLANARDET_OPEN_MANIFEST_SHA256",
    "PLANARDET_REPOSITORY",
    "PLANARDET_REVISION",
    "PLANARDET_TEST_GROUPS_METADATA_ONLY",
    "PLANARDET_TRAIN_GROUPS",
    "PLANARDET_VAL_GROUPS",
    "PRIMARY_BOX_COX_EPSILON",
    "RELEASED_COORDINATE_ORDER",
    "RELEASED_SPATIAL_SHAPE_XY",
    "SCHEMA",
    "SOURCE_SENSITIVITY_BOX_COX_EPSILON",
    "TIME_CADENCE",
    "TIME_FIRST",
    "TRAJECTORY_CANONICAL_SHAPE",
    "TRAJECTORY_DTYPE",
    "TRAJECTORY_KEY",
    "TRAJECTORY_RELEASED_SHAPE",
    "PlanarDetMetadata",
    "StreamingChannelMoments",
    "analyze_planardet_open_dataset",
    "git_blob_oid",
    "load_planardet_metadata",
    "load_planardet_trajectory",
    "metadata_manifest_entries",
    "planardet_normalizer",
    "sha256_file",
    "trajectory_diagnostics",
    "trajectory_relative_path",
    "validate_local_open_tree",
    "validate_planardet_open_manifest",
    "verify_manifest_file",
]
