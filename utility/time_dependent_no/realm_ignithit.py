"""Pinned IgnitHIT acquisition and real-data replay contracts for D088.

The generic metric and recurrence semantics live in :mod:`realm_benchmark`.
This module owns only the exact released IgnitHIT files, array schema, streaming
train statistics, local-object verification, and pre-model reference replay.
It never opens a test trajectory.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch

from utility.time_dependent_no.realm_benchmark import (
    IGNITHIT_FIELDS,
    IGNITHIT_GROUPS,
    ManifestEntry,
    RealmNormalizer,
    decoded_spatial_pearson,
    grouped_normalized_prediction_error,
)

SCHEMA = "d088_ignithit_p1b_replay_v1"
METADATA_KEYS = frozenset(
    {
        "coords",
        "num_chemical",
        "num_density",
        "num_pressure",
        "num_temperature",
        "num_velocity",
        "spatial_size",
        "test_groups",
        "times",
        "train_groups",
        "val_groups",
        "variables",
    }
)
TRAJECTORY_KEY = "data"
TRAJECTORY_SHAPE = (30, 12, 128, 128)
TRAJECTORY_DTYPE = np.dtype("float32")

TRAIN_GROUPS = (
    "phi=s_15_4_s",
    "phi=c_5_1_c",
    "phi=c_15_1_c",
    "phi=s_15_3_s",
    "phi=c_5_3_c",
    "phi=c_10_4_c",
    "phi=_t_5_3_t",
    "phi=_t_10_3_t",
    "phi=_t_5_4_t",
    "phi=s_15_2_s",
    "phi=s_10_4_s",
    "phi=s_5_4_s",
    "phi=s_5_2_s",
    "phi=s_15_1_s",
    "phi=s_5_1_s",
    "phi=_t_5_1_t",
    "phi=c_10_2_c",
    "phi=_t_15_2_t",
    "phi=c_15_2_c",
    "phi=c_10_3_c",
    "phi=_t_15_4_t",
    "phi=s_10_3_s",
    "phi=_t_10_2_t",
    "phi=c_10_1_c",
    "phi=s_10_2_s",
    "phi=s_10_1_s",
)
VAL_GROUPS = (
    "phi=_t_15_3_t",
    "phi=c_5_4_c",
    "phi=_t_15_1_t",
    "phi=c_15_3_c",
    "phi=c_5_2_c",
)
TEST_GROUPS = (
    "phi=c_15_4_c",
    "phi=_t_5_2_t",
    "phi=s_5_3_s",
    "phi=_t_10_4_t",
    "phi=_t_10_1_t",
)

PRIMARY_BOX_COX_EPSILON = 1.0e-8
SOURCE_SENSITIVITY_BOX_COX_EPSILON = 1.0e-40
BOX_COX_LAMBDA = 0.1
SCALE_STABILIZER = 1.0e-10
STD_CORRECTION = 1
BOUNDEDNESS_QUANTILE = 1.0
BOUNDEDNESS_EXPANSION_FACTOR = 10.0
FRONT_LOW_QUANTILE = 0.10
FRONT_HIGH_QUANTILE = 0.90
BOUNDARY_BAND_CELLS = 4
FLOAT64_REENCODE_TOLERANCE = 1.0e-10
FLOAT32_REENCODE_TOLERANCE = 1.0e-4
SELF_CORRELATION_TOLERANCE = 1.0e-6
COORDINATE_UNIFORMITY_RTOL = 3.0e-4


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
    """Require exactly the registered open files and reject every test path."""

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
        parts = relative.parts
        if any(part.casefold() == "test" for part in parts):
            raise ValueError(f"sealed test path exists under data root: {relative}")
        if path.is_symlink():
            raise ValueError(f"symlinks are forbidden under data root: {relative}")
        if path.is_file():
            actual.add(PurePosixPath(*parts).as_posix())
        elif path.is_dir():
            actual_directories.add(PurePosixPath(*parts).as_posix())
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
    allowed = TRAIN_GROUPS if split == "train" else VAL_GROUPS
    if group not in allowed:
        raise ValueError(f"unregistered {split} trajectory group: {group}")
    if "/" in group or "\\" in group or group in {"", ".", ".."}:
        raise ValueError("trajectory group is not a canonical file stem")
    return f"data/{split}/{group}.npz"


@dataclass(frozen=True)
class IgnitHITMetadata:
    coords: np.ndarray
    x: np.ndarray
    y: np.ndarray
    dx: float
    dy: float
    times: tuple[float, ...]
    train_groups: tuple[str, ...]
    val_groups: tuple[str, ...]
    test_groups: tuple[str, ...]

    def summary(self) -> dict[str, Any]:
        radial_max = math.hypot(1.0 / (2.0 * self.dx), 1.0 / (2.0 * self.dy))
        axis_nyquist = min(1.0 / (2.0 * self.dx), 1.0 / (2.0 * self.dy))
        spectrum_edges = (
            0.0,
            axis_nyquist / 8.0,
            axis_nyquist / 4.0,
            axis_nyquist / 2.0,
            axis_nyquist,
            radial_max,
        )
        return {
            "spatial_shape": [128, 128],
            "coordinate_array_shape": list(self.coords.shape),
            "coordinate_dtype": str(self.coords.dtype),
            "coordinate_order": ["y", "x"],
            "x_descending": bool(np.all(np.diff(self.x) < 0.0)),
            "y_descending": bool(np.all(np.diff(self.y) < 0.0)),
            "x_center_min": float(self.x.min()),
            "x_center_max": float(self.x.max()),
            "y_center_min": float(self.y.min()),
            "y_center_max": float(self.y.max()),
            "dx_native": self.dx,
            "dy_native": self.dy,
            "boundary_band_width_native": BOUNDARY_BAND_CELLS * max(self.dx, self.dy),
            "spectrum_band_edges_cycles_per_native_unit": list(spectrum_edges),
            "time_count": len(self.times),
            "time_first": self.times[0],
            "time_last": self.times[-1],
            "time_cadence_native": self.times[1] - self.times[0],
            "field_order": list(IGNITHIT_FIELDS),
            "train_groups": list(self.train_groups),
            "val_groups": list(self.val_groups),
            "test_group_names_metadata_only": list(self.test_groups),
            "coordinate_units": None,
            "time_units": None,
            "centering": None,
        }


def _tuple_of_strings(value: np.ndarray, name: str) -> tuple[str, ...]:
    if value.ndim != 1 or value.dtype.kind not in {"U", "S"}:
        raise ValueError(f"metadata {name} must be a one-dimensional string array")
    result = tuple(str(item) for item in value.tolist())
    if len(result) != len(set(result)):
        raise ValueError(f"metadata {name} entries must be unique")
    return result


def load_ignithit_metadata(path: Path) -> IgnitHITMetadata:
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != METADATA_KEYS:
            raise ValueError("IgnitHIT metadata keys do not match the pinned schema")
        counts = (
            int(payload["num_chemical"]),
            int(payload["num_temperature"]),
            int(payload["num_density"]),
            int(payload["num_velocity"]),
            int(payload["num_pressure"]),
        )
        if counts != (8, 1, 1, 2, 0):
            raise ValueError("IgnitHIT channel-group counts changed")
        if tuple(int(value) for value in payload["spatial_size"].tolist()) != (
            128,
            128,
        ):
            raise ValueError("IgnitHIT spatial size changed")
        variables = tuple(
            value.removesuffix(".npy")
            for value in _tuple_of_strings(payload["variables"], "variables")
        )
        if variables != IGNITHIT_FIELDS:
            raise ValueError("IgnitHIT field order changed")
        train_groups = _tuple_of_strings(payload["train_groups"], "train_groups")
        val_groups = _tuple_of_strings(payload["val_groups"], "val_groups")
        test_groups = _tuple_of_strings(payload["test_groups"], "test_groups")
        if train_groups != TRAIN_GROUPS or val_groups != VAL_GROUPS:
            raise ValueError("IgnitHIT open split membership or order changed")
        if test_groups != TEST_GROUPS:
            raise ValueError("IgnitHIT sealed split metadata changed")
        if set(train_groups) & set(val_groups) or (
            set(train_groups) | set(val_groups)
        ) & set(test_groups):
            raise ValueError("IgnitHIT split groups overlap")
        raw_times = _tuple_of_strings(payload["times"], "times")
        times = tuple(float(value) for value in raw_times)
        expected_times = np.arange(1, 31, dtype=np.float64) * 1.0e-5
        if not np.allclose(times, expected_times, rtol=0.0, atol=1.0e-15):
            raise ValueError("IgnitHIT time cadence changed")
        coords = np.asarray(payload["coords"])

    if coords.dtype != np.dtype("float32") or coords.shape != (2, 128, 128):
        raise ValueError("IgnitHIT coordinate array contract changed")
    if not np.isfinite(coords).all():
        raise ValueError("IgnitHIT coordinates must be finite")
    if not np.all(coords[0] == coords[0, :, :1]):
        raise ValueError("coordinate channel 0 must vary only along y")
    if not np.all(coords[1] == coords[1, :1, :]):
        raise ValueError("coordinate channel 1 must vary only along x")
    y = np.asarray(coords[0, :, 0], dtype=np.float64)
    x = np.asarray(coords[1, 0, :], dtype=np.float64)
    difference_y = np.diff(y)
    difference_x = np.diff(x)
    if not np.all(difference_y < 0.0) or not np.all(difference_x < 0.0):
        raise ValueError("pinned IgnitHIT coordinate axes must be descending")
    dy = float(abs(np.median(difference_y)))
    dx = float(abs(np.median(difference_x)))
    if not np.allclose(
        abs(difference_y), dy, rtol=COORDINATE_UNIFORMITY_RTOL, atol=1.0e-12
    ):
        raise ValueError("IgnitHIT y spacing is not uniform")
    if not np.allclose(
        abs(difference_x), dx, rtol=COORDINATE_UNIFORMITY_RTOL, atol=1.0e-12
    ):
        raise ValueError("IgnitHIT x spacing is not uniform")
    return IgnitHITMetadata(
        coords=coords.copy(),
        x=x,
        y=y,
        dx=dx,
        dy=dy,
        times=times,
        train_groups=train_groups,
        val_groups=val_groups,
        test_groups=test_groups,
    )


def load_ignithit_trajectory(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        if payload.files != [TRAJECTORY_KEY]:
            raise ValueError(f"trajectory must contain only {TRAJECTORY_KEY!r}")
        state = np.asarray(payload[TRAJECTORY_KEY])
    if state.dtype != TRAJECTORY_DTYPE:
        raise ValueError("trajectory dtype must be float32")
    if state.shape != TRAJECTORY_SHAPE:
        raise ValueError(f"trajectory shape must be {TRAJECTORY_SHAPE}")
    if not np.isfinite(state).all():
        raise ValueError("trajectory contains a native-nonfinite value")
    return state


def trajectory_diagnostics(state: np.ndarray) -> dict[str, Any]:
    if state.shape != TRAJECTORY_SHAPE:
        raise ValueError("trajectory diagnostics received the wrong shape")
    minima = state.min(axis=(0, 2, 3)).astype(np.float64)
    maxima = state.max(axis=(0, 2, 3)).astype(np.float64)
    negative_species = (state[:, :8] < 0.0).sum(axis=(0, 2, 3))
    return {
        "finite": bool(np.isfinite(state).all()),
        "admissible_released_state": bool(
            np.all(negative_species == 0)
            and np.all(state[:, 8] > 0.0)
            and np.all(state[:, 9] > 0.0)
        ),
        "min_by_channel": minima.tolist(),
        "max_by_channel": maxima.tolist(),
        "negative_species_count_by_channel": negative_species.astype(int).tolist(),
        "nonpositive_temperature_count": int(np.count_nonzero(state[:, 8] <= 0.0)),
        "nonpositive_density_count": int(np.count_nonzero(state[:, 9] <= 0.0)),
    }


class StreamingChannelMoments:
    """Stable per-channel moments over ``[time, channel, y, x]`` batches."""

    def __init__(self, *, box_cox_epsilon: float | None) -> None:
        if box_cox_epsilon is not None and (
            not math.isfinite(box_cox_epsilon) or box_cox_epsilon <= 0.0
        ):
            raise ValueError("box_cox_epsilon must be positive when supplied")
        self.box_cox_epsilon = box_cox_epsilon
        self.count = 0
        self.mean = np.zeros(len(IGNITHIT_FIELDS), dtype=np.float64)
        self.m2 = np.zeros(len(IGNITHIT_FIELDS), dtype=np.float64)
        self.minimum = np.full(len(IGNITHIT_FIELDS), np.inf, dtype=np.float64)
        self.maximum = np.full(len(IGNITHIT_FIELDS), -np.inf, dtype=np.float64)

    def update(self, state: np.ndarray) -> None:
        if state.shape != TRAJECTORY_SHAPE or state.dtype != TRAJECTORY_DTYPE:
            raise ValueError("streaming moments require a pinned trajectory tensor")
        batch_count = state.shape[0] * state.shape[2] * state.shape[3]
        batch_mean = np.empty(len(IGNITHIT_FIELDS), dtype=np.float64)
        batch_m2 = np.empty(len(IGNITHIT_FIELDS), dtype=np.float64)
        batch_min = np.empty(len(IGNITHIT_FIELDS), dtype=np.float64)
        batch_max = np.empty(len(IGNITHIT_FIELDS), dtype=np.float64)
        for channel in range(len(IGNITHIT_FIELDS)):
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
        variance = self.m2 / (self.count - STD_CORRECTION)
        variance = np.maximum(variance, 0.0)
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
            "box_cox_lambda": BOX_COX_LAMBDA
            if self.box_cox_epsilon is not None
            else None,
            "std_correction": STD_CORRECTION,
            "scale_stabilizer": SCALE_STABILIZER,
        }


def _normalizer(stats: Mapping[str, Any], *, dtype: torch.dtype) -> RealmNormalizer:
    epsilon = stats["box_cox_epsilon"]
    if not isinstance(epsilon, float):
        raise TypeError("transformed statistics require a Box-Cox epsilon")
    return RealmNormalizer(
        mean=torch.as_tensor(stats["mean"], dtype=dtype),
        scale=torch.as_tensor(stats["scale"], dtype=dtype),
        transformed_channels=tuple(range(8)),
        channel_axis=2,
        box_cox_lambda=BOX_COX_LAMBDA,
        box_cox_epsilon=epsilon,
        std_correction=STD_CORRECTION,
        scale_stabilizer=SCALE_STABILIZER,
    )


def _json_statistics(stats: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in stats.items()
    }


def analyze_ignithit_open_dataset(
    root: Path,
    entries: Sequence[ManifestEntry],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Audit all train/validation arrays and replay reference-only metrics."""

    inventory = validate_local_open_tree(root, entries)
    metadata = load_ignithit_metadata(root / "data" / "data.npz")
    expected_trajectories = {
        trajectory_relative_path(split, group)
        for split, groups in (("train", TRAIN_GROUPS), ("val", VAL_GROUPS))
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
    envelope = np.zeros(len(IGNITHIT_FIELDS), dtype=np.float64)
    front_temperature: list[np.ndarray] = []
    front_oh: list[np.ndarray] = []
    trajectory_rows: list[dict[str, Any]] = []

    for split, groups in (("train", TRAIN_GROUPS), ("val", VAL_GROUPS)):
        for group in groups:
            relative = trajectory_relative_path(split, group)
            state = load_ignithit_trajectory(
                root.joinpath(*PurePosixPath(relative).parts)
            )
            diagnostics = trajectory_diagnostics(state)
            trajectory_rows.append(
                {
                    "split": split,
                    "group": group,
                    "relative_path": relative,
                    **diagnostics,
                }
            )
            if split == "train":
                primary_moments.update(state)
                sensitivity_moments.update(state)
                raw_moments.update(state)
                envelope = np.maximum(
                    envelope, np.abs(state).max(axis=(0, 2, 3)).astype(np.float64)
                )
                front_temperature.append(state[:, 8].reshape(-1).copy())
                front_oh.append(state[:, 7].reshape(-1).copy())

    primary = primary_moments.finalize()
    sensitivity = sensitivity_moments.finalize()
    raw = raw_moments.finalize()
    temperature_values = np.concatenate(front_temperature).astype(np.float64)
    oh_values = np.concatenate(front_oh).astype(np.float64)
    front_thresholds = {
        "quantiles": [FRONT_LOW_QUANTILE, FRONT_HIGH_QUANTILE],
        "numpy_quantile_method": "linear",
        "T": np.quantile(
            temperature_values, [FRONT_LOW_QUANTILE, FRONT_HIGH_QUANTILE]
        ).tolist(),
        "OH": np.quantile(
            oh_values, [FRONT_LOW_QUANTILE, FRONT_HIGH_QUANTILE]
        ).tolist(),
    }
    del temperature_values, oh_values

    roundtrip_max = {"float64": 0.0, "float32": 0.0}
    identity_npe_mean = 0.0
    identity_npe_sum = 0.0
    minimum_self_correlation = 1.0
    observable_correlation_count = 0
    pearson_status_counts: Counter[str] = Counter()
    normalizers = {
        "float64": _normalizer(primary, dtype=torch.float64),
        "float32": _normalizer(primary, dtype=torch.float32),
    }
    for split, groups in (("train", TRAIN_GROUPS), ("val", VAL_GROUPS)):
        for group in groups:
            relative = trajectory_relative_path(split, group)
            state = load_ignithit_trajectory(
                root.joinpath(*PurePosixPath(relative).parts)
            )
            for dtype_name, normalizer in normalizers.items():
                dtype = torch.float64 if dtype_name == "float64" else torch.float32
                native = torch.from_numpy(state).to(dtype=dtype).unsqueeze(0)
                encoded = normalizer.encode(native)
                decoded = normalizer.decode(encoded)
                reencoded = normalizer.encode(decoded)
                if not bool(torch.isfinite(decoded).all()):
                    raise ValueError("normalizer decode produced a nonfinite reference")
                error = float((reencoded - encoded).abs().max().item())
                roundtrip_max[dtype_name] = max(roundtrip_max[dtype_name], error)
            if split == "val":
                normalized = normalizers["float64"].encode(
                    torch.from_numpy(state).to(dtype=torch.float64).unsqueeze(0)
                )
                identity = grouped_normalized_prediction_error(
                    normalized[:, 1:], normalized[:, 1:], groups=IGNITHIT_GROUPS
                )
                identity_npe_mean = max(identity_npe_mean, identity.realm_npe_mean)
                identity_npe_sum = max(identity_npe_sum, identity.realm_npe_sum_source)
                decoded_state = (
                    torch.from_numpy(state).to(dtype=torch.float64).unsqueeze(0)
                )
                correlation = decoded_spatial_pearson(
                    decoded_state[:, 1:], decoded_state[:, 1:]
                )
                for case in correlation.statuses:
                    for call in case:
                        pearson_status_counts.update(call)
                observable = correlation.values[torch.isfinite(correlation.values)]
                if observable.numel():
                    observable_correlation_count += observable.numel()
                    minimum_self_correlation = min(
                        minimum_self_correlation, float(observable.min().item())
                    )

    all_admissible = all(
        bool(row["finite"]) and bool(row["admissible_released_state"])
        for row in trajectory_rows
    )
    gates = {
        "exact_file_inventory": True,
        "metadata_schema": True,
        "all_open_trajectories_finite_and_admissible": all_admissible,
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
        "scope": "open_train_validation_reference_replay_no_model",
        "metadata": metadata.summary(),
        "inventory": inventory,
        "trajectory_rows": trajectory_rows,
        "train_statistics_primary": _json_statistics(primary),
        "train_statistics_source_sensitivity": _json_statistics(sensitivity),
        "train_statistics_raw": _json_statistics(raw),
        "boundedness": {
            "quantile": BOUNDEDNESS_QUANTILE,
            "expansion_factor": BOUNDEDNESS_EXPANSION_FACTOR,
            "train_max_abs_by_channel": envelope.tolist(),
            "inclusive_limits_by_channel": (
                envelope * BOUNDEDNESS_EXPANSION_FACTOR
            ).tolist(),
        },
        "front_thresholds_train_only": front_thresholds,
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
            "released-species checks are not complete composition conservation",
            "native coordinate and time units remain unresolved",
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
        "train_max_abs": envelope,
    }
    return report, arrays


__all__ = [
    "BOUNDARY_BAND_CELLS",
    "BOUNDEDNESS_EXPANSION_FACTOR",
    "BOUNDEDNESS_QUANTILE",
    "BOX_COX_LAMBDA",
    "COORDINATE_UNIFORMITY_RTOL",
    "FLOAT32_REENCODE_TOLERANCE",
    "FLOAT64_REENCODE_TOLERANCE",
    "FRONT_HIGH_QUANTILE",
    "FRONT_LOW_QUANTILE",
    "METADATA_KEYS",
    "PRIMARY_BOX_COX_EPSILON",
    "SCHEMA",
    "SELF_CORRELATION_TOLERANCE",
    "SOURCE_SENSITIVITY_BOX_COX_EPSILON",
    "TEST_GROUPS",
    "TRAIN_GROUPS",
    "TRAJECTORY_DTYPE",
    "TRAJECTORY_KEY",
    "TRAJECTORY_SHAPE",
    "VAL_GROUPS",
    "IgnitHITMetadata",
    "StreamingChannelMoments",
    "analyze_ignithit_open_dataset",
    "git_blob_oid",
    "load_ignithit_metadata",
    "load_ignithit_trajectory",
    "sha256_file",
    "trajectory_diagnostics",
    "trajectory_relative_path",
    "validate_local_open_tree",
    "verify_manifest_file",
]
