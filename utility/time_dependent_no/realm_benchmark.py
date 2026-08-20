"""Synthetic-contract utilities for the staged REALM benchmark study.

This module intentionally contains no network, dataset-loader, checkpoint, or
training code.  It freezes the preprocessing, recurrence, metric, manifest,
and decoded-diagnostic semantics needed before any REALM trajectory is opened.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import PurePosixPath
from typing import Any, Literal

import torch

SCHEMA = "realm_benchmark_contract_v1"
MANIFEST_SCHEMA = "realm_huggingface_manifest_v1"

IGNITHIT_REPOSITORY = "realm-bench/realm-bench-IgnitHIT"
IGNITHIT_REVISION = "a0736b4d8c6c58a2688127e32addc30085e824c3"
IGNITHIT_FIELDS = (
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
)
IGNITHIT_OPEN_MANIFEST_SHA256 = (
    "85e8dcdaf4a5f7726ac5a7ad18a1bc131785fc1212acc9318ff69b94be965205"
)
IGNITHIT_OPEN_BYTES = 552_023_019
IGNITHIT_TRAJECTORY_BYTES = 552_014_385
IGNITHIT_METADATA_BYTES = 8_634

_HEX_40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX_64 = re.compile(r"[0-9a-f]{64}\Z")
_WINDOWS_DRIVE_PATH = re.compile(r"[A-Za-z]:/")
_GROUP_NAMES = ("chem", "T", "rho", "u", "p")


def _channel_axis(axis: int, ndim: int) -> int:
    resolved = axis if axis >= 0 else ndim + axis
    if resolved < 0 or resolved >= ndim:
        raise ValueError(f"channel_axis={axis} is invalid for ndim={ndim}")
    return resolved


def _require_float_tensor(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must use a floating dtype")


def _validate_box_cox_parameters(lam: float, epsilon: float) -> None:
    if not math.isfinite(lam) or lam < 0.0:
        raise ValueError("lambda must be finite and nonnegative")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")


def box_cox(
    values: torch.Tensor,
    *,
    lam: float = 0.1,
    epsilon: float,
) -> torch.Tensor:
    """Apply the registered clamp-then-Box-Cox transform.

    REALM's documented species transform uses ``lambda=0.1``.  ``lambda=0``
    is supported as the continuous logarithmic limit for independent tests.
    Negative and zero inputs are both clamped, rather than silently creating
    non-real powers.
    """

    _require_float_tensor(values, "values")
    _validate_box_cox_parameters(lam, epsilon)
    clamped = values.clamp_min(epsilon)
    if lam == 0.0:
        return torch.log(clamped)
    return (torch.pow(clamped, lam) - 1.0) / lam


InverseDomainPolicy = Literal["raise", "nan"]


def inverse_box_cox(
    transformed: torch.Tensor,
    *,
    lam: float = 0.1,
    epsilon: float,
    domain_policy: InverseDomainPolicy = "raise",
) -> torch.Tensor:
    """Invert Box-Cox and expose, rather than conceal, domain violations."""

    _require_float_tensor(transformed, "transformed")
    _validate_box_cox_parameters(lam, epsilon)
    if domain_policy not in ("raise", "nan"):
        raise ValueError("domain_policy must be 'raise' or 'nan'")
    if lam == 0.0:
        return torch.exp(transformed)

    base = lam * transformed + 1.0
    invalid = base < 0.0
    if bool(invalid.any()) and domain_policy == "raise":
        raise ValueError("inverse Box-Cox input is outside the real domain")
    if domain_policy == "nan":
        base = torch.where(invalid, torch.full_like(base, torch.nan), base)
    return torch.pow(base, 1.0 / lam)


def _transform_selected_channels(
    values: torch.Tensor,
    *,
    channel_axis: int,
    channels: Sequence[int],
    lam: float,
    epsilon: float,
    inverse: bool,
    inverse_domain_policy: InverseDomainPolicy = "raise",
) -> torch.Tensor:
    axis = _channel_axis(channel_axis, values.ndim)
    indices = tuple(channels)
    if len(set(indices)) != len(indices):
        raise ValueError("transformed channel indices must be unique")
    if any(index < 0 or index >= values.shape[axis] for index in indices):
        raise ValueError("transformed channel index is out of range")
    result = values.clone()
    for index in indices:
        selection = [slice(None)] * values.ndim
        selection[axis] = index
        key = tuple(selection)
        if inverse:
            result[key] = inverse_box_cox(
                values[key],
                lam=lam,
                epsilon=epsilon,
                domain_policy=inverse_domain_policy,
            )
        else:
            result[key] = box_cox(values[key], lam=lam, epsilon=epsilon)
    return result


@dataclass(frozen=True)
class RealmNormalizer:
    """Per-channel train-stat normalizer with an explicit species transform."""

    mean: torch.Tensor
    scale: torch.Tensor
    transformed_channels: tuple[int, ...]
    channel_axis: int = 2
    box_cox_lambda: float = 0.1
    box_cox_epsilon: float = 1.0e-40
    std_correction: int = 1
    scale_stabilizer: float = 1.0e-10

    def __post_init__(self) -> None:
        _require_float_tensor(self.mean, "mean")
        _require_float_tensor(self.scale, "scale")
        if self.mean.ndim != 1 or self.scale.ndim != 1:
            raise ValueError("mean and scale must be one-dimensional channel vectors")
        if self.mean.shape != self.scale.shape:
            raise ValueError("mean and scale must have the same shape")
        if not bool(torch.isfinite(self.mean).all()):
            raise ValueError("normalizer mean must be finite")
        if not bool(torch.isfinite(self.scale).all()) or not bool(
            (self.scale > 0.0).all()
        ):
            raise ValueError("normalizer scale must be finite and positive")
        _validate_box_cox_parameters(self.box_cox_lambda, self.box_cox_epsilon)
        if self.std_correction not in (0, 1):
            raise ValueError("std_correction must be 0 or 1")
        if not math.isfinite(self.scale_stabilizer) or self.scale_stabilizer <= 0:
            raise ValueError("scale_stabilizer must be finite and positive")
        if len(set(self.transformed_channels)) != len(self.transformed_channels):
            raise ValueError("transformed channel indices must be unique")
        if any(
            index < 0 or index >= self.mean.numel()
            for index in self.transformed_channels
        ):
            raise ValueError("transformed channel index is out of range")

    @classmethod
    def fit_train_only(
        cls,
        train_values: torch.Tensor,
        *,
        transformed_channels: Sequence[int],
        channel_axis: int = 2,
        box_cox_lambda: float = 0.1,
        box_cox_epsilon: float,
        std_correction: int = 1,
        scale_stabilizer: float = 1.0e-10,
    ) -> RealmNormalizer:
        """Fit over every axis except channel; callers must pass train data only."""

        _require_float_tensor(train_values, "train_values")
        if train_values.ndim < 2 or train_values.numel() == 0:
            raise ValueError("train_values must be a nonempty multi-axis tensor")
        axis = _channel_axis(channel_axis, train_values.ndim)
        if std_correction not in (0, 1):
            raise ValueError("std_correction must be 0 or 1")
        sample_count = train_values.numel() // train_values.shape[axis]
        if sample_count <= std_correction:
            raise ValueError(
                "not enough train samples for the requested std correction"
            )
        transformed = _transform_selected_channels(
            train_values,
            channel_axis=axis,
            channels=transformed_channels,
            lam=box_cox_lambda,
            epsilon=box_cox_epsilon,
            inverse=False,
        )
        reduction_axes = tuple(i for i in range(transformed.ndim) if i != axis)
        mean = transformed.mean(dim=reduction_axes)
        std = transformed.std(dim=reduction_axes, correction=std_correction)
        scale = torch.where(
            std < scale_stabilizer,
            torch.ones_like(std),
            std + scale_stabilizer,
        )
        return cls(
            mean=mean.detach().clone(),
            scale=scale.detach().clone(),
            transformed_channels=tuple(transformed_channels),
            channel_axis=channel_axis,
            box_cox_lambda=box_cox_lambda,
            box_cox_epsilon=box_cox_epsilon,
            std_correction=std_correction,
            scale_stabilizer=scale_stabilizer,
        )

    def _statistics_view(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        axis = _channel_axis(self.channel_axis, values.ndim)
        if values.shape[axis] != self.mean.numel():
            raise ValueError("input channel count does not match normalizer statistics")
        shape = [1] * values.ndim
        shape[axis] = self.mean.numel()
        mean = self.mean.to(device=values.device, dtype=values.dtype).reshape(shape)
        scale = self.scale.to(device=values.device, dtype=values.dtype).reshape(shape)
        return mean, scale

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        _require_float_tensor(values, "values")
        transformed = _transform_selected_channels(
            values,
            channel_axis=self.channel_axis,
            channels=self.transformed_channels,
            lam=self.box_cox_lambda,
            epsilon=self.box_cox_epsilon,
            inverse=False,
        )
        mean, scale = self._statistics_view(transformed)
        return (transformed - mean) / scale

    def decode(
        self,
        normalized: torch.Tensor,
        *,
        inverse_domain_policy: InverseDomainPolicy = "raise",
    ) -> torch.Tensor:
        _require_float_tensor(normalized, "normalized")
        mean, scale = self._statistics_view(normalized)
        transformed = normalized * scale + mean
        return _transform_selected_channels(
            transformed,
            channel_axis=self.channel_axis,
            channels=self.transformed_channels,
            lam=self.box_cox_lambda,
            epsilon=self.box_cox_epsilon,
            inverse=True,
            inverse_domain_policy=inverse_domain_policy,
        )


@dataclass(frozen=True)
class ChannelGroups:
    """Contiguous REALM field groups in released channel order."""

    chem: int
    temperature: int
    density: int
    velocity: int
    pressure: int

    def __post_init__(self) -> None:
        values = (
            self.chem,
            self.temperature,
            self.density,
            self.velocity,
            self.pressure,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in values
        ):
            raise TypeError("channel-group counts must be integers")
        if any(value < 0 for value in values) or sum(values) == 0:
            raise ValueError("channel-group counts must be nonnegative and nonempty")

    @property
    def total_channels(self) -> int:
        return (
            self.chem + self.temperature + self.density + self.velocity + self.pressure
        )

    def slices(self, *, expected_channels: int | None = None) -> dict[str, slice]:
        if expected_channels is not None and expected_channels != self.total_channels:
            raise ValueError("channel-group total does not match tensor channels")
        counts = (
            self.chem,
            self.temperature,
            self.density,
            self.velocity,
            self.pressure,
        )
        result: dict[str, slice] = {}
        start = 0
        for name, count in zip(_GROUP_NAMES, counts, strict=True):
            result[name] = slice(start, start + count)
            start += count
        return result


IGNITHIT_GROUPS = ChannelGroups(
    chem=8,
    temperature=1,
    density=1,
    velocity=2,
    pressure=0,
)


@dataclass(frozen=True)
class NormalizedPredictionError:
    grouped_per_call: Mapping[str, torch.Tensor]
    total_per_call: torch.Tensor
    per_case_mean: torch.Tensor
    per_case_sum: torch.Tensor
    realm_npe_mean: float
    realm_npe_sum_source: float


def grouped_normalized_prediction_error(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    *,
    groups: ChannelGroups = IGNITHIT_GROUPS,
) -> NormalizedPredictionError:
    """Compute separately named paper-average and released-source-sum channels.

    Inputs use ``[case, call, channel, *spatial]``.  Group MSE averages over
    channel and spatial axes, groups are summed, and cases receive equal weight.
    """

    _require_float_tensor(prediction, "prediction")
    _require_float_tensor(truth, "truth")
    if prediction.shape != truth.shape:
        raise ValueError("prediction and truth must have the same shape")
    if prediction.ndim < 4 or prediction.shape[0] == 0 or prediction.shape[1] == 0:
        raise ValueError("metrics require nonempty [case, call, channel, *spatial]")
    groups.slices(expected_channels=prediction.shape[2])
    if not bool(torch.isfinite(truth).all()):
        raise ValueError("truth must be finite")

    grouped: dict[str, torch.Tensor] = {}
    total = torch.zeros(
        prediction.shape[:2], device=prediction.device, dtype=prediction.dtype
    )
    reduction_axes = tuple(range(2, prediction.ndim))
    for name, channel_slice in groups.slices().items():
        if channel_slice.start == channel_slice.stop:
            continue
        difference = prediction[:, :, channel_slice] - truth[:, :, channel_slice]
        squared = torch.where(
            torch.isfinite(difference),
            difference.square(),
            torch.full_like(difference, torch.inf),
        )
        value = squared.mean(dim=reduction_axes)
        grouped[name] = value
        total = total + value

    per_case_mean = total.mean(dim=1)
    per_case_sum = total.sum(dim=1)
    return NormalizedPredictionError(
        grouped_per_call=grouped,
        total_per_call=total,
        per_case_mean=per_case_mean,
        per_case_sum=per_case_sum,
        realm_npe_mean=float(per_case_mean.mean().item()),
        realm_npe_sum_source=float(per_case_sum.mean().item()),
    )


PearsonStatus = Literal[
    "ok",
    "truth_nonfinite",
    "prediction_nonfinite",
    "constant_truth",
    "constant_prediction",
]


@dataclass(frozen=True)
class PearsonSummary:
    values: torch.Tensor
    statuses: tuple[tuple[tuple[PearsonStatus, ...], ...], ...]
    per_case_mean: torch.Tensor
    population_case_first_mean: float


def decoded_spatial_pearson(
    prediction: torch.Tensor,
    truth: torch.Tensor,
) -> PearsonSummary:
    """Spatial Pearson correlation per case/call/channel with reason codes."""

    _require_float_tensor(prediction, "prediction")
    _require_float_tensor(truth, "truth")
    if prediction.shape != truth.shape:
        raise ValueError("prediction and truth must have the same shape")
    if prediction.ndim < 4 or min(prediction.shape[:3]) == 0:
        raise ValueError("correlation requires [case, call, channel, *spatial]")
    spatial_size = math.prod(prediction.shape[3:])
    if spatial_size < 2:
        raise ValueError("correlation requires at least two spatial values")

    values = torch.full(
        prediction.shape[:3],
        torch.nan,
        device=prediction.device,
        dtype=prediction.dtype,
    )
    status_cases: list[tuple[tuple[PearsonStatus, ...], ...]] = []
    for case in range(prediction.shape[0]):
        status_calls: list[tuple[PearsonStatus, ...]] = []
        for call in range(prediction.shape[1]):
            status_channels: list[PearsonStatus] = []
            for channel in range(prediction.shape[2]):
                predicted = prediction[case, call, channel].reshape(-1)
                target = truth[case, call, channel].reshape(-1)
                if not bool(torch.isfinite(target).all()):
                    status: PearsonStatus = "truth_nonfinite"
                elif not bool(torch.isfinite(predicted).all()):
                    status = "prediction_nonfinite"
                else:
                    centered_target = target - target.mean()
                    centered_prediction = predicted - predicted.mean()
                    target_energy = torch.dot(centered_target, centered_target)
                    prediction_energy = torch.dot(
                        centered_prediction, centered_prediction
                    )
                    if float(target_energy.item()) == 0.0:
                        status = "constant_truth"
                    elif float(prediction_energy.item()) == 0.0:
                        status = "constant_prediction"
                    else:
                        values[case, call, channel] = torch.dot(
                            centered_prediction, centered_target
                        ) / torch.sqrt(target_energy * prediction_energy)
                        status = "ok"
                status_channels.append(status)
            status_calls.append(tuple(status_channels))
        status_cases.append(tuple(status_calls))

    per_case_values: list[torch.Tensor] = []
    for case in range(values.shape[0]):
        observable = values[case][torch.isfinite(values[case])]
        per_case_values.append(
            observable.mean()
            if observable.numel()
            else torch.full((), torch.nan, device=values.device, dtype=values.dtype)
        )
    per_case_mean = torch.stack(per_case_values)
    population = (
        float(per_case_mean.mean().item())
        if bool(torch.isfinite(per_case_mean).all())
        else math.nan
    )
    return PearsonSummary(
        values=values,
        statuses=tuple(status_cases),
        per_case_mean=per_case_mean,
        population_case_first_mean=population,
    )


Parameterization = Literal["direct", "residual"]
StepModel = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def apply_parameterization(
    current: torch.Tensor,
    model_output: torch.Tensor,
    *,
    parameterization: Parameterization,
) -> torch.Tensor:
    if current.shape != model_output.shape:
        raise ValueError("current and model_output must have the same shape")
    if parameterization == "direct":
        return model_output
    if parameterization == "residual":
        return current + model_output
    raise ValueError("parameterization must be 'direct' or 'residual'")


def predict_one_call(
    model: StepModel,
    current: torch.Tensor,
    static_coordinates: torch.Tensor,
    *,
    parameterization: Parameterization,
) -> torch.Tensor:
    """One learned call; the API deliberately has no truth argument."""

    output = model(current, static_coordinates)
    return apply_parameterization(current, output, parameterization=parameterization)


def predict_two_call_final(
    model: StepModel,
    current: torch.Tensor,
    static_coordinates: torch.Tensor,
    *,
    parameterization: Parameterization,
) -> torch.Tensor:
    """Two learned calls with call one detached and loss intended on call two."""

    device_type = current.device.type
    autocast_enabled = torch.is_autocast_enabled(device_type)
    autocast_dtype = (
        torch.get_autocast_dtype(device_type) if autocast_enabled else None
    )
    # An outer autocast context caches parameter casts.  If the detached first
    # call populates that cache under no_grad, call two can reuse detached
    # weights and silently lose parameter gradients.  Keep call one's casts
    # uncached while preserving the caller's autocast dtype and state.
    with torch.no_grad(), torch.autocast(
        device_type=device_type,
        dtype=autocast_dtype,
        enabled=autocast_enabled,
        cache_enabled=False,
    ):
        first = predict_one_call(
            model,
            current,
            static_coordinates,
            parameterization=parameterization,
        )
    first = first.detach()
    return predict_one_call(
        model,
        first,
        static_coordinates,
        parameterization=parameterization,
    )


@dataclass(frozen=True, order=True)
class ManifestEntry:
    """One immutable Hugging Face repository entry."""

    path: str
    size: int
    oid: str
    lfs: bool

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("manifest path must be a nonempty string")
        parsed = PurePosixPath(self.path)
        if (
            "\\" in self.path
            or _WINDOWS_DRIVE_PATH.match(self.path) is not None
            or parsed.is_absolute()
            or parsed.as_posix() != self.path
            or any(part in ("", ".", "..") for part in parsed.parts)
        ):
            raise ValueError("manifest path must be canonical relative POSIX")
        if (
            isinstance(self.size, bool)
            or not isinstance(self.size, int)
            or self.size < 0
        ):
            raise ValueError("manifest size must be a nonnegative integer")
        if not isinstance(self.lfs, bool):
            raise TypeError("manifest lfs flag must be bool")
        if not isinstance(self.oid, str):
            raise TypeError("manifest oid must be a string")
        normalized = self.oid.lower()
        expected = _HEX_64 if self.lfs else _HEX_40
        if expected.fullmatch(normalized) is None:
            kind = "64-hex LFS SHA-256" if self.lfs else "40-hex Git object ID"
            raise ValueError(f"manifest oid must be a {kind}")
        object.__setattr__(self, "oid", normalized)


def canonical_manifest_bytes(entries: Sequence[ManifestEntry]) -> bytes:
    ordered = sorted(entries, key=lambda entry: entry.path)
    if not ordered:
        raise ValueError("manifest must contain at least one entry")
    paths = [entry.path for entry in ordered]
    if len(paths) != len(set(paths)):
        raise ValueError("manifest paths must be unique")
    return "".join(
        f"{entry.path}\t{entry.size}\t{entry.oid}\n" for entry in ordered
    ).encode("utf-8")


def manifest_sha256(entries: Sequence[ManifestEntry]) -> str:
    return hashlib.sha256(canonical_manifest_bytes(entries)).hexdigest()


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    """Hash finite JSON with sorted keys and compact separators."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_manifest_payload(
    payload: Mapping[str, Any],
) -> tuple[str, str, tuple[ManifestEntry, ...]]:
    required_fields = {"schema", "repository", "revision", "entries"}
    if set(payload) != required_fields:
        raise ValueError(
            "manifest root must contain exactly schema, repository, revision, "
            "and entries"
        )
    if payload.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"manifest schema must be {MANIFEST_SCHEMA!r}")
    repository = payload.get("repository")
    revision = payload.get("revision")
    raw_entries = payload.get("entries")
    if not isinstance(repository, str) or not repository:
        raise ValueError("manifest repository must be a nonempty string")
    if not isinstance(revision, str) or _HEX_40.fullmatch(revision.lower()) is None:
        raise ValueError("manifest revision must be a 40-hex Git commit")
    if not isinstance(raw_entries, list):
        raise TypeError("manifest entries must be a list")
    entries: list[ManifestEntry] = []
    for raw in raw_entries:
        if not isinstance(raw, Mapping) or set(raw) != {"path", "size", "oid", "lfs"}:
            raise ValueError("each manifest entry must have path, size, oid, and lfs")
        entries.append(
            ManifestEntry(
                path=raw["path"],
                size=raw["size"],
                oid=raw["oid"],
                lfs=raw["lfs"],
            )
        )
    canonical_manifest_bytes(entries)
    return repository, revision.lower(), tuple(entries)


def _contains_forbidden_test_component(path: str) -> bool:
    return any(part.casefold() == "test" for part in PurePosixPath(path).parts)


def validate_open_manifest_contract(
    *,
    repository: str,
    revision: str,
    entries: Sequence[ManifestEntry],
    expected_repository: str,
    expected_revision: str,
    expected_sha256: str,
    expected_bytes: int,
) -> dict[str, Any]:
    """Validate an exact open-population manifest without opening any object."""

    if not expected_repository:
        raise ValueError("expected repository must be nonempty")
    if (
        _HEX_40.fullmatch(revision.lower()) is None
        or _HEX_40.fullmatch(expected_revision.lower()) is None
    ):
        raise ValueError("repository revisions must be 40-hex Git commits")
    if _HEX_64.fullmatch(expected_sha256.lower()) is None:
        raise ValueError("expected manifest SHA-256 must be 64-hex")
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes < 0
    ):
        raise ValueError("expected manifest bytes must be a nonnegative integer")
    if repository != expected_repository:
        raise ValueError("repository identity mismatch")
    if revision.lower() != expected_revision.lower():
        raise ValueError("repository revision mismatch")
    forbidden = sorted(
        entry.path
        for entry in entries
        if _contains_forbidden_test_component(entry.path)
    )
    if forbidden:
        raise ValueError(f"sealed test paths are forbidden: {forbidden}")
    unexpected_splits = sorted(
        entry.path
        for entry in entries
        if len(PurePosixPath(entry.path).parts) >= 3
        and PurePosixPath(entry.path).parts[0].casefold() == "data"
        and PurePosixPath(entry.path).parts[1].casefold() not in {"train", "val"}
    )
    if unexpected_splits:
        raise ValueError(
            f"trajectory split paths must be exactly train or val: {unexpected_splits}"
        )
    digest = manifest_sha256(entries)
    if digest != expected_sha256.lower():
        raise ValueError("open manifest SHA-256 mismatch")
    total_bytes = sum(entry.size for entry in entries)
    if total_bytes != expected_bytes:
        raise ValueError("open manifest byte total mismatch")
    return {
        "schema": SCHEMA,
        "repository": repository,
        "revision": revision.lower(),
        "manifest_sha256": digest,
        "entry_count": len(entries),
        "total_bytes": total_bytes,
        "sealed_test_paths_present": False,
    }


def validate_ignithit_open_manifest(
    repository: str,
    revision: str,
    entries: Sequence[ManifestEntry],
) -> dict[str, Any]:
    return validate_open_manifest_contract(
        repository=repository,
        revision=revision,
        entries=entries,
        expected_repository=IGNITHIT_REPOSITORY,
        expected_revision=IGNITHIT_REVISION,
        expected_sha256=IGNITHIT_OPEN_MANIFEST_SHA256,
        expected_bytes=IGNITHIT_OPEN_BYTES,
    )


@dataclass(frozen=True)
class AdmissibilitySummary:
    finite: bool
    admissible: bool
    violations: tuple[str, ...]
    min_species: float | None
    min_temperature: float | None
    min_density: float | None
    min_pressure: float | None


def decoded_admissibility(
    state: torch.Tensor,
    *,
    groups: ChannelGroups = IGNITHIT_GROUPS,
    channel_axis: int = -3,
) -> AdmissibilitySummary:
    """Check only released-state invariant-domain conditions.

    This is not a complete species-simplex, elemental-conservation, or physical
    validity test because omitted species and conservative volumes are unknown.
    """

    _require_float_tensor(state, "state")
    axis = _channel_axis(channel_axis, state.ndim)
    slices = groups.slices(expected_channels=state.shape[axis])
    finite = bool(torch.isfinite(state).all())
    if not finite:
        return AdmissibilitySummary(
            finite=False,
            admissible=False,
            violations=("native_nonfinite",),
            min_species=None,
            min_temperature=None,
            min_density=None,
            min_pressure=None,
        )

    def selected(channel_slice: slice) -> torch.Tensor | None:
        if channel_slice.start == channel_slice.stop:
            return None
        key = [slice(None)] * state.ndim
        key[axis] = channel_slice
        return state[tuple(key)]

    species = selected(slices["chem"])
    temperature = selected(slices["T"])
    density = selected(slices["rho"])
    pressure = selected(slices["p"])
    violations: list[str] = []
    if species is not None and bool((species < 0.0).any()):
        violations.append("negative_released_species")
    if temperature is not None and bool((temperature <= 0.0).any()):
        violations.append("nonpositive_temperature")
    if density is not None and bool((density <= 0.0).any()):
        violations.append("nonpositive_density")
    if pressure is not None and bool((pressure <= 0.0).any()):
        violations.append("nonpositive_pressure")

    def minimum(value: torch.Tensor | None) -> float | None:
        return None if value is None else float(value.min().item())

    return AdmissibilitySummary(
        finite=True,
        admissible=not violations,
        violations=tuple(violations),
        min_species=minimum(species),
        min_temperature=minimum(temperature),
        min_density=minimum(density),
        min_pressure=minimum(pressure),
    )


@dataclass(frozen=True)
class MagnitudeEnvelope:
    max_abs: torch.Tensor
    quantile: float
    channel_axis: int

    def __post_init__(self) -> None:
        _require_float_tensor(self.max_abs, "max_abs")
        if self.max_abs.ndim != 1 or not bool(torch.isfinite(self.max_abs).all()):
            raise ValueError("max_abs must be a finite channel vector")
        if not bool((self.max_abs >= 0.0).all()):
            raise ValueError("max_abs must be nonnegative")
        if not math.isfinite(self.quantile) or not 0.0 < self.quantile <= 1.0:
            raise ValueError("quantile must lie in (0, 1]")


def fit_train_magnitude_envelope(
    train_values: torch.Tensor,
    *,
    quantile: float,
    channel_axis: int = 2,
) -> MagnitudeEnvelope:
    _require_float_tensor(train_values, "train_values")
    if not bool(torch.isfinite(train_values).all()):
        raise ValueError("train envelope values must be finite")
    if not math.isfinite(quantile) or not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must lie in (0, 1]")
    axis = _channel_axis(channel_axis, train_values.ndim)
    moved = train_values.movedim(axis, 0).reshape(train_values.shape[axis], -1)
    if moved.shape[1] == 0:
        raise ValueError("train envelope requires at least one sample")
    return MagnitudeEnvelope(
        max_abs=torch.quantile(moved.abs(), quantile, dim=1),
        quantile=quantile,
        channel_axis=channel_axis,
    )


@dataclass(frozen=True)
class BoundednessSummary:
    finite: bool
    bounded: bool
    max_ratio_by_channel: tuple[float, ...]


def decoded_boundedness(
    state: torch.Tensor,
    envelope: MagnitudeEnvelope,
    *,
    expansion_factor: float,
    channel_axis: int | None = None,
) -> BoundednessSummary:
    _require_float_tensor(state, "state")
    if not math.isfinite(expansion_factor) or expansion_factor <= 0.0:
        raise ValueError("expansion_factor must be finite and positive")
    axis = _channel_axis(
        envelope.channel_axis if channel_axis is None else channel_axis,
        state.ndim,
    )
    if state.shape[axis] != envelope.max_abs.numel():
        raise ValueError("state channel count does not match envelope")
    if not bool(torch.isfinite(state).all()):
        return BoundednessSummary(
            False, False, tuple(math.inf for _ in envelope.max_abs)
        )
    moved = state.movedim(axis, 0).reshape(state.shape[axis], -1).abs()
    observed = moved.max(dim=1).values
    limit = (
        envelope.max_abs.to(device=state.device, dtype=state.dtype) * expansion_factor
    )
    ratios = torch.where(
        limit > 0.0,
        observed / limit,
        torch.where(
            observed == 0.0,
            torch.zeros_like(observed),
            torch.full_like(observed, torch.inf),
        ),
    )
    return BoundednessSummary(
        finite=True,
        bounded=bool((observed <= limit).all()),
        max_ratio_by_channel=tuple(float(value) for value in ratios.tolist()),
    )


def cartesian_boundary_band_mask(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    band_width: float,
) -> torch.Tensor:
    """Return an inclusive fixed-physical-width boundary mask on a tensor grid."""

    _require_float_tensor(x, "x")
    _require_float_tensor(y, "y")
    if x.ndim != 1 or y.ndim != 1 or x.numel() < 2 or y.numel() < 2:
        raise ValueError(
            "x and y must be one-dimensional grids with at least two points"
        )
    if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(y).all()):
        raise ValueError("x and y must be finite")
    if not bool((x[1:] > x[:-1]).all()) or not bool((y[1:] > y[:-1]).all()):
        raise ValueError("x and y must be strictly increasing")
    if not math.isfinite(band_width) or band_width <= 0.0:
        raise ValueError("band_width must be finite and positive")
    x_edge = (x - x[0] <= band_width) | (x[-1] - x <= band_width)
    y_edge = (y - y[0] <= band_width) | (y[-1] - y <= band_width)
    return y_edge[:, None] | x_edge[None, :]


@dataclass(frozen=True)
class BoundaryBandError:
    boundary_mse: float
    interior_mse: float
    boundary_to_interior_ratio: float


def boundary_band_error(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    boundary_mask: torch.Tensor,
) -> BoundaryBandError:
    _require_float_tensor(prediction, "prediction")
    _require_float_tensor(truth, "truth")
    if prediction.shape != truth.shape or prediction.ndim < 2:
        raise ValueError("prediction and truth must match and end in [y, x]")
    if boundary_mask.dtype != torch.bool or tuple(boundary_mask.shape) != tuple(
        prediction.shape[-2:]
    ):
        raise ValueError("boundary_mask must be bool with the spatial [y, x] shape")
    if not bool(boundary_mask.any()) or bool(boundary_mask.all()):
        raise ValueError("boundary mask must contain boundary and interior points")
    if not bool(torch.isfinite(truth).all()):
        raise ValueError("truth must be finite")
    error = prediction - truth
    squared = torch.where(
        torch.isfinite(error), error.square(), torch.full_like(error, torch.inf)
    )
    flat = squared.reshape(-1, *boundary_mask.shape)
    boundary_mse = flat[:, boundary_mask].mean()
    interior_mse = flat[:, ~boundary_mask].mean()
    ratio = torch.where(
        interior_mse > 0.0,
        boundary_mse / interior_mse,
        torch.where(
            boundary_mse == 0.0,
            torch.ones_like(boundary_mse),
            torch.full_like(boundary_mse, torch.inf),
        ),
    )
    return BoundaryBandError(
        boundary_mse=float(boundary_mse.item()),
        interior_mse=float(interior_mse.item()),
        boundary_to_interior_ratio=float(ratio.item()),
    )


@dataclass(frozen=True)
class CartesianFrontMetrics:
    high_area: float
    interface_length: float
    transition_area: float
    thickness_proxy: float
    centroid_x: float | None
    centroid_y: float | None
    jump_strength: float | None
    status: str


def cartesian_front_metrics(
    field: torch.Tensor,
    *,
    dx: float,
    dy: float,
    low_threshold: float,
    high_threshold: float,
) -> CartesianFrontMetrics:
    """Measure a 2-D scalar front with explicit physical grid spacing.

    High/low regions use inclusive bounds.  Transition area uses strict bounds,
    so a binary step has zero transition thickness.  Interface length counts
    neighbor crossings of the high threshold.
    """

    _require_float_tensor(field, "field")
    if field.ndim != 2 or min(field.shape) < 2:
        raise ValueError("front field must be a 2-D grid with both dimensions >= 2")
    if not bool(torch.isfinite(field).all()):
        raise ValueError("front field must be finite")
    if not math.isfinite(dx) or not math.isfinite(dy) or dx <= 0.0 or dy <= 0.0:
        raise ValueError("dx and dy must be finite and positive")
    if (
        not math.isfinite(low_threshold)
        or not math.isfinite(high_threshold)
        or low_threshold >= high_threshold
    ):
        raise ValueError("front thresholds must be finite with low < high")

    high = field >= high_threshold
    low = field <= low_threshold
    transition = (field > low_threshold) & (field < high_threshold)
    cell_area = dx * dy
    high_area = float(high.sum().item()) * cell_area
    transition_area = float(transition.sum().item()) * cell_area
    x_crossings = torch.logical_xor(high[:, 1:], high[:, :-1]).sum()
    y_crossings = torch.logical_xor(high[1:, :], high[:-1, :]).sum()
    interface_length = float(x_crossings.item()) * dy + float(y_crossings.item()) * dx
    thickness = transition_area / interface_length if interface_length > 0.0 else 0.0

    grad_y, grad_x = torch.gradient(field, spacing=(dy, dx), dim=(0, 1))
    gradient_weight = torch.sqrt(grad_x.square() + grad_y.square())
    weight_sum = gradient_weight.sum()
    if float(weight_sum.item()) > 0.0:
        x_coords = (
            torch.arange(field.shape[1], device=field.device, dtype=field.dtype) * dx
        )
        y_coords = (
            torch.arange(field.shape[0], device=field.device, dtype=field.dtype) * dy
        )
        centroid_x = float(
            (gradient_weight * x_coords[None, :]).sum().div(weight_sum).item()
        )
        centroid_y = float(
            (gradient_weight * y_coords[:, None]).sum().div(weight_sum).item()
        )
    else:
        centroid_x = None
        centroid_y = None

    if bool(high.any()) and bool(low.any()):
        jump_strength = float((field[high].mean() - field[low].mean()).item())
    else:
        jump_strength = None
    status = "ok" if interface_length > 0.0 else "no_interface"
    return CartesianFrontMetrics(
        high_area=high_area,
        interface_length=interface_length,
        transition_area=transition_area,
        thickness_proxy=thickness,
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        jump_strength=jump_strength,
        status=status,
    )


@dataclass(frozen=True)
class Spectrum2D:
    band_edges: tuple[float, ...]
    band_energy: tuple[float, ...]
    physical_energy: float
    spectral_energy: float


def physical_spectrum_2d(
    field: torch.Tensor,
    *,
    dx: float,
    dy: float,
    band_edges: Sequence[float],
    demean: bool = False,
) -> Spectrum2D:
    """Full-FFT radial energy in cycles per physical unit with Parseval closure."""

    _require_float_tensor(field, "field")
    if field.ndim != 2 or min(field.shape) < 2:
        raise ValueError("spectrum field must be a 2-D grid")
    if not bool(torch.isfinite(field).all()):
        raise ValueError("spectrum field must be finite")
    if not math.isfinite(dx) or not math.isfinite(dy) or dx <= 0.0 or dy <= 0.0:
        raise ValueError("dx and dy must be finite and positive")
    edges = tuple(float(edge) for edge in band_edges)
    if (
        len(edges) < 2
        or any(not math.isfinite(edge) or edge < 0.0 for edge in edges)
        or any(right <= left for left, right in pairwise(edges))
    ):
        raise ValueError("band edges must be finite, nonnegative, and increasing")

    analyzed = field - field.mean() if demean else field
    coefficients = torch.fft.fft2(analyzed, norm="ortho")
    energy = coefficients.abs().square()
    frequency_x = torch.fft.fftfreq(
        field.shape[1], d=dx, device=field.device, dtype=field.dtype
    )
    frequency_y = torch.fft.fftfreq(
        field.shape[0], d=dy, device=field.device, dtype=field.dtype
    )
    radial = torch.sqrt(frequency_y[:, None].square() + frequency_x[None, :].square())
    if edges[-1] < float(radial.max().item()):
        raise ValueError("last band edge must include the maximum grid frequency")
    band_energy: list[float] = []
    for index, (left, right) in enumerate(pairwise(edges)):
        mask = (radial >= left) & (
            radial <= right if index == len(edges) - 2 else radial < right
        )
        band_energy.append(float(energy[mask].sum().item()))
    return Spectrum2D(
        band_edges=edges,
        band_energy=tuple(band_energy),
        physical_energy=float(analyzed.square().sum().item()),
        spectral_energy=float(energy.sum().item()),
    )


__all__ = [
    "IGNITHIT_FIELDS",
    "IGNITHIT_GROUPS",
    "IGNITHIT_METADATA_BYTES",
    "IGNITHIT_OPEN_BYTES",
    "IGNITHIT_OPEN_MANIFEST_SHA256",
    "IGNITHIT_REPOSITORY",
    "IGNITHIT_REVISION",
    "IGNITHIT_TRAJECTORY_BYTES",
    "MANIFEST_SCHEMA",
    "SCHEMA",
    "AdmissibilitySummary",
    "BoundaryBandError",
    "BoundednessSummary",
    "CartesianFrontMetrics",
    "ChannelGroups",
    "MagnitudeEnvelope",
    "ManifestEntry",
    "NormalizedPredictionError",
    "PearsonSummary",
    "RealmNormalizer",
    "Spectrum2D",
    "apply_parameterization",
    "boundary_band_error",
    "box_cox",
    "canonical_json_sha256",
    "canonical_manifest_bytes",
    "cartesian_boundary_band_mask",
    "cartesian_front_metrics",
    "decoded_admissibility",
    "decoded_boundedness",
    "decoded_spatial_pearson",
    "fit_train_magnitude_envelope",
    "grouped_normalized_prediction_error",
    "inverse_box_cox",
    "manifest_sha256",
    "parse_manifest_payload",
    "physical_spectrum_2d",
    "predict_one_call",
    "predict_two_call_final",
    "validate_ignithit_open_manifest",
    "validate_open_manifest_contract",
]
