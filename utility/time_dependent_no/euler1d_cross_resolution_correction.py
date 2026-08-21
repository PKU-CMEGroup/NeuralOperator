"""Common-source 1D Euler cross-resolution correction primitives.

The module contains no checkpoint or dataset loader.  Reference error appears
only in offline fit/score records; a deployed correction uses a predicted
native increment, a synchronized fine-minus-native predicted increment, and a
coefficient frozen on another population.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

DENOMINATOR_FLOOR = 1.0e-8
MAXIMUM_RELATIVE_CORRECTION = 0.10
PHYSICAL_COMPONENT_SCALES = np.asarray((1.0, 1.0, 2.5), dtype=np.float64)


@dataclass(frozen=True)
class Euler1DResolutionContract:
    """Three strictly nested uniform cell-average grids."""

    coarse_cells: int = 128
    native_cells: int = 256
    fine_cells: int = 512

    def __post_init__(self) -> None:
        values = (self.coarse_cells, self.native_cells, self.fine_cells)
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < 1
            for value in values
        ):
            raise ValueError("cell counts must be positive integers")
        if not self.coarse_cells < self.native_cells < self.fine_cells:
            raise ValueError("coarse/native/fine cell counts must be ordered")
        if self.native_cells % self.coarse_cells:
            raise ValueError("coarse cells must divide native cells")
        if self.fine_cells % self.native_cells:
            raise ValueError("native cells must divide fine cells")


@dataclass(frozen=True)
class CommonSourceStates:
    """Conservative cell averages derived from one fine physical state."""

    coarse: np.ndarray
    native: np.ndarray
    fine: np.ndarray


@dataclass(frozen=True)
class NativeIncrementBasis1D:
    """Synchronized predicted increments expressed on the native grid."""

    native_increment: np.ndarray
    coarse_on_native: np.ndarray
    fine_on_native: np.ndarray
    native_minus_coarse: np.ndarray
    fine_minus_native: np.ndarray


@dataclass(frozen=True)
class CosineProjector1D:
    """Physical-volume weighted QR basis for ordered 1D cosine modes."""

    modes: tuple[int, ...]
    volumes: np.ndarray
    square_root_mass: np.ndarray
    q_matrix: np.ndarray
    maximum_orthogonality_error: float

    @property
    def num_cells(self) -> int:
        return int(self.q_matrix.shape[0])

    @property
    def rank(self) -> int:
        return int(self.q_matrix.shape[1])

    def coordinates(
        self,
        field: np.ndarray,
        *,
        component_scale: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        values = _field(field, self.num_cells, "field")
        scale = _component_scale(component_scale, values.shape[1])
        weighted = self.square_root_mass[:, None] * (values / scale[None, :])
        return np.asarray(self.q_matrix.T @ weighted, dtype=np.float64)

    def reconstruct(
        self,
        coordinates: np.ndarray,
        *,
        component_scale: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(coordinates, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] != self.rank:
            raise ValueError("coordinates must have shape [rank, components]")
        if not np.isfinite(values).all():
            raise ValueError("coordinates must be finite")
        scale = _component_scale(component_scale, values.shape[1])
        weighted = self.q_matrix @ values
        return weighted / self.square_root_mass[:, None] * scale[None, :]

    def project(
        self,
        field: np.ndarray,
        *,
        active_modes: Sequence[int],
        component_scale: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        requested = tuple(active_modes)
        if any(
            isinstance(mode, (bool, np.bool_))
            or not isinstance(mode, (int, np.integer))
            for mode in requested
        ):
            raise ValueError("active modes must be integers")
        indices = []
        for mode in requested:
            try:
                indices.append(self.modes.index(int(mode)))
            except ValueError as error:
                raise ValueError(f"mode {mode} is absent from the projector") from error
        coordinates = self.coordinates(field, component_scale=component_scale)
        inactive = np.ones(self.rank, dtype=np.bool_)
        inactive[np.asarray(indices, dtype=np.int64)] = False
        coordinates[inactive] = 0.0
        return self.reconstruct(coordinates, component_scale=component_scale)


@dataclass(frozen=True)
class ScalarFitRecord:
    """One case/call feature-target pair for offline scalar fitting."""

    case_id: str
    input_call: int
    feature: np.ndarray
    target_correction: np.ndarray
    native_increment: np.ndarray
    volumes: np.ndarray
    component_scale: np.ndarray


@dataclass(frozen=True)
class ScoreRecord:
    """One case/call target and already constructed correction."""

    case_id: str
    input_call: int
    target_correction: np.ndarray
    correction: np.ndarray
    volumes: np.ndarray
    component_scale: np.ndarray


@dataclass(frozen=True)
class ScalarFit:
    coefficient: float | None
    feature_rms: float
    target_rms: float
    cross: float
    denominator: float
    status: str


@dataclass(frozen=True)
class CorrectionAudit:
    status: str
    coefficient: float
    native_increment_rms: float
    raw_correction_rms: float
    correction_rms: float
    correction_to_native_increment: float | None
    applied_scale: float
    cap_active: bool
    maximum_component_integral_abs: float


def _field(value: np.ndarray, cells: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != cells or array.shape[1] < 1:
        raise ValueError(f"{name} must have shape [{cells}, components]")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _volumes(value: Sequence[float] | np.ndarray, cells: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (cells,) or not np.isfinite(array).all():
        raise ValueError("volumes must be finite and align with cells")
    if np.any(array <= 0.0):
        raise ValueError("volumes must be positive")
    return array


def _component_scale(
    value: Sequence[float] | np.ndarray,
    components: int,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (components,) or not np.isfinite(array).all():
        raise ValueError("component_scale must be finite and align with components")
    if np.any(array <= 0.0):
        raise ValueError("component_scale must be positive")
    return array


def restrict_cell_averages(field: np.ndarray, target_cells: int) -> np.ndarray:
    """Conservatively block-average a nested 1D cell-average field."""

    values = np.asarray(field, dtype=np.float64)
    if values.ndim < 2 or values.shape[-1] < 1:
        raise ValueError("field must have shape [..., cells, components]")
    source_cells = int(values.shape[-2])
    if (
        isinstance(target_cells, (bool, np.bool_))
        or not isinstance(target_cells, (int, np.integer))
        or int(target_cells) < 1
    ):
        raise ValueError("target_cells must be a positive integer")
    target = int(target_cells)
    if source_cells % target:
        raise ValueError("target cell count must divide the source cell count")
    if not np.isfinite(values).all():
        raise ValueError("field must be finite")
    ratio = source_cells // target
    shape = (*values.shape[:-2], target, ratio, values.shape[-1])
    return np.asarray(values.reshape(shape).mean(axis=-2), dtype=np.float64)


def prolong_piecewise_constant(field: np.ndarray, target_cells: int) -> np.ndarray:
    """Repeat each coarse cell average on all of its nested children."""

    values = np.asarray(field, dtype=np.float64)
    if values.ndim < 2 or values.shape[-1] < 1:
        raise ValueError("field must have shape [..., cells, components]")
    source_cells = int(values.shape[-2])
    if (
        isinstance(target_cells, (bool, np.bool_))
        or not isinstance(target_cells, (int, np.integer))
        or int(target_cells) < 1
    ):
        raise ValueError("target_cells must be a positive integer")
    target = int(target_cells)
    if target % source_cells:
        raise ValueError("source cell count must divide the target cell count")
    if not np.isfinite(values).all():
        raise ValueError("field must be finite")
    return np.repeat(values, target // source_cells, axis=-2)


def common_source_states(
    fine_conservative: np.ndarray,
    contract: Euler1DResolutionContract,
) -> CommonSourceStates:
    fine = _field(fine_conservative, contract.fine_cells, "fine_conservative")
    return CommonSourceStates(
        coarse=restrict_cell_averages(fine, contract.coarse_cells),
        native=restrict_cell_averages(fine, contract.native_cells),
        fine=np.array(fine, copy=True),
    )


def mapped_increment_basis(
    states: CommonSourceStates,
    predicted_next: CommonSourceStates,
    contract: Euler1DResolutionContract,
) -> NativeIncrementBasis1D:
    coarse_increment = _field(
        predicted_next.coarse - states.coarse,
        contract.coarse_cells,
        "coarse_increment",
    )
    native_increment = _field(
        predicted_next.native - states.native,
        contract.native_cells,
        "native_increment",
    )
    fine_increment = _field(
        predicted_next.fine - states.fine,
        contract.fine_cells,
        "fine_increment",
    )
    coarse_on_native = prolong_piecewise_constant(
        coarse_increment, contract.native_cells
    )
    fine_on_native = restrict_cell_averages(fine_increment, contract.native_cells)
    return NativeIncrementBasis1D(
        native_increment=native_increment,
        coarse_on_native=coarse_on_native,
        fine_on_native=fine_on_native,
        native_minus_coarse=native_increment - coarse_on_native,
        fine_minus_native=fine_on_native - native_increment,
    )


def build_cosine_projector(
    num_cells: int,
    *,
    rank: int,
    volumes: Sequence[float] | np.ndarray | None = None,
) -> CosineProjector1D:
    """Build ordered modes ``cos(k*pi*x)``, starting with the constant."""

    if (
        isinstance(num_cells, (bool, np.bool_))
        or not isinstance(num_cells, (int, np.integer))
        or int(num_cells) < 1
    ):
        raise ValueError("num_cells must be a positive integer")
    cells = int(num_cells)
    if (
        isinstance(rank, (bool, np.bool_))
        or not isinstance(rank, (int, np.integer))
        or int(rank) < 1
        or int(rank) > cells
    ):
        raise ValueError("rank must be an integer in [1, num_cells]")
    mass = (
        np.full(cells, 1.0 / cells, dtype=np.float64)
        if volumes is None
        else _volumes(volumes, cells)
    )
    x = (np.arange(cells, dtype=np.float64) + 0.5) / cells
    modes = tuple(range(int(rank)))
    design = np.column_stack([np.cos(np.pi * mode * x) for mode in modes])
    root_mass = np.sqrt(mass)
    weighted = root_mass[:, None] * design
    q_matrix, r_matrix = np.linalg.qr(weighted, mode="reduced")
    singular_values = np.linalg.svd(r_matrix, compute_uv=False)
    tolerance = (
        max(weighted.shape)
        * np.finfo(np.float64).eps
        * max(float(singular_values[0]), 1.0)
    )
    if int(np.count_nonzero(singular_values > tolerance)) != int(rank):
        raise ValueError("cosine basis is rank deficient")
    orthogonality = q_matrix.T @ q_matrix - np.eye(int(rank), dtype=np.float64)
    return CosineProjector1D(
        modes=modes,
        volumes=mass,
        square_root_mass=root_mass,
        q_matrix=np.asarray(q_matrix, dtype=np.float64),
        maximum_orthogonality_error=float(np.max(np.abs(orthogonality))),
    )


def weighted_scaled_inner(
    first: np.ndarray,
    second: np.ndarray,
    *,
    volumes: Sequence[float] | np.ndarray,
    component_scale: Sequence[float] | np.ndarray,
) -> float:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] < 1:
        raise ValueError("fields must have equal [cells, components] shapes")
    left = _field(left, left.shape[0], "first")
    right = _field(right, right.shape[0], "second")
    mass = _volumes(volumes, left.shape[0])
    scale = _component_scale(component_scale, left.shape[1])
    normalized = (left / scale[None, :]) * (right / scale[None, :])
    return float(
        np.einsum("n,nc->", mass, normalized, optimize=True)
        / (float(mass.sum()) * left.shape[1])
    )


def weighted_scaled_rms(
    field: np.ndarray,
    *,
    volumes: Sequence[float] | np.ndarray,
    component_scale: Sequence[float] | np.ndarray,
) -> float:
    value = weighted_scaled_inner(
        field,
        field,
        volumes=volumes,
        component_scale=component_scale,
    )
    return float(np.sqrt(max(value, 0.0)))


def component_integrals(
    field: np.ndarray,
    *,
    volumes: Sequence[float] | np.ndarray,
    component_scale: Sequence[float] | np.ndarray,
) -> np.ndarray:
    values = np.asarray(field, dtype=np.float64)
    values = _field(values, values.shape[0], "field")
    mass = _volumes(volumes, values.shape[0])
    scale = _component_scale(component_scale, values.shape[1])
    return np.einsum("n,nc->c", mass, values / scale[None, :], optimize=True)


def apply_capped_correction(
    native_increment: np.ndarray,
    projected_feature: np.ndarray,
    *,
    coefficient: float,
    volumes: Sequence[float] | np.ndarray,
    component_scale: Sequence[float] | np.ndarray,
    maximum_relative_norm: float = MAXIMUM_RELATIVE_CORRECTION,
    denominator_floor: float = DENOMINATOR_FLOOR,
) -> tuple[np.ndarray, CorrectionAudit]:
    """Apply one scalar correction with the registered target-free norm cap."""

    native = np.asarray(native_increment, dtype=np.float64)
    feature = np.asarray(projected_feature, dtype=np.float64)
    if native.shape != feature.shape or native.ndim != 2:
        raise ValueError("native increment and projected feature must align")
    native = _field(native, native.shape[0], "native_increment")
    feature = _field(feature, feature.shape[0], "projected_feature")
    if not np.isfinite(coefficient):
        raise ValueError("coefficient must be finite")
    if not np.isfinite(maximum_relative_norm) or maximum_relative_norm <= 0.0:
        raise ValueError("maximum_relative_norm must be finite and positive")
    if not np.isfinite(denominator_floor) or denominator_floor <= 0.0:
        raise ValueError("denominator_floor must be finite and positive")
    mass = _volumes(volumes, native.shape[0])
    scale = _component_scale(component_scale, native.shape[1])
    native_rms = weighted_scaled_rms(native, volumes=mass, component_scale=scale)
    raw = float(coefficient) * feature
    raw_rms = weighted_scaled_rms(raw, volumes=mass, component_scale=scale)
    correction = np.zeros_like(native)
    status = "ok"
    applied_scale = 0.0
    cap_active = False
    ratio = None
    if native_rms <= denominator_floor:
        status = "small_native_increment"
    elif raw_rms <= denominator_floor:
        status = "small_correction"
        ratio = 0.0
    else:
        applied_scale = min(1.0, maximum_relative_norm * native_rms / raw_rms)
        cap_active = applied_scale < 1.0
        correction = applied_scale * raw
        ratio = (
            weighted_scaled_rms(correction, volumes=mass, component_scale=scale)
            / native_rms
        )
    correction_rms = weighted_scaled_rms(
        correction, volumes=mass, component_scale=scale
    )
    integrals = component_integrals(
        correction,
        volumes=mass,
        component_scale=scale,
    )
    return correction, CorrectionAudit(
        status=status,
        coefficient=float(coefficient),
        native_increment_rms=native_rms,
        raw_correction_rms=raw_rms,
        correction_rms=correction_rms,
        correction_to_native_increment=ratio,
        applied_scale=float(applied_scale),
        cap_active=cap_active,
        maximum_component_integral_abs=float(np.max(np.abs(integrals))),
    )


def _validate_fit_record(record: ScalarFitRecord) -> None:
    if not isinstance(record.case_id, str) or not record.case_id:
        raise ValueError("case_id must be a nonempty string")
    if (
        isinstance(record.input_call, (bool, np.bool_))
        or not isinstance(record.input_call, (int, np.integer))
        or int(record.input_call) < 0
    ):
        raise ValueError("input_call must be a nonnegative integer")
    feature = np.asarray(record.feature, dtype=np.float64)
    target = np.asarray(record.target_correction, dtype=np.float64)
    native = np.asarray(record.native_increment, dtype=np.float64)
    if feature.shape != target.shape or feature.shape != native.shape:
        raise ValueError("feature, target, and native increment must align")
    _field(feature, feature.shape[0], "feature")
    _field(target, target.shape[0], "target_correction")
    _field(native, native.shape[0], "native_increment")
    _volumes(record.volumes, feature.shape[0])
    _component_scale(record.component_scale, feature.shape[1])


def _validate_score_record(record: ScoreRecord) -> None:
    if not isinstance(record.case_id, str) or not record.case_id:
        raise ValueError("case_id must be a nonempty string")
    if (
        isinstance(record.input_call, (bool, np.bool_))
        or not isinstance(record.input_call, (int, np.integer))
        or int(record.input_call) < 0
    ):
        raise ValueError("input_call must be a nonnegative integer")
    target = np.asarray(record.target_correction, dtype=np.float64)
    correction = np.asarray(record.correction, dtype=np.float64)
    if target.shape != correction.shape:
        raise ValueError("target and correction must align")
    _field(target, target.shape[0], "target_correction")
    _field(correction, correction.shape[0], "correction")
    _volumes(record.volumes, target.shape[0])
    _component_scale(record.component_scale, target.shape[1])


def validate_record_inventory(
    records: Sequence[ScalarFitRecord] | Sequence[ScoreRecord],
    *,
    expected_case_ids: Sequence[str],
    expected_input_calls: Sequence[int],
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Fail closed on duplicate, missing, or extra case/call records."""

    cases = tuple(expected_case_ids)
    calls = tuple(expected_input_calls)
    if not cases or any(
        not isinstance(case_id, str) or not case_id for case_id in cases
    ):
        raise ValueError("expected_case_ids must contain nonempty strings")
    if len(set(cases)) != len(cases):
        raise ValueError("expected_case_ids must be unique")
    if not calls or any(
        isinstance(call, (bool, np.bool_))
        or not isinstance(call, (int, np.integer))
        or int(call) < 0
        for call in calls
    ):
        raise ValueError("expected_input_calls must be nonnegative integers")
    normalized_calls = tuple(int(call) for call in calls)
    if len(set(normalized_calls)) != len(normalized_calls):
        raise ValueError("expected_input_calls must be unique")
    expected = {(case_id, call) for case_id in cases for call in normalized_calls}
    actual: set[tuple[str, int]] = set()
    for record in records:
        if isinstance(record, ScalarFitRecord):
            _validate_fit_record(record)
        elif isinstance(record, ScoreRecord):
            _validate_score_record(record)
        else:
            raise TypeError("records must be scalar-fit or score records")
        key = (record.case_id, int(record.input_call))
        if key in actual:
            raise ValueError(f"duplicate case/call record: {key!r}")
        actual.add(key)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(f"record inventory mismatch: missing={missing}, extra={extra}")
    return tuple(sorted(cases)), tuple(sorted(normalized_calls))


def _case_first_fit_sums(
    records: Sequence[ScalarFitRecord],
) -> tuple[float, float, float, int]:
    ordered = sorted(records, key=lambda row: (row.case_id, int(row.input_call)))
    if not ordered:
        raise ValueError("scalar fitting requires records")
    reference_scale = np.asarray(ordered[0].component_scale)
    if any(
        not np.array_equal(np.asarray(record.component_scale), reference_scale)
        for record in ordered[1:]
    ):
        raise ValueError("all fit records must share component scales")
    per_case: dict[str, list[tuple[float, float, float]]] = {}
    for record in ordered:
        _validate_fit_record(record)
        feature_square = weighted_scaled_inner(
            record.feature,
            record.feature,
            volumes=record.volumes,
            component_scale=record.component_scale,
        )
        cross = weighted_scaled_inner(
            record.feature,
            record.target_correction,
            volumes=record.volumes,
            component_scale=record.component_scale,
        )
        target_square = weighted_scaled_inner(
            record.target_correction,
            record.target_correction,
            volumes=record.volumes,
            component_scale=record.component_scale,
        )
        per_case.setdefault(record.case_id, []).append(
            (feature_square, cross, target_square)
        )
    feature_sum = 0.0
    cross_sum = 0.0
    target_sum = 0.0
    for values in per_case.values():
        array = np.asarray(values, dtype=np.float64)
        feature_sum += float(np.mean(array[:, 0]))
        cross_sum += float(np.mean(array[:, 1]))
        target_sum += float(np.mean(array[:, 2]))
    count = len(per_case)
    return feature_sum / count, cross_sum / count, target_sum / count, count


def fit_case_first_scalar(
    records: Sequence[ScalarFitRecord],
    *,
    denominator_floor: float = DENOMINATOR_FLOOR,
) -> ScalarFit:
    """Fit one through-origin coefficient with equal case and call weights."""

    feature_square, cross, target_square, _case_count = _case_first_fit_sums(records)
    feature_rms = float(np.sqrt(max(feature_square, 0.0)))
    target_rms = float(np.sqrt(max(target_square, 0.0)))
    if not all(np.isfinite(value) for value in (feature_square, cross, target_square)):
        return ScalarFit(
            None, feature_rms, target_rms, cross, feature_square, "nonfinite"
        )
    if feature_rms <= denominator_floor:
        return ScalarFit(
            None,
            feature_rms,
            target_rms,
            cross,
            feature_square,
            "small_feature_denominator",
        )
    return ScalarFit(
        coefficient=float(cross / feature_square),
        feature_rms=feature_rms,
        target_rms=target_rms,
        cross=cross,
        denominator=feature_square,
        status="ok",
    )


def score_records(
    records: Sequence[ScoreRecord],
    *,
    denominator_floor: float = DENOMINATOR_FLOOR,
) -> dict[str, Any]:
    """Case-first error skill and per-case signed relation diagnostics."""

    ordered = sorted(records, key=lambda row: (row.case_id, int(row.input_call)))
    if not ordered:
        raise ValueError("scoring requires records")
    by_case: dict[str, list[ScoreRecord]] = {}
    for record in ordered:
        _validate_score_record(record)
        by_case.setdefault(record.case_id, []).append(record)
    case_rows = []
    population_zero = 0.0
    population_corrected = 0.0
    population_correction_square = 0.0
    target_centered = 0.0
    correction_centered = 0.0
    centered_cross = 0.0
    for case_id, rows in sorted(by_case.items()):
        zero_values = []
        corrected_values = []
        feature_values = []
        target_values = []
        feature_means = []
        target_means = []
        for record in rows:
            target = np.asarray(record.target_correction, dtype=np.float64)
            correction = np.asarray(record.correction, dtype=np.float64)
            zero_values.append(
                weighted_scaled_inner(
                    target,
                    target,
                    volumes=record.volumes,
                    component_scale=record.component_scale,
                )
            )
            corrected_values.append(
                weighted_scaled_inner(
                    target - correction,
                    target - correction,
                    volumes=record.volumes,
                    component_scale=record.component_scale,
                )
            )
            feature_values.append(
                weighted_scaled_inner(
                    correction,
                    correction,
                    volumes=record.volumes,
                    component_scale=record.component_scale,
                )
            )
            target_values.append(
                weighted_scaled_inner(
                    correction,
                    target,
                    volumes=record.volumes,
                    component_scale=record.component_scale,
                )
            )
            mass = _volumes(record.volumes, target.shape[0])
            scale = _component_scale(record.component_scale, target.shape[1])
            normalized_weight = mass / (float(mass.sum()) * target.shape[1])
            feature_means.append(
                float(
                    np.einsum(
                        "n,nc->",
                        normalized_weight,
                        correction / scale[None, :],
                        optimize=True,
                    )
                )
            )
            target_means.append(
                float(
                    np.einsum(
                        "n,nc->",
                        normalized_weight,
                        target / scale[None, :],
                        optimize=True,
                    )
                )
            )
        zero_sse = float(np.mean(zero_values))
        corrected_sse = float(np.mean(corrected_values))
        correction_square = float(np.mean(feature_values))
        correction_target = float(np.mean(target_values))
        correction_mean = float(np.mean(feature_means))
        target_mean = float(np.mean(target_means))
        centered_target = max(zero_sse - target_mean * target_mean, 0.0)
        centered_correction = max(
            correction_square - correction_mean * correction_mean, 0.0
        )
        centered_relation = correction_target - correction_mean * target_mean
        target_rms = float(np.sqrt(max(zero_sse, 0.0)))
        correction_rms = float(np.sqrt(max(correction_square, 0.0)))
        cosine_denominator = float(np.sqrt(max(zero_sse * correction_square, 0.0)))
        correlation_denominator = float(
            np.sqrt(max(centered_target * centered_correction, 0.0))
        )
        skill = (
            float(1.0 - corrected_sse / zero_sse)
            if target_rms > denominator_floor
            else None
        )
        cosine = (
            float(correction_target / cosine_denominator)
            if target_rms > denominator_floor and correction_rms > denominator_floor
            else None
        )
        correlation = (
            float(centered_relation / correlation_denominator)
            if np.sqrt(centered_target) > denominator_floor
            and np.sqrt(centered_correction) > denominator_floor
            else None
        )
        case_rows.append(
            {
                "case_id": case_id,
                "input_calls": [int(row.input_call) for row in rows],
                "zero_sse": zero_sse,
                "corrected_sse": corrected_sse,
                "skill_vs_zero": skill,
                "rms_ratio_vs_zero": (
                    float(np.sqrt(corrected_sse / zero_sse))
                    if skill is not None
                    else None
                ),
                "cosine": cosine,
                "correlation": correlation,
                "target_rms": target_rms,
                "correction_rms": correction_rms,
                "cosine_denominator": cosine_denominator,
                "correlation_denominator": correlation_denominator,
                "skill_status": "ok" if skill is not None else "small_target",
                "cosine_status": (
                    "ok" if cosine is not None else "small_target_or_correction"
                ),
                "correlation_status": (
                    "ok"
                    if correlation is not None
                    else "small_centered_target_or_correction"
                ),
            }
        )
        population_zero += zero_sse
        population_corrected += corrected_sse
        population_correction_square += correction_square
        target_centered += centered_target
        correction_centered += centered_correction
        centered_cross += centered_relation
    count = len(case_rows)
    population_zero /= count
    population_corrected /= count
    population_correction_square /= count
    target_centered /= count
    correction_centered /= count
    centered_cross /= count
    target_rms = float(np.sqrt(max(population_zero, 0.0)))
    skill = (
        float(1.0 - population_corrected / population_zero)
        if target_rms > denominator_floor
        else None
    )
    centered_r2 = (
        float(1.0 - population_corrected / target_centered)
        if np.sqrt(target_centered) > denominator_floor
        else None
    )
    resolved_cosines = [
        float(row["cosine"])
        for row in case_rows
        if row["cosine_status"] == "ok" and row["cosine"] is not None
    ]
    resolved_correlations = [
        float(row["correlation"])
        for row in case_rows
        if row["correlation_status"] == "ok" and row["correlation"] is not None
    ]
    return {
        "case_count": count,
        "zero_sse_case_mean": population_zero,
        "corrected_sse_case_mean": population_corrected,
        "target_rms": target_rms,
        "correction_rms": float(np.sqrt(max(population_correction_square, 0.0))),
        "correction_centered_rms": float(np.sqrt(max(correction_centered, 0.0))),
        "skill_vs_zero": skill,
        "rms_ratio_vs_zero": (
            float(np.sqrt(population_corrected / population_zero))
            if skill is not None
            else None
        ),
        "centered_r2": centered_r2,
        "median_case_cosine": (
            float(np.median(resolved_cosines))
            if len(resolved_cosines) == count
            else None
        ),
        "median_case_correlation": (
            float(np.median(resolved_correlations))
            if len(resolved_correlations) == count
            else None
        ),
        "skill_status": "ok" if skill is not None else "small_target",
        "centered_r2_status": (
            "ok" if centered_r2 is not None else "small_centered_target"
        ),
        "signed_metrics_status": (
            "ok"
            if len(resolved_cosines) == count and len(resolved_correlations) == count
            else "unresolved_case_denominator"
        ),
        "case_scores": case_rows,
        "centered_cross_case_mean": centered_cross,
    }


def corrections_for_coefficient(
    records: Sequence[ScalarFitRecord],
    coefficient: float,
    *,
    maximum_relative_norm: float = MAXIMUM_RELATIVE_CORRECTION,
) -> tuple[list[ScoreRecord], list[dict[str, Any]]]:
    """Build target-free capped corrections for a frozen scalar."""

    scores = []
    audits = []
    for record in sorted(records, key=lambda row: (row.case_id, int(row.input_call))):
        _validate_fit_record(record)
        correction, audit = apply_capped_correction(
            record.native_increment,
            record.feature,
            coefficient=coefficient,
            volumes=record.volumes,
            component_scale=record.component_scale,
            maximum_relative_norm=maximum_relative_norm,
        )
        scores.append(
            ScoreRecord(
                case_id=record.case_id,
                input_call=int(record.input_call),
                target_correction=record.target_correction,
                correction=correction,
                volumes=record.volumes,
                component_scale=record.component_scale,
            )
        )
        audits.append(
            {
                "case_id": record.case_id,
                "input_call": int(record.input_call),
                **asdict(audit),
            }
        )
    return scores, audits


def crossfit_case_groups(
    records: Sequence[ScalarFitRecord],
    *,
    expected_groups: Mapping[str, Sequence[str]],
    expected_input_calls: Sequence[int],
) -> dict[str, Any]:
    """Fit one scalar with each declared case group held out."""

    if len(expected_groups) < 2:
        raise ValueError("cross-fitting requires at least two groups")
    case_to_group: dict[str, str] = {}
    canonical_groups: dict[str, tuple[str, ...]] = {}
    for group_id, members in expected_groups.items():
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("group IDs must be nonempty strings")
        cases = tuple(members)
        if not cases or any(
            not isinstance(case_id, str) or not case_id for case_id in cases
        ):
            raise ValueError("groups must contain nonempty case IDs")
        if len(set(cases)) != len(cases):
            raise ValueError("a group contains duplicate case IDs")
        for case_id in cases:
            if case_id in case_to_group:
                raise ValueError("a case appears in more than one group")
            case_to_group[case_id] = group_id
        canonical_groups[group_id] = tuple(sorted(cases))
    expected_cases = tuple(sorted(case_to_group))
    validate_record_inventory(
        records,
        expected_case_ids=expected_cases,
        expected_input_calls=expected_input_calls,
    )
    ordered = tuple(sorted(records, key=lambda row: (row.case_id, int(row.input_call))))
    reference_volumes = np.asarray(ordered[0].volumes)
    reference_scale = np.asarray(ordered[0].component_scale)
    if any(
        not np.array_equal(np.asarray(record.volumes), reference_volumes)
        for record in ordered[1:]
    ):
        raise ValueError("cross-fit records must share native volumes")
    if any(
        not np.array_equal(np.asarray(record.component_scale), reference_scale)
        for record in ordered[1:]
    ):
        raise ValueError("cross-fit records must share component scales")
    folds = []
    all_oof_records: list[ScoreRecord] = []
    for held_out_group in sorted(canonical_groups):
        train = [row for row in ordered if case_to_group[row.case_id] != held_out_group]
        held_out = [
            row for row in ordered if case_to_group[row.case_id] == held_out_group
        ]
        fit = fit_case_first_scalar(train)
        fold_scores = None
        audits: list[dict[str, Any]] = []
        score_records_for_fold: list[ScoreRecord] = []
        if fit.status == "ok" and fit.coefficient is not None:
            score_records_for_fold, audits = corrections_for_coefficient(
                held_out, fit.coefficient
            )
            fold_scores = score_records(score_records_for_fold)
            all_oof_records.extend(score_records_for_fold)
        folds.append(
            {
                "held_out_group": held_out_group,
                "fit_case_ids": sorted({row.case_id for row in train}),
                "held_out_case_ids": sorted({row.case_id for row in held_out}),
                "fit": asdict(fit),
                "score": fold_scores,
                "audit_statuses": sorted({row["status"] for row in audits}),
            }
        )
    oof_score = (
        score_records(all_oof_records) if len(all_oof_records) == len(ordered) else None
    )
    coefficients = [
        float(fold["fit"]["coefficient"])
        for fold in folds
        if fold["fit"]["status"] == "ok" and fold["fit"]["coefficient"] is not None
    ]
    relative_iqr = None
    if len(coefficients) == len(folds):
        q25, q75 = np.percentile(np.asarray(coefficients, dtype=np.float64), (25, 75))
        median = float(np.median(coefficients))
        relative_iqr = float((q75 - q25) / max(abs(median), DENOMINATOR_FLOOR))
    return {
        "expected_groups": {
            group_id: list(canonical_groups[group_id])
            for group_id in sorted(canonical_groups)
        },
        "expected_input_calls": sorted(int(call) for call in expected_input_calls),
        "folds": folds,
        "oof_score": oof_score,
        "fold_coefficients": coefficients,
        "relative_iqr": relative_iqr,
        "status": (
            "ok"
            if len(coefficients) == len(folds) and oof_score is not None
            else "failed"
        ),
    }


def relation_from_fit_records(records: Sequence[ScalarFitRecord]) -> dict[str, Any]:
    """Offline signed feature/target relation with an oracle scalar diagnostic."""

    fit = fit_case_first_scalar(records)
    unit_records = [
        ScoreRecord(
            case_id=record.case_id,
            input_call=int(record.input_call),
            target_correction=record.target_correction,
            correction=record.feature,
            volumes=record.volumes,
            component_scale=record.component_scale,
        )
        for record in records
    ]
    unit_score = score_records(unit_records)
    oracle_score = None
    if fit.status == "ok" and fit.coefficient is not None:
        oracle_records = [
            ScoreRecord(
                case_id=record.case_id,
                input_call=int(record.input_call),
                target_correction=record.target_correction,
                correction=float(fit.coefficient) * record.feature,
                volumes=record.volumes,
                component_scale=record.component_scale,
            )
            for record in records
        ]
        oracle_score = score_records(oracle_records)
    return {
        "fit": asdict(fit),
        "unit_feature_score": unit_score,
        "uncapped_oracle_score": oracle_score,
    }
