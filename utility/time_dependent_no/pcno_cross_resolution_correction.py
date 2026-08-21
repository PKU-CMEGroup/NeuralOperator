"""Synthetic-tested W26-L5 cross-resolution correction primitives.

This A1 utility has no checkpoint or dataset loader.  It binds the nested-map
algebra, mapped predicted-increment features, scalar through-origin fits,
grouped cross-fitting, and one-native-state recurrence bookkeeping needed
before a separately authorized A2 evaluator is added.

Reference error is used only as an offline calibration/scoring label.  The two
deployable features contain predicted increments only.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

from utility.time_dependent_no.pcno_resolution_transfer import (
    Resolution,
    as_model_state,
    restrict_nested_state,
)
from utility.time_dependent_no.pcno_scale_separated_drift import (
    project_dct_bands,
)
from utility.time_dependent_no.shock_vortex_coarse_cfd import (
    prolong_piecewise_constant,
)

DENOMINATOR_FLOOR = 1.0e-8
MODEL_NAMES = ("zero", "coarse_only", "fine_only", "two_term")
ModelName = Literal["zero", "coarse_only", "fine_only", "two_term"]


@dataclass(frozen=True)
class ResolutionContract:
    """Three nested uniform Cartesian cell-average grids."""

    coarse: Resolution
    native: Resolution
    fine: Resolution

    def __post_init__(self) -> None:
        coarse_nx, coarse_ny = self.coarse
        native_nx, native_ny = self.native
        fine_nx, fine_ny = self.fine
        values = (*self.coarse, *self.native, *self.fine)
        if any(
            isinstance(value, bool) or int(value) != value or value < 1
            for value in values
        ):
            raise ValueError("resolution dimensions must be positive integers")
        if not (coarse_nx < native_nx < fine_nx and coarse_ny < native_ny < fine_ny):
            raise ValueError("coarse, native, and fine roles must be strictly ordered")
        if native_nx % coarse_nx or native_ny % coarse_ny:
            raise ValueError("coarse resolution must divide native resolution")
        if fine_nx % native_nx or fine_ny % native_ny:
            raise ValueError("native resolution must divide fine resolution")


@dataclass(frozen=True)
class NativeIncrementBasis:
    """Predicted increments and the two registered native-grid features."""

    native_increment: np.ndarray
    coarse_on_native: np.ndarray
    fine_on_native: np.ndarray
    native_minus_coarse: np.ndarray
    fine_minus_native: np.ndarray


@dataclass(frozen=True)
class PreparedCommonNativeInputs:
    """Pre-model and model-boundary views derived from one native state."""

    native_physical_state: np.ndarray
    pre_model_inputs: Mapping[Resolution, np.ndarray]
    model_inputs: Mapping[Resolution, np.ndarray]
    nesting_floors: Mapping[str, float]


@dataclass(frozen=True)
class TransferFloors:
    """D074-B direction-specific query/native transfer fields."""

    query_round_trip: np.ndarray
    native_information_mismatch: np.ndarray
    pipeline_truth_floor: np.ndarray
    mapped_information_mismatch: np.ndarray
    closure_residual: np.ndarray


@dataclass(frozen=True)
class ScalarCorrectionStatistics:
    """Additive sufficient statistics for two scalar correction features."""

    snapshot_count: int
    value_count: int
    weight_sum: float
    feature_sum: np.ndarray
    target_sum: float
    gram: np.ndarray
    cross: np.ndarray
    target_square: float


@dataclass(frozen=True)
class ScalarCorrectionFit:
    model: ModelName
    coefficients: tuple[float, float] | None
    active_dimension: int
    rank: int
    gram_eigenvalues: tuple[float, ...]
    condition_number: float | None
    feature_rms: tuple[float, ...]
    status: str


@dataclass(frozen=True)
class ScalarCorrectionScore:
    zero_sse: float
    corrected_sse: float
    target_rms: float
    correction_rms: float
    target_centered_rms: float
    correction_centered_rms: float
    cosine_denominator: float
    correlation_denominator: float
    skill_vs_zero: float | None
    rms_ratio_vs_zero: float | None
    centered_r2: float | None
    cosine: float | None
    correlation: float | None
    skill_status: str
    centered_r2_status: str
    cosine_status: str
    correlation_status: str
    status: str


@dataclass(frozen=True)
class DiagnosticSnapshot:
    """One case/call field record; nodes are never statistical samples."""

    case_id: str
    group_id: str
    input_call: int
    basis: NativeIncrementBasis
    target_correction: np.ndarray
    volumes: np.ndarray
    component_scale: np.ndarray
    masks: Mapping[str, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class SynchronizedFusionStep:
    """One truth-free step that retains only a native recurrent state."""

    prepared_inputs: PreparedCommonNativeInputs
    predictions: Mapping[Resolution, np.ndarray]
    basis: NativeIncrementBasis
    next_native_state: np.ndarray


def _resolution_size(resolution: Resolution) -> int:
    return int(resolution[0]) * int(resolution[1])


def _field(
    value: np.ndarray,
    *,
    resolution: Resolution,
    name: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    expected = (_resolution_size(resolution), 4)
    if array.shape != expected:
        raise ValueError(f"{name} must have shape {expected}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def prolong_nested_state(
    state: np.ndarray,
    *,
    coarse_resolution: Resolution,
    fine_resolution: Resolution,
) -> np.ndarray:
    """Float64 piecewise-constant injection for nested cell averages."""

    return prolong_piecewise_constant(
        np.asarray(state, dtype=np.float64),
        target_nx=int(fine_resolution[0]),
        target_ny=int(fine_resolution[1]),
        coarse_nx=int(coarse_resolution[0]),
        coarse_ny=int(coarse_resolution[1]),
    )


def common_native_representations(
    native_state: np.ndarray,
    *,
    contract: ResolutionContract,
) -> dict[Resolution, np.ndarray]:
    """Derive deployable coarse/native/fine views of one native physical state."""

    native = _field(native_state, resolution=contract.native, name="native_state")
    coarse = restrict_nested_state(
        native,
        fine_resolution=contract.native,
        coarse_resolution=contract.coarse,
    )
    fine = prolong_nested_state(
        native,
        coarse_resolution=contract.native,
        fine_resolution=contract.fine,
    )
    return {
        contract.coarse: np.asarray(coarse, dtype=np.float64),
        contract.native: np.array(native, copy=True),
        contract.fine: np.asarray(fine, dtype=np.float64),
    }


def _maximum_absolute(value: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(value, dtype=np.float64))))


def prepare_common_native_inputs(
    native_state: np.ndarray,
    *,
    contract: ResolutionContract,
) -> PreparedCommonNativeInputs:
    """Prepare all model inputs from one state and record FP32 nesting floors."""

    pre_model = common_native_representations(native_state, contract=contract)
    model_inputs = {
        resolution: as_model_state(state) for resolution, state in pre_model.items()
    }
    pre_coarse_gap = pre_model[contract.coarse] - restrict_nested_state(
        pre_model[contract.native],
        fine_resolution=contract.native,
        coarse_resolution=contract.coarse,
    )
    pre_fine_gap = (
        restrict_nested_state(
            pre_model[contract.fine],
            fine_resolution=contract.fine,
            coarse_resolution=contract.native,
        )
        - pre_model[contract.native]
    )
    post_coarse_gap = model_inputs[contract.coarse] - restrict_nested_state(
        model_inputs[contract.native],
        fine_resolution=contract.native,
        coarse_resolution=contract.coarse,
    )
    post_fine_gap = (
        restrict_nested_state(
            model_inputs[contract.fine],
            fine_resolution=contract.fine,
            coarse_resolution=contract.native,
        )
        - model_inputs[contract.native]
    )
    nesting_floors = {
        "pre_model_coarse_from_native_max_abs": _maximum_absolute(pre_coarse_gap),
        "pre_model_fine_to_native_max_abs": _maximum_absolute(pre_fine_gap),
        "post_fp32_coarse_from_native_max_abs": _maximum_absolute(post_coarse_gap),
        "post_fp32_fine_to_native_max_abs": _maximum_absolute(post_fine_gap),
        **{
            f"model_rounding_{resolution[0]}x{resolution[1]}_max_abs": (
                _maximum_absolute(model_inputs[resolution] - pre_model[resolution])
            )
            for resolution in (contract.coarse, contract.native, contract.fine)
        },
    }
    return PreparedCommonNativeInputs(
        native_physical_state=np.array(native_state, dtype=np.float64, copy=True),
        pre_model_inputs=pre_model,
        model_inputs=model_inputs,
        nesting_floors=nesting_floors,
    )


def _native_increment_basis(
    *,
    coarse_current: np.ndarray,
    coarse_prediction: np.ndarray,
    native_current: np.ndarray,
    native_prediction: np.ndarray,
    fine_current: np.ndarray,
    fine_prediction: np.ndarray,
    contract: ResolutionContract,
) -> NativeIncrementBasis:
    """Map increments, never full predictions, before forming the two features."""

    coarse_now = _field(
        coarse_current, resolution=contract.coarse, name="coarse_current"
    )
    coarse_next = _field(
        coarse_prediction,
        resolution=contract.coarse,
        name="coarse_prediction",
    )
    native_now = _field(
        native_current, resolution=contract.native, name="native_current"
    )
    native_next = _field(
        native_prediction,
        resolution=contract.native,
        name="native_prediction",
    )
    fine_now = _field(fine_current, resolution=contract.fine, name="fine_current")
    fine_next = _field(
        fine_prediction, resolution=contract.fine, name="fine_prediction"
    )

    coarse_increment = coarse_next - coarse_now
    native_increment = native_next - native_now
    fine_increment = fine_next - fine_now
    coarse_on_native = prolong_nested_state(
        coarse_increment,
        coarse_resolution=contract.coarse,
        fine_resolution=contract.native,
    )
    fine_on_native = restrict_nested_state(
        fine_increment,
        fine_resolution=contract.fine,
        coarse_resolution=contract.native,
    )
    return NativeIncrementBasis(
        native_increment=np.asarray(native_increment, dtype=np.float64),
        coarse_on_native=np.asarray(coarse_on_native, dtype=np.float64),
        fine_on_native=np.asarray(fine_on_native, dtype=np.float64),
        native_minus_coarse=np.asarray(
            native_increment - coarse_on_native, dtype=np.float64
        ),
        fine_minus_native=np.asarray(
            fine_on_native - native_increment, dtype=np.float64
        ),
    )


def common_native_increment_basis(
    prepared: PreparedCommonNativeInputs,
    predictions: Mapping[Resolution, np.ndarray],
    *,
    contract: ResolutionContract,
) -> NativeIncrementBasis:
    """Form deployable features only after verifying one-native-state inputs."""

    expected = prepare_common_native_inputs(
        prepared.native_physical_state,
        contract=contract,
    )
    expected_keys = {contract.coarse, contract.native, contract.fine}
    if set(prepared.pre_model_inputs) != expected_keys:
        raise ValueError("prepared pre-model input inventory is not exact")
    if set(prepared.model_inputs) != expected_keys:
        raise ValueError("prepared model-input inventory is not exact")
    if set(predictions) != expected_keys:
        raise ValueError("prediction inventory must contain exactly three grids")
    if set(prepared.nesting_floors) != set(expected.nesting_floors):
        raise ValueError("prepared nesting-floor inventory is not exact")
    for resolution in expected_keys:
        if not np.array_equal(
            prepared.pre_model_inputs[resolution],
            expected.pre_model_inputs[resolution],
        ):
            raise ValueError("pre-model inputs are not derived from one native state")
        if not np.array_equal(
            prepared.model_inputs[resolution],
            expected.model_inputs[resolution],
        ):
            raise ValueError("model inputs violate the common-native FP32 contract")
    if any(
        prepared.nesting_floors[name] != expected.nesting_floors[name]
        for name in expected.nesting_floors
    ):
        raise ValueError("prepared nesting floors do not match the model inputs")
    return _native_increment_basis(
        coarse_current=prepared.model_inputs[contract.coarse],
        coarse_prediction=predictions[contract.coarse],
        native_current=prepared.model_inputs[contract.native],
        native_prediction=predictions[contract.native],
        fine_current=prepared.model_inputs[contract.fine],
        fine_prediction=predictions[contract.fine],
        contract=contract,
    )


def corrected_native_prediction(
    native_prediction: np.ndarray,
    basis: NativeIncrementBasis,
    *,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """Correct a raw prediction; zero returns a bitwise copy of that prediction."""

    raw = np.asarray(native_prediction)
    if not np.isfinite(alpha) or not np.isfinite(beta):
        raise ValueError("correction coefficients must be finite")
    if alpha == 0.0 and beta == 0.0:
        return np.array(raw, copy=True)
    return np.asarray(
        raw
        + float(alpha) * basis.native_minus_coarse
        + float(beta) * basis.fine_minus_native,
        dtype=np.float64,
    )


def _map_query_to_native(
    value: np.ndarray,
    *,
    query_resolution: Resolution,
    native_resolution: Resolution,
) -> np.ndarray:
    if query_resolution == native_resolution:
        return np.array(value, dtype=np.float64, copy=True)
    if (
        query_resolution[0] > native_resolution[0]
        and query_resolution[1] > native_resolution[1]
    ):
        return restrict_nested_state(
            np.asarray(value, dtype=np.float64),
            fine_resolution=query_resolution,
            coarse_resolution=native_resolution,
        )
    if (
        query_resolution[0] < native_resolution[0]
        and query_resolution[1] < native_resolution[1]
    ):
        return prolong_nested_state(
            value,
            coarse_resolution=query_resolution,
            fine_resolution=native_resolution,
        )
    raise ValueError("query and native grids must be nested in the same direction")


def _map_native_to_query(
    value: np.ndarray,
    *,
    native_resolution: Resolution,
    query_resolution: Resolution,
) -> np.ndarray:
    if query_resolution == native_resolution:
        return np.array(value, dtype=np.float64, copy=True)
    if (
        query_resolution[0] > native_resolution[0]
        and query_resolution[1] > native_resolution[1]
    ):
        return prolong_nested_state(
            value,
            coarse_resolution=native_resolution,
            fine_resolution=query_resolution,
        )
    if (
        query_resolution[0] < native_resolution[0]
        and query_resolution[1] < native_resolution[1]
    ):
        return restrict_nested_state(
            np.asarray(value, dtype=np.float64),
            fine_resolution=native_resolution,
            coarse_resolution=query_resolution,
        )
    raise ValueError("query and native grids must be nested in the same direction")


def transfer_floor_fields(
    query_state: np.ndarray,
    native_state: np.ndarray,
    *,
    query_resolution: Resolution,
    native_resolution: Resolution,
) -> TransferFloors:
    """Return D074-B fields and exact ``o = S m + p`` closure."""

    query = _field(query_state, resolution=query_resolution, name="query_state")
    native = _field(native_state, resolution=native_resolution, name="native_state")
    query_on_native = _map_query_to_native(
        query,
        query_resolution=query_resolution,
        native_resolution=native_resolution,
    )
    native_on_query = _map_native_to_query(
        native,
        native_resolution=native_resolution,
        query_resolution=query_resolution,
    )
    round_trip = _map_native_to_query(
        query_on_native,
        native_resolution=native_resolution,
        query_resolution=query_resolution,
    )
    information_mismatch = query_on_native - native
    mapped_mismatch = _map_native_to_query(
        information_mismatch,
        native_resolution=native_resolution,
        query_resolution=query_resolution,
    )
    query_round_trip = round_trip - query
    pipeline_floor = native_on_query - query
    closure = query_round_trip - (mapped_mismatch + pipeline_floor)
    return TransferFloors(
        query_round_trip=query_round_trip,
        native_information_mismatch=information_mismatch,
        pipeline_truth_floor=pipeline_floor,
        mapped_information_mismatch=mapped_mismatch,
        closure_residual=closure,
    )


def mapped_increment_errors(
    basis: NativeIncrementBasis,
    *,
    native_reference_increment: np.ndarray,
    contract: ResolutionContract,
) -> dict[str, np.ndarray]:
    """Compare every mapped prediction with one common native reference increment."""

    native_reference = _field(
        native_reference_increment,
        resolution=contract.native,
        name="native_reference_increment",
    )
    return {
        "coarse_on_native": basis.coarse_on_native - native_reference,
        "native": basis.native_increment - native_reference,
        "fine_on_native": basis.fine_on_native - native_reference,
    }


def _metric_arrays(
    feature_a: np.ndarray,
    feature_b: np.ndarray,
    target: np.ndarray,
    *,
    volumes: np.ndarray,
    component_scale: np.ndarray,
    node_mask: np.ndarray | None,
    components: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left = np.asarray(feature_a, dtype=np.float64)
    right = np.asarray(feature_b, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if left.shape != right.shape or left.shape != truth.shape:
        raise ValueError("features and target must have identical shapes")
    if left.ndim != 2 or left.shape[1] != 4 or not np.isfinite(left).all():
        raise ValueError("features and target must be finite [nodes, 4] fields")
    if not np.isfinite(right).all() or not np.isfinite(truth).all():
        raise ValueError("features and target must be finite")

    mass = np.asarray(volumes, dtype=np.float64).reshape(-1)
    scale = np.asarray(component_scale, dtype=np.float64).reshape(-1)
    if mass.shape != (left.shape[0],) or np.any(mass <= 0.0):
        raise ValueError("volumes must be positive and align with nodes")
    if scale.shape != (4,) or np.any(scale <= 0.0) or not np.isfinite(scale).all():
        raise ValueError("component_scale must contain four positive values")
    selected = np.ones(left.shape[0], dtype=bool)
    if node_mask is not None:
        selected = np.asarray(node_mask)
        if selected.shape != (left.shape[0],) or selected.dtype != np.bool_:
            raise ValueError("node_mask must be a boolean vector")
    if not bool(selected.any()):
        raise ValueError("node_mask selects no nodes")

    component_indices = tuple(int(value) for value in components)
    if (
        not component_indices
        or len(set(component_indices)) != len(component_indices)
        or any(value < 0 or value >= 4 for value in component_indices)
    ):
        raise ValueError("components must be unique indices in [0, 3]")
    selected_mass = mass[selected]
    selected_mass = selected_mass / float(selected_mass.sum())
    weights = np.repeat(
        selected_mass[:, None] / len(component_indices),
        len(component_indices),
        axis=1,
    ).reshape(-1)

    def scaled_flat(value: np.ndarray) -> np.ndarray:
        return (
            value[selected][:, component_indices]
            / scale[np.asarray(component_indices)][None, :]
        ).reshape(-1)

    features = np.stack((scaled_flat(left), scaled_flat(right)), axis=1)
    return features, scaled_flat(truth), weights


def scalar_correction_statistics(
    basis: NativeIncrementBasis,
    target_correction: np.ndarray,
    *,
    volumes: np.ndarray,
    component_scale: np.ndarray,
    node_mask: np.ndarray | None = None,
    components: Sequence[int] = (0, 1, 2, 3),
) -> ScalarCorrectionStatistics:
    """Build one equally weighted case-time snapshot of sufficient statistics."""

    features, target, weights = _metric_arrays(
        basis.native_minus_coarse,
        basis.fine_minus_native,
        target_correction,
        volumes=volumes,
        component_scale=component_scale,
        node_mask=node_mask,
        components=components,
    )
    weighted_features = features * weights[:, None]
    return ScalarCorrectionStatistics(
        snapshot_count=1,
        value_count=int(target.size),
        weight_sum=float(weights.sum()),
        feature_sum=np.einsum("v,vk->k", weights, features, optimize=True),
        target_sum=float(np.dot(weights, target)),
        gram=features.T @ weighted_features,
        cross=np.einsum("vk,v,v->k", features, target, weights, optimize=True),
        target_square=float(np.dot(weights, np.square(target))),
    )


def sum_scalar_correction_statistics(
    values: Sequence[ScalarCorrectionStatistics],
) -> ScalarCorrectionStatistics:
    if not values:
        raise ValueError("at least one statistics record is required")
    return ScalarCorrectionStatistics(
        snapshot_count=sum(int(value.snapshot_count) for value in values),
        value_count=sum(int(value.value_count) for value in values),
        weight_sum=float(sum(float(value.weight_sum) for value in values)),
        feature_sum=np.sum([value.feature_sum for value in values], axis=0),
        target_sum=float(sum(float(value.target_sum) for value in values)),
        gram=np.sum([value.gram for value in values], axis=0),
        cross=np.sum([value.cross for value in values], axis=0),
        target_square=float(sum(float(value.target_square) for value in values)),
    )


def _scale_statistics(
    value: ScalarCorrectionStatistics,
    factor: float,
) -> ScalarCorrectionStatistics:
    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError("statistics scale factor must be positive and finite")
    return ScalarCorrectionStatistics(
        snapshot_count=value.snapshot_count,
        value_count=value.value_count,
        weight_sum=float(value.weight_sum * factor),
        feature_sum=np.asarray(value.feature_sum * factor, dtype=np.float64),
        target_sum=float(value.target_sum * factor),
        gram=np.asarray(value.gram * factor, dtype=np.float64),
        cross=np.asarray(value.cross * factor, dtype=np.float64),
        target_square=float(value.target_square * factor),
    )


def case_first_statistics(
    rows: Sequence[tuple[str, ScalarCorrectionStatistics]],
) -> ScalarCorrectionStatistics:
    """Average calls within cases and cases within the population."""

    grouped: dict[str, list[ScalarCorrectionStatistics]] = {}
    for case_id, statistics in rows:
        grouped.setdefault(str(case_id), []).append(statistics)
    if not grouped:
        raise ValueError("case-first aggregation requires at least one case")
    case_means = []
    for case_id in sorted(grouped):
        case_values = grouped[case_id]
        case_means.append(
            _scale_statistics(
                sum_scalar_correction_statistics(case_values),
                1.0 / len(case_values),
            )
        )
    return _scale_statistics(
        sum_scalar_correction_statistics(case_means),
        1.0 / len(case_means),
    )


def _active_indices(model: ModelName) -> tuple[int, ...]:
    if model == "zero":
        return ()
    if model == "coarse_only":
        return (0,)
    if model == "fine_only":
        return (1,)
    if model == "two_term":
        return (0, 1)
    raise ValueError(f"unsupported correction model: {model}")


def fit_scalar_correction(
    statistics: ScalarCorrectionStatistics,
    *,
    model: ModelName,
    denominator_floor: float = DENOMINATOR_FLOOR,
    rcond: float = 1.0e-12,
) -> ScalarCorrectionFit:
    """Fit fixed no-intercept scalars with no ridge or pseudoinverse fallback."""

    active = _active_indices(model)
    if model == "zero":
        return ScalarCorrectionFit(
            model=model,
            coefficients=(0.0, 0.0),
            active_dimension=0,
            rank=0,
            gram_eigenvalues=(),
            condition_number=None,
            feature_rms=(),
            status="ok",
        )
    arrays = (
        statistics.feature_sum,
        statistics.gram,
        statistics.cross,
    )
    if (
        statistics.weight_sum <= 0.0
        or not np.isfinite(statistics.weight_sum)
        or not np.isfinite(statistics.target_square)
        or any(not np.isfinite(value).all() for value in arrays)
    ):
        return ScalarCorrectionFit(
            model=model,
            coefficients=None,
            active_dimension=len(active),
            rank=0,
            gram_eigenvalues=(),
            condition_number=None,
            feature_rms=(),
            status="nonfinite",
        )
    gram = statistics.gram[np.ix_(active, active)]
    cross = statistics.cross[np.asarray(active)]
    eigenvalues = np.linalg.eigvalsh(gram)
    feature_rms = np.sqrt(np.maximum(np.diag(gram) / statistics.weight_sum, 0.0))
    if np.any(feature_rms <= float(denominator_floor)):
        return ScalarCorrectionFit(
            model=model,
            coefficients=None,
            active_dimension=len(active),
            rank=int(np.linalg.matrix_rank(gram)),
            gram_eigenvalues=tuple(float(value) for value in eigenvalues),
            condition_number=None,
            feature_rms=tuple(float(value) for value in feature_rms),
            status="zero_denominator",
        )
    maximum = float(eigenvalues[-1])
    tolerance = float(rcond) * maximum
    rank = int(np.count_nonzero(eigenvalues > tolerance))
    if rank != len(active):
        return ScalarCorrectionFit(
            model=model,
            coefficients=None,
            active_dimension=len(active),
            rank=rank,
            gram_eigenvalues=tuple(float(value) for value in eigenvalues),
            condition_number=None,
            feature_rms=tuple(float(value) for value in feature_rms),
            status="rank_deficient",
        )
    condition_number = float(eigenvalues[-1] / eigenvalues[0])
    solved = np.linalg.solve(gram, cross)
    coefficients = np.zeros(2, dtype=np.float64)
    coefficients[np.asarray(active)] = solved
    return ScalarCorrectionFit(
        model=model,
        coefficients=(float(coefficients[0]), float(coefficients[1])),
        active_dimension=len(active),
        rank=rank,
        gram_eigenvalues=tuple(float(value) for value in eigenvalues),
        condition_number=condition_number,
        feature_rms=tuple(float(value) for value in feature_rms),
        status="ok",
    )


def score_scalar_correction(
    statistics: ScalarCorrectionStatistics,
    coefficients: tuple[float, float],
    *,
    denominator_floor: float = DENOMINATOR_FLOOR,
) -> ScalarCorrectionScore:
    """Score from sufficient statistics against the exact zero correction."""

    theta = np.asarray(coefficients, dtype=np.float64)
    if theta.shape != (2,) or not np.isfinite(theta).all():
        raise ValueError("coefficients must contain two finite scalars")
    correction_sum = float(np.dot(theta, statistics.feature_sum))
    correction_square = float(theta @ statistics.gram @ theta)
    correction_target = float(np.dot(theta, statistics.cross))
    zero_sse = float(statistics.target_square)
    corrected_sse = float(zero_sse - 2.0 * correction_target + correction_square)
    roundoff = (
        64.0
        * np.finfo(np.float64).eps
        * max(zero_sse, correction_square, abs(correction_target), 1.0)
    )
    if corrected_sse < 0.0 and abs(corrected_sse) <= roundoff:
        corrected_sse = 0.0
    centered_target = float(
        zero_sse - statistics.target_sum * statistics.target_sum / statistics.weight_sum
    )
    centered_correction = float(
        correction_square - correction_sum * correction_sum / statistics.weight_sum
    )
    centered_cross = float(
        correction_target
        - correction_sum * statistics.target_sum / statistics.weight_sum
    )
    for name, value in (
        ("centered target", centered_target),
        ("centered correction", centered_correction),
    ):
        if value < 0.0 and abs(value) <= roundoff:
            if name == "centered target":
                centered_target = 0.0
            else:
                centered_correction = 0.0
    normalized = float(statistics.weight_sum)
    target_rms = float(np.sqrt(max(zero_sse / normalized, 0.0)))
    correction_rms = float(np.sqrt(max(correction_square / normalized, 0.0)))
    target_centered_rms = float(np.sqrt(max(centered_target / normalized, 0.0)))
    correction_centered_rms = float(np.sqrt(max(centered_correction / normalized, 0.0)))
    cosine_denominator = float(np.sqrt(max(correction_square * zero_sse, 0.0)))
    correlation_denominator = float(
        np.sqrt(max(centered_correction * centered_target, 0.0))
    )
    if (
        corrected_sse < 0.0
        or centered_target < 0.0
        or centered_correction < 0.0
        or not all(
            np.isfinite(value)
            for value in (
                corrected_sse,
                target_rms,
                correction_rms,
                target_centered_rms,
                correction_centered_rms,
                cosine_denominator,
                correlation_denominator,
            )
        )
    ):
        return ScalarCorrectionScore(
            zero_sse=zero_sse,
            corrected_sse=corrected_sse,
            target_rms=target_rms,
            correction_rms=correction_rms,
            target_centered_rms=target_centered_rms,
            correction_centered_rms=correction_centered_rms,
            cosine_denominator=cosine_denominator,
            correlation_denominator=correlation_denominator,
            skill_vs_zero=None,
            rms_ratio_vs_zero=None,
            centered_r2=None,
            cosine=None,
            correlation=None,
            skill_status="nonfinite",
            centered_r2_status="nonfinite",
            cosine_status="nonfinite",
            correlation_status="nonfinite",
            status="nonfinite",
        )
    floor = float(denominator_floor)
    skill = None
    rms_ratio = None
    skill_status = "ok"
    if target_rms > floor:
        skill = float(1.0 - corrected_sse / zero_sse)
        rms_ratio = float(np.sqrt(corrected_sse / zero_sse))
    else:
        skill_status = "zero_target_denominator"
    cosine_value = (
        float(correction_target / cosine_denominator)
        if target_rms > floor and correction_rms > floor
        else None
    )
    cosine_status = (
        "ok" if cosine_value is not None else "zero_target_or_correction_denominator"
    )
    correlation = (
        float(centered_cross / correlation_denominator)
        if target_centered_rms > floor and correction_centered_rms > floor
        else None
    )
    correlation_status = (
        "ok"
        if correlation is not None
        else "zero_centered_target_or_correction_denominator"
    )
    centered_r2 = (
        float(1.0 - corrected_sse / centered_target)
        if target_centered_rms > floor
        else None
    )
    centered_r2_status = (
        "ok" if centered_r2 is not None else "zero_centered_target_denominator"
    )
    metric_statuses = (
        skill_status,
        centered_r2_status,
        cosine_status,
        correlation_status,
    )
    status = (
        "ok"
        if all(metric_status == "ok" for metric_status in metric_statuses)
        else "unresolved_metric_denominator"
    )
    return ScalarCorrectionScore(
        zero_sse=zero_sse,
        corrected_sse=corrected_sse,
        target_rms=target_rms,
        correction_rms=correction_rms,
        target_centered_rms=target_centered_rms,
        correction_centered_rms=correction_centered_rms,
        cosine_denominator=cosine_denominator,
        correlation_denominator=correlation_denominator,
        skill_vs_zero=skill,
        rms_ratio_vs_zero=rms_ratio,
        centered_r2=centered_r2,
        cosine=cosine_value,
        correlation=correlation,
        skill_status=skill_status,
        centered_r2_status=centered_r2_status,
        cosine_status=cosine_status,
        correlation_status=correlation_status,
        status=status,
    )


def snapshot_statistics(
    snapshot: DiagnosticSnapshot,
    *,
    resolution: Resolution,
    band: str | None = None,
    region: str | None = None,
    components: Sequence[int] = (0, 1, 2, 3),
) -> tuple[ScalarCorrectionStatistics, dict[str, float]]:
    """Build statistics for one frozen component/band/region view."""

    if _resolution_size(resolution) != snapshot.target_correction.shape[0]:
        raise ValueError("snapshot does not match the declared native resolution")
    if band is None:
        basis = snapshot.basis
        target = snapshot.target_correction
        closure = {
            "maximum_reconstruction_abs_residual_scaled": 0.0,
            "maximum_instantaneous_energy_relative_closure": 0.0,
        }
    else:
        stacked = np.stack(
            (
                snapshot.basis.native_minus_coarse,
                snapshot.basis.fine_minus_native,
                snapshot.target_correction,
            ),
            axis=0,
        )
        projected, closure = project_dct_bands(
            stacked,
            resolution=resolution,
            component_scale=snapshot.component_scale,
        )
        if band not in projected:
            raise ValueError(f"unknown DCT band: {band}")
        selected = projected[band]
        basis = NativeIncrementBasis(
            native_increment=np.zeros_like(selected[0]),
            coarse_on_native=np.zeros_like(selected[0]),
            fine_on_native=np.zeros_like(selected[0]),
            native_minus_coarse=selected[0],
            fine_minus_native=selected[1],
        )
        target = selected[2]
    node_mask = None
    if region is not None:
        if region not in snapshot.masks:
            raise ValueError(f"snapshot lacks required region mask: {region}")
        node_mask = snapshot.masks[region]
    return (
        scalar_correction_statistics(
            basis,
            target,
            volumes=snapshot.volumes,
            component_scale=snapshot.component_scale,
            node_mask=node_mask,
            components=components,
        ),
        closure,
    )


def grouped_crossfit(
    snapshots: Sequence[DiagnosticSnapshot],
    *,
    resolution: Resolution,
    expected_groups: Mapping[str, Sequence[str]],
    expected_input_calls: Sequence[int],
) -> dict[str, Any]:
    """Run the frozen large-band selector with one declared group held out."""

    if not snapshots:
        raise ValueError("cross-fitting requires snapshots")
    if not expected_groups or len(expected_groups) < 2:
        raise ValueError("expected_groups must declare at least two groups")
    expected_case_to_group: dict[str, str] = {}
    for group_id, members in expected_groups.items():
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("group IDs must be nonempty strings")
        if isinstance(members, (str, bytes)):
            raise TypeError("each expected group must contain a case sequence")
        group_cases = tuple(members)
        if not group_cases:
            raise ValueError(f"expected group {group_id!r} is empty")
        if any(not isinstance(case_id, str) or not case_id for case_id in group_cases):
            raise ValueError("case IDs must be nonempty strings")
        if len(set(group_cases)) != len(group_cases):
            raise ValueError(f"expected group {group_id!r} contains duplicate cases")
        for case_id in group_cases:
            if case_id in expected_case_to_group:
                raise ValueError(f"case {case_id!r} appears in multiple groups")
            expected_case_to_group[case_id] = group_id
    expected_calls = tuple(expected_input_calls)
    if not expected_calls:
        raise ValueError("expected_input_calls must not be empty")
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < 0
        for value in expected_calls
    ):
        raise ValueError("expected input calls must be nonnegative integers")
    expected_calls = tuple(sorted(int(value) for value in expected_calls))
    if len(set(expected_calls)) != len(expected_calls):
        raise ValueError("expected input calls must be unique")

    expected_pairs = {
        (case_id, input_call)
        for case_id in expected_case_to_group
        for input_call in expected_calls
    }
    actual_pairs: set[tuple[str, int]] = set()
    for snapshot in snapshots:
        if not isinstance(snapshot.case_id, str) or not snapshot.case_id:
            raise ValueError("snapshot case IDs must be nonempty strings")
        if not isinstance(snapshot.group_id, str) or not snapshot.group_id:
            raise ValueError("snapshot group IDs must be nonempty strings")
        if (
            isinstance(snapshot.input_call, (bool, np.bool_))
            or not isinstance(snapshot.input_call, (int, np.integer))
            or int(snapshot.input_call) < 0
        ):
            raise ValueError("snapshot input calls must be nonnegative integers")
        expected_group = expected_case_to_group.get(snapshot.case_id)
        if expected_group is None:
            raise ValueError(f"unexpected case ID: {snapshot.case_id!r}")
        if snapshot.group_id != expected_group:
            raise ValueError(
                f"case {snapshot.case_id!r} is assigned to the wrong group"
            )
        pair = (snapshot.case_id, int(snapshot.input_call))
        if pair in actual_pairs:
            raise ValueError(f"duplicate case/call snapshot: {pair!r}")
        actual_pairs.add(pair)
    missing_pairs = sorted(expected_pairs - actual_pairs)
    extra_pairs = sorted(actual_pairs - expected_pairs)
    if missing_pairs or extra_pairs:
        raise ValueError(
            "case/call inventory does not match the frozen design: "
            f"missing={missing_pairs}, extra={extra_pairs}"
        )

    ordered_snapshots = tuple(
        sorted(snapshots, key=lambda value: (value.case_id, int(value.input_call)))
    )
    reference_volumes = np.asarray(ordered_snapshots[0].volumes)
    reference_scales = np.asarray(ordered_snapshots[0].component_scale)
    if any(
        not np.array_equal(np.asarray(snapshot.volumes), reference_volumes)
        for snapshot in ordered_snapshots[1:]
    ):
        raise ValueError("all snapshots must use identical native volumes")
    if any(
        not np.array_equal(np.asarray(snapshot.component_scale), reference_scales)
        for snapshot in ordered_snapshots[1:]
    ):
        raise ValueError("all snapshots must use identical component scales")

    groups = sorted(expected_groups)
    case_ids = sorted(expected_case_to_group)
    statistics_by_snapshot: list[
        tuple[DiagnosticSnapshot, ScalarCorrectionStatistics]
    ] = []
    maximum_closure = 0.0
    for snapshot in ordered_snapshots:
        statistics, closure = snapshot_statistics(
            snapshot,
            resolution=resolution,
            band="large",
        )
        statistics_by_snapshot.append((snapshot, statistics))
        maximum_closure = max(
            maximum_closure,
            float(closure["maximum_reconstruction_abs_residual_scaled"]),
            float(closure["maximum_instantaneous_energy_relative_closure"]),
        )

    model_rows: dict[str, Any] = {}
    for model in MODEL_NAMES:
        folds = []
        weighted_zero = 0.0
        weighted_corrected = 0.0
        scored_cases = 0
        all_ok = True
        for held_out_group in groups:
            train_rows = [
                (snapshot.case_id, statistics)
                for snapshot, statistics in statistics_by_snapshot
                if snapshot.group_id != held_out_group
            ]
            held_out_rows = [
                (snapshot.case_id, statistics)
                for snapshot, statistics in statistics_by_snapshot
                if snapshot.group_id == held_out_group
            ]
            if not train_rows or not held_out_rows:
                raise AssertionError("group fold has an empty train or validation side")
            train_statistics = case_first_statistics(train_rows)
            held_out_statistics = case_first_statistics(held_out_rows)
            fit = fit_scalar_correction(train_statistics, model=model)
            held_out_cases = sorted({case_id for case_id, _ in held_out_rows})
            train_cases = sorted({case_id for case_id, _ in train_rows})
            score = None
            case_scores: list[dict[str, Any]] = []
            if fit.status == "ok" and fit.coefficients is not None:
                score = score_scalar_correction(
                    held_out_statistics,
                    fit.coefficients,
                )
                for held_out_case in held_out_cases:
                    case_statistics = case_first_statistics(
                        [
                            (case_id, statistics)
                            for case_id, statistics in held_out_rows
                            if case_id == held_out_case
                        ]
                    )
                    case_score = score_scalar_correction(
                        case_statistics,
                        fit.coefficients,
                    )
                    case_scores.append({"case_id": held_out_case, **asdict(case_score)})
                if score.skill_status == "ok":
                    case_weight = len(held_out_cases)
                    weighted_zero += case_weight * score.zero_sse
                    weighted_corrected += case_weight * score.corrected_sse
                    scored_cases += case_weight
                else:
                    all_ok = False
            else:
                all_ok = False
            resolved_cosines = [
                float(case_score["cosine"])
                for case_score in case_scores
                if case_score["cosine_status"] == "ok"
                and case_score["cosine"] is not None
            ]
            resolved_correlations = [
                float(case_score["correlation"])
                for case_score in case_scores
                if case_score["correlation_status"] == "ok"
                and case_score["correlation"] is not None
            ]
            folds.append(
                {
                    "held_out_group": held_out_group,
                    "held_out_case_ids": held_out_cases,
                    "fit_case_ids": train_cases,
                    "fit_status": fit.status,
                    "coefficients": fit.coefficients,
                    "condition_number": fit.condition_number,
                    "population_skill_status": (
                        None if score is None else score.skill_status
                    ),
                    "skill_vs_zero": None if score is None else score.skill_vs_zero,
                    "rms_ratio_vs_zero": (
                        None if score is None else score.rms_ratio_vs_zero
                    ),
                    "case_scores": case_scores,
                    "median_case_cosine": (
                        float(np.median(resolved_cosines))
                        if len(resolved_cosines) == len(held_out_cases)
                        else None
                    ),
                    "median_case_correlation": (
                        float(np.median(resolved_correlations))
                        if len(resolved_correlations) == len(held_out_cases)
                        else None
                    ),
                    "all_case_signed_metrics_resolved": (
                        len(resolved_cosines) == len(held_out_cases)
                        and len(resolved_correlations) == len(held_out_cases)
                    ),
                }
            )
        skill = None
        rms_ratio = None
        if all_ok and scored_cases == len(case_ids) and weighted_zero > 0.0:
            ratio = weighted_corrected / weighted_zero
            skill = float(1.0 - ratio)
            rms_ratio = float(np.sqrt(ratio))
        out_of_fold_case_scores = sorted(
            [case_score for fold in folds for case_score in fold["case_scores"]],
            key=lambda row: str(row["case_id"]),
        )
        all_oof_cosines = [
            float(row["cosine"])
            for row in out_of_fold_case_scores
            if row["cosine_status"] == "ok" and row["cosine"] is not None
        ]
        all_oof_correlations = [
            float(row["correlation"])
            for row in out_of_fold_case_scores
            if row["correlation_status"] == "ok" and row["correlation"] is not None
        ]
        model_rows[model] = {
            "status": "ok" if skill is not None else "failed",
            "skill_vs_zero": skill,
            "rms_ratio_vs_zero": rms_ratio,
            "zero_sse_case_sum": weighted_zero if scored_cases else None,
            "corrected_sse_case_sum": weighted_corrected if scored_cases else None,
            "scored_case_count": scored_cases,
            "out_of_fold_case_scores": out_of_fold_case_scores,
            "median_oof_case_cosine": (
                float(np.median(all_oof_cosines))
                if len(all_oof_cosines) == len(case_ids)
                else None
            ),
            "median_oof_case_correlation": (
                float(np.median(all_oof_correlations))
                if len(all_oof_correlations) == len(case_ids)
                else None
            ),
            "folds": folds,
        }

    eligible = [
        model
        for model in MODEL_NAMES
        if model_rows[model]["status"] == "ok"
        and model_rows[model]["rms_ratio_vs_zero"] is not None
    ]
    if not eligible:
        raise ValueError("no correction model completed every group fold")
    complexity = {"zero": 0, "coarse_only": 1, "fine_only": 1, "two_term": 2}
    selected = min(
        eligible,
        key=lambda model: (
            float(model_rows[model]["rms_ratio_vs_zero"]),
            complexity[model],
            MODEL_NAMES.index(model),
        ),
    )
    full_statistics = case_first_statistics(
        [
            (snapshot.case_id, statistics)
            for snapshot, statistics in statistics_by_snapshot
        ]
    )
    full_fit = fit_scalar_correction(full_statistics, model=selected)
    return {
        "selected_model": selected,
        "selected_coefficients": full_fit.coefficients,
        "selected_fit_status": full_fit.status,
        "selected_condition_number": full_fit.condition_number,
        "selected_full_fit": {
            **asdict(full_fit),
            "gram": full_statistics.gram.tolist(),
            "cross": full_statistics.cross.tolist(),
            "target_square": full_statistics.target_square,
            "target_rms": float(
                np.sqrt(full_statistics.target_square / full_statistics.weight_sum)
            ),
            "weight_sum": full_statistics.weight_sum,
        },
        "models": model_rows,
        "case_ids": case_ids,
        "groups": groups,
        "expected_groups": {
            group_id: sorted(expected_groups[group_id]) for group_id in groups
        },
        "expected_input_calls": list(expected_calls),
        "selection_band": "large",
        "maximum_band_closure": maximum_closure,
    }


def field_relation_metrics(
    feature: np.ndarray,
    target: np.ndarray,
    *,
    volumes: np.ndarray,
    component_scale: np.ndarray,
    node_mask: np.ndarray | None = None,
    components: Sequence[int] = (0, 1, 2, 3),
) -> dict[str, Any]:
    """Offline signed relation between one field and one true-error field."""

    zero = np.zeros_like(feature, dtype=np.float64)
    basis = NativeIncrementBasis(
        native_increment=zero,
        coarse_on_native=zero,
        fine_on_native=zero,
        native_minus_coarse=np.asarray(feature, dtype=np.float64),
        fine_minus_native=zero,
    )
    statistics = scalar_correction_statistics(
        basis,
        target,
        volumes=volumes,
        component_scale=component_scale,
        node_mask=node_mask,
        components=components,
    )
    fit = fit_scalar_correction(statistics, model="coarse_only")
    unit_score = score_scalar_correction(statistics, (1.0, 0.0))
    fitted_score = (
        score_scalar_correction(statistics, fit.coefficients)
        if fit.status == "ok" and fit.coefficients is not None
        else None
    )
    return {
        "feature_rms": unit_score.correction_rms,
        "target_rms": unit_score.target_rms,
        "signed_cosine": unit_score.cosine,
        "signed_cosine_denominator": unit_score.cosine_denominator,
        "signed_cosine_status": unit_score.cosine_status,
        "pearson": unit_score.correlation,
        "pearson_denominator": unit_score.correlation_denominator,
        "pearson_status": unit_score.correlation_status,
        "fitted_slope": (
            None if fit.coefficients is None else float(fit.coefficients[0])
        ),
        "fit_status": fit.status,
        "fitted_skill_vs_zero": (
            None if fitted_score is None else fitted_score.skill_vs_zero
        ),
        "fitted_skill_status": (
            None if fitted_score is None else fitted_score.skill_status
        ),
    }


def predicted_error_relationships(
    errors: Mapping[str, np.ndarray],
    *,
    volumes: np.ndarray,
    component_scale: np.ndarray,
) -> list[dict[str, Any]]:
    """Pairwise offline error-field relations; never deployable features."""

    required = ("coarse_on_native", "native", "fine_on_native")
    missing = [name for name in required if name not in errors]
    if missing:
        raise ValueError(f"mapped error fields are missing: {missing}")
    pairs = (
        ("coarse_on_native", "native"),
        ("fine_on_native", "native"),
        ("coarse_on_native", "fine_on_native"),
    )
    return [
        {
            "feature_error": left,
            "target_error": right,
            **field_relation_metrics(
                errors[left],
                errors[right],
                volumes=volumes,
                component_scale=component_scale,
            ),
            "offline_only": True,
        }
        for left, right in pairs
    ]


def synchronized_fusion_step(
    native_state: np.ndarray,
    *,
    contract: ResolutionContract,
    predictor: Callable[[Resolution, np.ndarray], np.ndarray],
    alpha: float,
    beta: float,
) -> SynchronizedFusionStep:
    """Call three grids from one state and update only the native prediction."""

    prepared = prepare_common_native_inputs(native_state, contract=contract)
    predictions: dict[Resolution, np.ndarray] = {}
    for resolution in (contract.coarse, contract.native, contract.fine):
        predictions[resolution] = _field(
            predictor(
                resolution,
                np.array(prepared.model_inputs[resolution], copy=True),
            ),
            resolution=resolution,
            name=f"prediction_{resolution[0]}x{resolution[1]}",
        )
    basis = common_native_increment_basis(
        prepared,
        predictions,
        contract=contract,
    )
    next_native = corrected_native_prediction(
        predictions[contract.native],
        basis,
        alpha=alpha,
        beta=beta,
    )
    return SynchronizedFusionStep(
        prepared_inputs=prepared,
        predictions=predictions,
        basis=basis,
        next_native_state=next_native,
    )
