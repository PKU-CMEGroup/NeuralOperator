"""Same-state exact/coast branch closure for W26-L5 A46.

Truth is intentionally absent from :func:`build_same_state_branch_pair`.
It enters only through :func:`score_same_state_branch_pair` after both branch
states and the frozen master update have been materialized.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np

from utility.time_dependent_no.pcno_resolution_transfer import weighted_scaled_rms

MasterRoute = Literal["exact", "coast"]
PreferredBranch = Literal["exact", "coast", "tie"]
LogicalCallRole = Literal[
    "shadow_native",
    "accepted_native",
    "accepted_fine",
]

PAIRED_CALL_ROLES: tuple[LogicalCallRole, ...] = (
    "shadow_native",
    "accepted_native",
    "accepted_fine",
)


@dataclass(frozen=True)
class BranchCandidate:
    """One already-built native candidate."""

    next_native_state: np.ndarray


@dataclass(frozen=True)
class SameStateBranchPair:
    """Exact and coast candidates built around one immutable state pair."""

    master_route: MasterRoute
    shared_shadow_prediction: np.ndarray
    exact_next_native_state: np.ndarray
    coast_next_native_state: np.ndarray
    master_next_native_state: np.ndarray
    next_shadow_native_state: np.ndarray
    logical_call_roles: tuple[LogicalCallRole, ...]
    accepted_input_sha256: str
    shadow_input_sha256: str
    maximum_input_mutation_abs: float
    maximum_master_identity_abs: float
    maximum_shadow_identity_abs: float


@dataclass(frozen=True)
class SameStateBranchUtility:
    """Physical-volume error and signed exact-refresh utility for one view."""

    exact_rms: float
    coast_rms: float
    master_rms: float
    exact_mse: float
    coast_mse: float
    master_mse: float
    signed_exact_refresh_utility: float
    relative_exact_refresh_utility: float | None
    relative_denominator: float
    relative_denominator_status: str
    preferred_branch: PreferredBranch
    tie_threshold: float


ModelPredictor: TypeAlias = Callable[
    [LogicalCallRole, np.ndarray],
    np.ndarray,
]
OrderedRolePredictor: TypeAlias = Callable[
    [LogicalCallRole, np.ndarray],
    np.ndarray,
]
CoastBranchBuilder: TypeAlias = Callable[
    [np.ndarray, np.ndarray, np.ndarray],
    BranchCandidate,
]
ExactBranchBuilder: TypeAlias = Callable[
    [np.ndarray, np.ndarray, np.ndarray, OrderedRolePredictor],
    BranchCandidate,
]


def _finite_matrix(
    value: np.ndarray,
    *,
    name: str,
    expected_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError(f"{name} must be a nonempty rank-2 array")
    if expected_shape is not None and array.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _readonly_copy(value: np.ndarray) -> np.ndarray:
    array = np.array(value, dtype=np.float64, order="C", copy=True)
    array.setflags(write=False)
    return array


def _maximum_abs(value: np.ndarray) -> float:
    array = np.asarray(value, dtype=np.float64)
    return 0.0 if array.size == 0 else float(np.max(np.abs(array)))


def array_sha256(value: np.ndarray) -> str:
    """Hash one finite rank-2 array in a canonical float64 representation."""

    array = np.ascontiguousarray(_finite_matrix(value, name="value"))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _validated_candidate(
    candidate: BranchCandidate,
    *,
    name: str,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    if not isinstance(candidate, BranchCandidate):
        raise TypeError(f"{name} builder must return BranchCandidate")
    return _readonly_copy(
        _finite_matrix(
            candidate.next_native_state,
            name=f"{name}_next_native_state",
            expected_shape=expected_shape,
        )
    )


class _OrderedModelCalls:
    """Fail-closed role order around the one predictor visible to A46."""

    def __init__(self, predictor: ModelPredictor):
        self._predictor = predictor
        self._roles: list[LogicalCallRole] = []

    @property
    def roles(self) -> tuple[LogicalCallRole, ...]:
        return tuple(self._roles)

    def __call__(
        self,
        role: LogicalCallRole,
        model_input: np.ndarray,
    ) -> np.ndarray:
        index = len(self._roles)
        if index >= len(PAIRED_CALL_ROLES):
            raise ValueError("logical model call inventory exceeded")
        expected = PAIRED_CALL_ROLES[index]
        if role != expected:
            raise ValueError(
                f"logical model call {index} must be {expected}, not {role}"
            )
        value = _finite_matrix(model_input, name=f"{role}_input")
        prediction = _finite_matrix(
            self._predictor(role, _readonly_copy(value)),
            name=f"{role}_prediction",
        )
        self._roles.append(role)
        return _readonly_copy(prediction)

    def require_complete(self) -> None:
        if self.roles != PAIRED_CALL_ROLES:
            raise ValueError(
                "logical model call sequence is incomplete: "
                f"expected {PAIRED_CALL_ROLES}, observed {self.roles}"
            )


def build_same_state_branch_pair(
    accepted_native_state: np.ndarray,
    shadow_native_state: np.ndarray,
    *,
    master_route: MasterRoute,
    model_predictor: ModelPredictor,
    exact_builder: ExactBranchBuilder,
    coast_builder: CoastBranchBuilder,
) -> SameStateBranchPair:
    """Build exact/coast probes from one state and retain only a frozen route.

    The orchestrator calls ``model_predictor`` for the shadow exactly once.
    Coast is built first and receives no predictor. The exact builder receives
    the ordered role wrapper and must then call accepted-native followed by
    accepted-fine. Builders receive separate read-only copies, so neither
    candidate can affect the other or the master input.
    """

    if master_route not in ("exact", "coast"):
        raise ValueError("master_route must be 'exact' or 'coast'")
    accepted = _finite_matrix(accepted_native_state, name="accepted_native_state")
    shadow = _finite_matrix(
        shadow_native_state,
        name="shadow_native_state",
        expected_shape=accepted.shape,
    )
    accepted_before = np.array(accepted, copy=True)
    shadow_before = np.array(shadow, copy=True)

    calls = _OrderedModelCalls(model_predictor)
    shared_shadow = _finite_matrix(
        calls("shadow_native", shadow_before),
        name="shared_shadow_prediction",
        expected_shape=accepted.shape,
    )
    shared_shadow = _readonly_copy(shared_shadow)

    def branch_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            _readonly_copy(accepted_before),
            _readonly_copy(shadow_before),
            _readonly_copy(shared_shadow),
        )

    coast = _validated_candidate(
        coast_builder(*branch_inputs()),
        name="coast",
        expected_shape=accepted.shape,
    )
    exact = _validated_candidate(
        exact_builder(*branch_inputs(), calls),
        name="exact",
        expected_shape=accepted.shape,
    )
    calls.require_complete()

    accepted_after = np.asarray(accepted_native_state, dtype=np.float64)
    shadow_after = np.asarray(shadow_native_state, dtype=np.float64)
    input_mutation = max(
        _maximum_abs(accepted_after - accepted_before),
        _maximum_abs(shadow_after - shadow_before),
    )
    if input_mutation != 0.0:
        raise ValueError("a branch mutated an input state")

    selected = exact if master_route == "exact" else coast
    master = _readonly_copy(selected)
    next_shadow = _readonly_copy(shared_shadow)
    return SameStateBranchPair(
        master_route=master_route,
        shared_shadow_prediction=_readonly_copy(shared_shadow),
        exact_next_native_state=exact,
        coast_next_native_state=coast,
        master_next_native_state=master,
        next_shadow_native_state=next_shadow,
        logical_call_roles=calls.roles,
        accepted_input_sha256=array_sha256(accepted_before),
        shadow_input_sha256=array_sha256(shadow_before),
        maximum_input_mutation_abs=input_mutation,
        maximum_master_identity_abs=_maximum_abs(master - selected),
        maximum_shadow_identity_abs=_maximum_abs(next_shadow - shared_shadow),
    )


def score_same_state_branch_pair(
    pair: SameStateBranchPair,
    reference_next_native_state: np.ndarray,
    *,
    volumes: np.ndarray,
    component_scale: Sequence[float] | np.ndarray,
    mask: np.ndarray | None = None,
    relative_denominator_floor: float = 1.0e-24,
    tie_relative_tolerance: float = 1.0e-12,
) -> SameStateBranchUtility:
    """Score a completed pair against real truth; positive utility favors exact."""

    if not isinstance(pair, SameStateBranchPair):
        raise TypeError("pair must be SameStateBranchPair")
    if not np.isfinite(relative_denominator_floor) or relative_denominator_floor <= 0:
        raise ValueError("relative_denominator_floor must be finite and positive")
    if not np.isfinite(tie_relative_tolerance) or tie_relative_tolerance < 0:
        raise ValueError("tie_relative_tolerance must be finite and nonnegative")

    exact = _finite_matrix(pair.exact_next_native_state, name="exact_next")
    reference = _finite_matrix(
        reference_next_native_state,
        name="reference_next_native_state",
        expected_shape=exact.shape,
    )
    coast = _finite_matrix(
        pair.coast_next_native_state,
        name="coast_next",
        expected_shape=exact.shape,
    )
    master = _finite_matrix(
        pair.master_next_native_state,
        name="master_next",
        expected_shape=exact.shape,
    )
    weights = np.asarray(volumes, dtype=np.float64).reshape(-1)
    if weights.shape != (exact.shape[0],) or not np.isfinite(weights).all():
        raise ValueError("volumes must be finite and align with the node axis")
    if np.any(weights <= 0.0):
        raise ValueError("volumes must be strictly positive")
    scale = np.asarray(component_scale, dtype=np.float64)
    if scale.shape != (exact.shape[1],) or not np.isfinite(scale).all():
        raise ValueError("component_scale must be finite and match components")
    if np.any(scale <= 0.0):
        raise ValueError("component_scale must be strictly positive")

    metric_kwargs = {
        "volumes": weights,
        "component_scale": scale,
        "mask": mask,
    }
    exact_rms = weighted_scaled_rms(exact - reference, **metric_kwargs)
    coast_rms = weighted_scaled_rms(coast - reference, **metric_kwargs)
    master_rms = weighted_scaled_rms(master - reference, **metric_kwargs)
    exact_mse = float(exact_rms * exact_rms)
    coast_mse = float(coast_rms * coast_rms)
    master_mse = float(master_rms * master_rms)
    utility = float(coast_mse - exact_mse)

    if coast_mse <= relative_denominator_floor:
        relative = None
        denominator_status = "unresolved_small_coast_mse"
    else:
        relative = float(utility / coast_mse)
        denominator_status = "resolved"

    tie_threshold = float(
        tie_relative_tolerance * max(exact_mse, coast_mse, relative_denominator_floor)
    )
    if utility > tie_threshold:
        preferred: PreferredBranch = "exact"
    elif utility < -tie_threshold:
        preferred = "coast"
    else:
        preferred = "tie"

    return SameStateBranchUtility(
        exact_rms=exact_rms,
        coast_rms=coast_rms,
        master_rms=master_rms,
        exact_mse=exact_mse,
        coast_mse=coast_mse,
        master_mse=master_mse,
        signed_exact_refresh_utility=utility,
        relative_exact_refresh_utility=relative,
        relative_denominator=coast_mse,
        relative_denominator_status=denominator_status,
        preferred_branch=preferred,
        tie_threshold=tie_threshold,
    )
