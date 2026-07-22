"""Frozen conservative-correction oracle for finite-volume flow maps.

The oracle asks a deliberately narrow question: can the error of a frozen
cell-state prediction be reduced by a small correction decoded from shared
interior-face impulses?  It is an analysis tool, not a learned model and not a
time-integration scheme.  Truth may be used to choose the allowed face set and
the least-squares direction, so its result is an upper-bound diagnostic rather
than an inference-time procedure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

AntiSmearingCheck = Callable[[np.ndarray], bool]


@dataclass(frozen=True)
class ConservativeCorrectionOracleResult:
    """Result of one constrained face-impulse oracle solve."""

    face_impulse: np.ndarray
    cell_correction: np.ndarray
    corrected_state: np.ndarray
    selected_alpha: float
    baseline_error: float
    corrected_error: float
    relative_error_reduction: float
    candidate_update_ratio: float
    applied_update_ratio: float
    allowed_face_fraction: float
    active_face_fraction: float
    conservation_balance_max_abs: float
    feasible_line_search_points: int
    anti_smearing_checked: bool
    lsqr_stop_codes: tuple[int, ...]

    def summary(self) -> dict[str, float | int | bool | list[int] | None]:
        """Return scalar diagnostics suitable for JSON metadata."""

        return {
            "selected_alpha": self.selected_alpha,
            "baseline_error": self.baseline_error,
            "corrected_error": self.corrected_error,
            "relative_error_reduction": self.relative_error_reduction,
            "candidate_update_ratio": (
                self.candidate_update_ratio
                if np.isfinite(self.candidate_update_ratio)
                else None
            ),
            "applied_update_ratio": self.applied_update_ratio,
            "allowed_face_fraction": self.allowed_face_fraction,
            "active_face_fraction": self.active_face_fraction,
            "conservation_balance_max_abs": self.conservation_balance_max_abs,
            "feasible_line_search_points": self.feasible_line_search_points,
            "anti_smearing_checked": self.anti_smearing_checked,
            "lsqr_stop_codes": list(self.lsqr_stop_codes),
        }


def decode_face_impulse_correction(
    face_impulse: np.ndarray,
    *,
    cell_volume: np.ndarray,
    face_owner: np.ndarray,
    face_neighbor: np.ndarray,
) -> np.ndarray:
    """Decode owner-oriented cumulative face impulses into cell increments.

    Interior impulses are subtracted from the owner and added to the neighbor.
    A boundary impulse (``neighbor == -1``) changes only its owner and therefore
    represents boundary exchange.  The constrained oracle below excludes such
    impulses so that its correction is globally conservative by construction.
    """

    impulse = np.asarray(face_impulse, dtype=np.float64)
    volume = np.asarray(cell_volume, dtype=np.float64)
    owner = np.asarray(face_owner, dtype=np.int64)
    neighbor = np.asarray(face_neighbor, dtype=np.int64)
    _validate_geometry(volume, owner, neighbor)
    if impulse.ndim != 2 or impulse.shape[0] != owner.size:
        raise ValueError("face_impulse must have shape [faces, variables]")
    if not np.all(np.isfinite(impulse)):
        raise ValueError("face_impulse must be finite")

    correction = np.zeros((volume.size, impulse.shape[1]), dtype=np.float64)
    np.add.at(correction, owner, -impulse / volume[owner, None])
    interior = neighbor >= 0
    np.add.at(
        correction,
        neighbor[interior],
        impulse[interior] / volume[neighbor[interior], None],
    )
    return correction


def euler2d_pressure(
    conservative: np.ndarray,
    *,
    gamma: float = 1.4,
) -> np.ndarray:
    """Return ideal-gas pressure for ``[rho, rho*u, rho*v, E]`` states."""

    state = np.asarray(conservative, dtype=np.float64)
    if state.ndim != 2 or state.shape[1] != 4:
        raise ValueError("conservative must have shape [cells, 4]")
    if gamma <= 1.0:
        raise ValueError("gamma must be greater than one")
    rho = state[:, 0]
    kinetic = 0.5 * (state[:, 1] ** 2 + state[:, 2] ** 2) / rho
    return (gamma - 1.0) * (state[:, 3] - kinetic)


def euler2d_state_is_admissible(
    conservative: np.ndarray,
    *,
    gamma: float = 1.4,
    density_floor: float = 0.0,
    pressure_floor: float = 0.0,
) -> bool:
    """Return whether every conservative state is finite and admissible."""

    state = np.asarray(conservative, dtype=np.float64)
    if state.ndim != 2 or state.shape[1] != 4 or not np.all(np.isfinite(state)):
        return False
    if density_floor < 0.0 or pressure_floor < 0.0:
        raise ValueError("admissibility floors must be nonnegative")
    if np.any(state[:, 0] <= density_floor):
        return False
    pressure = euler2d_pressure(state, gamma=gamma)
    return bool(np.all(np.isfinite(pressure)) and np.all(pressure > pressure_floor))


def select_endpoint_error_faces(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    face_owner: np.ndarray,
    face_neighbor: np.ndarray,
    state_scale: np.ndarray,
    max_face_fraction: float,
) -> np.ndarray:
    """Select the deterministic top-error interior-face support.

    Each cell is scored by the Euclidean norm of its component-normalized
    endpoint truth error.  An interior face receives the larger score of its
    owner and neighbor cells.  Ties are resolved by the stored face index, and
    the selected count is floored so the requested fraction is never exceeded.
    Boundary faces are always excluded.
    """

    prediction = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    owner = np.asarray(face_owner, dtype=np.int64)
    neighbor = np.asarray(face_neighbor, dtype=np.int64)
    scale = np.asarray(state_scale, dtype=np.float64)
    if prediction.ndim != 2 or prediction.shape[1] != 4:
        raise ValueError("predicted must have shape [cells, 4]")
    if truth.shape != prediction.shape:
        raise ValueError("target must match predicted")
    if not np.all(np.isfinite(prediction)) or not np.all(np.isfinite(truth)):
        raise ValueError("predicted and target must be finite")
    if scale.shape != (4,) or not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("state_scale must be finite, positive, and have shape [4]")
    if not 0.0 <= max_face_fraction <= 1.0:
        raise ValueError("max_face_fraction must lie in [0, 1]")
    if owner.ndim != 1 or neighbor.shape != owner.shape:
        raise ValueError("face_owner and face_neighbor must share shape [faces]")
    if np.any(owner < 0) or np.any(owner >= prediction.shape[0]):
        raise ValueError("face_owner contains an invalid cell index")
    if np.any(neighbor < -1) or np.any(neighbor >= prediction.shape[0]):
        raise ValueError("face_neighbor contains an invalid cell index")
    if np.any(owner == neighbor):
        raise ValueError("a face cannot have the same owner and neighbor")

    interior = np.flatnonzero(neighbor >= 0)
    selected_count = int(np.floor(max_face_fraction * interior.size + 1.0e-12))
    selected = np.zeros(owner.shape, dtype=bool)
    if selected_count == 0:
        return selected
    cell_score = np.linalg.norm((truth - prediction) / scale[None, :], axis=1)
    face_score = np.maximum(cell_score[owner[interior]], cell_score[neighbor[interior]])
    order = np.lexsort((interior, -face_score))
    selected[interior[order[:selected_count]]] = True
    return selected


def conservative_face_correction_oracle(
    predicted: np.ndarray,
    target: np.ndarray,
    current: np.ndarray,
    *,
    cell_volume: np.ndarray,
    face_owner: np.ndarray,
    face_neighbor: np.ndarray,
    allowed_face_mask: np.ndarray,
    state_scale: np.ndarray,
    max_face_fraction: float,
    max_update_ratio: float,
    gamma: float = 1.4,
    density_floor: float = 0.0,
    pressure_floor: float = 0.0,
    ridge: float = 1e-8,
    line_search_steps: int = 65,
    anti_smearing_check: AntiSmearingCheck | None = None,
) -> ConservativeCorrectionOracleResult:
    """Fit and constrain one truth-informed conservative correction direction.

    The least-squares solve uses only ``allowed_face_mask`` and interior faces.
    The decoded update is then restricted to ``max_update_ratio`` times the
    frozen global update in a volume-weighted, component-normalized norm.  A
    one-dimensional search retains only admissible states and, when supplied,
    states accepted by ``anti_smearing_check``.  The callback must accept the
    uncorrected prediction; it should express *no degradation relative to the
    frozen baseline*, rather than an absolute quality threshold.
    """

    prediction = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    initial = np.asarray(current, dtype=np.float64)
    volume = np.asarray(cell_volume, dtype=np.float64)
    owner = np.asarray(face_owner, dtype=np.int64)
    neighbor = np.asarray(face_neighbor, dtype=np.int64)
    allowed = np.asarray(allowed_face_mask, dtype=bool)
    scale = np.asarray(state_scale, dtype=np.float64)
    _validate_oracle_inputs(
        prediction,
        truth,
        initial,
        volume,
        owner,
        neighbor,
        allowed,
        scale,
        max_face_fraction=max_face_fraction,
        max_update_ratio=max_update_ratio,
        ridge=ridge,
        line_search_steps=line_search_steps,
    )
    for name, state in (
        ("current", initial),
        ("predicted", prediction),
        ("target", truth),
    ):
        if not euler2d_state_is_admissible(
            state,
            gamma=gamma,
            density_floor=density_floor,
            pressure_floor=pressure_floor,
        ):
            raise ValueError(f"{name} state is not raw-admissible")
    if anti_smearing_check is not None and not bool(anti_smearing_check(prediction)):
        raise ValueError("anti_smearing_check must accept the frozen prediction")

    interior = neighbor >= 0
    if np.any(allowed & ~interior):
        raise ValueError("allowed correction faces must all be interior")
    interior_count = int(np.count_nonzero(interior))
    allowed_count = int(np.count_nonzero(allowed))
    allowed_fraction = allowed_count / max(interior_count, 1)
    if allowed_fraction > max_face_fraction + 1e-12:
        raise ValueError(
            "allowed_face_mask exceeds max_face_fraction: "
            f"{allowed_fraction:.6g} > {max_face_fraction:.6g}"
        )

    face_impulse, stop_codes = _least_squares_face_impulse(
        prediction,
        truth,
        volume=volume,
        owner=owner,
        neighbor=neighbor,
        allowed=allowed,
        state_scale=scale,
        ridge=ridge,
    )
    candidate_correction = decode_face_impulse_correction(
        face_impulse,
        cell_volume=volume,
        face_owner=owner,
        face_neighbor=neighbor,
    )
    reference_norm = _weighted_state_norm(prediction - initial, volume, scale)
    candidate_norm = _weighted_state_norm(candidate_correction, volume, scale)
    if candidate_norm == 0.0:
        max_alpha = 0.0
        candidate_ratio = 0.0
    elif reference_norm == 0.0:
        max_alpha = 0.0
        candidate_ratio = float("inf")
    else:
        candidate_ratio = candidate_norm / reference_norm
        max_alpha = min(1.0, max_update_ratio / candidate_ratio)

    baseline_error = _weighted_state_norm(truth - prediction, volume, scale)
    best_alpha = 0.0
    best_state = prediction.copy()
    best_error = baseline_error
    feasible_points = 0
    for alpha in np.linspace(0.0, max_alpha, line_search_steps, dtype=np.float64):
        corrected = prediction + float(alpha) * candidate_correction
        if not euler2d_state_is_admissible(
            corrected,
            gamma=gamma,
            density_floor=density_floor,
            pressure_floor=pressure_floor,
        ):
            continue
        if anti_smearing_check is not None and not bool(anti_smearing_check(corrected)):
            continue
        feasible_points += 1
        error = _weighted_state_norm(truth - corrected, volume, scale)
        if error < best_error:
            best_alpha = float(alpha)
            best_state = corrected
            best_error = error

    applied_impulse = best_alpha * face_impulse
    applied_correction = best_alpha * candidate_correction
    total_balance = np.sum(volume[:, None] * applied_correction, axis=0)
    face_magnitude = np.linalg.norm(applied_impulse / scale[None, :], axis=1)
    magnitude_reference = float(np.max(face_magnitude, initial=0.0))
    active = face_magnitude > max(1e-12 * magnitude_reference, 1e-14)
    active_fraction = int(np.count_nonzero(active)) / max(interior_count, 1)
    applied_ratio = (
        _weighted_state_norm(applied_correction, volume, scale) / reference_norm
        if reference_norm > 0.0
        else 0.0
    )
    reduction = (
        (baseline_error - best_error) / baseline_error if baseline_error > 0.0 else 0.0
    )
    return ConservativeCorrectionOracleResult(
        face_impulse=applied_impulse,
        cell_correction=applied_correction,
        corrected_state=best_state,
        selected_alpha=best_alpha,
        baseline_error=baseline_error,
        corrected_error=best_error,
        relative_error_reduction=reduction,
        candidate_update_ratio=candidate_ratio,
        applied_update_ratio=applied_ratio,
        allowed_face_fraction=allowed_fraction,
        active_face_fraction=active_fraction,
        conservation_balance_max_abs=float(np.max(np.abs(total_balance), initial=0.0)),
        feasible_line_search_points=feasible_points,
        anti_smearing_checked=anti_smearing_check is not None,
        lsqr_stop_codes=stop_codes,
    )


def _least_squares_face_impulse(
    prediction: np.ndarray,
    truth: np.ndarray,
    *,
    volume: np.ndarray,
    owner: np.ndarray,
    neighbor: np.ndarray,
    allowed: np.ndarray,
    state_scale: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, tuple[int, ...]]:
    try:
        from scipy import sparse  # type: ignore[import-not-found]
        from scipy.sparse.linalg import lsqr  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError("SciPy is required for the sparse oracle solve") from exc

    num_cells, num_variables = prediction.shape
    selected = np.flatnonzero(allowed)
    full_impulse = np.zeros((owner.size, num_variables), dtype=np.float64)
    if selected.size == 0:
        return full_impulse, tuple(0 for _ in range(num_variables))

    selected_owner = owner[selected]
    selected_neighbor = neighbor[selected]
    columns = np.arange(selected.size, dtype=np.int64)
    rows = np.concatenate((selected_owner, selected_neighbor))
    cols = np.concatenate((columns, columns))
    data = np.concatenate(
        (-1.0 / volume[selected_owner], 1.0 / volume[selected_neighbor])
    )
    divergence = sparse.coo_matrix(
        (data, (rows, cols)), shape=(num_cells, selected.size)
    ).tocsr()
    dual_volume = (
        2.0
        * volume[selected_owner]
        * volume[selected_neighbor]
        / (volume[selected_owner] + volume[selected_neighbor])
    )
    weighted_operator = (
        sparse.diags(np.sqrt(volume)) @ divergence @ sparse.diags(dual_volume)
    )
    normalized_error = (truth - prediction) / state_scale[None, :]
    stop_codes: list[int] = []
    for component in range(num_variables):
        rhs = np.sqrt(volume) * normalized_error[:, component]
        solution = lsqr(
            weighted_operator,
            rhs,
            damp=float(np.sqrt(ridge)),
            atol=1e-10,
            btol=1e-10,
        )
        full_impulse[selected, component] = (
            solution[0] * dual_volume * state_scale[component]
        )
        stop_codes.append(int(solution[1]))
    return full_impulse, tuple(stop_codes)


def _weighted_state_norm(
    value: np.ndarray,
    volume: np.ndarray,
    state_scale: np.ndarray,
) -> float:
    normalized = np.asarray(value, dtype=np.float64) / state_scale[None, :]
    energy = np.sum(volume[:, None] * normalized**2) / np.sum(volume)
    return float(np.sqrt(energy))


def _validate_geometry(
    volume: np.ndarray,
    owner: np.ndarray,
    neighbor: np.ndarray,
) -> None:
    if volume.ndim != 1 or volume.size == 0:
        raise ValueError("cell_volume must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(volume)) or np.any(volume <= 0.0):
        raise ValueError("cell_volume must be finite and positive")
    if owner.ndim != 1 or neighbor.shape != owner.shape:
        raise ValueError("face_owner and face_neighbor must share shape [faces]")
    if np.any(owner < 0) or np.any(owner >= volume.size):
        raise ValueError("face_owner contains an invalid cell index")
    if np.any(neighbor < -1) or np.any(neighbor >= volume.size):
        raise ValueError("face_neighbor contains an invalid cell index")
    if np.any(owner == neighbor):
        raise ValueError("a face cannot have the same owner and neighbor")


def _validate_oracle_inputs(
    prediction: np.ndarray,
    truth: np.ndarray,
    initial: np.ndarray,
    volume: np.ndarray,
    owner: np.ndarray,
    neighbor: np.ndarray,
    allowed: np.ndarray,
    scale: np.ndarray,
    *,
    max_face_fraction: float,
    max_update_ratio: float,
    ridge: float,
    line_search_steps: int,
) -> None:
    if prediction.ndim != 2 or prediction.shape[1] != 4:
        raise ValueError("predicted must have shape [cells, 4]")
    if truth.shape != prediction.shape or initial.shape != prediction.shape:
        raise ValueError("target and current must match predicted shape")
    if not all(np.all(np.isfinite(value)) for value in (prediction, truth, initial)):
        raise ValueError("states must be finite")
    _validate_geometry(volume, owner, neighbor)
    if volume.size != prediction.shape[0]:
        raise ValueError("cell_volume must align with state cells")
    if allowed.shape != owner.shape:
        raise ValueError("allowed_face_mask must have shape [faces]")
    if scale.shape != (prediction.shape[1],):
        raise ValueError("state_scale must have shape [4]")
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("state_scale must be finite and positive")
    if not 0.0 <= max_face_fraction <= 1.0:
        raise ValueError("max_face_fraction must lie in [0, 1]")
    if max_update_ratio < 0.0 or not np.isfinite(max_update_ratio):
        raise ValueError("max_update_ratio must be finite and nonnegative")
    if ridge < 0.0 or not np.isfinite(ridge):
        raise ValueError("ridge must be finite and nonnegative")
    if line_search_steps < 2:
        raise ValueError("line_search_steps must be at least two")
