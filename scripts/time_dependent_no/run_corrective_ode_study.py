"""Run the bounded exact and learned corrective-ODE calibration study.

The state lies on ``S^1 x R``. Neural models receive the periodic embedding
``(cos(theta), sin(theta), r)`` but predict lifted-coordinate residuals
``(delta_theta, delta_r)``. Rollout path error uses the continuous phase lift;
chordal error is reported separately on the embedded state space.
Solver-relative quantities in this file are offline diagnostics; no ODE result
is evidence for an ODE-to-PDE ranking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from utility.time_dependent_no.pcno_artifacts import (  # noqa: E402
    atomic_write_json,
    digest_array,
    git_state,
    sha256_file,
    write_csv,
)

STUDY_SCHEMA = "corrective_ode_study_v1"
EXACT_SCHEMA = "corrective_ode_exact_v1"
LEARNED_SCHEMA = "corrective_ode_learned_v1"
MANIFEST_SCHEMA = "corrective_ode_manifest_v1"
PREDICTION_SCHEMA = "corrective_ode_predictions_v1"

SOURCE_FILES = (
    "scripts/time_dependent_no/run_corrective_ode_study.py",
    "tests/time_dependent_no/test_corrective_ode_study.py",
    "docs/time_dependent_no/EXPERIMENT_PLAN.md",
)


def _finite_array(value: Any, *, name: str, last_dimension: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim < 1 or array.shape[-1] != last_dimension:
        raise ValueError(f"{name} must end in dimension {last_dimension}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def state_embedding(state: np.ndarray) -> np.ndarray:
    state_array = _finite_array(state, name="state", last_dimension=2)
    theta = state_array[..., 0]
    radius = state_array[..., 1]
    return np.stack((np.cos(theta), np.sin(theta), radius), axis=-1)


def chordal_error(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    embedded_left = state_embedding(left)
    embedded_right = state_embedding(right)
    return np.linalg.norm(embedded_left - embedded_right, axis=-1)


def lifted_coordinate_error(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_array = _finite_array(left, name="left", last_dimension=2)
    right_array = _finite_array(right, name="right", last_dimension=2)
    if left_array.shape != right_array.shape:
        raise ValueError("lifted-coordinate error inputs must have identical shapes")
    phase = left_array[..., 0] - right_array[..., 0]
    normal = left_array[..., 1] - right_array[..., 1]
    return np.sqrt(phase * phase + normal * normal)


def continuous_flow_coefficients(
    *, kappa: float, coupling: float, step_size: float
) -> tuple[float, float]:
    if not all(math.isfinite(value) for value in (kappa, coupling, step_size)):
        raise ValueError("flow coefficients must be finite")
    if step_size <= 0.0:
        raise ValueError("step_size must be positive")
    exponent = kappa * step_size
    normal_gain = math.exp(exponent)
    if kappa == 0.0:
        phase_gain = coupling * step_size
    else:
        phase_gain = coupling * math.expm1(exponent) / kappa
    return normal_gain, phase_gain


@dataclass(frozen=True)
class LinearFlow:
    omega_h: float
    normal_gain: float
    normal_to_phase: float

    def validated(self) -> "LinearFlow":
        values = (self.omega_h, self.normal_gain, self.normal_to_phase)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("flow values must be finite")
        if self.normal_gain <= 0.0:
            raise ValueError("an exact scalar ODE flow must have positive normal gain")
        return self

    @classmethod
    def from_continuous(
        cls,
        *,
        omega: float,
        kappa: float,
        coupling: float,
        step_size: float,
    ) -> "LinearFlow":
        normal_gain, phase_gain = continuous_flow_coefficients(
            kappa=kappa,
            coupling=coupling,
            step_size=step_size,
        )
        return cls(
            omega_h=omega * step_size,
            normal_gain=normal_gain,
            normal_to_phase=phase_gain,
        ).validated()

    def advance(self, state: np.ndarray) -> np.ndarray:
        state_array = _finite_array(state, name="state", last_dimension=2)
        result = np.empty_like(state_array)
        result[..., 0] = (
            state_array[..., 0]
            + self.omega_h
            + self.normal_to_phase * state_array[..., 1]
        )
        result[..., 1] = self.normal_gain * state_array[..., 1]
        return result


@dataclass(frozen=True)
class AffineLearnedMap:
    name: str
    phase_forcing: float
    normal_forcing: float
    normal_gain: float
    normal_to_phase: float

    def validated(self) -> "AffineLearnedMap":
        if not self.name.strip():
            raise ValueError("map name must be nonempty")
        values = (
            self.phase_forcing,
            self.normal_forcing,
            self.normal_gain,
            self.normal_to_phase,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("learned-map coefficients must be finite")
        return self

    def advance(self, flow: LinearFlow, state: np.ndarray) -> np.ndarray:
        state_array = _finite_array(state, name="state", last_dimension=2)
        result = np.empty_like(state_array)
        result[..., 0] = (
            state_array[..., 0]
            + flow.omega_h
            + self.phase_forcing
            + self.normal_to_phase * state_array[..., 1]
        )
        result[..., 1] = self.normal_forcing + self.normal_gain * state_array[..., 1]
        return result

    def corrected(self, rho: float) -> "AffineLearnedMap":
        if not math.isfinite(rho) or not 0.0 <= rho <= 1.0:
            raise ValueError("rho must lie in [0, 1]")
        return AffineLearnedMap(
            name=f"{self.name}+C_{rho:g}",
            phase_forcing=self.phase_forcing,
            normal_forcing=rho * self.normal_forcing,
            normal_gain=rho * self.normal_gain,
            normal_to_phase=self.normal_to_phase,
        )

    @property
    def response_block(self) -> np.ndarray:
        return np.asarray(
            [[1.0, self.normal_to_phase], [0.0, self.normal_gain]],
            dtype=np.float64,
        )


def rollout(
    step: Callable[[np.ndarray], np.ndarray],
    initial: np.ndarray,
    steps: int,
) -> np.ndarray:
    if steps < 1:
        raise ValueError("steps must be positive")
    initial_array = _finite_array(initial, name="initial", last_dimension=2)
    values = np.empty((steps + 1, *initial_array.shape), dtype=np.float64)
    values[0] = initial_array
    for index in range(steps):
        next_state = _finite_array(
            step(values[index]), name="next_state", last_dimension=2
        )
        if next_state.shape != initial_array.shape:
            raise ValueError("step changed the state shape")
        values[index + 1] = next_state
    return values


def _geometric_sums(normal_gain: float, steps: int) -> tuple[np.ndarray, np.ndarray]:
    if not math.isfinite(normal_gain):
        raise ValueError("normal_gain must be finite")
    if steps < 1:
        raise ValueError("steps must be positive")
    indices = np.arange(steps + 1, dtype=np.float64)
    delta = normal_gain - 1.0
    if abs(delta) <= 1.0e-7:
        first = (
            indices
            + indices * (indices - 1.0) * delta / 2.0
            + indices * (indices - 1.0) * (indices - 2.0) * delta * delta / 6.0
        )
        second = (
            indices * (indices - 1.0) / 2.0
            + indices * (indices - 1.0) * (indices - 2.0) * delta / 6.0
            + indices
            * (indices - 1.0)
            * (indices - 2.0)
            * (indices - 3.0)
            * delta
            * delta
            / 24.0
        )
        return first, second

    powers = np.power(normal_gain, indices)
    first = (1.0 - powers) / (1.0 - normal_gain)
    second = indices / (1.0 - normal_gain) - (1.0 - powers) / ((1.0 - normal_gain) ** 2)
    return first, second


def closed_clean_rollout(
    flow: LinearFlow,
    learned_map: AffineLearnedMap,
    *,
    theta0: float,
    steps: int,
) -> np.ndarray:
    first, second = _geometric_sums(learned_map.normal_gain, steps)
    indices = np.arange(steps + 1, dtype=np.float64)
    result = np.empty((steps + 1, 2), dtype=np.float64)
    result[:, 1] = learned_map.normal_forcing * first
    phase_error = (
        indices * learned_map.phase_forcing
        + learned_map.normal_to_phase * learned_map.normal_forcing * second
    )
    result[:, 0] = theta0 + indices * flow.omega_h + phase_error
    return result


def precision_tolerance(*, horizon: int, scale: float) -> float:
    if horizon < 1 or not math.isfinite(scale) or scale < 0.0:
        raise ValueError("invalid precision-tolerance inputs")
    return 512.0 * np.finfo(np.float64).eps * (horizon + 1) * max(1.0, scale)


def _first_exit(values: np.ndarray, radius: float) -> int | None:
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("tube radius must be finite and positive")
    indices = np.flatnonzero(np.abs(np.asarray(values, dtype=np.float64)) > radius)
    return int(indices[0]) if indices.size else None


def _evaluate_affine_map(
    *,
    scenario: str,
    flow: LinearFlow,
    learned_map: AffineLearnedMap,
    steps: int,
    tube_radius: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    learned_map.validated()
    initial = np.asarray([0.0, 0.0], dtype=np.float64)
    iterated = rollout(lambda state: learned_map.advance(flow, state), initial, steps)
    closed = closed_clean_rollout(flow, learned_map, theta0=0.0, steps=steps)
    reference = rollout(flow.advance, initial, steps)
    recurrence_residual = float(np.max(np.abs(iterated - closed)))
    scale = float(max(np.max(np.abs(iterated)), np.max(np.abs(closed)), 1.0))
    tolerance = precision_tolerance(horizon=steps, scale=scale)
    phase_error = iterated[:, 0] - reference[:, 0]
    normal_error = iterated[:, 1] - reference[:, 1]
    lifted_values = np.sqrt(phase_error * phase_error + normal_error * normal_error)
    chordal_values = chordal_error(iterated, reference)
    clean_coordinate_error = math.hypot(
        learned_map.phase_forcing, learned_map.normal_forcing
    )
    clean_chordal_error = math.sqrt(
        4.0 * math.sin(learned_map.phase_forcing / 2.0) ** 2
        + learned_map.normal_forcing**2
    )
    trusted_block = np.asarray(
        [[1.0, flow.normal_to_phase], [0.0, flow.normal_gain]], dtype=np.float64
    )
    response_defect = float(np.linalg.norm(learned_map.response_block - trusted_block))
    fixed_point_denominator = 1.0 - learned_map.normal_gain
    if fixed_point_denominator != 0.0:
        normal_fixed_point_type = "unique"
        normal_fixed_point = learned_map.normal_forcing / fixed_point_denominator
        normal_fixed_point_stable: bool | None = abs(learned_map.normal_gain) < 1.0
        final_fixed_point_distance: float | None = abs(
            float(iterated[-1, 1]) - normal_fixed_point
        )
    elif learned_map.normal_forcing == 0.0:
        normal_fixed_point_type = "nonunique_all"
        normal_fixed_point = None
        normal_fixed_point_stable = None
        final_fixed_point_distance = None
    else:
        normal_fixed_point_type = "none"
        normal_fixed_point = None
        normal_fixed_point_stable = None
        final_fixed_point_distance = None
    summary = {
        "scenario": scenario,
        "map": learned_map.name,
        "steps": steps,
        "tube_radius": tube_radius,
        "clean_coordinate_error": clean_coordinate_error,
        "clean_chordal_error": clean_chordal_error,
        "phase_forcing": learned_map.phase_forcing,
        "normal_forcing": learned_map.normal_forcing,
        "normal_gain": learned_map.normal_gain,
        "normal_to_phase": learned_map.normal_to_phase,
        "trusted_response_defect_frobenius": response_defect,
        "recurrence_max_abs": recurrence_residual,
        "recurrence_tolerance": tolerance,
        "recurrence_pass": recurrence_residual <= tolerance,
        "normal_fixed_point_type": normal_fixed_point_type,
        "normal_fixed_point": normal_fixed_point,
        "normal_fixed_point_stable": normal_fixed_point_stable,
        "final_distance_to_normal_fixed_point": final_fixed_point_distance,
        "first_tube_exit": _first_exit(iterated[:, 1], tube_radius),
        "max_tube_distance": float(np.max(np.abs(iterated[:, 1]))),
        "final_normal": float(iterated[-1, 1]),
        "final_phase_error_unwrapped": float(phase_error[-1]),
        "final_lifted_coordinate_error": float(lifted_values[-1]),
        "final_chordal_error": float(chordal_values[-1]),
    }
    rows = []
    for index in range(steps + 1):
        rows.append(
            {
                "scenario": scenario,
                "map": learned_map.name,
                "step": index,
                "phase": iterated[index, 0],
                "normal": iterated[index, 1],
                "phase_error_unwrapped": phase_error[index],
                "tube_distance": abs(iterated[index, 1]),
                "lifted_coordinate_error": lifted_values[index],
                "chordal_error": chordal_values[index],
            }
        )
    return summary, rows


def _flow_with_discrete_response(
    *, omega_h: float, normal_gain: float, normal_to_phase: float, step_size: float
) -> LinearFlow:
    if normal_gain <= 0.0:
        raise ValueError("normal_gain must be positive")
    kappa = math.log(normal_gain) / step_size
    if abs(normal_gain - 1.0) <= 1.0e-14:
        coupling = normal_to_phase / step_size
    else:
        coupling = normal_to_phase * kappa / math.expm1(kappa * step_size)
    return LinearFlow.from_continuous(
        omega=omega_h / step_size,
        kappa=kappa,
        coupling=coupling,
        step_size=step_size,
    )


def run_exact_study() -> (
    tuple[
        dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]
    ]
):
    base_flow = _flow_with_discrete_response(
        omega_h=0.25,
        normal_gain=0.8,
        normal_to_phase=0.5,
        step_size=0.25,
    )
    summaries: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []

    registered = (
        (
            "ranking_reversal",
            AffineLearnedMap("A_small_clean", 0.0, 1.0e-3, 1.15, 0.5),
            30,
            0.05,
        ),
        (
            "ranking_reversal",
            AffineLearnedMap("B_larger_clean", 0.0, 5.0e-3, 0.0, 0.0),
            30,
            0.05,
        ),
        (
            "retention_without_accuracy",
            AffineLearnedMap("wrong_phase", 0.02, 0.0, 0.0, 0.0),
            40,
            0.05,
        ),
        (
            "bounded_false_attractor",
            AffineLearnedMap("false_attractor", 0.0, 0.002, 0.98, 0.0),
            300,
            0.05,
        ),
    )
    for scenario, learned_map, steps, radius in registered:
        summary, rows = _evaluate_affine_map(
            scenario=scenario,
            flow=base_flow,
            learned_map=learned_map,
            steps=steps,
            tube_radius=radius,
        )
        summaries.append(summary)
        trajectories.extend(rows)

    ranking = {
        row["map"]: row for row in summaries if row["scenario"] == "ranking_reversal"
    }
    wrong_phase = next(
        row for row in summaries if row["scenario"] == "retention_without_accuracy"
    )
    false_attractor = next(
        row for row in summaries if row["scenario"] == "bounded_false_attractor"
    )

    crossover_rows: list[dict[str, Any]] = []
    for trusted_normal_gain in np.linspace(0.05, 0.98, 94):
        kappa = math.log(float(trusted_normal_gain)) / 0.2
        trusted_phase_gain = 2.0 * math.expm1(kappa * 0.2) / kappa
        flow = LinearFlow(
            omega_h=0.2,
            normal_gain=float(trusted_normal_gain),
            normal_to_phase=trusted_phase_gain,
        ).validated()
        maps = (
            AffineLearnedMap("RECOVERY", 0.002, 0.001, 0.05, 0.0),
            AffineLearnedMap(
                "DYN_RELABEL",
                0.0005,
                0.001,
                flow.normal_gain,
                flow.normal_to_phase,
            ),
        )
        for learned_map in maps:
            summary, _ = _evaluate_affine_map(
                scenario="recovery_relabeling_crossover",
                flow=flow,
                learned_map=learned_map,
                steps=80,
                tube_radius=0.05,
            )
            crossover_rows.append(
                {
                    "trusted_normal_gain": flow.normal_gain,
                    "trusted_normal_to_phase": flow.normal_to_phase,
                    "map": learned_map.name,
                    "clean_coordinate_error": summary["clean_coordinate_error"],
                    "trusted_response_defect_frobenius": summary[
                        "trusted_response_defect_frobenius"
                    ],
                    "final_lifted_coordinate_error": summary[
                        "final_lifted_coordinate_error"
                    ],
                    "final_tube_distance": abs(summary["final_normal"]),
                    "final_phase_error_unwrapped": summary[
                        "final_phase_error_unwrapped"
                    ],
                    "recurrence_max_abs": summary["recurrence_max_abs"],
                    "recurrence_tolerance": summary["recurrence_tolerance"],
                    "recurrence_pass": summary["recurrence_pass"],
                }
            )

    crossover_by_gain: dict[float, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in crossover_rows:
        crossover_by_gain[float(row["trusted_normal_gain"])][str(row["map"])] = row
    relabel_wins = [
        gain
        for gain, rows in crossover_by_gain.items()
        if rows["DYN_RELABEL"]["final_lifted_coordinate_error"]
        < rows["RECOVERY"]["final_lifted_coordinate_error"]
    ]
    recovery_wins = [
        gain
        for gain, rows in crossover_by_gain.items()
        if rows["RECOVERY"]["final_lifted_coordinate_error"]
        < rows["DYN_RELABEL"]["final_lifted_coordinate_error"]
    ]
    substantial_gain = min(crossover_by_gain, key=lambda value: abs(value - 0.95))
    substantial_rows = crossover_by_gain[substantial_gain]

    raw = AffineLearnedMap("raw_predictor", 0.0, 0.002, 1.2, 0.5)
    composition_rows = []
    composition_probe = np.asarray([[0.0, 0.0], [0.3, 0.04]], dtype=np.float64)
    composition_radius = 0.05
    for rho in (1.0, 0.5, 0.0):
        corrected = raw.corrected(rho)
        explicit_one_step = raw.advance(base_flow, composition_probe)
        explicit_one_step[:, 1] *= rho
        factored_one_step = corrected.advance(base_flow, composition_probe)

        def explicit_step(state: np.ndarray, *, current_rho: float = rho) -> np.ndarray:
            value = raw.advance(base_flow, state)
            value[:, 1] *= current_rho
            return value

        explicit_rollout = rollout(explicit_step, composition_probe, 20)
        factored_rollout = rollout(
            lambda state, current=corrected: current.advance(base_flow, state),
            composition_probe,
            20,
        )
        corrected_closed = closed_clean_rollout(
            base_flow, corrected, theta0=float(composition_probe[0, 0]), steps=20
        )
        corrected_recurrence_residual = float(
            np.max(np.abs(explicit_rollout[:, 0, :] - corrected_closed))
        )
        corrected_recurrence_scale = float(
            max(
                np.max(np.abs(explicit_rollout[:, 0, :])),
                np.max(np.abs(corrected_closed)),
                1.0,
            )
        )
        corrected_recurrence_tolerance = precision_tolerance(
            horizon=20, scale=corrected_recurrence_scale
        )
        tube_left = (
            corrected.normal_gain * composition_radius + corrected.normal_forcing
        )
        composition_rows.append(
            {
                "rho": rho,
                "normal_forcing": corrected.normal_forcing,
                "normal_gain": corrected.normal_gain,
                "normal_to_phase": corrected.normal_to_phase,
                "expected_normal_forcing": rho * raw.normal_forcing,
                "expected_normal_gain": rho * raw.normal_gain,
                "one_step_composition_max_abs": float(
                    np.max(np.abs(explicit_one_step - factored_one_step))
                ),
                "rollout_composition_max_abs": float(
                    np.max(np.abs(explicit_rollout - factored_rollout))
                ),
                "corrected_recurrence_max_abs": corrected_recurrence_residual,
                "corrected_recurrence_tolerance": corrected_recurrence_tolerance,
                "corrected_recurrence_pass": bool(
                    corrected_recurrence_residual <= corrected_recurrence_tolerance
                ),
                "tube_radius": composition_radius,
                "tube_closure_left": tube_left,
                "tube_closure_pass": bool(tube_left <= composition_radius),
                "identity_parity": bool(
                    rho != 1.0
                    or (
                        corrected.phase_forcing == raw.phase_forcing
                        and corrected.normal_forcing == raw.normal_forcing
                        and corrected.normal_gain == raw.normal_gain
                        and corrected.normal_to_phase == raw.normal_to_phase
                        and np.array_equal(explicit_one_step, factored_one_step)
                    )
                ),
            }
        )

    checks = {
        "all_recurrences_close": bool(
            all(row["recurrence_pass"] for row in summaries)
            and all(row["recurrence_pass"] for row in crossover_rows)
            and all(row["corrected_recurrence_pass"] for row in composition_rows)
        ),
        "ranking_reversal": bool(
            ranking["A_small_clean"]["clean_coordinate_error"]
            < ranking["B_larger_clean"]["clean_coordinate_error"]
            and ranking["A_small_clean"]["final_lifted_coordinate_error"]
            > ranking["B_larger_clean"]["final_lifted_coordinate_error"]
        ),
        "retention_without_accuracy": bool(
            wrong_phase["max_tube_distance"] == 0.0
            and abs(wrong_phase["final_phase_error_unwrapped"] - 0.8) <= 1.0e-13
        ),
        "bounded_false_attractor": bool(
            false_attractor["first_tube_exit"] is not None
            and abs(false_attractor["final_normal"] - 0.1) <= 3.0e-4
            and math.isfinite(false_attractor["final_lifted_coordinate_error"])
        ),
        "conditional_crossover": bool(relabel_wins and recovery_wins),
        "recovery_wins_substantial_response": bool(
            substantial_rows["RECOVERY"]["final_lifted_coordinate_error"]
            < substantial_rows["DYN_RELABEL"]["final_lifted_coordinate_error"]
            and substantial_rows["RECOVERY"]["trusted_response_defect_frobenius"]
            > substantial_rows["DYN_RELABEL"]["trusted_response_defect_frobenius"]
        ),
        "composition_coefficients_close": all(
            abs(row["normal_forcing"] - row["expected_normal_forcing"]) <= 1.0e-15
            and abs(row["normal_gain"] - row["expected_normal_gain"]) <= 1.0e-15
            and row["normal_to_phase"] == raw.normal_to_phase
            and row["one_step_composition_max_abs"] == 0.0
            and row["rollout_composition_max_abs"] == 0.0
            for row in composition_rows
        ),
        "composition_tube_closure_classification": bool(
            not composition_rows[0]["tube_closure_pass"]
            and composition_rows[1]["tube_closure_pass"]
            and composition_rows[2]["tube_closure_pass"]
        ),
        "identity_parity": composition_rows[0]["identity_parity"],
    }
    exact_summary = {
        "schema": EXACT_SCHEMA,
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "base_flow": asdict(base_flow),
        "scenarios": summaries,
        "crossover": {
            "grid_size": len(crossover_by_gain),
            "relabel_win_gain_range": [min(relabel_wins), max(relabel_wins)],
            "recovery_win_gain_range": [min(recovery_wins), max(recovery_wins)],
            "substantial_gain": substantial_gain,
            "substantial_gain_rows": substantial_rows,
            "declared_tradeoff": (
                "RECOVERY has larger clean phase forcing but stronger return; "
                "DYN_RELABEL has smaller clean forcing and exact displaced response."
            ),
        },
        "composition": composition_rows,
    }
    return exact_summary, summaries, trajectories, crossover_rows


def _regular_polygon_contains(points: np.ndarray, query: np.ndarray) -> bool:
    points_array = _finite_array(points, name="points", last_dimension=2)
    query_array = _finite_array(query, name="query", last_dimension=2)
    if query_array.shape != (2,):
        raise ValueError("query must be one two-dimensional point")
    edges = np.roll(points_array, -1, axis=0) - points_array
    offsets = query_array[None, :] - points_array
    crosses = edges[:, 0] * offsets[:, 1] - edges[:, 1] * offsets[:, 0]
    return bool(np.all(crosses >= -1.0e-12))


def proxy_geometry_calibration(
    *, clean_points: int = 128, local_neighbors: int = 12
) -> list[dict[str, Any]]:
    if clean_points < 16 or local_neighbors < 3 or local_neighbors >= clean_points:
        raise ValueError("invalid proxy calibration sizes")
    phases = np.linspace(-np.pi, np.pi, clean_points, endpoint=False)
    clean = np.column_stack((np.cos(phases), np.sin(phases), np.zeros(clean_points)))
    polygon = clean[:, :2]
    queries = (
        ("on_manifold_midpoint", phases[0] + np.pi / clean_points, 1.0, 0.0),
        ("normal_displacement", phases[0] + np.pi / clean_points, 1.0, 0.1),
        ("ambient_inward_chord", phases[0], 0.8, 0.0),
        ("ambient_center", 0.0, 0.0, 0.0),
    )
    rows = []
    for name, theta, planar_radius, normal in queries:
        query = np.asarray(
            [planar_radius * math.cos(theta), planar_radius * math.sin(theta), normal],
            dtype=np.float64,
        )
        distances = np.linalg.norm(clean - query[None, :], axis=1)
        neighbor_indices = np.argsort(distances)[:local_neighbors]
        neighborhood = clean[neighbor_indices]
        center = neighborhood.mean(axis=0)
        _, _, right = np.linalg.svd(neighborhood - center, full_matrices=False)
        tangent = right[0]
        local_residual = float(
            np.linalg.norm((query - center) - np.dot(query - center, tangent) * tangent)
        )
        exact_embedded_distance = math.sqrt((planar_radius - 1.0) ** 2 + normal**2)
        inside = bool(
            abs(normal) <= 1.0e-12 and _regular_polygon_contains(polygon, query[:2])
        )
        rows.append(
            {
                "query": name,
                "exact_embedded_distance": exact_embedded_distance,
                "knn_distance": float(np.min(distances)),
                "local_pca_rank1_residual": local_residual,
                "sampled_convex_hull_inside": inside,
                "convex_hull_ood": not inside,
                "clean_points": clean_points,
                "local_neighbors": local_neighbors,
            }
        )
    return rows


class ResidualMLP(nn.Module):
    """Periodic input embedding with lifted-coordinate residual output."""

    def __init__(self, *, width: int, hidden_layers: int) -> None:
        super().__init__()
        if width < 1 or hidden_layers < 1:
            raise ValueError("MLP width and hidden_layers must be positive")
        layers: list[nn.Module] = []
        input_width = 3
        for _ in range(hidden_layers):
            layers.extend((nn.Linear(input_width, width), nn.Tanh()))
            input_width = width
        layers.append(nn.Linear(input_width, 2))
        self.network = nn.Sequential(*layers)

    def residual(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim != 2 or state.shape[1] != 2:
            raise ValueError("state must have shape [batch, 2]")
        embedded = torch.stack(
            (torch.cos(state[:, 0]), torch.sin(state[:, 0]), state[:, 1]), dim=1
        )
        return self.network(embedded)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return state + self.residual(state)


@dataclass(frozen=True)
class LearnedStudyConfig:
    omega_h: float = 0.25
    trusted_normal_gain: float = 0.8
    trusted_normal_to_phase: float = 0.5
    train_phases: int = 256
    max_displacement: float = 0.2
    hidden_width: int = 32
    hidden_layers: int = 2
    learning_rate: float = 3.0e-3
    batch_size: int = 128
    training_steps: int = 1500
    seeds: tuple[int, ...] = (17, 29, 43)
    data_seed: int = 20260830
    evaluation_phases: int = 128
    query_radii: tuple[float, ...] = (-0.2, -0.1, -0.05, 0.05, 0.1, 0.2)
    rollout_steps: int = 160
    impulse_steps: int = 40
    impulse_radius: float = 0.1
    tube_radius: float = 0.1

    def validated(self) -> "LearnedStudyConfig":
        if self.train_phases < 16 or self.train_phases % 2:
            raise ValueError("train_phases must be an even integer at least 16")
        if self.evaluation_phases < 8:
            raise ValueError("evaluation_phases must be at least 8")
        if self.train_phases % self.evaluation_phases:
            raise ValueError(
                "evaluation_phases must divide train_phases for a disjoint half-grid"
            )
        if self.training_steps < 1 or self.batch_size < 1:
            raise ValueError("training_steps and batch_size must be positive")
        if self.hidden_width < 1 or self.hidden_layers < 1:
            raise ValueError("hidden dimensions must be positive")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be nonempty and unique")
        if self.max_displacement <= 0.0 or self.tube_radius <= 0.0:
            raise ValueError("displacement and tube radii must be positive")
        if self.impulse_radius <= 0.0:
            raise ValueError("impulse radius must be positive")
        if not all(
            0.0 < abs(value) <= self.max_displacement for value in self.query_radii
        ):
            raise ValueError("query radii must be nonzero and inside training range")
        return self

    @property
    def flow(self) -> LinearFlow:
        return LinearFlow(
            omega_h=self.omega_h,
            normal_gain=self.trusted_normal_gain,
            normal_to_phase=self.trusted_normal_to_phase,
        ).validated()


def build_training_contract(
    config: LearnedStudyConfig,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    config.validated()
    flow = config.flow
    phases = np.linspace(-np.pi, np.pi, config.train_phases, endpoint=False)
    clean = np.column_stack((phases, np.zeros_like(phases)))
    rng = np.random.default_rng(config.data_seed)
    displacements = np.linspace(
        -config.max_displacement,
        config.max_displacement,
        config.train_phases,
        endpoint=True,
        dtype=np.float64,
    )
    permutation = rng.permutation(config.train_phases)
    displacements = displacements[permutation]
    displaced = clean.copy()
    displaced[:, 1] = displacements
    clean_target = flow.advance(clean)
    displaced_target = flow.advance(displaced)

    datasets = {
        "CLEAN": (
            np.concatenate((clean, clean), axis=0),
            np.concatenate((clean_target, clean_target), axis=0),
        ),
        "RECOVERY": (
            np.concatenate((clean, displaced), axis=0),
            np.concatenate((clean_target, clean_target), axis=0),
        ),
        "DYN_RELABEL": (
            np.concatenate((clean, displaced), axis=0),
            np.concatenate((clean_target, displaced_target), axis=0),
        ),
    }
    metadata = {
        "clean_digest": digest_array(clean),
        "displaced_digest": digest_array(displaced),
        "clean_target_digest": digest_array(clean_target),
        "displaced_target_digest": digest_array(displaced_target),
        "recovery_displaced_input_digest": digest_array(
            datasets["RECOVERY"][0][config.train_phases :]
        ),
        "dyn_relabel_displaced_input_digest": digest_array(
            datasets["DYN_RELABEL"][0][config.train_phases :]
        ),
        "displacement_min": float(np.min(displacements)),
        "displacement_max": float(np.max(displacements)),
        "displacement_mean": float(np.mean(displacements)),
    }
    if (
        metadata["recovery_displaced_input_digest"]
        != metadata["dyn_relabel_displaced_input_digest"]
    ):
        raise AssertionError("recovery and relabeling inputs do not match")
    return datasets, metadata


def _state_dict_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _batch_plan(config: LearnedStudyConfig, seed: int, rows: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 10_000_019)
    return rng.integers(
        0,
        rows,
        size=(config.training_steps, config.batch_size),
        dtype=np.int64,
    )


def train_matched_models(
    config: LearnedStudyConfig,
    *,
    checkpoint_dir: Path,
) -> tuple[dict[int, dict[str, ResidualMLP]], list[dict[str, Any]], dict[str, Any]]:
    config.validated()
    datasets, data_metadata = build_training_contract(config)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    trained: dict[int, dict[str, ResidualMLP]] = {}
    training_rows: list[dict[str, Any]] = []
    paired_metadata: dict[str, Any] = {"data": data_metadata, "seeds": {}}

    for seed in config.seeds:
        torch.manual_seed(seed)
        template = ResidualMLP(
            width=config.hidden_width, hidden_layers=config.hidden_layers
        ).to(dtype=torch.float32)
        initial_state = {
            name: value.detach().clone()
            for name, value in template.state_dict().items()
        }
        initial_digest = _state_dict_digest(initial_state)
        rows = next(iter(datasets.values()))[0].shape[0]
        batches = _batch_plan(config, seed, rows)
        batch_digest = digest_array(batches)
        trained[seed] = {}
        seed_metadata = {
            "initial_parameter_digest": initial_digest,
            "batch_plan_digest": batch_digest,
            "arms": {},
        }

        for arm in ("CLEAN", "RECOVERY", "DYN_RELABEL"):
            model = ResidualMLP(
                width=config.hidden_width, hidden_layers=config.hidden_layers
            ).to(dtype=torch.float32)
            model.load_state_dict(initial_state)
            if _state_dict_digest(model.state_dict()) != initial_digest:
                raise AssertionError("paired initialization did not close")
            inputs_np, targets_np = datasets[arm]
            inputs = torch.as_tensor(inputs_np, dtype=torch.float32)
            targets = torch.as_tensor(targets_np, dtype=torch.float32)
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
            with torch.no_grad():
                initial_loss = float(torch.mean((model(inputs) - targets) ** 2).item())
            for batch in batches:
                indices = torch.from_numpy(batch)
                prediction = model(inputs[indices])
                loss = torch.mean((prediction - targets[indices]) ** 2)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite {arm} loss at seed {seed}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                final_loss = float(torch.mean((model(inputs) - targets) ** 2).item())
            checkpoint_path = checkpoint_dir / f"seed_{seed}_{arm.lower()}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            reloaded = ResidualMLP(
                width=config.hidden_width, hidden_layers=config.hidden_layers
            ).to(dtype=torch.float32)
            reloaded.load_state_dict(
                torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            )
            reloaded.eval()
            if _state_dict_digest(reloaded.state_dict()) != _state_dict_digest(
                model.state_dict()
            ):
                raise AssertionError("checkpoint round trip changed model parameters")
            trained[seed][arm] = reloaded
            seed_metadata["arms"][arm] = {
                "checkpoint": checkpoint_path.name,
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "parameter_digest": _state_dict_digest(reloaded.state_dict()),
            }
            training_rows.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "initial_loss": initial_loss,
                    "final_loss": final_loss,
                    "updates": config.training_steps,
                    "checkpoint": checkpoint_path.name,
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                    "initial_parameter_digest": initial_digest,
                    "batch_plan_digest": batch_digest,
                }
            )
        paired_metadata["seeds"][str(seed)] = seed_metadata
    return trained, training_rows, paired_metadata


def _model_step(
    model: ResidualMLP,
    state: np.ndarray,
    *,
    rho: float,
) -> np.ndarray:
    if not math.isfinite(rho) or not 0.0 <= rho <= 1.0:
        raise ValueError("rho must lie in [0, 1]")
    state_array = _finite_array(state, name="state", last_dimension=2)
    original_shape = state_array.shape
    flat = state_array.reshape(-1, 2)
    with torch.no_grad():
        raw = model(torch.as_tensor(flat, dtype=torch.float32)).cpu().numpy()
    raw = np.asarray(raw, dtype=np.float64).reshape(original_shape)
    if not np.isfinite(raw).all():
        raise FloatingPointError("model produced a non-finite state")
    if rho == 1.0:
        return raw
    corrected = raw.copy()
    corrected[..., 1] *= rho
    return corrected


def _test_phases(config: LearnedStudyConfig) -> np.ndarray:
    config.validated()
    train_spacing = 2.0 * np.pi / config.train_phases
    stride = config.train_phases // config.evaluation_phases
    return -np.pi + (stride * np.arange(config.evaluation_phases) + 0.5) * train_spacing


def _least_squares_response(input_normal: np.ndarray, output: np.ndarray) -> float:
    inputs = np.asarray(input_normal, dtype=np.float64).reshape(-1)
    outputs = np.asarray(output, dtype=np.float64).reshape(-1)
    if inputs.shape != outputs.shape or not inputs.size:
        raise ValueError("response arrays must be nonempty and aligned")
    denominator = float(np.dot(inputs, inputs))
    if denominator <= 0.0:
        raise ValueError("response inputs must have positive energy")
    return float(np.dot(inputs, outputs) / denominator)


def estimate_learned_response(
    model: ResidualMLP,
    config: LearnedStudyConfig,
    *,
    rho: float,
) -> dict[str, float]:
    phases = _test_phases(config)
    clean = np.column_stack((phases, np.zeros_like(phases)))
    clean_output = _model_step(model, clean, rho=rho)
    input_values = []
    phase_responses = []
    normal_responses = []
    for radius in config.query_radii:
        displaced = clean.copy()
        displaced[:, 1] = radius
        displaced_output = _model_step(model, displaced, rho=rho)
        input_values.append(np.full(phases.shape, radius, dtype=np.float64))
        phase_responses.append(displaced_output[:, 0] - clean_output[:, 0])
        normal_responses.append(displaced_output[:, 1] - clean_output[:, 1])
    inputs = np.concatenate(input_values)
    phase_response = np.concatenate(phase_responses)
    normal_response = np.concatenate(normal_responses)
    phase_gain = _least_squares_response(inputs, phase_response)
    normal_gain = _least_squares_response(inputs, normal_response)
    return {
        "normal_gain": normal_gain,
        "normal_to_phase": phase_gain,
        "trusted_response_defect": math.hypot(
            normal_gain - config.trusted_normal_gain,
            phase_gain - config.trusted_normal_to_phase,
        ),
        "recovery_response_distance": math.hypot(normal_gain, phase_gain),
    }


def _rollout_metrics(
    *,
    step: Callable[[np.ndarray], np.ndarray],
    initial: np.ndarray,
    reference: np.ndarray,
    steps: int,
    tube_radius: float,
    seed: int,
    arm: str,
    assay: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predicted = rollout(step, initial, steps)
    if reference.shape != predicted.shape:
        raise ValueError("reference and learned rollout shapes differ")
    rows = []
    exits = np.full(initial.shape[0], steps + 1, dtype=np.int64)
    exited = np.zeros(initial.shape[0], dtype=bool)
    lifted_by_step = []
    phase_by_step = []
    tube_by_step = []
    chordal_by_step = []
    for index in range(steps + 1):
        phase_error = predicted[index, :, 0] - reference[index, :, 0]
        normal_error = predicted[index, :, 1] - reference[index, :, 1]
        lifted_values = np.sqrt(phase_error * phase_error + normal_error * normal_error)
        chordal_values = chordal_error(predicted[index], reference[index])
        tube_values = np.abs(predicted[index, :, 1])
        new_exits = (~exited) & (tube_values > tube_radius)
        exits[new_exits] = index
        exited |= new_exits
        lifted_rms = float(np.sqrt(np.mean(lifted_values**2)))
        phase_rms = float(np.sqrt(np.mean(phase_error**2)))
        tube_rms = float(np.sqrt(np.mean(tube_values**2)))
        chordal_rms = float(np.sqrt(np.mean(chordal_values**2)))
        lifted_by_step.append(lifted_rms)
        phase_by_step.append(phase_rms)
        tube_by_step.append(tube_rms)
        chordal_by_step.append(chordal_rms)
        rows.append(
            {
                "seed": seed,
                "arm": arm,
                "assay": assay,
                "step": index,
                "lifted_coordinate_error_rms": lifted_rms,
                "phase_error_unwrapped_rms": phase_rms,
                "tube_distance_rms": tube_rms,
                "chordal_error_rms": chordal_rms,
                "tube_exit_fraction": float(np.mean(exited)),
            }
        )
    aggregate = {
        "final_lifted_coordinate_error": lifted_by_step[-1],
        "final_phase_error_unwrapped": phase_by_step[-1],
        "final_tube_distance": tube_by_step[-1],
        "final_chordal_error": chordal_by_step[-1],
        "max_lifted_coordinate_error": max(lifted_by_step),
        "max_tube_distance": max(tube_by_step),
        "lifted_coordinate_error_auc": float(np.mean(lifted_by_step[1:])),
        "phase_error_auc": float(np.mean(phase_by_step[1:])),
        "tube_distance_auc": float(np.mean(tube_by_step[1:])),
        "tube_exit_fraction": float(np.mean(exited)),
        "median_exit_step": float(np.median(exits)),
    }
    return rows, aggregate


def evaluate_learned_models(
    trained: Mapping[int, Mapping[str, ResidualMLP]],
    config: LearnedStudyConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    config.validated()
    phases = _test_phases(config)
    clean_initial = np.column_stack((phases, np.zeros_like(phases)))
    impulse_initial = clean_initial.copy()
    impulse_initial[:, 1] = config.impulse_radius
    clean_reference = rollout(config.flow.advance, clean_initial, config.rollout_steps)
    impulse_trusted_reference = rollout(
        config.flow.advance, impulse_initial, config.impulse_steps
    )
    impulse_clean_reference = rollout(
        config.flow.advance, clean_initial, config.impulse_steps
    )
    metric_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    parity: dict[str, Any] = {}

    for seed in config.seeds:
        models = trained[seed]
        deployed = (
            ("CLEAN", models["CLEAN"], 1.0, "CLEAN"),
            ("CLEAN_C0", models["CLEAN"], 0.0, "CLEAN"),
            ("RECOVERY", models["RECOVERY"], 1.0, "RECOVERY"),
            ("DYN_RELABEL", models["DYN_RELABEL"], 1.0, "DYN_RELABEL"),
        )
        raw_clean = _model_step(models["CLEAN"], clean_initial, rho=1.0)
        identity_clean = _model_step(models["CLEAN"], clean_initial, rho=1.0)
        identity_parity = bool(np.array_equal(raw_clean, identity_clean))
        parity[str(seed)] = {"identity_parity": identity_parity}
        if not identity_parity:
            raise AssertionError("identity corrector changed the raw predictor")

        for arm, model, rho, parent in deployed:
            response = estimate_learned_response(model, config, rho=rho)
            clean_output = _model_step(model, clean_initial, rho=rho)
            exact_clean_output = config.flow.advance(clean_initial)
            clean_phase_error = clean_output[:, 0] - exact_clean_output[:, 0]
            clean_normal_error = clean_output[:, 1] - exact_clean_output[:, 1]
            clean_lifted = np.sqrt(clean_phase_error**2 + clean_normal_error**2)
            clean_chordal = chordal_error(clean_output, exact_clean_output)

            step = lambda value, current_model=model, current_rho=rho: _model_step(
                current_model, value, rho=current_rho
            )
            id_rows, id_metrics = _rollout_metrics(
                step=step,
                initial=clean_initial,
                reference=clean_reference,
                steps=config.rollout_steps,
                tube_radius=config.tube_radius,
                seed=seed,
                arm=arm,
                assay="id_clean_start",
            )
            trusted_rows, trusted_metrics = _rollout_metrics(
                step=step,
                initial=impulse_initial,
                reference=impulse_trusted_reference,
                steps=config.impulse_steps,
                tube_radius=config.tube_radius,
                seed=seed,
                arm=arm,
                assay="normal_impulse_displaced_truth",
            )
            recovery_rows, recovery_metrics = _rollout_metrics(
                step=step,
                initial=impulse_initial,
                reference=impulse_clean_reference,
                steps=config.impulse_steps,
                tube_radius=config.tube_radius,
                seed=seed,
                arm=arm,
                assay="normal_impulse_clean_reference",
            )
            rollout_rows.extend(id_rows)
            rollout_rows.extend(trusted_rows)
            rollout_rows.extend(recovery_rows)
            metric_rows.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "parent_predictor": parent,
                    "rho": rho,
                    "clean_lifted_coordinate_error_rms": float(
                        np.sqrt(np.mean(clean_lifted**2))
                    ),
                    "clean_phase_error_rms": float(
                        np.sqrt(np.mean(clean_phase_error**2))
                    ),
                    "clean_normal_error_rms": float(
                        np.sqrt(np.mean(clean_normal_error**2))
                    ),
                    "clean_chordal_error_rms": float(
                        np.sqrt(np.mean(clean_chordal**2))
                    ),
                    **response,
                    **{f"id_{key}": value for key, value in id_metrics.items()},
                    **{
                        f"impulse_trusted_{key}": value
                        for key, value in trusted_metrics.items()
                    },
                    **{
                        f"impulse_clean_{key}": value
                        for key, value in recovery_metrics.items()
                    },
                }
            )

    rows_by_seed_arm = {(int(row["seed"]), str(row["arm"])): row for row in metric_rows}
    seed_checks = {}
    for seed in config.seeds:
        clean = rows_by_seed_arm[(seed, "CLEAN")]
        projected = rows_by_seed_arm[(seed, "CLEAN_C0")]
        recovery = rows_by_seed_arm[(seed, "RECOVERY")]
        relabel = rows_by_seed_arm[(seed, "DYN_RELABEL")]
        seed_checks[str(seed)] = {
            "recovery_moves_toward_zero": bool(
                recovery["recovery_response_distance"]
                < relabel["recovery_response_distance"]
            ),
            "relabel_matches_trusted_response": bool(
                relabel["trusted_response_defect"] < recovery["trusted_response_defect"]
            ),
            "projection_zero_normal_gain": bool(
                abs(projected["normal_gain"]) <= 1.0e-10
            ),
            "projection_preserves_phase_coupling": bool(
                abs(projected["normal_to_phase"] - clean["normal_to_phase"]) <= 1.0e-10
            ),
            "relabel_tracks_displaced_truth": bool(
                relabel["impulse_trusted_lifted_coordinate_error_auc"]
                < recovery["impulse_trusted_lifted_coordinate_error_auc"]
            ),
            "recovery_returns_to_clean_path": bool(
                recovery["impulse_clean_lifted_coordinate_error_auc"]
                < relabel["impulse_clean_lifted_coordinate_error_auc"]
            ),
            "identity_parity": parity[str(seed)]["identity_parity"],
        }
    all_checks = {
        key: all(checks[key] for checks in seed_checks.values())
        for key in next(iter(seed_checks.values()))
    }
    return (
        metric_rows,
        rollout_rows,
        {
            "seed_checks": seed_checks,
            "all_seed_checks": all_checks,
            "mechanism_realization_pass": all(all_checks.values()),
        },
    )


def _aggregate_learned_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, float]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["arm"])].append(row)
    ignored = {"seed", "arm", "parent_predictor"}
    result: dict[str, dict[str, dict[str, float]]] = {}
    for arm, arm_rows in sorted(grouped.items()):
        metrics: dict[str, dict[str, float]] = {}
        for key in arm_rows[0]:
            if key in ignored:
                continue
            values = [row[key] for row in arm_rows]
            if not all(
                isinstance(value, (int, float, np.integer, np.floating))
                for value in values
            ):
                continue
            array = np.asarray(values, dtype=np.float64)
            metrics[key] = {
                "mean": float(np.mean(array)),
                "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
                "min": float(np.min(array)),
                "max": float(np.max(array)),
            }
        result[arm] = metrics
    return result


def registered_predictions() -> dict[str, Any]:
    return {
        "schema": PREDICTION_SCHEMA,
        "frozen_before_training": True,
        "predictions": [
            {
                "id": "P1_RECOVERY_RESPONSE",
                "statement": "RECOVERY moves (a,b) closer to (0,0) than DYN_RELABEL on every paired seed.",
            },
            {
                "id": "P2_RELABEL_RESPONSE",
                "statement": "DYN_RELABEL has lower trusted (a,b) response defect than RECOVERY on every paired seed.",
            },
            {
                "id": "P3_OPERATIONAL_PROJECTION",
                "statement": "CLEAN+C0 has zero deployed normal gain and preserves the CLEAN predictor's phase coupling.",
            },
            {
                "id": "P4_IMPULSE_TARGET_TRADEOFF",
                "statement": "DYN_RELABEL better follows displaced truth while RECOVERY returns more strongly to the clean reference path.",
            },
        ],
        "non_prediction": (
            "No autonomous ID rollout ranking is required; dense ODE coverage "
            "may make all clean-start rollouts accurate."
        ),
    }


def _plot_study(
    *,
    exact_trajectories: Sequence[Mapping[str, Any]],
    crossover_rows: Sequence[Mapping[str, Any]],
    learned_rows: Sequence[Mapping[str, Any]],
    output_pdf: Path,
    output_png: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.8), constrained_layout=True)
    colors = {
        "CLEAN": "#4C78A8",
        "CLEAN_C0": "#72B7B2",
        "RECOVERY": "#F58518",
        "DYN_RELABEL": "#E45756",
    }

    ranking_rows = [
        row for row in exact_trajectories if row["scenario"] == "ranking_reversal"
    ]
    for name in ("A_small_clean", "B_larger_clean"):
        selected = [row for row in ranking_rows if row["map"] == name]
        axes[0, 0].plot(
            [row["step"] for row in selected],
            [max(float(row["lifted_coordinate_error"]), 1.0e-8) for row in selected],
            label=name.replace("_", " "),
        )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("rollout step")
    axes[0, 0].set_ylabel("lifted-coordinate error")
    axes[0, 0].set_title("(a) Smaller clean error, worse rollout")
    axes[0, 0].legend(frameon=False, fontsize=8)

    crossover_grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in crossover_rows:
        crossover_grouped[str(row["map"])].append(row)
    for name in ("RECOVERY", "DYN_RELABEL"):
        selected = sorted(
            crossover_grouped[name], key=lambda row: row["trusted_normal_gain"]
        )
        axes[0, 1].plot(
            [row["trusted_normal_gain"] for row in selected],
            [row["final_lifted_coordinate_error"] for row in selected],
            label=name.replace("_", "-"),
            color=colors[name],
        )
    axes[0, 1].set_xlabel(r"trusted normal response $a_\star$")
    axes[0, 1].set_ylabel("final lifted-coordinate error")
    axes[0, 1].set_title("(b) Conditional recovery--relabeling crossover")
    axes[0, 1].legend(frameon=False, fontsize=8)

    aggregate = _aggregate_learned_metrics(learned_rows)
    arms = ["CLEAN", "RECOVERY", "DYN_RELABEL", "CLEAN_C0"]
    x = np.arange(len(arms), dtype=np.float64)
    width = 0.36
    normal_means = [aggregate[arm]["normal_gain"]["mean"] for arm in arms]
    normal_stds = [aggregate[arm]["normal_gain"]["std"] for arm in arms]
    phase_means = [aggregate[arm]["normal_to_phase"]["mean"] for arm in arms]
    phase_stds = [aggregate[arm]["normal_to_phase"]["std"] for arm in arms]
    axes[1, 0].bar(
        x - width / 2,
        normal_means,
        width,
        yerr=normal_stds,
        label="normal gain a",
        color="#4C78A8",
        capsize=2,
    )
    axes[1, 0].bar(
        x + width / 2,
        phase_means,
        width,
        yerr=phase_stds,
        label="normal-to-phase b",
        color="#F58518",
        capsize=2,
    )
    axes[1, 0].axhline(0.8, color="#4C78A8", linestyle="--", linewidth=1)
    axes[1, 0].axhline(0.5, color="#F58518", linestyle="--", linewidth=1)
    axes[1, 0].set_xticks(x, [arm.replace("_", "\n") for arm in arms], fontsize=7)
    axes[1, 0].set_ylabel("frozen-bank response coefficient")
    axes[1, 0].set_title(r"(c) Matched learned response (mean $\pm$ s.d.)")
    axes[1, 0].legend(frameon=False, fontsize=8)

    trusted_means = [
        aggregate[arm]["impulse_trusted_lifted_coordinate_error_auc"]["mean"]
        for arm in arms
    ]
    trusted_stds = [
        aggregate[arm]["impulse_trusted_lifted_coordinate_error_auc"]["std"]
        for arm in arms
    ]
    clean_means = [
        aggregate[arm]["impulse_clean_lifted_coordinate_error_auc"]["mean"]
        for arm in arms
    ]
    clean_stds = [
        aggregate[arm]["impulse_clean_lifted_coordinate_error_auc"]["std"]
        for arm in arms
    ]
    axes[1, 1].bar(
        x - width / 2,
        trusted_means,
        width,
        yerr=trusted_stds,
        label="to displaced truth",
        color="#E45756",
        capsize=2,
    )
    axes[1, 1].bar(
        x + width / 2,
        clean_means,
        width,
        yerr=clean_stds,
        label="to clean path",
        color="#72B7B2",
        capsize=2,
    )
    axes[1, 1].set_xticks(x, [arm.replace("_", "\n") for arm in arms], fontsize=7)
    axes[1, 1].set_ylabel("normal-impulse error AUC")
    axes[1, 1].set_title("(d) One displacement, two declared targets")
    axes[1, 1].legend(frameon=False, fontsize=8)

    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.18, linewidth=0.6)
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _source_records() -> dict[str, dict[str, Any]]:
    records = {}
    for relative_name in SOURCE_FILES:
        path = REPOSITORY_ROOT / relative_name
        if not path.is_file():
            raise FileNotFoundError(path)
        records[relative_name] = {
            "sha256": sha256_file(path),
            "bytes": int(path.stat().st_size),
        }
    return records


def _output_records(output_dir: Path) -> dict[str, dict[str, Any]]:
    records = {}
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name in {"manifest.json", "closeout.json"}:
            continue
        relative_name = path.relative_to(output_dir).as_posix()
        records[relative_name] = {
            "sha256": sha256_file(path),
            "bytes": int(path.stat().st_size),
        }
    return records


def _write_packet_closeout(
    *,
    output_dir: Path,
    config: LearnedStudyConfig,
    mode: str,
    summary_status: str,
) -> dict[str, Any]:
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "study_schema": STUDY_SCHEMA,
        "mode": mode,
        "summary_status": summary_status,
        "canonical_contract": asdict(config) == asdict(LearnedStudyConfig()),
        "git": git_state(),
        "sources": _source_records(),
        "outputs": _output_records(output_dir),
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    closeout = {
        "schema": "corrective_ode_closeout_v1",
        "status": "complete",
        "manifest_sha256": sha256_file(output_dir / "manifest.json"),
    }
    atomic_write_json(output_dir / "closeout.json", closeout)
    return verify_study_packet(output_dir)


def verify_study_packet(
    output_dir: str | Path, *, verify_current_sources: bool = True
) -> dict[str, Any]:
    root = Path(output_dir)
    manifest_path = root / "manifest.json"
    closeout_path = root / "closeout.json"
    if not manifest_path.is_file() or not closeout_path.is_file():
        raise FileNotFoundError("study packet lacks manifest.json or closeout.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    closeout = json.loads(closeout_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unsupported corrective-ODE manifest schema")
    if closeout.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("closeout manifest hash mismatch")
    for relative_name, record in manifest.get("outputs", {}).items():
        path = root / relative_name
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise ValueError(f"output hash mismatch: {relative_name}")
        if int(path.stat().st_size) != int(record.get("bytes", -1)):
            raise ValueError(f"output size mismatch: {relative_name}")
    if verify_current_sources:
        for relative_name, record in manifest.get("sources", {}).items():
            path = REPOSITORY_ROOT / relative_name
            if not path.is_file() or sha256_file(path) != record.get("sha256"):
                raise ValueError(f"current source mismatch: {relative_name}")
    return {
        "status": "verified",
        "manifest_sha256": sha256_file(manifest_path),
        "outputs": len(manifest.get("outputs", {})),
        "sources": len(manifest.get("sources", {})),
        "current_sources_checked": verify_current_sources,
        "canonical_contract": bool(manifest.get("canonical_contract")),
        "summary_status": manifest.get("summary_status"),
    }


def run_study(
    *,
    output_dir: str | Path,
    mode: str,
    config: LearnedStudyConfig | None = None,
) -> dict[str, Any]:
    if mode not in {"exact", "all"}:
        raise ValueError("mode must be 'exact' or 'all'")
    study_config = (config or LearnedStudyConfig()).validated()
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    atomic_write_json(
        root / "config.json",
        {
            "schema": "corrective_ode_config_v1",
            "mode": mode,
            "learned": asdict(study_config),
        },
    )
    atomic_write_json(root / "predictions.json", registered_predictions())

    exact, exact_summaries, exact_trajectories, crossover_rows = run_exact_study()
    if exact["status"] != "pass":
        raise AssertionError("registered exact ODE contracts failed")
    proxy_rows = proxy_geometry_calibration()
    atomic_write_json(root / "exact_summary.json", exact)
    write_csv(root / "exact_scenarios.csv", exact_summaries)
    write_csv(root / "exact_trajectories.csv", exact_trajectories)
    write_csv(root / "crossover_sweep.csv", crossover_rows)
    write_csv(root / "proxy_geometry.csv", proxy_rows)

    learned_summary = None
    if mode == "all":
        torch.set_num_threads(1)
        trained, training_rows, paired_metadata = train_matched_models(
            study_config, checkpoint_dir=root / "checkpoints"
        )
        learned_metrics, learned_rollouts, learned_checks = evaluate_learned_models(
            trained, study_config
        )
        aggregate = _aggregate_learned_metrics(learned_metrics)
        learned_summary = {
            "schema": LEARNED_SCHEMA,
            "status": (
                "mechanism_realization_pass"
                if learned_checks["mechanism_realization_pass"]
                else "mechanism_realization_falsified"
            ),
            "checks": learned_checks,
            "aggregate": aggregate,
            "paired_contract": paired_metadata,
        }
        write_csv(root / "training.csv", training_rows)
        write_csv(root / "learned_metrics.csv", learned_metrics)
        write_csv(root / "learned_rollouts.csv", learned_rollouts)
        atomic_write_json(root / "learned_summary.json", learned_summary)
        _plot_study(
            exact_trajectories=exact_trajectories,
            crossover_rows=crossover_rows,
            learned_rows=learned_metrics,
            output_pdf=root / "figure3_ode_calibration.pdf",
            output_png=root / "figure3_ode_calibration.png",
        )

    summary_status = (
        "pass"
        if learned_summary is None
        or learned_summary["status"] == "mechanism_realization_pass"
        else "exact_pass_learned_falsified"
    )
    summary = {
        "schema": STUDY_SCHEMA,
        "status": summary_status,
        "scope": "supporting_ode_calibration_only",
        "ode_to_pde_transfer_claim": False,
        "exact": exact,
        "learned": learned_summary,
    }
    atomic_write_json(root / "summary.json", summary)
    persisted_summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    verification = _write_packet_closeout(
        output_dir=root,
        config=study_config,
        mode=mode,
        summary_status=summary_status,
    )
    return {
        "summary": persisted_summary,
        "verification": verification,
        "output_dir": str(root),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run or verify the bounded exact/learned corrective-ODE study. "
            "This command does not access PDE data, checkpoints, or solvers."
        )
    )
    parser.add_argument("--mode", choices=("exact", "all", "verify"), default="all")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/time_dependent_no/corrective_ode_study_20260830d"),
    )
    parser.add_argument("--training-steps", type=int, default=1500)
    parser.add_argument("--hidden-width", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3.0e-3)
    parser.add_argument("--seeds", type=int, nargs="+", default=(17, 29, 43))
    parser.add_argument(
        "--packet-only",
        action="store_true",
        help=(
            "with --mode verify, check the persisted packet hashes without "
            "requiring the current checkout to match its bound sources"
        ),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.mode == "verify":
        result = verify_study_packet(
            args.output_dir, verify_current_sources=not args.packet_only
        )
    else:
        if args.packet_only:
            raise ValueError("--packet-only is valid only with --mode verify")
        config = LearnedStudyConfig(
            training_steps=args.training_steps,
            hidden_width=args.hidden_width,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seeds=tuple(args.seeds),
        )
        result = run_study(output_dir=args.output_dir, mode=args.mode, config=config)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
