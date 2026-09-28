"""Finite-amplitude, path-conditioned diagnostics for learned time steppers.

The historical path-conditioned metrics use the same archived next reference
state for both inputs; they do not measure solver-relative off-path fidelity.
The separate paired solver-response assay additionally requires a trusted next
state from the displaced input and distinguishes recovery from faithful dynamics.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

METRIC_SCHEMA = "path_conditioned_tube_probe_v1"
AGGREGATE_SCHEMA = "path_conditioned_tube_tail_score_v1"
PAIRED_RESPONSE_SCHEMA = "paired_solver_response_probe_v1"


def _field(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError(f"{name} must have shape [nodes, channels]")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _metric_geometry(
    *,
    nodes: int,
    channels: int,
    node_weights: np.ndarray,
    component_scale: Sequence[float] | np.ndarray,
    node_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    weights = np.asarray(node_weights, dtype=np.float64)
    if weights.ndim == 2:
        if weights.shape[0] != nodes:
            raise ValueError("node_weights must align with the node axis")
        weights = weights.sum(axis=1)
    else:
        weights = weights.reshape(-1)
    if weights.shape != (nodes,):
        raise ValueError("node_weights must have shape [nodes] or [nodes, measures]")
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("node_weights must be finite and nonnegative")

    if node_mask is not None:
        mask = np.asarray(node_mask, dtype=np.float64).reshape(-1)
        if mask.shape != (nodes,):
            raise ValueError("node_mask must align with the node axis")
        if not np.isfinite(mask).all() or np.any((mask != 0.0) & (mask != 1.0)):
            raise ValueError("node_mask must contain only zeros and ones")
        weights = weights * mask

    scale = np.asarray(component_scale, dtype=np.float64).reshape(-1)
    if scale.shape != (channels,):
        raise ValueError("component_scale must contain one value per channel")
    if not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise ValueError("component_scale must be finite and strictly positive")

    weight_sum = float(weights.sum())
    if not math.isfinite(weight_sum) or weight_sum <= 0.0:
        raise ValueError("effective node weights must have positive mass")
    return weights, scale, weight_sum * channels


def path_conditioned_tube_metrics(
    *,
    reference_input: np.ndarray,
    displaced_input: np.ndarray,
    reference_prediction: np.ndarray,
    displaced_prediction: np.ndarray,
    reference_next: np.ndarray,
    node_weights: np.ndarray,
    component_scale: Sequence[float] | np.ndarray,
    node_mask: np.ndarray | None = None,
) -> dict[str, float | str | None]:
    """Measure signed finite-amplitude escape from one archived reference path.

    Let ``eta`` be the input displacement, ``d`` the clean one-step defect, and
    ``p`` the learned response to ``eta``.  The primary quantity is

    ``(||d + p|| - ||d||) / ||eta||``.

    It subtracts clean one-step error, retains the sign of helpful versus
    harmful defect--response alignment, and is bounded in magnitude by the
    learned secant gain ``||p|| / ||eta||``.  A zero displacement is reported as
    unresolved rather than hidden behind an epsilon.
    """

    fields = {
        "reference_input": _field(reference_input, name="reference_input"),
        "displaced_input": _field(displaced_input, name="displaced_input"),
        "reference_prediction": _field(
            reference_prediction, name="reference_prediction"
        ),
        "displaced_prediction": _field(
            displaced_prediction, name="displaced_prediction"
        ),
        "reference_next": _field(reference_next, name="reference_next"),
    }
    shapes = {value.shape for value in fields.values()}
    if len(shapes) != 1:
        raise ValueError("all path-conditioned fields must have identical shapes")
    nodes, channels = next(iter(shapes))
    weights, scale, normalization = _metric_geometry(
        nodes=nodes,
        channels=channels,
        node_weights=node_weights,
        component_scale=component_scale,
        node_mask=node_mask,
    )

    def inner(left: np.ndarray, right: np.ndarray) -> float:
        return float(
            np.einsum(
                "n,nc,nc->",
                weights,
                left / scale[None, :],
                right / scale[None, :],
                optimize=True,
            )
            / normalization
        )

    eta = fields["displaced_input"] - fields["reference_input"]
    clean_defect = fields["reference_prediction"] - fields["reference_next"]
    response = fields["displaced_prediction"] - fields["reference_prediction"]
    displaced_defect = fields["displaced_prediction"] - fields["reference_next"]
    closure = displaced_defect - (clean_defect + response)

    input_energy = max(inner(eta, eta), 0.0)
    clean_energy = max(inner(clean_defect, clean_defect), 0.0)
    response_energy = max(inner(response, response), 0.0)
    displaced_energy = max(inner(displaced_defect, displaced_defect), 0.0)
    cross_inner = inner(clean_defect, response)
    closure_energy = max(inner(closure, closure), 0.0)

    input_rms = math.sqrt(input_energy)
    clean_rms = math.sqrt(clean_energy)
    response_rms = math.sqrt(response_energy)
    displaced_rms = math.sqrt(displaced_energy)
    closure_rms = math.sqrt(closure_energy)
    energy_identity_residual = displaced_energy - (
        clean_energy + response_energy + 2.0 * cross_inner
    )
    energy_identity_scale = max(
        displaced_energy,
        clean_energy + response_energy + 2.0 * abs(cross_inner),
        np.finfo(np.float64).tiny,
    )

    if clean_rms == 0.0 or response_rms == 0.0:
        cosine = None
    else:
        raw_cosine = cross_inner / (clean_rms * response_rms)
        if abs(raw_cosine) > 1.0 + 1.0e-12:
            raise ArithmeticError("defect--response cosine violates Cauchy--Schwarz")
        cosine = float(np.clip(raw_cosine, -1.0, 1.0))

    if input_rms == 0.0:
        secant_gain = None
        tube_escape_slope = None
        squared_excess_per_input_energy = None
        alignment_per_input_energy = None
        slope_identity_absolute = None
        secant_bound_relative_violation = None
    else:
        secant_gain = float(response_rms / input_rms)
        tube_escape_slope = float((displaced_rms - clean_rms) / input_rms)
        squared_excess_per_input_energy = float(
            (displaced_energy - clean_energy) / input_energy
        )
        alignment_per_input_energy = float(2.0 * cross_inner / input_energy)
        reconstructed_slope = (
            float(
                (response_energy + 2.0 * cross_inner)
                / (input_rms * (displaced_rms + clean_rms))
            )
            if displaced_rms + clean_rms > 0.0
            else 0.0
        )
        slope_identity_absolute = float(
            abs(tube_escape_slope - reconstructed_slope)
        )
        secant_bound_relative_violation = float(
            max(abs(tube_escape_slope) - secant_gain, 0.0)
            / max(secant_gain, np.finfo(np.float64).tiny)
        )

    return {
        "schema": METRIC_SCHEMA,
        "input_displacement_scaled_rms": input_rms,
        "clean_defect_scaled_rms": clean_rms,
        "learned_response_scaled_rms": response_rms,
        "displaced_defect_scaled_rms": displaced_rms,
        "learned_secant_gain": secant_gain,
        "tube_escape_slope": tube_escape_slope,
        "defect_response_cosine": cosine,
        "defect_response_scaled_inner": cross_inner,
        "alignment_per_input_energy": alignment_per_input_energy,
        "squared_excess_per_input_energy": squared_excess_per_input_energy,
        "output_closure_scaled_rms": closure_rms,
        "energy_identity_relative": float(
            abs(energy_identity_residual) / energy_identity_scale
        ),
        "slope_identity_absolute": slope_identity_absolute,
        "secant_bound_relative_violation": secant_bound_relative_violation,
    }


def paired_solver_response_metrics(
    *,
    reference_input: np.ndarray,
    displaced_input: np.ndarray,
    reference_prediction: np.ndarray,
    displaced_prediction: np.ndarray,
    reference_next: np.ndarray,
    trusted_displaced_next: np.ndarray,
    node_weights: np.ndarray,
    component_scale: Sequence[float] | np.ndarray,
    node_mask: np.ndarray | None = None,
) -> dict[str, float | str | None]:
    """Separate recovery-target error from trusted displaced-state fidelity.

    All fields have shape ``[nodes, channels]`` and use the historical metric's
    node-weighted, component-scaled RMS geometry. Let ``b = Psi(u) - S(u)``,
    ``p = Psi(u + eta) - Psi(u)`` and ``r = S(u + eta) - S(u)``. The recovery
    error is ``||b + p||``; displaced-dynamics error is ``||b + p - r||``.
    The response defect ``p - r`` cancels a common additive prediction bias.

    These are finite-amplitude secants, not derivatives, manifold coordinates,
    or ID/OOD classifications. The caller must supply a qualified trusted map,
    the complete deployed learned map, and matched state/forcing conventions.
    For a stochastic map, use common random tapes within each input pair and
    assess tape variability separately. This function neither samples nor
    averages predictions. All inputs must be real-valued; complex arrays are
    rejected before conversion. Zero-denominator gains/cosines are ``None``.
    """
    fields = {
        name: np.asarray(value)
        for name, value in (
            ("reference_input", reference_input),
            ("displaced_input", displaced_input),
            ("reference_prediction", reference_prediction),
            ("displaced_prediction", displaced_prediction),
            ("reference_next", reference_next),
            ("trusted_displaced_next", trusted_displaced_next),
        )
    }
    for name, value in (
        *fields.items(),
        ("node_weights", node_weights),
        ("component_scale", component_scale),
        ("node_mask", node_mask),
    ):
        if np.iscomplexobj(np.asarray(value)):
            raise ValueError(f"{name} must be real-valued")
    fields = {name: _field(value, name=name) for name, value in fields.items()}
    shapes = {value.shape for value in fields.values()}
    if len(shapes) != 1:
        raise ValueError("all paired solver-response fields must have identical shapes")
    nodes, channels = next(iter(shapes))
    raw_weights = np.asarray(node_weights, dtype=np.float64)
    if raw_weights.ndim not in (1, 2):
        raise ValueError("node_weights must have shape [nodes] or [nodes, measures]")
    if not np.isfinite(raw_weights).all() or np.any(raw_weights < 0.0):
        raise ValueError("node_weights must be finite and nonnegative")
    weights, scale, normalization = _metric_geometry(
        nodes=nodes,
        channels=channels,
        node_weights=node_weights,
        component_scale=component_scale,
        node_mask=node_mask,
    )
    if not math.isfinite(normalization):
        raise ValueError("metric normalization must be finite")

    def inner(left: np.ndarray, right: np.ndarray) -> float:
        value = float(
            np.einsum(
                "n,nc,nc->",
                weights,
                left / scale[None, :],
                right / scale[None, :],
                optimize=True,
            )
            / normalization
        )
        if not math.isfinite(value):
            raise ArithmeticError("paired solver-response arithmetic is non-finite")
        return value

    def norm(value: np.ndarray) -> float:
        return math.sqrt(max(inner(value, value), 0.0))

    def ratio(numerator: float, denominator: float) -> float | None:
        if denominator == 0.0:
            return None
        value = numerator / denominator
        if not math.isfinite(value):
            raise ArithmeticError("paired solver-response ratio is non-finite")
        return value

    def cosine(cross: float, left_norm: float, right_norm: float) -> float | None:
        if left_norm == 0.0 or right_norm == 0.0:
            return None
        value = (cross / left_norm) / right_norm
        if not math.isfinite(value) or abs(value) > 1.0 + 1.0e-12:
            raise ArithmeticError("paired response cosine violates Cauchy--Schwarz")
        return float(np.clip(value, -1.0, 1.0))

    eta = fields["displaced_input"] - fields["reference_input"]
    clean = fields["reference_prediction"] - fields["reference_next"]
    learned = fields["displaced_prediction"] - fields["reference_prediction"]
    trusted = fields["trusted_displaced_next"] - fields["reference_next"]
    recovery = fields["displaced_prediction"] - fields["reference_next"]
    dynamics = fields["displaced_prediction"] - fields["trusted_displaced_next"]
    response_defect = learned - trusted
    input_norm = norm(eta)
    clean_norm = norm(clean)
    learned_norm = norm(learned)
    trusted_norm = norm(trusted)
    response_defect_norm = norm(response_defect)
    response_cross = inner(learned, trusted)
    defect_cross = inner(clean, response_defect)

    return {
        "schema": PAIRED_RESPONSE_SCHEMA,
        "input_displacement_scaled_rms": input_norm,
        "clean_defect_scaled_rms": clean_norm,
        "recovery_target_error_scaled_rms": norm(recovery),
        "displaced_dynamics_error_scaled_rms": norm(dynamics),
        "learned_response_scaled_rms": learned_norm,
        "trusted_response_scaled_rms": trusted_norm,
        "response_defect_scaled_rms": response_defect_norm,
        "learned_secant_gain": ratio(learned_norm, input_norm),
        "trusted_secant_gain": ratio(trusted_norm, input_norm),
        "response_defect_gain": ratio(response_defect_norm, input_norm),
        "learned_trusted_response_scaled_inner": response_cross,
        "learned_trusted_response_cosine": cosine(
            response_cross, learned_norm, trusted_norm
        ),
        "clean_response_defect_scaled_inner": defect_cross,
        "clean_response_defect_cosine": cosine(
            defect_cross, clean_norm, response_defect_norm
        ),
        "recovery_closure_scaled_rms": norm(recovery - (clean + learned)),
        "dynamics_closure_scaled_rms": norm(dynamics - (clean + response_defect)),
        "response_defect_closure_scaled_rms": norm(
            response_defect - (dynamics - clean)
        ),
    }


def aggregate_tube_escape_tail_score(
    rows: Sequence[Mapping[str, Any]],
    *,
    prefix_horizon: int,
    tail_quantile: float = 0.9,
) -> dict[str, Any]:
    """Aggregate a rectangular, cross-fitted probe bank without model weighting.

    The score is the equal-weighted mean of one upper-tail slope per
    ``(case, amplitude)`` cell.  Each recipient must see exactly the same probe
    identities, and the global donor and recipient model sets must be disjoint.
    Lower scores indicate less harmful path-conditioned response.
    """

    if not rows:
        raise ValueError("tube score requires at least one probe row")
    if prefix_horizon < 1:
        raise ValueError("prefix_horizon must be positive")
    if not 0.0 < tail_quantile < 1.0:
        raise ValueError("tail_quantile must lie strictly between zero and one")

    required = {
        "recipient_model_id",
        "donor_model_id",
        "case_id",
        "call_index",
        "amplitude_multiplier",
        "input_admissible",
        "input_displacement_scaled_rms",
        "tube_escape_slope",
    }
    recipients: set[str] = set()
    donors: set[str] = set()
    records_by_recipient: dict[str, dict[tuple[str, int, str, float], float]] = (
        defaultdict(dict)
    )
    for row in rows:
        missing = required - set(row)
        if missing:
            raise ValueError(f"tube probe row is missing fields: {sorted(missing)}")
        identities = {
            name: row[name]
            for name in ("recipient_model_id", "donor_model_id", "case_id")
        }
        if any(
            not isinstance(value, str) or not value.strip()
            for value in identities.values()
        ):
            raise TypeError("model and case identities must be nonempty strings")
        recipient = identities["recipient_model_id"]
        donor = identities["donor_model_id"]
        case = identities["case_id"]

        call_value = row["call_index"]
        if isinstance(call_value, (bool, np.bool_)) or not isinstance(
            call_value, (int, np.integer)
        ):
            raise TypeError("call_index must be an integer")
        call = int(call_value)

        numeric_values = {
            "amplitude_multiplier": row["amplitude_multiplier"],
            "input_displacement_scaled_rms": row[
                "input_displacement_scaled_rms"
            ],
            "tube_escape_slope": row["tube_escape_slope"],
        }
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            for value in numeric_values.values()
        ):
            raise TypeError("probe metrics and amplitudes must be numeric scalars")
        amplitude = float(numeric_values["amplitude_multiplier"])
        displacement = float(numeric_values["input_displacement_scaled_rms"])
        slope = float(numeric_values["tube_escape_slope"])
        if not 1 <= call <= prefix_horizon:
            raise ValueError("probe call lies outside the frozen prefix horizon")
        if not math.isfinite(amplitude) or amplitude <= 0.0:
            raise ValueError("amplitude_multiplier must be finite and positive")
        if not isinstance(row["input_admissible"], (bool, np.bool_)) or not bool(
            row["input_admissible"]
        ):
            raise ValueError("the primary tube bank may contain only admissible inputs")
        if not math.isfinite(displacement) or displacement <= 0.0:
            raise ValueError("probe displacement must be finite and nonzero")
        if not math.isfinite(slope):
            raise ValueError("tube_escape_slope must be finite")
        key = (case, call, donor, amplitude)
        if key in records_by_recipient[recipient]:
            raise ValueError("duplicate probe identity for one recipient")
        records_by_recipient[recipient][key] = slope
        recipients.add(recipient)
        donors.add(donor)

    overlap = recipients & donors
    if overlap:
        raise ValueError(f"donor and recipient model sets overlap: {sorted(overlap)}")
    probe_key_sets = {frozenset(records) for records in records_by_recipient.values()}
    if len(probe_key_sets) != 1:
        raise ValueError("all recipients must use the identical rectangular probe bank")

    results: dict[str, Any] = {}
    for recipient, records in sorted(records_by_recipient.items()):
        cell_values: dict[tuple[str, float], list[float]] = defaultdict(list)
        for (case, _, _, amplitude), slope in records.items():
            cell_values[(case, amplitude)].append(slope)
        cases = sorted({case for case, _ in cell_values})
        amplitudes = sorted({amplitude for _, amplitude in cell_values})
        expected_cells = {(case, amplitude) for case in cases for amplitude in amplitudes}
        if set(cell_values) != expected_cells:
            raise ValueError("case-by-amplitude cells must form a complete rectangle")
        tail_values = {
            f"{case}|{amplitude:.17g}": float(
                np.quantile(
                    np.asarray(values, dtype=np.float64),
                    tail_quantile,
                    method="linear",
                )
            )
            for (case, amplitude), values in sorted(cell_values.items())
        }
        cell_probe_counts = {
            f"{case}|{amplitude:.17g}": len(values)
            for (case, amplitude), values in sorted(cell_values.items())
        }
        results[recipient] = {
            "tube_escape_tail_score": float(np.mean(list(tail_values.values()))),
            "case_count": len(cases),
            "case_ids": cases,
            "amplitude_multipliers": amplitudes,
            "probe_count": len(records),
            "cell_tail_scores": tail_values,
            "cell_probe_counts": cell_probe_counts,
        }

    return {
        "schema": AGGREGATE_SCHEMA,
        "prefix_horizon": prefix_horizon,
        "tail_quantile": tail_quantile,
        "quantile_method": "linear",
        "donor_model_ids": sorted(donors),
        "recipient_model_ids": sorted(recipients),
        "probe_count_per_recipient": len(next(iter(records_by_recipient.values()))),
        "models": results,
    }


__all__ = [
    "AGGREGATE_SCHEMA",
    "METRIC_SCHEMA",
    "PAIRED_RESPONSE_SCHEMA",
    "aggregate_tube_escape_tail_score",
    "paired_solver_response_metrics",
    "path_conditioned_tube_metrics",
]
