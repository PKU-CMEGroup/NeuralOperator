#!/usr/bin/env python3
"""Audit graph-frequency conditioning of the finite-volume face decoder.

D049 is a zero-training, CPU diagnostic. Physical perturbation evidence is
restricted to one accepted dynamic reference on its validated 250x100 mesh.
The optional 2x/4x meshes contain analytic Cartesian geometry only and are
reported solely as algebraic refinement controls, never as fine-grid shock
evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no.evaluate_pcno_shock_vortex_baseline import (  # noqa: E402
    endpoint_metrics,
)
from utility.time_dependent_no.fv_impulse_diagnostics import (  # noqa: E402
    CLAIM_BOUNDARY,
    FVImpulseOperators,
    InteriorDivergenceBandMode,
    build_fv_impulse_operators,
    decoder_gain_summary,
    factorize_direct_minimum_winv_norm_projector,
    interior_divergence_band_modes,
)
from utility.time_dependent_no.pcno_ripple_diagnostics import (  # noqa: E402
    raw_admissibility_summary,
)
from utility.time_dependent_no.shock_vortex_fv import (  # noqa: E402
    ShockVortexFVConfig,
    make_structured_fv_geometry,
)

SCHEMA = "fv_divergence_conditioning_audit_v1"
EXPERIMENT_ID = "D049"
DEFAULT_INTERVALS = (1, 6, 12)
DEFAULT_LEVELS = (0.01, 0.05, 0.10)
DEFAULT_SEEDS = (20260722, 20260723, 20260724)
DEFAULT_RESOLUTIONS = ((250, 100), (500, 200), (1000, 400))
DOCUMENTED_D048_ERROR_GAIN_RANGE = (1.995, 2.002)
GATE_THRESHOLDS = {
    "canonical_closure_relative_l2_max": 1.0e-8,
    "cycle_decoded_gain_max": 1.0e-7,
    "band_order_fraction_min": 1.0,
    "high_to_low_gain_ratio_min": 4.0,
    "high_to_target_gain_ratio_min": 100.0,
    "flux_gain_refinement_ratio_min": 1.8,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--intervals", type=int, nargs="+", default=DEFAULT_INTERVALS)
    parser.add_argument("--levels", type=float, nargs="+", default=DEFAULT_LEVELS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--lowpass-steps", type=int, default=64)
    parser.add_argument("--transition-steps", type=int, default=8)
    parser.add_argument("--shock-quantile", type=float, default=0.90)
    parser.add_argument("--macro-dt", type=float, default=0.01)
    parser.add_argument(
        "--resolutions",
        nargs="+",
        default=[f"{nx}x{ny}" for nx, ny in DEFAULT_RESOLUTIONS],
    )
    return parser.parse_args(argv)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest_mapping(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_resolution(value: str) -> tuple[int, int]:
    try:
        nx_text, ny_text = value.lower().split("x", maxsplit=1)
        nx, ny = int(nx_text), int(ny_text)
    except (TypeError, ValueError) as error:
        raise ValueError(f"resolution must have form NXxNY, got {value!r}") from error
    if nx < 2 or ny < 2:
        raise ValueError("resolution dimensions must be at least two")
    return nx, ny


def _load_reference(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    summary_path = path.with_name("summary.json")
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "passed":
        raise ValueError("reference summary is not passed")
    checks = summary.get("contract_checks")
    if not isinstance(checks, dict) or not checks or not all(checks.values()):
        raise ValueError("reference summary has an unclosed contract check")
    artifact_sha256 = _sha256_file(path)
    if artifact_sha256 != summary.get("reference_artifact_sha256"):
        raise ValueError("reference artifact digest does not match its summary")

    required = {
        "schema",
        "conservative_states",
        "physical_times",
        "cumulative_accepted_substep_face_impulses",
        "cell_centers",
        "cell_volume",
        "face_centers",
        "face_measure",
        "face_normal",
        "face_owner",
        "face_neighbor",
        "face_boundary_tag",
        "coordinate_convention",
        "face_orientation_convention",
        "state_convention",
        "boundary_mode",
        "config_json",
    }
    with np.load(path, allow_pickle=False) as artifact:
        missing = sorted(required - set(artifact.files))
        if missing:
            raise ValueError(f"reference artifact is missing arrays: {missing}")
        if str(artifact["schema"].item()) != "shock_vortex_fv_reference_v2":
            raise ValueError("reference artifact schema is not supported")
        arrays = {
            name: np.array(artifact[name])
            for name in required
            if name not in {"schema", "config_json"}
        }
        config = json.loads(str(artifact["config_json"].item()))
    config_digest = _digest_mapping(config)
    if config_digest != summary.get("config_digest"):
        raise ValueError("reference configuration digest does not match its summary")

    states = arrays["conservative_states"]
    times = arrays["physical_times"]
    impulses = arrays["cumulative_accepted_substep_face_impulses"]
    if states.ndim != 3 or states.shape[-1] != 4:
        raise ValueError("reference states must have shape [T,N,4]")
    if times.shape != (states.shape[0],):
        raise ValueError("reference times and states disagree")
    if impulses.shape != (states.shape[0] - 1, arrays["face_owner"].size, 4):
        raise ValueError("reference face impulses have incompatible shape")
    if not np.all(np.diff(times) > 0.0):
        raise ValueError("reference times must be strictly increasing")
    if not all(np.all(np.isfinite(value)) for value in (states, times, impulses)):
        raise ValueError("reference dynamic arrays must be finite")
    metadata = {
        "artifact_path": path.as_posix(),
        "artifact_sha256": artifact_sha256,
        "summary_path": summary_path.as_posix(),
        "summary_sha256": _sha256_file(summary_path),
        "config": config,
        "config_digest": config_digest,
        "schema": "shock_vortex_fv_reference_v2",
    }
    return arrays, metadata


def _operators_from_reference(arrays: Mapping[str, np.ndarray]) -> FVImpulseOperators:
    return build_fv_impulse_operators(
        cell_centers=arrays["cell_centers"],
        cell_volume=arrays["cell_volume"],
        face_centers=arrays["face_centers"],
        face_measure=arrays["face_measure"],
        face_owner=arrays["face_owner"],
        face_neighbor=arrays["face_neighbor"],
        face_boundary_tag=arrays["face_boundary_tag"],
    )


def _winv_component_norm(field: np.ndarray, face_weight: np.ndarray) -> np.ndarray:
    values = np.asarray(field, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    return np.sqrt(np.sum(values * values / face_weight[:, None], axis=0))


def _winv_norm(field: np.ndarray, face_weight: np.ndarray) -> float:
    return float(np.linalg.norm(_winv_component_norm(field, face_weight)))


def _volume_component_norm(field: np.ndarray, volume: np.ndarray) -> np.ndarray:
    values = np.asarray(field, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    return np.sqrt(np.sum(volume[:, None] * values * values, axis=0))


def _volume_scaled_relative_l2(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    volume: np.ndarray,
    component_scale: np.ndarray,
) -> float:
    scaled_error = (np.asarray(actual) - np.asarray(expected)) / component_scale
    scaled_expected = np.asarray(expected) / component_scale
    numerator = float(np.sum(volume[:, None] * scaled_error**2))
    denominator = float(np.sum(volume[:, None] * scaled_expected**2))
    return float(np.sqrt(numerator / max(denominator, 1.0e-30)))


def _scale_component_field(
    field: np.ndarray,
    *,
    face_weight: np.ndarray,
    target_component_norm: np.ndarray,
    level: float,
    polarity: int,
) -> np.ndarray:
    values = np.asarray(field, dtype=np.float64)
    source_norm = _winv_component_norm(values, face_weight)
    scaled = np.zeros_like(values)
    for component in range(values.shape[1]):
        if target_component_norm[component] == 0.0:
            continue
        if source_norm[component] == 0.0:
            raise ValueError("cannot scale a zero face component to nonzero norm")
        scaled[:, component] = (
            polarity
            * level
            * target_component_norm[component]
            * values[:, component]
            / source_norm[component]
        )
    return scaled


def _scale_band_mode(
    mode: InteriorDivergenceBandMode,
    *,
    target_component_norm: np.ndarray,
    level: float,
    polarity: int,
) -> np.ndarray:
    return (
        polarity * level * mode.face_impulse[:, None] * target_component_norm[None, :]
    )


def _evaluate_error(
    *,
    error_face: np.ndarray,
    canonical_face: np.ndarray,
    current: np.ndarray,
    target: np.ndarray,
    operators: FVImpulseOperators,
    edges: np.ndarray,
    component_scale: np.ndarray,
    gamma: float,
    shock_quantile: float,
) -> dict[str, Any]:
    error_face_norm = _winv_norm(error_face, operators.face_weight)
    canonical_face_norm = _winv_norm(canonical_face, operators.face_weight)
    decoded_error = -np.asarray(operators.incidence @ error_face) / (
        operators.cell_volume[:, None]
    )
    prediction = target + decoded_error
    target_increment = target - current
    decoded_error_norm = float(
        np.linalg.norm(_volume_component_norm(decoded_error, operators.cell_volume))
    )
    increment_norm = float(
        np.linalg.norm(_volume_component_norm(target_increment, operators.cell_volume))
    )
    relative_face_error = error_face_norm / max(canonical_face_norm, 1.0e-30)
    relative_increment_error = decoded_error_norm / max(increment_norm, 1.0e-30)
    relative_error_gain = relative_increment_error / max(relative_face_error, 1.0e-30)
    admissibility = raw_admissibility_summary(prediction, gamma=gamma)
    structure = endpoint_metrics(
        prediction,
        target,
        positions=operators.cell_centers,
        edges=edges,
        volumes=operators.cell_volume,
        component_scale=component_scale,
        gamma=gamma,
        shock_quantile=shock_quantile,
    )
    return {
        "face_winv_norm": error_face_norm,
        "canonical_face_winv_norm": canonical_face_norm,
        "relative_face_error": relative_face_error,
        "decoded_error_volume_l2": decoded_error_norm,
        "target_increment_volume_l2": increment_norm,
        "physical_decoded_gain": decoded_error_norm / max(error_face_norm, 1.0e-30),
        "relative_increment_error": relative_increment_error,
        "relative_error_gain": relative_error_gain,
        "state_scaled_relative_l2": _volume_scaled_relative_l2(
            prediction,
            target,
            volume=operators.cell_volume,
            component_scale=component_scale,
        ),
        **{f"admissibility_{key}": value for key, value in admissibility.items()},
        **structure,
    }


def _interior_edges(operators: FVImpulseOperators) -> np.ndarray:
    index = operators.interior_face_indices
    return np.stack(
        (operators.face_owner[index], operators.face_neighbor[index]), axis=-1
    )


def _physical_rows(
    arrays: Mapping[str, np.ndarray],
    operators: FVImpulseOperators,
    *,
    intervals: Sequence[int],
    levels: Sequence[float],
    seeds: Sequence[int],
    lowpass_steps: int,
    transition_steps: int,
    shock_quantile: float,
    gamma: float,
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]
]:
    states = np.asarray(arrays["conservative_states"], dtype=np.float64)
    times = np.asarray(arrays["physical_times"], dtype=np.float64)
    impulses = np.asarray(
        arrays["cumulative_accepted_substep_face_impulses"], dtype=np.float64
    )
    if any(interval <= 0 or interval >= states.shape[0] for interval in intervals):
        raise ValueError("every interval must select a saved endpoint after t=0")
    if any(level <= 0.0 or not np.isfinite(level) for level in levels):
        raise ValueError("perturbation levels must be finite and positive")
    component_scale = np.std(states.reshape(-1, 4), axis=0)
    if np.any(component_scale <= 0.0) or not np.all(np.isfinite(component_scale)):
        raise ValueError("reference-derived component scales must be positive")

    modes_by_seed = {
        int(seed): interior_divergence_band_modes(
            operators,
            seed=int(seed),
            lowpass_steps=lowpass_steps,
            transition_steps=transition_steps,
        )
        for seed in seeds
    }
    mesh_nx = int(np.unique(operators.cell_centers[:, 0]).size)
    mesh_ny = int(np.unique(operators.cell_centers[:, 1]).size)
    band_rows = [
        {
            "mesh_nx": mesh_nx,
            "mesh_ny": mesh_ny,
            "evidence_scope": "validated_dynamic_reference_mesh",
            **mode.summary(),
        }
        for modes in modes_by_seed.values()
        for mode in modes
    ]
    projector = factorize_direct_minimum_winv_norm_projector(operators)
    boundary_index = operators.boundary_face_indices
    edges = _interior_edges(operators)
    canonical_rows: list[dict[str, Any]] = []
    perturbation_rows: list[dict[str, Any]] = []

    for interval in intervals:
        current = states[0]
        target = states[interval]
        reference_face = np.sum(impulses[:interval], axis=0)
        target_cell_integral = -operators.cell_volume[:, None] * (target - current)
        canonical = projector.solve(
            target_cell_integral,
            reference_face[boundary_index],
        )
        canonical_face = canonical.face_impulse
        cycle = reference_face - canonical_face
        boundary_field = np.zeros_like(canonical_face)
        boundary_field[boundary_index] = canonical_face[boundary_index]
        target_increment = target - current
        canonical_component_norm = _winv_component_norm(
            canonical_face, operators.face_weight
        )
        target_component_norm = _volume_component_norm(
            target_increment, operators.cell_volume
        )
        target_component_gain = target_component_norm / np.maximum(
            canonical_component_norm, 1.0e-30
        )
        cycle_decoded = -np.asarray(operators.incidence @ cycle) / (
            operators.cell_volume[:, None]
        )
        cycle_gain = float(
            np.linalg.norm(_volume_component_norm(cycle_decoded, operators.cell_volume))
            / max(_winv_norm(cycle, operators.face_weight), 1.0e-30)
        )
        canonical_rows.append(
            {
                "interval": int(interval),
                "time_start": float(times[0]),
                "time_end": float(times[interval]),
                "physical_horizon": float(times[interval] - times[0]),
                "canonical_face_winv_norm": _winv_norm(
                    canonical_face, operators.face_weight
                ),
                "reference_face_winv_norm": _winv_norm(
                    reference_face, operators.face_weight
                ),
                "cycle_face_winv_norm": _winv_norm(cycle, operators.face_weight),
                "boundary_face_winv_norm": _winv_norm(
                    boundary_field, operators.face_weight
                ),
                "cycle_decoded_gain": cycle_gain,
                "cycle_energy_fraction": (
                    _winv_norm(cycle, operators.face_weight) ** 2
                    / max(
                        _winv_norm(reference_face, operators.face_weight) ** 2, 1.0e-30
                    )
                ),
                "canonical_component_winv_norm": canonical_component_norm.tolist(),
                "target_increment_component_volume_l2": target_component_norm.tolist(),
                "target_component_decoded_gain": target_component_gain.tolist(),
                **canonical.summary(),
            }
        )

        for level in levels:
            for polarity in (-1, 1):
                common = {
                    "interval": int(interval),
                    "time_end": float(times[interval]),
                    "physical_horizon": float(times[interval] - times[0]),
                    "level": float(level),
                    "polarity": int(polarity),
                }
                for subspace, source in (
                    ("cycle_nullspace", cycle),
                    ("boundary_exchange", boundary_field),
                ):
                    error_face = _scale_component_field(
                        source,
                        face_weight=operators.face_weight,
                        target_component_norm=canonical_component_norm,
                        level=float(level),
                        polarity=int(polarity),
                    )
                    perturbation_rows.append(
                        {
                            **common,
                            "subspace": subspace,
                            "seed": None,
                            "band": None,
                            "mode_decoded_gain": None,
                            "mode_normalized_frequency": None,
                            **_evaluate_error(
                                error_face=error_face,
                                canonical_face=canonical_face,
                                current=current,
                                target=target,
                                operators=operators,
                                edges=edges,
                                component_scale=component_scale,
                                gamma=gamma,
                                shock_quantile=shock_quantile,
                            ),
                        }
                    )
                for seed, modes in modes_by_seed.items():
                    for mode in modes:
                        error_face = _scale_band_mode(
                            mode,
                            target_component_norm=canonical_component_norm,
                            level=float(level),
                            polarity=int(polarity),
                        )
                        perturbation_rows.append(
                            {
                                **common,
                                "subspace": "divergence_active_band",
                                "seed": int(seed),
                                "band": mode.band,
                                "mode_decoded_gain": mode.decoded_gain,
                                "mode_normalized_frequency": (
                                    mode.normalized_frequency
                                ),
                                **_evaluate_error(
                                    error_face=error_face,
                                    canonical_face=canonical_face,
                                    current=current,
                                    target=target,
                                    operators=operators,
                                    edges=edges,
                                    component_scale=component_scale,
                                    gamma=gamma,
                                    shock_quantile=shock_quantile,
                                ),
                            }
                        )

    return canonical_rows, band_rows, perturbation_rows, projector.summary()


def _geometry_resolution_rows(
    *,
    resolutions: Sequence[tuple[int, int]],
    seeds: Sequence[int],
    lowpass_steps: int,
    transition_steps: int,
    macro_dt: float,
    reference_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for nx, ny in resolutions:
        config = ShockVortexFVConfig(
            nx=nx,
            ny=ny,
            coarse_nx=nx,
            coarse_ny=ny,
            x_min=float(reference_config["x_min"]),
            x_max=float(reference_config["x_max"]),
            y_min=float(reference_config["y_min"]),
            y_max=float(reference_config["y_max"]),
            gamma=float(reference_config["gamma"]),
        )
        geometry = make_structured_fv_geometry(config)
        operators = build_fv_impulse_operators(
            cell_centers=geometry.cell_centers,
            cell_volume=geometry.cell_volume,
            face_centers=geometry.face_centers,
            face_measure=geometry.face_measure,
            face_owner=geometry.face_owner,
            face_neighbor=geometry.face_neighbor,
            face_boundary_tag=geometry.face_boundary_tag,
        )
        top = decoder_gain_summary(
            operators,
            dt=macro_dt,
            tolerance=1.0e-8,
            max_iterations=25_000,
        )
        for seed in seeds:
            modes = interior_divergence_band_modes(
                operators,
                seed=int(seed),
                lowpass_steps=lowpass_steps,
                transition_steps=transition_steps,
            )
            by_band = {mode.band: mode for mode in modes}
            rows.append(
                {
                    "mesh_nx": int(nx),
                    "mesh_ny": int(ny),
                    "num_cells": operators.topology.num_cells,
                    "num_faces": operators.topology.num_faces,
                    "seed": int(seed),
                    "macro_dt": float(macro_dt),
                    "impulse_top_gain": top.impulse_gain.value,
                    "impulse_top_gain_relative_residual": (
                        top.impulse_gain.relative_residual
                    ),
                    "flux_top_gain": top.flux_gain.value,
                    "flux_top_gain_relative_residual": top.flux_gain.relative_residual,
                    **{
                        f"{band}_decoded_gain": by_band[band].decoded_gain
                        for band in ("low", "mid", "high")
                    },
                    **{
                        f"{band}_normalized_frequency": (
                            by_band[band].normalized_frequency
                        )
                        for band in ("low", "mid", "high")
                    },
                    "evidence_scope": (
                        "analytic_cartesian_geometry_only; no native dynamic "
                        "states or reference face impulses"
                    ),
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            )
    return rows


def _aggregate(
    canonical_rows: Sequence[Mapping[str, Any]],
    band_rows: Sequence[Mapping[str, Any]],
    resolution_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not canonical_rows or not band_rows or not resolution_rows:
        raise ValueError("D049 aggregation requires all three row families")
    canonical_closure = max(
        float(row["decoded_residual_relative_l2"]) for row in canonical_rows
    )
    cycle_gain = max(float(row["cycle_decoded_gain"]) for row in canonical_rows)

    physical_by_seed: dict[int, dict[str, float]] = {}
    for row in band_rows:
        physical_by_seed.setdefault(int(row["seed"]), {})[str(row["band"])] = float(
            row["decoded_gain"]
        )
    if any(
        set(values) != {"low", "mid", "high"} for values in physical_by_seed.values()
    ):
        raise ValueError("each physical seed must contain low, mid, and high bands")
    ordered = [
        values["low"] < values["mid"] < values["high"]
        for values in physical_by_seed.values()
    ]
    band_order_fraction = float(np.mean(ordered))
    high_to_low = min(
        values["high"] / max(values["low"], 1.0e-30)
        for values in physical_by_seed.values()
    )
    band_medians = {
        band: float(np.median([values[band] for values in physical_by_seed.values()]))
        for band in ("low", "mid", "high")
    }
    maximum_target_gain = max(
        max(float(value) for value in row["target_component_decoded_gain"])
        for row in canonical_rows
    )
    high_to_target = min(values["high"] for values in physical_by_seed.values()) / max(
        maximum_target_gain, 1.0e-30
    )

    flux_by_resolution: dict[tuple[int, int], list[float]] = {}
    for row in resolution_rows:
        key = (int(row["mesh_nx"]), int(row["mesh_ny"]))
        flux_by_resolution.setdefault(key, []).append(float(row["flux_top_gain"]))
    ordered_resolutions = sorted(flux_by_resolution, key=lambda item: item[0] * item[1])
    if len(ordered_resolutions) < 2:
        raise ValueError("D049 needs at least two algebraic resolution controls")
    median_flux = {
        key: float(np.median(flux_by_resolution[key])) for key in ordered_resolutions
    }
    refinement_ratios = [
        median_flux[current] / max(median_flux[previous], 1.0e-30)
        for previous, current in zip(
            ordered_resolutions[:-1], ordered_resolutions[1:], strict=True
        )
    ]
    measurements = {
        "canonical_closure_relative_l2_max": canonical_closure,
        "cycle_decoded_gain_max": cycle_gain,
        "band_order_fraction_min": band_order_fraction,
        "high_to_low_gain_ratio_min": high_to_low,
        "high_to_target_gain_ratio_min": high_to_target,
        "flux_gain_refinement_ratio_min": min(refinement_ratios),
    }
    checks = {}
    for name, threshold in GATE_THRESHOLDS.items():
        if name.endswith("_max"):
            checks[name] = bool(measurements[name] <= threshold)
        else:
            checks[name] = bool(measurements[name] >= threshold)
    passed = all(checks.values())
    d048_low, d048_high = DOCUMENTED_D048_ERROR_GAIN_RANGE
    if d048_low > band_medians["mid"] and d048_high <= band_medians["high"]:
        d048_context = "documented_error_gain_lies_between_mid_and_high_band_medians"
    else:
        d048_context = "documented_error_gain_not_bracketed_by_mid_and_high_medians"
    return {
        "passed": passed,
        "classification": (
            "supports_face_norm_discrete_divergence_conditioning"
            if passed
            else "divergence_conditioning_gate_not_closed"
        ),
        "next_route": (
            "authorize_D050_zero_training_residual_to_face_preflight_only"
            if passed
            else "stop_before_D050_and_reaudit_D049"
        ),
        "measurements": measurements,
        "thresholds": dict(GATE_THRESHOLDS),
        "checks": checks,
        "physical_band_median_decoded_gain": band_medians,
        "maximum_canonical_target_component_gain": maximum_target_gain,
        "geometry_only_flux_top_gain": {
            f"{nx}x{ny}": median_flux[(nx, ny)] for nx, ny in ordered_resolutions
        },
        "geometry_only_flux_refinement_ratios": refinement_ratios,
        "documented_D048_error_gain_range": list(DOCUMENTED_D048_ERROR_GAIN_RANGE),
        "documented_D048_context": d048_context,
    }


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, allow_nan=False)
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table {path.name}")
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _validated_arguments(args: argparse.Namespace) -> dict[str, Any]:
    intervals = tuple(dict.fromkeys(int(value) for value in args.intervals))
    levels = tuple(dict.fromkeys(float(value) for value in args.levels))
    seeds = tuple(dict.fromkeys(int(value) for value in args.seeds))
    resolutions = tuple(
        dict.fromkeys(_parse_resolution(value) for value in args.resolutions)
    )
    if not intervals or not levels or not seeds:
        raise ValueError("intervals, levels, and seeds must be nonempty")
    if len(resolutions) < 2:
        raise ValueError("at least two geometry-only resolutions are required")
    if not 0.0 < args.shock_quantile < 1.0:
        raise ValueError("shock quantile must lie strictly between zero and one")
    if args.macro_dt <= 0.0 or not np.isfinite(args.macro_dt):
        raise ValueError("macro dt must be finite and positive")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    return {
        "intervals": intervals,
        "levels": levels,
        "seeds": seeds,
        "resolutions": resolutions,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    values = _validated_arguments(args)
    arrays, source = _load_reference(args.reference_artifact)
    operators = _operators_from_reference(arrays)
    config = source["config"]
    canonical_rows, band_rows, perturbation_rows, projector = _physical_rows(
        arrays,
        operators,
        intervals=values["intervals"],
        levels=values["levels"],
        seeds=values["seeds"],
        lowpass_steps=args.lowpass_steps,
        transition_steps=args.transition_steps,
        shock_quantile=args.shock_quantile,
        gamma=float(config["gamma"]),
    )
    expected_perturbations = (
        len(values["intervals"])
        * len(values["levels"])
        * 2
        * (2 + 3 * len(values["seeds"]))
    )
    if len(perturbation_rows) != expected_perturbations:
        raise RuntimeError("perturbation table is incomplete")
    resolution_rows = _geometry_resolution_rows(
        resolutions=values["resolutions"],
        seeds=values["seeds"],
        lowpass_steps=args.lowpass_steps,
        transition_steps=args.transition_steps,
        macro_dt=args.macro_dt,
        reference_config=config,
    )
    aggregate = _aggregate(canonical_rows, band_rows, resolution_rows)

    args.output_dir.mkdir(parents=True)
    tables = {
        "canonical_rows.csv": canonical_rows,
        "band_rows.csv": band_rows,
        "perturbation_rows.csv": perturbation_rows,
        "resolution_rows.csv": resolution_rows,
    }
    for name, rows in tables.items():
        _write_csv(args.output_dir / name, rows)

    run_config = {
        "intervals": list(values["intervals"]),
        "levels": list(values["levels"]),
        "seeds": list(values["seeds"]),
        "lowpass_steps": int(args.lowpass_steps),
        "transition_steps": int(args.transition_steps),
        "shock_quantile": float(args.shock_quantile),
        "macro_dt": float(args.macro_dt),
        "geometry_only_resolutions": [
            [int(nx), int(ny)] for nx, ny in values["resolutions"]
        ],
    }
    summary = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "passed" if aggregate["passed"] else "failed",
        "passed": bool(aggregate["passed"]),
        "entrypoint": Path(__file__).relative_to(ROOT).as_posix(),
        "entrypoint_sha256": _sha256_file(Path(__file__)),
        "configuration": run_config,
        "configuration_digest": _digest_mapping(run_config),
        "source_reference": source,
        "physical_mesh_topology": operators.topology.to_dict(),
        "direct_projector": projector,
        "coordinate_convention": str(arrays["coordinate_convention"].item()),
        "face_orientation_convention": str(
            arrays["face_orientation_convention"].item()
        ),
        "state_convention": str(arrays["state_convention"].item()),
        "boundary_mode": str(arrays["boundary_mode"].item()),
        "diagnostic_weights": {
            "cell_state_norm": "physical cell volume",
            "face_field_norm": "sum_f I_f^2 / (A_f d_f)",
            "face_weight": "validated face measure times owner-neighbor dual width",
            "component_scaling": "standard deviation over accepted reference states",
        },
        "row_counts": {
            "canonical": len(canonical_rows),
            "physical_band": len(band_rows),
            "perturbation": len(perturbation_rows),
            "geometry_resolution": len(resolution_rows),
        },
        "result": aggregate,
        "evidence_scope": {
            "physical": (
                "accepted dynamic states, geometry, orientation, boundary exchange, "
                "and cumulative face impulses on the common validated 250x100 mesh"
            ),
            "refinement": (
                "analytic Cartesian decoder algebra only; no native 500x200 or "
                "1000x400 dynamic state or face-impulse evidence"
            ),
            "D048_context": (
                "documented prior gain range only; D048 arrays are not regenerated "
                "or reinterpreted as physical flux truth"
            ),
        },
        "unsupported_claims": [
            "a learned PCNO branch caused the measured conditioning",
            "native fine-grid shock trajectories exhibit the same scaling",
            "a decoded learned face field is a physically correct flux",
            "D050 will improve the frozen D044 rollout",
            "the finding generalizes beyond this mesh family and norm contract",
        ],
        "intervention": "none; frozen zero-training diagnostic",
        "claim_boundary": CLAIM_BOUNDARY,
        "table_sha256": {name: _sha256_file(args.output_dir / name) for name in tables},
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "status": summary["status"],
                "classification": aggregate["classification"],
                "next_route": aggregate["next_route"],
                "summary": summary_path.as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0 if aggregate["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
