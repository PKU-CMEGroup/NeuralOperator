#!/usr/bin/env python3
"""Audit three nested D037 shock--vortex finite-volume reference artifacts."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utility.time_dependent_no.fv_impulse_diagnostics import (  # noqa: E402
    CLAIM_BOUNDARY as IMPULSE_CLAIM_BOUNDARY,
    FVImpulseOperators,
    build_fv_impulse_operators,
    decoder_gain_summary,
    decompose_interior_impulse_error,
)
from utility.time_dependent_no.shock_vortex_fv import (  # noqa: E402
    BOUNDARY_TAG_NAMES,
    REFERENCE_CONTRACT_CHECK_KEYS,
    ShockVortexFVConfig,
    make_structured_fv_geometry,
)

CANONICAL_RESOLUTIONS = ((250, 100), (500, 200), (1000, 400))
CANONICAL_COARSE_RESOLUTION = (250, 100)
AUDIT_SCHEMA = "shock_vortex_fv_convergence_audit_v3"
PRIMARY_ARTIFACT_SCHEMA = "shock_vortex_fv_reference_v2"
PRIMARY_DTYPE = "float64"
PRIMARY_SOLVER_METHOD = "dimension-by-dimension primitive WENO5-JS + HLLC + SSPRK3"
PRIMARY_BOUNDARY_MODE = "linear x extrapolation; y symmetry"
COORDINATE_CONVENTION = "row-major cell averages; x increases right, y increases up"
STATE_CONVENTION = "[rho,rho*u,rho*v,total_energy]"
FACE_ORIENTATION_CONVENTION = (
    "owner outward on boundary; owner-to-neighbor on interior faces"
)
INITIAL_QUADRATURE_SCOPE = (
    "full fine-grid conservative cell averages compared before restriction"
)
INITIAL_STATE_CONTRACT = (
    "conservative cell averages; shock-crossing cells split exactly; "
    "full fine-grid tensor Gauss-Legendre quadrature certified at doubled order"
)
INDEPENDENT_CONTRACT_CHECK_KEYS = {
    "shock_vortex_pyro_reference_v1": frozenset(
        {
            "all_saved_times_reached_exactly",
            "raw_states_finite",
            "raw_density_pressure_admissible",
            "cell_order_and_geometry_valid",
        }
    ),
    "shock_vortex_sharpclaw_reference_v1": frozenset(
        {
            "pinned_clawpack_version_and_build",
            "solver_contract_exact",
            "all_saved_times_reached_exactly",
            "initial_cell_average_quadrature_certified",
            "initial_state_loaded_without_reordering",
            "raw_states_finite",
            "raw_density_pressure_admissible",
            "cell_order_and_geometry_valid",
        }
    ),
    "shock_vortex_sharpclaw_reference_v2": frozenset(
        {
            "pinned_clawpack_version_and_build",
            "solver_contract_exact",
            "all_saved_times_reached_exactly",
            "initial_cell_average_quadrature_certified",
            "initial_state_loaded_without_reordering",
            "raw_states_finite",
            "raw_density_pressure_admissible",
            "cell_order_and_geometry_valid",
        }
    ),
}
ACCEPTED_LSMR_STOP_CODES = frozenset({0, 1, 2, 4, 5})
IMPULSE_DECOMPOSITION_TOLERANCE = 1.0e-11
CYCLE_DIVERGENCE_RELATIVE_TOLERANCE = 1.0e-8
# The benchmark is nondimensional with O(1) cumulative boundary impulses.
# This is about 45 float64 eps at unit scale and remains 100x tighter than the
# reference solver's absolute boundary-balance floor.
WALL_ZERO_FLUX_ABSOLUTE_TOLERANCE = 1.0e-14
BOUNDARY_EXCHANGE_ACTIVE_CONTRACTION_MASK = (
    (True, True, True, True),
    (True, True, True, True),
    (False, False, True, False),
    (False, False, True, False),
)


@dataclass(frozen=True)
class ReferenceRun:
    root: Path
    summary: dict[str, Any]
    artifact_schema: str
    config: dict[str, Any]
    metadata: dict[str, Any]
    states: np.ndarray
    face_impulses: np.ndarray
    times: np.ndarray
    cell_centers: np.ndarray
    cell_volume: np.ndarray
    face_centers: np.ndarray
    face_measure: np.ndarray
    face_normal: np.ndarray
    face_owner: np.ndarray
    face_neighbor: np.ndarray
    face_axis: np.ndarray
    face_boundary_tag: np.ndarray
    boundary_tag_names: tuple[str, ...]
    coordinate_convention: str
    face_orientation_convention: str
    state_convention: str
    boundary_mode: str
    solver_method: str

    @property
    def resolution(self) -> tuple[int, int]:
        return int(self.config["nx"]), int(self.config["ny"])


@dataclass(frozen=True)
class IndependentRun:
    root: Path
    summary: dict[str, Any]
    config: dict[str, Any]
    metadata: dict[str, Any]
    states: np.ndarray
    times: np.ndarray
    cell_centers: np.ndarray
    cell_volume: np.ndarray
    solver_name: str
    solver_version: str
    artifact_schema: str
    coordinate_convention: str
    state_convention: str
    boundary_mode: str

    @property
    def resolution(self) -> tuple[int, int]:
        return int(self.config["nx"]), int(self.config["ny"])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--independent-run-dir", type=Path)
    parser.add_argument(
        "--allow-noncanonical-resolutions",
        action="store_true",
        help="Marks the result smoke-only even if numerical checks pass.",
    )
    parser.add_argument("--successive-error-ratio", type=float, default=0.9)
    parser.add_argument(
        "--initial-state-relative-tolerance", type=float, default=1.0e-10
    )
    parser.add_argument("--vortex-core-relative-tolerance", type=float, default=0.02)
    parser.add_argument(
        "--independent-state-envelope-factor",
        type=float,
        default=1.5,
        help="Multiplier on the summed coarse-medium and medium-fine state error.",
    )
    parser.add_argument(
        "--independent-vortex-core-relative-tolerance",
        type=float,
        default=0.05,
    )
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_json_mapping(
    artifact: Any,
    field: str,
    *,
    root: Path,
) -> dict[str, Any]:
    try:
        payload = json.loads(str(artifact[field].item()))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid or missing {field} in artifact: {root}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{field} must encode a JSON object: {root}")
    return payload


def _require_exact_contract_checks(
    summary: dict[str, Any],
    expected_keys: frozenset[str],
    *,
    root: Path,
) -> None:
    checks = summary.get("contract_checks")
    if not isinstance(checks, dict) or set(checks) != expected_keys:
        raise ValueError(
            f"contract-check key set mismatch under {root}: "
            f"expected {sorted(expected_keys)}, got "
            f"{sorted(checks) if isinstance(checks, dict) else type(checks).__name__}"
        )
    if not all(isinstance(value, bool) and value for value in checks.values()):
        raise ValueError(
            f"artifact contains a failed or non-boolean contract check: {root}"
        )


def _reconcile_artifact_payloads(
    *,
    summary: dict[str, Any],
    artifact_schema: str,
    config: dict[str, Any],
    metadata: dict[str, Any],
    root: Path,
) -> None:
    if summary.get("schema") != artifact_schema:
        raise ValueError(f"summary/artifact schema mismatch: {root}")
    if summary.get("config") != config:
        raise ValueError(f"summary/artifact config mismatch: {root}")
    for key, value in metadata.items():
        if summary.get(key) != value:
            raise ValueError(f"summary/artifact metadata mismatch in {key}: {root}")


def _config_from_payload(payload: dict[str, Any]) -> ShockVortexFVConfig:
    field_names = ShockVortexFVConfig.__dataclass_fields__
    try:
        arguments = {name: payload[name] for name in field_names}
    except KeyError as exc:
        raise ValueError(
            f"reference config omits required field {exc.args[0]}"
        ) from exc
    arguments["output_times"] = tuple(
        float(value) for value in arguments["output_times"]
    )
    return ShockVortexFVConfig(**arguments).validated()


def _validate_serialized_geometry(run: ReferenceRun) -> None:
    expected = make_structured_fv_geometry(_config_from_payload(run.config))
    for field in (
        "cell_centers",
        "cell_volume",
        "face_centers",
        "face_measure",
        "face_normal",
        "face_owner",
        "face_neighbor",
        "face_axis",
        "face_boundary_tag",
    ):
        if not np.array_equal(getattr(run, field), getattr(expected, field)):
            raise ValueError(
                f"serialized structured geometry mismatch in {field}: {run.root}"
            )
    if run.boundary_tag_names != BOUNDARY_TAG_NAMES:
        raise ValueError(f"boundary tag names or ordering mismatch: {run.root}")


def load_reference_run(root: Path) -> ReferenceRun:
    summary_path = root / "summary.json"
    artifact_path = root / "reference.npz"
    if not summary_path.is_file() or not artifact_path.is_file():
        raise FileNotFoundError(f"missing summary.json or reference.npz under {root}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "passed":
        raise ValueError(f"reference run did not pass its contract: {root}")
    _require_exact_contract_checks(
        summary,
        REFERENCE_CONTRACT_CHECK_KEYS,
        root=root,
    )
    expected_hash = str(summary.get("reference_artifact_sha256", ""))
    if _sha256(artifact_path) != expected_hash:
        raise ValueError(f"reference artifact hash mismatch: {root}")
    with np.load(artifact_path, allow_pickle=False) as artifact:
        artifact_schema = str(artifact["schema"].item())
        if artifact_schema != PRIMARY_ARTIFACT_SCHEMA:
            raise ValueError(f"unsupported reference schema: {root}")
        config = _load_json_mapping(artifact, "config_json", root=root)
        metadata = _load_json_mapping(artifact, "metadata_json", root=root)
        _reconcile_artifact_payloads(
            summary=summary,
            artifact_schema=artifact_schema,
            config=config,
            metadata=metadata,
            root=root,
        )
        if summary.get("config_digest") != _json_digest(config):
            raise ValueError(f"reference config digest mismatch: {root}")
        times = np.array(artifact["physical_times"], copy=True)
        physical_delta_t = np.array(artifact["physical_delta_t"], copy=True)
        if physical_delta_t.shape != (max(times.size - 1, 0),) or not np.array_equal(
            physical_delta_t,
            np.diff(times),
        ):
            raise ValueError(f"physical_delta_t does not match saved times: {root}")
        try:
            boundary_tag_names = tuple(
                str(value)
                for value in json.loads(str(artifact["boundary_tag_names_json"].item()))
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid boundary_tag_names_json: {root}") from exc
        run = ReferenceRun(
            root=root,
            summary=summary,
            artifact_schema=artifact_schema,
            config=config,
            metadata=metadata,
            states=np.array(artifact["conservative_states"], copy=True),
            face_impulses=np.array(
                artifact["cumulative_accepted_substep_face_impulses"], copy=True
            ),
            times=times,
            cell_centers=np.array(artifact["cell_centers"], copy=True),
            cell_volume=np.array(artifact["cell_volume"], copy=True),
            face_centers=np.array(artifact["face_centers"], copy=True),
            face_measure=np.array(artifact["face_measure"], copy=True),
            face_normal=np.array(artifact["face_normal"], copy=True),
            face_owner=np.array(artifact["face_owner"], copy=True),
            face_neighbor=np.array(artifact["face_neighbor"], copy=True),
            face_axis=np.array(artifact["face_axis"], copy=True),
            face_boundary_tag=np.array(artifact["face_boundary_tag"], copy=True),
            boundary_tag_names=boundary_tag_names,
            coordinate_convention=str(artifact["coordinate_convention"].item()),
            face_orientation_convention=str(
                artifact["face_orientation_convention"].item()
            ),
            state_convention=str(artifact["state_convention"].item()),
            boundary_mode=str(artifact["boundary_mode"].item()),
            solver_method=str(artifact["solver_method"].item()),
        )
    _validate_serialized_geometry(run)
    return run


def load_independent_run(root: Path) -> IndependentRun:
    summary_path = root / "summary.json"
    artifact_path = root / "reference.npz"
    if not summary_path.is_file() or not artifact_path.is_file():
        raise FileNotFoundError(
            f"missing independent summary.json or reference.npz under {root}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "passed":
        raise ValueError(f"independent run did not pass its contract: {root}")
    if _sha256(artifact_path) != str(summary.get("reference_artifact_sha256", "")):
        raise ValueError(f"independent artifact hash mismatch: {root}")
    with np.load(artifact_path, allow_pickle=False) as artifact:
        artifact_schema = str(artifact["schema"].item())
        registry = {
            "shock_vortex_pyro_reference_v1": (
                "pyro",
                "pyro_version",
            ),
            "shock_vortex_sharpclaw_reference_v1": (
                "sharpclaw",
                "clawpack_version",
            ),
            "shock_vortex_sharpclaw_reference_v2": (
                "sharpclaw",
                "clawpack_version",
            ),
        }
        if artifact_schema not in registry:
            raise ValueError(f"unsupported independent reference schema: {root}")
        _require_exact_contract_checks(
            summary,
            INDEPENDENT_CONTRACT_CHECK_KEYS[artifact_schema],
            root=root,
        )
        config = _load_json_mapping(artifact, "config_json", root=root)
        metadata = _load_json_mapping(artifact, "metadata_json", root=root)
        _reconcile_artifact_payloads(
            summary=summary,
            artifact_schema=artifact_schema,
            config=config,
            metadata=metadata,
            root=root,
        )
        times = np.array(artifact["physical_times"], copy=True)
        if "physical_delta_t" in artifact.files:
            physical_delta_t = np.array(artifact["physical_delta_t"], copy=True)
            if physical_delta_t.shape != (
                max(times.size - 1, 0),
            ) or not np.array_equal(
                physical_delta_t,
                np.diff(times),
            ):
                raise ValueError(
                    f"independent physical_delta_t does not match saved times: {root}"
                )
        elif artifact_schema.startswith("shock_vortex_sharpclaw_reference_"):
            raise ValueError(f"SharpClaw artifact omits physical_delta_t: {root}")
        solver_name, version_field = registry[artifact_schema]
        solver_version = str(summary.get(version_field, ""))
        if not solver_version:
            raise ValueError(f"independent run omits required {version_field}: {root}")
        return IndependentRun(
            root=root,
            summary=summary,
            config=config,
            metadata=metadata,
            states=np.array(artifact["conservative_states"], copy=True),
            times=times,
            cell_centers=np.array(artifact["cell_centers"], copy=True),
            cell_volume=np.array(artifact["cell_volume"], copy=True),
            solver_name=solver_name,
            solver_version=solver_version,
            artifact_schema=artifact_schema,
            coordinate_convention=str(artifact["coordinate_convention"].item()),
            state_convention=str(artifact["state_convention"].item()),
            boundary_mode=str(artifact["boundary_mode"].item()),
        )


def _primitive(conservative: np.ndarray, gamma: float) -> np.ndarray:
    rho = conservative[..., 0]
    u = conservative[..., 1] / rho
    v = conservative[..., 2] / rho
    pressure = (gamma - 1.0) * (conservative[..., 3] - 0.5 * rho * (u * u + v * v))
    return np.stack((rho, u, v, pressure), axis=-1)


def pair_metrics(
    candidate_states: np.ndarray,
    reference_states: np.ndarray,
    *,
    physical_times: np.ndarray,
    cell_volume: np.ndarray,
    cell_centers: np.ndarray,
    coarse_nx: int,
    coarse_ny: int,
    gamma: float,
) -> list[dict[str, float]]:
    """Compute time-resolved nested-grid differences on the common coarse mesh."""

    candidate = np.asarray(candidate_states, dtype=np.float64)
    reference = np.asarray(reference_states, dtype=np.float64)
    times = np.asarray(physical_times, dtype=np.float64)
    volume = np.asarray(cell_volume, dtype=np.float64)
    centers = np.asarray(cell_centers, dtype=np.float64)
    if candidate.shape != reference.shape or candidate.ndim != 3:
        raise ValueError("candidate and reference states must match [time,cells,4]")
    if times.shape != (candidate.shape[0],):
        raise ValueError("physical_times must align with the state time axis")
    if candidate.shape[1:] != (coarse_nx * coarse_ny, 4):
        raise ValueError("state shape does not match declared coarse mesh")
    if volume.shape != (candidate.shape[1],) or centers.shape != (
        candidate.shape[1],
        2,
    ):
        raise ValueError("cell geometry does not align with states")

    weighted_difference = np.sum(
        volume[None, :, None] * np.abs(candidate - reference), axis=1
    )
    weighted_reference = np.sum(volume[None, :, None] * np.abs(reference), axis=1)
    component_relative_l1 = weighted_difference / np.maximum(
        weighted_reference, 1.0e-30
    )
    normalized_state_l1 = np.mean(component_relative_l1, axis=1)

    candidate_primitive = _primitive(candidate, gamma)
    reference_primitive = _primitive(reference, gamma)
    y_values = centers[:, 1].reshape(coarse_ny, coarse_nx)[:, 0]
    center_rows = np.argsort(np.abs(y_values - 0.5))[:2]
    candidate_primitive_grid = candidate_primitive.reshape(
        candidate.shape[0], coarse_ny, coarse_nx, 4
    )
    reference_primitive_grid = reference_primitive.reshape(
        reference.shape[0], coarse_ny, coarse_nx, 4
    )
    candidate_density = (
        candidate_primitive[..., 0]
        .reshape(candidate.shape[0], coarse_ny, coarse_nx)[:, center_rows]
        .mean(axis=1)
    )
    reference_density = (
        reference_primitive[..., 0]
        .reshape(reference.shape[0], coarse_ny, coarse_nx)[:, center_rows]
        .mean(axis=1)
    )
    centerline_density_l1 = np.mean(
        np.abs(candidate_density - reference_density), axis=1
    ) / np.maximum(np.mean(np.abs(reference_density), axis=1), 1.0e-30)
    x_values = centers[:, 0].reshape(coarse_ny, coarse_nx)[0]
    x_faces = 0.5 * (x_values[1:] + x_values[:-1])
    outer_rows = np.abs(y_values - 0.5) >= 0.25
    if not np.any(outer_rows):
        outer_rows = np.ones_like(y_values, dtype=bool)
    candidate_pressure_profile = candidate_primitive_grid[:, outer_rows, :, 3].mean(
        axis=1
    )
    reference_pressure_profile = reference_primitive_grid[:, outer_rows, :, 3].mean(
        axis=1
    )
    shock_face_mask = (x_faces >= 0.3) & (x_faces <= 0.7)
    if not np.any(shock_face_mask):
        raise ValueError("coarse grid does not resolve the fixed shock search window")
    shock_faces = x_faces[shock_face_mask]
    candidate_shock = shock_faces[
        np.argmax(
            np.abs(np.diff(candidate_pressure_profile, axis=1))[:, shock_face_mask],
            axis=1,
        )
    ]
    reference_shock = shock_faces[
        np.argmax(
            np.abs(np.diff(reference_pressure_profile, axis=1))[:, shock_face_mask],
            axis=1,
        )
    ]

    x_grid = centers[:, 0].reshape(coarse_ny, coarse_nx)
    y_grid = centers[:, 1].reshape(coarse_ny, coarse_nx)
    upstream_velocity = 1.1 * np.sqrt(gamma)
    shock_arrival = (0.5 - 0.25) / upstream_velocity
    expected_vortex_x = np.where(
        times <= shock_arrival,
        0.25 + upstream_velocity * times,
        0.5 + 1.1133 * (times - shock_arrival),
    )
    candidate_core = np.empty(times.size, dtype=np.float64)
    reference_core = np.empty(times.size, dtype=np.float64)
    for time_index, center_x in enumerate(expected_vortex_x):
        vortex_window = (x_grid - center_x) ** 2 + (y_grid - 0.5) ** 2 <= 0.18**2
        if not np.any(vortex_window):
            raise ValueError("coarse grid does not resolve the moving vortex window")
        candidate_core[time_index] = np.min(
            candidate_primitive_grid[time_index, :, :, 0][vortex_window]
        )
        reference_core[time_index] = np.min(
            reference_primitive_grid[time_index, :, :, 0][vortex_window]
        )

    rows: list[dict[str, float]] = []
    for time_index in range(candidate.shape[0]):
        rows.append(
            {
                "time_index": float(time_index),
                "normalized_state_relative_l1": float(normalized_state_l1[time_index]),
                "density_relative_l1": float(component_relative_l1[time_index, 0]),
                "x_momentum_relative_l1": float(component_relative_l1[time_index, 1]),
                "y_momentum_relative_l1": float(component_relative_l1[time_index, 2]),
                "energy_relative_l1": float(component_relative_l1[time_index, 3]),
                "centerline_density_relative_l1": float(
                    centerline_density_l1[time_index]
                ),
                "shock_position_difference": float(
                    abs(candidate_shock[time_index] - reference_shock[time_index])
                ),
                "vortex_core_density_relative_difference": float(
                    abs(candidate_core[time_index] - reference_core[time_index])
                    / max(abs(reference_core[time_index]), 1.0e-30)
                ),
            }
        )
    return rows


def _boundary_exchange_l2(
    face_impulses: np.ndarray,
    face_boundary_tag: np.ndarray,
) -> np.ndarray:
    impulses = np.asarray(face_impulses, dtype=np.float64)
    tags = np.asarray(face_boundary_tag)
    if impulses.ndim != 3 or impulses.shape[1] != tags.size or impulses.shape[2] != 4:
        raise ValueError("face impulses must have shape [intervals,faces,4]")
    if not np.all(np.isfinite(impulses)):
        raise ValueError("face impulses must be finite")
    exchanges = []
    for tag in range(1, 5):
        selected = tags == tag
        if not np.any(selected):
            raise ValueError(f"boundary tag {tag} has no faces")
        net_exchange = np.sum(impulses[:, selected], axis=1)
        exchanges.append(np.linalg.norm(net_exchange, axis=0))
    return np.stack(exchanges, axis=0)


def _boundary_exchange_gate(
    coarse_error_l2: np.ndarray,
    fine_error_l2: np.ndarray,
    primary_exchange_l2: np.ndarray,
    *,
    contraction_limit: float,
    wall_zero_flux_absolute_tolerance: float = WALL_ZERO_FLUX_ABSOLUTE_TOLERANCE,
) -> dict[str, Any]:
    """Gate active boundary exchange by contraction and exact wall zeros absolutely."""

    coarse = np.asarray(coarse_error_l2, dtype=np.float64)
    fine = np.asarray(fine_error_l2, dtype=np.float64)
    primary = np.asarray(primary_exchange_l2, dtype=np.float64)
    expected_shape = (4, 4)
    if coarse.shape != expected_shape or fine.shape != expected_shape:
        raise ValueError(
            "boundary exchange errors must have shape [4 tags,4 components]"
        )
    if (
        primary.ndim != 3
        or primary.shape[0] == 0
        or primary.shape[1:] != expected_shape
    ):
        raise ValueError(
            "primary boundary exchanges must have shape [runs,4 tags,4 components]"
        )
    if not all(np.all(np.isfinite(array)) for array in (coarse, fine, primary)):
        raise ValueError("boundary exchange diagnostics must be finite")
    if any(np.any(array < 0.0) for array in (coarse, fine, primary)):
        raise ValueError("boundary exchange diagnostics must be nonnegative")
    if not 0.0 < contraction_limit < 1.0:
        raise ValueError("boundary exchange contraction limit must lie in (0,1)")
    if (
        not np.isfinite(wall_zero_flux_absolute_tolerance)
        or wall_zero_flux_absolute_tolerance <= 0.0
    ):
        raise ValueError("wall zero-flux absolute tolerance must be positive")

    active_mask = np.asarray(
        BOUNDARY_EXCHANGE_ACTIVE_CONTRACTION_MASK,
        dtype=bool,
    )
    wall_zero_flux_mask = ~active_mask
    contraction = fine / np.maximum(coarse, 1.0e-30)
    active_passed = bool(np.all(contraction[active_mask] <= contraction_limit))
    wall_zero_flux_error_passed = bool(
        np.all(coarse[wall_zero_flux_mask] <= wall_zero_flux_absolute_tolerance)
        and np.all(fine[wall_zero_flux_mask] <= wall_zero_flux_absolute_tolerance)
    )
    wall_zero_flux_primary_exchange_passed = bool(
        np.all(primary[:, wall_zero_flux_mask] <= wall_zero_flux_absolute_tolerance)
    )
    wall_zero_flux_passed = bool(
        wall_zero_flux_error_passed and wall_zero_flux_primary_exchange_passed
    )
    return {
        "contraction": contraction,
        "active_mask": active_mask,
        "wall_zero_flux_mask": wall_zero_flux_mask,
        "active_passed": active_passed,
        "wall_zero_flux_error_passed": wall_zero_flux_error_passed,
        "wall_zero_flux_primary_exchange_passed": (
            wall_zero_flux_primary_exchange_passed
        ),
        "wall_zero_flux_passed": wall_zero_flux_passed,
        "combined_passed": bool(active_passed and wall_zero_flux_passed),
    }


def impulse_pair_metrics(
    candidate: ReferenceRun,
    reference: ReferenceRun,
    operators: FVImpulseOperators,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compare same-mesh restricted face impulses and their decoded updates."""

    if candidate.face_impulses.shape != reference.face_impulses.shape:
        raise ValueError("paired face-impulse arrays must have identical shapes")
    expected_shape = (
        reference.times.size - 1,
        operators.topology.num_faces,
        4,
    )
    if reference.face_impulses.shape != expected_shape:
        raise ValueError("face-impulse shape does not match time and geometry")
    error_energy = 0.0
    divergence_energy = 0.0
    cycle_energy = 0.0
    boundary_energy = 0.0
    error_energy_by_component = np.zeros(4, dtype=np.float64)
    divergence_energy_by_component = np.zeros(4, dtype=np.float64)
    cycle_energy_by_component = np.zeros(4, dtype=np.float64)
    boundary_energy_by_component = np.zeros(4, dtype=np.float64)
    boundary_exchange_error_energy = np.zeros((4, 4), dtype=np.float64)
    component_error_l1 = np.zeros(4, dtype=np.float64)
    component_reference_l1 = np.zeros(4, dtype=np.float64)
    maximum_reconstruction = 0.0
    maximum_cycle_divergence = 0.0
    maximum_orthogonality = 0.0
    maximum_decoded_residual = 0.0
    maximum_decoded_max_abs = 0.0
    all_lsmr_stop_codes: set[int] = set()
    rows: list[dict[str, Any]] = []
    for interval_index, dt in enumerate(np.diff(reference.times)):
        error = (
            candidate.face_impulses[interval_index]
            - reference.face_impulses[interval_index]
        )
        component_error_l1 += np.sum(np.abs(error), axis=0)
        component_reference_l1 += np.sum(
            np.abs(reference.face_impulses[interval_index]), axis=0
        )
        decomposition = decompose_interior_impulse_error(
            operators,
            error,
            tolerance=IMPULSE_DECOMPOSITION_TOLERANCE,
        )
        summary = decomposition.summary()
        interval_error_energy = (
            decomposition.interior_energy + decomposition.boundary_energy
        )
        error_energy += interval_error_energy
        divergence_energy += decomposition.divergence_active_energy
        cycle_energy += decomposition.cycle_energy
        boundary_energy += decomposition.boundary_energy
        interior_by_component = np.asarray(
            summary["interior_energy_by_component"], dtype=np.float64
        )
        boundary_by_component = np.asarray(
            summary["boundary_energy_by_component"], dtype=np.float64
        )
        error_energy_by_component += interior_by_component + boundary_by_component
        divergence_energy_by_component += np.asarray(
            summary["divergence_active_energy_by_component"], dtype=np.float64
        )
        cycle_energy_by_component += np.asarray(
            summary["cycle_energy_by_component"], dtype=np.float64
        )
        boundary_energy_by_component += boundary_by_component
        all_lsmr_stop_codes.update(int(value) for value in summary["lsmr_stop_codes"])

        decoded = -(operators.incidence @ error) / operators.cell_volume[:, None]
        expected = (
            candidate.states[interval_index + 1]
            - candidate.states[interval_index]
            - reference.states[interval_index + 1]
            + reference.states[interval_index]
        )
        integrated_residual = operators.cell_volume[:, None] * (decoded - expected)
        integrated_expected = operators.cell_volume[:, None] * expected
        decoded_residual = float(
            np.linalg.norm(integrated_residual)
            / max(float(np.linalg.norm(integrated_expected)), 1.0e-30)
        )
        decoded_max_abs = float(np.max(np.abs(integrated_residual), initial=0.0))
        maximum_reconstruction = max(
            maximum_reconstruction,
            float(summary["reconstruction_relative_l2"]),
        )
        maximum_cycle_divergence = max(
            maximum_cycle_divergence,
            float(summary["cycle_divergence_relative_l2"]),
        )
        maximum_orthogonality = max(
            maximum_orthogonality,
            float(summary["winv_orthogonality_relative"]),
        )
        maximum_decoded_residual = max(maximum_decoded_residual, decoded_residual)
        maximum_decoded_max_abs = max(maximum_decoded_max_abs, decoded_max_abs)
        row: dict[str, Any] = {
            "interval_index": interval_index,
            "start_time": float(reference.times[interval_index]),
            "end_time": float(reference.times[interval_index + 1]),
            "dt": float(dt),
            "full_winv_error_norm": float(np.sqrt(interval_error_energy)),
            "divergence_active_winv_error_norm": float(
                np.sqrt(decomposition.divergence_active_energy)
            ),
            "cycle_winv_error_norm": float(np.sqrt(decomposition.cycle_energy)),
            "boundary_winv_error_norm": float(np.sqrt(decomposition.boundary_energy)),
            "divergence_active_energy_fraction": float(
                decomposition.divergence_active_energy
                / max(decomposition.interior_energy, 1.0e-30)
            ),
            "cycle_energy_fraction": float(
                decomposition.cycle_energy / max(decomposition.interior_energy, 1.0e-30)
            ),
            "observed_interior_decoded_gain": float(
                summary["observed_interior_decoded_gain"]
            ),
            "observed_full_decoded_gain": float(summary["observed_full_decoded_gain"]),
            "decomposition_reconstruction_relative_l2": float(
                summary["reconstruction_relative_l2"]
            ),
            "cycle_divergence_relative_l2": float(
                summary["cycle_divergence_relative_l2"]
            ),
            "cycle_divergence_l2": float(summary["cycle_divergence_l2"]),
            "winv_orthogonality_relative": float(
                summary["winv_orthogonality_relative"]
            ),
            "decoded_transition_relative_l2": decoded_residual,
            "decoded_transition_max_abs_integrated": decoded_max_abs,
            "lsmr_stop_codes": ";".join(
                str(value) for value in summary["lsmr_stop_codes"]
            ),
            "lsmr_max_iterations": max(summary["lsmr_iterations"], default=0),
        }
        for component, name in enumerate(
            ("mass", "x_momentum", "y_momentum", "energy")
        ):
            row[f"{name}_face_impulse_relative_l1"] = float(
                np.sum(np.abs(error[:, component]))
                / max(
                    float(
                        np.sum(
                            np.abs(
                                reference.face_impulses[interval_index, :, component]
                            )
                        )
                    ),
                    1.0e-30,
                )
            )
        for tag in range(1, 5):
            selected = operators.face_boundary_tag == tag
            exchange_error = np.sum(error[selected], axis=0)
            boundary_exchange_error_energy[tag - 1] += exchange_error**2
            reference_exchange = np.sum(
                reference.face_impulses[interval_index, selected], axis=0
            )
            row[f"boundary_tag_{tag}_exchange_error_l2"] = float(
                np.linalg.norm(exchange_error)
            )
            row[f"boundary_tag_{tag}_exchange_relative_l2"] = float(
                np.linalg.norm(exchange_error)
                / max(float(np.linalg.norm(reference_exchange)), 1.0e-30)
            )
        rows.append(row)

    return rows, {
        "full_winv_error_norm": float(np.sqrt(error_energy)),
        "divergence_active_winv_error_norm": float(np.sqrt(divergence_energy)),
        "cycle_winv_error_norm": float(np.sqrt(cycle_energy)),
        "boundary_winv_error_norm": float(np.sqrt(boundary_energy)),
        "full_winv_error_norm_by_component": np.sqrt(
            error_energy_by_component
        ).tolist(),
        "divergence_active_winv_error_norm_by_component": np.sqrt(
            divergence_energy_by_component
        ).tolist(),
        "cycle_winv_error_norm_by_component": np.sqrt(
            cycle_energy_by_component
        ).tolist(),
        "boundary_winv_error_norm_by_component": np.sqrt(
            boundary_energy_by_component
        ).tolist(),
        "boundary_tag_component_exchange_error_l2": np.sqrt(
            boundary_exchange_error_energy
        ).tolist(),
        "lsmr_stop_codes": sorted(all_lsmr_stop_codes),
        "component_face_impulse_relative_l1": (
            component_error_l1 / np.maximum(component_reference_l1, 1.0e-30)
        ).tolist(),
        "maximum_decomposition_reconstruction_relative_l2": maximum_reconstruction,
        "maximum_cycle_divergence_relative_l2": maximum_cycle_divergence,
        "maximum_winv_orthogonality_relative": maximum_orthogonality,
        "maximum_decoded_transition_relative_l2": maximum_decoded_residual,
        "maximum_decoded_transition_max_abs_integrated": maximum_decoded_max_abs,
    }


def _validate_matched_contract(runs: list[ReferenceRun]) -> None:
    if len(runs) != 3:
        raise ValueError("exactly three nested reference runs are required")
    finest = runs[-1]
    ignored = {"nx", "ny", "dx", "dy", "restriction_x", "restriction_y"}
    canonical = {
        key: value for key, value in finest.config.items() if key not in ignored
    }
    for run in runs:
        compared = {
            key: value for key, value in run.config.items() if key not in ignored
        }
        if compared != canonical:
            raise ValueError(f"reference configs are not matched: {run.root}")
        if (
            run.summary["solver_utility_sha256"]
            != finest.summary["solver_utility_sha256"]
        ):
            raise ValueError("reference runs use different solver source hashes")
        if run.summary["entrypoint_sha256"] != finest.summary["entrypoint_sha256"]:
            raise ValueError("reference runs use different generator source hashes")
        if run.summary.get("dtype") != finest.summary.get("dtype"):
            raise ValueError("reference runs use different numerical dtypes")
        if not np.array_equal(run.times, finest.times):
            raise ValueError("reference runs do not share saved physical times")
        for field in (
            "artifact_schema",
            "boundary_tag_names",
            "coordinate_convention",
            "face_orientation_convention",
            "state_convention",
            "boundary_mode",
            "solver_method",
        ):
            if getattr(run, field) != getattr(finest, field):
                raise ValueError(f"reference runs disagree in {field}")
        for field in (
            "cell_centers",
            "cell_volume",
            "face_centers",
            "face_measure",
            "face_normal",
            "face_owner",
            "face_neighbor",
            "face_axis",
            "face_boundary_tag",
        ):
            if not np.array_equal(getattr(run, field), getattr(finest, field)):
                raise ValueError(f"coarse geometry mismatch in {field}: {run.root}")


def _canonical_primary_contract_checks(
    runs: list[ReferenceRun],
) -> dict[str, bool]:
    resolutions = tuple(run.resolution for run in runs)
    exact_configs = all(
        run.config == ShockVortexFVConfig(nx=nx, ny=ny).to_dict()
        for run, (nx, ny) in zip(runs, CANONICAL_RESOLUTIONS, strict=True)
    )
    comparison_mesh = all(
        (
            int(run.config["coarse_nx"]),
            int(run.config["coarse_ny"]),
        )
        == CANONICAL_COARSE_RESOLUTION
        for run in runs
    )
    exact_times = np.asarray(ShockVortexFVConfig().output_times, dtype=np.float64)
    physical_time_contract = all(np.array_equal(run.times, exact_times) for run in runs)
    solver_contract = all(
        run.artifact_schema == PRIMARY_ARTIFACT_SCHEMA
        and run.summary.get("dtype") == PRIMARY_DTYPE
        and all(
            array.dtype == np.dtype(np.float64)
            for array in (
                run.states,
                run.face_impulses,
                run.times,
                run.cell_centers,
                run.cell_volume,
                run.face_centers,
                run.face_measure,
                run.face_normal,
            )
        )
        and run.coordinate_convention == COORDINATE_CONVENTION
        and run.face_orientation_convention == FACE_ORIENTATION_CONVENTION
        and run.state_convention == STATE_CONVENTION
        and run.boundary_mode == PRIMARY_BOUNDARY_MODE
        and run.solver_method == PRIMARY_SOLVER_METHOD
        and run.metadata.get("future_reference_boundary_values") is False
        and run.metadata.get("clipping_or_accepted_state_floors") is False
        and run.metadata.get("canonical_case") is True
        and run.metadata.get("initial_quadrature_scope") == INITIAL_QUADRATURE_SCOPE
        and run.metadata.get("initial_state_contract") == INITIAL_STATE_CONTRACT
        and run.metadata.get("reference_truth_source")
        == ("numerical finite-volume evolution from the published initial condition")
        and _is_sha256(run.summary.get("entrypoint_sha256"))
        and _is_sha256(run.summary.get("solver_utility_sha256"))
        for run in runs
    )
    return {
        "canonical_three_resolution_ladder": resolutions == CANONICAL_RESOLUTIONS,
        "canonical_comparison_mesh": comparison_mesh,
        "canonical_physical_configuration_and_save_times": bool(
            exact_configs and physical_time_contract
        ),
        "canonical_primary_solver_dtype_and_metadata": solver_contract,
    }


def _validate_independent_contract(
    independent: IndependentRun,
    finest: ReferenceRun,
    *,
    coarse_nx: int,
    coarse_ny: int,
) -> None:
    if independent.resolution != (coarse_nx, coarse_ny):
        raise ValueError(
            "independent run must use the common coarse resolution "
            f"{coarse_nx}x{coarse_ny}, got {independent.resolution}"
        )
    if not np.array_equal(independent.times, finest.times):
        raise ValueError("independent run does not share saved physical times")
    if not np.allclose(
        independent.cell_centers,
        finest.cell_centers,
        rtol=0.0,
        atol=2.0e-14,
    ):
        raise ValueError("independent run uses a different coarse cell ordering")
    if not np.allclose(
        independent.cell_volume,
        finest.cell_volume,
        rtol=0.0,
        atol=2.0e-14,
    ):
        raise ValueError("independent run uses different coarse cell volumes")
    if independent.coordinate_convention != finest.coordinate_convention:
        raise ValueError("independent run uses a different coordinate convention")
    if independent.state_convention != finest.state_convention:
        raise ValueError("independent run uses a different state convention")
    for key in ("gamma", "x_min", "x_max", "y_min", "y_max", "t_final"):
        if independent.config[key] != finest.config[key]:
            raise ValueError(f"independent run config mismatch in {key}")


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _canonical_independent_solver_contract(independent: IndependentRun) -> bool:
    if independent.artifact_schema != "shock_vortex_sharpclaw_reference_v2":
        return False
    expected_config = {
        "nx": 250,
        "ny": 100,
        "x_min": 0.0,
        "x_max": 2.0,
        "y_min": 0.0,
        "y_max": 1.0,
        "gamma": 1.4,
        "t_final": 0.6,
        "output_times": list(ShockVortexFVConfig().output_times),
        "solver": "pyclaw.SharpClawSolver2D",
        "riemann_solver": "riemann.euler_4wave_2D",
        "kernel_language": "Fortran",
        "lim_type": 2,
        "weno_order": 5,
        "char_decomp": 0,
        "time_integrator": "SSP33",
        "cfl_desired": 0.35,
        "cfl_max": 0.5,
        "num_ghost": 3,
        "initial_quadrature_order": 8,
        "x_boundary": "custom linear extrapolation in primitive variables",
        "y_boundary": "reflecting wall",
    }
    return bool(
        independent.config == expected_config
        and all(
            array.dtype == np.dtype(np.float64)
            for array in (
                independent.states,
                independent.times,
                independent.cell_centers,
                independent.cell_volume,
            )
        )
        and independent.coordinate_convention == COORDINATE_CONVENTION
        and independent.state_convention == STATE_CONVENTION
        and independent.boundary_mode
        == (
            "SharpClaw custom linear primitive-variable extrapolation in x; "
            "reflecting wall in y"
        )
        and independent.metadata.get("clawpack_version") == "5.9.0"
        and independent.metadata.get("clawpack_build") == "py311h3d4ca6a_1"
        and independent.metadata.get("future_reference_boundary_values") is False
        and independent.metadata.get("clipping_or_positive_floors") is False
        and independent.metadata.get("provides_reference_face_impulses") is False
        and independent.metadata.get("state_only_independent_comparison") is True
        and _is_sha256(independent.metadata.get("clawpack_conda_record_sha256"))
        and _is_sha256(independent.metadata.get("adapter_sha256"))
    )


def independent_agreement_metrics(
    independent_rows: list[dict[str, float]],
    coarse_rows: list[dict[str, float]],
    fine_rows: list[dict[str, float]],
    *,
    coarse_dx: float,
    state_envelope_factor: float,
    vortex_core_relative_tolerance: float,
    initial_state_relative_tolerance: float = 1.0e-10,
) -> tuple[dict[str, bool], dict[str, float]]:
    """Apply the predeclared public-solver agreement envelope."""

    if not independent_rows or not (
        len(independent_rows) == len(coarse_rows) == len(fine_rows)
    ):
        raise ValueError("agreement rows must be non-empty and time-aligned")
    if state_envelope_factor <= 0.0:
        raise ValueError("independent state envelope factor must be positive")
    if vortex_core_relative_tolerance <= 0.0:
        raise ValueError("independent vortex tolerance must be positive")
    if initial_state_relative_tolerance <= 0.0:
        raise ValueError("independent initial-state tolerance must be positive")

    final_limit = state_envelope_factor * (
        coarse_rows[-1]["normalized_state_relative_l1"]
        + fine_rows[-1]["normalized_state_relative_l1"]
    )
    time_mean_limit = state_envelope_factor * (
        np.mean([row["normalized_state_relative_l1"] for row in coarse_rows[1:]])
        + np.mean([row["normalized_state_relative_l1"] for row in fine_rows[1:]])
    )
    final_error = independent_rows[-1]["normalized_state_relative_l1"]
    time_mean_error = float(
        np.mean([row["normalized_state_relative_l1"] for row in independent_rows[1:]])
    )
    checks = {
        "independent_initial_state_matches_primary": bool(
            independent_rows[0]["normalized_state_relative_l1"]
            <= initial_state_relative_tolerance
        ),
        "independent_final_state_within_discretization_envelope": bool(
            final_error <= final_limit
        ),
        "independent_time_mean_state_within_discretization_envelope": bool(
            time_mean_error <= time_mean_limit
        ),
        "independent_final_shock_position_within_one_coarse_cell": bool(
            independent_rows[-1]["shock_position_difference"] <= coarse_dx + 1.0e-14
        ),
        "independent_final_vortex_core_density_agrees": bool(
            independent_rows[-1]["vortex_core_density_relative_difference"]
            <= vortex_core_relative_tolerance
        ),
    }
    metrics = {
        "independent_initial_state_relative_l1": float(
            independent_rows[0]["normalized_state_relative_l1"]
        ),
        "independent_final_state_relative_l1": float(final_error),
        "independent_final_state_envelope": float(final_limit),
        "independent_time_mean_state_relative_l1": time_mean_error,
        "independent_time_mean_state_envelope": float(time_mean_limit),
        "independent_final_shock_position_difference": float(
            independent_rows[-1]["shock_position_difference"]
        ),
        "independent_final_vortex_core_density_relative_difference": float(
            independent_rows[-1]["vortex_core_density_relative_difference"]
        ),
    }
    return checks, metrics


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(
            f"output directory already exists; choose a new path: {args.output_dir}"
        )
    if not 0.0 < args.successive_error_ratio < 1.0:
        raise ValueError("successive error ratio must lie in (0,1)")
    if args.vortex_core_relative_tolerance <= 0.0:
        raise ValueError("vortex core tolerance must be positive")
    if args.initial_state_relative_tolerance <= 0.0:
        raise ValueError("initial state relative tolerance must be positive")
    if args.independent_state_envelope_factor <= 0.0:
        raise ValueError("independent state envelope factor must be positive")
    if args.independent_vortex_core_relative_tolerance <= 0.0:
        raise ValueError("independent vortex tolerance must be positive")
    runs = sorted(
        (load_reference_run(path) for path in args.run_dir), key=lambda x: x.resolution
    )
    _validate_matched_contract(runs)
    resolutions = tuple(run.resolution for run in runs)
    canonical_resolutions = resolutions == CANONICAL_RESOLUTIONS
    canonical_primary_checks = _canonical_primary_contract_checks(runs)
    canonical_primary_contract = all(canonical_primary_checks.values())
    if not canonical_resolutions and not args.allow_noncanonical_resolutions:
        raise ValueError(
            f"expected canonical resolutions {CANONICAL_RESOLUTIONS}, got {resolutions}"
        )

    coarse_nx = int(runs[-1].config["coarse_nx"])
    coarse_ny = int(runs[-1].config["coarse_ny"])
    gamma = float(runs[-1].config["gamma"])
    pair_names = (
        f"{runs[0].resolution[0]}x{runs[0].resolution[1]}_vs_{runs[1].resolution[0]}x{runs[1].resolution[1]}",
        f"{runs[1].resolution[0]}x{runs[1].resolution[1]}_vs_{runs[2].resolution[0]}x{runs[2].resolution[1]}",
    )
    all_rows: list[dict[str, Any]] = []
    pair_rows: list[list[dict[str, float]]] = []
    for pair_index, (candidate, reference) in enumerate(
        zip(runs[:-1], runs[1:], strict=True)
    ):
        rows = pair_metrics(
            candidate.states,
            reference.states,
            physical_times=reference.times,
            cell_volume=reference.cell_volume,
            cell_centers=reference.cell_centers,
            coarse_nx=coarse_nx,
            coarse_ny=coarse_ny,
            gamma=gamma,
        )
        pair_rows.append(rows)
        for row, physical_time in zip(rows, reference.times, strict=True):
            all_rows.append(
                {
                    "pair": pair_names[pair_index],
                    "physical_time": float(physical_time),
                    **row,
                }
            )

    impulse_operators = build_fv_impulse_operators(
        cell_centers=runs[-1].cell_centers,
        cell_volume=runs[-1].cell_volume,
        face_centers=runs[-1].face_centers,
        face_measure=runs[-1].face_measure,
        face_owner=runs[-1].face_owner,
        face_neighbor=runs[-1].face_neighbor,
        face_boundary_tag=runs[-1].face_boundary_tag,
    )
    saved_dt = np.diff(runs[-1].times)
    if not np.allclose(saved_dt, saved_dt[0], rtol=0.0, atol=1.0e-14):
        raise ValueError("decoder-gain audit currently requires uniform saved dt")
    decoder_gains = decoder_gain_summary(
        impulse_operators,
        dt=float(saved_dt[0]),
    )
    all_impulse_rows: list[dict[str, Any]] = []
    impulse_pair_summaries: list[dict[str, Any]] = []
    for pair_name, candidate, reference in zip(
        pair_names,
        runs[:-1],
        runs[1:],
        strict=True,
    ):
        rows, pair_summary = impulse_pair_metrics(
            candidate,
            reference,
            impulse_operators,
        )
        impulse_pair_summaries.append({"pair": pair_name, **pair_summary})
        all_impulse_rows.extend({"pair": pair_name, **row} for row in rows)

    coarse_impulse = impulse_pair_summaries[0]
    fine_impulse = impulse_pair_summaries[1]

    def contraction_ratio(metric: str) -> float:
        return float(fine_impulse[metric] / max(float(coarse_impulse[metric]), 1.0e-30))

    impulse_contraction = {
        "full_winv_error_ratio": contraction_ratio("full_winv_error_norm"),
        "divergence_active_winv_error_ratio": contraction_ratio(
            "divergence_active_winv_error_norm"
        ),
        "cycle_winv_error_ratio": contraction_ratio("cycle_winv_error_norm"),
        "boundary_winv_error_ratio": contraction_ratio("boundary_winv_error_norm"),
    }

    def component_contraction_ratios(metric: str) -> list[float]:
        coarse_values = np.asarray(coarse_impulse[metric], dtype=np.float64)
        fine_values = np.asarray(fine_impulse[metric], dtype=np.float64)
        return (
            (fine_values / np.maximum(coarse_values, 1.0e-30))
            .astype(np.float64)
            .tolist()
        )

    impulse_component_contraction = {
        "full_winv_error_ratio": component_contraction_ratios(
            "full_winv_error_norm_by_component"
        ),
        "divergence_active_winv_error_ratio": component_contraction_ratios(
            "divergence_active_winv_error_norm_by_component"
        ),
        "cycle_winv_error_ratio": component_contraction_ratios(
            "cycle_winv_error_norm_by_component"
        ),
        "boundary_winv_error_ratio": component_contraction_ratios(
            "boundary_winv_error_norm_by_component"
        ),
    }
    coarse_boundary_exchange = np.asarray(
        coarse_impulse["boundary_tag_component_exchange_error_l2"],
        dtype=np.float64,
    )
    fine_boundary_exchange = np.asarray(
        fine_impulse["boundary_tag_component_exchange_error_l2"],
        dtype=np.float64,
    )
    primary_boundary_exchange = np.stack(
        [
            _boundary_exchange_l2(
                run.face_impulses,
                run.face_boundary_tag,
            )
            for run in runs
        ],
        axis=0,
    )
    boundary_exchange_gate = _boundary_exchange_gate(
        coarse_boundary_exchange,
        fine_boundary_exchange,
        primary_boundary_exchange,
        contraction_limit=args.successive_error_ratio,
    )
    boundary_exchange_contraction = boundary_exchange_gate["contraction"]
    all_lsmr_stop_codes = sorted(
        {
            int(code)
            for pair_summary in impulse_pair_summaries
            for code in pair_summary["lsmr_stop_codes"]
        }
    )
    maximum_impulse_reconstruction = max(
        float(row["decomposition_reconstruction_relative_l2"])
        for row in all_impulse_rows
    )
    maximum_cycle_divergence_relative = max(
        float(row["cycle_divergence_relative_l2"]) for row in all_impulse_rows
    )
    maximum_cycle_divergence_absolute = max(
        float(row["cycle_divergence_l2"]) for row in all_impulse_rows
    )
    maximum_impulse_orthogonality = max(
        float(row["winv_orthogonality_relative"]) for row in all_impulse_rows
    )
    maximum_decoded_transition_abs = max(
        float(row["decoded_transition_max_abs_integrated"]) for row in all_impulse_rows
    )

    coarse_pair = pair_rows[0]
    fine_pair = pair_rows[1]
    final_state_ratio = fine_pair[-1]["normalized_state_relative_l1"] / max(
        coarse_pair[-1]["normalized_state_relative_l1"], 1.0e-30
    )
    mean_state_ratio = np.mean(
        [row["normalized_state_relative_l1"] for row in fine_pair[1:]]
    ) / max(
        np.mean([row["normalized_state_relative_l1"] for row in coarse_pair[1:]]),
        1.0e-30,
    )
    centerline_ratio = fine_pair[-1]["centerline_density_relative_l1"] / max(
        coarse_pair[-1]["centerline_density_relative_l1"], 1.0e-30
    )
    initial_state_relative_l1 = max(
        coarse_pair[0]["normalized_state_relative_l1"],
        fine_pair[0]["normalized_state_relative_l1"],
    )
    initial_cell_average_contract = all(
        run.artifact_schema == PRIMARY_ARTIFACT_SCHEMA
        and run.metadata.get("initial_quadrature_scope") == INITIAL_QUADRATURE_SCOPE
        and run.metadata.get("initial_state_contract") == INITIAL_STATE_CONTRACT
        for run in runs
    )
    coarse_dx = (
        float(runs[-1].config["x_max"]) - float(runs[-1].config["x_min"])
    ) / coarse_nx
    independent = (
        load_independent_run(args.independent_run_dir)
        if args.independent_run_dir is not None
        else None
    )
    independent_checks: dict[str, bool] = {}
    independent_metrics: dict[str, float] = {}
    independent_agrees = False
    if independent is not None:
        _validate_independent_contract(
            independent,
            runs[-1],
            coarse_nx=coarse_nx,
            coarse_ny=coarse_ny,
        )
        independent_rows = pair_metrics(
            independent.states,
            runs[-1].states,
            physical_times=runs[-1].times,
            cell_volume=runs[-1].cell_volume,
            cell_centers=runs[-1].cell_centers,
            coarse_nx=coarse_nx,
            coarse_ny=coarse_ny,
            gamma=gamma,
        )
        for row, physical_time in zip(independent_rows, runs[-1].times, strict=True):
            all_rows.append(
                {
                    "pair": (
                        f"{independent.solver_name}_{independent.resolution[0]}x"
                        f"{independent.resolution[1]}_vs_"
                        f"{runs[-1].resolution[0]}x{runs[-1].resolution[1]}"
                    ),
                    "physical_time": float(physical_time),
                    **row,
                }
            )
        independent_checks, independent_metrics = independent_agreement_metrics(
            independent_rows,
            coarse_pair,
            fine_pair,
            coarse_dx=coarse_dx,
            state_envelope_factor=args.independent_state_envelope_factor,
            vortex_core_relative_tolerance=(
                args.independent_vortex_core_relative_tolerance
            ),
            initial_state_relative_tolerance=args.initial_state_relative_tolerance,
        )
        independent_checks["independent_artifact_contract"] = True
        independent_checks["independent_matched_solver_contract"] = (
            _canonical_independent_solver_contract(independent)
        )
        independent_agrees = all(independent_checks.values())
    checks = {
        **canonical_primary_checks,
        "conservative_cell_average_initialization": bool(initial_cell_average_contract),
        "restricted_initial_states_agree": bool(
            initial_cell_average_contract
            and initial_state_relative_l1 <= args.initial_state_relative_tolerance
        ),
        "final_state_difference_contracts": bool(
            final_state_ratio <= args.successive_error_ratio
        ),
        "time_mean_state_difference_contracts": bool(
            mean_state_ratio <= args.successive_error_ratio
        ),
        "final_centerline_density_difference_contracts": bool(
            centerline_ratio <= args.successive_error_ratio
        ),
        "medium_fine_shock_position_within_one_coarse_cell": bool(
            fine_pair[-1]["shock_position_difference"] <= coarse_dx + 1.0e-14
        ),
        "medium_fine_vortex_core_density_agrees": bool(
            fine_pair[-1]["vortex_core_density_relative_difference"]
            <= args.vortex_core_relative_tolerance
        ),
        "impulse_topology_matches_connected_mesh": bool(
            impulse_operators.topology.num_connected_components == 1
            and impulse_operators.topology.interior_rank
            == impulse_operators.topology.num_cells - 1
            and impulse_operators.topology.interior_nullity
            == (
                impulse_operators.topology.num_interior_faces
                - impulse_operators.topology.num_cells
                + 1
            )
        ),
        "impulse_decoder_gain_solve_converged": bool(
            decoder_gains.impulse_gain.relative_residual <= 1.0e-8
            and decoder_gains.flux_gain.relative_residual <= 1.0e-8
        ),
        "impulse_decomposition_reconstructs": bool(
            maximum_impulse_reconstruction <= 1.0e-10
        ),
        "impulse_cycle_is_divergence_free": bool(
            maximum_cycle_divergence_relative <= CYCLE_DIVERGENCE_RELATIVE_TOLERANCE
        ),
        "impulse_lsmr_stop_codes_converged": bool(
            set(all_lsmr_stop_codes) <= ACCEPTED_LSMR_STOP_CODES
        ),
        "impulse_div_cycle_are_winv_orthogonal": bool(
            maximum_impulse_orthogonality <= 1.0e-10
        ),
        "impulse_difference_decodes_state_transition": bool(
            maximum_decoded_transition_abs <= 1.0e-10
        ),
        "divergence_active_impulse_error_contracts": bool(
            impulse_contraction["divergence_active_winv_error_ratio"]
            <= args.successive_error_ratio
        ),
        "divergence_active_impulse_error_contracts_by_component": bool(
            np.all(
                np.asarray(
                    impulse_component_contraction["divergence_active_winv_error_ratio"]
                )
                <= args.successive_error_ratio
            )
        ),
        "boundary_impulse_error_contracts": bool(
            impulse_contraction["boundary_winv_error_ratio"]
            <= args.successive_error_ratio
        ),
        "boundary_exchange_error_contracts_by_tag_and_component": bool(
            boundary_exchange_gate["combined_passed"]
        ),
        "active_boundary_exchange_error_contracts_by_tag_and_component": bool(
            boundary_exchange_gate["active_passed"]
        ),
        "wall_zero_flux_boundary_exchange_errors_below_absolute_tolerance": bool(
            boundary_exchange_gate["wall_zero_flux_error_passed"]
        ),
        "wall_zero_flux_primary_exchanges_below_absolute_tolerance": bool(
            boundary_exchange_gate["wall_zero_flux_primary_exchange_passed"]
        ),
        "full_reference_impulse_error_contracts": bool(
            impulse_contraction["full_winv_error_ratio"] <= args.successive_error_ratio
        ),
        "full_reference_impulse_error_contracts_by_component": bool(
            np.all(
                np.asarray(impulse_component_contraction["full_winv_error_ratio"])
                <= args.successive_error_ratio
            )
        ),
        **independent_checks,
        "independent_public_solver_agreement": independent_agrees,
    }
    convergence_checks = {
        key: value
        for key, value in checks.items()
        if not key.startswith("independent_")
        and not key.startswith("full_reference_impulse_")
        and (canonical_resolutions or not key.startswith("canonical_"))
    }
    convergence_passed = all(convergence_checks.values())
    if not canonical_resolutions:
        status = "smoke_only"
    elif not convergence_passed:
        status = "failed_convergence"
    elif independent is None:
        status = "convergence_passed_independent_solver_pending"
    elif not independent_agrees:
        status = "failed_independent_solver_agreement"
    else:
        status = "benchmark_contract_closed"

    args.output_dir.mkdir(parents=True)
    pair_metrics_path = args.output_dir / "pair_metrics.csv"
    with pair_metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    impulse_metrics_path = args.output_dir / "impulse_metrics.csv"
    with impulse_metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_impulse_rows[0]))
        writer.writeheader()
        writer.writerows(all_impulse_rows)
    audit_path = Path(__file__).resolve()
    impulse_utility_path = (
        ROOT / "utility/time_dependent_no/fv_impulse_diagnostics.py"
    ).resolve()
    summary = {
        "schema": AUDIT_SCHEMA,
        "status": status,
        "audit_entrypoint_sha256": _sha256(audit_path),
        "fv_impulse_utility_sha256": _sha256(impulse_utility_path),
        "pair_metrics_sha256": _sha256(pair_metrics_path),
        "impulse_metrics_sha256": _sha256(impulse_metrics_path),
        "resolutions": [list(value) for value in resolutions],
        "coarse_resolution": [coarse_nx, coarse_ny],
        "checks": checks,
        "convergence_passed": convergence_passed,
        "benchmark_contract_closed": bool(
            canonical_primary_contract
            and convergence_passed
            and checks["independent_public_solver_agreement"]
        ),
        "direct_reference_impulse_contract_closed": bool(
            canonical_primary_contract
            and convergence_passed
            and independent_agrees
            and checks["full_reference_impulse_error_contracts"]
            and checks["full_reference_impulse_error_contracts_by_component"]
        ),
        "successive_error_ratio_limit": args.successive_error_ratio,
        "impulse_decomposition_lsmr_tolerance": (IMPULSE_DECOMPOSITION_TOLERANCE),
        "cycle_divergence_relative_tolerance": (CYCLE_DIVERGENCE_RELATIVE_TOLERANCE),
        "wall_zero_flux_absolute_tolerance": (WALL_ZERO_FLUX_ABSOLUTE_TOLERANCE),
        "boundary_exchange_tag_order": list(BOUNDARY_TAG_NAMES[1:]),
        "boundary_exchange_component_order": [
            "mass",
            "x_momentum",
            "y_momentum",
            "energy",
        ],
        "boundary_exchange_active_contraction_mask": boundary_exchange_gate[
            "active_mask"
        ].tolist(),
        "initial_state_relative_tolerance": args.initial_state_relative_tolerance,
        "vortex_core_relative_tolerance": args.vortex_core_relative_tolerance,
        "independent_state_envelope_factor": (args.independent_state_envelope_factor),
        "independent_vortex_core_relative_tolerance": (
            args.independent_vortex_core_relative_tolerance
        ),
        "metrics": {
            "final_state_successive_ratio": float(final_state_ratio),
            "time_mean_state_successive_ratio": float(mean_state_ratio),
            "final_centerline_density_successive_ratio": float(centerline_ratio),
            "maximum_initial_state_relative_l1": float(initial_state_relative_l1),
            "medium_fine_final_shock_position_difference": float(
                fine_pair[-1]["shock_position_difference"]
            ),
            "medium_fine_final_vortex_core_density_relative_difference": float(
                fine_pair[-1]["vortex_core_density_relative_difference"]
            ),
            "impulse_contraction": impulse_contraction,
            "impulse_component_contraction": impulse_component_contraction,
            "boundary_exchange_contraction_by_tag_and_component": (
                boundary_exchange_contraction.tolist()
            ),
            "coarse_boundary_exchange_error_l2_by_tag_and_component": (
                coarse_boundary_exchange.tolist()
            ),
            "fine_boundary_exchange_error_l2_by_tag_and_component": (
                fine_boundary_exchange.tolist()
            ),
            "primary_boundary_exchange_l2_by_run_tag_and_component": (
                primary_boundary_exchange.tolist()
            ),
            "maximum_impulse_reconstruction_relative_l2": float(
                maximum_impulse_reconstruction
            ),
            "maximum_cycle_divergence_relative_l2": float(
                maximum_cycle_divergence_relative
            ),
            "maximum_cycle_divergence_absolute_l2": float(
                maximum_cycle_divergence_absolute
            ),
            "lsmr_stop_codes": all_lsmr_stop_codes,
            "maximum_impulse_winv_orthogonality_relative": float(
                maximum_impulse_orthogonality
            ),
            "maximum_decoded_transition_max_abs_integrated": float(
                maximum_decoded_transition_abs
            ),
            **independent_metrics,
        },
        "impulse_diagnostics": {
            "topology": impulse_operators.topology.to_dict(),
            "decoder_gains": decoder_gains.to_dict(),
            "decomposition_lsmr_tolerance": IMPULSE_DECOMPOSITION_TOLERANCE,
            "pair_summaries": impulse_pair_summaries,
            "weight_convention": "W_f=face_measure_f*dual_width_f",
            "incidence_convention": "+1 owner, -1 interior neighbor",
            "claim_boundary": IMPULSE_CLAIM_BOUNDARY,
        },
        "inputs": [
            {
                "root_name": run.root.name,
                "resolution": list(run.resolution),
                "reference_artifact_sha256": run.summary["reference_artifact_sha256"],
                "solver_utility_sha256": run.summary["solver_utility_sha256"],
            }
            for run in runs
        ]
        + (
            [
                {
                    "kind": "independent_public_solver",
                    "solver_name": independent.solver_name,
                    "solver_version": independent.solver_version,
                    "artifact_schema": independent.artifact_schema,
                    "root_name": independent.root.name,
                    "resolution": list(independent.resolution),
                    "reference_artifact_sha256": independent.summary[
                        "reference_artifact_sha256"
                    ],
                    "adapter_sha256": independent.summary["adapter_sha256"],
                }
            ]
            if independent is not None
            else []
        ),
        "claim_boundary": {
            "verified_if_convergence_passed": "nested-grid state, componentwise divergence-active impulse, and per-tag/component boundary exchange of this WENO5-HLLC-SSPRK3 implementation; active exchanges must self-converge and analytically zero reflecting-wall exchanges must remain below the recorded absolute tolerance",
            "verified_if_contract_closed": "state agreement with one pinned, boundary-matched SharpClaw solver under the predeclared discretization envelope",
            "verified_if_direct_reference_impulse_contract_closed": "same-solver full face-impulse contraction in addition to the closed state benchmark; the cycle field remains discretization-specific",
            "still_required": (
                "a frozen perturbation-family and split contract before neural training-data generation"
                if independent_agrees
                else "state agreement with an independent public solver before reference or training-data promotion"
            ),
            "unsupported": "neural baseline quality, oracle correction headroom, a learned corrector, or uniqueness of the 2D face-field cycle component",
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if status in {"failed_convergence", "failed_independent_solver_agreement"}:
        raise RuntimeError(f"canonical shock-vortex audit failed with status {status}")


if __name__ == "__main__":
    main()
