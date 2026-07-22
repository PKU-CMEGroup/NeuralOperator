"""Frozen perturbation-family contract for the 2D shock--vortex benchmark.

The canonical D037 reference closes the numerical data contract for one
Mach-1.1 shock--vortex interaction.  This module extends that contract only
along two physically safe initial-condition axes: vortex strength and vertical
position.  Shock Mach number and both shock states remain fixed because the
current generator does not derive a new Rankine--Hugoniot pair when Mach
changes.

The family is deterministic and deliberately narrow.  Every physical
trajectory, including all of its saved time transitions, belongs to exactly
one split.  Validation holds out the two boundary-nearest vortex positions;
test holds out the three strongest vortices.  The split is therefore grouped
and parameter-OOD rather than a random transition-level split.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from utility.time_dependent_no.shock_vortex_fv import (
    REFERENCE_CONTRACT_CHECK_KEYS,
    ShockVortexFVConfig,
)

FAMILY_SCHEMA = "shock_vortex_perturbation_family_v1"
FAMILY_ID = "shock_vortex_eps_y_15x9_fine1000_to_250_dt001_v1"
REFERENCE_ARTIFACT_SCHEMA = "shock_vortex_fv_reference_v2"

VORTEX_EPSILON_VALUES = tuple(0.2125 + 0.0125 * index for index in range(15))
VORTEX_Y_VALUES = tuple(0.35 + 0.0375 * index for index in range(9))
VALIDATION_Y_INDICES = frozenset({0, 8})
TEST_EPSILON_INDICES = frozenset({12, 13, 14})
CANONICAL_CASE_ID = "sv_e07_y04"
SMOKE_CASE_IDS = (
    CANONICAL_CASE_ID,
    "sv_e07_y00",
    "sv_e13_y04",
)


def canonical_json_digest(payload: Mapping[str, Any]) -> str:
    """Return the stable SHA256 digest used by family and artifact metadata."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _family_output_times() -> tuple[float, ...]:
    # Integer construction avoids cumulative floating-point drift while keeping
    # the exact 60-call, t=0.6 contract visible in the manifest.
    return tuple(index / 100.0 for index in range(61))


def _base_reference_config() -> ShockVortexFVConfig:
    return ShockVortexFVConfig(
        nx=1000,
        ny=400,
        coarse_nx=250,
        coarse_ny=100,
        t_final=0.6,
        output_times=_family_output_times(),
        cfl=0.35,
        initial_quadrature_order=8,
    ).validated()


def _split_for_case(epsilon_index: int, y_index: int) -> tuple[str, str, str]:
    if epsilon_index in TEST_EPSILON_INDICES:
        return (
            "test",
            f"strength_ood_e{epsilon_index:02d}",
            "held-out high-vortex-strength band",
        )
    if y_index in VALIDATION_Y_INDICES:
        return (
            "validation",
            f"position_ood_y{y_index:02d}",
            "held-out boundary-nearest vortex-position band",
        )
    return (
        "train",
        f"train_e{epsilon_index:02d}",
        "seen-strength interior-position training support",
    )


def _manifest_payload_without_digest() -> dict[str, Any]:
    base_config = _base_reference_config()
    cases: list[dict[str, Any]] = []
    trajectory_index = 0
    for epsilon_index, epsilon in enumerate(VORTEX_EPSILON_VALUES):
        for y_index, vortex_y in enumerate(VORTEX_Y_VALUES):
            split, split_group_id, role = _split_for_case(epsilon_index, y_index)
            case_id = f"sv_e{epsilon_index:02d}_y{y_index:02d}"
            cases.append(
                {
                    "case_id": case_id,
                    "trajectory_index": trajectory_index,
                    "split": split,
                    "split_group_id": split_group_id,
                    "split_role": role,
                    "epsilon_index": epsilon_index,
                    "y_index": y_index,
                    "parameters": {
                        "vortex_epsilon": epsilon,
                        "vortex_y": vortex_y,
                    },
                    "canonical_physical_case": case_id == CANONICAL_CASE_ID,
                }
            )
            trajectory_index += 1

    split_counts = {
        split: sum(case["split"] == split for case in cases)
        for split in ("train", "validation", "test")
    }
    split_group_counts = {
        split: len({case["split_group_id"] for case in cases if case["split"] == split})
        for split in ("train", "validation", "test")
    }
    return {
        "schema": FAMILY_SCHEMA,
        "family_id": FAMILY_ID,
        "status": "frozen_before_generation_or_training",
        "benchmark": "Mach-1.1 shock--isentropic-vortex interaction",
        "reference_config": base_config.to_dict(),
        "reference_fidelity": {
            "evolution_grid": [base_config.nx, base_config.ny],
            "stored_model_grid": [base_config.coarse_nx, base_config.coarse_ny],
            "restriction": "conservative cell-average and owner-oriented face-impulse restriction",
            "dtype": "float64",
            "canonical_convergence_contract": (
                "shock_vortex_fv_convergence_cellavg_sharpclaw_matchedbc_20260720c"
            ),
            "interpretation": (
                "high-fidelity primary-solver trajectories restricted to the "
                "declared 250x100 model grid; not arbitrary dataset downsampling"
            ),
        },
        "time_contract": {
            "t_final": 0.6,
            "saved_delta_t": 0.01,
            "saved_calls": 60,
            "output_times": list(base_config.output_times),
            "accepted_substeps": (
                "all accepted SSPRK3 steps are recorded and accumulated into each saved interval"
            ),
        },
        "fixed_physics": {
            "gamma": base_config.gamma,
            "shock_mach": base_config.shock_mach,
            "shock_x": base_config.shock_x,
            "right_state": [
                base_config.right_rho,
                base_config.right_u,
                0.0,
                base_config.right_pressure,
            ],
            "vortex_x": base_config.vortex_x,
            "vortex_alpha": base_config.vortex_alpha,
            "vortex_radius": base_config.vortex_radius,
            "boundary_mode": "linear x extrapolation; y symmetry",
        },
        "varied_parameters": {
            "vortex_epsilon": {
                "values": list(VORTEX_EPSILON_VALUES),
                "units": "nondimensional",
            },
            "vortex_y": {
                "values": list(VORTEX_Y_VALUES),
                "units": "domain y coordinate",
            },
        },
        "split_contract": {
            "unit": "complete physical trajectory; time transitions never cross splits",
            "train": (
                "epsilon indices 0..11 and y indices 1..7; seen strength range "
                "with boundary-nearest positions excluded"
            ),
            "validation": (
                "epsilon indices 0..11 and y indices {0,8}; position OOD for checkpoint selection"
            ),
            "test": (
                "epsilon indices 12..14 and all y indices; untouched high-strength parameter OOD"
            ),
            "random_split": False,
            "split_counts": split_counts,
            "split_group_counts": split_group_counts,
        },
        "canonical_case_id": CANONICAL_CASE_ID,
        "smoke_case_ids": list(SMOKE_CASE_IDS),
        "cases": cases,
        "claim_boundary": {
            "supports_after_complete_audit": (
                "one frozen two-parameter dynamic finite-volume training family with grouped "
                "position and strength OOD evaluation"
            ),
            "does_not_support": (
                "geometry OOD, Mach-number transfer, neural baseline quality, oracle correction "
                "headroom, or learned stabilization"
            ),
        },
    }


def build_shock_vortex_family_manifest() -> dict[str, Any]:
    """Build the exact frozen family manifest and attach its stable digest."""

    payload = _manifest_payload_without_digest()
    payload["manifest_digest_sha256"] = canonical_json_digest(payload)
    return payload


def validate_shock_vortex_family_manifest(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless *payload* is exactly the frozen family contract."""

    normalized = json.loads(json.dumps(payload, allow_nan=False))
    expected = build_shock_vortex_family_manifest()
    if normalized != expected:
        received_digest = normalized.get("manifest_digest_sha256")
        unsigned = dict(normalized)
        unsigned.pop("manifest_digest_sha256", None)
        recomputed = canonical_json_digest(unsigned)
        if received_digest != recomputed:
            raise ValueError("shock-vortex family manifest digest mismatch")
        raise ValueError(
            "shock-vortex family manifest differs from the frozen contract"
        )
    return normalized


def load_shock_vortex_family_manifest(path: Path) -> dict[str, Any]:
    """Load and validate the exact frozen family manifest."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("shock-vortex family manifest must contain a JSON object")
    return validate_shock_vortex_family_manifest(payload)


def family_case_by_id(manifest: Mapping[str, Any], case_id: str) -> dict[str, Any]:
    """Return one validated case, rejecting unknown or duplicate identifiers."""

    validated = validate_shock_vortex_family_manifest(manifest)
    matches = [case for case in validated["cases"] if case["case_id"] == case_id]
    if len(matches) != 1:
        raise ValueError(f"family case_id is not unique and present: {case_id}")
    return matches[0]


def config_for_family_case(
    manifest: Mapping[str, Any], case_id: str
) -> ShockVortexFVConfig:
    """Construct the exact solver configuration declared for one family case."""

    validated = validate_shock_vortex_family_manifest(manifest)
    case = family_case_by_id(validated, case_id)
    fields = ShockVortexFVConfig.__dataclass_fields__
    arguments = {
        name: validated["reference_config"][name]
        for name in fields
        if name in validated["reference_config"]
    }
    arguments["output_times"] = tuple(arguments["output_times"])
    base = ShockVortexFVConfig(**arguments).validated()
    return replace(base, **case["parameters"]).validated()


def family_case_provenance(manifest: Mapping[str, Any], case_id: str) -> dict[str, Any]:
    """Return the immutable trajectory mapping embedded in each artifact."""

    validated = validate_shock_vortex_family_manifest(manifest)
    case = family_case_by_id(validated, case_id)
    return {
        "manifest_schema": validated["schema"],
        "family_id": validated["family_id"],
        "manifest_digest_sha256": validated["manifest_digest_sha256"],
        "case_id": case["case_id"],
        "trajectory_index": case["trajectory_index"],
        "split": case["split"],
        "split_group_id": case["split_group_id"],
        "parameters": case["parameters"],
        "canonical_physical_case": case["canonical_physical_case"],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_num_faces(nx: int, ny: int) -> int:
    return (nx + 1) * ny + nx * (ny + 1)


def _case_audit_row(
    manifest: Mapping[str, Any],
    artifact_root: Path,
    case_id: str,
) -> dict[str, Any]:
    case = family_case_by_id(manifest, case_id)
    expected_provenance = family_case_provenance(manifest, case_id)
    expected_config = config_for_family_case(manifest, case_id).to_dict()
    case_dir = artifact_root / case_id
    failures: list[str] = []
    summary_path = case_dir / "summary.json"
    artifact_path = case_dir / "reference.npz"
    if not summary_path.is_file():
        failures.append("summary_missing")
    if not artifact_path.is_file():
        failures.append("reference_artifact_missing")
    if failures:
        return {
            "case_id": case_id,
            "trajectory_index": case["trajectory_index"],
            "split": case["split"],
            "status": "failed",
            "failed_checks": failures,
        }

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "case_id": case_id,
            "trajectory_index": case["trajectory_index"],
            "split": case["split"],
            "status": "failed",
            "failed_checks": [f"summary_unreadable:{type(exc).__name__}"],
        }

    checks = summary.get("contract_checks")
    if summary.get("status") != "passed":
        failures.append("reference_status_not_passed")
    if not isinstance(checks, dict) or set(checks) != REFERENCE_CONTRACT_CHECK_KEYS:
        failures.append("contract_check_key_set_mismatch")
    elif not all(checks.values()):
        failures.append("reference_contract_check_failed")
    if summary.get("schema") != REFERENCE_ARTIFACT_SCHEMA:
        failures.append("summary_schema_mismatch")
    if summary.get("config") != expected_config:
        failures.append("config_mismatch")
    if summary.get("family_contract") != expected_provenance:
        failures.append("family_provenance_mismatch")
    if summary.get("reference_artifact") != artifact_path.name:
        failures.append("reference_artifact_name_mismatch")
    artifact_sha256 = _sha256(artifact_path)
    if summary.get("reference_artifact_sha256") != artifact_sha256:
        failures.append("reference_artifact_digest_mismatch")

    minimum_density = float("nan")
    minimum_pressure = float("nan")
    try:
        with np.load(artifact_path, allow_pickle=False) as artifact:
            schema = artifact["schema"].item()
            states = artifact["conservative_states"]
            times = artifact["physical_times"]
            impulses = artifact["cumulative_accepted_substep_face_impulses"]
            config_payload = json.loads(artifact["config_json"].item())
            metadata = json.loads(artifact["metadata_json"].item())
            provenance = json.loads(artifact["family_contract_json"].item())
            expected_times = np.asarray(
                expected_config["output_times"], dtype=np.float64
            )
            num_cells = expected_config["coarse_nx"] * expected_config["coarse_ny"]
            num_faces = _expected_num_faces(
                expected_config["coarse_nx"], expected_config["coarse_ny"]
            )
            if schema != REFERENCE_ARTIFACT_SCHEMA:
                failures.append("artifact_schema_mismatch")
            if config_payload != expected_config:
                failures.append("artifact_config_mismatch")
            if metadata.get("family_contract") != expected_provenance:
                failures.append("artifact_metadata_family_mismatch")
            if provenance != expected_provenance:
                failures.append("artifact_family_provenance_mismatch")
            if states.shape != (expected_times.size, num_cells, 4):
                failures.append("state_shape_mismatch")
            if impulses.shape != (expected_times.size - 1, num_faces, 4):
                failures.append("impulse_shape_mismatch")
            if not np.array_equal(times, expected_times):
                failures.append("physical_times_mismatch")
            if not np.all(np.isfinite(states)):
                failures.append("states_nonfinite")
            else:
                density = states[..., 0]
                momentum_squared = states[..., 1] ** 2 + states[..., 2] ** 2
                pressure = (expected_config["gamma"] - 1.0) * (
                    states[..., 3] - 0.5 * momentum_squared / density
                )
                minimum_density = float(np.min(density))
                minimum_pressure = float(np.min(pressure))
                if minimum_density <= 0.0:
                    failures.append("density_not_positive")
                if minimum_pressure <= 0.0:
                    failures.append("pressure_not_positive")
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        failures.append(f"artifact_unreadable:{type(exc).__name__}")

    return {
        "case_id": case_id,
        "trajectory_index": case["trajectory_index"],
        "split": case["split"],
        "split_group_id": case["split_group_id"],
        "parameters": case["parameters"],
        "status": "passed" if not failures else "failed",
        "failed_checks": failures,
        "reference_artifact_sha256": artifact_sha256,
        "artifact_bytes": artifact_path.stat().st_size,
        "minimum_density": minimum_density,
        "minimum_pressure": minimum_pressure,
        "accepted_steps": summary.get("accepted_steps"),
        "rejected_attempts": summary.get("rejected_attempts"),
        "face_reconstruction_fallbacks": summary.get("face_reconstruction_fallbacks"),
        "maximum_interval_closure_relative_l2": summary.get(
            "maximum_interval_closure_relative_l2"
        ),
    }


def audit_shock_vortex_family_artifacts(
    manifest: Mapping[str, Any],
    artifact_root: Path,
    case_ids: Iterable[str],
) -> dict[str, Any]:
    """Audit a declared complete or smoke cohort against the frozen manifest."""

    validated = validate_shock_vortex_family_manifest(manifest)
    requested = list(case_ids)
    if not requested:
        raise ValueError("at least one family case must be audited")
    if len(requested) != len(set(requested)):
        raise ValueError("family audit case_ids must be unique")
    known = {case["case_id"] for case in validated["cases"]}
    unknown = sorted(set(requested) - known)
    if unknown:
        raise ValueError(f"unknown family case_ids: {unknown}")
    rows = [_case_audit_row(validated, artifact_root, case_id) for case_id in requested]
    passed = all(row["status"] == "passed" for row in rows)
    artifact_set_digest = canonical_json_digest(
        {
            "manifest_digest_sha256": validated["manifest_digest_sha256"],
            "artifacts": [
                {
                    "case_id": row["case_id"],
                    "reference_artifact_sha256": row.get("reference_artifact_sha256"),
                }
                for row in rows
            ],
        }
    )
    return {
        "schema": "shock_vortex_family_artifact_audit_v1",
        "status": "passed" if passed else "failed",
        "family_id": validated["family_id"],
        "manifest_digest_sha256": validated["manifest_digest_sha256"],
        "artifact_set_digest_sha256": artifact_set_digest,
        "artifact_root": str(artifact_root),
        "requested_case_ids": requested,
        "case_count": len(rows),
        "split_counts": {
            split: sum(row["split"] == split for row in rows)
            for split in ("train", "validation", "test")
        },
        "passed_case_count": sum(row["status"] == "passed" for row in rows),
        "failed_case_count": sum(row["status"] != "passed" for row in rows),
        "rows": rows,
        "claim_boundary": {
            "verified_if_passed": (
                "the requested artifact cohort matches the frozen trajectory, split, "
                "time, geometry, admissibility, and provenance contract"
            ),
            "unsupported": (
                "neural fit, rollout quality, correction headroom, geometry OOD, or "
                "completion of unrequested family cases"
            ),
        },
    }
