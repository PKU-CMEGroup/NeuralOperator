"""Frozen resource contract for the A46 same-state refresh experiment.

This module owns two prerequisites that were missing from the metadata-only
A46-A2 audit: an independently initialized D060-compatible training command
and a newly generated dynamic-FV population.  It does not implement the A46
branch counterfactual or any recurrent controller.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from utility.time_dependent_no.pcno_resolution_transfer import restrict_nested_state
from utility.time_dependent_no.pcno_same_state_refresh_metadata import (
    EXPECTED_FINE_RESOLUTION,
    EXPECTED_INPUT_CALLS,
    EXPECTED_MODEL_DELTA_T,
    EXPECTED_NATIVE_RESOLUTION,
    EXPECTED_OUTPUT_CALLS,
    EXPECTED_SELECTOR_THRESHOLD,
    POPULATION_SCHEMA,
    SELECTOR_WORKING_ID,
    canonical_payload_sha256,
    validate_population_manifest,
    with_payload_sha256,
)
from utility.time_dependent_no.shock_vortex_fv import (
    ShockVortexFVConfig,
    ShockVortexReferenceResult,
    StructuredFVGeometry,
    make_structured_fv_geometry,
)

RESOURCE_BUILD_WORKING_ID = "W26-L5-P6-RFB19-A46-A2-R1-MATCHED-RESOURCE-BUILD"
RESOURCE_PLAN_SCHEMA = "pcno_same_state_refresh_resource_plan_v1"
DUAL_REFERENCE_SCHEMA = "pcno_same_state_refresh_dual_resolution_reference_v1"
RESOURCE_AUDIT_SCHEMA = "pcno_same_state_refresh_resource_audit_v1"

REFERENCE_CHECKPOINT_SHA256 = (
    "95e6c180a3298c4b38662d8c6cb77301c0e7fd0286e9a53571bddc487b8379c9"
)
EXPECTED_DATA_MANIFEST_DIGEST = (
    "f8d228ae3c6e08fe6697f37df21476fe621abc354d0b0a6eed2a623ed2b9f96c"
)
EXPECTED_NORMALIZATION_DIGEST = (
    "9df7efb2f1aadf315f399dcabf3b366bf1b2f46794b026cbc5b7d80ba22e1a1a"
)
EXPECTED_NORMALIZATION = {
    "gamma": 1.4,
    "mach_mean": 1.0999999999999996,
    "mach_scale": 0.1,
    "mach_scale_floor": 0.1,
    "residual_scale": [
        0.004629576317875613,
        0.01090066324277191,
        0.016072209961331355,
        0.02287761670497486,
    ],
    "state_mean": [
        1.1259979427545967,
        1.3008588802569192,
        0.0020348638026725743,
        3.7132099831077374,
    ],
    "state_scale": [
        0.07523402869739157,
        0.05461684041716393,
        0.043580576660120066,
        0.2345080527811244,
    ],
    "weight_provenance": "validated_physical_cell_volume_normalized",
}
EXPECTED_MODEL_CONFIG = {
    "model": "PCNOEuler2DResidual",
    "in_dim": 12,
    "out_dim": 4,
    "k_max": 8,
    "domain_lengths": [2.0, 1.0],
    "layers": [128, 128, 128, 128, 128],
    "fc_dim": 128,
    "nmeasures": 1,
    "node_type_feature_mode": "one_hot",
    "boundary_field_mode": "none",
    "boundary_field_names": [],
    "boundary_residual_mode": "none",
    "boundary_residual_names": [],
    "boundary_residual_width": 64,
}

TRAINING_SEED = 20260814
SPLIT_SEED = 20260718
MIDPOINT_STRENGTHS = (0.21875, 0.24375, 0.26875, 0.29375, 0.31875, 0.34375)
BRANCH_STRENGTH_INDICES = frozenset({0, 2, 3, 5})
RECURRENCE_STRENGTH_INDICES = frozenset({1, 4})
POSITIONS = (
    ("y01", 0.3875, True),
    ("y04", 0.5, False),
    ("y07", 0.6125, True),
)
EVOLUTION_RESOLUTION = (1000, 400)
SAVED_DELTA_T = 0.01
FINAL_TIME = 0.6

SOURCE_PATHS = (
    "pcno/pcno.py",
    "utility/time_dependent_no/pcno_euler2d.py",
    "utility/time_dependent_no/pcno_resolution_transfer.py",
    "utility/time_dependent_no/pcno_response_filtered_block.py",
    "utility/time_dependent_no/pcno_same_state_refresh_metadata.py",
    "utility/time_dependent_no/pcno_same_state_refresh_resources.py",
    "utility/time_dependent_no/shock_vortex_fv.py",
    "scripts/time_dependent_no/analyze_pcno_binary_position_phase_rescore.py",
    "scripts/time_dependent_no/train_pcno_euler2d_residual.py",
    "scripts/time_dependent_no/build_pcno_same_state_refresh_resources.py",
)


@dataclass(frozen=True)
class ResourcePositionDescriptor:
    status: str
    vertical_centroid: float | None
    normalized_wall_distance: float | None
    transverse_velocity_l1: float


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def _output_times() -> tuple[float, ...]:
    return tuple(index / 100.0 for index in range(61))


def _base_reference_config() -> ShockVortexFVConfig:
    return ShockVortexFVConfig(
        nx=EVOLUTION_RESOLUTION[0],
        ny=EVOLUTION_RESOLUTION[1],
        coarse_nx=EXPECTED_FINE_RESOLUTION[0],
        coarse_ny=EXPECTED_FINE_RESOLUTION[1],
        t_final=FINAL_TIME,
        output_times=_output_times(),
        cfl=0.35,
        initial_quadrature_order=8,
    ).validated()


def source_sha256(root: str | Path) -> dict[str, str]:
    base = Path(root)
    missing = [path for path in SOURCE_PATHS if not (base / path).is_file()]
    if missing:
        raise FileNotFoundError(f"resource source files are missing: {missing}")
    return {path: sha256_file(base / path) for path in SOURCE_PATHS}


def _case_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for strength_index, epsilon in enumerate(MIDPOINT_STRENGTHS):
        if strength_index in BRANCH_STRENGTH_INDICES:
            population = "branch_label"
        elif strength_index in RECURRENCE_STRENGTH_INDICES:
            population = "recurrence"
        else:  # pragma: no cover - guarded by the frozen constants
            raise AssertionError("each strength must have exactly one population")
        group = f"a46_m{strength_index:02d}"
        for position_index, (position_id, vortex_y, active) in enumerate(POSITIONS):
            rows.append(
                {
                    "case_id": f"sv_a46_m{strength_index:02d}_{position_id}",
                    "trajectory_index": 3 * strength_index + position_index,
                    "strength_group": group,
                    "strength_index": strength_index,
                    "position_id": position_id,
                    "population": population,
                    "parameters": {
                        "vortex_epsilon": epsilon,
                        "vortex_y": vortex_y,
                    },
                    "expected_selector_active": active,
                    "inactive_control_retained": not active,
                    "project_test_member": False,
                    "outcome_opened": False,
                    "generated": False,
                }
            )
    return rows


def frozen_selector_active(normalized_wall_distance: float, input_call: int) -> bool:
    """Reproduce the frozen A31 position/phase decision without analysis imports."""

    if isinstance(input_call, bool) or not isinstance(input_call, (int, np.integer)):
        raise TypeError("input_call must be an integer")
    call = int(input_call)
    if call not in EXPECTED_INPUT_CALLS:
        raise ValueError("input_call must be in 0..29")
    distance = float(normalized_wall_distance)
    if not np.isfinite(distance):
        raise ValueError("normalized wall distance must be finite")
    return distance <= EXPECTED_SELECTOR_THRESHOLD and (call <= 7 or call >= 22)


def resource_position_descriptor(
    conservative_state: np.ndarray,
    *,
    nodes: np.ndarray,
    volumes: np.ndarray,
    y_min: float,
    y_max: float,
    density_floor: float = 1.0e-12,
    signal_floor: float = 1.0e-12,
) -> ResourcePositionDescriptor:
    """Compute the exact A31 call-zero transverse-velocity centroid."""

    state = np.asarray(conservative_state, dtype=np.float64)
    positions = np.asarray(nodes, dtype=np.float64)
    mass = np.asarray(volumes, dtype=np.float64)
    if (
        state.ndim != 2
        or state.shape[1] < 3
        or not np.isfinite(state).all()
        or positions.shape != (state.shape[0], 2)
        or not np.isfinite(positions).all()
        or mass.shape != (state.shape[0],)
        or not np.isfinite(mass).all()
        or np.any(mass <= 0.0)
    ):
        raise ValueError("state, nodes, and volumes must be finite and node-aligned")
    if (
        not np.isfinite([y_min, y_max, density_floor, signal_floor]).all()
        or y_min >= y_max
        or density_floor <= 0.0
        or signal_floor <= 0.0
    ):
        raise ValueError("position-descriptor bounds and floors are invalid")
    if np.any(positions[:, 1] < y_min) or np.any(positions[:, 1] > y_max):
        raise ValueError("node y coordinates lie outside the physical domain")
    density = state[:, 0]
    if np.any(density <= density_floor):
        raise ValueError("position descriptor requires positive density")
    weights = mass * np.abs(state[:, 2] / density)
    signal = float(np.sum(weights))
    if signal <= signal_floor * float(np.sum(mass)):
        return ResourcePositionDescriptor(
            status="unresolved_transverse_velocity",
            vertical_centroid=None,
            normalized_wall_distance=None,
            transverse_velocity_l1=signal,
        )
    centroid = float(np.dot(weights, positions[:, 1]) / signal)
    wall_distance = min(centroid - y_min, y_max - centroid)
    normalized = float(2.0 * wall_distance / (y_max - y_min))
    if not -1.0e-12 <= normalized <= 1.0 + 1.0e-12:
        raise ValueError("transverse-velocity centroid lies outside the domain")
    return ResourcePositionDescriptor(
        status="ok",
        vertical_centroid=centroid,
        normalized_wall_distance=float(np.clip(normalized, 0.0, 1.0)),
        transverse_velocity_l1=signal,
    )


def build_resource_plan(root: str | Path) -> dict[str, Any]:
    """Build the exact source-bound resource plan frozen before outcomes."""

    base = _base_reference_config()
    cases = _case_rows()
    active_branch = sorted(
        row["case_id"]
        for row in cases
        if row["population"] == "branch_label" and row["expected_selector_active"]
    )
    active_recurrence = sorted(
        row["case_id"]
        for row in cases
        if row["population"] == "recurrence" and row["expected_selector_active"]
    )
    payload: dict[str, Any] = {
        "schema": RESOURCE_PLAN_SCHEMA,
        "working_id": RESOURCE_BUILD_WORKING_ID,
        "status": "frozen_before_generation_training_or_outcome_opening",
        "frozen_before_outcomes": True,
        "reference_arrays_opened": False,
        "truth_arrays_loaded": False,
        "family": "dynamic_shock_vortex_fv",
        "source_sha256": source_sha256(root),
        "reference_contract": {
            "solver": "WENO5-HLLC-SSPRK3",
            "dtype": "float64",
            "evolution_resolution": list(EVOLUTION_RESOLUTION),
            "retained_fine_resolution": list(EXPECTED_FINE_RESOLUTION),
            "retained_native_resolution": list(EXPECTED_NATIVE_RESOLUTION),
            "native_construction": (
                "exact conservative block restriction of the retained fine state "
                "from the same high-fidelity evolution"
            ),
            "boundary_policy": "linear x extrapolation; y symmetry",
            "accepted_state_clipping_or_floors": False,
            "future_reference_boundary_values": False,
            "base_config": base.to_dict(),
            "saved_delta_t": SAVED_DELTA_T,
            "model_delta_t": EXPECTED_MODEL_DELTA_T,
            "model_step_stride": 2,
            "input_calls": list(EXPECTED_INPUT_CALLS),
            "output_calls": list(EXPECTED_OUTPUT_CALLS),
        },
        "selector": {
            "working_id": SELECTOR_WORKING_ID,
            "truth_free": True,
            "feature_clock": "call_zero_native_physical_state",
            "normalized_wall_distance_max": EXPECTED_SELECTOR_THRESHOLD,
            "expected_active_position_ids": ["y01", "y07"],
            "expected_inactive_position_ids": ["y04"],
            "source_path": (
                "scripts/time_dependent_no/"
                "analyze_pcno_binary_position_phase_rescore.py"
            ),
        },
        "allocation": {
            "rule": "frozen_balanced_strength_span_before_generation",
            "branch_strength_groups": ["a46_m00", "a46_m02", "a46_m03", "a46_m05"],
            "recurrence_strength_groups": ["a46_m01", "a46_m04"],
            "branch_fit_case_ids": active_branch,
            "recurrence_case_ids": active_recurrence,
        },
        "training_contract": {
            "reference_checkpoint_sha256": REFERENCE_CHECKPOINT_SHA256,
            "fresh_seed": TRAINING_SEED,
            "split_seed": SPLIT_SEED,
            "independent_initialization_required": True,
            "init_checkpoint": None,
            "resume_checkpoint": None,
            "data_manifest_digest": EXPECTED_DATA_MANIFEST_DIGEST,
            "normalization_digest": EXPECTED_NORMALIZATION_DIGEST,
            "normalization": EXPECTED_NORMALIZATION,
            "model_config": EXPECTED_MODEL_CONFIG,
            "model_node_type_input": "physical",
            "node_type_channel_control": "standard",
            "boundary_field_mode": "none",
            "boundary_residual_mode": "none",
            "boundary_mode": "model_all_nodes",
            "step_stride": 2,
            "gradient_policy": (
                "current physical-FV affine-reproducing gradient implementation; "
                "no gradient edit or ablation in this resource build"
            ),
        },
        "cases": cases,
        "activity_boundary": {
            "a46_branch_predictions": 0,
            "a46_controllers": 0,
            "animations": 0,
            "bump_cases": 0,
        },
    }
    return with_payload_sha256(payload)


def validate_resource_plan(
    value: Mapping[str, Any], root: str | Path
) -> dict[str, Any]:
    """Fail closed unless a resource plan equals the current frozen contract."""

    normalized = json.loads(json.dumps(value, allow_nan=False))
    if normalized.get("payload_sha256") != canonical_payload_sha256(normalized):
        raise ValueError("resource plan payload digest mismatch")
    expected = build_resource_plan(root)
    if normalized != expected:
        raise ValueError("resource plan differs from the frozen contract")
    return normalized


def load_resource_plan(path: str | Path, root: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError("resource plan must contain one JSON object")
    return validate_resource_plan(value, root)


def case_by_id(plan: Mapping[str, Any], case_id: str) -> dict[str, Any]:
    matches = [row for row in plan["cases"] if row["case_id"] == case_id]
    if len(matches) != 1:
        raise ValueError(f"resource case is not uniquely present: {case_id}")
    return dict(matches[0])


def config_for_case(plan: Mapping[str, Any], case_id: str) -> ShockVortexFVConfig:
    case = case_by_id(plan, case_id)
    fields = ShockVortexFVConfig.__dataclass_fields__
    stored = plan["reference_contract"]["base_config"]
    arguments = {name: stored[name] for name in fields if name in stored}
    arguments["output_times"] = tuple(arguments["output_times"])
    base = ShockVortexFVConfig(**arguments).validated()
    return replace(base, **case["parameters"]).validated()


def native_config(config: ShockVortexFVConfig) -> ShockVortexFVConfig:
    return replace(
        config,
        coarse_nx=EXPECTED_NATIVE_RESOLUTION[0],
        coarse_ny=EXPECTED_NATIVE_RESOLUTION[1],
    ).validated()


def derive_nested_reference(
    states: np.ndarray,
    *,
    fine_geometry: StructuredFVGeometry,
    fine_resolution: tuple[int, int],
    native_geometry: StructuredFVGeometry,
    native_resolution: tuple[int, int],
) -> dict[str, Any]:
    """Conservatively restrict states and verify volume-integrated consistency."""

    native_states = restrict_nested_state(
        states,
        fine_resolution=fine_resolution,
        coarse_resolution=native_resolution,
    )
    fine_totals = np.einsum(
        "n,tnc->tc", fine_geometry.cell_volume, states, optimize=True
    )
    native_totals = np.einsum(
        "n,tnc->tc", native_geometry.cell_volume, native_states, optimize=True
    )
    total_delta = native_totals - fine_totals
    return {
        "native_states": native_states,
        "native_geometry": native_geometry,
        "nested_restriction_max_abs": 0.0,
        "integrated_state_max_abs": float(np.max(np.abs(total_delta))),
        "integrated_state_relative_l2": float(
            np.linalg.norm(total_delta.reshape(-1))
            / max(np.linalg.norm(fine_totals.reshape(-1)), 1.0e-300)
        ),
        "fine_volume_min": float(np.min(fine_geometry.cell_volume)),
        "native_volume_min": float(np.min(native_geometry.cell_volume)),
    }


def derive_native_reference(
    result: ShockVortexReferenceResult,
) -> dict[str, Any]:
    """Derive native truth/geometry from one retained-fine solver result."""

    fine_resolution = (result.config.coarse_nx, result.config.coarse_ny)
    if fine_resolution != EXPECTED_FINE_RESOLUTION:
        raise ValueError("solver result does not use the frozen retained fine grid")
    geometry = make_structured_fv_geometry(native_config(result.config))
    return derive_nested_reference(
        result.states,
        fine_geometry=result.geometry,
        fine_resolution=EXPECTED_FINE_RESOLUTION,
        native_geometry=geometry,
        native_resolution=EXPECTED_NATIVE_RESOLUTION,
    )


def matched_training_arguments(
    data_dir: str | Path,
    output_dir: str | Path,
) -> list[str]:
    """Return the explicit current-trainer arguments for the fresh checkpoint."""

    return [
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(output_dir),
        "--seed",
        str(TRAINING_SEED),
        "--split-seed",
        str(SPLIT_SEED),
        "--split-mode",
        "manifest",
        "--val-count",
        "30",
        "--step-stride",
        "2",
        "--epochs",
        "50",
        "--presentation-mode",
        "balanced",
        "--presentations-per-epoch",
        "1024",
        "--val-presentations",
        "256",
        "--batch-size",
        "4",
        "--gradient-accumulation-steps",
        "1",
        "--tiny-pairs",
        "0",
        "--tiny-fit-rel-l2",
        "0.01",
        "--tiny-fit-loss-ratio",
        "0.01",
        "--stats-time-stride",
        "1",
        "--mach-scale-floor",
        "0.1",
        "--k-max",
        "8",
        "--domain-lengths",
        "2.0",
        "1.0",
        "--model-node-type-input",
        "physical",
        "--node-type-channel-control",
        "standard",
        "--boundary-field-mode",
        "none",
        "--boundary-residual-mode",
        "none",
        "--boundary-residual-width",
        "64",
        "--layers",
        "128",
        "128",
        "128",
        "128",
        "128",
        "--fc-dim",
        "128",
        "--learning-rate",
        "0.001",
        "--weight-decay",
        "0.00001",
        "--scheduler",
        "constant",
        "--lr-div-factor",
        "10",
        "--lr-final-div-factor",
        "1000",
        "--gradient-clip",
        "1.0",
        "--input-noise-std",
        "0.003",
        "--generated-state-exposure-weight",
        "0.0",
        "--multistep-loss-steps",
        "1",
        "--multistep-loss-weight",
        "0.0",
        "--multistep-recurrent-input",
        "attached_prediction",
        "--rollout-every",
        "5",
        "--rollout-val-count",
        "24",
        "--rollout-steps",
        "30",
        "--selection-mode",
        "historical_all_node",
        "--selection-short-horizon",
        "20",
        "--rollout-start-frame",
        "0",
        "--boundary-mode",
        "model_all_nodes",
        "--boundary-max-source-hops",
        "3",
        "--boundary-rho-inf",
        "1.4",
        "--boundary-p-inf",
        "1.0",
        "--primary-objective",
        "normal_closed",
        "--boundary-auxiliary",
        "none",
        "--boundary-auxiliary-weight",
        "0.0",
        "--near-boundary-hops",
        "2",
        "--max-wall-hours",
        "0.0",
        "--checkpoint-every",
        "1",
        "--device",
        "cuda",
        "--amp",
        "bf16",
        "--init-boundary-mode-transition",
        "none",
    ]


def build_population_manifest(
    plan: Mapping[str, Any],
    *,
    plan_file_sha256: str,
    summary_rows: Sequence[Mapping[str, Any]],
    opened_case_ids: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind summary-only case metadata without opening any truth array."""

    summaries = {str(row.get("case_id")): row for row in summary_rows}
    expected_ids = {row["case_id"] for row in plan["cases"]}
    if set(summaries) != expected_ids or len(summaries) != len(summary_rows):
        raise ValueError("resource summaries do not exactly cover the frozen cases")
    source_hash = plan["source_sha256"]
    solver_sha = source_hash["utility/time_dependent_no/shock_vortex_fv.py"]
    selector_sha = source_hash[
        "scripts/time_dependent_no/analyze_pcno_binary_position_phase_rescore.py"
    ]
    time_hashes: set[str] = set()
    rows: list[dict[str, Any]] = []
    for planned in plan["cases"]:
        summary = summaries[planned["case_id"]]
        if summary.get("status") != "passed":
            raise ValueError(f"resource case did not pass: {planned['case_id']}")
        if summary.get("resource_plan_payload_sha256") != plan["payload_sha256"]:
            raise ValueError("resource case plan binding mismatch")
        if summary.get("source_sha256") != source_hash:
            raise ValueError("resource case source hashes differ")
        if summary.get("parameters") != planned["parameters"]:
            raise ValueError("resource case parameters differ")
        if summary.get("selector_active") is not planned["expected_selector_active"]:
            raise ValueError("resource case selector decision differs")
        time_hash = str(summary.get("physical_times_sha256"))
        time_hashes.add(time_hash)
        rows.append(
            {
                "case_id": planned["case_id"],
                "strength_group": planned["strength_group"],
                "position_id": planned["position_id"],
                "population": planned["population"],
                "parameters": planned["parameters"],
                "selector_active": planned["expected_selector_active"],
                "generated": True,
                "outcome_opened": False,
                "project_test_member": False,
                "retained_evolved_native_truth": True,
                "retained_evolved_fine_truth": True,
                "physical_volume_min": min(
                    float(summary["native_volume_min"]),
                    float(summary["fine_volume_min"]),
                ),
                "truth_metadata_manifest_sha256": summary["summary_sha256"],
                "truth_array_sha256": summary["reference_artifact_sha256"],
                "solver_source_sha256": solver_sha,
                "configuration_sha256": summary["configuration_sha256"],
                "physical_times_sha256": time_hash,
                "inactive_control_retained": not planned["expected_selector_active"],
            }
        )
    if len(time_hashes) != 1:
        raise ValueError("resource cases have different physical-time hashes")
    physical_times_sha256 = next(iter(time_hashes))
    population = with_payload_sha256(
        {
            "schema": POPULATION_SCHEMA,
            "working_id": RESOURCE_BUILD_WORKING_ID,
            "family": "dynamic_shock_vortex_fv",
            "frozen_before_outcomes": True,
            "reference_arrays_opened": False,
            "truth_arrays_loaded": False,
            "resource_plan_sha256": plan_file_sha256,
            "resource_plan_payload_sha256": plan["payload_sha256"],
            "selector": {
                "working_id": SELECTOR_WORKING_ID,
                "truth_free": True,
                "feature_clock": "call_zero_native_physical_state",
                "normalized_wall_distance_max": EXPECTED_SELECTOR_THRESHOLD,
                "source_sha256": selector_sha,
            },
            "generation": {
                "inventory_complete": True,
                "newly_generated": True,
                "project_test_split_excluded": True,
                "solver": "WENO5-HLLC-SSPRK3",
                "solver_source_sha256": solver_sha,
                "configuration_sha256": canonical_json_sha256(
                    plan["reference_contract"]
                ),
                "family_manifest_sha256": plan_file_sha256,
                "single_evolution_dual_restriction": True,
            },
            "calls": {
                "input_calls": list(EXPECTED_INPUT_CALLS),
                "output_calls": list(EXPECTED_OUTPUT_CALLS),
                "model_delta_t": EXPECTED_MODEL_DELTA_T,
                "native_resolution": list(EXPECTED_NATIVE_RESOLUTION),
                "fine_resolution": list(EXPECTED_FINE_RESOLUTION),
                "physical_times_sha256": physical_times_sha256,
            },
            "branch_fit_case_ids": plan["allocation"]["branch_fit_case_ids"],
            "recurrence_case_ids": plan["allocation"]["recurrence_case_ids"],
            "cases": rows,
            "activity": {
                "truth_array_loads_for_manifest": 0,
                "outcomes_opened": 0,
                "checkpoint_calls": 0,
                "a46_branch_predictions": 0,
                "controllers_run": 0,
            },
        }
    )
    report = validate_population_manifest(population, opened_case_ids=opened_case_ids)
    if not report["valid"]:
        raise ValueError(
            f"population metadata failed closed: {report['failed_checks']}"
        )
    return population, report
