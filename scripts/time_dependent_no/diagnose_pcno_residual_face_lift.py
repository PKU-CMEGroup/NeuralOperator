#!/usr/bin/env python3
"""Run the D050 frozen PCNO-residual-to-face zero-training preflight.

The legal row allocates the PCNO-predicted global cell integral over x-boundary
faces by a deterministic minimum-W^-1 rule and then lifts the remaining
compatible residual through the fixed-mesh direct projector. It uses no future
reference, but its boundary allocation is algebraic rather than physical. A
separate oracle row retains accepted reference boundary impulses to measure
headroom and is never an autonomous-solver result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no.evaluate_pcno_shock_vortex_baseline import (  # noqa: E402
    endpoint_metrics,
    physical_call_metrics,
)
from utility.time_dependent_no.fv_impulse_diagnostics import (  # noqa: E402
    CLAIM_BOUNDARY,
    DirectMinimumNormProjector,
    FVImpulseOperators,
    build_fv_impulse_operators,
    factorize_direct_minimum_winv_norm_projector,
    minimum_winv_boundary_impulse_for_totals,
)
from utility.time_dependent_no.pcno_ripple_diagnostics import (  # noqa: E402
    raw_admissibility_summary,
    weighted_relative_l2_numpy,
)

SCHEMA = "pcno_residual_face_lift_preflight_v1"
EXPERIMENT_ID = "D050"
DEFAULT_TRAJECTORIES = (
    "sv_e00_y00",
    "sv_e00_y08",
    "sv_e06_y00",
    "sv_e06_y08",
    "sv_e11_y00",
    "sv_e11_y08",
)
DEFAULT_CALLS = (1, 10, 30, 60)
STRUCTURAL_THRESHOLDS = {
    "legal_state_reconstruction_relative_l2_max": 1.0e-8,
    "legal_lift_closure_relative_l2_max": 1.0e-8,
    "oracle_lift_closure_relative_l2_max": 1.0e-8,
    "legal_wall_exchange_absolute_max": 1.0e-14,
}
HEADROOM_THRESHOLDS = {
    "h60_state_error_reduction_median_min": 0.15,
    "h60_budget_defect_reduction_median_min": 0.80,
    "correction_to_raw_update_norm_max": 0.10,
    "h60_anti_smearing_ratio_max": 1.05,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--evaluation-summary", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--d049-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trajectories", nargs="+", default=list(DEFAULT_TRAJECTORIES))
    parser.add_argument("--calls", type=int, nargs="+", default=DEFAULT_CALLS)
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


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _validate_source_contract(
    *,
    evaluation_path: Path,
    handoff_path: Path,
    d049_path: Path,
    trajectories: Sequence[str],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    evaluation = _load_json(evaluation_path)
    handoff = _load_json(handoff_path)
    d049 = _load_json(d049_path)
    if (
        evaluation.get("schema") != "pcno_shock_vortex_physical_baseline_v1"
        or evaluation.get("status") != "complete"
    ):
        raise ValueError("D044 evaluation summary is not complete")
    if (
        handoff.get("schema") != "line3_to_line4_physical_baseline_handoff_v1"
        or handoff.get("status") != "frozen"
    ):
        raise ValueError("D044 handoff is not frozen")
    physical = handoff.get("physical_baseline", {})
    evaluation_sha256 = _sha256_file(evaluation_path)
    if evaluation_sha256 != physical.get("evaluation_report_sha256"):
        raise ValueError("D044 evaluation digest does not match the frozen handoff")
    checkpoint = evaluation.get("checkpoint", {})
    if checkpoint.get("sha256") != physical.get("checkpoint_sha256"):
        raise ValueError("D044 checkpoint digest differs between source reports")
    if checkpoint.get("config_digest") != physical.get("config_digest"):
        raise ValueError("D044 configuration digest differs between source reports")
    if (
        d049.get("schema") != "fv_divergence_conditioning_audit_v1"
        or d049.get("experiment_id") != "D049"
        or not d049.get("passed")
        or d049.get("result", {}).get("next_route")
        != "authorize_D050_zero_training_residual_to_face_preflight_only"
    ):
        raise ValueError("D049 does not authorize the D050 preflight")

    artifacts = evaluation.get("trajectory_artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("D044 trajectory artifact index is missing")
    by_key = {
        str(row["trajectory"]): row
        for row in artifacts
        if isinstance(row, dict) and "trajectory" in row
    }
    missing = sorted(set(trajectories) - set(by_key))
    if missing:
        raise ValueError(f"D044 evaluation is missing trajectories: {missing}")
    source = {
        "evaluation_summary": evaluation_path.as_posix(),
        "evaluation_summary_sha256": evaluation_sha256,
        "handoff": handoff_path.as_posix(),
        "handoff_sha256": _sha256_file(handoff_path),
        "d049_summary": d049_path.as_posix(),
        "d049_summary_sha256": _sha256_file(d049_path),
        "checkpoint_sha256": checkpoint["sha256"],
        "checkpoint_config_digest": checkpoint["config_digest"],
        "data_manifest_digest": evaluation["data_contract"]["data_manifest_digest"],
        "geometry_contract_digest": evaluation["data_contract"][
            "geometry_contract_digest"
        ],
        "geometry_digest": evaluation["data_contract"]["unique_geometry_digests"][0],
        "normalization_digest": evaluation["data_contract"]["normalization_digest"],
    }
    return evaluation, by_key, source


def _load_trajectory(
    path: Path,
    *,
    key: str,
    index_row: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    artifact_sha256 = _sha256_file(path)
    if artifact_sha256 != index_row.get("sha256"):
        raise ValueError(f"D044 trajectory digest mismatch for {key}")
    required = {
        "schema",
        "trajectory",
        "split",
        "parameters_json",
        "reference_config_json",
        "gamma",
        "state_component_scale",
        "shock_quantile",
        "physical_times",
        "initial_state",
        "targets",
        "predictions",
        "positions",
        "edges",
        "physical_cell_volumes",
        "mesh_cell_to_graph_node",
        "face_centers",
        "face_measures",
        "face_normals",
        "face_owner",
        "face_neighbor",
        "face_boundary_tag",
        "boundary_tag_names_json",
        "coordinate_convention",
        "face_orientation_convention",
        "reference_cumulative_accepted_substep_face_impulses",
        "reference_interval_boundary_exchange",
        "valid_length",
        "failure_cause",
        "checkpoint_sha256",
        "checkpoint_config_digest",
        "data_manifest_digest",
        "geometry_digest",
        "normalization_digest",
        "boundary_mode",
        "raw_recurrence",
        "inference_interventions_json",
    }
    with np.load(path, allow_pickle=False) as artifact:
        missing = sorted(required - set(artifact.files))
        if missing:
            raise ValueError(f"D044 trajectory {key} is missing arrays: {missing}")
        scalar = {
            name: artifact[name].item()
            for name in (
                "schema",
                "trajectory",
                "split",
                "gamma",
                "shock_quantile",
                "valid_length",
                "failure_cause",
                "checkpoint_sha256",
                "checkpoint_config_digest",
                "data_manifest_digest",
                "geometry_digest",
                "normalization_digest",
                "boundary_mode",
                "raw_recurrence",
                "coordinate_convention",
                "face_orientation_convention",
            )
        }
        data = {
            "parameters": json.loads(str(artifact["parameters_json"].item())),
            "reference_config": json.loads(
                str(artifact["reference_config_json"].item())
            ),
            "boundary_tag_names": tuple(
                json.loads(str(artifact["boundary_tag_names_json"].item()))
            ),
            "inference_interventions": json.loads(
                str(artifact["inference_interventions_json"].item())
            ),
        }
        for name in (
            required
            - set(scalar)
            - {
                "parameters_json",
                "reference_config_json",
                "boundary_tag_names_json",
                "inference_interventions_json",
            }
        ):
            data[name] = np.array(artifact[name])
    data.update(scalar)
    data["artifact_sha256"] = artifact_sha256

    if data["schema"] != "pcno_shock_vortex_physical_baseline_v1":
        raise ValueError(f"unsupported D044 trajectory schema for {key}")
    if data["trajectory"] != key or data["split"] != "validation":
        raise ValueError(f"trajectory identity/split mismatch for {key}")
    if int(data["valid_length"]) != 60 or data["failure_cause"] != "completed":
        raise ValueError(f"D044 trajectory {key} did not complete all 60 calls")
    if not bool(data["raw_recurrence"]) or any(
        bool(value) for value in data["inference_interventions"].values()
    ):
        raise ValueError(
            f"D044 trajectory {key} is not raw intervention-free recurrence"
        )
    digest_pairs = {
        "checkpoint_sha256": "checkpoint_sha256",
        "checkpoint_config_digest": "checkpoint_config_digest",
        "data_manifest_digest": "data_manifest_digest",
        "normalization_digest": "normalization_digest",
    }
    for field, source_field in digest_pairs.items():
        if data[field] != source[source_field]:
            raise ValueError(f"D044 trajectory {key} has mismatched {field}")
    if data["geometry_digest"] != source["geometry_digest"]:
        raise ValueError(f"D044 trajectory {key} has mismatched geometry_digest")
    mapping = np.asarray(data["mesh_cell_to_graph_node"], dtype=np.int64)
    if not np.array_equal(mapping, np.arange(mapping.size)):
        raise ValueError(f"D044 trajectory {key} does not use identity mesh ordering")
    return data


def _operators(data: Mapping[str, Any]) -> FVImpulseOperators:
    return build_fv_impulse_operators(
        cell_centers=data["positions"],
        cell_volume=data["physical_cell_volumes"],
        face_centers=data["face_centers"],
        face_measure=data["face_measures"],
        face_owner=data["face_owner"],
        face_neighbor=data["face_neighbor"],
        face_boundary_tag=data["face_boundary_tag"],
    )


def _winv_norm(field: np.ndarray, face_weight: np.ndarray) -> float:
    values = np.asarray(field, dtype=np.float64)
    return float(np.sqrt(np.sum(values * values / face_weight[:, None])))


def _volume_scaled_norm(
    field: np.ndarray, volume: np.ndarray, component_scale: np.ndarray
) -> float:
    values = np.asarray(field, dtype=np.float64) / np.asarray(component_scale)[None, :]
    return float(np.sqrt(np.sum(np.asarray(volume)[:, None] * values**2)))


def _safe_reduction(candidate: float, baseline: float) -> float:
    if baseline <= 0.0:
        return 0.0 if candidate == 0.0 else float("-inf")
    return 1.0 - candidate / baseline


def _safe_ratio(candidate: float, baseline: float) -> float:
    if baseline <= 0.0:
        return 1.0 if candidate == 0.0 else float("inf")
    return candidate / baseline


def _vortex_center(data: Mapping[str, Any], call: int) -> tuple[float, float]:
    config = data["reference_config"]
    absolute_time = float(np.asarray(data["physical_times"])[call])
    upstream_speed = float(config["shock_mach"]) * math.sqrt(float(config["gamma"]))
    shock_arrival = (
        float(config["shock_x"]) - float(config["vortex_x"])
    ) / upstream_speed
    vortex_x = (
        float(config["vortex_x"]) + upstream_speed * absolute_time
        if absolute_time <= shock_arrival
        else float(config["shock_x"])
        + float(config["right_u"]) * (absolute_time - shock_arrival)
    )
    return vortex_x, float(data["parameters"]["vortex_y"])


def _state_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    initial: np.ndarray,
    *,
    data: Mapping[str, Any],
    cumulative_exchange: np.ndarray,
    call: int,
) -> dict[str, Any]:
    volume = np.asarray(data["physical_cell_volumes"], dtype=np.float64)
    component_scale = np.asarray(data["state_component_scale"], dtype=np.float64)
    admissibility = raw_admissibility_summary(prediction, gamma=float(data["gamma"]))
    physical = physical_call_metrics(
        prediction,
        target,
        initial,
        volumes=volume,
        reference_cumulative_boundary_exchange=cumulative_exchange,
        component_scale=component_scale,
    )
    structure = endpoint_metrics(
        prediction,
        target,
        positions=data["positions"],
        edges=data["edges"],
        volumes=volume,
        component_scale=component_scale,
        gamma=float(data["gamma"]),
        shock_quantile=float(data["shock_quantile"]),
        vortex_center=_vortex_center(data, call),
    )
    return {
        "scaled_relative_l2_physical_volume": weighted_relative_l2_numpy(
            prediction, target, volume, component_scale
        ),
        **{f"admissibility_{key}": value for key, value in admissibility.items()},
        **physical,
        **structure,
    }


def _trajectory_rows(
    data: Mapping[str, Any],
    operators: FVImpulseOperators,
    projector: DirectMinimumNormProjector,
    *,
    calls: Sequence[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    predictions = np.asarray(data["predictions"], dtype=np.float64)
    targets = np.asarray(data["targets"], dtype=np.float64)
    initial = np.asarray(data["initial_state"], dtype=np.float64)
    volume = np.asarray(data["physical_cell_volumes"], dtype=np.float64)
    component_scale = np.asarray(data["state_component_scale"], dtype=np.float64)
    reference_impulse = np.asarray(
        data["reference_cumulative_accepted_substep_face_impulses"],
        dtype=np.float64,
    )
    interval_exchange = np.asarray(
        data["reference_interval_boundary_exchange"], dtype=np.float64
    )
    expected_state_shape = (60, operators.topology.num_cells, 4)
    expected_face_shape = (60, operators.topology.num_faces, 4)
    if (
        predictions.shape != expected_state_shape
        or targets.shape != expected_state_shape
    ):
        raise ValueError(
            f"D044 state arrays have incompatible shape for {data['trajectory']}"
        )
    if initial.shape != expected_state_shape[1:]:
        raise ValueError(
            f"D044 initial state has incompatible shape for {data['trajectory']}"
        )
    if reference_impulse.shape != expected_face_shape or interval_exchange.shape != (
        60,
        4,
    ):
        raise ValueError(
            f"D044 reference impulse arrays have incompatible shape for {data['trajectory']}"
        )
    dynamic_arrays = (
        predictions,
        targets,
        initial,
        reference_impulse,
        interval_exchange,
    )
    if not all(np.all(np.isfinite(value)) for value in dynamic_arrays):
        raise ValueError(f"D044 dynamic arrays are nonfinite for {data['trajectory']}")

    boundary_index = operators.boundary_face_indices
    boundary_tags = operators.face_boundary_tag[boundary_index]
    names = tuple(data["boundary_tag_names"])
    x_tags = [index for index, name in enumerate(names) if name in {"x_min", "x_max"}]
    if len(x_tags) != 2:
        raise ValueError("D050 requires named x_min and x_max boundary tags")
    allowed_x = np.isin(boundary_tags, x_tags)
    reference_exchange_from_faces = np.sum(
        reference_impulse[:, boundary_index, :], axis=1
    )
    exchange_relative = float(
        np.linalg.norm(reference_exchange_from_faces - interval_exchange)
        / max(float(np.linalg.norm(interval_exchange)), 1.0e-30)
    )
    if exchange_relative > 1.0e-10:
        raise ValueError("reference face impulses and boundary exchange disagree")

    state_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    for call in calls:
        prediction = predictions[call - 1]
        target = targets[call - 1]
        target_cell_integral = -volume[:, None] * (prediction - initial)
        predicted_total = np.sum(target_cell_integral, axis=0)
        legal_boundary = minimum_winv_boundary_impulse_for_totals(
            operators,
            predicted_total,
            allowed_boundary=allowed_x,
        )
        reference_face = np.sum(reference_impulse[:call], axis=0)
        oracle_boundary = reference_face[boundary_index]
        legal = projector.solve(target_cell_integral, legal_boundary)
        oracle = projector.solve(target_cell_integral, oracle_boundary)
        legal_closure = legal.reduced_solve_residual_relative_l2
        oracle_closure = oracle.reduced_solve_residual_relative_l2
        if legal_closure is None or oracle_closure is None:
            raise RuntimeError("D050 requires direct-projector closure diagnostics")
        legal_state = initial - legal.decoded_cell_integral / volume[:, None]
        oracle_state = initial - oracle.decoded_cell_integral / volume[:, None]
        cumulative_exchange = np.sum(interval_exchange[:call], axis=0)
        common = {
            "trajectory": data["trajectory"],
            "split_group_id": "position_ood",
            "call": int(call),
            "physical_time": float(
                np.asarray(data["physical_times"])[call]
                - np.asarray(data["physical_times"])[0]
            ),
        }
        raw_metrics = _state_metrics(
            prediction,
            target,
            initial,
            data=data,
            cumulative_exchange=cumulative_exchange,
            call=call,
        )
        legal_metrics = _state_metrics(
            legal_state,
            target,
            initial,
            data=data,
            cumulative_exchange=cumulative_exchange,
            call=call,
        )
        oracle_metrics = _state_metrics(
            oracle_state,
            target,
            initial,
            data=data,
            cumulative_exchange=cumulative_exchange,
            call=call,
        )

        legal_face_norm = _winv_norm(legal.face_impulse, operators.face_weight)
        oracle_face_norm = _winv_norm(oracle.face_impulse, operators.face_weight)
        legal_reconstruction = weighted_relative_l2_numpy(
            legal_state, prediction, volume, component_scale
        )
        raw_update_norm = _volume_scaled_norm(
            prediction - initial, volume, component_scale
        )
        correction_norm = _volume_scaled_norm(
            oracle_state - prediction, volume, component_scale
        )
        legal_wall_exchange = float(
            np.max(np.abs(legal.face_impulse[boundary_index[~allowed_x]]), initial=0.0)
        )
        state_rows.extend(
            (
                {
                    **common,
                    "variant": "raw_D044_state_residual",
                    "uses_future_reference_boundary": False,
                    "boundary_contract": "D044_model_all_nodes_state_prediction",
                    "physical_boundary_claim": False,
                    "face_winv_norm": None,
                    "lift_decoded_gain": None,
                    "lift_closure_relative_l2": None,
                    **raw_metrics,
                },
                {
                    **common,
                    "variant": "legal_algebraic_minimum_norm_face_lift",
                    "uses_future_reference_boundary": False,
                    "boundary_contract": (
                        "predicted_total_minimum_Winv_x_boundary_allocation; "
                        "algebraic_not_physical"
                    ),
                    "physical_boundary_claim": False,
                    "face_winv_norm": legal_face_norm,
                    "lift_decoded_gain": _volume_scaled_norm(
                        legal_state - initial, volume, np.ones(4)
                    )
                    / max(legal_face_norm, 1.0e-30),
                    "lift_closure_relative_l2": legal_closure,
                    "target_projection_relative_l2": (
                        legal.decoded_residual_relative_l2
                    ),
                    **legal_metrics,
                },
                {
                    **common,
                    "variant": "oracle_reference_boundary_face_lift",
                    "uses_future_reference_boundary": True,
                    "boundary_contract": (
                        "accepted_reference_cumulative_boundary_impulse; "
                        "headroom_only_not_autonomous"
                    ),
                    "physical_boundary_claim": False,
                    "face_winv_norm": oracle_face_norm,
                    "lift_decoded_gain": _volume_scaled_norm(
                        oracle_state - initial, volume, np.ones(4)
                    )
                    / max(oracle_face_norm, 1.0e-30),
                    "lift_closure_relative_l2": oracle_closure,
                    "target_projection_relative_l2": (
                        oracle.decoded_residual_relative_l2
                    ),
                    **oracle_metrics,
                },
            )
        )
        raw_error = float(raw_metrics["scaled_relative_l2_physical_volume"])
        oracle_error = float(oracle_metrics["scaled_relative_l2_physical_volume"])
        raw_budget = float(
            raw_metrics["prediction_reference_balance_component_scaled_rmse"]
        )
        oracle_budget = float(
            oracle_metrics["prediction_reference_balance_component_scaled_rmse"]
        )
        paired_rows.append(
            {
                **common,
                "legal_state_reconstruction_relative_l2": legal_reconstruction,
                "legal_lift_closure_relative_l2": legal_closure,
                "oracle_lift_closure_relative_l2": oracle_closure,
                "legal_target_projection_relative_l2": (
                    legal.decoded_residual_relative_l2
                ),
                "oracle_target_projection_relative_l2": (
                    oracle.decoded_residual_relative_l2
                ),
                "legal_compatibility_relative_l2": legal.compatibility_relative_l2,
                "oracle_compatibility_relative_l2": oracle.compatibility_relative_l2,
                "legal_wall_exchange_absolute": legal_wall_exchange,
                "raw_state_error": raw_error,
                "oracle_state_error": oracle_error,
                "oracle_state_error_reduction": _safe_reduction(
                    oracle_error, raw_error
                ),
                "raw_budget_defect": raw_budget,
                "oracle_budget_defect": oracle_budget,
                "oracle_budget_defect_reduction": _safe_reduction(
                    oracle_budget, raw_budget
                ),
                "correction_to_raw_update_norm": correction_norm
                / max(raw_update_norm, 1.0e-30),
                "oracle_to_raw_front_chamfer_ratio": _safe_ratio(
                    float(oracle_metrics["front_symmetric_chamfer"]),
                    float(raw_metrics["front_symmetric_chamfer"]),
                ),
                "oracle_to_raw_shock_thickness_error_ratio": _safe_ratio(
                    float(oracle_metrics["shock_thickness_log_error"]),
                    float(raw_metrics["shock_thickness_log_error"]),
                ),
                "oracle_to_raw_shock_strength_error_ratio": _safe_ratio(
                    float(oracle_metrics["shock_strength_log_error"]),
                    float(raw_metrics["shock_strength_log_error"]),
                ),
                "oracle_to_raw_smooth_error_ratio": _safe_ratio(
                    float(oracle_metrics["smooth_region_scaled_relative_l2"]),
                    float(raw_metrics["smooth_region_scaled_relative_l2"]),
                ),
                "raw_all_admissible": bool(raw_metrics["admissibility_all_admissible"]),
                "legal_all_admissible": bool(
                    legal_metrics["admissibility_all_admissible"]
                ),
                "oracle_all_admissible": bool(
                    oracle_metrics["admissibility_all_admissible"]
                ),
            }
        )

    provenance = {"reference_boundary_exchange_relative_l2": exchange_relative}
    return state_rows, paired_rows, provenance


def _aggregate(paired_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not paired_rows:
        raise ValueError("D050 requires paired endpoint rows")
    h60 = [row for row in paired_rows if int(row["call"]) == 60]
    if not h60:
        raise ValueError("D050 requires the frozen H60 endpoint")
    structural_measurements = {
        "legal_state_reconstruction_relative_l2_max": max(
            float(row["legal_state_reconstruction_relative_l2"]) for row in paired_rows
        ),
        "legal_lift_closure_relative_l2_max": max(
            float(row["legal_lift_closure_relative_l2"]) for row in paired_rows
        ),
        "oracle_lift_closure_relative_l2_max": max(
            float(row["oracle_lift_closure_relative_l2"]) for row in paired_rows
        ),
        "legal_wall_exchange_absolute_max": max(
            float(row["legal_wall_exchange_absolute"]) for row in paired_rows
        ),
    }
    structural_checks = {
        name: bool(structural_measurements[name] <= threshold)
        for name, threshold in STRUCTURAL_THRESHOLDS.items()
    }
    structural_checks["all_raw_legal_oracle_admissible"] = all(
        bool(row["raw_all_admissible"])
        and bool(row["legal_all_admissible"])
        and bool(row["oracle_all_admissible"])
        for row in paired_rows
    )
    anti_smearing_metrics = (
        "oracle_to_raw_front_chamfer_ratio",
        "oracle_to_raw_shock_thickness_error_ratio",
        "oracle_to_raw_shock_strength_error_ratio",
        "oracle_to_raw_smooth_error_ratio",
    )
    anti_smearing_medians = {
        name: float(np.median([float(row[name]) for row in h60]))
        for name in anti_smearing_metrics
    }
    headroom_measurements = {
        "h60_state_error_reduction_median_min": float(
            np.median([float(row["oracle_state_error_reduction"]) for row in h60])
        ),
        "h60_budget_defect_reduction_median_min": float(
            np.median([float(row["oracle_budget_defect_reduction"]) for row in h60])
        ),
        "correction_to_raw_update_norm_max": max(
            float(row["correction_to_raw_update_norm"]) for row in paired_rows
        ),
        "h60_anti_smearing_ratio_max": max(anti_smearing_medians.values()),
    }
    headroom_checks = {}
    for name, threshold in HEADROOM_THRESHOLDS.items():
        if name.endswith("_max"):
            headroom_checks[name] = bool(headroom_measurements[name] <= threshold)
        else:
            headroom_checks[name] = bool(headroom_measurements[name] >= threshold)
    structural_passed = all(structural_checks.values())
    headroom_passed = all(headroom_checks.values())
    if not structural_passed:
        classification = "residual_to_face_lift_contract_failed"
        next_route = "stop_and_reaudit_D050_no_face_method"
    elif headroom_passed:
        classification = (
            "lift_is_structurally_valid_and_oracle_boundary_headroom_is_material"
        )
        next_route = (
            "design_legal_current_state_boundary_budget_preflight_no_training_yet"
        )
    else:
        classification = (
            "lift_is_structurally_valid_but_oracle_boundary_headroom_is_insufficient"
        )
        next_route = "stop_face_route_keep_D044_state_residual_baseline"
    return {
        "structural_passed": structural_passed,
        "oracle_headroom_passed": headroom_passed,
        "classification": classification,
        "next_route": next_route,
        "structural_measurements": structural_measurements,
        "structural_thresholds": dict(STRUCTURAL_THRESHOLDS),
        "structural_checks": structural_checks,
        "headroom_measurements": headroom_measurements,
        "headroom_thresholds": dict(HEADROOM_THRESHOLDS),
        "headroom_checks": headroom_checks,
        "h60_anti_smearing_median_ratios": anti_smearing_medians,
        "h60_raw_state_error_median": float(
            np.median([float(row["raw_state_error"]) for row in h60])
        ),
        "h60_oracle_state_error_median": float(
            np.median([float(row["oracle_state_error"]) for row in h60])
        ),
        "h60_raw_budget_defect_median": float(
            np.median([float(row["raw_budget_defect"]) for row in h60])
        ),
        "h60_oracle_budget_defect_median": float(
            np.median([float(row["oracle_budget_defect"]) for row in h60])
        ),
    }


def _csv_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
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


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    trajectories = tuple(dict.fromkeys(str(value) for value in args.trajectories))
    calls = tuple(dict.fromkeys(int(value) for value in args.calls))
    if trajectories != DEFAULT_TRAJECTORIES:
        raise ValueError("D050 trajectory cohort is frozen and may not be changed")
    if calls != DEFAULT_CALLS:
        raise ValueError("D050 endpoint calls are frozen and may not be changed")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    evaluation, artifact_index, source = _validate_source_contract(
        evaluation_path=args.evaluation_summary,
        handoff_path=args.handoff,
        d049_path=args.d049_summary,
        trajectories=trajectories,
    )

    state_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    trajectory_sources: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {}
    operators: FVImpulseOperators | None = None
    projector: DirectMinimumNormProjector | None = None
    for key in trajectories:
        index_row = artifact_index[key]
        path = args.trajectory_dir / Path(str(index_row["artifact"])).name
        data = _load_trajectory(
            path,
            key=key,
            index_row=index_row,
            source=source,
        )
        if operators is None:
            operators = _operators(data)
            projector = factorize_direct_minimum_winv_norm_projector(operators)
        if projector is None or operators is None:
            raise RuntimeError("D050 projector initialization failed")
        rows, pairs, checks = _trajectory_rows(
            data,
            operators,
            projector,
            calls=calls,
        )
        state_rows.extend(rows)
        paired_rows.extend(pairs)
        provenance[key] = checks
        trajectory_sources.append(
            {
                "trajectory": key,
                "artifact": path.as_posix(),
                "artifact_sha256": data["artifact_sha256"],
                "reference_artifact_sha256": index_row["reference_artifact_sha256"],
            }
        )
    if len(state_rows) != 3 * len(trajectories) * len(calls):
        raise RuntimeError("D050 state table is incomplete")
    if len(paired_rows) != len(trajectories) * len(calls):
        raise RuntimeError("D050 paired table is incomplete")
    result = _aggregate(paired_rows)

    args.output_dir.mkdir(parents=True)
    tables = {"state_rows.csv": state_rows, "paired_rows.csv": paired_rows}
    for name, rows in tables.items():
        _write_csv(args.output_dir / name, rows)

    configuration = {
        "trajectories": list(trajectories),
        "calls": list(calls),
        "split": "validation_position_OOD_predeclared_D013_cohort",
        "legal_boundary": (
            "minimum_Winv allocation of PCNO-predicted total cell integral over "
            "x_min/x_max faces only"
        ),
        "oracle_boundary": "accepted cumulative reference boundary impulse",
        "projector": "once-factored direct weighted graph Laplacian",
    }
    summary = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "complete" if result["structural_passed"] else "failed",
        "entrypoint": Path(__file__).relative_to(ROOT).as_posix(),
        "entrypoint_sha256": _sha256_file(Path(__file__)),
        "configuration": configuration,
        "configuration_digest": _digest_mapping(configuration),
        "source_contract": source,
        "source_trajectories": trajectory_sources,
        "source_evaluation_contract": {
            "schema": evaluation["schema"],
            "split": evaluation["evaluation"]["split"],
            "raw_recurrence": evaluation["evaluation"]["raw_recurrence"],
            "boundary_mode": evaluation["evaluation"]["boundary_mode"],
            "inference_interventions": evaluation["evaluation"][
                "inference_interventions"
            ],
            "checkpoint": evaluation["checkpoint"],
            "data_contract": evaluation["data_contract"],
        },
        "physical_mesh_topology": operators.topology.to_dict(),
        "direct_projector": projector.summary(),
        "reference_boundary_checks": provenance,
        "row_counts": {
            "state": len(state_rows),
            "paired": len(paired_rows),
        },
        "result": result,
        "information_contracts": {
            "raw_D044": (
                "frozen intervention-free state rollout already generated by D044"
            ),
            "legal_lift": (
                "uses only frozen predicted state residual, validated geometry, and "
                "an algebraic predicted-total boundary allocation; no future reference"
            ),
            "oracle_lift": (
                "uses accepted future reference boundary impulse and is headroom only, "
                "never an autonomous result"
            ),
        },
        "diagnostic_weights": {
            "state": "validated physical cell volume and D044 training scale",
            "face": "sum_f I_f^2 / (A_f d_f)",
            "compatibility_projection": "componentwise arithmetic mean cell integral",
            "lift_closure": "relative residual of the reduced compatible direct system",
            "target_projection": (
                "decoded residual against the original unprojected D044 cell integral"
            ),
        },
        "training_or_model_execution": False,
        "test_trajectory_access": [],
        "inference_intervention_added_to_D044": False,
        "unsupported_claims": [
            "the algebraic legal boundary allocation is a physical boundary flux",
            "the oracle-reference-boundary row is an autonomous solver",
            "deterministic lifting improves the D044 recurrent dynamics",
            "a learned face or boundary model is authorized",
            "physical conservation follows without a validated autonomous boundary law",
        ],
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
                "classification": result["classification"],
                "next_route": result["next_route"],
                "summary": summary_path.as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0 if result["structural_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
