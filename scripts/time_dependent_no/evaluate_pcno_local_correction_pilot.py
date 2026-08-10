#!/usr/bin/env python3
"""D080-B: deterministic native rollout pilot for causal local corrections."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.time_dependent_no.evaluate_pcno_deterministic_response_controller as d078
import scripts.time_dependent_no.evaluate_pcno_native_residual_correction as parent
import scripts.time_dependent_no.evaluate_pcno_response_gain_controller as d076
from scripts.time_dependent_no.evaluate_pcno_node_type_interventions import CaseData
from utility.time_dependent_no.pcno_defect_corrections import (
    DissipationResult,
    controlled_graph_dissipation,
)
from utility.time_dependent_no.pcno_residual_structure import shock_vortex_regions

SCHEMA = "pcno_local_correction_pilot_d080b_v1"
EXPERIMENT_CONTRACT = "d080b_deterministic_local_correction_pilot"
D080A_SUMMARY_SHA256 = (
    "8570ac47b1ae989c5aef5ae29617d6f67bc97729d0ff6fb0f52d2bc1d59e767d"
)
SENSOR_QUANTILE = 0.8
LOCAL_MEAN_LIMIT = 1.0e-12
LOCAL_CAP_LIMIT = 1.0e-12
CONTACT_SUPPORT_LIMIT = 0.0
ENDPOINT_WORST_LIMIT = 1.02
CONTROL_WORST_LIMIT = 1.05
VISUAL_ARMS = (
    "zero",
    "persistent",
    "combined_shock_isotropic",
    "combined_shock_normal",
    "combined_vortex_isotropic",
)


@dataclass(frozen=True)
class LocalArmSpec:
    name: str
    pathway: str | None
    cap: float
    include_persistent: bool
    local_start_call: int = 1
    local_stop_call: int | None = None


ARM_SPECS = (
    LocalArmSpec("zero", None, 0.0, False),
    LocalArmSpec("persistent", None, 0.0, True),
    LocalArmSpec("local_shock_isotropic", "shock_isotropic", 0.01, False),
    LocalArmSpec("local_shock_normal", "shock_normal", 0.01, False),
    LocalArmSpec("local_vortex_isotropic", "vortex_isotropic", 0.005, False),
    LocalArmSpec("combined_shock_isotropic", "shock_isotropic", 0.01, True),
    LocalArmSpec("combined_shock_normal", "shock_normal", 0.01, True),
    LocalArmSpec("combined_vortex_isotropic", "vortex_isotropic", 0.005, True),
)


@dataclass
class LocalRolloutPayload:
    result: parent.RolloutResult
    persistent_corrections: np.ndarray
    local_corrections: np.ndarray
    local_audit_rows: list[dict[str, Any]]


def _unique_undirected_edges(edges: np.ndarray) -> np.ndarray:
    directed = np.asarray(edges, dtype=np.int64)
    if directed.ndim != 2 or directed.shape[1] != 2:
        raise ValueError("D080-B edges must have shape [E,2]")
    ordered = np.sort(directed, axis=1)
    ordered = ordered[ordered[:, 0] != ordered[:, 1]]
    unique = np.unique(ordered, axis=0)
    if unique.shape[0] != 49_650:
        raise ValueError("D080-B native undirected edge inventory changed")
    return unique


def _weighted_energy(field: np.ndarray, case: CaseData, scale: np.ndarray) -> float:
    values = np.asarray(field, dtype=np.float64) / scale[None, :]
    return float(
        np.einsum("n,nc,nc->", case.weights, values, values, optimize=True)
        / case.weights.sum()
    )


def _relative_l2(first: np.ndarray, second: np.ndarray) -> float:
    difference = np.asarray(first, dtype=np.float64) - np.asarray(
        second, dtype=np.float64
    )
    denominator = float(np.linalg.norm(np.asarray(second, dtype=np.float64)))
    return (
        0.0 if denominator == 0.0 else float(np.linalg.norm(difference) / denominator)
    )


def _shock_edge_multiplier(
    nodes: np.ndarray, edges: np.ndarray, normals: np.ndarray
) -> np.ndarray:
    left, right = edges.T
    direction = nodes[right] - nodes[left]
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    edge_normal = normals[left] + normals[right]
    edge_normal /= np.linalg.norm(edge_normal, axis=1, keepdims=True)
    return np.clip(np.square(np.einsum("ec,ec->e", direction, edge_normal)), 0.0, 1.0)


def _local_correction(
    update: np.ndarray,
    current: np.ndarray,
    case: CaseData,
    unique_edges: np.ndarray,
    arm: LocalArmSpec,
) -> tuple[DissipationResult, int]:
    if arm.pathway is None:
        zero = np.zeros_like(update, dtype=np.float64)
        return (
            DissipationResult(
                correction=zero,
                sensor=np.zeros(update.shape[0], dtype=np.float64),
                uncapped_relative_norm=0.0,
                applied_relative_norm=0.0,
                applied_scale=0.0,
                weighted_mean_closure=np.zeros(update.shape[1], dtype=np.float64),
                eligible_edge_count=0,
            ),
            0,
        )
    if case.resolution is None:
        raise ValueError("D080-B requires the native structured resolution")
    resolution = tuple(int(value) for value in case.resolution)
    interior = case.physical_node_type == 0
    masks, _, normals = shock_vortex_regions(
        current,
        case.nodes,
        resolution=resolution,
        gamma=case.gamma,
    )
    if arm.pathway.startswith("shock_"):
        gate = masks["shock_envelope_le_0.05"] & interior
    elif arm.pathway == "vortex_isotropic":
        gate = masks["partition_vortex"] & interior
    else:
        raise ValueError(f"unsupported D080-B local pathway: {arm.pathway}")
    multiplier = (
        _shock_edge_multiplier(case.nodes, unique_edges, normals)
        if arm.pathway == "shock_normal"
        else None
    )
    result = controlled_graph_dissipation(
        update,
        unique_edges,
        case.weights,
        interior,
        component_scale=case.residual_scale,
        sensor_quantile=SENSOR_QUANTILE,
        norm_cap=arm.cap,
        node_gate=gate.astype(np.float64),
        edge_multiplier=multiplier,
        edges_are_unique_undirected=True,
    )
    return result, int(gate.sum())


def _arm_candidate(arm: LocalArmSpec, selected: parent.Candidate) -> parent.Candidate:
    if arm.name == "zero":
        return parent.Candidate(key="zero", rank=0, gain=0.0)
    if arm.name == "persistent":
        return selected
    rank = 8 if arm.include_persistent and not selected.is_zero else 1
    gain = selected.gain if rank == 8 else arm.cap
    return parent.Candidate(key=arm.name, rank=rank, gain=gain)


def _sequence_array(values: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    return (
        np.asarray(values, dtype=np.float64)
        if values
        else np.empty((0, *shape), dtype=np.float64)
    )


def _local_active_on_call(arm: LocalArmSpec, call: int) -> bool:
    if call < 1:
        raise ValueError("D080-B local-correction calls are one-indexed")
    if arm.local_start_call < 1:
        raise ValueError("D080-B local-correction start call must be positive")
    if arm.local_stop_call is not None and arm.local_stop_call < arm.local_start_call:
        raise ValueError("D080-B local-correction stop precedes its start")
    return bool(
        arm.pathway is not None
        and call >= arm.local_start_call
        and (arm.local_stop_call is None or call <= arm.local_stop_call)
    )


def _rollout_arm(
    model: torch.nn.Module,
    case: CaseData,
    selected: parent.Candidate,
    *,
    bias_sequence: np.ndarray,
    arm: LocalArmSpec,
    shock_quantile: float,
    calls: int,
) -> LocalRolloutPayload:
    available_calls = case.reference_states.shape[0] - 1
    if calls < 1 or calls > available_calls:
        raise ValueError("D080-B rollout call count lies outside the reference")
    if bias_sequence.shape != case.reference_states[1:].shape:
        raise ValueError("D080-B persistent bias does not match the native trajectory")
    unique_edges = _unique_undirected_edges(case.edges)
    interior = case.physical_node_type == 0
    current = np.array(case.reference_states[0], copy=True)
    states = [current.copy()]
    base_defects: list[np.ndarray] = []
    total_corrections: list[np.ndarray] = []
    persistent_corrections: list[np.ndarray] = []
    local_corrections: list[np.ndarray] = []
    defects: list[np.ndarray] = []
    truth_increments: list[np.ndarray] = []
    admissibility_rows: list[dict[str, Any]] = []
    local_audit_rows: list[dict[str, Any]] = []
    valid_length = 0
    for call in range(1, calls + 1):
        base_prediction = parent._predict(model, case, current)
        base_summary = parent._safe_admissibility_summary(
            base_prediction, gamma=case.gamma
        )
        if not base_summary["finite"]:
            admissibility_rows.append(
                {
                    "call": call,
                    "accepted": False,
                    "failure_stage": "base_prediction_nonfinite",
                    **base_summary,
                }
            )
            break
        base_update = base_prediction - current
        local_arm = (
            arm
            if _local_active_on_call(arm, call)
            else replace(arm, pathway=None, cap=0.0)
        )
        local_result, active_nodes = _local_correction(
            base_update, current, case, unique_edges, local_arm
        )
        persistent = (
            -selected.gain * bias_sequence[call - 1]
            if arm.include_persistent and not selected.is_zero
            else np.zeros_like(base_prediction)
        )
        local = local_result.correction
        total_correction = persistent + local
        prediction = base_prediction + total_correction
        truth_increment = case.reference_states[call] - case.reference_states[call - 1]
        base_defect = base_update - truth_increment
        defect = base_defect + total_correction
        admissibility = parent._safe_admissibility_summary(prediction, gamma=case.gamma)
        accepted = bool(admissibility["finite"] and admissibility["admissible"])
        admissibility_rows.append(
            {
                "call": call,
                "accepted": accepted,
                "failure_stage": None if accepted else "proposal_inadmissible",
                **admissibility,
            }
        )
        mean_closure_scaled = float(
            np.max(np.abs(local_result.weighted_mean_closure / case.residual_scale))
        )
        support_leakage = (
            float(np.max(np.abs(local[~interior]))) if np.any(~interior) else 0.0
        )
        local_audit_rows.append(
            {
                "family": case.family,
                "case_id": case.case_id,
                "resolution": case.resolution_name,
                "arm": arm.name,
                "pathway": arm.pathway or "none",
                "norm_cap": arm.cap,
                "call": call,
                "physical_time": float(case.physical_times[call]),
                "active_node_count": active_nodes,
                "eligible_edge_count": local_result.eligible_edge_count,
                "uncapped_relative_norm": local_result.uncapped_relative_norm,
                "applied_relative_norm": local_result.applied_relative_norm,
                "cap_excess": max(0.0, local_result.applied_relative_norm - arm.cap),
                "maximum_local_mean_closure_scaled": mean_closure_scaled,
                "maximum_non_type0_local_correction": support_leakage,
                "base_update_residual_scaled_energy": _weighted_energy(
                    base_update, case, case.residual_scale
                ),
                "persistent_correction_residual_scaled_energy": _weighted_energy(
                    persistent, case, case.residual_scale
                ),
                "local_correction_residual_scaled_energy": _weighted_energy(
                    local, case, case.residual_scale
                ),
                "total_correction_residual_scaled_energy": _weighted_energy(
                    total_correction, case, case.residual_scale
                ),
            }
        )
        if not admissibility["finite"]:
            break
        states.append(np.asarray(prediction, dtype=np.float64))
        base_defects.append(np.asarray(base_defect, dtype=np.float64))
        total_corrections.append(np.asarray(total_correction, dtype=np.float64))
        persistent_corrections.append(np.asarray(persistent, dtype=np.float64))
        local_corrections.append(np.asarray(local, dtype=np.float64))
        defects.append(np.asarray(defect, dtype=np.float64))
        truth_increments.append(np.asarray(truth_increment, dtype=np.float64))
        if not accepted:
            break
        current = prediction
        valid_length = call

    shape = (case.nodes.shape[0], len(parent.COMPONENTS))
    state_array = np.asarray(states, dtype=np.float64)
    base_array = _sequence_array(base_defects, shape)
    correction_array = _sequence_array(total_corrections, shape)
    persistent_array = _sequence_array(persistent_corrections, shape)
    local_array = _sequence_array(local_corrections, shape)
    defect_array = _sequence_array(defects, shape)
    truth_array = _sequence_array(truth_increments, shape)
    complete = bool(valid_length == calls)
    if defect_array.shape[0] < 1:
        summary: dict[str, Any] = {
            "complete": False,
            "valid_length": valid_length,
            "final_state_error": None,
            "residual_rms": None,
            "controls": {},
        }
    else:
        summary = parent._summarize_rollout(
            case,
            states=state_array,
            defects=defect_array,
            complete=complete,
            shock_quantile=shock_quantile,
        )
        summary["valid_length"] = valid_length
    candidate = _arm_candidate(arm, selected)
    summary["correction_integral_audit"] = parent._correction_integral_audit(
        correction_array, case, candidate
    )
    result = parent.RolloutResult(
        candidate=candidate,
        complete=complete,
        valid_length=valid_length,
        states=state_array,
        base_defects=base_array,
        corrections=correction_array,
        defects=defect_array,
        truth_increments=truth_array,
        admissibility_rows=admissibility_rows,
        summary=summary,
    )
    return LocalRolloutPayload(
        result=result,
        persistent_corrections=persistent_array,
        local_corrections=local_array,
        local_audit_rows=local_audit_rows,
    )


def _prefix_replay_row(
    case_id: str,
    arm: str,
    prefix: parent.RolloutResult,
    full: parent.RolloutResult,
    *,
    probe_calls: int,
) -> dict[str, Any]:
    if not prefix.complete or not full.complete:
        return {
            "case_id": case_id,
            "arm": arm,
            "probe_calls": probe_calls,
            "maximum_absolute": None,
            "relative_l2": None,
            "exact_equal": False,
            "passed": False,
        }
    left = prefix.states[: probe_calls + 1]
    right = full.states[: probe_calls + 1]
    maximum_absolute = float(np.max(np.abs(left - right)))
    relative = _relative_l2(left, right)
    return {
        "case_id": case_id,
        "arm": arm,
        "probe_calls": probe_calls,
        "maximum_absolute": maximum_absolute,
        "relative_l2": relative,
        "exact_equal": bool(np.array_equal(left, right)),
        "absolute_limit": d076.PROBE_ABSOLUTE_REPLAY_LIMIT,
        "relative_limit": d076.PROBE_RELATIVE_REPLAY_LIMIT,
        "passed": bool(
            maximum_absolute <= d076.PROBE_ABSOLUTE_REPLAY_LIMIT
            and relative <= d076.PROBE_RELATIVE_REPLAY_LIMIT
        ),
    }


def _output_template() -> dict[str, list[dict[str, Any]]]:
    return {
        "evaluation_case_summary.csv": [],
        "evaluation_comparisons.csv": [],
        "evaluation_probe_replay.csv": [],
        "evaluation_call_metrics.csv": [],
        "projection_metrics.csv": [],
        "projection_component_metrics.csv": [],
        "projection_status.csv": [],
        "signed_component_budgets.csv": [],
        "sequence_summaries.csv": [],
        "sequence_time_metrics.csv": [],
        "lag_correlations.csv": [],
        "pod_summaries.csv": [],
        "completion.csv": [],
        "correction_integral_audit.csv": [],
        "local_correction_audit.csv": [],
        "visual_payload_inventory.csv": [],
    }


def _materialized_csv_rows(
    outputs: Mapping[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """Omit intentionally inapplicable empty tables from the strict CSV writer."""

    return {name: rows for name, rows in outputs.items() if rows}


def _save_visual_payload(
    path: Path,
    case: CaseData,
    payloads: Mapping[str, LocalRolloutPayload],
    *,
    visual_arms: Sequence[str] = VISUAL_ARMS,
    schema: str = SCHEMA,
) -> dict[str, Any]:
    arrays: dict[str, np.ndarray] = {
        "schema": np.asarray(schema),
        "case_id": np.asarray(case.case_id),
        "resolution": np.asarray(case.resolution_name),
        "nodes": case.nodes.astype(np.float64),
        "weights": case.weights.astype(np.float64),
        "node_type": case.physical_node_type.astype(np.int64),
        "physical_times": case.physical_times.astype(np.float64),
        "state_scale": case.state_scale.astype(np.float64),
        "residual_scale": case.residual_scale.astype(np.float64),
        "reference_states": case.reference_states.astype(np.float32),
        "true_increment": np.diff(case.reference_states, axis=0).astype(np.float32),
    }
    maximum_cumulative_closure = 0.0
    for arm in visual_arms:
        payload = payloads[arm]
        result = payload.result
        if not result.complete:
            raise ValueError("D080-B visual payload requires complete rollout arms")
        cumulative = np.cumsum(result.defects, axis=0)
        state_error = result.states[1:] - case.reference_states[1:]
        maximum_cumulative_closure = max(
            maximum_cumulative_closure,
            float(np.max(np.abs(cumulative - state_error))),
        )
        arrays[f"states__{arm}"] = result.states.astype(np.float32)
        arrays[f"base_increment__{arm}"] = (
            result.base_defects + result.truth_increments
        ).astype(np.float32)
        arrays[f"persistent_correction__{arm}"] = payload.persistent_corrections.astype(
            np.float32
        )
        arrays[f"local_correction__{arm}"] = payload.local_corrections.astype(
            np.float32
        )
        arrays[f"defect__{arm}"] = result.defects.astype(np.float32)
        arrays[f"cumulative_defect__{arm}"] = cumulative.astype(np.float32)
    np.savez_compressed(path, **arrays)
    return {
        "family": case.family,
        "case_id": case.case_id,
        "resolution": case.resolution_name,
        "relative_path": str(path),
        "arm_count": len(visual_arms),
        "calls": int(case.reference_states.shape[0] - 1),
        "maximum_cumulative_closure": maximum_cumulative_closure,
    }


def _promotion(
    comparisons: Sequence[Mapping[str, Any]],
    case_summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    summary_by_key = {
        (str(row["case_id"]), str(row["arm"])): row for row in case_summaries
    }
    expected_case_count = len({str(row["case_id"]) for row in case_summaries})
    combined_specs = [arm for arm in ARM_SPECS if arm.name.startswith("combined_")]
    candidate_rows = []
    for arm in combined_specs:
        rows = [
            row
            for row in comparisons
            if row["comparison"] == f"{arm.name}_vs_persistent"
        ]
        if len(rows) != expected_case_count:
            raise ValueError(f"D080-B comparison inventory is incomplete: {arm.name}")
        endpoint = np.asarray(
            [float(row["endpoint_state_ratio"]) for row in rows], dtype=np.float64
        )
        residual = np.asarray(
            [float(row["residual_rms_ratio"]) for row in rows], dtype=np.float64
        )
        cumulative = []
        region = []
        for row in rows:
            case_id = str(row["case_id"])
            numerator = summary_by_key[(case_id, arm.name)]
            denominator = summary_by_key[(case_id, "persistent")]
            cumulative.append(
                parent._required_ratio(
                    float(numerator["net_defect_rms"]),
                    float(denominator["net_defect_rms"]),
                )
            )
            controls = json.loads(str(row["control_ratios_json"]))
            region_key = (
                "endpoint_state__vortex"
                if arm.pathway == "vortex_isotropic"
                else "endpoint_state__shock"
            )
            region.append(float(controls[region_key]))
        cumulative_array = np.asarray(cumulative, dtype=np.float64)
        region_array = np.asarray(region, dtype=np.float64)
        maximum_control = max(float(row["maximum_control_ratio"]) for row in rows)
        gates = {
            "median_endpoint_improves": float(np.median(endpoint)) < 1.0,
            "median_residual_improves": float(np.median(residual)) < 1.0,
            "median_cumulative_defect_improves": (
                float(np.median(cumulative_array)) < 1.0
            ),
            "endpoint_nonworse_four_of_six": int(np.count_nonzero(endpoint <= 1.0))
            >= 4,
            "maximum_endpoint_within_limit": float(endpoint.max())
            <= ENDPOINT_WORST_LIMIT,
            "maximum_control_within_limit": maximum_control <= CONTROL_WORST_LIMIT,
            "local_region_median_nonworse": float(np.median(region_array)) <= 1.0,
        }
        candidate_rows.append(
            {
                "arm": arm.name,
                "pathway": arm.pathway,
                "norm_cap": arm.cap,
                "median_endpoint_ratio": float(np.median(endpoint)),
                "median_residual_ratio": float(np.median(residual)),
                "median_cumulative_defect_ratio": float(np.median(cumulative_array)),
                "median_local_region_ratio": float(np.median(region_array)),
                "maximum_endpoint_ratio": float(endpoint.max()),
                "maximum_control_ratio": maximum_control,
                "endpoint_nonworse_count": int(np.count_nonzero(endpoint <= 1.0)),
                "gates": gates,
                "efficacy_passed": bool(all(gates.values())),
            }
        )
    passing = [row for row in candidate_rows if row["efficacy_passed"]]
    passing.sort(
        key=lambda row: (
            row["median_endpoint_ratio"],
            row["median_residual_ratio"],
            row["arm"],
        )
    )
    return {
        "adaptive_open_population": True,
        "d080a_summary_sha256": D080A_SUMMARY_SHA256,
        "candidate_rows": candidate_rows,
        "selected_arm": None if not passing else passing[0]["arm"],
        "efficacy_passed": bool(passing),
        "passed": bool(passing),
    }


def _maximum(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [
        abs(float(row[key]))
        for row in rows
        if row.get(key) is not None and np.isfinite(float(row[key]))
    ]
    return 0.0 if not values else max(values)


def _evaluate_local_arms(
    args: Any,
    *,
    model: torch.nn.Module,
    device: torch.device,
    probe: Mapping[str, Any],
    arm_specs: Sequence[LocalArmSpec] = ARM_SPECS,
    visual_arms: Sequence[str] = VISUAL_ARMS,
    schema: str = SCHEMA,
    promotion_runner: Callable[
        [Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]],
        dict[str, Any],
    ] = _promotion,
) -> dict[str, Any]:
    outputs = _output_template()
    all_complete = True
    visual_ids = (
        set(args.visualization_cases)
        if args.visualization_cases
        else {
            probe["cases"][0].case_id,
            probe["cases"][len(probe["cases"]) // 2].case_id,
            probe["cases"][-1].case_id,
        }
    )
    if not visual_ids <= {case.case_id for case in probe["cases"]}:
        raise ValueError("D080-B visualization case lies outside evaluation")

    for index, case in enumerate(probe["cases"], start=1):
        selected = probe["selected_by_case"][case.case_id]
        bias = probe["bias_by_case"][case.case_id]
        payloads: dict[str, LocalRolloutPayload] = {}
        for arm in arm_specs:
            payloads[arm.name] = _rollout_arm(
                model,
                case,
                selected,
                bias_sequence=bias,
                arm=arm,
                shock_quantile=args.shock_quantile,
                calls=args.rollout_calls,
            )
        all_complete = bool(
            all_complete
            and all(payload.result.complete for payload in payloads.values())
        )

        for arm in arm_specs:
            payload = payloads[arm.name]
            result = payload.result
            d076._append_arm_diagnostics(
                outputs,
                case,
                arm=arm.name,
                result=result,
                basis=None,
                applied_coefficients=None,
            )
            outputs["projection_status.csv"].append(
                {
                    "family": case.family,
                    "case_id": case.case_id,
                    "resolution": case.resolution_name,
                    "arm": arm.name,
                    "status": "not_applied_mixed_local_correction",
                }
            )
            outputs["local_correction_audit.csv"].extend(payload.local_audit_rows)
            prefix = _rollout_arm(
                model,
                case,
                selected,
                bias_sequence=bias,
                arm=arm,
                shock_quantile=args.shock_quantile,
                calls=args.probe_calls,
            )
            outputs["evaluation_probe_replay.csv"].append(
                _prefix_replay_row(
                    case.case_id,
                    arm.name,
                    prefix.result,
                    result,
                    probe_calls=args.probe_calls,
                )
            )

        zero = payloads["zero"].result
        persistent = payloads["persistent"].result
        for arm in arm_specs[1:]:
            outputs["evaluation_comparisons.csv"].append(
                d076._comparison_row(
                    case.case_id,
                    payloads[arm.name].result,
                    zero,
                    comparison=f"{arm.name}_vs_zero",
                )
            )
        for arm in arm_specs[2:]:
            outputs["evaluation_comparisons.csv"].append(
                d076._comparison_row(
                    case.case_id,
                    payloads[arm.name].result,
                    persistent,
                    comparison=f"{arm.name}_vs_persistent",
                )
            )

        if case.case_id in visual_ids:
            visual_path = (
                args.output_dir
                / "visual_payloads"
                / f"dynamic_fv_{case.case_id}_{case.resolution_name}.npz"
            )
            visual = _save_visual_payload(
                visual_path,
                case,
                payloads,
                visual_arms=visual_arms,
                schema=schema,
            )
            visual["relative_path"] = str(
                visual_path.relative_to(args.output_dir)
            ).replace("\\", "/")
            visual["sha256"] = d076.sha256_file(visual_path)
            outputs["visual_payload_inventory.csv"].append(visual)
        print(
            f"D080-B evaluated {index}/{len(probe['cases'])} "
            f"{case.case_id} {case.resolution_name}",
            flush=True,
        )
        del payloads
        if device.type == "cuda":
            torch.cuda.empty_cache()

    case_count = len(probe["cases"])
    arm_count = len(arm_specs)
    calls = args.rollout_calls
    expected_comparisons = case_count * ((arm_count - 1) + (arm_count - 2))
    inventory = {
        "case_summary": len(outputs["evaluation_case_summary.csv"])
        == case_count * arm_count,
        "call_metrics": len(outputs["evaluation_call_metrics.csv"])
        == case_count * arm_count * calls,
        "completion": len(outputs["completion.csv"]) == case_count * arm_count * calls,
        "local_audit": len(outputs["local_correction_audit.csv"])
        == case_count * arm_count * calls,
        "prefix_replay": len(outputs["evaluation_probe_replay.csv"])
        == case_count * arm_count,
        "comparisons": len(outputs["evaluation_comparisons.csv"])
        == expected_comparisons,
        "visual_payloads": len(outputs["visual_payload_inventory.csv"])
        == len(visual_ids),
    }
    inventory_pass = bool(all(inventory.values()))
    completion_pass = bool(
        all(bool(row["accepted"]) for row in outputs["completion.csv"])
    )
    prefix_pass = bool(
        all(bool(row["passed"]) for row in outputs["evaluation_probe_replay.csv"])
    )
    local_rows = outputs["local_correction_audit.csv"]
    maximum_local_mean = _maximum(local_rows, "maximum_local_mean_closure_scaled")
    maximum_cap_excess = _maximum(local_rows, "cap_excess")
    maximum_support = _maximum(local_rows, "maximum_non_type0_local_correction")
    call_rows = outputs["evaluation_call_metrics.csv"]
    maximum_recurrence = _maximum(call_rows, "recurrence_closure_rms")
    maximum_growth = _maximum(call_rows, "growth_closure")
    maximum_same_input_energy = _maximum(call_rows, "same_input_energy_closure")
    maximum_same_input_pointwise = _maximum(
        call_rows, "same_input_pointwise_closure_max"
    )
    visual_rows = outputs["visual_payload_inventory.csv"]
    visual_hash_pass = bool(
        all(
            d076.sha256_file(args.output_dir / str(row["relative_path"]))
            == str(row["sha256"])
            for row in visual_rows
        )
    )
    maximum_visual_cumulative = _maximum(visual_rows, "maximum_cumulative_closure")
    closure_pass = bool(
        maximum_local_mean <= LOCAL_MEAN_LIMIT
        and maximum_cap_excess <= LOCAL_CAP_LIMIT
        and maximum_support <= CONTACT_SUPPORT_LIMIT
        and maximum_recurrence <= d076.PROBE_ABSOLUTE_REPLAY_LIMIT
        and maximum_growth <= 2.0e-5
        and maximum_same_input_energy <= 1.0e-10
        and maximum_same_input_pointwise <= 1.0e-12
        and maximum_visual_cumulative <= 2.0e-5
        and visual_hash_pass
    )
    promotion = promotion_runner(
        outputs["evaluation_comparisons.csv"],
        outputs["evaluation_case_summary.csv"],
    )
    contract_pass = bool(
        inventory_pass and completion_pass and prefix_pass and closure_pass
    )
    promotion["contract_passed"] = contract_pass
    promotion["passed"] = bool(
        promotion["efficacy_passed"] and contract_pass and not args.smoke
    )
    if args.smoke:
        promotion["status"] = "non_scientific_smoke"
        promotion["selected_arm"] = None
    return {
        "outputs": outputs,
        "promotion": promotion,
        "checks": {
            "inventory": inventory,
            "inventory_pass": inventory_pass,
            "all_evaluation_complete": all_complete,
            "completion_pass": completion_pass,
            "prefix_replay_pass": prefix_pass,
            "closure_pass": closure_pass,
            "maximum_local_mean_closure_scaled": maximum_local_mean,
            "maximum_local_cap_excess": maximum_cap_excess,
            "maximum_non_type0_local_correction": maximum_support,
            "maximum_recurrence_closure_rms": maximum_recurrence,
            "maximum_growth_closure_absolute": maximum_growth,
            "maximum_same_input_energy_closure": maximum_same_input_energy,
            "maximum_same_input_pointwise_closure": maximum_same_input_pointwise,
            "maximum_visual_cumulative_closure": maximum_visual_cumulative,
            "visual_payload_hash_pass": visual_hash_pass,
            "maximum_probe_prefix_absolute": _maximum(
                outputs["evaluation_probe_replay.csv"], "maximum_absolute"
            ),
            "maximum_probe_prefix_relative_l2": _maximum(
                outputs["evaluation_probe_replay.csv"], "relative_l2"
            ),
        },
    }


def _conditional_evaluation(
    args: Any,
    *,
    spec: d076.ResponseExperimentSpec,
    model: torch.nn.Module,
    checkpoint: Mapping[str, Any],
    store: Any,
    device: torch.device,
    evaluation_ids: Sequence[str],
    candidates: Sequence[parent.Candidate],
    frozen_coefficients: np.ndarray,
    frozen_policy: Mapping[str, Any],
    response_table_rows: Sequence[Mapping[str, Any]],
    selector_sha: str,
    phase_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    probe = d076._freeze_evaluation_probe_choices(
        args,
        spec=spec,
        model=model,
        checkpoint=checkpoint,
        store=store,
        device=device,
        evaluation_ids=evaluation_ids,
        candidates=candidates,
        frozen_coefficients=frozen_coefficients,
        frozen_policy=frozen_policy,
        response_table_rows=response_table_rows,
        selector_sha=selector_sha,
        phase_rows=phase_rows,
    )
    if d076.sha256_file(probe["record_path"]) != probe["record_sha"] or any(
        d076.sha256_file(args.output_dir / name) != digest
        for name, digest in probe["artifact_sha256"].items()
    ):
        raise ValueError("D080-B evaluation choice bundle changed before rollout")
    evaluated = _evaluate_local_arms(
        args,
        model=model,
        device=device,
        probe=probe,
    )
    if d076.sha256_file(probe["record_path"]) != probe["record_sha"] or any(
        d076.sha256_file(args.output_dir / name) != digest
        for name, digest in probe["artifact_sha256"].items()
    ):
        raise ValueError("D080-B evaluation choice bundle changed during rollout")
    output_rows = {
        **probe["output_rows"],
        **_materialized_csv_rows(evaluated["outputs"]),
        "evaluation_reference_checks.csv": probe["reference_checks"],
    }
    checks = {
        "evaluation_choice_bundle_revalidated": True,
        "evaluation_inventory_pass": evaluated["checks"]["inventory_pass"],
        "evaluation_inventory_details": evaluated["checks"]["inventory"],
        "all_evaluation_complete": bool(
            evaluated["checks"]["all_evaluation_complete"]
            and evaluated["checks"]["completion_pass"]
        ),
        "probe_prefix_replay_pass": evaluated["checks"]["prefix_replay_pass"],
        **evaluated["checks"],
    }
    return {
        "hook": probe["hook"],
        "reference_checks": probe["reference_checks"],
        "case_contract_rows": probe["case_contract_rows"],
        "output_rows": output_rows,
        "promotion": evaluated["promotion"],
        "choice_bundle": {
            "record_path": probe["record_path"],
            "record_sha": probe["record_sha"],
            "artifact_sha256": probe["artifact_sha256"],
        },
        "checks": checks,
    }


D080_SPEC = replace(
    d078.D078_SPEC,
    experiment_id="D080",
    schema=SCHEMA,
    experiment_contract=EXPERIMENT_CONTRACT,
    description=__doc__ or "D080-B local correction pilot",
    method_claim=(
        "D078 persistent low-rank controller plus three D080-A-selected causal "
        "local graph-dissipation directions; adaptive open-validation pilot; "
        f"D080-A summary SHA-256 {D080A_SUMMARY_SHA256}; not data assimilation"
    ),
    extra_source_paths=(*d078.D078_SPEC.extra_source_paths, Path(__file__)),
    evaluation_runner=_conditional_evaluation,
)


def parse_args(argv: Sequence[str] | None = None):
    return d076.parse_args_for_experiment(argv, D080_SPEC)


def run(args: Any):
    return d076.run(args, spec=D080_SPEC)


def main(argv: Sequence[str] | None = None) -> int:
    summary = run(parse_args(argv))
    print(
        f"D080-B dynamic_fv status={summary['status']} "
        f"calibration_qualified={summary['calibration_qualification']['passed']} "
        f"promotion={summary['promotion'].get('passed')}",
        flush=True,
    )
    return 0 if summary["contract_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
