#!/usr/bin/env python3
"""Test realizability of a bounded correction on D055's legal support.

D056 is a truth-informed, zero-training upper bound.  D055 fixes support using
only the current state and frozen PCNO proposal.  Truth then supplies a desired
state correction inside that support; the correction is projected to zero
physical-volume integral, capped at ten percent of the frozen update norm, and
line-searched under raw admissibility and anti-smearing constraints.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no.decompose_pcno_euler2d_rollout_error import (  # noqa: E402
    _artifact_scalar,
    _reference_masks,
    _source_records,
    _state_tensor,
)
from scripts.time_dependent_no.diagnose_pcno_euler2d_fresh_defect_locality import (  # noqa: E402
    SHOCK_QUANTILE,
    build_graph_adjacency,
    graph_hop_distance,
)
from scripts.time_dependent_no.diagnose_pcno_euler2d_proposal_sensor import (  # noqa: E402
    CURRENT_SHOCK_EXCLUSION_HOPS,
    EXPECTED_CALLS,
    SENSOR_ELIGIBLE_NODE_FRACTION,
    _top_score_mask,
    proposal_sensor_metrics,
)
from scripts.time_dependent_no.diagnose_pcno_euler2d_ripples import (  # noqa: E402
    append_jsonl,
    build_model,
    load_checkpoint,
    model_call,
    select_device,
    sha256_file,
    write_json,
)
from scripts.time_dependent_no.evaluate_pcno_shock_vortex_correction_oracle import (  # noqa: E402
    _expected_vortex_center,
    _highpass_reduction,
    metric_acceptance,
    metric_report_passed,
)
from utility.time_dependent_no.conservative_correction_oracle import (  # noqa: E402
    euler2d_state_is_admissible,
)
from utility.time_dependent_no.pcno_euler2d import (  # noqa: E402
    PCNOEuler2DShardStore,
    reference_smooth_region_mask,
)
from utility.time_dependent_no.pcno_ripple_diagnostics import (  # noqa: E402
    RIPPLE_DIAGNOSTIC_SCHEMA,
    node_highpass_field,
    normalized_node_weights,
    raw_admissibility_summary,
    weighted_relative_l2_numpy,
)
from utility.time_dependent_no.shock_vortex_family import (  # noqa: E402
    family_case_by_id,
    load_shock_vortex_family_manifest,
)
from utility.time_dependent_no.shock_vortex_metrics import (  # noqa: E402
    endpoint_metrics,
)


REALIZABILITY_SCHEMA = "pcno_euler2d_proposal_correction_oracle_d056_v1"
EXPECTED_TRAJECTORY_COUNT = 6
LATE_CALLS = (30, 60)
REQUIRED_CASES = 5
MAX_SUPPORT_FRACTION = 0.20
MAX_UPDATE_RATIO = 0.10
BALANCE_MAX_ABSOLUTE = 1.0e-10
MIN_MEDIAN_STATE_REDUCTION = 0.15
MIN_MEDIAN_SMOOTH_HIGHPASS_REDUCTION = 0.20
SOURCE_REPLAY_RELATIVE_TOLERANCE = 2.0e-4
LINE_SEARCH_POINTS = 65


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--d055-dir", type=Path, required=True)
    parser.add_argument("--family-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")


def volume_weighted_scaled_norm(
    field: np.ndarray,
    volumes: np.ndarray,
    component_scale: np.ndarray,
) -> float:
    values = np.asarray(field, dtype=np.float64)
    volume = np.asarray(volumes, dtype=np.float64).reshape(-1)
    scale = np.asarray(component_scale, dtype=np.float64).reshape(-1)
    if values.ndim != 2 or values.shape != (volume.size, scale.size):
        raise ValueError("field, volumes, and component_scale have incompatible shapes")
    mass = normalized_node_weights(volume, name="D056 update norm")
    return float(np.sqrt(np.sum(mass[:, None] * (values / scale[None, :]) ** 2)))


def volume_balanced_support_direction(
    prediction: np.ndarray,
    target: np.ndarray,
    support_mask: np.ndarray,
    volumes: np.ndarray,
) -> np.ndarray:
    """Project the desired supported correction to zero volume integral."""

    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    support = np.asarray(support_mask, dtype=bool)
    volume = np.asarray(volumes, dtype=np.float64).reshape(-1)
    if pred.ndim != 2 or truth.shape != pred.shape or pred.shape[0] != volume.size:
        raise ValueError("prediction, target, and volumes have incompatible shapes")
    if support.shape != (pred.shape[0],) or not np.any(support):
        raise ValueError("support_mask must select at least one state row")
    support_volume = float(np.sum(volume[support]))
    if support_volume <= 0.0:
        raise ValueError("support volume must be positive")
    desired = truth[support] - pred[support]
    mean = np.sum(volume[support, None] * desired, axis=0) / support_volume
    direction = np.zeros_like(pred)
    direction[support] = desired - mean[None, :]
    return direction


def bounded_realizability_oracle(
    prediction: np.ndarray,
    target: np.ndarray,
    current: np.ndarray,
    support_mask: np.ndarray,
    *,
    positions: np.ndarray,
    edges: np.ndarray,
    volumes: np.ndarray,
    component_scale: np.ndarray,
    gamma: float,
    vortex_center: tuple[float, float],
) -> tuple[dict[str, Any], np.ndarray]:
    """Return the best fixed-line-search feasible balanced correction."""

    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    first = np.asarray(current, dtype=np.float64)
    volume = np.asarray(volumes, dtype=np.float64).reshape(-1)
    scale = np.asarray(component_scale, dtype=np.float64).reshape(-1)
    support = np.asarray(support_mask, dtype=bool)
    if not all(
        euler2d_state_is_admissible(state, gamma=gamma)
        for state in (pred, truth, first)
    ):
        raise ValueError("D056 inputs must all be raw-admissible")

    baseline_metrics = endpoint_metrics(
        pred,
        truth,
        positions=positions,
        edges=edges,
        volumes=volume,
        component_scale=scale,
        gamma=gamma,
        shock_quantile=SHOCK_QUANTILE,
        vortex_center=vortex_center,
    )
    if not metric_report_passed(metric_acceptance(baseline_metrics, baseline_metrics)):
        raise ValueError("D056 anti-smearing metrics are unavailable")
    baseline_error = weighted_relative_l2_numpy(pred, truth, volume, scale)
    baseline_highpass = float(baseline_metrics["smooth_region_graph_highpass_energy"])

    direction = volume_balanced_support_direction(pred, truth, support, volume)
    update_norm = volume_weighted_scaled_norm(pred - first, volume, scale)
    direction_norm = volume_weighted_scaled_norm(direction, volume, scale)
    candidate_ratio = direction_norm / max(update_norm, 1.0e-30)
    cap_scale = min(1.0, MAX_UPDATE_RATIO / max(candidate_ratio, 1.0e-30))
    bounded_direction = cap_scale * direction
    bounded_norm = volume_weighted_scaled_norm(bounded_direction, volume, scale)

    best_state = pred.copy()
    best_metrics = baseline_metrics
    best_error = baseline_error
    best_alpha = 0.0
    feasible = 0
    for alpha in np.linspace(0.0, 1.0, LINE_SEARCH_POINTS):
        candidate = pred + float(alpha) * bounded_direction
        if not euler2d_state_is_admissible(candidate, gamma=gamma):
            continue
        candidate_metrics = endpoint_metrics(
            candidate,
            truth,
            positions=positions,
            edges=edges,
            volumes=volume,
            component_scale=scale,
            gamma=gamma,
            shock_quantile=SHOCK_QUANTILE,
            vortex_center=vortex_center,
        )
        if not metric_report_passed(metric_acceptance(baseline_metrics, candidate_metrics)):
            continue
        feasible += 1
        candidate_error = weighted_relative_l2_numpy(
            candidate, truth, volume, scale
        )
        if candidate_error < best_error - 1.0e-15:
            best_state = candidate
            best_metrics = candidate_metrics
            best_error = candidate_error
            best_alpha = float(alpha)

    correction = best_state - pred
    balance = np.sum(volume[:, None] * correction, axis=0)
    corrected_highpass = float(
        best_metrics["smooth_region_graph_highpass_energy"]
    )
    state_reduction = (baseline_error - best_error) / max(baseline_error, 1.0e-30)
    smooth_reduction = _highpass_reduction(baseline_highpass, corrected_highpass)
    applied_norm = volume_weighted_scaled_norm(correction, volume, scale)
    gate_report = metric_acceptance(baseline_metrics, best_metrics)
    return (
        {
            "selected_alpha": best_alpha,
            "feasible_line_search_points": feasible,
            "baseline_state_error": baseline_error,
            "corrected_state_error": best_error,
            "state_error_reduction": state_reduction,
            "baseline_smooth_highpass_energy": baseline_highpass,
            "corrected_smooth_highpass_energy": corrected_highpass,
            "smooth_highpass_error_reduction": smooth_reduction,
            "candidate_update_ratio": candidate_ratio,
            "bounded_direction_update_ratio": bounded_norm
            / max(update_norm, 1.0e-30),
            "applied_update_ratio": applied_norm / max(update_norm, 1.0e-30),
            "support_fraction_all_nodes": float(np.mean(support)),
            "volume_integral_correction": balance.tolist(),
            "volume_integral_correction_max_abs": float(
                np.max(np.abs(balance), initial=0.0)
            ),
            "raw_admissible": euler2d_state_is_admissible(best_state, gamma=gamma),
            "anti_smearing_pass": metric_report_passed(gate_report),
            "anti_smearing": gate_report,
        },
        best_state,
    )


def _median(values: Sequence[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return None if not finite else float(np.median(finite))


def realizability_selector(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the frozen D056 structural and late-call promotion gates."""

    row_list = [dict(row) for row in rows]
    keys = [
        (str(row.get("trajectory")), int(row.get("call_index", -1)))
        for row in row_list
    ]
    duplicate_keys = sorted({key for key in keys if keys.count(key) > 1})
    trajectories = sorted({trajectory for trajectory, _ in keys})
    calls_by_trajectory = {
        trajectory: sorted(call for key, call in keys if key == trajectory)
        for trajectory in trajectories
    }
    replay_errors = [
        max(float(value) for value in row.get("d055_replay", {}).values())
        for row in row_list
    ]
    structural = all(
        row.get("oracle", {}).get("raw_admissible") is True
        and row.get("oracle", {}).get("anti_smearing_pass") is True
        and float(row.get("support_fraction_interior", math.inf))
        <= MAX_SUPPORT_FRACTION + 1.0e-12
        and float(row.get("oracle", {}).get("applied_update_ratio", math.inf))
        <= MAX_UPDATE_RATIO + 1.0e-12
        and float(
            row.get("oracle", {}).get(
                "volume_integral_correction_max_abs", math.inf
            )
        )
        <= BALANCE_MAX_ABSOLUTE
        for row in row_list
    )
    checks = {
        "schema": bool(row_list)
        and all(row.get("schema") == REALIZABILITY_SCHEMA for row in row_list),
        "exact_six_by_four_rows": (
            len(row_list) == EXPECTED_TRAJECTORY_COUNT * len(EXPECTED_CALLS)
            and len(trajectories) == EXPECTED_TRAJECTORY_COUNT
            and not duplicate_keys
            and all(calls == list(EXPECTED_CALLS) for calls in calls_by_trajectory.values())
        ),
        "d055_support_replay": bool(replay_errors)
        and max(replay_errors) <= SOURCE_REPLAY_RELATIVE_TOLERANCE,
        "all_rows_bounded_balanced_admissible_and_anti_smearing": structural,
    }
    contract_complete = all(checks.values())

    late: dict[str, Any] = {}
    promoted = True
    for call in LATE_CALLS:
        selected = [row for row in row_list if int(row.get("call_index", -1)) == call]
        state = [float(row["oracle"]["state_error_reduction"]) for row in selected]
        smooth = [
            float(row["oracle"]["smooth_highpass_error_reduction"])
            for row in selected
        ]
        joint_nonworse = sum(
            state_value >= -1.0e-12 and smooth_value >= -1.0e-12
            for state_value, smooth_value in zip(state, smooth, strict=True)
        )
        median_state = _median(state)
        median_smooth = _median(smooth)
        call_pass = (
            len(selected) == EXPECTED_TRAJECTORY_COUNT
            and median_state is not None
            and median_state >= MIN_MEDIAN_STATE_REDUCTION
            and median_smooth is not None
            and median_smooth >= MIN_MEDIAN_SMOOTH_HIGHPASS_REDUCTION
            and joint_nonworse >= REQUIRED_CASES
        )
        promoted &= call_pass
        late[str(call)] = {
            "row_count": len(selected),
            "median_state_error_reduction": median_state,
            "median_smooth_highpass_error_reduction": median_smooth,
            "joint_nonworse_case_count": joint_nonworse,
            "passed": call_pass,
        }

    if not contract_complete:
        classification = "incomplete_contract"
        route = "stop_without_training"
    elif promoted:
        classification = "bounded_causal_support_correction_realizable"
        route = "authorize_one_frozen_global_tiny_detail_fit_implementation"
    else:
        classification = "bounded_causal_support_correction_insufficient"
        route = "reject_learned_detail_route_no_training"
    return {
        "version": "proposal_correction_realizability_selector_d056_v1",
        "contract_complete": contract_complete,
        "contract_checks": checks,
        "duplicate_keys": [list(key) for key in duplicate_keys],
        "row_count": len(row_list),
        "trajectory_count": len(trajectories),
        "late_calls": late,
        "promotion_passed": promoted,
        "classification": classification,
        "route": route,
        "thresholds": {
            "support_fraction_interior_max": MAX_SUPPORT_FRACTION,
            "update_ratio_max": MAX_UPDATE_RATIO,
            "volume_integral_correction_max_abs": BALANCE_MAX_ABSOLUTE,
            "median_state_error_reduction_min": MIN_MEDIAN_STATE_REDUCTION,
            "median_smooth_highpass_error_reduction_min": (
                MIN_MEDIAN_SMOOTH_HIGHPASS_REDUCTION
            ),
            "joint_nonworse_cases_min": REQUIRED_CASES,
            "line_search_points": LINE_SEARCH_POINTS,
            "source_replay_relative_tolerance": SOURCE_REPLAY_RELATIVE_TOLERANCE,
        },
        "claim_boundary": (
            "truth-informed state-space realizability on a legally selected support; "
            "not learnability, rollout improvement, face flux, or local conservation"
        ),
    }


def _load_rows(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row["trajectory"]), int(row["call_index"]))
        if key in result:
            raise ValueError(f"duplicate D055 row: {key}")
        result[key] = row
    return result


def _relative_error(value: float, reference: Any) -> float:
    expected = float(reference)
    return abs(float(value) - expected) / max(abs(expected), 1.0e-12)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_args(args)
    device = select_device(args.device)

    source_summary_path = args.source_dir / "summary.json"
    d055_summary_path = args.d055_dir / "summary.json"
    if not source_summary_path.is_file() or not d055_summary_path.is_file():
        raise FileNotFoundError("D052 and D055 summaries are required")
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    d055_summary = json.loads(d055_summary_path.read_text(encoding="utf-8"))
    if source_summary.get("status") != "complete":
        raise ValueError("D056 requires completed D052")
    if (
        d055_summary.get("status") != "complete"
        or d055_summary.get("selector", {}).get("contract_complete") is not True
        or d055_summary.get("selector", {}).get("classification")
        != "proposal_self_sensor_candidate"
    ):
        raise ValueError("D056 requires the positive D055 selector")
    source_digest = sha256_file(source_summary_path)
    if d055_summary.get("source", {}).get("d052_summary_sha256") != source_digest:
        raise ValueError("D052 source differs from the D055 binding")
    d055_rows_path = args.d055_dir / "proposal_sensor.jsonl"
    d055_rows = _load_rows(d055_rows_path)
    records = _source_records(source_summary)
    family = load_shock_vortex_family_manifest(args.family_manifest)
    reference_config = family["reference_config"]

    checkpoint = load_checkpoint(args.checkpoint)
    checkpoint_digest = sha256_file(args.checkpoint)
    if d055_summary.get("checkpoint", {}).get("sha256") != checkpoint_digest:
        raise ValueError("checkpoint SHA-256 differs from D055")
    if checkpoint.get("boundary_mode") != "model_all_nodes" or not bool(
        checkpoint.get("raw_recurrence")
    ):
        raise ValueError("D056 requires legal model-all-node raw recurrence")
    store = PCNOEuler2DShardStore(args.data_dir)
    if store.manifest_digest != checkpoint.get("data_manifest_digest"):
        raise ValueError("checkpoint and shard manifest digests differ")
    if store.manifest.get("weight_provenance") != "validated_physical_cell_volume_normalized":
        raise ValueError("D056 requires validated physical cell-volume weights")
    data_manifest_digest = store.manifest_digest
    model = build_model(checkpoint, device)
    component_scale = model.state_scale.detach().cpu().numpy().astype(np.float64).reshape(-1)

    args.output_dir.mkdir(parents=True)
    rows_path = args.output_dir / "realizability.jsonl"
    rows: list[dict[str, Any]] = []
    try:
        for record in records:
            key = str(record["trajectory"])
            case = family_case_by_id(family, key)
            if case["split"] != "validation":
                raise ValueError(f"D056 case {key} is not validation")
            if key not in {str(value) for value in checkpoint.get("val_keys", [])}:
                raise ValueError(f"source trajectory {key} is not a checkpoint validation case")
            artifact_path = args.source_dir / str(record["artifact"])
            if not artifact_path.is_file():
                raise FileNotFoundError(artifact_path)
            with np.load(artifact_path, allow_pickle=False) as artifact:
                if _artifact_scalar(artifact, "schema") != RIPPLE_DIAGNOSTIC_SCHEMA:
                    raise ValueError(f"unexpected source schema for {key}")
                if str(_artifact_scalar(artifact, "trajectory_key")) != key:
                    raise ValueError("source trajectory key and record differ")
                if _artifact_scalar(artifact, "failure_cause") != "completed":
                    raise ValueError(f"source trajectory {key} is incomplete")
                if _artifact_scalar(artifact, "checkpoint_sha256") != checkpoint_digest:
                    raise ValueError("source trajectory and checkpoint SHA-256 differ")
                start_frame = int(_artifact_scalar(artifact, "start_frame"))
                step_stride = int(_artifact_scalar(artifact, "step_stride"))
                arrays = {
                    name: np.array(artifact[name], copy=True)
                    for name in (
                        "positions",
                        "edges",
                        "physical_cell_volume_weights",
                        "reference_currents",
                        "targets",
                        "physical_target_times",
                    )
                }
            if step_stride != int(checkpoint["step_stride"]):
                raise ValueError("source trajectory and checkpoint strides differ")
            volumes = arrays["physical_cell_volume_weights"]
            stored_weights = np.asarray(store.array(key, "node_weights"), dtype=np.float64).sum(axis=-1)
            if not np.allclose(volumes, stored_weights, rtol=1.0e-6, atol=1.0e-9):
                raise ValueError(f"source and shard weights differ for {key}")
            sample = store.tensor_sample(key, start_frame, step_stride=step_stride, device=device)
            adjacency = build_graph_adjacency(arrays["positions"].shape[0], arrays["edges"])

            for call_index in EXPECTED_CALLS:
                offset = call_index - 1
                current_np = arrays["reference_currents"][offset]
                target_np = arrays["targets"][offset]
                current = _state_tensor(current_np, device)
                target = _state_tensor(target_np, device)
                with torch.no_grad():
                    teacher_prediction = model_call(model, sample, current)
                teacher_np = teacher_prediction[0].float().cpu().numpy()
                interior, smooth, mask_contract = _reference_masks(
                    current,
                    target,
                    sample,
                    shock_quantile=SHOCK_QUANTILE,
                    dilation_hops=2,
                )
                node_type = sample["node_type"].reshape(1, -1, 1)
                interior_tensor = node_type == 0
                if not bool(interior_tensor.any()):
                    interior_tensor = sample["node_mask"].to(dtype=torch.bool)
                current_smooth = reference_smooth_region_mask(
                    current,
                    sample["directed_edges"],
                    sample["node_mask"],
                    interior_mask=interior_tensor,
                    shock_quantile=SHOCK_QUANTILE,
                    dilation_hops=0,
                )
                current_shock_seed = (
                    interior_tensor[0, :, 0] & ~current_smooth[0, :, 0]
                ).cpu().numpy().astype(bool)

                teacher_value = np.asarray(teacher_np, dtype=np.float64)
                current_value = np.asarray(current_np, dtype=np.float64)
                target_value = np.asarray(target_np, dtype=np.float64)
                proposed_update = (teacher_value - current_value) / component_scale[None, :]
                fresh = (teacher_value - target_value) / component_scale[None, :]
                proposal_highpass = node_highpass_field(proposed_update, arrays["edges"])
                fresh_highpass = node_highpass_field(fresh, arrays["edges"])
                sensor = proposal_sensor_metrics(
                    proposal_highpass,
                    fresh_highpass,
                    adjacency=adjacency,
                    weights=volumes,
                    interior_mask=interior,
                    d053_smooth_mask=smooth,
                    current_shock_seed=current_shock_seed,
                )
                hop_distance = graph_hop_distance(adjacency, current_shock_seed)
                eligible = interior & (hop_distance > CURRENT_SHOCK_EXCLUSION_HOPS)
                support = _top_score_mask(
                    np.linalg.norm(proposal_highpass, axis=-1),
                    eligible,
                    fraction=SENSOR_ELIGIBLE_NODE_FRACTION,
                )
                support_fraction = float(
                    np.count_nonzero(support) / np.count_nonzero(interior)
                )

                absolute_time = float(arrays["physical_target_times"][offset])
                vortex_center = _expected_vortex_center(
                    reference_config,
                    case["parameters"],
                    absolute_time=absolute_time,
                    gamma=model.gamma,
                )
                oracle, _ = bounded_realizability_oracle(
                    teacher_value,
                    target_value,
                    current_value,
                    support,
                    positions=arrays["positions"],
                    edges=arrays["edges"],
                    volumes=volumes,
                    component_scale=component_scale,
                    gamma=model.gamma,
                    vortex_center=vortex_center,
                )
                d055_row = d055_rows.get((key, call_index))
                if d055_row is None:
                    raise ValueError(f"D055 row missing for {(key, call_index)}")
                row = {
                    "schema": REALIZABILITY_SCHEMA,
                    "trajectory": key,
                    "call_index": call_index,
                    "physical_target_time": absolute_time,
                    "support_fraction_interior": support_fraction,
                    "oracle": oracle,
                    "mask_contract": mask_contract,
                    "teacher_admissibility": raw_admissibility_summary(teacher_np, gamma=model.gamma),
                    "d055_replay": {
                        "support_fraction_relative_error": _relative_error(
                            support_fraction,
                            d055_row.get("sensor", {}).get("selected_fraction_interior"),
                        ),
                        "fresh_energy_capture_relative_error": _relative_error(
                            sensor["smooth_highpass_energy_capture"],
                            d055_row.get("sensor", {}).get("smooth_highpass_energy_capture"),
                        ),
                    },
                    "claim_boundary": (
                        "support is legal and frozen before truth defines the balanced "
                        "oracle direction; no face-flux or learnability claim"
                    ),
                }
                rows.append(row)
                append_jsonl(rows_path, row)
            print(json.dumps({"trajectory": key, "rows": len(EXPECTED_CALLS)}, sort_keys=True), flush=True)
    finally:
        store.close()

    selector = realizability_selector(rows)
    summary = {
        "schema": REALIZABILITY_SCHEMA,
        "status": "complete" if selector["contract_complete"] else "failed_contract",
        "source": {
            "d052_artifact_dir_name": args.source_dir.name,
            "d052_summary_sha256": source_digest,
            "d055_artifact_dir_name": args.d055_dir.name,
            "d055_summary_sha256": sha256_file(d055_summary_path),
            "d055_rows_sha256": sha256_file(d055_rows_path),
            "family_manifest_sha256": sha256_file(args.family_manifest),
            "family_manifest_digest": family["manifest_digest_sha256"],
        },
        "checkpoint": {
            "sha256": checkpoint_digest,
            "config_digest": str(checkpoint["config_digest"]),
            "data_manifest_digest": data_manifest_digest,
            "boundary_mode": checkpoint["boundary_mode"],
            "raw_recurrence": bool(checkpoint["raw_recurrence"]),
        },
        "evaluation": {
            "device": str(device),
            "trajectory_keys": [str(record["trajectory"]) for record in records],
            "calls": list(EXPECTED_CALLS),
            "row_count": len(rows),
            "test_trajectory_access": [],
            "training_or_inference_intervention": False,
        },
        "selector": selector,
        "artifact_schema": {
            "realizability.jsonl": "one scalar oracle row per trajectory/selected call"
        },
        "claim_boundary": {
            "verified": (
                "bounded, balanced state-space correction headroom on D055's "
                "legally selected validation support"
            ),
            "plausible": "one frozen-global tiny detail-head fit only after a full pass",
            "unsupported": (
                "learnability, autonomous rollout gain, face flux, local conservation, "
                "or strength-OOD behavior"
            ),
        },
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(selector, indent=2, sort_keys=True), flush=True)
    if not selector["contract_complete"]:
        raise RuntimeError("D056 failed its frozen diagnostic contract")


if __name__ == "__main__":
    main()
