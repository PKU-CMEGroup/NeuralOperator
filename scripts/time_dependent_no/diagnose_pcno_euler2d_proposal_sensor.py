#!/usr/bin/env python3
"""Test one legal self-sensor for D054's sparse fresh ripple pockets.

D055 performs no training and changes no rollout.  It ranks nodes by the graph
high-pass amplitude of the frozen PCNO's own proposed conservative update,
excluding a two-hop current-state pressure-jump halo.  The target is used only
after selection to measure how much D053 fresh-defect high-pass energy the
causal support captured.
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
    _weighted_norm,
)
from scripts.time_dependent_no.diagnose_pcno_euler2d_fresh_defect_locality import (  # noqa: E402
    ORACLE_NODE_FRACTION,
    SHOCK_QUANTILE,
    _top_fraction_mask,
    build_graph_adjacency,
    graph_hop_distance,
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
from utility.time_dependent_no.pcno_euler2d import (  # noqa: E402
    PCNOEuler2DShardStore,
    reference_smooth_region_mask,
)
from utility.time_dependent_no.pcno_ripple_diagnostics import (  # noqa: E402
    RIPPLE_DIAGNOSTIC_SCHEMA,
    node_highpass_field,
    normalized_node_weights,
    raw_admissibility_summary,
    spatial_correlation_summary,
)


PROPOSAL_SENSOR_SCHEMA = "pcno_euler2d_proposal_sensor_d055_v1"
EXPECTED_TRAJECTORY_COUNT = 6
EXPECTED_CALLS = (1, 10, 30, 60)
LATE_CALLS = (30, 60)
REQUIRED_CASES = 5
CURRENT_SHOCK_EXCLUSION_HOPS = 2
SENSOR_ELIGIBLE_NODE_FRACTION = 0.20
SENSOR_SUPPORT_FRACTION_MAX = 0.20
SENSOR_ENERGY_CAPTURE_MIN = 0.50
SOURCE_REPLAY_RELATIVE_TOLERANCE = 2.0e-4


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--d053-dir", type=Path, required=True)
    parser.add_argument("--d054-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")


def _top_score_mask(
    score: np.ndarray,
    eligible_mask: np.ndarray,
    *,
    fraction: float,
) -> np.ndarray:
    values = np.asarray(score, dtype=np.float64)
    eligible = np.asarray(eligible_mask, dtype=bool)
    if values.ndim != 1 or eligible.shape != values.shape:
        raise ValueError("score and eligible_mask must share shape [N]")
    if not np.all(np.isfinite(values)):
        raise ValueError("proposal score must be finite")
    available = np.flatnonzero(eligible)
    if available.size == 0:
        raise ValueError("proposal sensor has no eligible nodes")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0,1]")
    count = max(1, int(math.floor(fraction * available.size + 1.0e-12)))
    order = np.lexsort((available, -values[available]))
    selected = np.zeros_like(eligible)
    selected[available[order[:count]]] = True
    return selected


def proposal_sensor_metrics(
    proposed_update_highpass: np.ndarray,
    fresh_defect_highpass: np.ndarray,
    *,
    adjacency: Sequence[Sequence[int]],
    weights: np.ndarray,
    interior_mask: np.ndarray,
    d053_smooth_mask: np.ndarray,
    current_shock_seed: np.ndarray,
) -> dict[str, Any]:
    """Evaluate one fixed target-free proposal score against fresh error."""

    proposal = np.asarray(proposed_update_highpass, dtype=np.float64)
    fresh = np.asarray(fresh_defect_highpass, dtype=np.float64)
    if proposal.ndim != 2 or fresh.shape != proposal.shape:
        raise ValueError("proposal and fresh high-pass fields must share shape [N,C]")
    interior = np.asarray(interior_mask, dtype=bool)
    smooth = np.asarray(d053_smooth_mask, dtype=bool)
    shock_seed = np.asarray(current_shock_seed, dtype=bool)
    expected = (proposal.shape[0],)
    if any(mask.shape != expected for mask in (interior, smooth, shock_seed)):
        raise ValueError("node masks must match proposal fields")
    if not np.any(interior) or not np.any(smooth) or not np.any(shock_seed):
        raise ValueError("interior, smooth, and current shock masks must be nonempty")
    if np.any(smooth & ~interior) or np.any(shock_seed & ~interior):
        raise ValueError("smooth and shock masks must lie inside the interior")

    mass = normalized_node_weights(weights, name="D055 proposal sensor")
    proposal_score = np.linalg.norm(proposal, axis=-1)
    fresh_amplitude = np.linalg.norm(fresh, axis=-1)
    fresh_energy = mass * fresh_amplitude**2
    smooth_energy = float(np.sum(fresh_energy[smooth]))
    if smooth_energy <= 1.0e-30:
        return {"status": "zero_smooth_highpass_energy"}

    hop_distance = graph_hop_distance(adjacency, shock_seed)
    current_shock_halo = (
        interior
        & (hop_distance >= 0)
        & (hop_distance <= CURRENT_SHOCK_EXCLUSION_HOPS)
    )
    eligible = interior & ~current_shock_halo
    selected = _top_score_mask(
        proposal_score,
        eligible,
        fraction=SENSOR_ELIGIBLE_NODE_FRACTION,
    )
    truth_oracle = _top_fraction_mask(
        fresh_energy,
        smooth,
        ORACLE_NODE_FRACTION,
    )

    def capture(mask: np.ndarray) -> float:
        return float(np.sum(fresh_energy[mask & smooth]) / smooth_energy)

    correlation = spatial_correlation_summary(
        fresh_amplitude,
        {"proposal_update_highpass_amplitude": proposal_score},
        mask=smooth,
    )["proposal_update_highpass_amplitude"]
    return {
        "status": "available",
        "score_contract": (
            "norm of graph high-pass of (G(current)-current)/training_scale; "
            "current and frozen proposal only"
        ),
        "current_shock_exclusion_hops": CURRENT_SHOCK_EXCLUSION_HOPS,
        "eligible_node_count": int(np.count_nonzero(eligible)),
        "eligible_fraction_interior": float(
            np.count_nonzero(eligible) / np.count_nonzero(interior)
        ),
        "selected_node_count": int(np.count_nonzero(selected)),
        "selected_fraction_interior": float(
            np.count_nonzero(selected) / np.count_nonzero(interior)
        ),
        "selected_fraction_eligible": float(
            np.count_nonzero(selected) / np.count_nonzero(eligible)
        ),
        "smooth_highpass_energy_capture": capture(selected),
        "truth_oracle_smooth_highpass_energy_capture": capture(truth_oracle),
        "truth_oracle_node_recall": float(
            np.count_nonzero(selected & truth_oracle)
            / max(np.count_nonzero(truth_oracle), 1)
        ),
        "proposal_score_fresh_amplitude_correlation": correlation,
    }


def _median(values: Sequence[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return None if not finite else float(np.median(finite))


def proposal_sensor_selector(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the frozen D055 six-case proposal-sensor gate."""

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
        max(float(value) for value in row.get("source_replay", {}).values())
        for row in row_list
    ]
    checks = {
        "schema": bool(row_list)
        and all(row.get("schema") == PROPOSAL_SENSOR_SCHEMA for row in row_list),
        "exact_six_by_four_rows": (
            len(row_list) == EXPECTED_TRAJECTORY_COUNT * len(EXPECTED_CALLS)
            and len(trajectories) == EXPECTED_TRAJECTORY_COUNT
            and not duplicate_keys
            and all(calls == list(EXPECTED_CALLS) for calls in calls_by_trajectory.values())
        ),
        "all_sensor_metrics_available": all(
            row.get("sensor", {}).get("status") == "available" for row in row_list
        ),
        "all_teacher_predictions_admissible": all(
            row.get("teacher_admissibility", {}).get("all_admissible") is True
            for row in row_list
        ),
        "no_smooth_mask_fallback": all(
            row.get("mask_contract", {}).get("smooth_fallback_to_interior") is False
            for row in row_list
        ),
        "source_replay": bool(replay_errors)
        and max(replay_errors) <= SOURCE_REPLAY_RELATIVE_TOLERANCE,
        "support_cap": all(
            float(row.get("sensor", {}).get("selected_fraction_interior", math.inf))
            <= SENSOR_SUPPORT_FRACTION_MAX + 1.0e-12
            for row in row_list
        ),
    }
    contract_complete = all(checks.values())

    late: dict[str, Any] = {}
    repeated = True
    for call in LATE_CALLS:
        selected = [row for row in row_list if int(row.get("call_index", -1)) == call]
        captures = [
            float(row["sensor"]["smooth_highpass_energy_capture"])
            for row in selected
            if row.get("sensor", {}).get("status") == "available"
        ]
        supports = [
            float(row["sensor"]["selected_fraction_interior"])
            for row in selected
            if row.get("sensor", {}).get("status") == "available"
        ]
        passing = sum(
            capture >= SENSOR_ENERGY_CAPTURE_MIN
            and support <= SENSOR_SUPPORT_FRACTION_MAX
            for capture, support in zip(captures, supports, strict=True)
        )
        repeated &= len(selected) == EXPECTED_TRAJECTORY_COUNT and passing >= REQUIRED_CASES
        late[str(call)] = {
            "row_count": len(selected),
            "passing_case_count": passing,
            "median_energy_capture": _median(captures),
            "median_support_fraction_interior": _median(supports),
        }

    if not contract_complete:
        classification = "incomplete_contract"
        route = "stop_without_training"
    elif repeated:
        classification = "proposal_self_sensor_candidate"
        route = "authorize_one_matched_tiny_fit_contract_only"
    else:
        classification = "sparse_but_not_legally_self_localizable"
        route = "reject_sensor_gated_local_detail_no_training"
    return {
        "version": "proposal_self_sensor_selector_d055_v1",
        "contract_complete": contract_complete,
        "contract_checks": checks,
        "duplicate_keys": [list(key) for key in duplicate_keys],
        "row_count": len(row_list),
        "trajectory_count": len(trajectories),
        "late_calls": late,
        "causal_capture_repeated": repeated,
        "classification": classification,
        "route": route,
        "thresholds": {
            "current_shock_exclusion_hops": CURRENT_SHOCK_EXCLUSION_HOPS,
            "selected_fraction_of_eligible": SENSOR_ELIGIBLE_NODE_FRACTION,
            "support_fraction_interior_max": SENSOR_SUPPORT_FRACTION_MAX,
            "smooth_highpass_energy_capture_min": SENSOR_ENERGY_CAPTURE_MIN,
            "required_cases_per_late_call": REQUIRED_CASES,
            "source_replay_relative_tolerance": SOURCE_REPLAY_RELATIVE_TOLERANCE,
        },
        "claim_boundary": (
            "one frozen legal proposal score on validation calls; not a learned "
            "sensor, correction, rollout improvement, or conservation result"
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
            raise ValueError(f"duplicate source row: {key}")
        result[key] = row
    return result


def _relative_error(value: float | None, reference: Any) -> float:
    if value is None or reference is None:
        return math.inf
    expected = float(reference)
    return abs(float(value) - expected) / max(abs(expected), 1.0e-12)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_args(args)
    device = select_device(args.device)

    source_summary_path = args.source_dir / "summary.json"
    d053_summary_path = args.d053_dir / "summary.json"
    d054_summary_path = args.d054_dir / "summary.json"
    summaries = []
    for path in (source_summary_path, d053_summary_path, d054_summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    source_summary, d053_summary, d054_summary = summaries
    if source_summary.get("status") != "complete":
        raise ValueError("D055 requires completed D052")
    if (
        d053_summary.get("status") != "complete"
        or d053_summary.get("selector", {}).get("classification") != "mixed_or_split"
        or d053_summary.get("selector", {}).get("contract_complete") is not True
    ):
        raise ValueError("D055 requires completed mixed/split D053")
    if (
        d054_summary.get("status") != "complete"
        or d054_summary.get("selector", {}).get("classification")
        != "localized_but_not_causally_shock_localizable"
        or d054_summary.get("selector", {}).get("contract_complete") is not True
    ):
        raise ValueError("D055 requires D054's oracle-only locality result")
    source_digest = sha256_file(source_summary_path)
    if d053_summary.get("source", {}).get("summary_sha256") != source_digest:
        raise ValueError("D052 source differs from the D053 binding")
    if d054_summary.get("source", {}).get("d052_summary_sha256") != source_digest:
        raise ValueError("D052 source differs from the D054 binding")

    d053_rows_path = args.d053_dir / "error_sources.jsonl"
    d054_rows_path = args.d054_dir / "fresh_defect_locality.jsonl"
    d053_rows = _load_rows(d053_rows_path)
    d054_rows = _load_rows(d054_rows_path)
    records = _source_records(source_summary)

    checkpoint = load_checkpoint(args.checkpoint)
    checkpoint_digest = sha256_file(args.checkpoint)
    for summary, name in ((d053_summary, "D053"), (d054_summary, "D054")):
        if summary.get("checkpoint", {}).get("sha256") != checkpoint_digest:
            raise ValueError(f"checkpoint SHA-256 differs from {name}")
    if checkpoint.get("boundary_mode") != "model_all_nodes" or not bool(
        checkpoint.get("raw_recurrence")
    ):
        raise ValueError("D055 requires legal model-all-node raw recurrence")
    store = PCNOEuler2DShardStore(args.data_dir)
    if store.manifest_digest != checkpoint.get("data_manifest_digest"):
        raise ValueError("checkpoint and shard manifest digests differ")
    if store.manifest.get("weight_provenance") != "validated_physical_cell_volume_normalized":
        raise ValueError("D055 requires validated physical cell-volume weights")
    data_manifest_digest = store.manifest_digest
    model = build_model(checkpoint, device)
    component_scale = model.state_scale.detach().cpu().numpy().astype(np.float64).reshape(1, -1)

    args.output_dir.mkdir(parents=True)
    rows_path = args.output_dir / "proposal_sensor.jsonl"
    rows: list[dict[str, Any]] = []
    try:
        for record in records:
            key = str(record["trajectory"])
            if key not in {str(value) for value in checkpoint.get("val_keys", [])}:
                raise ValueError(f"source trajectory {key} is not a validation case")
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
                if _artifact_scalar(artifact, "boundary_mode") != "model_all_nodes":
                    raise ValueError("source used an incompatible boundary mode")
                if _artifact_scalar(artifact, "weight_provenance") != (
                    "validated_physical_cell_volume_normalized"
                ):
                    raise ValueError("source lacks physical-volume provenance")
                if _artifact_scalar(artifact, "checkpoint_sha256") != checkpoint_digest:
                    raise ValueError("source trajectory and checkpoint SHA-256 differ")
                start_frame = int(_artifact_scalar(artifact, "start_frame"))
                step_stride = int(_artifact_scalar(artifact, "step_stride"))
                delta_t = float(_artifact_scalar(artifact, "delta_t"))
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
            stored_weights = np.asarray(store.array(key, "node_weights"), dtype=np.float64).sum(axis=-1)
            if not np.allclose(
                arrays["physical_cell_volume_weights"], stored_weights, rtol=1.0e-6, atol=1.0e-9
            ):
                raise ValueError(f"source and shard weights differ for {key}")
            if not np.allclose(arrays["positions"], store.array(key, "nodes")):
                raise ValueError(f"source and shard positions differ for {key}")
            if not np.array_equal(arrays["edges"], store.array(key, "edges")):
                raise ValueError(f"source and shard edges differ for {key}")

            sample = store.tensor_sample(key, start_frame, step_stride=step_stride, device=device)
            adjacency = build_graph_adjacency(arrays["positions"].shape[0], arrays["edges"])
            weights = arrays["physical_cell_volume_weights"]
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
                proposed_update = (teacher_value - current_value) / component_scale
                fresh = (teacher_value - target_value) / component_scale
                proposal_highpass = node_highpass_field(proposed_update, arrays["edges"])
                fresh_highpass = node_highpass_field(fresh, arrays["edges"])
                sensor = proposal_sensor_metrics(
                    proposal_highpass,
                    fresh_highpass,
                    adjacency=adjacency,
                    weights=weights,
                    interior_mask=interior,
                    d053_smooth_mask=smooth,
                    current_shock_seed=current_shock_seed,
                )

                key_call = (key, call_index)
                d053_row = d053_rows.get(key_call)
                d054_row = d054_rows.get(key_call)
                if d053_row is None or d054_row is None:
                    raise ValueError(f"source diagnostic row missing for {key_call}")
                fresh_full_norm = _weighted_norm(fresh, weights, interior)
                fresh_highpass_norm = _weighted_norm(fresh_highpass, weights, smooth)
                row = {
                    "schema": PROPOSAL_SENSOR_SCHEMA,
                    "trajectory": key,
                    "call_index": call_index,
                    "physical_target_time": float(arrays["physical_target_times"][offset]),
                    "delta_t": delta_t,
                    "sensor": sensor,
                    "mask_contract": mask_contract,
                    "source_replay": {
                        "d053_interior_fresh_norm_relative_error": _relative_error(
                            fresh_full_norm,
                            d053_row.get("regions", {}).get("interior_full", {}).get("fresh_defect_norm"),
                        ),
                        "d053_smooth_highpass_fresh_norm_relative_error": _relative_error(
                            fresh_highpass_norm,
                            d053_row.get("regions", {}).get("smooth_highpass", {}).get("fresh_defect_norm"),
                        ),
                        "d054_truth_oracle_capture_relative_error": _relative_error(
                            sensor.get("truth_oracle_smooth_highpass_energy_capture"),
                            d054_row.get("locality", {}).get("truth_oracle", {}).get("smooth_highpass_energy_capture"),
                        ),
                    },
                    "teacher_admissibility": raw_admissibility_summary(teacher_np, gamma=model.gamma),
                    "claim_boundary": (
                        "support selection uses only current state and frozen PCNO proposal; "
                        "target enters evaluation after support is frozen"
                    ),
                }
                rows.append(row)
                append_jsonl(rows_path, row)
            print(json.dumps({"trajectory": key, "rows": len(EXPECTED_CALLS)}, sort_keys=True), flush=True)
    finally:
        store.close()

    selector = proposal_sensor_selector(rows)
    summary = {
        "schema": PROPOSAL_SENSOR_SCHEMA,
        "status": "complete" if selector["contract_complete"] else "failed_contract",
        "source": {
            "d052_artifact_dir_name": args.source_dir.name,
            "d052_summary_sha256": source_digest,
            "d053_artifact_dir_name": args.d053_dir.name,
            "d053_summary_sha256": sha256_file(d053_summary_path),
            "d053_rows_sha256": sha256_file(d053_rows_path),
            "d054_artifact_dir_name": args.d054_dir.name,
            "d054_summary_sha256": sha256_file(d054_summary_path),
            "d054_rows_sha256": sha256_file(d054_rows_path),
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
            "proposal_sensor.jsonl": "one scalar row per trajectory/selected call"
        },
        "claim_boundary": {
            "verified": "coverage of one fixed legal proposal score on frozen validation calls",
            "plausible": "one detail-target tiny fit only if every selector gate passes",
            "unsupported": (
                "learned localization, corrected rollout, OOD behavior, flux prediction, "
                "or conservation by construction"
            ),
        },
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(selector, indent=2, sort_keys=True), flush=True)
    if not selector["contract_complete"]:
        raise RuntimeError("D055 failed its frozen diagnostic contract")


if __name__ == "__main__":
    main()
