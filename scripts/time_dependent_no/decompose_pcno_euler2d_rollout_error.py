"""Decompose frozen PCNO rollout error into propagation and fresh defect.

D053 performs no training and applies no inference intervention. It consumes a
completed D052/D013 rollout bundle, evaluates the same legal raw PCNO map only
on saved reference currents, and records the exact additive error identity.
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
)


ERROR_SOURCE_SCHEMA = "pcno_euler2d_error_source_d053_v1"
EXPECTED_TRAJECTORY_COUNT = 6
EXPECTED_CALLS = tuple(range(1, 61))
LATE_CALLS = (30, 60)
REQUIRED_CASES = 5
PROPAGATION_SHARE_MIN = 0.65
FRESH_DEFECT_SHARE_MAX = 0.35
IDENTITY_RELATIVE_TOLERANCE = 1e-6
CALL1_PROPAGATED_SHARE_MAX = 1e-6
SOURCE_REPLAY_MAX_ABSOLUTE_TOLERANCE = 1e-5


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--shock-quantile", type=float, default=0.9)
    parser.add_argument("--shock-dilation-hops", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2403)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.shock_quantile < 1.0:
        raise ValueError("shock quantile must lie in (0,1)")
    if args.shock_dilation_hops < 0:
        raise ValueError("shock dilation hops must be nonnegative")
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")


def _state_tensor(state: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(
        np.array(state, copy=True), dtype=torch.float32, device=device
    ).unsqueeze(0)


def _weighted_norm(
    field: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
) -> float | None:
    values = np.asarray(field, dtype=np.float64)
    selected = np.asarray(mask, dtype=bool)
    if values.ndim != 2 or selected.shape != (values.shape[0],):
        raise ValueError("field and mask must have shapes [N,C] and [N]")
    if not np.any(selected):
        return None
    mass = normalized_node_weights(weights, name="D053 decomposition")
    energy = float(np.sum(mass[selected, None] * values[selected] ** 2))
    return float(np.sqrt(max(energy, 0.0)))


def weighted_decomposition_metrics(
    total: np.ndarray,
    propagated: np.ndarray,
    fresh_defect: np.ndarray,
    weights: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Measure an additive vector decomposition under one fixed node measure."""

    total_value = np.asarray(total, dtype=np.float64)
    propagated_value = np.asarray(propagated, dtype=np.float64)
    fresh_value = np.asarray(fresh_defect, dtype=np.float64)
    if (
        total_value.ndim != 2
        or propagated_value.shape != total_value.shape
        or fresh_value.shape != total_value.shape
    ):
        raise ValueError("decomposition fields must share shape [N,C]")
    selected = (
        np.ones(total_value.shape[0], dtype=bool)
        if mask is None
        else np.asarray(mask, dtype=bool)
    )
    if selected.shape != (total_value.shape[0],):
        raise ValueError("decomposition mask must have shape [N]")
    if not np.any(selected):
        return {"status": "empty_region", "selected_node_count": 0}
    mass = normalized_node_weights(weights, name="D053 decomposition")
    chosen_mass = mass[selected, None]

    def inner(left: np.ndarray, right: np.ndarray) -> float:
        return float(np.sum(chosen_mass * left[selected] * right[selected]))

    total_energy = inner(total_value, total_value)
    propagated_energy = inner(propagated_value, propagated_value)
    fresh_energy = inner(fresh_value, fresh_value)
    cross_inner = inner(propagated_value, fresh_value)
    residual = total_value - propagated_value - fresh_value
    residual_energy = inner(residual, residual)
    total_norm = math.sqrt(max(total_energy, 0.0))
    propagated_norm = math.sqrt(max(propagated_energy, 0.0))
    fresh_norm = math.sqrt(max(fresh_energy, 0.0))
    residual_norm = math.sqrt(max(residual_energy, 0.0))
    magnitude_sum = propagated_norm + fresh_norm
    identity_energy = propagated_energy + fresh_energy + 2.0 * cross_inner
    norm_scale = max(total_norm, magnitude_sum, 1e-30)
    energy_scale = max(
        abs(total_energy),
        propagated_energy + fresh_energy + 2.0 * abs(cross_inner),
        1e-30,
    )
    cosine = None
    if propagated_norm > 1e-30 and fresh_norm > 1e-30:
        cosine = cross_inner / (propagated_norm * fresh_norm)
    total_fraction_scale = max(abs(total_energy), 1e-30)
    return {
        "status": "available",
        "selected_node_count": int(np.count_nonzero(selected)),
        "selected_weight": float(mass[selected].sum()),
        "total_norm": total_norm,
        "propagated_norm": propagated_norm,
        "fresh_defect_norm": fresh_norm,
        "propagated_magnitude_share": (
            None if magnitude_sum <= 1e-30 else propagated_norm / magnitude_sum
        ),
        "propagated_fresh_cosine": cosine,
        "propagated_energy_fraction_of_total": (
            propagated_energy / total_fraction_scale
        ),
        "fresh_defect_energy_fraction_of_total": fresh_energy / total_fraction_scale,
        "cross_energy_fraction_of_total": 2.0 * cross_inner / total_fraction_scale,
        "relative_reconstruction_residual": residual_norm / norm_scale,
        "relative_energy_identity_residual": (
            abs(total_energy - identity_energy) / energy_scale
        ),
    }


def _gain(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 1e-30:
        return None
    return float(numerator / denominator)


def _median(values: Sequence[float | None]) -> float | None:
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return None if not finite else float(np.median(finite))


def error_source_selector(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the frozen six-case, 60-call D053 routing contract."""

    row_list = [dict(row) for row in rows]
    keys = [
        (str(row.get("trajectory")), int(row.get("call_index", -1))) for row in row_list
    ]
    duplicate_keys = sorted({key for key in keys if keys.count(key) > 1})
    trajectories = sorted({key[0] for key in keys})
    calls_by_trajectory = {
        trajectory: sorted(key[1] for key in keys if key[0] == trajectory)
        for trajectory in trajectories
    }

    def region(row: Mapping[str, Any], name: str) -> Mapping[str, Any]:
        return row.get("regions", {}).get(name, {})

    identity_residuals = [
        float(region(row, name).get(metric, math.inf))
        for row in row_list
        for name in ("interior_full", "smooth_highpass")
        for metric in (
            "relative_reconstruction_residual",
            "relative_energy_identity_residual",
        )
    ]
    all_admissible = all(
        bool(row.get("admissibility", {}).get(source, {}).get("all_admissible"))
        for row in row_list
        for source in ("rollout_prediction", "teacher_prediction")
    )
    regions_available = all(
        region(row, "interior_full").get("status") == "available"
        and region(row, "smooth_highpass").get("status") == "available"
        for row in row_list
    )
    no_mask_fallback = all(
        not bool(row.get("mask_contract", {}).get("smooth_fallback_to_interior"))
        for row in row_list
    )
    call1_shares = [
        region(row, name).get("propagated_magnitude_share")
        for row in row_list
        if int(row.get("call_index", -1)) == 1
        for name in ("interior_full", "smooth_highpass")
    ]
    call1_zero = bool(call1_shares) and all(
        value is not None and float(value) <= CALL1_PROPAGATED_SHARE_MAX
        for value in call1_shares
    )
    exact_shape = (
        len(trajectories) == EXPECTED_TRAJECTORY_COUNT
        and all(calls == list(EXPECTED_CALLS) for calls in calls_by_trajectory.values())
        and len(row_list) == EXPECTED_TRAJECTORY_COUNT * len(EXPECTED_CALLS)
        and not duplicate_keys
    )
    source_replay_errors = [
        float(row.get("source_replay", {}).get("max_absolute_error", math.inf))
        for row in row_list
    ]
    checks = {
        "exact_six_by_sixty_rows": exact_shape,
        "schema": bool(row_list)
        and all(row.get("schema") == ERROR_SOURCE_SCHEMA for row in row_list),
        "identity_closure": bool(identity_residuals)
        and max(identity_residuals) <= IDENTITY_RELATIVE_TOLERANCE,
        "raw_admissibility": all_admissible,
        "required_regions_available": regions_available,
        "no_smooth_mask_fallback": no_mask_fallback,
        "call1_zero_propagation": call1_zero,
        "source_rollout_replay": bool(source_replay_errors)
        and max(source_replay_errors) <= SOURCE_REPLAY_MAX_ABSOLUTE_TOLERANCE,
    }
    contract_complete = all(checks.values())

    late: dict[str, Any] = {}
    for call_index in LATE_CALLS:
        selected = [
            row for row in row_list if int(row.get("call_index", -1)) == call_index
        ]
        full_shares = [
            region(row, "interior_full").get("propagated_magnitude_share")
            for row in selected
        ]
        high_shares = [
            region(row, "smooth_highpass").get("propagated_magnitude_share")
            for row in selected
        ]
        paired = [
            (float(full), float(high))
            for full, high in zip(full_shares, high_shares, strict=True)
            if full is not None and high is not None
        ]
        late[str(call_index)] = {
            "row_count": len(selected),
            "median_full_propagated_share": _median(full_shares),
            "median_smooth_highpass_propagated_share": _median(high_shares),
            "propagation_dominated_case_count": sum(
                full >= PROPAGATION_SHARE_MIN and high >= PROPAGATION_SHARE_MIN
                for full, high in paired
            ),
            "fresh_defect_dominated_case_count": sum(
                full <= FRESH_DEFECT_SHARE_MAX and high <= FRESH_DEFECT_SHARE_MAX
                for full, high in paired
            ),
        }
    propagation_pass = all(
        late[str(call)]["propagation_dominated_case_count"] >= REQUIRED_CASES
        for call in LATE_CALLS
    )
    fresh_pass = all(
        late[str(call)]["fresh_defect_dominated_case_count"] >= REQUIRED_CASES
        for call in LATE_CALLS
    )
    if not contract_complete:
        classification = "incomplete_contract"
        route = "no_learned_method"
    elif propagation_pass and not fresh_pass:
        classification = "propagation_dominated"
        route = "one_matched_short_generated_state_exposure_capacity_test"
    elif fresh_pass and not propagation_pass:
        classification = "fresh_teacher_defect_dominated"
        route = "target_or_representation_diagnostic_without_recurrence_training"
    else:
        classification = "mixed_or_split"
        route = "no_learned_method"
    return {
        "version": "exact_rollout_error_selector_d053_v1",
        "contract_complete": contract_complete,
        "contract_checks": checks,
        "duplicate_keys": [list(key) for key in duplicate_keys],
        "trajectory_count": len(trajectories),
        "row_count": len(row_list),
        "late_calls": late,
        "thresholds": {
            "required_cases": REQUIRED_CASES,
            "propagation_share_min": PROPAGATION_SHARE_MIN,
            "fresh_defect_share_max": FRESH_DEFECT_SHARE_MAX,
            "identity_relative_tolerance": IDENTITY_RELATIVE_TOLERANCE,
            "call1_propagated_share_max": CALL1_PROPAGATED_SHARE_MAX,
            "source_replay_max_absolute_tolerance": (
                SOURCE_REPLAY_MAX_ABSOLUTE_TOLERANCE
            ),
        },
        "classification": classification,
        "route": route,
        "claim_boundary": (
            "exact additive attribution for this frozen learned map; it does not "
            "identify a universal PDE or architecture mechanism"
        ),
    }


def _source_records(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = [dict(record) for record in summary.get("trajectories", [])]
    if not records:
        raise ValueError("source summary contains no trajectories")
    if any(not bool(record.get("completed")) for record in records):
        raise ValueError("D053 requires completed source trajectories")
    return records


def _artifact_scalar(artifact: Mapping[str, np.ndarray], name: str) -> Any:
    if name not in artifact:
        raise ValueError(f"source trajectory artifact is missing {name}")
    value = np.asarray(artifact[name])
    if value.size != 1:
        raise ValueError(f"source trajectory field {name} must be scalar")
    return value.item()


def _reference_masks(
    current: torch.Tensor,
    target: torch.Tensor,
    sample: Mapping[str, torch.Tensor],
    *,
    shock_quantile: float,
    dilation_hops: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    node_type = sample["node_type"].reshape(1, -1, 1)
    interior = node_type == 0
    interior_fallback = not bool(interior.any())
    if interior_fallback:
        interior = sample["node_mask"].to(dtype=torch.bool)
    current_smooth = reference_smooth_region_mask(
        current,
        sample["directed_edges"],
        sample["node_mask"],
        interior_mask=interior,
        shock_quantile=shock_quantile,
        dilation_hops=dilation_hops,
    )
    target_smooth = reference_smooth_region_mask(
        target,
        sample["directed_edges"],
        sample["node_mask"],
        interior_mask=interior,
        shock_quantile=shock_quantile,
        dilation_hops=dilation_hops,
    )
    interior_np = interior[0, :, 0].cpu().numpy().astype(bool)
    smooth = (
        (current_smooth[0, :, 0] & target_smooth[0, :, 0]).cpu().numpy().astype(bool)
    )
    smooth_fallback = not bool(np.any(smooth))
    if smooth_fallback:
        smooth = interior_np.copy()
    shock = interior_np & ~smooth
    return (
        interior_np,
        smooth,
        {
            "source": "reference_current_and_target_shock_union",
            "shock_quantile": shock_quantile,
            "dilation_hops": dilation_hops,
            "interior_fallback_to_valid_nodes": interior_fallback,
            "smooth_fallback_to_interior": smooth_fallback,
            "interior_node_count": int(np.count_nonzero(interior_np)),
            "smooth_node_count": int(np.count_nonzero(smooth)),
            "shock_support_node_count": int(np.count_nonzero(shock)),
        },
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = select_device(args.device)
    source_summary_path = args.source_dir / "summary.json"
    if not source_summary_path.is_file():
        raise FileNotFoundError(source_summary_path)
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    if source_summary.get("status") != "complete":
        raise ValueError("D053 requires a completed source diagnostic")
    records = _source_records(source_summary)

    checkpoint = load_checkpoint(args.checkpoint)
    checkpoint_digest = sha256_file(args.checkpoint)
    source_checkpoint = source_summary.get("checkpoint", {})
    if source_checkpoint.get("sha256") != checkpoint_digest:
        raise ValueError("source diagnostic and checkpoint SHA-256 differ")
    if checkpoint.get("boundary_mode") != "model_all_nodes" or not bool(
        checkpoint.get("raw_recurrence")
    ):
        raise ValueError("D053 requires legal model-all-node raw recurrence")
    store = PCNOEuler2DShardStore(args.data_dir)
    if store.manifest_digest != checkpoint.get("data_manifest_digest"):
        raise ValueError("checkpoint and shard manifest digests differ")
    if store.manifest.get("weight_provenance") != (
        "validated_physical_cell_volume_normalized"
    ):
        raise ValueError("D053 requires validated physical cell-volume weights")
    model = build_model(checkpoint, device)
    component_scale = (
        model.state_scale.detach().cpu().numpy().astype(np.float64).reshape(1, -1)
    )
    args.output_dir.mkdir(parents=True)
    rows_path = args.output_dir / "error_sources.jsonl"
    rows: list[dict[str, Any]] = []

    try:
        for record in records:
            key = str(record["trajectory"])
            if key not in {str(value) for value in checkpoint.get("val_keys", [])}:
                raise ValueError(
                    f"source trajectory {key} is not in checkpoint val_keys"
                )
            artifact_path = args.source_dir / str(record["artifact"])
            if not artifact_path.is_file():
                raise FileNotFoundError(artifact_path)
            with np.load(artifact_path, allow_pickle=False) as artifact:
                if _artifact_scalar(artifact, "schema") != RIPPLE_DIAGNOSTIC_SCHEMA:
                    raise ValueError(f"unexpected source schema for {key}")
                if str(_artifact_scalar(artifact, "trajectory_key")) != key:
                    raise ValueError("source trajectory key and filename record differ")
                if _artifact_scalar(artifact, "failure_cause") != "completed":
                    raise ValueError(f"source trajectory {key} is incomplete")
                if _artifact_scalar(artifact, "boundary_mode") != "model_all_nodes":
                    raise ValueError(
                        "source trajectory used an incompatible boundary mode"
                    )
                if _artifact_scalar(artifact, "weight_provenance") != (
                    "validated_physical_cell_volume_normalized"
                ):
                    raise ValueError(
                        "source trajectory lacks physical-volume provenance"
                    )
                if _artifact_scalar(artifact, "checkpoint_sha256") != checkpoint_digest:
                    raise ValueError("trajectory and checkpoint SHA-256 differ")
                if _artifact_scalar(artifact, "config_digest") != str(
                    checkpoint["config_digest"]
                ):
                    raise ValueError("trajectory and checkpoint config digests differ")
                valid_length = int(_artifact_scalar(artifact, "valid_length"))
                start_frame = int(_artifact_scalar(artifact, "start_frame"))
                step_stride = int(_artifact_scalar(artifact, "step_stride"))
                delta_t = float(_artifact_scalar(artifact, "delta_t"))
                if step_stride != int(checkpoint["step_stride"]):
                    raise ValueError("source trajectory and checkpoint strides differ")
                arrays = {
                    name: np.array(artifact[name], copy=True)
                    for name in (
                        "positions",
                        "edges",
                        "node_type",
                        "physical_cell_volume_weights",
                        "rollout_currents",
                        "reference_currents",
                        "predictions",
                        "targets",
                        "physical_target_times",
                    )
                }
            trajectory_arrays = (
                arrays["rollout_currents"],
                arrays["reference_currents"],
                arrays["predictions"],
                arrays["targets"],
            )
            if any(
                array.shape != trajectory_arrays[0].shape for array in trajectory_arrays
            ):
                raise ValueError(f"source state arrays differ in shape for {key}")
            if trajectory_arrays[0].shape != (
                valid_length,
                arrays["positions"].shape[0],
                4,
            ):
                raise ValueError(f"invalid source state-array shape for {key}")
            if arrays["physical_target_times"].shape != (valid_length,):
                raise ValueError(f"invalid physical-time shape for {key}")

            stored_weights = np.asarray(
                store.array(key, "node_weights"), dtype=np.float64
            ).sum(axis=-1)
            if not np.allclose(
                arrays["physical_cell_volume_weights"],
                stored_weights,
                rtol=1e-6,
                atol=1e-9,
            ):
                raise ValueError(
                    f"source and shard cell-volume weights differ for {key}"
                )
            if not np.allclose(arrays["positions"], store.array(key, "nodes")):
                raise ValueError(f"source and shard positions differ for {key}")
            if not np.array_equal(arrays["edges"], store.array(key, "edges")):
                raise ValueError(f"source and shard edges differ for {key}")
            if not np.array_equal(
                arrays["node_type"].reshape(-1),
                np.asarray(store.array(key, "node_type")).reshape(-1),
            ):
                raise ValueError(f"source and shard node types differ for {key}")

            sample = store.tensor_sample(
                key, start_frame, step_stride=step_stride, device=device
            )
            weights = arrays["physical_cell_volume_weights"]
            for offset in range(valid_length):
                call_index = offset + 1
                reference_current_np = arrays["reference_currents"][offset]
                target_np = arrays["targets"][offset]
                rollout_current_np = arrays["rollout_currents"][offset]
                rollout_prediction_np = arrays["predictions"][offset]
                rollout_current = _state_tensor(rollout_current_np, device)
                reference_current = _state_tensor(reference_current_np, device)
                target_tensor = _state_tensor(target_np, device)
                same_current = bool(
                    np.array_equal(rollout_current_np, reference_current_np)
                )
                with torch.no_grad():
                    rollout_replay = model_call(model, sample, rollout_current)
                    teacher_prediction = (
                        rollout_replay
                        if same_current
                        else model_call(model, sample, reference_current)
                    )
                rollout_replay_np = rollout_replay[0].float().cpu().numpy()
                teacher_prediction_np = teacher_prediction[0].float().cpu().numpy()
                source_replay_max_absolute_error = float(
                    np.max(np.abs(rollout_replay_np - rollout_prediction_np))
                )
                interior, smooth, mask_contract = _reference_masks(
                    reference_current,
                    target_tensor,
                    sample,
                    shock_quantile=args.shock_quantile,
                    dilation_hops=args.shock_dilation_hops,
                )
                shock = interior & ~smooth

                rollout_prediction_value = np.asarray(
                    rollout_replay_np, dtype=np.float64
                )
                teacher_prediction_value = np.asarray(
                    teacher_prediction_np, dtype=np.float64
                )
                target_value = np.asarray(target_np, dtype=np.float64)
                rollout_current_value = np.asarray(rollout_current_np, dtype=np.float64)
                reference_current_value = np.asarray(
                    reference_current_np, dtype=np.float64
                )
                total = (rollout_prediction_value - target_value) / component_scale
                propagated = (
                    rollout_prediction_value - teacher_prediction_value
                ) / component_scale
                fresh = (teacher_prediction_value - target_value) / component_scale
                current_error = (
                    rollout_current_value - reference_current_value
                ) / component_scale
                total_highpass = node_highpass_field(total, arrays["edges"])
                propagated_highpass = node_highpass_field(propagated, arrays["edges"])
                fresh_highpass = node_highpass_field(fresh, arrays["edges"])
                current_highpass = node_highpass_field(current_error, arrays["edges"])

                regions = {
                    "interior_full": weighted_decomposition_metrics(
                        total, propagated, fresh, weights, mask=interior
                    ),
                    "smooth_full": weighted_decomposition_metrics(
                        total, propagated, fresh, weights, mask=smooth
                    ),
                    "shock_support_full": weighted_decomposition_metrics(
                        total, propagated, fresh, weights, mask=shock
                    ),
                    "smooth_highpass": weighted_decomposition_metrics(
                        total_highpass,
                        propagated_highpass,
                        fresh_highpass,
                        weights,
                        mask=smooth,
                    ),
                    "shock_support_highpass": weighted_decomposition_metrics(
                        total_highpass,
                        propagated_highpass,
                        fresh_highpass,
                        weights,
                        mask=shock,
                    ),
                }
                input_full_norm = _weighted_norm(current_error, weights, interior)
                input_highpass_norm = _weighted_norm(current_highpass, weights, smooth)
                row = {
                    "schema": ERROR_SOURCE_SCHEMA,
                    "trajectory": key,
                    "call_index": call_index,
                    "physical_target_time": float(
                        arrays["physical_target_times"][offset]
                    ),
                    "delta_t": delta_t,
                    "regions": regions,
                    "input_error": {
                        "interior_full_norm": input_full_norm,
                        "smooth_highpass_norm": input_highpass_norm,
                        "full_propagation_gain": _gain(
                            regions["interior_full"].get("propagated_norm"),
                            input_full_norm,
                        ),
                        "smooth_highpass_propagation_gain": _gain(
                            regions["smooth_highpass"].get("propagated_norm"),
                            input_highpass_norm,
                        ),
                    },
                    "mask_contract": mask_contract,
                    "source_replay": {
                        "same_current_as_reference": same_current,
                        "max_absolute_error": source_replay_max_absolute_error,
                        "tolerance": SOURCE_REPLAY_MAX_ABSOLUTE_TOLERANCE,
                    },
                    "admissibility": {
                        "rollout_prediction": raw_admissibility_summary(
                            rollout_replay_np, gamma=model.gamma
                        ),
                        "teacher_prediction": raw_admissibility_summary(
                            teacher_prediction_np, gamma=model.gamma
                        ),
                    },
                    "claim_boundary": (
                        "exact frozen-map error identity; reference current/target "
                        "masks are diagnostic only and never enter autonomous inference"
                    ),
                }
                rows.append(row)
                append_jsonl(rows_path, row)
            print(json.dumps({"trajectory": key, "rows": valid_length}, sort_keys=True))
    finally:
        store.close()

    selector = error_source_selector(rows)
    max_source_replay_error = max(
        float(row["source_replay"]["max_absolute_error"]) for row in rows
    )
    max_identity_residual = max(
        float(row["regions"][name][metric])
        for row in rows
        for name in ("interior_full", "smooth_highpass")
        for metric in (
            "relative_reconstruction_residual",
            "relative_energy_identity_residual",
        )
    )
    summary = {
        "schema": ERROR_SOURCE_SCHEMA,
        "status": "complete",
        "source": {
            "artifact_dir_name": args.source_dir.name,
            "summary_sha256": sha256_file(source_summary_path),
            "summary_schema": source_summary.get("schema"),
            "branch_sensitivity_schema": source_summary.get(
                "branch_gain_sensitivity", {}
            )
            .get("selector", {})
            .get("version"),
        },
        "checkpoint": {
            "sha256": checkpoint_digest,
            "config_digest": str(checkpoint["config_digest"]),
            "data_manifest_digest": store.manifest_digest,
            "boundary_mode": checkpoint["boundary_mode"],
            "raw_recurrence": bool(checkpoint["raw_recurrence"]),
        },
        "evaluation": {
            "device": str(device),
            "trajectory_keys": [str(record["trajectory"]) for record in records],
            "trajectory_count": len(records),
            "row_count": len(rows),
            "shock_quantile": args.shock_quantile,
            "shock_dilation_hops": args.shock_dilation_hops,
            "weights": "validated_physical_cell_volume_normalized",
            "recurrence": (
                "saved legal raw D052 rollout currents with same-process map "
                "replay; no intervention"
            ),
        },
        "decomposition_contract": {
            "identity": ("G(uhat_t)-u_(t+1) = [G(uhat_t)-G(u_t)] + [G(u_t)-u_(t+1)]"),
            "highpass": "linear self-plus-neighbor graph high-pass",
            "mask": "reference-current/target shock union; diagnostic only",
            "maximum_identity_relative_residual": max_identity_residual,
            "maximum_source_rollout_replay_absolute_error": (max_source_replay_error),
            "source_rollout_replay_absolute_tolerance": (
                SOURCE_REPLAY_MAX_ABSOLUTE_TOLERANCE
            ),
            "physical_conservation": (
                "not evaluated; physical volumes weight error attribution but do "
                "not turn a state residual into a flux or conservation guarantee"
            ),
        },
        "selector": selector,
        "artifact_schema": {
            "error_sources_jsonl": (
                "one row per trajectory/call with full, regional, and high-pass "
                "additive error attribution"
            )
        },
        "claim_boundary": {
            "verified": "frozen D044/D052 map behavior on the selected validation cohort",
            "plausible": "selector classification of propagated versus fresh defect",
            "unsupported": (
                "universal ripple causation, strength-OOD behavior, trained-method "
                "improvement, or physical flux/conservation"
            ),
        },
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary["selector"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
