#!/usr/bin/env python3
"""Evaluate retained D094 maps on exact and common propagated inputs.

For each ``n={128,256}``, the selected checkpoint first generates one frozen
H79 path.  Step 20,480, selected, and terminal checkpoints are then called on
the same exact-reference inputs and on that same selected-checkpoint path.
Candidate outputs never feed back during the comparison.  The diagnostic is
FP32, inference-only, and restricted to the 28 registered outside-selection
development trajectories.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.time_dependent_no.evaluate_pcno_bump_b1_c4 import (
    EXPECTED_CHECKPOINT_SOURCE_SET_DIGEST,
    EXPECTED_DATA_MANIFEST_DIGEST,
    EXPECTED_SPLIT_PARTITION_DIGEST,
    TRAJECTORY_COUNTS,
    discover_checkpoints,
)
from scripts.time_dependent_no.evaluate_pcno_bump_scaling_holdout import (
    _build_boundary_policies,
    _load_json,
    outside_selection_keys,
)
from utility.time_dependent_no.pcno_artifacts import (
    atomic_write_json,
    runtime_environment,
    sha256_file,
    write_csv,
    write_source_snapshot,
)
from utility.time_dependent_no.pcno_euler2d import (
    PCNOEuler2DShardStore,
    conservative_admissibility,
)
from utility.time_dependent_no.pcno_rollout import (
    build_bump_checkpoint_model,
    contract_forward_sample,
    load_bump_checkpoint,
)
from utility.time_dependent_no.pcno_runtime import (
    expand_homogeneous_sample,
    select_device,
)

SCHEMA = "d094_b1_c5_fixed_map_evaluation_v1"
ROW_SCHEMA = "d094_b1_c5_fixed_map_row_v1"
ARTIFACT_SCHEMA = "d094_b1_c5_fixed_map_artifacts_v1"
SPLIT_SCHEMA = "d094_bump_trajectory_scaling_split_v1"
CHECKPOINT_ROLES = ("matched_20480", "selected", "terminal")
EXPECTED_STEPS = {
    "matched_20480": 20_480,
    "selected": 38_400,
    "terminal": 40_960,
}
INPUT_VIEWS = ("exact_reference", "common_selected_path")
ENDPOINT_CALLS = (1, 5, 10, 20, 40, 60, 79)
CALL_WINDOWS = {
    "early_1_20": (1, 20),
    "middle_21_40": (21, 40),
    "late_41_60": (41, 60),
    "tail_61_79": (61, 79),
    "all_1_79": (1, 79),
}
EXPECTED_EXECUTION_IDS = ("fp32_1", "fp32_2")
MAX_EFFECT_FLOOR_FRACTION = 0.25
MAX_SELECTED_PATH_REPLAY_DRIFT_FRACTION = 1.0e-4
MAX_QUADRATIC_CLOSURE_RELATIVE = 1.0e-10
EXTRA_SOURCE_FILES = (
    "docs/time_dependent_no/D094_BUMP_SCALING_PREREGISTRATION.md",
    "docs/time_dependent_no/D094_BUMP_SCALING_SPLIT_MANIFEST.json",
    "scripts/time_dependent_no/evaluate_pcno_bump_scaling_holdout.py",
    "scripts/time_dependent_no/evaluate_pcno_bump_scaling_ladder.py",
    "scripts/time_dependent_no/evaluate_pcno_bump_b1_c4.py",
    "scripts/time_dependent_no/evaluate_pcno_bump_b1_c5_fixed_map.py",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=(
            REPO_ROOT / "docs/time_dependent_no/D094_BUMP_SCALING_SPLIT_MANIFEST.json"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-id", choices=EXPECTED_EXECUTION_IDS, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=20260823)
    return parser


def select_fixed_map_descriptors(
    descriptors: Sequence[Mapping[str, Any]],
) -> dict[int, dict[str, Mapping[str, Any]]]:
    """Select the exact three-map inventory for each trajectory count."""

    selected: dict[int, dict[str, Mapping[str, Any]]] = {
        count: {} for count in TRAJECTORY_COUNTS
    }
    for descriptor in descriptors:
        count = int(descriptor["trajectory_count"])
        role = str(descriptor["checkpoint_role"])
        if count not in selected or role not in CHECKPOINT_ROLES:
            continue
        if role in selected[count]:
            raise ValueError(f"duplicate fixed-map descriptor: n={count}, {role}")
        if int(descriptor["optimizer_step"]) != EXPECTED_STEPS[role]:
            raise ValueError(f"fixed-map step changed for n={count}, {role}")
        selected[count][role] = descriptor
    expected = set(CHECKPOINT_ROLES)
    if any(set(by_role) != expected for by_role in selected.values()):
        raise ValueError("fixed-map inventory must contain three roles per count")
    for count, by_role in selected.items():
        normalization_digests = {
            str(descriptor["normalization_digest"]) for descriptor in by_role.values()
        }
        config_digests = {
            str(descriptor["config_digest"]) for descriptor in by_role.values()
        }
        if len(normalization_digests) != 1 or len(config_digests) != 1:
            raise ValueError(
                f"n={count} fixed maps do not share model and normalization contracts"
            )
    return selected


def _metric_geometry(
    value: torch.Tensor,
    node_weights: torch.Tensor,
    node_mask: torch.Tensor,
    component_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if value.ndim != 3 or value.shape[0] != 1 or value.shape[-1] != 4:
        raise ValueError("map fields must have shape [1,N,4]")
    if node_weights.ndim != 3 or node_weights.shape[:2] != value.shape[:2]:
        raise ValueError("node weights must align with map fields")
    if node_mask.shape != value.shape[:2] + (1,):
        raise ValueError("node mask must align with map fields")
    if component_scale.numel() != 4:
        raise ValueError("component scale must contain four values")
    weights = node_weights.detach().to(dtype=torch.float64).sum(
        dim=-1, keepdim=True
    ) * node_mask.detach().to(dtype=torch.float64)
    scale = component_scale.detach().to(dtype=torch.float64).reshape(1, 1, 4)
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0.0).any()):
        raise ValueError("proxy weights must be finite and nonnegative")
    if not bool(torch.isfinite(scale).all()) or not bool((scale > 0.0).all()):
        raise ValueError("component scale must be finite and positive")
    weight_sum = float(weights.sum().detach().cpu())
    if weight_sum <= 0.0:
        raise ValueError("proxy weights must have positive mass")
    return weights, scale, node_mask, weight_sum * value.shape[-1]


def paired_map_metrics(
    *,
    candidate_prediction: torch.Tensor,
    selected_prediction: torch.Tensor,
    target: torch.Tensor,
    input_state: torch.Tensor,
    node_weights: torch.Tensor,
    node_mask: torch.Tensor,
    component_scale: torch.Tensor,
) -> dict[str, float | None]:
    """Compare two maps in one fixed proxy-weighted component geometry."""

    tensors = {
        "candidate_prediction": candidate_prediction,
        "selected_prediction": selected_prediction,
        "target": target,
        "input_state": input_state,
    }
    shapes = {tuple(value.shape) for value in tensors.values()}
    if len(shapes) != 1:
        raise ValueError("candidate, selected, target, and input must share a shape")
    if any(not bool(torch.isfinite(value).all()) for value in tensors.values()):
        raise ValueError("fixed-map metrics require finite tensors")
    weights, scale, _, normalization = _metric_geometry(
        candidate_prediction, node_weights, node_mask, component_scale
    )
    values = {
        name: value.detach().to(dtype=torch.float64) for name, value in tensors.items()
    }
    candidate_error = (values["candidate_prediction"] - values["target"]) / scale
    selected_error = (values["selected_prediction"] - values["target"]) / scale
    drift = (values["candidate_prediction"] - values["selected_prediction"]) / scale
    candidate_residual = (
        values["candidate_prediction"] - values["input_state"]
    ) / scale
    selected_residual = (values["selected_prediction"] - values["input_state"]) / scale
    target_scaled = values["target"] / scale

    def energy(field: torch.Tensor) -> torch.Tensor:
        return (weights * field.square()).sum() / normalization

    candidate_error_mse_t = energy(candidate_error)
    selected_error_mse_t = energy(selected_error)
    drift_mse_t = energy(drift)
    candidate_residual_mse_t = energy(candidate_residual)
    selected_residual_mse_t = energy(selected_residual)
    target_energy_t = energy(target_scaled)
    cross_t = (weights * selected_error * drift).sum() / normalization
    (
        candidate_error_mse,
        selected_error_mse,
        drift_mse,
        candidate_residual_mse,
        selected_residual_mse,
        target_energy,
        cross,
    ) = (
        torch.stack(
            (
                candidate_error_mse_t,
                selected_error_mse_t,
                drift_mse_t,
                candidate_residual_mse_t,
                selected_residual_mse_t,
                target_energy_t,
                cross_t,
            )
        )
        .detach()
        .cpu()
        .tolist()
    )
    quadratic_rhs = selected_error_mse + 2.0 * cross + drift_mse
    closure_scale = max(
        abs(candidate_error_mse),
        abs(selected_error_mse) + 2.0 * abs(cross) + abs(drift_mse),
        torch.finfo(torch.float64).tiny,
    )
    drift_rms = math.sqrt(max(drift_mse, 0.0))
    selected_residual_rms = math.sqrt(max(selected_residual_mse, 0.0))
    error_drift_denominator = math.sqrt(max(selected_error_mse * drift_mse, 0.0))
    selected_error_rms = math.sqrt(max(selected_error_mse, 0.0))
    return {
        "candidate_state_relative_l2": math.sqrt(
            candidate_error_mse / max(target_energy, torch.finfo(torch.float64).tiny)
        ),
        "selected_state_relative_l2": math.sqrt(
            selected_error_mse / max(target_energy, torch.finfo(torch.float64).tiny)
        ),
        "candidate_minus_selected_relative_l2": math.sqrt(
            candidate_error_mse / max(target_energy, torch.finfo(torch.float64).tiny)
        )
        - math.sqrt(
            selected_error_mse / max(target_energy, torch.finfo(torch.float64).tiny)
        ),
        "candidate_over_selected_relative_l2": (
            None
            if selected_error_rms <= 1.0e-30
            else math.sqrt(max(candidate_error_mse, 0.0)) / selected_error_rms
        ),
        "candidate_error_scaled_mse": candidate_error_mse,
        "selected_error_scaled_mse": selected_error_mse,
        "candidate_minus_selected_scaled_mse": (
            candidate_error_mse - selected_error_mse
        ),
        "drift_scaled_mse": drift_mse,
        "drift_scaled_rms": drift_rms,
        "candidate_residual_scaled_rms": math.sqrt(max(candidate_residual_mse, 0.0)),
        "selected_residual_scaled_rms": selected_residual_rms,
        "drift_over_selected_residual": (
            None
            if selected_residual_rms <= 1.0e-30
            else drift_rms / selected_residual_rms
        ),
        "selected_error_drift_cross_scaled": cross,
        "selected_error_drift_cosine": (
            None
            if error_drift_denominator <= 1.0e-30
            else cross / error_drift_denominator
        ),
        "quadratic_closure_relative": abs(candidate_error_mse - quadratic_rhs)
        / closure_scale,
    }


def _admissibility_record(
    state: torch.Tensor, *, gamma: float
) -> dict[str, bool | float]:
    diagnostics = conservative_admissibility(state, gamma=gamma)
    values = (
        torch.stack(
            (
                diagnostics["finite_components"].all().to(dtype=torch.float64),
                diagnostics["admissible"].all().to(dtype=torch.float64),
                diagnostics["density"].min().to(dtype=torch.float64),
                diagnostics["internal_energy"].min().to(dtype=torch.float64),
                diagnostics["pressure"].min().to(dtype=torch.float64),
            )
        )
        .detach()
        .cpu()
        .tolist()
    )
    return {
        "all_finite": bool(values[0]),
        "all_admissible": bool(values[1]),
        "minimum_density": float(values[2]),
        "minimum_internal_energy": float(values[3]),
        "minimum_pressure": float(values[4]),
    }


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    values = [
        float(row[field])
        for row in rows
        if row.get(field) is not None and math.isfinite(float(row[field]))
    ]
    return None if not values else float(np.mean(values))


def _aggregate_scope(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate an empty fixed-map scope")
    drift_mse = _mean(rows, "drift_scaled_mse")
    return {
        "row_count": len(rows),
        "candidate_state_relative_l2_mean": _mean(rows, "candidate_state_relative_l2"),
        "selected_state_relative_l2_mean": _mean(rows, "selected_state_relative_l2"),
        "candidate_minus_selected_relative_l2_mean": _mean(
            rows, "candidate_minus_selected_relative_l2"
        ),
        "candidate_over_selected_relative_l2_mean": _mean(
            rows, "candidate_over_selected_relative_l2"
        ),
        "candidate_error_scaled_mse_mean": _mean(rows, "candidate_error_scaled_mse"),
        "selected_error_scaled_mse_mean": _mean(rows, "selected_error_scaled_mse"),
        "candidate_minus_selected_scaled_mse_mean": _mean(
            rows, "candidate_minus_selected_scaled_mse"
        ),
        "pooled_drift_scaled_rms": (
            None if drift_mse is None else math.sqrt(max(drift_mse, 0.0))
        ),
        "drift_scaled_rms_mean": _mean(rows, "drift_scaled_rms"),
        "drift_over_selected_residual_mean": _mean(
            rows, "drift_over_selected_residual"
        ),
        "selected_error_drift_cross_scaled_mean": _mean(
            rows, "selected_error_drift_cross_scaled"
        ),
        "selected_error_drift_cosine_mean": _mean(rows, "selected_error_drift_cosine"),
        "quadratic_closure_relative_max": max(
            float(row["quadratic_closure_relative"]) for row in rows
        ),
        "candidate_admissible_call_fraction": float(
            np.mean(
                [bool(row["candidate_admissibility"]["all_admissible"]) for row in rows]
            )
        ),
        "selected_admissible_call_fraction": float(
            np.mean(
                [bool(row["selected_admissibility"]["all_admissible"]) for row in rows]
            )
        ),
    }


def aggregate_map_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Create fixed endpoint/window curves with equal case-call weighting."""

    result: dict[str, Any] = {}
    for count in TRAJECTORY_COUNTS:
        count_result: dict[str, Any] = {}
        for role in CHECKPOINT_ROLES:
            role_result: dict[str, Any] = {}
            for view in INPUT_VIEWS:
                selected = [
                    row
                    for row in rows
                    if int(row["trajectory_count"]) == count
                    and row["checkpoint_role"] == role
                    and row["input_view"] == view
                ]
                endpoints = {
                    str(call): _aggregate_scope(
                        [row for row in selected if int(row["call_index"]) == call]
                    )
                    for call in ENDPOINT_CALLS
                }
                curve = {
                    str(call): _aggregate_scope(
                        [row for row in selected if int(row["call_index"]) == call]
                    )
                    for call in range(1, 80)
                }
                windows = {
                    name: _aggregate_scope(
                        [
                            row
                            for row in selected
                            if first <= int(row["call_index"]) <= last
                        ]
                    )
                    for name, (first, last) in CALL_WINDOWS.items()
                }
                role_result[view] = {
                    "endpoints": endpoints,
                    "windows": windows,
                    "curve": curve,
                }
            count_result[role] = role_result
        result[str(count)] = count_result
    return result


def _aggregate_csv_rows(aggregates: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for count in TRAJECTORY_COUNTS:
        for role in CHECKPOINT_ROLES:
            for view in INPUT_VIEWS:
                cell = aggregates[str(count)][role][view]
                for scope_kind in ("endpoints", "windows"):
                    for scope, metrics in cell[scope_kind].items():
                        rows.append(
                            {
                                "trajectory_count": count,
                                "checkpoint_role": role,
                                "optimizer_step": EXPECTED_STEPS[role],
                                "input_view": view,
                                "scope_kind": scope_kind,
                                "scope": scope,
                                **metrics,
                            }
                        )
    return rows


def _validate_checkpoint_bindings(
    checkpoints: Mapping[int, Mapping[str, Mapping[str, Any]]],
    *,
    outside_keys: Sequence[str],
    policy_metadata: Mapping[str, Mapping[str, Any]],
    store: PCNOEuler2DShardStore,
) -> None:
    for count, by_role in checkpoints.items():
        normalizers = set()
        for role, checkpoint in by_role.items():
            if checkpoint.get("data_manifest_digest") != store.manifest_digest:
                raise ValueError(f"n={count} {role} data manifest changed")
            if checkpoint.get("source_snapshot", {}).get("source_set_digest") != (
                EXPECTED_CHECKPOINT_SOURCE_SET_DIGEST
            ):
                raise ValueError(f"n={count} {role} source-set digest changed")
            if int(checkpoint.get("step_stride", -1)) != 1:
                raise ValueError(f"n={count} {role} learned stride changed")
            if checkpoint.get("test_keys") not in (None, []):
                raise ValueError(f"n={count} {role} exposes a sealed/test population")
            policy_digests = checkpoint.get("boundary_contract", {}).get(
                "policy_digests", {}
            )
            if any(
                str(policy_digests.get(key))
                != str(policy_metadata[key]["policy_digest"])
                for key in outside_keys
            ):
                raise ValueError(f"n={count} {role} boundary policy changed")
            normalizers.add(str(checkpoint["normalization_digest"]))
        if len(normalizers) != 1:
            raise ValueError(f"n={count} fixed maps do not share one normalizer")


@torch.inference_mode()
def _evaluate_count(
    *,
    trajectory_count: int,
    descriptors: Mapping[str, Mapping[str, Any]],
    checkpoints: Mapping[str, Mapping[str, Any]],
    store: PCNOEuler2DShardStore,
    outside_keys: Sequence[str],
    policies: Mapping[str, Mapping[str, Any]],
    device: torch.device,
    execution_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    models = {
        role: build_bump_checkpoint_model(checkpoint, device)
        for role, checkpoint in checkpoints.items()
    }
    scales = {
        role: model.state_scale.detach().float() for role, model in models.items()
    }
    if any(
        not torch.equal(scales[role], scales["selected"]) for role in CHECKPOINT_ROLES
    ):
        raise ValueError(f"n={trajectory_count} map state scales differ")
    gammas = {float(model.gamma) for model in models.values()}
    if len(gammas) != 1:
        raise ValueError(f"n={trajectory_count} map gamma values differ")
    gamma = next(iter(gammas))
    rows: list[dict[str, Any]] = []
    trajectory_records = []
    try:
        for key in outside_keys:
            states_np = np.array(store.states(key)[:80], dtype=np.float32, copy=True)
            if (
                states_np.shape[0] != 80
                or states_np.ndim != 3
                or states_np.shape[-1] != 4
                or not np.isfinite(states_np).all()
            ):
                raise ValueError(f"trajectory {key} lacks a finite H79 reference")
            states = torch.as_tensor(states_np, dtype=torch.float32, device=device)
            sample = store.tensor_sample(key, 0, step_stride=1, device=device)
            batch_sample = expand_homogeneous_sample(sample, 2)
            policy = policies[key]

            common_inputs: list[torch.Tensor] = []
            common_next: list[torch.Tensor] = []
            current = states[0:1]
            for call_index in range(1, 80):
                common_inputs.append(current)
                proposal, _, _ = contract_forward_sample(
                    models["selected"], sample, current, boundary_policy=policy
                )
                if not bool(torch.isfinite(proposal).all()):
                    raise ValueError(
                        f"n={trajectory_count} trajectory {key} selected path "
                        f"is nonfinite at call {call_index}"
                    )
                common_next.append(proposal)
                current = proposal

            call1_inputs_equal = torch.equal(common_inputs[0], states[0:1])
            recurrence_inputs_equal = all(
                torch.equal(common_inputs[index], common_next[index - 1])
                for index in range(1, 79)
            )
            if not call1_inputs_equal or not recurrence_inputs_equal:
                raise RuntimeError("selected-path recurrence bookkeeping changed")

            for call_index in range(1, 80):
                exact_input = states[call_index - 1 : call_index]
                common_input = common_inputs[call_index - 1]
                target = states[call_index : call_index + 1]
                input_batch = torch.cat((exact_input, common_input), dim=0)
                predictions = {}
                for role in CHECKPOINT_ROLES:
                    prediction, _, _ = contract_forward_sample(
                        models[role],
                        batch_sample,
                        input_batch,
                        boundary_policy=policy,
                    )
                    if not bool(torch.isfinite(prediction).all()):
                        raise ValueError(
                            f"n={trajectory_count} {role} trajectory {key} "
                            f"is nonfinite at call {call_index}"
                        )
                    predictions[role] = prediction

                selected_exact = predictions["selected"][0:1]
                selected_common = common_next[call_index - 1]
                candidate_admissibility = {
                    role: {
                        "exact_reference": _admissibility_record(
                            prediction[0:1], gamma=gamma
                        ),
                        "common_selected_path": _admissibility_record(
                            prediction[1:2], gamma=gamma
                        ),
                    }
                    for role, prediction in predictions.items()
                }
                selected_admissibility = {
                    "exact_reference": candidate_admissibility["selected"][
                        "exact_reference"
                    ],
                    "common_selected_path": _admissibility_record(
                        selected_common, gamma=gamma
                    ),
                }
                for role in CHECKPOINT_ROLES:
                    views = (
                        (
                            "exact_reference",
                            predictions[role][0:1],
                            selected_exact,
                            exact_input,
                        ),
                        (
                            "common_selected_path",
                            predictions[role][1:2],
                            selected_common,
                            common_input,
                        ),
                    )
                    for view, candidate, selected_prediction, input_state in views:
                        metrics = paired_map_metrics(
                            candidate_prediction=candidate,
                            selected_prediction=selected_prediction,
                            target=target,
                            input_state=input_state,
                            node_weights=sample["node_weights"],
                            node_mask=sample["node_mask"],
                            component_scale=scales["selected"],
                        )
                        row = {
                            "schema": ROW_SCHEMA,
                            "execution_id": execution_id,
                            "trajectory_count": trajectory_count,
                            "checkpoint_role": role,
                            "optimizer_step": EXPECTED_STEPS[role],
                            "checkpoint_sha256": str(
                                descriptors[role]["checkpoint_sha256"]
                            ),
                            "trajectory": str(key),
                            "call_index": call_index,
                            "input_view": view,
                            **metrics,
                            "candidate_admissibility": candidate_admissibility[role][
                                view
                            ],
                            "selected_admissibility": selected_admissibility[view],
                        }
                        rows.append(row)
            trajectory_records.append(
                {
                    "trajectory": str(key),
                    "completed_calls": 79,
                    "call1_exact_common_inputs_equal": call1_inputs_equal,
                    "selected_path_recurrence_exact": recurrence_inputs_equal,
                }
            )
            print(
                json.dumps(
                    {
                        "execution_id": execution_id,
                        "trajectory_count": trajectory_count,
                        "trajectory": str(key),
                        "completed_calls": 79,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows, {
        "trajectory_count": trajectory_count,
        "trajectories": trajectory_records,
    }


def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise ValueError("B1-C5-B output directory must not already exist")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    descriptors = select_fixed_map_descriptors(discover_checkpoints(args.run_root))
    split_manifest = _load_json(args.split_manifest)
    if (
        split_manifest.get("schema") != SPLIT_SCHEMA
        or split_manifest.get("partition_digest") != EXPECTED_SPLIT_PARTITION_DIGEST
    ):
        raise ValueError("D094 split manifest changed")
    flat_descriptors = [
        descriptors[count][role]
        for count in TRAJECTORY_COUNTS
        for role in CHECKPOINT_ROLES
    ]
    validation_keys, selection_keys, outside_keys = outside_selection_keys(
        split_manifest, [descriptor["split"] for descriptor in flat_descriptors]
    )
    device = select_device(args.device)
    args.output_dir.mkdir(parents=True)
    source_snapshot = write_source_snapshot(
        args.output_dir, extra_source_files=EXTRA_SOURCE_FILES
    )
    store = PCNOEuler2DShardStore(args.data_dir)
    try:
        if store.manifest_digest != EXPECTED_DATA_MANIFEST_DIGEST:
            raise ValueError("B1-C5-B evaluation data manifest changed")
        if not set(outside_keys) <= set(store.keys):
            raise ValueError("outside-selection keys are absent from the shard store")
        checkpoint_payloads = {
            count: {
                role: load_bump_checkpoint(Path(descriptors[count][role]["checkpoint"]))
                for role in CHECKPOINT_ROLES
            }
            for count in TRAJECTORY_COUNTS
        }
        reference_checkpoint = checkpoint_payloads[TRAJECTORY_COUNTS[0]]["selected"]
        policies, policy_metadata = _build_boundary_policies(
            store, outside_keys, reference_checkpoint, device
        )
        _validate_checkpoint_bindings(
            checkpoint_payloads,
            outside_keys=outside_keys,
            policy_metadata=policy_metadata,
            store=store,
        )
        rows: list[dict[str, Any]] = []
        trajectory_records = []
        for count in TRAJECTORY_COUNTS:
            count_rows, count_record = _evaluate_count(
                trajectory_count=count,
                descriptors=descriptors[count],
                checkpoints=checkpoint_payloads[count],
                store=store,
                outside_keys=outside_keys,
                policies=policies,
                device=device,
                execution_id=args.execution_id,
            )
            rows.extend(count_rows)
            trajectory_records.append(count_record)
    finally:
        store.close()

    expected_rows = (
        len(TRAJECTORY_COUNTS)
        * len(CHECKPOINT_ROLES)
        * len(outside_keys)
        * 79
        * len(INPUT_VIEWS)
    )
    selected_common_rows = [
        row
        for row in rows
        if row["checkpoint_role"] == "selected"
        and row["input_view"] == "common_selected_path"
    ]
    replay_fractions = [
        float(row["drift_over_selected_residual"])
        for row in selected_common_rows
        if row["drift_over_selected_residual"] is not None
    ]
    replay_drift_mse = np.mean(
        [float(row["drift_scaled_mse"]) for row in selected_common_rows]
    )
    replay_residual_mse = np.mean(
        [
            float(row["selected_residual_scaled_rms"]) ** 2
            for row in selected_common_rows
        ]
    )
    replay_fraction = (
        math.inf
        if replay_residual_mse <= 0.0
        else math.sqrt(max(replay_drift_mse, 0.0) / replay_residual_mse)
    )
    maximum_per_row_replay_fraction = max(replay_fractions, default=math.inf)
    maximum_quadratic_closure = max(
        (float(row["quadratic_closure_relative"]) for row in rows), default=math.inf
    )
    contract_checks = {
        "outside_population_has_28_unique_cases": len(outside_keys) == 28
        and len(set(outside_keys)) == 28,
        "historical_test_population_not_accessed": True,
        "checkpoint_reselection_not_performed": True,
        "exact_three_maps_per_count": len(flat_descriptors) == 6,
        "all_rows_present": len(rows) == expected_rows,
        "all_candidate_outputs_finite": all(
            bool(row["candidate_admissibility"]["all_finite"]) for row in rows
        ),
        "selected_path_replay_below_ceiling": replay_fraction
        <= MAX_SELECTED_PATH_REPLAY_DRIFT_FRACTION,
        "quadratic_identity_closes": maximum_quadratic_closure
        <= MAX_QUADRATIC_CLOSURE_RELATIVE,
        "selected_paths_complete": all(
            record["completed_calls"] == 79
            and record["call1_exact_common_inputs_equal"]
            and record["selected_path_recurrence_exact"]
            for count_record in trajectory_records
            for record in count_record["trajectories"]
        ),
    }
    if not all(contract_checks.values()):
        raise RuntimeError(f"B1-C5-B contract checks failed: {contract_checks}")
    aggregates = aggregate_map_rows(rows)

    rows_path = args.output_dir / "map_response.jsonl"
    with rows_path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    write_csv(
        args.output_dir / "aggregate_summary.csv", _aggregate_csv_rows(aggregates)
    )

    checkpoint_bindings = {
        str(count): {
            role: {
                "optimizer_step": EXPECTED_STEPS[role],
                "checkpoint_sha256": str(descriptors[count][role]["checkpoint_sha256"]),
                "config_digest": str(descriptors[count][role]["config_digest"]),
                "normalization_digest": str(
                    descriptors[count][role]["normalization_digest"]
                ),
            }
            for role in CHECKPOINT_ROLES
        }
        for count in TRAJECTORY_COUNTS
    }
    summary = {
        "schema": SCHEMA,
        "status": "complete",
        "execution_id": args.execution_id,
        "scientific_scope": (
            "single_seed_fixed_checkpoint_map_response_on_exact_and_common_"
            "selected_checkpoint_inputs"
        ),
        "historical_test_population_accessed": False,
        "checkpoint_reselection_performed": False,
        "new_training_performed": False,
        "precision": "float32_no_autocast",
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "split_partition_digest": split_manifest["partition_digest"],
        "data_manifest_sha256": sha256_file(args.data_dir / "manifest.json"),
        "data_manifest_digest": store.manifest_digest,
        "checkpoint_source_set_digest": EXPECTED_CHECKPOINT_SOURCE_SET_DIGEST,
        "evaluator_source_snapshot": source_snapshot,
        "checkpoint_bindings": checkpoint_bindings,
        "validation_keys": validation_keys,
        "selection_keys": selection_keys,
        "outside_selection_keys": outside_keys,
        "boundary_policy_digests": {
            key: record["policy_digest"] for key, record in policy_metadata.items()
        },
        "rollout_horizon": 79,
        "input_views": {
            "exact_reference": (
                "each checkpoint receives exact U_t and is scored against U_(t+1)"
            ),
            "common_selected_path": (
                "each checkpoint receives the same Uhat_t path generated once by "
                "that count's selected checkpoint; candidate outputs never feed back"
            ),
        },
        "learned_residual_definition": "F_c(x) = G_c(x) - x after denormalization",
        "functional_drift_definition": (
            "F_c(x)-F_selected(x)=G_c(x)-G_selected(x) on identical x"
        ),
        "node_measure": "reconstructed bump proxy weights; not physical volumes",
        "primary_directional_rule": {
            "comparison": "terminal minus selected on common_selected_path",
            "calls": [20, 79],
            "expected_crossover": {
                "20": "negative candidate-minus-selected relative L2",
                "79": "positive candidate-minus-selected relative L2",
            },
            "required_counts": list(TRAJECTORY_COUNTS),
            "fresh_process_executions": list(EXPECTED_EXECUTION_IDS),
            "effect_and_drift_floor_fraction_ceiling": (MAX_EFFECT_FLOOR_FRACTION),
            "selected_path_replay_drift_fraction_ceiling": (
                MAX_SELECTED_PATH_REPLAY_DRIFT_FRACTION
            ),
            "classification": (
                "resolved functional-map crossover only if both executions and "
                "both counts agree on early-help/tail-harm and every paired effect "
                "and drift range is at most 25% of its smaller magnitude"
            ),
        },
        "contract_checks": contract_checks,
        "contract_complete": all(contract_checks.values()),
        "selected_path_replay_drift_fraction": replay_fraction,
        "maximum_per_row_selected_path_replay_drift_fraction": (
            maximum_per_row_replay_fraction
        ),
        "maximum_quadratic_closure_relative": maximum_quadratic_closure,
        "row_count": len(rows),
        "trajectory_records": trajectory_records,
        "aggregates": aggregates,
        "evaluation_numerics": {
            "amp": "none",
            "deterministic_algorithms_enabled": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        },
        "runtime_environment": runtime_environment(device),
        "claims_not_supported": [
            "hidden-feature or representation equivalence",
            "causal attribution to data count, optimizer, schedule, or architecture",
            "optimization or representation convergence",
            "multi-seed scaling or capacity conclusions",
            "physical conservation from bump proxy weights",
            "historical test performance",
        ],
    }
    atomic_write_json(args.output_dir / "summary.json", summary)
    artifact_files = ("summary.json", "map_response.jsonl", "aggregate_summary.csv")
    artifact_manifest = {
        "schema": ARTIFACT_SCHEMA,
        "execution_id": args.execution_id,
        "historical_test_population_accessed": False,
        "files": {
            name: {
                "bytes": (args.output_dir / name).stat().st_size,
                "sha256": sha256_file(args.output_dir / name),
            }
            for name in artifact_files
        },
    }
    atomic_write_json(args.output_dir / "artifact_manifest.json", artifact_manifest)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    summary = run(argv)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "execution_id": summary["execution_id"],
                "row_count": summary["row_count"],
                "contract_complete": summary["contract_complete"],
                "selected_path_replay_drift_fraction": summary[
                    "selected_path_replay_drift_fraction"
                ],
                "historical_test_population_accessed": False,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
