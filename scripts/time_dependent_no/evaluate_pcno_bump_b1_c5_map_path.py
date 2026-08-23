#!/usr/bin/env python3
"""Evaluate the registered D094 B1-C5-C symmetric map--path square.

For each ``n={128,256}``, retained selected and terminal checkpoints generate
their own H79 paths from the same exact frame zero.  Both maps are evaluated on
both paths at every call.  Cross-path outputs and batched numerical replays
never feed back.  The diagnostic is FP32, inference-only, and restricted to the
28 registered outside-selection development trajectories.
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
from scripts.time_dependent_no.evaluate_pcno_bump_b1_c5_fixed_map import (
    CALL_WINDOWS,
    ENDPOINT_CALLS,
    EXPECTED_EXECUTION_IDS,
    EXPECTED_STEPS,
    MAX_SELECTED_PATH_REPLAY_DRIFT_FRACTION,
    SPLIT_SCHEMA,
    _admissibility_record,
    _metric_geometry,
    _validate_checkpoint_bindings,
    select_fixed_map_descriptors,
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
from utility.time_dependent_no.pcno_euler2d import PCNOEuler2DShardStore
from utility.time_dependent_no.pcno_rollout import (
    build_bump_checkpoint_model,
    contract_forward_sample,
    load_bump_checkpoint,
)
from utility.time_dependent_no.pcno_runtime import (
    expand_homogeneous_sample,
    select_device,
)

SCHEMA = "d094_b1_c5_map_path_evaluation_v1"
ROW_SCHEMA = "d094_b1_c5_map_path_row_v1"
ARTIFACT_SCHEMA = "d094_b1_c5_map_path_artifacts_v1"
CHECKPOINT_ROLES = ("selected", "terminal")
OUTPUT_LABELS = ("ss", "ts", "st", "tt")
MAX_SCALAR_CLOSURE_RELATIVE = 1.0e-12
MAX_OUTPUT_CLOSURE_RELATIVE = 1.0e-10
EXTRA_SOURCE_FILES = (
    "docs/time_dependent_no/D094_BUMP_SCALING_PREREGISTRATION.md",
    "docs/time_dependent_no/D094_BUMP_SCALING_SPLIT_MANIFEST.json",
    "scripts/time_dependent_no/evaluate_pcno_bump_scaling_holdout.py",
    "scripts/time_dependent_no/evaluate_pcno_bump_scaling_ladder.py",
    "scripts/time_dependent_no/evaluate_pcno_bump_b1_c4.py",
    "scripts/time_dependent_no/evaluate_pcno_bump_b1_c5_fixed_map.py",
    "scripts/time_dependent_no/evaluate_pcno_bump_b1_c5_map_path.py",
    "scripts/time_dependent_no/analyze_pcno_bump_b1_c5_map_path.py",
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


def four_way_map_path_metrics(
    *,
    predictions: Mapping[str, torch.Tensor],
    selected_input: torch.Tensor,
    terminal_input: torch.Tensor,
    target: torch.Tensor,
    node_weights: torch.Tensor,
    node_mask: torch.Tensor,
    component_scale: torch.Tensor,
) -> dict[str, float]:
    """Close scalar error and output-space identities for one map--path square."""

    if set(predictions) != set(OUTPUT_LABELS):
        raise ValueError("map--path predictions must contain ss, ts, st, and tt")
    tensors = {
        **predictions,
        "selected_input": selected_input,
        "terminal_input": terminal_input,
        "target": target,
    }
    shapes = {tuple(value.shape) for value in tensors.values()}
    if len(shapes) != 1:
        raise ValueError("all map--path tensors must share one shape")
    if any(not bool(torch.isfinite(value).all()) for value in tensors.values()):
        raise ValueError("map--path metrics require finite tensors")
    weights, scale, _, normalization = _metric_geometry(
        predictions["ss"], node_weights, node_mask, component_scale
    )
    values = {
        name: value.detach().to(dtype=torch.float64) for name, value in tensors.items()
    }

    def energy(field: torch.Tensor) -> torch.Tensor:
        return (weights * field.square()).sum() / normalization

    target_scaled = values["target"] / scale
    target_energy = float(energy(target_scaled).detach().cpu())
    error_energy = {
        label: float(energy((values[label] - values["target"]) / scale).detach().cpu())
        for label in OUTPUT_LABELS
    }
    target_floor = max(target_energy, torch.finfo(torch.float64).tiny)
    errors = {
        label: math.sqrt(max(error_energy[label], 0.0) / target_floor)
        for label in OUTPUT_LABELS
    }

    map_selected = errors["ts"] - errors["ss"]
    map_terminal = errors["tt"] - errors["st"]
    path_selected = errors["st"] - errors["ss"]
    path_terminal = errors["tt"] - errors["ts"]
    interaction = map_terminal - map_selected
    interaction_from_paths = path_terminal - path_selected
    total = errors["tt"] - errors["ss"]
    zero_interaction = map_selected + path_selected
    scalar_residuals = (
        total - (map_selected + path_selected + interaction),
        total - (map_selected + path_terminal),
        total - (path_selected + map_terminal),
        interaction - interaction_from_paths,
    )
    scalar_scale = max(
        abs(total),
        abs(map_selected) + abs(path_selected) + abs(interaction),
        abs(map_selected) + abs(path_terminal),
        abs(path_selected) + abs(map_terminal),
        torch.finfo(torch.float64).tiny,
    )

    output_fields = {
        "map_selected": (values["ts"] - values["ss"]) / scale,
        "path_selected": (values["st"] - values["ss"]) / scale,
        "interaction": (values["tt"] - values["ts"] - values["st"] + values["ss"])
        / scale,
        "total": (values["tt"] - values["ss"]) / scale,
        "input_path": (values["terminal_input"] - values["selected_input"]) / scale,
    }
    output_energy = {
        name: float(energy(field).detach().cpu())
        for name, field in output_fields.items()
    }
    output_rms = {
        name: math.sqrt(max(value, 0.0)) for name, value in output_energy.items()
    }
    output_closure = output_fields["total"] - (
        output_fields["map_selected"]
        + output_fields["path_selected"]
        + output_fields["interaction"]
    )
    output_closure_rms = math.sqrt(
        max(float(energy(output_closure).detach().cpu()), 0.0)
    )
    output_scale = max(
        output_rms["total"],
        output_rms["map_selected"]
        + output_rms["path_selected"]
        + output_rms["interaction"],
        torch.finfo(torch.float64).tiny,
    )
    return {
        **{f"error_{label}": errors[label] for label in OUTPUT_LABELS},
        "map_effect_selected_path": map_selected,
        "map_effect_terminal_path": map_terminal,
        "path_effect_selected_map": path_selected,
        "path_effect_terminal_map": path_terminal,
        "map_path_interaction": interaction,
        "autonomous_total_effect": total,
        "zero_interaction_counterfactual_effect": zero_interaction,
        "maximum_scalar_closure_relative": max(abs(value) for value in scalar_residuals)
        / scalar_scale,
        **{f"output_{name}_scaled_mse": output_energy[name] for name in output_energy},
        **{f"output_{name}_scaled_rms": output_rms[name] for name in output_rms},
        "output_closure_relative": output_closure_rms / output_scale,
    }


def replay_metrics(
    *,
    replay_prediction: torch.Tensor,
    primary_prediction: torch.Tensor,
    input_state: torch.Tensor,
    node_weights: torch.Tensor,
    node_mask: torch.Tensor,
    component_scale: torch.Tensor,
) -> dict[str, float | None]:
    """Measure batch-context replay drift without changing either primary path."""

    tensors = (replay_prediction, primary_prediction, input_state)
    if len({tuple(value.shape) for value in tensors}) != 1:
        raise ValueError("replay tensors must share one shape")
    if any(not bool(torch.isfinite(value).all()) for value in tensors):
        raise ValueError("replay metrics require finite tensors")
    weights, scale, _, normalization = _metric_geometry(
        primary_prediction, node_weights, node_mask, component_scale
    )

    def energy(field: torch.Tensor) -> float:
        return float(((weights * field.square()).sum() / normalization).cpu())

    primary = primary_prediction.detach().to(dtype=torch.float64)
    replay = replay_prediction.detach().to(dtype=torch.float64)
    input_value = input_state.detach().to(dtype=torch.float64)
    drift_mse = energy((replay - primary) / scale)
    residual_mse = energy((primary - input_value) / scale)
    fraction = (
        None if residual_mse <= 0.0 else math.sqrt(max(drift_mse, 0.0) / residual_mse)
    )
    return {
        "replay_drift_scaled_mse": drift_mse,
        "owner_residual_scaled_mse": residual_mse,
        "replay_drift_over_owner_residual": fraction,
    }


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    values = [
        float(row[field])
        for row in rows
        if row.get(field) is not None and math.isfinite(float(row[field]))
    ]
    return None if not values else float(np.mean(values))


AGGREGATE_FIELDS = (
    *(f"error_{label}" for label in OUTPUT_LABELS),
    "map_effect_selected_path",
    "map_effect_terminal_path",
    "path_effect_selected_map",
    "path_effect_terminal_map",
    "map_path_interaction",
    "autonomous_total_effect",
    "zero_interaction_counterfactual_effect",
    "output_map_selected_scaled_rms",
    "output_path_selected_scaled_rms",
    "output_interaction_scaled_rms",
    "output_total_scaled_rms",
    "output_input_path_scaled_rms",
)


def _aggregate_scope(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate an empty map--path scope")
    return {
        "row_count": len(rows),
        **{f"{field}_mean": _mean(rows, field) for field in AGGREGATE_FIELDS},
        "maximum_scalar_closure_relative": max(
            float(row["maximum_scalar_closure_relative"]) for row in rows
        ),
        "maximum_output_closure_relative": max(
            float(row["output_closure_relative"]) for row in rows
        ),
        **{
            f"{label}_admissible_call_fraction": float(
                np.mean(
                    [
                        bool(row["admissibility"][label]["all_admissible"])
                        for row in rows
                    ]
                )
            )
            for label in OUTPUT_LABELS
        },
    }


def aggregate_map_path_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for count in TRAJECTORY_COUNTS:
        selected = [row for row in rows if int(row["trajectory_count"]) == count]
        result[str(count)] = {
            "endpoints": {
                str(call): _aggregate_scope(
                    [row for row in selected if int(row["call_index"]) == call]
                )
                for call in ENDPOINT_CALLS
            },
            "windows": {
                name: _aggregate_scope(
                    [row for row in selected if first <= int(row["call_index"]) <= last]
                )
                for name, (first, last) in CALL_WINDOWS.items()
            },
            "curve": {
                str(call): _aggregate_scope(
                    [row for row in selected if int(row["call_index"]) == call]
                )
                for call in range(1, 80)
            },
        }
    return result


def _aggregate_csv_rows(aggregates: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for count in TRAJECTORY_COUNTS:
        for kind in ("endpoints", "windows"):
            for scope, metrics in aggregates[str(count)][kind].items():
                rows.append(
                    {
                        "trajectory_count": count,
                        "scope_kind": kind,
                        "scope": scope,
                        **metrics,
                    }
                )
    return rows


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
    if not torch.equal(scales["selected"], scales["terminal"]):
        raise ValueError(f"n={trajectory_count} selected and terminal scales differ")
    gammas = {float(model.gamma) for model in models.values()}
    if len(gammas) != 1:
        raise ValueError(f"n={trajectory_count} selected and terminal gamma differ")
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
            current_selected = states[0:1]
            current_terminal = states[0:1]
            call1_inputs_equal = torch.equal(current_selected, current_terminal)
            selected_recurrence_exact = True
            terminal_recurrence_exact = True
            previous_selected = None
            previous_terminal = None
            for call_index in range(1, 80):
                if previous_selected is not None:
                    selected_recurrence_exact &= torch.equal(
                        current_selected, previous_selected
                    )
                    terminal_recurrence_exact &= torch.equal(
                        current_terminal, previous_terminal
                    )
                target = states[call_index : call_index + 1]
                y_ss, _, _ = contract_forward_sample(
                    models["selected"], sample, current_selected, boundary_policy=policy
                )
                y_tt, _, _ = contract_forward_sample(
                    models["terminal"], sample, current_terminal, boundary_policy=policy
                )
                y_ts, _, _ = contract_forward_sample(
                    models["terminal"], sample, current_selected, boundary_policy=policy
                )
                y_st, _, _ = contract_forward_sample(
                    models["selected"], sample, current_terminal, boundary_policy=policy
                )
                input_batch = torch.cat((current_selected, current_terminal), dim=0)
                selected_replay, _, _ = contract_forward_sample(
                    models["selected"],
                    batch_sample,
                    input_batch,
                    boundary_policy=policy,
                )
                terminal_replay, _, _ = contract_forward_sample(
                    models["terminal"],
                    batch_sample,
                    input_batch,
                    boundary_policy=policy,
                )
                primary = {"ss": y_ss, "ts": y_ts, "st": y_st, "tt": y_tt}
                replay = {
                    "ss": selected_replay[0:1],
                    "st": selected_replay[1:2],
                    "ts": terminal_replay[0:1],
                    "tt": terminal_replay[1:2],
                }
                if any(
                    not bool(torch.isfinite(value).all())
                    for value in (*primary.values(), *replay.values())
                ):
                    raise ValueError(
                        f"n={trajectory_count} trajectory {key} has a nonfinite "
                        f"map--path output at call {call_index}"
                    )
                metrics = four_way_map_path_metrics(
                    predictions=primary,
                    selected_input=current_selected,
                    terminal_input=current_terminal,
                    target=target,
                    node_weights=sample["node_weights"],
                    node_mask=sample["node_mask"],
                    component_scale=scales["selected"],
                )
                replay_records = {
                    label: replay_metrics(
                        replay_prediction=replay[label],
                        primary_prediction=primary[label],
                        input_state=(
                            current_selected if label[1] == "s" else current_terminal
                        ),
                        node_weights=sample["node_weights"],
                        node_mask=sample["node_mask"],
                        component_scale=scales["selected"],
                    )
                    for label in OUTPUT_LABELS
                }
                rows.append(
                    {
                        "schema": ROW_SCHEMA,
                        "execution_id": execution_id,
                        "trajectory_count": trajectory_count,
                        "trajectory": str(key),
                        "call_index": call_index,
                        "selected_optimizer_step": EXPECTED_STEPS["selected"],
                        "terminal_optimizer_step": EXPECTED_STEPS["terminal"],
                        "selected_checkpoint_sha256": str(
                            descriptors["selected"]["checkpoint_sha256"]
                        ),
                        "terminal_checkpoint_sha256": str(
                            descriptors["terminal"]["checkpoint_sha256"]
                        ),
                        **metrics,
                        "replay": replay_records,
                        "replay_all_finite": True,
                        "admissibility": {
                            label: _admissibility_record(value, gamma=gamma)
                            for label, value in primary.items()
                        },
                    }
                )
                previous_selected = y_ss
                previous_terminal = y_tt
                current_selected = y_ss
                current_terminal = y_tt
            trajectory_records.append(
                {
                    "trajectory": str(key),
                    "selected_completed_calls": 79,
                    "terminal_completed_calls": 79,
                    "call1_path_inputs_equal": call1_inputs_equal,
                    "selected_path_recurrence_exact": selected_recurrence_exact,
                    "terminal_path_recurrence_exact": terminal_recurrence_exact,
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


def _pooled_replay_fraction(rows: Sequence[Mapping[str, Any]], label: str) -> float:
    drift = float(
        np.mean(
            [float(row["replay"][label]["replay_drift_scaled_mse"]) for row in rows]
        )
    )
    residual = float(
        np.mean(
            [float(row["replay"][label]["owner_residual_scaled_mse"]) for row in rows]
        )
    )
    return math.inf if residual <= 0.0 else math.sqrt(max(drift, 0.0) / residual)


def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise ValueError("B1-C5-C output directory must not already exist")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    discovered = select_fixed_map_descriptors(discover_checkpoints(args.run_root))
    descriptors = {
        count: {role: discovered[count][role] for role in CHECKPOINT_ROLES}
        for count in TRAJECTORY_COUNTS
    }
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
            raise ValueError("B1-C5-C evaluation data manifest changed")
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
        data_manifest_digest = store.manifest_digest
    finally:
        store.close()

    expected_rows = len(TRAJECTORY_COUNTS) * len(outside_keys) * 79
    replay_fractions = {
        label: _pooled_replay_fraction(rows, label) for label in OUTPUT_LABELS
    }
    owner_replay_fractions = {
        "selected_on_selected_path": replay_fractions["ss"],
        "terminal_on_terminal_path": replay_fractions["tt"],
    }
    maximum_scalar_closure = max(
        float(row["maximum_scalar_closure_relative"]) for row in rows
    )
    maximum_output_closure = max(float(row["output_closure_relative"]) for row in rows)
    contract_checks = {
        "outside_population_has_28_unique_cases": len(outside_keys) == 28
        and len(set(outside_keys)) == 28,
        "historical_test_population_not_accessed": True,
        "checkpoint_reselection_not_performed": True,
        "exact_selected_and_terminal_maps_per_count": len(flat_descriptors) == 4,
        "all_rows_present": len(rows) == expected_rows,
        "all_primary_and_replay_outputs_finite": all(
            bool(row["admissibility"][label]["all_finite"])
            for row in rows
            for label in OUTPUT_LABELS
        )
        and all(bool(row["replay_all_finite"]) for row in rows),
        "owner_replays_below_ceiling": max(owner_replay_fractions.values())
        <= MAX_SELECTED_PATH_REPLAY_DRIFT_FRACTION,
        "scalar_identities_close": maximum_scalar_closure
        <= MAX_SCALAR_CLOSURE_RELATIVE,
        "output_identity_closes": maximum_output_closure <= MAX_OUTPUT_CLOSURE_RELATIVE,
        "both_paths_complete": all(
            record["selected_completed_calls"] == 79
            and record["terminal_completed_calls"] == 79
            and record["call1_path_inputs_equal"]
            and record["selected_path_recurrence_exact"]
            and record["terminal_path_recurrence_exact"]
            for count_record in trajectory_records
            for record in count_record["trajectories"]
        ),
    }
    if not all(contract_checks.values()):
        raise RuntimeError(f"B1-C5-C contract checks failed: {contract_checks}")
    aggregates = aggregate_map_path_rows(rows)
    rows_path = args.output_dir / "map_path_response.jsonl"
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
        "scientific_scope": "single_seed_selected_terminal_symmetric_map_path_decomposition",
        "historical_test_population_accessed": False,
        "checkpoint_reselection_performed": False,
        "new_training_performed": False,
        "precision": "float32_no_autocast",
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "split_partition_digest": split_manifest["partition_digest"],
        "data_manifest_sha256": sha256_file(args.data_dir / "manifest.json"),
        "data_manifest_digest": data_manifest_digest,
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
        "map_path_labels": {
            "ss": "selected map on selected path",
            "ts": "terminal map on selected path",
            "st": "selected map on terminal path",
            "tt": "terminal map on terminal path",
        },
        "recurrence_contract": (
            "only separate ss and tt calls advance their owning paths; cross-path "
            "and batched replay outputs never feed back"
        ),
        "effect_identity": "T = M_s + P_s + I = M_s + P_t = P_s + M_t",
        "node_measure": "reconstructed bump proxy weights; not physical volumes",
        "numerical_floor_contract": {
            "fresh_processes": list(EXPECTED_EXECUTION_IDS),
            "owner_replay_drift_fraction_ceiling": (
                MAX_SELECTED_PATH_REPLAY_DRIFT_FRACTION
            ),
            "scalar_closure_relative_ceiling": MAX_SCALAR_CLOSURE_RELATIVE,
            "output_closure_relative_ceiling": MAX_OUTPUT_CLOSURE_RELATIVE,
        },
        "contract_checks": contract_checks,
        "contract_complete": all(contract_checks.values()),
        "replay_drift_fractions": replay_fractions,
        "owner_replay_drift_fractions": owner_replay_fractions,
        "maximum_scalar_closure_relative": maximum_scalar_closure,
        "maximum_output_closure_relative": maximum_output_closure,
        "row_count": len(rows),
        "trajectory_records": trajectory_records,
        "aggregates": aggregates,
        "evaluation_numerics": {
            "amp": "none",
            "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        },
        "runtime_environment": runtime_environment(device),
        "claims_not_supported": [
            "optimizer, representation, data-count, capacity, or gradient cause",
            "optimization or representation convergence",
            "architecture comparison or multi-seed scaling",
            "physical conservation from bump proxy weights",
            "historical test performance",
        ],
    }
    atomic_write_json(args.output_dir / "summary.json", summary)
    artifact_files = (
        "summary.json",
        "map_path_response.jsonl",
        "aggregate_summary.csv",
    )
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
                "maximum_owner_replay_drift_fraction": max(
                    summary["owner_replay_drift_fractions"].values()
                ),
                "historical_test_population_accessed": False,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
