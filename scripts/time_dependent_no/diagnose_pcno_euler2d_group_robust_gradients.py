#!/usr/bin/env python3
"""Build D058's zero-step geometry-group robust common descent direction.

The audit uses every one of the 84 full-resolution training trajectories at
calls 30 and 60.  Seven vortex-position state/recurrence tasks and one global
smooth-high-pass task are combined by a fixed 500-step minimum-norm
Frank--Wolfe solve.  Validation gradients are evaluation only; no optimizer is
created and the D044 checkpoint is never modified.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no.diagnose_pcno_euler2d_joint_objective_gradients import (  # noqa: E402
    OBJECTIVES,
    SHOCK_QUANTILE,
    SMOOTH_DILATION_HOPS,
    VALIDATION_CALLS,
    VALIDATION_CASES,
    _admissibility_row,
    _dot64,
    _gradient_vector,
    _norm64,
    _objective_loss,
    _parameter_digest,
    _parameter_layout,
    smooth_highpass_error_mse,
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
    weighted_scaled_mse,
    weighted_scaled_relative_l2,
)
from utility.time_dependent_no.shock_vortex_family import (  # noqa: E402
    load_shock_vortex_family_manifest,
)


SCHEMA = "pcno_euler2d_group_robust_gradients_d058_v1"
TRAIN_CALLS = (30, 60)
TRAIN_Y_INDICES = tuple(range(1, 8))
TRAIN_EPSILON_INDICES = tuple(range(12))
FRANK_WOLFE_ITERATIONS = 500
FRANK_WOLFE_GAP_MAX = 1.0e-6
TRAIN_TASK_COSINE_MIN = 0.02
VALIDATION_COSINE_MIN = 0.02
REQUIRED_VALIDATION_CASES = 5


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--d057-dir", type=Path, required=True)
    parser.add_argument("--family-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")


def minimum_norm_frank_wolfe(
    task_vectors: Mapping[str, torch.Tensor],
    *,
    iterations: int = FRANK_WOLFE_ITERATIONS,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return a fixed-iteration minimum-norm convex task combination."""

    if iterations < 1:
        raise ValueError("Frank-Wolfe iterations must be positive")
    names = sorted(task_vectors)
    if len(names) < 2:
        raise ValueError("MGDA requires at least two tasks")
    vectors = {name: task_vectors[name].detach().cpu().float() for name in names}
    shapes = {tuple(value.shape) for value in vectors.values()}
    if len(shapes) != 1 or next(iter(shapes))[0] < 1:
        raise ValueError("MGDA task vectors must share nonempty one-dimensional shape")
    units: dict[str, torch.Tensor] = {}
    input_norms: dict[str, float] = {}
    for name in names:
        value = vectors[name]
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"MGDA task {name} is nonfinite")
        norm = _norm64(value)
        if not math.isfinite(norm) or norm <= 0.0:
            raise ValueError(f"MGDA task {name} has invalid norm")
        input_norms[name] = norm
        units[name] = value / norm

    gram = np.empty((len(names), len(names)), dtype=np.float64)
    for left, left_name in enumerate(names):
        for right in range(left + 1):
            value = _dot64(units[left_name], units[names[right]])
            gram[left, right] = value
            gram[right, left] = value
    coefficients = np.zeros(len(names), dtype=np.float64)
    coefficients[0] = 1.0
    for _ in range(iterations):
        gram_alpha = gram @ coefficients
        vertex = int(np.argmin(gram_alpha))
        direction = -coefficients
        direction[vertex] += 1.0
        denominator = float(direction @ gram @ direction)
        numerator = float(-(coefficients @ gram @ direction))
        step = 0.0 if denominator <= 1.0e-30 else float(
            np.clip(numerator / denominator, 0.0, 1.0)
        )
        coefficients += step * direction

    gram_alpha = gram @ coefficients
    objective = float(coefficients @ gram_alpha)
    gap = float(objective - np.min(gram_alpha))
    combined = torch.zeros_like(units[names[0]])
    for coefficient, name in zip(coefficients, names, strict=True):
        combined.add_(units[name], alpha=float(coefficient))
    combined_norm = _norm64(combined)
    if not math.isfinite(combined_norm) or combined_norm <= 0.0:
        raise ValueError("MGDA direction is zero or nonfinite")
    task_cosines = {
        name: _dot64(units[name], combined) / combined_norm for name in names
    }
    return combined, {
        "task_names": names,
        "task_input_norms": input_norms,
        "gram_matrix": gram.tolist(),
        "coefficients": {
            name: float(value)
            for name, value in zip(names, coefficients, strict=True)
        },
        "iterations": iterations,
        "objective_squared_norm": objective,
        "direction_norm": combined_norm,
        "frank_wolfe_gap": gap,
        "task_directional_cosines": task_cosines,
    }


def _batched_gradient_triplet(
    model: torch.nn.Module,
    store: PCNOEuler2DShardStore,
    key: str,
    *,
    device: torch.device,
    layout: Sequence[tuple[str, torch.nn.Parameter, int, int]],
    parameter_size: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    time_indices = [call - 1 for call in TRAIN_CALLS]
    sample = store.tensor_batch(key, time_indices, step_stride=1, device=device)
    losses: dict[str, float] = {}
    gradients: dict[str, torch.Tensor] = {}

    prediction = model_call(model, sample, sample["current"])
    clean_loss = weighted_scaled_mse(
        prediction,
        sample["target"],
        sample["node_weights"],
        sample["node_mask"],
        model.state_scale,
    )
    gradients["clean_state"] = _gradient_vector(
        model, clean_loss, layout, parameter_size
    )
    losses["clean_state"] = float(clean_loss.detach().cpu())

    prediction = model_call(model, sample, sample["current"])
    highpass_loss, smooth_mass = smooth_highpass_error_mse(
        prediction,
        sample["target"],
        sample["current"],
        directed_edges=sample["directed_edges"],
        node_weights=sample["node_weights"],
        node_mask=sample["node_mask"],
        node_type=sample["node_type"],
        component_scale=model.state_scale,
        gamma=model.gamma,
    )
    gradients["smooth_highpass"] = _gradient_vector(
        model, highpass_loss, layout, parameter_size
    )
    losses["smooth_highpass"] = float(highpass_loss.detach().cpu())

    previous = store.tensor_batch(
        key,
        [index - 1 for index in time_indices],
        step_stride=1,
        device=device,
    )
    with torch.no_grad():
        generated_current = model_call(model, previous, previous["current"]).detach()
    admissibility = _admissibility_row(generated_current, gamma=model.gamma)
    if not admissibility["all_admissible"]:
        raise FloatingPointError("D058 generated input is not raw-admissible")
    generated_prediction = model_call(model, sample, generated_current)
    generated_loss = weighted_scaled_mse(
        generated_prediction,
        sample["target"],
        sample["node_weights"],
        sample["node_mask"],
        model.state_scale,
    )
    gradients["generated_state"] = _gradient_vector(
        model, generated_loss, layout, parameter_size
    )
    losses["generated_state"] = float(generated_loss.detach().cpu())
    generated_input_error = weighted_scaled_relative_l2(
        generated_current,
        sample["current"],
        sample["node_weights"],
        sample["node_mask"],
        model.state_scale,
    )
    return gradients, {
        "objective_losses": losses,
        "smooth_physical_mass_fraction": float(smooth_mass.detach().cpu()),
        "generated_input_relative_l2": float(generated_input_error.detach().cpu()),
        "generated_input_admissibility": admissibility,
    }


def _state_recurrence_task(
    clean: torch.Tensor,
    generated: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    clean_norm = _norm64(clean)
    generated_norm = _norm64(generated)
    if min(clean_norm, generated_norm) <= 0.0:
        raise ValueError("state/recurrence gradients must be nonzero")
    clean_unit = clean / clean_norm
    generated_unit = generated / generated_norm
    cosine = _dot64(clean_unit, generated_unit)
    combined = clean_unit + generated_unit
    combined_norm = _norm64(combined)
    if combined_norm <= 0.0:
        raise ValueError("state and recurrence gradients cancel exactly")
    return combined / combined_norm, {
        "clean_gradient_norm": clean_norm,
        "generated_gradient_norm": generated_norm,
        "clean_generated_cosine": cosine,
        "pre_normalization_sum_norm": combined_norm,
    }


def group_robust_selector(
    mgda: Mapping[str, Any],
    validation_rows: Sequence[Mapping[str, Any]],
    ledger: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply D058's frozen numerical, training-task, and transfer gates."""

    rows = [dict(row) for row in validation_rows]
    keys = [(str(row.get("trajectory")), int(row.get("call_index", -1))) for row in rows]
    expected_keys = {
        (trajectory, call)
        for trajectory in VALIDATION_CASES
        for call in VALIDATION_CALLS
    }
    duplicate_keys = sorted({key for key in keys if keys.count(key) > 1})
    task_cosines = mgda.get("task_directional_cosines", {})
    task_names = {f"state_recurrence_y{index:02d}" for index in TRAIN_Y_INDICES}
    task_names.add("smooth_highpass_global")
    row_structure = all(
        row.get("schema") == SCHEMA
        and set(row.get("directional_cosines", {})) == set(OBJECTIVES)
        and all(
            math.isfinite(float(row["directional_cosines"][name]))
            and math.isfinite(float(row["gradient_norms"][name]))
            and float(row["gradient_norms"][name]) > 0.0
            and math.isfinite(float(row["objective_losses"][name]))
            for name in OBJECTIVES
        )
        and float(row.get("smooth_physical_mass_fraction", 0.0)) > 0.0
        and row.get("generated_input_admissibility", {}).get("all_admissible") is True
        for row in rows
    )
    structural = {
        "exact_eight_training_tasks": set(task_cosines) == task_names,
        "finite_nonzero_direction": math.isfinite(float(mgda.get("direction_norm", math.nan)))
        and float(mgda.get("direction_norm", 0.0)) > 0.0,
        "frank_wolfe_gap": math.isfinite(float(mgda.get("frank_wolfe_gap", math.nan)))
        and float(mgda.get("frank_wolfe_gap", math.inf)) <= FRANK_WOLFE_GAP_MAX,
        "exact_validation_rows": len(rows) == len(expected_keys)
        and set(keys) == expected_keys
        and not duplicate_keys,
        "finite_validation_gradients_positive_masks_and_admissible_inputs": row_structure,
        "zero_optimizer_steps_and_unchanged_state": (
            ledger.get("optimizer_created") is False
            and int(ledger.get("optimizer_steps", -1)) == 0
            and ledger.get("parameter_state_unchanged") is True
        ),
        "test_split_unopened": ledger.get("test_trajectory_access") == [],
    }
    contract_complete = all(structural.values())
    training_passed = set(task_cosines) == task_names and all(
        math.isfinite(float(value)) and float(value) >= TRAIN_TASK_COSINE_MIN
        for value in task_cosines.values()
    )
    calls: dict[str, Any] = {}
    validation_passed = True
    for call in VALIDATION_CALLS:
        selected = [row for row in rows if int(row.get("call_index", -1)) == call]
        passing = sum(
            all(
                float(row["directional_cosines"][name]) >= VALIDATION_COSINE_MIN
                for name in OBJECTIVES
            )
            for row in selected
        )
        passed = len(selected) == len(VALIDATION_CASES) and passing >= REQUIRED_VALIDATION_CASES
        validation_passed &= passed
        calls[str(call)] = {
            "row_count": len(selected),
            "joint_positive_case_count": passing,
            "passed": passed,
        }
    promoted = contract_complete and training_passed and validation_passed
    if not contract_complete:
        classification = "incomplete_contract"
        route = "stop_without_training"
    elif promoted:
        classification = "geometry_group_common_descent_candidate"
        route = "authorize_short_group_mgda_continuation_contract_only"
    else:
        classification = "geometry_group_joint_objective_insufficient"
        route = "stop_joint_objective_route_no_training"
    return {
        "version": "geometry_group_gradient_selector_d058_v1",
        "contract_complete": contract_complete,
        "contract_checks": structural,
        "training_tasks_passed": training_passed,
        "validation_calls": calls,
        "promotion_passed": promoted,
        "classification": classification,
        "route": route,
        "duplicate_keys": [list(key) for key in duplicate_keys],
        "thresholds": {
            "frank_wolfe_gap_max": FRANK_WOLFE_GAP_MAX,
            "training_task_cosine_min": TRAIN_TASK_COSINE_MIN,
            "validation_directional_cosine_min": VALIDATION_COSINE_MIN,
            "validation_joint_cases_min": REQUIRED_VALIDATION_CASES,
        },
        "claim_boundary": (
            "zero-step local gradient compatibility using all training geometry "
            "groups; not finite-step learning, rollout gain, or conservation"
        ),
    }


def _training_groups(family: Mapping[str, Any]) -> dict[int, list[str]]:
    groups = {index: [] for index in TRAIN_Y_INDICES}
    for case in family["cases"]:
        if case["split"] != "train":
            continue
        y_index = int(case["y_index"])
        epsilon_index = int(case["epsilon_index"])
        if y_index not in groups or epsilon_index not in TRAIN_EPSILON_INDICES:
            raise ValueError("D058 encountered an unexpected training group")
        groups[y_index].append(str(case["case_id"]))
    for y_index, keys in groups.items():
        keys.sort(key=lambda key: int(key.split("_e", 1)[1].split("_", 1)[0]))
        epsilon_indices = {
            int(case["epsilon_index"])
            for case in family["cases"]
            if str(case["case_id"]) in keys
        }
        if len(keys) != len(TRAIN_EPSILON_INDICES) or epsilon_indices != set(
            TRAIN_EPSILON_INDICES
        ):
            raise ValueError(f"D058 training y group {y_index} is incomplete")
    return groups


def _validate_sources(
    args: argparse.Namespace,
    checkpoint: Mapping[str, Any],
    checkpoint_digest: str,
    store: PCNOEuler2DShardStore,
    family: Mapping[str, Any],
    groups: Mapping[int, Sequence[str]],
) -> dict[str, Any]:
    d057_path = args.d057_dir / "summary.json"
    if not d057_path.is_file():
        raise FileNotFoundError(d057_path)
    d057 = json.loads(d057_path.read_text(encoding="utf-8"))
    if (
        d057.get("status") != "complete"
        or d057.get("selector", {}).get("classification")
        != "naive_equal_gradient_scalarization_incompatible"
        or d057.get("checkpoint", {}).get("sha256") != checkpoint_digest
    ):
        raise ValueError("D058 requires the completed rejecting D057 result")
    if checkpoint.get("boundary_mode") != "model_all_nodes" or not bool(
        checkpoint.get("raw_recurrence")
    ):
        raise ValueError("D058 requires legal raw model-all-node recurrence")
    if int(checkpoint.get("step_stride", -1)) != 1:
        raise ValueError("D058 requires the D044 stride-one checkpoint")
    if checkpoint.get("data_manifest_digest") != store.manifest_digest:
        raise ValueError("checkpoint and shard manifest digests differ")
    if store.manifest.get("weight_provenance") != "validated_physical_cell_volume_normalized":
        raise ValueError("D058 requires validated physical cell-volume weights")
    training_args = checkpoint.get("training_args", {})
    if float(training_args.get("input_noise_std", 0.0)) <= 0.0:
        raise ValueError("D058 parent must be the noise-trained D044 checkpoint")
    if float(training_args.get("generated_state_exposure_weight", 0.0)) != 0.0:
        raise ValueError("D058 parent must not already use generated exposure")
    expected_train = {key for keys in groups.values() for key in keys}
    checkpoint_train = {str(key) for key in checkpoint.get("train_keys", [])}
    checkpoint_val = {str(key) for key in checkpoint.get("val_keys", [])}
    if expected_train != checkpoint_train or len(expected_train) != 84:
        raise ValueError("D058 requires the exact complete 84-case training split")
    if not set(VALIDATION_CASES).issubset(checkpoint_val):
        raise ValueError("D058 validation cohort differs from D044")
    for key in expected_train | set(VALIDATION_CASES):
        if int(store.entry(key)["num_steps"]) <= max(TRAIN_CALLS):
            raise ValueError(f"D058 trajectory is too short: {key}")
    return {
        "d057_summary_sha256": sha256_file(d057_path),
        "parent_input_noise_std": float(training_args["input_noise_std"]),
        "parent_generated_state_exposure_weight": float(
            training_args.get("generated_state_exposure_weight", 0.0)
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_args(args)
    device = select_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    checkpoint = load_checkpoint(args.checkpoint)
    checkpoint_digest = sha256_file(args.checkpoint)
    family = load_shock_vortex_family_manifest(args.family_manifest)
    groups = _training_groups(family)
    store = PCNOEuler2DShardStore(args.data_dir)
    try:
        source = _validate_sources(
            args,
            checkpoint,
            checkpoint_digest,
            store,
            family,
            groups,
        )
        model = build_model(checkpoint, device)
        model.eval()
        layout, parameter_size = _parameter_layout(model)
        before_digest = _parameter_digest(model)
        task_vectors: dict[str, torch.Tensor] = {}
        global_highpass = torch.zeros(parameter_size, dtype=torch.float32)
        group_rows: list[dict[str, Any]] = []
        backward_calls = 0

        for y_index in TRAIN_Y_INDICES:
            aggregate = {
                objective: torch.zeros(parameter_size, dtype=torch.float32)
                for objective in OBJECTIVES
            }
            losses = {objective: [] for objective in OBJECTIVES}
            smooth_mass: list[float] = []
            generated_error: list[float] = []
            for key in groups[y_index]:
                gradients, metrics = _batched_gradient_triplet(
                    model,
                    store,
                    key,
                    device=device,
                    layout=layout,
                    parameter_size=parameter_size,
                )
                for objective in OBJECTIVES:
                    aggregate[objective].add_(gradients[objective])
                    losses[objective].append(metrics["objective_losses"][objective])
                smooth_mass.append(metrics["smooth_physical_mass_fraction"])
                generated_error.append(metrics["generated_input_relative_l2"])
                backward_calls += len(OBJECTIVES)
                del gradients
            for objective in OBJECTIVES:
                aggregate[objective].div_(len(groups[y_index]))
            task_name = f"state_recurrence_y{y_index:02d}"
            task, task_report = _state_recurrence_task(
                aggregate["clean_state"],
                aggregate["generated_state"],
            )
            task_vectors[task_name] = task
            global_highpass.add_(aggregate["smooth_highpass"] / len(TRAIN_Y_INDICES))
            group_rows.append(
                {
                    "y_index": y_index,
                    "trajectory_count": len(groups[y_index]),
                    "trajectory_keys": list(groups[y_index]),
                    "calls": list(TRAIN_CALLS),
                    "objective_mean_losses": {
                        objective: float(np.mean(losses[objective]))
                        for objective in OBJECTIVES
                    },
                    "smooth_physical_mass_fraction_range": [
                        float(min(smooth_mass)),
                        float(max(smooth_mass)),
                    ],
                    "generated_input_relative_l2_range": [
                        float(min(generated_error)),
                        float(max(generated_error)),
                    ],
                    **task_report,
                }
            )
            print(
                json.dumps(
                    {"training_y_index": y_index, "trajectories": len(groups[y_index])},
                    sort_keys=True,
                ),
                flush=True,
            )
            del aggregate
            gc.collect()

        global_highpass_norm = _norm64(global_highpass)
        if global_highpass_norm <= 0.0:
            raise ValueError("D058 global high-pass gradient is zero")
        task_vectors["smooth_highpass_global"] = (
            global_highpass / global_highpass_norm
        )
        direction, mgda = minimum_norm_frank_wolfe(task_vectors)
        direction_norm = float(mgda["direction_norm"])
        del task_vectors, global_highpass
        gc.collect()

        args.output_dir.mkdir(parents=True)
        rows_path = args.output_dir / "validation_gradients.jsonl"
        validation_rows: list[dict[str, Any]] = []
        for call_index in VALIDATION_CALLS:
            time_index = call_index - 1
            for key in VALIDATION_CASES:
                losses: dict[str, float] = {}
                norms: dict[str, float] = {}
                cosines: dict[str, float] = {}
                row_info: dict[str, Any] = {}
                for objective in OBJECTIVES:
                    loss, info = _objective_loss(
                        model,
                        store,
                        key,
                        time_index,
                        objective,
                        device=device,
                    )
                    vector = _gradient_vector(model, loss, layout, parameter_size)
                    norm = _norm64(vector)
                    if not math.isfinite(norm) or norm <= 0.0:
                        raise FloatingPointError("D058 validation gradient is invalid")
                    losses[objective] = float(loss.detach().cpu())
                    norms[objective] = norm
                    cosines[objective] = _dot64(vector, direction) / (
                        norm * direction_norm
                    )
                    row_info.update(info)
                    backward_calls += 1
                row = {
                    "schema": SCHEMA,
                    "trajectory": key,
                    "call_index": call_index,
                    "time_index": time_index,
                    "objective_losses": losses,
                    "gradient_norms": norms,
                    "directional_cosines": cosines,
                    "smooth_physical_mass_fraction": row_info[
                        "smooth_physical_mass_fraction"
                    ],
                    "generated_input_relative_l2": row_info[
                        "generated_input_relative_l2"
                    ],
                    "generated_input_admissibility": row_info[
                        "generated_input_admissibility"
                    ],
                    "claim_boundary": (
                        "validation gradient transfer only; no parameter update or "
                        "finite-step improvement claim"
                    ),
                }
                validation_rows.append(row)
                append_jsonl(rows_path, row)
                print(
                    json.dumps(
                        {"trajectory": key, "call_index": call_index},
                        sort_keys=True,
                    ),
                    flush=True,
                )

        after_digest = _parameter_digest(model)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_memory = int(torch.cuda.max_memory_allocated(device))
        else:
            peak_memory = None
        ledger = {
            "optimizer_created": False,
            "optimizer_steps": 0,
            "backward_calls": backward_calls,
            "parameter_state_before_sha256": before_digest,
            "parameter_state_after_sha256": after_digest,
            "parameter_state_unchanged": before_digest == after_digest,
            "test_trajectory_access": [],
            "amp": False,
            "input_noise_sampled": False,
            "peak_memory_allocated_bytes": peak_memory,
            "wall_seconds": perf_counter() - started,
        }
        selector = group_robust_selector(mgda, validation_rows, ledger)
        summary = {
            "schema": SCHEMA,
            "status": "complete" if selector["contract_complete"] else "failed_contract",
            "source": {
                **source,
                "d057_artifact_dir_name": args.d057_dir.name,
                "family_manifest_sha256": sha256_file(args.family_manifest),
                "family_manifest_digest": family["manifest_digest_sha256"],
            },
            "checkpoint": {
                "sha256": checkpoint_digest,
                "config_digest": str(checkpoint["config_digest"]),
                "data_manifest_digest": store.manifest_digest,
                "boundary_mode": checkpoint["boundary_mode"],
                "raw_recurrence": bool(checkpoint["raw_recurrence"]),
                "parameter_count": parameter_size,
            },
            "contract": {
                "train_trajectory_count": sum(len(keys) for keys in groups.values()),
                "train_y_indices": list(TRAIN_Y_INDICES),
                "train_epsilon_indices": list(TRAIN_EPSILON_INDICES),
                "train_calls": list(TRAIN_CALLS),
                "validation_cases": list(VALIDATION_CASES),
                "validation_calls": list(VALIDATION_CALLS),
                "shock_quantile": SHOCK_QUANTILE,
                "smooth_dilation_hops": SMOOTH_DILATION_HOPS,
                "frank_wolfe_iterations": FRANK_WOLFE_ITERATIONS,
                "gradient_precision": "float32_forward_backward_float64_scalar_dots",
            },
            "training_groups": group_rows,
            "mgda": mgda,
            "execution_ledger": ledger,
            "validation_rows_sha256": sha256_file(rows_path),
            "selector": selector,
            "claim_boundary": {
                "verified": (
                    "first-order geometry-group gradient geometry on the complete "
                    "D044 training split"
                ),
                "plausible": "one short group-MGDA continuation contract after a pass",
                "unsupported": (
                    "finite-step optimization, rollout improvement, OOD behavior, "
                    "flux prediction, or physical conservation"
                ),
            },
        }
        write_json(args.output_dir / "summary.json", summary)
        print(json.dumps(selector, indent=2, sort_keys=True), flush=True)
        if not selector["contract_complete"]:
            raise RuntimeError("D058 failed its frozen diagnostic contract")
    finally:
        store.close()


if __name__ == "__main__":
    main()
