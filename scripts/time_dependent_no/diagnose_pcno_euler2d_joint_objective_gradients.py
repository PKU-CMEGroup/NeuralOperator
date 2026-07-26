#!/usr/bin/env python3
"""Audit whether three mechanism-matched D044 losses share a descent direction.

D057 takes no optimizer step and never changes the checkpoint.  It computes
full-model float32 gradients for the clean state loss, the teacher-forced
smooth-region graph-high-pass defect loss, and one detached generated-state
loss.  The sole candidate direction is the sum of the three unit-normalized
training gradients.  Validation gradients measure transfer only.
"""

from __future__ import annotations

import argparse
import hashlib
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
    conservative_admissibility,
    graph_neighbor_highpass,
    proxy_mass_weights,
    reference_smooth_region_mask,
    weighted_scaled_mse,
    weighted_scaled_relative_l2,
)
from utility.time_dependent_no.shock_vortex_family import (  # noqa: E402
    family_case_by_id,
    load_shock_vortex_family_manifest,
)


SCHEMA = "pcno_euler2d_joint_objective_gradients_d057_v1"
OBJECTIVES = ("clean_state", "smooth_highpass", "generated_state")
TRAIN_PAIRS = (
    ("sv_e08_y07", 8),
    ("sv_e03_y03", 7),
    ("sv_e03_y01", 44),
    ("sv_e05_y07", 41),
)
VALIDATION_CASES = (
    "sv_e00_y00",
    "sv_e00_y08",
    "sv_e06_y00",
    "sv_e06_y08",
    "sv_e11_y00",
    "sv_e11_y08",
)
VALIDATION_CALLS = (30, 60)
TRAIN_COSINE_MIN = 0.05
VALIDATION_COSINE_MIN = 0.02
REQUIRED_VALIDATION_CASES = 5
SHOCK_QUANTILE = 0.9
SMOOTH_DILATION_HOPS = 2


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--d056-dir", type=Path, required=True)
    parser.add_argument("--family-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")


def smooth_highpass_error_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    current: torch.Tensor,
    *,
    directed_edges: torch.Tensor,
    node_weights: torch.Tensor,
    node_mask: torch.Tensor,
    node_type: torch.Tensor,
    component_scale: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return D057's target-smooth high-pass loss and retained mass fraction."""

    if prediction.shape != target.shape or current.shape != target.shape:
        raise ValueError("prediction, target, and current must share shape [B,N,4]")
    batch_size, num_nodes, _ = prediction.shape
    if node_mask.shape != (batch_size, num_nodes, 1):
        raise ValueError("node_mask must have shape [B,N,1]")
    if node_type.shape == (batch_size, num_nodes, 1):
        node_type = node_type[..., 0]
    if node_type.shape != (batch_size, num_nodes):
        raise ValueError("node_type must have shape [B,N] or [B,N,1]")
    valid_nodes = node_mask.to(dtype=torch.bool)
    interior = (node_type == 0).unsqueeze(-1) & valid_nodes
    if not bool(interior[..., 0].any(dim=1).all()):
        interior = valid_nodes
    current_smooth = reference_smooth_region_mask(
        current,
        directed_edges,
        node_mask,
        interior_mask=interior,
        shock_quantile=SHOCK_QUANTILE,
        dilation_hops=SMOOTH_DILATION_HOPS,
        gamma=gamma,
    )
    target_smooth = reference_smooth_region_mask(
        target,
        directed_edges,
        node_mask,
        interior_mask=interior,
        shock_quantile=SHOCK_QUANTILE,
        dilation_hops=SMOOTH_DILATION_HOPS,
        gamma=gamma,
    )
    smooth = current_smooth & target_smooth
    scale = component_scale.reshape(1, 1, 4).to(
        dtype=prediction.dtype,
        device=prediction.device,
    )
    error_highpass = graph_neighbor_highpass(
        (prediction - target) / scale,
        directed_edges,
        node_mask,
    )
    weights = proxy_mass_weights(node_weights, node_mask)
    smooth_weights = weights * smooth.to(dtype=weights.dtype)
    smooth_mass = smooth_weights.sum()
    if not bool((smooth_weights.sum(dim=1) > 0.0).all()):
        raise ValueError("D057 smooth mask must retain positive mass per sample")
    loss = (smooth_weights * error_highpass.square()).sum() / (
        smooth_mass * prediction.shape[-1]
    )
    return loss, smooth_mass / weights.sum()


def _dot64(left: torch.Tensor, right: torch.Tensor, *, chunk: int = 1_000_000) -> float:
    if left.ndim != 1 or right.shape != left.shape:
        raise ValueError("gradient vectors must share one-dimensional shape")
    total = 0.0
    for start in range(0, left.numel(), chunk):
        stop = min(start + chunk, left.numel())
        total += float(
            torch.dot(left[start:stop].double(), right[start:stop].double())
        )
    return total


def _norm64(value: torch.Tensor) -> float:
    return math.sqrt(max(_dot64(value, value), 0.0))


def normalized_gradient_sum(
    gradients: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return the one D057 direction and its frozen gradient geometry."""

    if set(gradients) != set(OBJECTIVES):
        raise ValueError(f"gradients must contain exactly {OBJECTIVES}")
    vectors = {name: gradients[name].detach().cpu().float() for name in OBJECTIVES}
    shapes = {tuple(value.shape) for value in vectors.values()}
    if len(shapes) != 1 or next(iter(shapes))[0] < 1:
        raise ValueError("objective gradients must share a nonempty vector shape")
    if any(not bool(torch.isfinite(value).all()) for value in vectors.values()):
        raise ValueError("objective gradients must be finite")
    norms = {name: _norm64(value) for name, value in vectors.items()}
    if any(not math.isfinite(value) or value <= 0.0 for value in norms.values()):
        raise ValueError("objective gradients must have finite positive norm")

    units = {name: vectors[name] / norms[name] for name in OBJECTIVES}
    direction = sum(units.values(), torch.zeros_like(next(iter(units.values()))))
    direction_norm = _norm64(direction)
    if not math.isfinite(direction_norm) or direction_norm <= 0.0:
        raise ValueError("normalized gradient sum is zero or nonfinite")

    pairwise: dict[str, dict[str, float]] = {}
    for left in OBJECTIVES:
        pairwise[left] = {
            right: _dot64(vectors[left], vectors[right])
            / (norms[left] * norms[right])
            for right in OBJECTIVES
        }
    direction_cosines = {
        name: _dot64(vectors[name], direction) / (norms[name] * direction_norm)
        for name in OBJECTIVES
    }
    return direction, {
        "gradient_norms": norms,
        "inverse_norm_loss_weights": {
            name: 1.0 / norms[name] for name in OBJECTIVES
        },
        "pairwise_cosines": pairwise,
        "direction_norm": direction_norm,
        "directional_cosines": direction_cosines,
    }


def _parameter_layout(
    model: torch.nn.Module,
) -> tuple[list[tuple[str, torch.nn.Parameter, int, int]], int]:
    layout: list[tuple[str, torch.nn.Parameter, int, int]] = []
    offset = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        stop = offset + parameter.numel()
        layout.append((name, parameter, offset, stop))
        offset = stop
    if offset == 0:
        raise ValueError("model has no trainable parameters")
    return layout, offset


def _parameter_group(name: str) -> str:
    parts = name.split(".")
    if len(parts) >= 3 and parts[0] == "backbone" and parts[2].isdigit():
        return ".".join(parts[:3])
    return ".".join(parts[: min(2, len(parts))])


def _gradient_vector(
    model: torch.nn.Module,
    loss: torch.Tensor,
    layout: Sequence[tuple[str, torch.nn.Parameter, int, int]],
    size: int,
) -> torch.Tensor:
    if loss.numel() != 1 or not bool(torch.isfinite(loss)):
        raise FloatingPointError("D057 objective loss must be one finite scalar")
    model.zero_grad(set_to_none=True)
    loss.backward()
    vector = torch.empty(size, dtype=torch.float32, device="cpu")
    for name, parameter, start, stop in layout:
        gradient = parameter.grad
        if gradient is None:
            raise RuntimeError(f"D057 objective did not reach parameter {name}")
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError(f"nonfinite D057 gradient for {name}")
        vector[start:stop].copy_(gradient.detach().reshape(-1).float().cpu())
    model.zero_grad(set_to_none=True)
    return vector


def _parameter_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _admissibility_row(state: torch.Tensor, *, gamma: float) -> dict[str, Any]:
    report = conservative_admissibility(state, gamma=gamma)
    return {
        "all_admissible": bool(report["admissible"].all()),
        "min_density": float(report["density"].min().detach().cpu()),
        "min_internal_energy": float(
            report["internal_energy"].min().detach().cpu()
        ),
        "min_pressure": float(report["pressure"].min().detach().cpu()),
    }


def _objective_loss(
    model: torch.nn.Module,
    store: PCNOEuler2DShardStore,
    key: str,
    time_index: int,
    objective: str,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown D057 objective {objective!r}")
    sample = store.tensor_sample(key, time_index, step_stride=1, device=device)
    if objective in ("clean_state", "smooth_highpass"):
        prediction = model_call(model, sample, sample["current"])
        if objective == "clean_state":
            loss = weighted_scaled_mse(
                prediction,
                sample["target"],
                sample["node_weights"],
                sample["node_mask"],
                model.state_scale,
            )
            return loss, {}
        loss, mass_fraction = smooth_highpass_error_mse(
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
        return loss, {"smooth_physical_mass_fraction": float(mass_fraction.detach().cpu())}

    if time_index < 1:
        raise ValueError("generated-state objective requires time_index >= 1")
    previous = store.tensor_sample(
        key,
        time_index - 1,
        step_stride=1,
        device=device,
    )
    with torch.no_grad():
        generated_current = model_call(model, previous, previous["current"]).detach()
    admissibility = _admissibility_row(generated_current, gamma=model.gamma)
    if not admissibility["all_admissible"]:
        raise FloatingPointError("D057 generated input is not raw-admissible")
    generated_prediction = model_call(model, sample, generated_current)
    loss = weighted_scaled_mse(
        generated_prediction,
        sample["target"],
        sample["node_weights"],
        sample["node_mask"],
        model.state_scale,
    )
    input_error = weighted_scaled_relative_l2(
        generated_current,
        sample["current"],
        sample["node_weights"],
        sample["node_mask"],
        model.state_scale,
    )
    return loss, {
        "generated_input_admissibility": admissibility,
        "generated_input_relative_l2": float(input_error.detach().cpu()),
    }


def _layerwise_norms(
    gradients: Mapping[str, torch.Tensor],
    layout: Sequence[tuple[str, torch.nn.Parameter, int, int]],
) -> dict[str, dict[str, float]]:
    groups: dict[str, list[tuple[int, int]]] = {}
    for name, _, start, stop in layout:
        groups.setdefault(_parameter_group(name), []).append((start, stop))
    result: dict[str, dict[str, float]] = {}
    for group, slices in groups.items():
        result[group] = {}
        for objective in OBJECTIVES:
            squared = sum(
                _dot64(gradients[objective][start:stop], gradients[objective][start:stop])
                for start, stop in slices
            )
            result[group][objective] = math.sqrt(max(squared, 0.0))
    return result


def joint_gradient_selector(
    training: Mapping[str, Any],
    validation_rows: Sequence[Mapping[str, Any]],
    ledger: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply D057's frozen structural and common-descent gates."""

    rows = [dict(row) for row in validation_rows]
    keys = [(str(row.get("trajectory")), int(row.get("call_index", -1))) for row in rows]
    duplicate_keys = sorted({key for key in keys if keys.count(key) > 1})
    expected_keys = {
        (trajectory, call)
        for trajectory in VALIDATION_CASES
        for call in VALIDATION_CALLS
    }
    train_cosines = training.get("directional_cosines", {})
    training_finite = all(
        math.isfinite(float(train_cosines.get(name, math.nan)))
        for name in OBJECTIVES
    )
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
        "exact_validation_rows": len(rows) == len(expected_keys)
        and set(keys) == expected_keys
        and not duplicate_keys,
        "finite_positive_training_gradients": bool(training.get("all_finite"))
        and training_finite,
        "finite_validation_gradients_positive_masks_and_admissible_inputs": row_structure,
        "zero_optimizer_steps_and_unchanged_state": (
            ledger.get("optimizer_created") is False
            and int(ledger.get("optimizer_steps", -1)) == 0
            and ledger.get("parameter_state_unchanged") is True
        ),
        "test_split_unopened": ledger.get("test_trajectory_access") == [],
    }
    contract_complete = all(structural.values())
    training_passed = training_finite and all(
        float(train_cosines[name]) >= TRAIN_COSINE_MIN for name in OBJECTIVES
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
        classification = "joint_objective_local_descent_compatible"
        route = "authorize_short_continuation_contract_only"
    else:
        classification = "naive_equal_gradient_scalarization_incompatible"
        route = "reject_equal_gradient_continuation_no_training"
    return {
        "version": "joint_objective_gradient_selector_d057_v1",
        "contract_complete": contract_complete,
        "contract_checks": structural,
        "training_direction_passed": training_passed,
        "validation_calls": calls,
        "promotion_passed": promoted,
        "classification": classification,
        "route": route,
        "duplicate_keys": [list(key) for key in duplicate_keys],
        "thresholds": {
            "training_directional_cosine_min": TRAIN_COSINE_MIN,
            "validation_directional_cosine_min": VALIDATION_COSINE_MIN,
            "validation_joint_cases_min": REQUIRED_VALIDATION_CASES,
        },
        "claim_boundary": (
            "local first-order objective compatibility at one frozen checkpoint; "
            "not finite-step optimization, rollout improvement, or conservation"
        ),
    }


def _validate_sources(
    args: argparse.Namespace,
    checkpoint: Mapping[str, Any],
    checkpoint_digest: str,
    store: PCNOEuler2DShardStore,
    family: Mapping[str, Any],
) -> dict[str, Any]:
    d056_path = args.d056_dir / "summary.json"
    if not d056_path.is_file():
        raise FileNotFoundError(d056_path)
    d056 = json.loads(d056_path.read_text(encoding="utf-8"))
    if (
        d056.get("status") != "complete"
        or d056.get("selector", {}).get("classification")
        != "bounded_causal_support_correction_insufficient"
        or d056.get("checkpoint", {}).get("sha256") != checkpoint_digest
    ):
        raise ValueError("D057 requires the completed rejecting D056 result")
    if checkpoint.get("boundary_mode") != "model_all_nodes" or not bool(
        checkpoint.get("raw_recurrence")
    ):
        raise ValueError("D057 requires legal raw model-all-node recurrence")
    if int(checkpoint.get("step_stride", -1)) != 1:
        raise ValueError("D057 requires the D044 stride-one checkpoint")
    if checkpoint.get("data_manifest_digest") != store.manifest_digest:
        raise ValueError("checkpoint and shard manifest digests differ")
    if store.manifest.get("weight_provenance") != "validated_physical_cell_volume_normalized":
        raise ValueError("D057 requires validated physical cell-volume weights")
    training_args = checkpoint.get("training_args", {})
    if float(training_args.get("input_noise_std", 0.0)) <= 0.0:
        raise ValueError("D057 parent must be the noise-trained D044 checkpoint")
    if float(training_args.get("generated_state_exposure_weight", 0.0)) != 0.0:
        raise ValueError("D057 parent must not already use generated exposure")

    train_keys = {str(key) for key in checkpoint.get("train_keys", [])}
    val_keys = {str(key) for key in checkpoint.get("val_keys", [])}
    test_keys = {str(key) for key in checkpoint.get("test_keys", [])}
    for key, time_index in TRAIN_PAIRS:
        if key not in train_keys or family_case_by_id(family, key)["split"] != "train":
            raise ValueError(f"D057 training pair has the wrong split: {key}")
        if time_index < 1 or time_index + 1 >= int(store.entry(key)["num_steps"]):
            raise ValueError(f"D057 training pair is out of range: {(key, time_index)}")
    for key in VALIDATION_CASES:
        if key not in val_keys or family_case_by_id(family, key)["split"] != "validation":
            raise ValueError(f"D057 validation case has the wrong split: {key}")
        if key in test_keys:
            raise ValueError(f"D057 validation case overlaps test: {key}")
    return {
        "d056_summary_sha256": sha256_file(d056_path),
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

    checkpoint = load_checkpoint(args.checkpoint)
    checkpoint_digest = sha256_file(args.checkpoint)
    family = load_shock_vortex_family_manifest(args.family_manifest)
    store = PCNOEuler2DShardStore(args.data_dir)
    try:
        source = _validate_sources(
            args,
            checkpoint,
            checkpoint_digest,
            store,
            family,
        )
        model = build_model(checkpoint, device)
        model.eval()
        layout, parameter_size = _parameter_layout(model)
        before_digest = _parameter_digest(model)
        train_gradients = {
            name: torch.zeros(parameter_size, dtype=torch.float32)
            for name in OBJECTIVES
        }
        train_losses = {name: [] for name in OBJECTIVES}
        train_rows: list[dict[str, Any]] = []
        backward_calls = 0

        for key, time_index in TRAIN_PAIRS:
            row: dict[str, Any] = {
                "trajectory": key,
                "time_index": time_index,
                "objective_losses": {},
            }
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
                train_gradients[objective].add_(vector)
                value = float(loss.detach().cpu())
                train_losses[objective].append(value)
                row["objective_losses"][objective] = value
                row.update(info)
                backward_calls += 1
            train_rows.append(row)
            print(json.dumps({"train_pair": [key, time_index]}, sort_keys=True), flush=True)

        for objective in OBJECTIVES:
            train_gradients[objective].div_(len(TRAIN_PAIRS))
        direction, training_geometry = normalized_gradient_sum(train_gradients)
        training_geometry["objective_mean_losses"] = {
            name: float(np.mean(train_losses[name])) for name in OBJECTIVES
        }
        training_geometry["layerwise_gradient_norms"] = _layerwise_norms(
            train_gradients,
            layout,
        )
        training_geometry["all_finite"] = all(
            bool(torch.isfinite(value).all()) for value in train_gradients.values()
        )
        direction_norm = float(training_geometry["direction_norm"])

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
                        raise FloatingPointError("validation gradient norm is invalid")
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
        }
        selector = joint_gradient_selector(training_geometry, validation_rows, ledger)
        summary = {
            "schema": SCHEMA,
            "status": "complete" if selector["contract_complete"] else "failed_contract",
            "source": {
                **source,
                "d056_artifact_dir_name": args.d056_dir.name,
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
                "objectives": list(OBJECTIVES),
                "train_pairs": [list(pair) for pair in TRAIN_PAIRS],
                "validation_cases": list(VALIDATION_CASES),
                "validation_calls": list(VALIDATION_CALLS),
                "shock_quantile": SHOCK_QUANTILE,
                "smooth_dilation_hops": SMOOTH_DILATION_HOPS,
                "gradient_precision": "float32_forward_backward_float64_scalar_dots",
                "candidate_direction": "sum_of_three_unit_aggregate_training_gradients",
            },
            "training": training_geometry,
            "training_pair_metrics": train_rows,
            "execution_ledger": ledger,
            "validation_rows_sha256": sha256_file(rows_path),
            "selector": selector,
            "claim_boundary": {
                "verified": "first-order gradient geometry at the frozen D044 checkpoint",
                "plausible": "one short continuation contract only after a full pass",
                "unsupported": (
                    "finite-step optimization, rollout improvement, OOD behavior, "
                    "flux prediction, or physical conservation"
                ),
            },
        }
        write_json(args.output_dir / "summary.json", summary)
        print(json.dumps(selector, indent=2, sort_keys=True), flush=True)
        if not selector["contract_complete"]:
            raise RuntimeError("D057 failed its frozen diagnostic contract")
    finally:
        store.close()


if __name__ == "__main__":
    main()
