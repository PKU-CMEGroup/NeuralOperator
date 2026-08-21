#!/usr/bin/env python3
"""Continue the three finite-inadmissible PCFNO bump trajectories through H79.

The exact stored post-boundary failed proposal is the recurrence entry.  Nothing
is repaired.  The script binds itself to the frozen W26-L2 checkpoint/runtime,
checks one-call replay compatibility, and distinguishes model recovery from
input/output boundary recovery.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no.evaluate_pcno_euler2d_residual import (
    CAUSAL_BOUNDARY_MODE,
    NATIVE_CAUSAL_BOUNDARY_SCOPE,
    PCNOEuler2DShardStore,
    apply_causal_boundary_conservative,
    build_graph_causal_boundary_policy,
    build_model,
    load_checkpoint,
    model_call,
    preprocessing_contract_audit,
    select_device,
    sha256_file,
    synchronize,
)
from scripts.time_dependent_no.train_pcno_bump_gradient_ablation import (
    B1_DATA_MANIFEST_SHA256,
    B1_SOURCE_SET_SHA256,
    checkpoint_differential_branch_mode,
    verify_frozen_b1_sources,
)

SCHEMA = "w26_l2_pcfno_inadmissibility_continuation_v1"
WORKING_ID = "W26-L2-PCFNO-C1-S20260718"
CHECKPOINT_SHA256 = "ff5c24fb8d1be813b97d44ebdfde8693d196c467adf4a9d5ddc7c1d4d28f9d0d"
TRAJECTORY_KEYS = ("54", "227", "233")
NUM_STEPS = 79
REPLAY_MAX_ABS = 1.0e-4
REPLAY_RELATIVE_L2 = 1.0e-6
NODE_TYPE_NAMES = {0: "normal", 1: "wall", 2: "outflow", 3: "inflow"}
EXPECTED_FAILURES: Mapping[str, Mapping[str, Any]] = {
    "54": {
        "call": 53,
        "node": 12977,
        "node_type": 0,
        "internal_energy": -0.1937860238,
    },
    "227": {
        "call": 78,
        "node": 19322,
        "node_type": 0,
        "internal_energy": -0.6283703198,
    },
    "233": {
        "call": 52,
        "node": 35,
        "node_type": 1,
        "internal_energy": -3.0206653809,
    },
}
SOURCE_PATHS = (
    "docs/time_dependent_no/W26_L2_PCFNO_INADMISSIBILITY_CONTINUATION_PREREGISTRATION.md",
    "scripts/time_dependent_no/evaluate_pcfno_inadmissibility_continuation.py",
    "scripts/time_dependent_no/visualize_pcfno_inadmissibility_continuation.py",
    "tests/time_dependent_no/test_pcfno_inadmissibility_continuation.py",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--training-data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--strict-rollout-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    return parser.parse_args(argv)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty CSV")
    columns: list[str] = []
    for row in rows:
        for name in row:
            if name not in columns:
                columns.append(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    name: (
                        json.dumps(_json_safe(value), sort_keys=True)
                        if isinstance(value, (dict, list, tuple))
                        else _json_safe(value)
                    )
                    for name, value in row.items()
                }
            )


def _source_hashes() -> dict[str, str | None]:
    return {
        relative: sha256_file(ROOT / relative) if (ROOT / relative).is_file() else None
        for relative in SOURCE_PATHS
    }


def conservative_fields(
    state: np.ndarray, *, gamma: float = 1.4
) -> dict[str, np.ndarray]:
    value = np.asarray(state, dtype=np.float64)
    rho = value[..., 0]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        velocity_x = value[..., 1] / rho
        velocity_y = value[..., 2] / rho
        internal = value[..., 3] - 0.5 * (value[..., 1] ** 2 + value[..., 2] ** 2) / rho
        pressure = (gamma - 1.0) * internal
    return {
        "density": rho,
        "velocity_x": velocity_x,
        "velocity_y": velocity_y,
        "internal_energy": internal,
        "pressure": pressure,
    }


def invalid_node_mask(state: np.ndarray, *, gamma: float = 1.4) -> np.ndarray:
    value = np.asarray(state, dtype=np.float64)
    fields = conservative_fields(value, gamma=gamma)
    return (
        ~np.isfinite(value).all(axis=-1)
        | ~np.isfinite(fields["internal_energy"])
        | ~np.isfinite(fields["pressure"])
        | (fields["density"] <= 0.0)
        | (fields["internal_energy"] <= 0.0)
        | (fields["pressure"] <= 0.0)
    )


def state_diagnostics(
    state: np.ndarray,
    *,
    target: np.ndarray,
    weights: np.ndarray,
    component_scale: np.ndarray,
    node_type: np.ndarray,
    state_mean: np.ndarray,
    reference_max_abs: float,
    gamma: float,
) -> dict[str, Any]:
    value = np.asarray(state, dtype=np.float64)
    invalid = invalid_node_mask(value, gamma=gamma)
    finite = bool(np.isfinite(value).all())
    fields = conservative_fields(value, gamma=gamma)
    active_types = np.asarray(node_type, dtype=np.int64).reshape(-1)
    centered = (value - np.asarray(state_mean, dtype=np.float64)) / np.asarray(
        component_scale, dtype=np.float64
    )
    finite_centered = np.abs(centered[np.isfinite(centered)])

    def finite_minimum(array: np.ndarray) -> float | None:
        selected = np.asarray(array)[np.isfinite(array)]
        return None if not selected.size else float(np.min(selected))

    max_abs = None
    finite_value = np.abs(value[np.isfinite(value)])
    if finite_value.size:
        max_abs = float(np.max(finite_value))
    result: dict[str, Any] = {
        "failure_cause": (
            "nonfinite_state"
            if not finite
            else "inadmissible_state"
            if bool(np.any(invalid))
            else "admissible"
        ),
        "admissible": finite and not bool(np.any(invalid)),
        "all_finite": finite,
        "invalid_node_count": int(np.count_nonzero(invalid)),
        "min_density": finite_minimum(fields["density"]),
        "min_internal_energy": finite_minimum(fields["internal_energy"]),
        "min_pressure": finite_minimum(fields["pressure"]),
        "max_abs_conservative": max_abs,
        "max_abs_to_reference_max_ratio": (
            None if max_abs is None else max_abs / reference_max_abs
        ),
        "maximum_normalizer_excursion": (
            None if not finite_centered.size else float(np.max(finite_centered))
        ),
    }
    total_components = value.size
    for threshold in (6.0, 10.0):
        result[f"normalizer_excursion_fraction_gt_{int(threshold)}"] = float(
            np.count_nonzero(np.abs(centered) > threshold) / total_components
        )
    for code, name in NODE_TYPE_NAMES.items():
        result[f"invalid_{name}_node_count"] = int(
            np.count_nonzero(invalid & (active_types == code))
        )
    masks = {
        "all": np.ones(active_types.shape[0], dtype=bool),
        "normal": active_types == 0,
        "boundary": active_types != 0,
    }
    for name, mask in masks.items():
        result[f"{name}_proxy_scaled_relative_l2"] = weighted_scaled_relative_l2(
            value, target, weights, component_scale, mask=mask
        )
        pressure_error = (
            fields["pressure"] - conservative_fields(target, gamma=gamma)["pressure"]
        )
        result[f"{name}_pressure_rmse"] = weighted_scalar_rms(
            pressure_error, weights, mask=mask
        )
    return result


def weighted_scalar_rms(
    value: np.ndarray, weights: np.ndarray, *, mask: np.ndarray | None = None
) -> float | None:
    array = np.asarray(value, dtype=np.float64)
    measure = np.asarray(weights, dtype=np.float64).reshape(-1)
    selected = np.ones(measure.shape, dtype=bool) if mask is None else np.asarray(mask)
    if not np.any(selected) or not np.isfinite(array[selected]).all():
        return None
    denominator = float(np.sum(measure[selected]))
    if denominator <= 0.0:
        return None
    return float(
        np.sqrt(np.sum(measure[selected] * array[selected] ** 2) / denominator)
    )


def weighted_scaled_rms(
    value: np.ndarray,
    weights: np.ndarray,
    component_scale: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> float | None:
    array = np.asarray(value, dtype=np.float64)
    measure = np.asarray(weights, dtype=np.float64).reshape(-1)
    selected = np.ones(measure.shape, dtype=bool) if mask is None else np.asarray(mask)
    if not np.any(selected) or not np.isfinite(array[selected]).all():
        return None
    scaled = array[selected] / np.asarray(component_scale, dtype=np.float64)
    denominator = float(np.sum(measure[selected]) * array.shape[-1])
    if denominator <= 0.0:
        return None
    return float(np.sqrt(np.sum(measure[selected, None] * scaled**2) / denominator))


def weighted_scaled_relative_l2(
    prediction: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    component_scale: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> float | None:
    prediction64 = np.asarray(prediction, dtype=np.float64)
    target64 = np.asarray(target, dtype=np.float64)
    measure = np.asarray(weights, dtype=np.float64).reshape(-1)
    selected = np.ones(measure.shape, dtype=bool) if mask is None else np.asarray(mask)
    if (
        not np.any(selected)
        or not np.isfinite(prediction64[selected]).all()
        or not np.isfinite(target64[selected]).all()
    ):
        return None
    scale = np.asarray(component_scale, dtype=np.float64)
    difference = (prediction64[selected] - target64[selected]) / scale
    reference = target64[selected] / scale
    numerator = float(np.sum(measure[selected, None] * difference**2))
    denominator = float(np.sum(measure[selected, None] * reference**2))
    if denominator <= 0.0:
        return None
    return float(np.sqrt(numerator / denominator))


def transition_label(
    before: Mapping[str, Any], after: Mapping[str, Any], stage: str
) -> str:
    before_valid = bool(before["admissible"])
    after_valid = bool(after["admissible"])
    if before_valid and after_valid:
        return f"{stage}_admissible"
    if before_valid and not after_valid:
        return f"{stage}_introduced_inadmissibility"
    if not before_valid and after_valid:
        return f"{stage}_recovery"
    return f"{stage}_persistent_inadmissibility"


def inadmissibility_episodes(
    calls: Sequence[int], invalid: Sequence[bool], *, terminal_call: int
) -> list[dict[str, Any]]:
    if len(calls) != len(invalid):
        raise ValueError("calls and invalid flags must have equal length")
    episodes: list[dict[str, Any]] = []
    start: int | None = None
    prior: int | None = None
    for call, flag in zip(calls, invalid, strict=True):
        call = int(call)
        if flag and start is None:
            start = call
        if not flag and start is not None:
            episodes.append(
                {
                    "start_call": start,
                    "end_call": int(prior),
                    "duration_calls": int(prior) - start + 1,
                    "recovered": True,
                    "recovery_call": call,
                    "durable_through_h79": all(
                        not bool(later_flag)
                        for later_call, later_flag in zip(calls, invalid, strict=True)
                        if int(later_call) >= call
                    )
                    and terminal_call == NUM_STEPS,
                }
            )
            start = None
        prior = call
    if start is not None and prior is not None:
        episodes.append(
            {
                "start_call": start,
                "end_call": prior,
                "duration_calls": prior - start + 1,
                "recovered": False,
                "recovery_call": None,
                "durable_through_h79": False,
            }
        )
    return episodes


def failure_location(
    state: np.ndarray, node_type: np.ndarray, *, gamma: float
) -> dict[str, Any]:
    value = np.asarray(state, dtype=np.float64)
    types = np.asarray(node_type, dtype=np.int64).reshape(-1)
    nonfinite = np.flatnonzero(~np.isfinite(value).all(axis=1))
    if nonfinite.size:
        index = int(nonfinite[0])
        quantity = "nonfinite_components"
        quantity_value = None
    else:
        fields = conservative_fields(value, gamma=gamma)
        candidates = [
            (float(np.min(fields[name])), name, int(np.argmin(fields[name])))
            for name in ("density", "internal_energy", "pressure")
        ]
        quantity_value, quantity, index = min(candidates, key=lambda item: item[0])
    return {
        "node": index,
        "node_type": int(types[index]),
        "node_type_name": NODE_TYPE_NAMES.get(int(types[index]), "unknown"),
        "quantity": quantity,
        "value": quantity_value,
    }


def replay_error(
    observed: np.ndarray,
    expected: np.ndarray,
    weights: np.ndarray,
    state_scale: np.ndarray,
) -> dict[str, Any]:
    difference = np.asarray(observed, dtype=np.float64) - np.asarray(
        expected, dtype=np.float64
    )
    return {
        "exact_equal": bool(np.array_equal(observed, expected)),
        "max_abs": float(np.max(np.abs(difference))),
        "proxy_scaled_relative_l2": weighted_scaled_relative_l2(
            observed, expected, weights, state_scale
        ),
    }


def error_decomposition(
    failed_proposal: np.ndarray,
    teacher_proposal: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    state_scale: np.ndarray,
    *,
    mask: np.ndarray,
) -> dict[str, Any]:
    total = np.asarray(failed_proposal, dtype=np.float64) - target
    fresh = np.asarray(teacher_proposal, dtype=np.float64) - target
    propagated = np.asarray(failed_proposal, dtype=np.float64) - teacher_proposal
    reconstruction = total - fresh - propagated
    selected = np.asarray(mask, dtype=bool)
    measure = np.asarray(weights, dtype=np.float64).reshape(-1)[selected]
    scale = np.asarray(state_scale, dtype=np.float64)
    fresh_scaled = fresh[selected] / scale
    propagated_scaled = propagated[selected] / scale
    inner = float(np.sum(measure[:, None] * fresh_scaled * propagated_scaled))
    fresh_energy = float(np.sum(measure[:, None] * fresh_scaled**2))
    propagated_energy = float(np.sum(measure[:, None] * propagated_scaled**2))
    cosine = (
        None
        if fresh_energy <= 0.0 or propagated_energy <= 0.0
        else inner / math.sqrt(fresh_energy * propagated_energy)
    )
    return {
        "total_norm": weighted_scaled_rms(total, weights, state_scale, mask=selected),
        "fresh_defect_norm": weighted_scaled_rms(
            fresh, weights, state_scale, mask=selected
        ),
        "propagated_response_norm": weighted_scaled_rms(
            propagated, weights, state_scale, mask=selected
        ),
        "reconstruction_residual_norm": weighted_scaled_rms(
            reconstruction, weights, state_scale, mask=selected
        ),
        "fresh_propagated_cosine": cosine,
    }


def _prefix(name: str, values: Mapping[str, Any]) -> dict[str, Any]:
    return {f"{name}_{key}": value for key, value in values.items()}


def _distance_to_set(point: np.ndarray, candidates: np.ndarray) -> float | None:
    if candidates.size == 0:
        return None
    return float(np.min(np.linalg.norm(candidates - point[None, :], axis=1)))


def _pressure_front_nodes(
    state: np.ndarray, edges: np.ndarray, *, gamma: float
) -> np.ndarray:
    pressure = conservative_fields(state, gamma=gamma)["pressure"]
    edge_index = np.asarray(edges, dtype=np.int64)
    jumps = np.abs(pressure[edge_index[:, 0]] - pressure[edge_index[:, 1]])
    finite = jumps[np.isfinite(jumps)]
    if not finite.size:
        return np.empty(0, dtype=np.int64)
    threshold = float(np.quantile(finite, 0.9))
    selected = edge_index[np.isfinite(jumps) & (jumps >= threshold)]
    return np.unique(selected.reshape(-1))


def _graph_degree(edges: np.ndarray, node: int, num_nodes: int) -> int:
    edge_index = np.asarray(edges, dtype=np.int64)
    degree = np.bincount(edge_index.reshape(-1), minlength=num_nodes)
    return int(degree[node])


def _neighbor_indices(edges: np.ndarray, node: int) -> np.ndarray:
    edge_index = np.asarray(edges, dtype=np.int64)
    left = edge_index[edge_index[:, 0] == node, 1]
    right = edge_index[edge_index[:, 1] == node, 0]
    return np.unique(np.concatenate(([node], left, right)))


def _percentile_rank(values: Sequence[float], observed: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.mean(array <= observed))


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def validate_strict_bundle(
    arrays: Mapping[str, np.ndarray],
    *,
    key: str,
    checkpoint_sha256: str,
    test_manifest_digest: str,
    reference: np.ndarray,
    positions: np.ndarray,
    node_type: np.ndarray,
    gamma: float,
) -> dict[str, Any]:
    expected = EXPECTED_FAILURES[key]

    def scalar(name: str) -> Any:
        if name not in arrays:
            raise KeyError(f"strict trajectory {key} lacks {name}")
        return np.asarray(arrays[name]).item()

    checks = {
        "schema": str(scalar("schema")) == "pcno_euler2d_official_rollout_v1",
        "trajectory": str(scalar("trajectory_key")) == key,
        "checkpoint_sha256": str(scalar("checkpoint_sha256")) == checkpoint_sha256,
        "test_manifest_digest": str(scalar("test_manifest_digest"))
        == test_manifest_digest,
        "boundary_mode": str(scalar("baseline_boundary_mode")) == CAUSAL_BOUNDARY_MODE,
        "valid_length": int(scalar("baseline_valid_length"))
        == int(expected["call"]) - 1,
        "failure_call": int(scalar("baseline_failure_call")) == int(expected["call"]),
        "failure_cause": str(scalar("baseline_failure_cause")) == "inadmissible_state",
        "reference": np.array_equal(
            arrays["reference_targets_conservative"], reference[1:]
        )
        and np.array_equal(arrays["initial_conservative"], reference[0]),
        "positions": np.array_equal(arrays["positions"], positions),
        "node_type": np.array_equal(arrays["node_type"].reshape(-1), node_type),
        "failed_proposal_present": "baseline_failed_proposal" in arrays,
    }
    failed = np.asarray(arrays["baseline_failed_proposal"], dtype=np.float32)
    location = failure_location(failed, node_type, gamma=gamma)
    fields = conservative_fields(failed, gamma=gamma)
    checks.update(
        {
            "failure_node": location["node"] == int(expected["node"]),
            "failure_node_type": location["node_type"] == int(expected["node_type"]),
            "failure_internal_energy": math.isclose(
                float(fields["internal_energy"][int(expected["node"])]),
                float(expected["internal_energy"]),
                rel_tol=0.0,
                abs_tol=5.0e-5,
            ),
        }
    )
    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    if failed_checks:
        raise ValueError(f"strict trajectory {key} failed closure: {failed_checks}")
    return {"checks": checks, "failure_location": location}


@torch.no_grad()
def contract_call(
    model: torch.nn.Module,
    sample: Mapping[str, torch.Tensor],
    current: torch.Tensor,
    boundary_policy: Mapping[str, Any],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    synchronize(device)
    model_current = apply_causal_boundary_conservative(
        current,
        boundary_policy,
        gamma=float(model.gamma),
        scope=NATIVE_CAUSAL_BOUNDARY_SCOPE,
    )
    raw = model_call(model, sample, model_current, branch_gains=None)
    deployed = apply_causal_boundary_conservative(
        raw,
        boundary_policy,
        gamma=float(model.gamma),
        scope=NATIVE_CAUSAL_BOUNDARY_SCOPE,
    )
    synchronize(device)
    return model_current, raw, deployed


def _as_numpy(value: torch.Tensor) -> np.ndarray:
    return value[0].detach().float().cpu().numpy().copy()


def run_case(
    model: torch.nn.Module,
    store: PCNOEuler2DShardStore,
    key: str,
    *,
    strict_rollout_dir: Path,
    checkpoint: Mapping[str, Any],
    checkpoint_sha256: str,
    device: torch.device,
    training_machs: Sequence[float],
    training_node_counts: Sequence[float],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, np.ndarray]]:
    expected = EXPECTED_FAILURES[key]
    failure_call = int(expected["call"])
    step_stride = int(checkpoint["step_stride"])
    if step_stride != 1:
        raise ValueError("the registered PCFNO continuation requires step_stride=1")
    states = store.states(key)
    if states.shape[0] < NUM_STEPS + 1:
        raise ValueError(f"trajectory {key} lacks H79 reference frames")
    reference = np.asarray(states[: NUM_STEPS + 1], dtype=np.float32)
    positions = np.asarray(store.array(key, "nodes"), dtype=np.float32)
    edges = np.asarray(store.array(key, "edges"), dtype=np.int64)
    node_type = np.asarray(store.array(key, "node_type"), dtype=np.int64).reshape(-1)
    weights = np.asarray(store.array(key, "node_weights"), dtype=np.float64).sum(
        axis=-1
    )
    strict_path = strict_rollout_dir / "trajectories" / f"trajectory_{key}.npz"
    strict = _load_npz(strict_path)
    strict_closure = validate_strict_bundle(
        strict,
        key=key,
        checkpoint_sha256=checkpoint_sha256,
        test_manifest_digest=store.manifest_digest,
        reference=reference,
        positions=positions,
        node_type=node_type,
        gamma=float(model.gamma),
    )
    sample = store.tensor_sample(key, 0, step_stride=1, device=device)
    native_contract = checkpoint["boundary_contract"]
    boundary_policy, boundary_metadata = build_graph_causal_boundary_policy(
        store,
        key,
        device=device,
        max_source_hops=int(native_contract["max_source_hops"]),
        rho_inf=float(native_contract["rho_inf"]),
        p_inf=float(native_contract["p_inf"]),
    )
    expected_digest = native_contract.get("policy_digests", {}).get(key)
    if (
        expected_digest is not None
        and boundary_metadata["policy_digest"] != expected_digest
    ):
        raise ValueError(f"trajectory {key} boundary-policy digest changed")

    state_scale = np.asarray(
        checkpoint["normalization"]["state_scale"], dtype=np.float64
    )
    state_mean = np.asarray(checkpoint["normalization"]["state_mean"], dtype=np.float64)
    residual_scale = np.asarray(
        checkpoint["normalization"].get("residual_scale", state_scale),
        dtype=np.float64,
    )
    reference_max_abs = float(np.max(np.abs(reference)))
    node_count = reference.shape[1]
    deployed_states = np.full((NUM_STEPS + 1, node_count, 4), np.nan, dtype=np.float32)
    raw_proposals = np.full((NUM_STEPS, node_count, 4), np.nan, dtype=np.float32)
    model_currents = np.full((NUM_STEPS, node_count, 4), np.nan, dtype=np.float32)
    deployed_states[0] = strict["initial_conservative"]
    strict_predictions = np.asarray(
        strict["pcno_baseline_predictions_conservative"], dtype=np.float32
    )
    deployed_states[1:failure_call] = strict_predictions
    failed_proposal = np.asarray(strict["baseline_failed_proposal"], dtype=np.float32)
    deployed_states[failure_call] = failed_proposal

    last_valid_tensor = torch.as_tensor(
        deployed_states[failure_call - 1][None], dtype=torch.float32, device=device
    )
    replay_model_current, replay_raw, replay_deployed = contract_call(
        model, sample, last_valid_tensor, boundary_policy, device=device
    )
    replay_deployed_np = _as_numpy(replay_deployed)
    replay = replay_error(replay_deployed_np, failed_proposal, weights, state_scale)
    replay["max_abs_gate"] = REPLAY_MAX_ABS
    replay["relative_l2_gate"] = REPLAY_RELATIVE_L2
    replay["passed"] = bool(
        replay["max_abs"] <= REPLAY_MAX_ABS
        and replay["proxy_scaled_relative_l2"] is not None
        and replay["proxy_scaled_relative_l2"] <= REPLAY_RELATIVE_L2
    )
    if not replay["passed"]:
        raise ValueError(f"trajectory {key} failed strict one-call replay: {replay}")
    model_currents[failure_call - 1] = _as_numpy(replay_model_current)
    raw_proposals[failure_call - 1] = _as_numpy(replay_raw)

    reference_previous = torch.as_tensor(
        reference[failure_call - 1][None], dtype=torch.float32, device=device
    )
    teacher_model_current, teacher_raw, teacher_deployed = contract_call(
        model, sample, reference_previous, boundary_policy, device=device
    )
    teacher_deployed_np = _as_numpy(teacher_deployed)

    current = torch.as_tensor(failed_proposal[None], dtype=torch.float32, device=device)
    termination_call: int | None = None
    for call in range(failure_call + 1, NUM_STEPS + 1):
        model_current, raw, deployed = contract_call(
            model, sample, current, boundary_policy, device=device
        )
        model_currents[call - 1] = _as_numpy(model_current)
        raw_proposals[call - 1] = _as_numpy(raw)
        deployed_states[call] = _as_numpy(deployed)
        current = deployed
        if not np.isfinite(deployed_states[call]).all():
            termination_call = call
            break

    masks = {
        "all": np.ones(node_count, dtype=bool),
        "normal": node_type == 0,
        "boundary": node_type != 0,
    }
    call_rows: list[dict[str, Any]] = []
    last_call = NUM_STEPS if termination_call is None else termination_call
    for call in range(1, last_call + 1):
        deployed = deployed_states[call]
        current_metrics = state_diagnostics(
            deployed_states[call - 1],
            target=reference[call - 1],
            weights=weights,
            component_scale=state_scale,
            node_type=node_type,
            state_mean=state_mean,
            reference_max_abs=reference_max_abs,
            gamma=float(model.gamma),
        )
        deployed_metrics = state_diagnostics(
            deployed,
            target=reference[call],
            weights=weights,
            component_scale=state_scale,
            node_type=node_type,
            state_mean=state_mean,
            reference_max_abs=reference_max_abs,
            gamma=float(model.gamma),
        )
        predicted_increment = deployed - deployed_states[call - 1]
        reference_increment = reference[call] - reference[call - 1]
        predicted_pressure_increment = (
            conservative_fields(deployed, gamma=float(model.gamma))["pressure"]
            - conservative_fields(deployed_states[call - 1], gamma=float(model.gamma))[
                "pressure"
            ]
        )
        reference_pressure_increment = (
            conservative_fields(reference[call], gamma=float(model.gamma))["pressure"]
            - conservative_fields(reference[call - 1], gamma=float(model.gamma))[
                "pressure"
            ]
        )
        row: dict[str, Any] = {
            "trajectory": key,
            "call": call,
            "physical_time": float(call * float(store.manifest["dt"])),
            "source": (
                "strict_prefix"
                if call < failure_call
                else "stored_failed_proposal"
                if call == failure_call
                else "finite_invalid_continuation"
            ),
            "is_last_admissible_call": call == failure_call - 1,
            "is_first_inadmissible_call": call == failure_call,
            "deployed_increment_proxy_scaled_rms": weighted_scaled_rms(
                predicted_increment, weights, residual_scale
            ),
            "reference_increment_proxy_scaled_rms": weighted_scaled_rms(
                reference_increment, weights, residual_scale
            ),
            "increment_error_proxy_scaled_rms": weighted_scaled_rms(
                predicted_increment - reference_increment, weights, residual_scale
            ),
            "pressure_residual_error_proxy_rms": weighted_scalar_rms(
                predicted_pressure_increment - reference_pressure_increment,
                weights,
            ),
            **_prefix("current", current_metrics),
            **_prefix("deployed", deployed_metrics),
        }
        if call >= failure_call and np.isfinite(model_currents[call - 1]).all():
            model_current_metrics = state_diagnostics(
                model_currents[call - 1],
                target=reference[call - 1],
                weights=weights,
                component_scale=state_scale,
                node_type=node_type,
                state_mean=state_mean,
                reference_max_abs=reference_max_abs,
                gamma=float(model.gamma),
            )
            raw_metrics = state_diagnostics(
                raw_proposals[call - 1],
                target=reference[call],
                weights=weights,
                component_scale=state_scale,
                node_type=node_type,
                state_mean=state_mean,
                reference_max_abs=reference_max_abs,
                gamma=float(model.gamma),
            )
            row.update(
                {
                    "input_transition": transition_label(
                        current_metrics, model_current_metrics, "input_boundary"
                    ),
                    "model_transition": transition_label(
                        model_current_metrics, raw_metrics, "model"
                    ),
                    "output_transition": transition_label(
                        raw_metrics, deployed_metrics, "output_boundary"
                    ),
                    **_prefix("model_current", model_current_metrics),
                    **_prefix("raw", raw_metrics),
                }
            )
        else:
            row.update(
                {
                    "input_transition": None,
                    "model_transition": None,
                    "output_transition": None,
                }
            )
        call_rows.append(row)

    failure_node = int(expected["node"])
    last_valid = deployed_states[failure_call - 1]
    target = reference[failure_call]
    previous_target = reference[failure_call - 1]
    failure_fields = conservative_fields(failed_proposal, gamma=float(model.gamma))
    last_fields = conservative_fields(last_valid, gamma=float(model.gamma))
    previous_fields = conservative_fields(previous_target, gamma=float(model.gamma))
    target_fields = conservative_fields(target, gamma=float(model.gamma))
    teacher_metrics = state_diagnostics(
        teacher_deployed_np,
        target=target,
        weights=weights,
        component_scale=state_scale,
        node_type=node_type,
        state_mean=state_mean,
        reference_max_abs=reference_max_abs,
        gamma=float(model.gamma),
    )
    decomposition = {
        name: error_decomposition(
            failed_proposal,
            teacher_deployed_np,
            target,
            weights,
            state_scale,
            mask=mask,
        )
        for name, mask in masks.items()
    }
    invalid_flags = [
        not bool(row["deployed_admissible"])
        for row in call_rows
        if int(row["call"]) >= failure_call
    ]
    invalid_calls = [
        int(row["call"]) for row in call_rows if int(row["call"]) >= failure_call
    ]
    episodes = inadmissibility_episodes(
        invalid_calls, invalid_flags, terminal_call=last_call
    )
    recovery_calls = [
        int(row["call"])
        for row in call_rows
        if int(row["call"]) > failure_call
        and bool(row["deployed_admissible"])
        and not bool(row["current_admissible"])
    ]
    model_recovery_calls = [
        int(row["call"])
        for row in call_rows
        if row.get("model_transition") == "model_recovery"
    ]
    input_recovery_calls = [
        int(row["call"])
        for row in call_rows
        if row.get("input_transition") == "input_boundary_recovery"
    ]
    output_recovery_calls = [
        int(row["call"])
        for row in call_rows
        if row.get("output_transition") == "output_boundary_recovery"
    ]

    front_nodes = _pressure_front_nodes(target, edges, gamma=float(model.gamma))
    boundary_nodes = np.flatnonzero(node_type != 0)
    neighbors = _neighbor_indices(edges, failure_node)
    last_pressure = last_fields["pressure"]
    true_pressure_residual = target_fields["pressure"] - previous_fields["pressure"]
    predicted_pressure_residual = failure_fields["pressure"] - last_fields["pressure"]
    pre_failure_error = call_rows[failure_call - 2][
        "deployed_all_proxy_scaled_relative_l2"
    ]
    post_failure_rows = [row for row in call_rows if int(row["call"]) >= failure_call]
    error_values = [
        float(row["deployed_all_proxy_scaled_relative_l2"])
        for row in post_failure_rows
        if row["deployed_all_proxy_scaled_relative_l2"] is not None
    ]
    durable_model_calls = [
        call
        for call in model_recovery_calls
        if last_call == NUM_STEPS
        and all(
            bool(row["deployed_admissible"])
            for row in call_rows
            if int(row["call"]) >= call
        )
    ]
    durable_boundary_calls = [
        call
        for call in input_recovery_calls + output_recovery_calls
        if last_call == NUM_STEPS
        and all(
            bool(row["deployed_admissible"])
            for row in call_rows
            if int(row["call"]) >= call
        )
    ]
    if durable_model_calls:
        recovery_verdict = "durable_model_self_recovery"
    elif model_recovery_calls:
        recovery_verdict = "temporary_model_self_recovery"
    elif durable_boundary_calls:
        recovery_verdict = "durable_boundary_recovery_only"
    elif input_recovery_calls or output_recovery_calls:
        recovery_verdict = "temporary_boundary_recovery_only"
    elif recovery_calls:
        recovery_verdict = "stage_attribution_unresolved_recovery"
    else:
        recovery_verdict = "no_deployed_recovery"

    case_summary: dict[str, Any] = {
        "trajectory": key,
        "failure_call": failure_call,
        "last_admissible_call": failure_call - 1,
        "failure_location": strict_closure["failure_location"],
        "strict_artifact": {
            "path": str(strict_path),
            "sha256": sha256_file(strict_path),
            "closure": strict_closure,
        },
        "boundary_policy": boundary_metadata,
        "replay": replay,
        "continuation": {
            "last_retained_call": last_call,
            "termination_call": termination_call,
            "terminal_admissible": bool(call_rows[-1]["deployed_admissible"]),
            "recovery_verdict": recovery_verdict,
            "deployed_reentry_recovery_calls": recovery_calls,
            "model_recovery_calls": model_recovery_calls,
            "input_boundary_recovery_calls": input_recovery_calls,
            "output_boundary_recovery_calls": output_recovery_calls,
            "durable_model_recovery_calls": durable_model_calls,
            "durable_boundary_recovery_calls": durable_boundary_calls,
            "inadmissibility_episodes": episodes,
            "pre_failure_relative_l2": pre_failure_error,
            "terminal_relative_l2": call_rows[-1][
                "deployed_all_proxy_scaled_relative_l2"
            ],
            "minimum_post_failure_relative_l2": min(error_values)
            if error_values
            else None,
            "accuracy_returned_below_pre_failure": bool(
                error_values and min(error_values) < float(pre_failure_error)
            ),
            "registered_blowup": any(
                (
                    row["deployed_max_abs_to_reference_max_ratio"] is not None
                    and float(row["deployed_max_abs_to_reference_max_ratio"]) >= 100.0
                )
                or (
                    row["deployed_all_proxy_scaled_relative_l2"] is not None
                    and float(row["deployed_all_proxy_scaled_relative_l2"]) >= 10.0
                )
                for row in post_failure_rows
            ),
        },
        "teacher_forced_failure_call": {
            "admissibility": teacher_metrics,
            "decomposition": decomposition,
            "dominant_source_normal": (
                "propagated_response"
                if float(decomposition["normal"]["propagated_response_norm"])
                > float(decomposition["normal"]["fresh_defect_norm"])
                else "fresh_defect"
            ),
        },
        "hardness": {
            "mach": float(store.entry(key)["mach"]),
            "mach_training_percentile_rank": _percentile_rank(
                training_machs, float(store.entry(key)["mach"])
            ),
            "num_nodes": node_count,
            "node_count_training_percentile_rank": _percentile_rank(
                training_node_counts, float(node_count)
            ),
            "failure_node_graph_degree": _graph_degree(edges, failure_node, node_count),
            "failure_node_distance_to_nearest_boundary": _distance_to_set(
                positions[failure_node], positions[boundary_nodes]
            ),
            "failure_node_distance_to_reference_pressure_front_proxy": _distance_to_set(
                positions[failure_node], positions[front_nodes]
            ),
            "pressure_front_proxy_quantile": 0.9,
            "failure_node_reference_internal_energy_before": float(
                previous_fields["internal_energy"][failure_node]
            ),
            "failure_node_pcfno_internal_energy_before": float(
                last_fields["internal_energy"][failure_node]
            ),
            "failure_node_reference_internal_energy_after": float(
                target_fields["internal_energy"][failure_node]
            ),
            "failure_node_failed_internal_energy": float(
                failure_fields["internal_energy"][failure_node]
            ),
            "failure_node_local_reference_pressure_range_before": float(
                np.ptp(previous_fields["pressure"][neighbors])
            ),
            "failure_node_local_pcfno_pressure_range_before": float(
                np.ptp(last_pressure[neighbors])
            ),
            "failure_node_true_pressure_residual": float(
                true_pressure_residual[failure_node]
            ),
            "failure_node_predicted_pressure_residual": float(
                predicted_pressure_residual[failure_node]
            ),
            "failure_node_pressure_residual_error": float(
                predicted_pressure_residual[failure_node]
                - true_pressure_residual[failure_node]
            ),
        },
        "claim_boundary": (
            "finite-invalid recurrence stress test with proxy weights; not a valid "
            "physical rollout, conservation result, or general PCFNO stability claim"
        ),
    }
    arrays = {
        "schema": np.asarray(SCHEMA),
        "working_id": np.asarray(WORKING_ID),
        "trajectory_key": np.asarray(key),
        "all_temporal_frames_retained": np.asarray(termination_call is None),
        "reference_states_conservative": reference,
        "deployed_states_conservative": deployed_states,
        "model_currents_conservative": model_currents,
        "raw_proposals_conservative": raw_proposals,
        "teacher_model_current_at_failure": _as_numpy(teacher_model_current),
        "teacher_raw_proposal_at_failure": _as_numpy(teacher_raw),
        "teacher_deployed_proposal_at_failure": teacher_deployed_np,
        "positions": positions,
        "edges": edges,
        "node_type": node_type,
        "reconstructed_node_weights_proxy": weights.astype(np.float32),
        "physical_times": np.arange(NUM_STEPS + 1, dtype=np.float64)
        * float(store.manifest["dt"]),
        "physical_delta_t": np.asarray(float(store.manifest["dt"]), dtype=np.float64),
        "failure_call": np.asarray(failure_call, dtype=np.int64),
        "last_retained_call": np.asarray(last_call, dtype=np.int64),
        "checkpoint_sha256": np.asarray(checkpoint_sha256),
        "test_manifest_digest": np.asarray(store.manifest_digest),
        "boundary_policy_digest": np.asarray(boundary_metadata["policy_digest"]),
    }
    return call_rows, case_summary, arrays


def _gradient_zero_audit(model: torch.nn.Module) -> dict[str, Any]:
    backbone = getattr(model, "backbone", model)
    layers = []
    for index, module in enumerate(backbone.gws):
        weight = module.gw2.weight.detach()
        layers.append(
            {
                "layer": index,
                "element_count": int(weight.numel()),
                "nonzero_count": int(torch.count_nonzero(weight).cpu()),
                "max_abs": float(torch.max(torch.abs(weight)).cpu()),
            }
        )
    if any(row["nonzero_count"] != 0 for row in layers):
        raise ValueError("PCFNO checkpoint has a nonzero serialized gw2 weight")
    return {"exact_zero_gw2": True, "layers": layers}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not args.strict_rollout_dir.is_dir():
        raise FileNotFoundError(args.strict_rollout_dir)
    device = select_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("the registered production continuation requires CUDA")
    frozen_sources = verify_frozen_b1_sources()
    checkpoint_sha256 = sha256_file(args.checkpoint)
    if checkpoint_sha256 != CHECKPOINT_SHA256:
        raise ValueError("checkpoint SHA-256 differs from the registered PCFNO")
    checkpoint = load_checkpoint(args.checkpoint)
    if checkpoint_differential_branch_mode(checkpoint) != "no_gradient":
        raise ValueError("checkpoint is not the registered PCFNO mode")
    training_store = PCNOEuler2DShardStore(args.training_data_dir)
    test_store = PCNOEuler2DShardStore(args.data_dir)
    if training_store.manifest_digest != B1_DATA_MANIFEST_SHA256:
        raise ValueError("training data manifest differs from the registered B1 data")
    preprocessing = preprocessing_contract_audit(checkpoint, training_store, test_store)
    missing = sorted(set(TRAJECTORY_KEYS) - set(test_store.keys))
    if missing:
        raise KeyError(f"test store lacks registered cases: {missing}")
    model = build_model(checkpoint, device)
    model.eval()
    gradient_audit = _gradient_zero_audit(model)

    args.output_dir.mkdir(parents=True)
    trajectory_dir = args.output_dir / "trajectories"
    trajectory_dir.mkdir()
    source_hashes = _source_hashes()
    if any(value is None for value in source_hashes.values()):
        raise FileNotFoundError("one or more registered extension sources are absent")
    run_contract = {
        "schema": SCHEMA,
        "working_id": WORKING_ID,
        "status": "running",
        "trajectory_keys": list(TRAJECTORY_KEYS),
        "num_steps": NUM_STEPS,
        "precision": "fp32",
        "recurrence_entry": "exact_stored_post_output_boundary_failed_proposal",
        "interventions": [],
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": checkpoint_sha256,
            "differential_branch_mode": "no_gradient",
        },
        "data": {
            "training_dir": str(args.training_data_dir.resolve()),
            "training_manifest_digest": training_store.manifest_digest,
            "test_dir": str(args.data_dir.resolve()),
            "test_manifest_digest": test_store.manifest_digest,
        },
        "strict_rollout_dir": str(args.strict_rollout_dir.resolve()),
        "frozen_b1_source_set_sha256": B1_SOURCE_SET_SHA256,
        "frozen_b1_sources": frozen_sources,
        "extension_source_sha256": source_hashes,
        "preprocessing": preprocessing,
        "gradient_audit": gradient_audit,
        "replay_gates": {
            "max_abs": REPLAY_MAX_ABS,
            "proxy_scaled_relative_l2": REPLAY_RELATIVE_L2,
        },
    }
    write_json(args.output_dir / "run_contract.json", run_contract)

    training_machs = [
        float(training_store.entry(key)["mach"]) for key in training_store.keys
    ]
    training_node_counts = [
        float(training_store.entry(key)["num_nodes"]) for key in training_store.keys
    ]
    all_call_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    case_summaries: dict[str, Any] = {}
    for key in TRAJECTORY_KEYS:
        call_rows, case_summary, arrays = run_case(
            model,
            test_store,
            key,
            strict_rollout_dir=args.strict_rollout_dir,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            device=device,
            training_machs=training_machs,
            training_node_counts=training_node_counts,
        )
        bundle_path = trajectory_dir / f"trajectory_{key}.npz"
        np.savez_compressed(bundle_path, **arrays)
        case_summary["artifact"] = {
            "path": str(bundle_path.relative_to(args.output_dir)),
            "sha256": sha256_file(bundle_path),
        }
        write_json(trajectory_dir / f"trajectory_{key}.json", case_summary)
        all_call_rows.extend(call_rows)
        case_summaries[key] = case_summary
        case_rows.append(
            {
                "trajectory": key,
                "failure_call": case_summary["failure_call"],
                "failure_node": case_summary["failure_location"]["node"],
                "failure_node_type": case_summary["failure_location"]["node_type_name"],
                "recovery_verdict": case_summary["continuation"]["recovery_verdict"],
                "model_recovery_calls": case_summary["continuation"][
                    "model_recovery_calls"
                ],
                "input_boundary_recovery_calls": case_summary["continuation"][
                    "input_boundary_recovery_calls"
                ],
                "output_boundary_recovery_calls": case_summary["continuation"][
                    "output_boundary_recovery_calls"
                ],
                "terminal_admissible": case_summary["continuation"][
                    "terminal_admissible"
                ],
                "pre_failure_relative_l2": case_summary["continuation"][
                    "pre_failure_relative_l2"
                ],
                "terminal_relative_l2": case_summary["continuation"][
                    "terminal_relative_l2"
                ],
                "minimum_post_failure_relative_l2": case_summary["continuation"][
                    "minimum_post_failure_relative_l2"
                ],
                "accuracy_returned_below_pre_failure": case_summary["continuation"][
                    "accuracy_returned_below_pre_failure"
                ],
                "teacher_forced_admissible": case_summary[
                    "teacher_forced_failure_call"
                ]["admissibility"]["admissible"],
                "normal_fresh_defect_norm": case_summary["teacher_forced_failure_call"][
                    "decomposition"
                ]["normal"]["fresh_defect_norm"],
                "normal_propagated_response_norm": case_summary[
                    "teacher_forced_failure_call"
                ]["decomposition"]["normal"]["propagated_response_norm"],
                "normal_total_error_norm": case_summary["teacher_forced_failure_call"][
                    "decomposition"
                ]["normal"]["total_norm"],
                "registered_blowup": case_summary["continuation"]["registered_blowup"],
            }
        )

    write_csv(args.output_dir / "call_metrics.csv", all_call_rows)
    write_csv(args.output_dir / "case_metrics.csv", case_rows)
    summary = {
        "schema": SCHEMA,
        "working_id": WORKING_ID,
        "status": "completed",
        "checkpoint_sha256": checkpoint_sha256,
        "trajectory_count": len(case_rows),
        "model_self_recovery_count": sum(
            bool(row["model_recovery_calls"]) for row in case_rows
        ),
        "boundary_recovery_count": sum(
            bool(row["input_boundary_recovery_calls"])
            or bool(row["output_boundary_recovery_calls"])
            for row in case_rows
        ),
        "durable_recovery_count": sum(
            str(row["recovery_verdict"]).startswith("durable") for row in case_rows
        ),
        "accuracy_recovery_count": sum(
            bool(row["accuracy_returned_below_pre_failure"]) for row in case_rows
        ),
        "registered_blowup_count": sum(
            bool(row["registered_blowup"]) for row in case_rows
        ),
        "cases": case_summaries,
        "claim_boundary": (
            "finite-invalid stress continuation through H79; admissibility and "
            "proxy-weighted accuracy are reported separately"
        ),
    }
    write_json(args.output_dir / "summary.json", summary)
    run_contract["status"] = "completed"
    write_json(args.output_dir / "run_contract.json", run_contract)
    outputs = sorted(
        path
        for path in args.output_dir.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    manifest = {
        "schema": SCHEMA,
        "working_id": WORKING_ID,
        "status": "completed",
        "source_sha256": source_hashes,
        "output_count": len(outputs),
        "output_sha256": {
            str(path.relative_to(args.output_dir)).replace("\\", "/"): sha256_file(path)
            for path in outputs
        },
    }
    write_json(args.output_dir / "manifest.json", manifest)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run(args)
    print(json.dumps(_json_safe(summary), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
