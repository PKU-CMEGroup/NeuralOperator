#!/usr/bin/env python3
"""Evaluate a frozen PCNO residual baseline on the validated FV benchmark.

This entry point is intentionally separate from the CPG-bump evaluator.  It
uses the shock--vortex family's physical cell volumes, oriented faces, and
recorded accepted-substep boundary impulses.  The PCNO remains a state model:
its "boundary exchange" below is the exchange implied by its cell-total
change, not a decoded or supervised face flux.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utility.time_dependent_no.pcno_euler2d import (  # noqa: E402
    Euler2DNormalization,
    PCNOEuler2DResidual,
    PCNOEuler2DShardStore,
    parameter_count,
)
from utility.time_dependent_no.pcno_ripple_diagnostics import (  # noqa: E402
    raw_admissibility_summary,
    weighted_relative_l2_numpy,
)
from utility.time_dependent_no.shock_vortex_family import (  # noqa: E402
    load_shock_vortex_family_manifest,
)
from utility.time_dependent_no.shock_vortex_metrics import (  # noqa: E402
    endpoint_metrics,
    physical_call_metrics,
)

SCHEMA = "pcno_shock_vortex_physical_baseline_v1"
CHECKPOINT_SCHEMA_VERSION = 4
DEFAULT_ENDPOINT_CALLS = (1, 5, 10, 20, 40, 60)
CONTROL_PARAMETER_NAMES = ("vortex_epsilon", "vortex_y")
CONTROL_NEIGHBOR_COUNT = 4


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--family-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--trajectory-keys", nargs="*", default=None)
    parser.add_argument("--expected-trajectory-count", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=60)
    parser.add_argument(
        "--endpoint-calls", type=int, nargs="+", default=list(DEFAULT_ENDPOINT_CALLS)
    )
    parser.add_argument("--shock-quantile", type=float, default=0.9)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--amp", choices=("checkpoint", "none", "bf16", "fp16"), default="checkpoint"
    )
    parser.add_argument("--latency-warmup", type=int, default=3)
    parser.add_argument("--latency-repeats", type=int, default=20)
    parser.add_argument("--throughput-batch-size", type=int, default=8)
    parser.add_argument("--throughput-repeats", type=int, default=10)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if str(key) not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serialized = {}
            for key, value in row.items():
                safe = json_safe(value)
                if isinstance(safe, (dict, list)):
                    safe = json.dumps(safe, sort_keys=True, separators=(",", ":"))
                serialized[str(key)] = safe
            writer.writerow(serialized)


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def autocast_context(device: torch.device, amp: str):
    if amp == "none":
        return nullcontext()
    if device.type != "cuda":
        raise ValueError("mixed-precision evaluation requires CUDA")
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    required = {
        "checkpoint_schema_version",
        "model_state",
        "model_config",
        "normalization",
        "normalization_digest",
        "data_manifest_digest",
        "data_contract",
        "config_digest",
        "boundary_mode",
        "raw_recurrence",
        "inference_interventions",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"checkpoint is missing frozen contract fields: {missing}")
    if int(checkpoint["checkpoint_schema_version"]) != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported PCNO checkpoint schema")
    if checkpoint["boundary_mode"] != "model_all_nodes":
        raise ValueError("physical baseline requires legal model_all_nodes recurrence")
    if checkpoint["raw_recurrence"] is not True:
        raise ValueError("checkpoint does not declare raw recurrence")
    if any(bool(value) for value in checkpoint["inference_interventions"].values()):
        raise ValueError(
            "official physical baseline cannot contain inference intervention"
        )
    return dict(checkpoint)


def build_model(
    checkpoint: Mapping[str, Any], device: torch.device
) -> PCNOEuler2DResidual:
    normalization = Euler2DNormalization.from_mapping(checkpoint["normalization"])
    config = checkpoint["model_config"]
    model = PCNOEuler2DResidual(
        normalization=normalization,
        k_max=int(config["k_max"]),
        domain_lengths=tuple(config["domain_lengths"]),
        layers=tuple(config["layers"]),
        fc_dim=int(config["fc_dim"]),
        nmeasures=int(config["nmeasures"]),
        zero_initialize=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model


@torch.no_grad()
def model_call(
    model: PCNOEuler2DResidual,
    sample: Mapping[str, torch.Tensor],
    current: torch.Tensor,
    *,
    device: torch.device,
    amp: str,
) -> torch.Tensor:
    with autocast_context(device, amp):
        return model(
            current,
            node_mask=sample["node_mask"],
            nodes=sample["nodes"],
            node_weights=sample["node_weights"],
            node_rhos=sample["node_rhos"],
            directed_edges=sample["directed_edges"],
            edge_gradient_weights=sample["edge_gradient_weights"],
            node_type=sample["node_type"],
            mach=sample["mach"],
        )


def raw_rollout(
    model: PCNOEuler2DResidual,
    store: PCNOEuler2DShardStore,
    key: str,
    *,
    start_frame: int,
    num_steps: int,
    step_stride: int,
    device: torch.device,
    amp: str,
) -> dict[str, Any]:
    states = store.states(key)
    final_index = start_frame + num_steps * step_stride
    if final_index >= states.shape[0]:
        raise ValueError(f"trajectory {key} lacks requested target frame {final_index}")
    sample = store.tensor_sample(
        key, start_frame, step_stride=step_stride, device=device
    )
    current = sample["current"]
    predictions: list[np.ndarray] = []
    call_seconds: list[float] = []
    failed_proposal: np.ndarray | None = None
    failure_cause = "completed"
    minimums = {
        "min_density": math.inf,
        "min_internal_energy": math.inf,
        "min_pressure": math.inf,
    }
    for _call in range(1, num_steps + 1):
        synchronize(device)
        started = perf_counter()
        proposal = model_call(model, sample, current, device=device, amp=amp)
        synchronize(device)
        call_seconds.append(perf_counter() - started)
        proposal_np = proposal[0].float().cpu().numpy()
        admissibility = raw_admissibility_summary(proposal_np, gamma=model.gamma)
        for name in minimums:
            value = admissibility[name]
            if value is not None:
                minimums[name] = min(minimums[name], float(value))
        if not admissibility["all_finite"]:
            failure_cause = "nonfinite_state"
        elif not admissibility["all_admissible"]:
            failure_cause = "inadmissible_state"
        if failure_cause != "completed":
            failed_proposal = proposal_np
            break
        predictions.append(proposal_np.copy())
        current = proposal
    prediction_array = (
        np.asarray(predictions, dtype=np.float32)
        if predictions
        else np.empty((0, states.shape[1], 4), dtype=np.float32)
    )
    return {
        "predictions": prediction_array,
        "valid_length": len(predictions),
        "completed": len(predictions) == num_steps,
        "failure_cause": failure_cause,
        "failure_call": None if failed_proposal is None else len(predictions) + 1,
        "failed_proposal": failed_proposal,
        "call_seconds": call_seconds,
        **{
            name: None if not math.isfinite(value) else float(value)
            for name, value in minimums.items()
        },
    }


def characteristic_travel_summary(
    conservative_states: np.ndarray,
    volumes: np.ndarray,
    *,
    gamma: float,
    macro_delta_t: float,
    physical_horizon: float,
) -> dict[str, float | str]:
    """Report physical and cell-width travel without calling it solver CFL."""

    primitive = conservative_to_primitive_raw(conservative_states, gamma=gamma)
    characteristic_speed = np.linalg.norm(primitive[..., 1:3], axis=-1) + np.sqrt(
        gamma * primitive[..., 3] / primitive[..., 0]
    )
    cell_width = np.sqrt(np.asarray(volumes, dtype=np.float64)).reshape(1, -1)
    travel_per_call = characteristic_speed * float(macro_delta_t) / cell_width
    maximum_speed = float(np.max(characteristic_speed))
    minimum_cell_width = float(np.min(cell_width))
    return {
        "contract": (
            "reference |velocity|+sound-speed travel divided by sqrt(cell volume); "
            "an effective macro-call CFL proxy, not the reference solver CFL"
        ),
        "minimum_cell_width": minimum_cell_width,
        "maximum_characteristic_speed": maximum_speed,
        "effective_cfl_proxy_p99": float(np.quantile(travel_per_call, 0.99)),
        "effective_cfl_proxy_max": float(np.max(travel_per_call)),
        "maximum_characteristic_travel_physical": maximum_speed
        * float(physical_horizon),
        "maximum_characteristic_travel_cell_widths": maximum_speed
        * float(physical_horizon)
        / minimum_cell_width,
    }


def _validate_reference_geometry(
    reference: Mapping[str, np.ndarray],
) -> dict[str, float]:
    centers = np.asarray(reference["cell_centers"], dtype=np.float64)
    volumes = np.asarray(reference["cell_volume"], dtype=np.float64)
    face_centers = np.asarray(reference["face_centers"], dtype=np.float64)
    measures = np.asarray(reference["face_measure"], dtype=np.float64)
    normals = np.asarray(reference["face_normal"], dtype=np.float64)
    owner = np.asarray(reference["face_owner"], dtype=np.int64)
    neighbor = np.asarray(reference["face_neighbor"], dtype=np.int64)
    boundary_tag = np.asarray(reference["face_boundary_tag"], dtype=np.int64)
    if (
        centers.ndim != 2
        or centers.shape[1] != 2
        or volumes.shape != (centers.shape[0],)
    ):
        raise ValueError("invalid physical cell geometry")
    if face_centers.shape != normals.shape or face_centers.shape != (owner.size, 2):
        raise ValueError("invalid physical face geometry")
    if (
        measures.shape != owner.shape
        or neighbor.shape != owner.shape
        or boundary_tag.shape != owner.shape
    ):
        raise ValueError("physical face arrays disagree")
    if np.any(volumes <= 0.0) or np.any(measures <= 0.0):
        raise ValueError("physical volumes and face measures must be positive")
    if np.any(owner < 0) or np.any(owner >= centers.shape[0]) or np.any(neighbor < -1):
        raise ValueError("oriented face connectivity is invalid")
    interior = neighbor >= 0
    if np.any(boundary_tag[interior] != 0) or np.any(boundary_tag[~interior] <= 0):
        raise ValueError("face connectivity and boundary tags disagree")
    if np.any(neighbor[interior] >= centers.shape[0]):
        raise ValueError("interior neighbor lies outside the cell axis")
    unit_error = float(np.max(np.abs(np.linalg.norm(normals, axis=1) - 1.0)))
    interior_orientation = np.einsum(
        "ij,ij->i",
        centers[neighbor[interior]] - centers[owner[interior]],
        normals[interior],
    )
    boundary_orientation = np.einsum(
        "ij,ij->i",
        face_centers[~interior] - centers[owner[~interior]],
        normals[~interior],
    )
    if (
        unit_error > 1.0e-12
        or np.any(interior_orientation <= 0.0)
        or np.any(boundary_orientation <= 0.0)
    ):
        raise ValueError("face normal or owner-orientation contract failed")
    return {
        "maximum_face_normal_unit_error": unit_error,
        "minimum_owner_oriented_face_distance": float(
            min(np.min(interior_orientation), np.min(boundary_orientation))
        ),
    }


def load_reference(
    family_root: Path, store: PCNOEuler2DShardStore, key: str
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    artifact_path = family_root / key / "reference.npz"
    expected_digest = store.entry(key).get("source_reference_sha256")
    digest = sha256_file(artifact_path)
    if digest != expected_digest:
        raise ValueError(f"reference digest mismatch for trajectory {key}")
    names = (
        "conservative_states",
        "physical_times",
        "cumulative_accepted_substep_face_impulses",
        "cell_centers",
        "cell_volume",
        "face_centers",
        "face_measure",
        "face_normal",
        "face_owner",
        "face_neighbor",
        "face_axis",
        "face_boundary_tag",
        "boundary_tag_names_json",
        "interval_boundary_exchange",
    )
    with np.load(artifact_path, allow_pickle=False) as artifact:
        missing = sorted(set(names) - set(artifact.files))
        if missing:
            raise ValueError(f"reference {key} is missing arrays: {missing}")
        reference = {name: np.array(artifact[name], copy=True) for name in names}
    geometry = _validate_reference_geometry(reference)
    shard_states = np.asarray(store.states(key), dtype=np.float64)
    shard_nodes = np.asarray(store.array(key, "nodes"), dtype=np.float64)
    shard_measures = np.asarray(
        store.array(key, "node_measures"), dtype=np.float64
    ).reshape(-1)
    if not np.allclose(
        shard_states, reference["conservative_states"], rtol=1.0e-6, atol=1.0e-7
    ):
        raise ValueError(f"shard/reference state mismatch for {key}")
    if not np.allclose(shard_nodes, reference["cell_centers"], rtol=0.0, atol=1.0e-7):
        raise ValueError(f"shard/reference cell-order mismatch for {key}")
    if not np.allclose(
        shard_measures, reference["cell_volume"], rtol=1.0e-6, atol=1.0e-10
    ):
        raise ValueError(f"shard/reference cell-volume mismatch for {key}")
    boundary = reference["face_neighbor"] < 0
    impulse_exchange = np.sum(
        reference["cumulative_accepted_substep_face_impulses"][:, boundary], axis=1
    )
    if not np.allclose(
        impulse_exchange,
        reference["interval_boundary_exchange"],
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise ValueError(f"reference boundary impulse accounting failed for {key}")
    state_change = np.einsum(
        "n,tnc->tc",
        reference["cell_volume"],
        np.diff(reference["conservative_states"], axis=0),
    )
    closure = state_change + reference["interval_boundary_exchange"]
    closure_scale = np.maximum(np.linalg.norm(state_change, axis=1), 1.0e-30)
    closure_relative = np.linalg.norm(closure, axis=1) / closure_scale
    return reference, {
        "reference_artifact_sha256": digest,
        "maximum_interval_balance_relative_l2": float(np.max(closure_relative)),
        **geometry,
    }


def _mean(rows: Sequence[Mapping[str, Any]], name: str) -> float | None:
    values = [float(row[name]) for row in rows if row.get(name) is not None]
    return float(np.mean(values)) if values else None


def aggregate_variant(
    trajectory_rows: Sequence[Mapping[str, Any]],
    call_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not trajectory_rows:
        raise ValueError("cannot aggregate an empty variant")
    completed = [row for row in trajectory_rows if row["completed"]]
    common_call = min(int(row["valid_length"]) for row in trajectory_rows)
    one_step = [row for row in call_rows if int(row["call"]) == 1]
    common = (
        [row for row in call_rows if int(row["call"]) == common_call]
        if common_call
        else []
    )
    final_by_trajectory = {str(row["trajectory"]): row for row in call_rows}
    completed_final = [
        final_by_trajectory[str(row["trajectory"])]
        for row in completed
        if str(row["trajectory"]) in final_by_trajectory
    ]
    failure_counts: dict[str, int] = {}
    for row in trajectory_rows:
        cause = str(row["failure_cause"])
        failure_counts[cause] = failure_counts.get(cause, 0) + 1
    metric = "scaled_relative_l2_physical_volume"
    budget = "prediction_reference_balance_component_scaled_rmse"
    return {
        "trajectories": len(trajectory_rows),
        "completed": len(completed),
        "completion_rate": len(completed) / len(trajectory_rows),
        "mean_survival_fraction": _mean(trajectory_rows, "survival_fraction"),
        "failure_cause_counts": failure_counts,
        "one_step_entry_gate_mean_relative_l2": _mean(one_step, metric),
        "mixed_prefix_mean_final_relative_l2": _mean(
            list(final_by_trajectory.values()), metric
        ),
        "completed_case_mean_final_relative_l2": _mean(completed_final, metric),
        "common_endpoint_call": common_call,
        "common_endpoint_mean_relative_l2": _mean(common, metric),
        "common_endpoint_mean_physical_budget_error": _mean(common, budget),
        "completed_case_mean_physical_budget_error": _mean(completed_final, budget),
        "mean_forward_seconds_per_trajectory": _mean(
            trajectory_rows, "total_forward_seconds"
        ),
    }


def endpoint_aggregates(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    variants = sorted({str(row["variant"]) for row in rows})
    calls = sorted({int(row["call"]) for row in rows})
    names = (
        "front_iou",
        "front_centroid_distance",
        "front_symmetric_chamfer",
        "shock_thickness_log_error",
        "shock_strength_log_error",
        "vortex_core_density_relative_error",
        "smooth_region_scaled_relative_l2",
        "smooth_region_graph_highpass_energy",
    )
    for variant in variants:
        for call in calls:
            selected = [
                row
                for row in rows
                if row["variant"] == variant and int(row["call"]) == call
            ]
            if selected:
                result.append(
                    {
                        "variant": variant,
                        "call": call,
                        "trajectories": len(selected),
                        **{f"mean_{name}": _mean(selected, name) for name in names},
                    }
                )
    return result


def benchmark_cost(
    model: PCNOEuler2DResidual,
    store: PCNOEuler2DShardStore,
    keys: Sequence[str],
    *,
    step_stride: int,
    device: torch.device,
    amp: str,
    latency_warmup: int,
    latency_repeats: int,
    throughput_batch_size: int,
    throughput_repeats: int,
) -> dict[str, Any]:
    first_key = keys[0]
    sample = store.tensor_sample(first_key, 0, step_stride=step_stride, device=device)
    current = sample["current"]
    for _ in range(latency_warmup):
        model_call(model, sample, current, device=device, amp=amp)
    synchronize(device)
    latency = []
    for _ in range(latency_repeats):
        synchronize(device)
        started = perf_counter()
        model_call(model, sample, current, device=device, amp=amp)
        synchronize(device)
        latency.append(perf_counter() - started)

    batch_keys = list(keys[: min(len(keys), throughput_batch_size)])
    geometry_digests = {store.entry(key).get("geometry_digest") for key in batch_keys}
    if len(geometry_digests) != 1:
        raise ValueError("amortized throughput batch requires identical geometry")
    batch_size = len(batch_keys)
    batch_sample = {
        name: value[:1].expand(batch_size, *value.shape[1:])
        for name, value in sample.items()
        if name not in {"current", "target", "mach"}
    }
    batch_current = torch.as_tensor(
        np.stack([owned_state_frame(store, key, 0) for key in batch_keys]),
        dtype=torch.float32,
        device=device,
    )
    batch_sample["mach"] = torch.as_tensor(
        [float(store.entry(key)["mach"]) for key in batch_keys],
        dtype=torch.float32,
        device=device,
    )
    for _ in range(latency_warmup):
        model_call(model, batch_sample, batch_current, device=device, amp=amp)
    synchronize(device)
    throughput = []
    for _ in range(throughput_repeats):
        synchronize(device)
        started = perf_counter()
        model_call(model, batch_sample, batch_current, device=device, amp=amp)
        synchronize(device)
        throughput.append(perf_counter() - started)
    latency_array = np.asarray(latency)
    throughput_array = np.asarray(throughput)
    return {
        "contract": (
            "synchronized device forward including PCNO basis construction; excludes "
            "host transfer, metrics, recurrence bookkeeping, and artifact compression"
        ),
        "batch1_repeats": latency_repeats,
        "batch1_seconds_median": float(np.median(latency_array)),
        "batch1_seconds_p95": float(np.quantile(latency_array, 0.95)),
        "throughput_batch_size": batch_size,
        "throughput_repeats": throughput_repeats,
        "throughput_samples_per_second": float(batch_size / np.mean(throughput_array)),
    }


def _resolved_keys(args: argparse.Namespace, store: PCNOEuler2DShardStore) -> list[str]:
    declared = [
        str(key) for key in store.manifest.get("splits", {}).get(args.split, [])
    ]
    if not declared:
        declared = [
            key for key in store.keys if store.entry(key).get("split") == args.split
        ]
    if args.trajectory_keys is None:
        keys = declared
    else:
        keys = [str(key) for key in args.trajectory_keys]
        unexpected = sorted(set(keys) - set(declared))
        if unexpected:
            raise ValueError(
                f"requested keys are outside the {args.split} split: {unexpected}"
            )
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("resolved trajectory keys must be nonempty and unique")
    if args.expected_trajectory_count and len(keys) != args.expected_trajectory_count:
        raise ValueError("resolved trajectory count disagrees with expectation")
    return keys


def parameter_time_control_weights(
    query_parameters: Mapping[str, Any],
    candidate_parameters: Mapping[str, Mapping[str, Any]],
    *,
    neighbor_count: int = CONTROL_NEIGHBOR_COUNT,
) -> dict[str, Any]:
    """Return deterministic train-only neighbors in normalized parameter space."""

    if neighbor_count < 1:
        raise ValueError("neighbor_count must be positive")
    keys = sorted(map(str, candidate_parameters))
    if not keys:
        raise ValueError("parameter-time controls require training candidates")
    if neighbor_count > len(keys):
        raise ValueError("neighbor_count exceeds the training candidate count")
    try:
        query = np.asarray(
            [float(query_parameters[name]) for name in CONTROL_PARAMETER_NAMES],
            dtype=np.float64,
        )
        candidates = np.asarray(
            [
                [
                    float(candidate_parameters[key][name])
                    for name in CONTROL_PARAMETER_NAMES
                ]
                for key in keys
            ],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "control parameters must provide finite vortex_epsilon and vortex_y"
        ) from exc
    if not np.all(np.isfinite(query)) or not np.all(np.isfinite(candidates)):
        raise ValueError("control parameters must be finite")
    minimum = np.min(candidates, axis=0)
    maximum = np.max(candidates, axis=0)
    span = maximum - minimum
    if np.any(span <= 0.0):
        raise ValueError("training control parameters must vary in every dimension")
    distances = np.linalg.norm((candidates - query[None, :]) / span[None, :], axis=1)
    order = sorted(range(len(keys)), key=lambda index: (distances[index], keys[index]))
    selected = order[:neighbor_count]
    selected_distances = distances[selected]
    if selected_distances[0] <= 1.0e-14:
        weights = np.zeros(neighbor_count, dtype=np.float64)
        weights[0] = 1.0
    else:
        weights = 1.0 / selected_distances
        weights /= np.sum(weights)
    return {
        "parameter_names": list(CONTROL_PARAMETER_NAMES),
        "normalization_minimum": minimum.tolist(),
        "normalization_maximum": maximum.tolist(),
        "selected_keys": [keys[index] for index in selected],
        "normalized_distances": selected_distances.tolist(),
        "inverse_distance_weights": weights.tolist(),
        "nearest_key": keys[selected[0]],
    }


def owned_store_array(
    store: PCNOEuler2DShardStore,
    key: str,
    name: str,
    *,
    dtype: np.dtype[Any] | type[Any],
) -> np.ndarray:
    """Copy a shard array before later cache eviction can close its mmap."""

    return np.array(store.array(key, name), dtype=dtype, copy=True)


def owned_state_frame(store: PCNOEuler2DShardStore, key: str, frame: int) -> np.ndarray:
    """Copy one state frame before loading another key can evict its mmap."""

    return np.array(store.states(key)[frame], dtype=np.float32, copy=True)


def validate_args(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    if args.expected_trajectory_count < 0:
        raise ValueError("--expected-trajectory-count must be nonnegative")
    if args.start_frame < 0 or args.num_steps < 1:
        raise ValueError("invalid rollout horizon")
    if not 0.0 < args.shock_quantile < 1.0:
        raise ValueError("--shock-quantile must lie in (0,1)")
    for name in (
        "latency_warmup",
        "latency_repeats",
        "throughput_batch_size",
        "throughput_repeats",
    ):
        if int(getattr(args, name)) < (0 if name == "latency_warmup" else 1):
            raise ValueError(f"--{name.replace('_', '-')} is invalid")
    calls = sorted(set(int(value) for value in args.endpoint_calls))
    args.endpoint_calls = [value for value in calls if 1 <= value <= args.num_steps]
    if not args.endpoint_calls:
        raise ValueError("no endpoint call lies inside the rollout horizon")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    device = select_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint)
    amp = (
        str(checkpoint.get("training_args", {}).get("amp", "none"))
        if args.amp == "checkpoint"
        else args.amp
    )
    if amp != "none" and device.type != "cuda":
        raise ValueError("resolved mixed precision requires CUDA")
    store = PCNOEuler2DShardStore(args.data_dir, max_cached_trajectories=2)
    if store.manifest.get("dataset") != "shock_vortex_fv_family":
        raise ValueError(
            "this evaluator only accepts the validated shock-vortex FV family"
        )
    if checkpoint["data_manifest_digest"] != store.manifest_digest:
        raise ValueError("checkpoint and evaluation shard manifests differ")
    if checkpoint["data_contract"]["data_manifest_digest"] != store.manifest_digest:
        raise ValueError("checkpoint data contract is not bound to these shards")
    family_manifest = load_shock_vortex_family_manifest(
        args.family_root / "family_manifest.json"
    )
    if (
        checkpoint["data_contract"]["source_family_manifest_digest"]
        != family_manifest["manifest_digest_sha256"]
    ):
        raise ValueError("checkpoint and reference family manifests differ")
    keys = _resolved_keys(args, store)
    step_stride = int(checkpoint["step_stride"])
    final_index = args.start_frame + args.num_steps * step_stride
    if any(final_index >= store.states(key).shape[0] for key in keys):
        raise ValueError("requested rollout exceeds one or more trajectories")
    model = build_model(checkpoint, device)
    normalization = Euler2DNormalization.from_mapping(checkpoint["normalization"])
    training_keys = [
        str(key) for key in store.manifest.get("splits", {}).get("train", [])
    ]
    if not training_keys:
        training_keys = [
            key for key in store.keys if store.entry(key).get("split") == "train"
        ]
    if len(training_keys) < CONTROL_NEIGHBOR_COUNT:
        raise ValueError("insufficient train-split trajectories for dataset controls")
    training_parameters = {
        key: store.entry(key).get("parameters", {}) for key in training_keys
    }

    args.output_dir.mkdir(parents=True)
    trajectory_dir = args.output_dir / "trajectories"
    trajectory_dir.mkdir()
    trajectory_rows: list[dict[str, Any]] = []
    call_rows: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []
    artifact_contracts: list[dict[str, Any]] = []
    reference_checks: list[dict[str, Any]] = []
    control_contracts: list[dict[str, Any]] = []

    for key in keys:
        states = np.asarray(store.states(key), dtype=np.float64)
        target_indices = args.start_frame + step_stride * np.arange(
            1, args.num_steps + 1
        )
        targets = states[target_indices]
        initial = states[args.start_frame]
        positions = owned_store_array(store, key, "nodes", dtype=np.float64)
        edges = owned_store_array(store, key, "edges", dtype=np.int64)
        node_type = owned_store_array(store, key, "node_type", dtype=np.int64)
        mapping = owned_store_array(
            store, key, "mesh_cell_to_graph_node", dtype=np.int64
        )
        if not np.array_equal(mapping, np.arange(mapping.size)):
            raise ValueError(f"mesh-to-graph identity map failed for {key}")
        reference, reference_check = load_reference(args.family_root, store, key)
        reference_config = family_manifest["reference_config"]
        upstream_speed = float(reference_config["shock_mach"]) * math.sqrt(
            float(reference_config["gamma"])
        )
        shock_arrival_time = (
            float(reference_config["shock_x"]) - float(reference_config["vortex_x"])
        ) / upstream_speed
        vortex_y = float(store.entry(key)["parameters"]["vortex_y"])
        macro_delta_t = step_stride * float(store.manifest["dt"])
        physical_horizon = args.num_steps * macro_delta_t
        reference_check["characteristic_travel"] = characteristic_travel_summary(
            reference["conservative_states"][args.start_frame : final_index + 1],
            reference["cell_volume"],
            gamma=model.gamma,
            macro_delta_t=macro_delta_t,
            physical_horizon=physical_horizon,
        )
        reference_checks.append({"trajectory": key, **reference_check})
        volumes = np.asarray(reference["cell_volume"], dtype=np.float64)
        interval_exchange = np.asarray(
            reference["interval_boundary_exchange"], dtype=np.float64
        )
        rollout = raw_rollout(
            model,
            store,
            key,
            start_frame=args.start_frame,
            num_steps=args.num_steps,
            step_stride=step_stride,
            device=device,
            amp=amp,
        )
        control_started = perf_counter()
        control = parameter_time_control_weights(
            store.entry(key).get("parameters", {}), training_parameters
        )
        control_keys = [str(value) for value in control["selected_keys"]]
        query_geometry_digest = str(store.entry(key).get("geometry_digest"))
        control_geometry_digests = {
            str(store.entry(control_key).get("geometry_digest"))
            for control_key in control_keys
        }
        if control_geometry_digests != {query_geometry_digest}:
            raise ValueError(
                "dataset controls require the audited identity-mesh remapping contract"
            )
        control_states = np.stack(
            [
                np.asarray(store.states(control_key)[target_indices], dtype=np.float64)
                for control_key in control_keys
            ],
            axis=0,
        )
        idw_predictions = np.tensordot(
            np.asarray(control["inverse_distance_weights"], dtype=np.float64),
            control_states,
            axes=(0, 0),
        )
        nearest_predictions = control_states[0]
        control_contracts.append(
            {
                "trajectory": key,
                "query_parameters": store.entry(key).get("parameters"),
                **control,
                "geometry_digest": query_geometry_digest,
                "mesh_remapping_operator": "audited identity cell/node ordering",
                "oracle_remapping_error_floor": 0.0,
                "construction_seconds": perf_counter() - control_started,
            }
        )
        variants = {
            "persistence": np.repeat(initial[None], args.num_steps, axis=0),
            "geometry_parameter_time_idw_k4": idw_predictions,
            "nearest_training_trajectory": nearest_predictions,
            "pcno_baseline": rollout["predictions"],
        }
        for variant, predictions in variants.items():
            valid_length = int(predictions.shape[0])
            is_pcno = variant == "pcno_baseline"
            variant_admissibility = (
                None
                if is_pcno
                else raw_admissibility_summary(
                    np.asarray(predictions, dtype=np.float64).reshape(-1, 4),
                    gamma=model.gamma,
                )
            )
            trajectory_rows.append(
                {
                    "trajectory": key,
                    "variant": variant,
                    "split": args.split,
                    "split_group_id": store.entry(key).get("split_group_id"),
                    "parameters": store.entry(key).get("parameters"),
                    "valid_length": valid_length,
                    "completed": valid_length == args.num_steps,
                    "survival_fraction": valid_length / args.num_steps,
                    "failure_cause": (
                        rollout["failure_cause"] if is_pcno else "completed"
                    ),
                    "failure_call": rollout["failure_call"] if is_pcno else None,
                    "min_density": (
                        rollout["min_density"]
                        if is_pcno
                        else variant_admissibility["min_density"]
                    ),
                    "min_internal_energy": (
                        rollout["min_internal_energy"]
                        if is_pcno
                        else variant_admissibility["min_internal_energy"]
                    ),
                    "min_pressure": (
                        rollout["min_pressure"]
                        if is_pcno
                        else variant_admissibility["min_pressure"]
                    ),
                    "total_forward_seconds": (
                        float(sum(rollout["call_seconds"])) if is_pcno else None
                    ),
                }
            )
            for call_index in range(1, valid_length + 1):
                prediction = predictions[call_index - 1]
                target = targets[call_index - 1]
                saved_stop = args.start_frame + call_index * step_stride
                cumulative_exchange = np.sum(
                    interval_exchange[args.start_frame : saved_stop], axis=0
                )
                physical = physical_call_metrics(
                    prediction,
                    target,
                    initial,
                    volumes=volumes,
                    reference_cumulative_boundary_exchange=cumulative_exchange,
                    component_scale=normalization.state_scale,
                )
                row = {
                    "trajectory": key,
                    "variant": variant,
                    "call": call_index,
                    "physical_time": float(
                        reference["physical_times"][saved_stop]
                        - reference["physical_times"][args.start_frame]
                    ),
                    "scaled_relative_l2_physical_volume": weighted_relative_l2_numpy(
                        prediction,
                        target,
                        volumes,
                        normalization.state_scale,
                    ),
                    **physical,
                }
                call_rows.append(row)
                if call_index in args.endpoint_calls:
                    absolute_time = float(reference["physical_times"][saved_stop])
                    expected_vortex_x = (
                        float(reference_config["vortex_x"])
                        + upstream_speed * absolute_time
                        if absolute_time <= shock_arrival_time
                        else float(reference_config["shock_x"])
                        + float(reference_config["right_u"])
                        * (absolute_time - shock_arrival_time)
                    )
                    endpoint_rows.append(
                        {
                            "trajectory": key,
                            "variant": variant,
                            "call": call_index,
                            **endpoint_metrics(
                                prediction,
                                target,
                                positions=positions,
                                edges=edges,
                                volumes=volumes,
                                component_scale=normalization.state_scale,
                                gamma=model.gamma,
                                shock_quantile=args.shock_quantile,
                                vortex_center=(expected_vortex_x, vortex_y),
                            ),
                        }
                    )

        artifact_name = f"trajectory_{key}.npz"
        artifact_payload: dict[str, Any] = {
            "schema": np.asarray(SCHEMA),
            "trajectory": np.asarray(key),
            "split": np.asarray(args.split),
            "split_group_id": np.asarray(str(store.entry(key).get("split_group_id"))),
            "parameters_json": np.asarray(
                json.dumps(store.entry(key).get("parameters"), sort_keys=True)
            ),
            "reference_config_json": np.asarray(
                json.dumps(reference_config, sort_keys=True)
            ),
            "gamma": np.asarray(model.gamma, dtype=np.float64),
            "state_component_scale": np.asarray(
                normalization.state_scale, dtype=np.float64
            ),
            "shock_quantile": np.asarray(args.shock_quantile, dtype=np.float64),
            "physical_times": reference["physical_times"][: final_index + 1],
            "physical_delta_t": np.diff(reference["physical_times"][: final_index + 1]),
            "initial_state": initial.astype(np.float32),
            "targets": targets.astype(np.float32),
            "predictions": rollout["predictions"],
            "geometry_parameter_time_idw_k4_predictions": idw_predictions.astype(
                np.float32
            ),
            "nearest_training_trajectory_predictions": nearest_predictions.astype(
                np.float32
            ),
            "parameter_time_control_keys_json": np.asarray(json.dumps(control_keys)),
            "parameter_time_control_distances": np.asarray(
                control["normalized_distances"], dtype=np.float64
            ),
            "parameter_time_control_weights": np.asarray(
                control["inverse_distance_weights"], dtype=np.float64
            ),
            "nearest_training_trajectory_key": np.asarray(control["nearest_key"]),
            "positions": positions.astype(np.float32),
            "edges": edges,
            "node_type": node_type,
            "physical_cell_volumes": volumes,
            "normalized_physical_cell_volume_weights": volumes / volumes.sum(),
            "mesh_cell_to_graph_node": mapping,
            "face_to_directed_edge": np.asarray(
                owned_store_array(store, key, "face_to_directed_edge", dtype=np.int64)
            ),
            "face_centers": reference["face_centers"],
            "face_measures": reference["face_measure"],
            "face_normals": reference["face_normal"],
            "face_owner": reference["face_owner"],
            "face_neighbor": reference["face_neighbor"],
            "face_axis": reference["face_axis"],
            "face_boundary_tag": reference["face_boundary_tag"],
            "boundary_tag_names_json": reference["boundary_tag_names_json"],
            "coordinate_convention": np.asarray(
                store.manifest["coordinate_convention"]
            ),
            "face_orientation_convention": np.asarray(
                "normal points outward from owner; positive impulse leaves owner"
            ),
            "reference_cumulative_accepted_substep_face_impulses": reference[
                "cumulative_accepted_substep_face_impulses"
            ][:final_index],
            "reference_interval_boundary_exchange": interval_exchange[:final_index],
            "valid_length": np.asarray(rollout["valid_length"], dtype=np.int64),
            "failure_cause": np.asarray(rollout["failure_cause"]),
            "checkpoint_sha256": np.asarray(sha256_file(args.checkpoint)),
            "checkpoint_config_digest": np.asarray(checkpoint["config_digest"]),
            "data_manifest_digest": np.asarray(store.manifest_digest),
            "geometry_digest": np.asarray(str(store.entry(key).get("geometry_digest"))),
            "normalization_digest": np.asarray(checkpoint["normalization_digest"]),
            "boundary_mode": np.asarray("model_all_nodes"),
            "raw_recurrence": np.asarray(True),
            "inference_interventions_json": np.asarray(
                json.dumps(checkpoint["inference_interventions"], sort_keys=True)
            ),
        }
        if rollout["failed_proposal"] is not None:
            artifact_payload["failed_proposal"] = rollout["failed_proposal"].astype(
                np.float32
            )
        np.savez_compressed(trajectory_dir / artifact_name, **artifact_payload)
        artifact_contracts.append(
            {
                "trajectory": key,
                "artifact": f"trajectories/{artifact_name}",
                "sha256": sha256_file(trajectory_dir / artifact_name),
                "reference_artifact_sha256": reference_check[
                    "reference_artifact_sha256"
                ],
            }
        )
        print(
            json.dumps(
                {
                    "trajectory": key,
                    "valid_length": rollout["valid_length"],
                    "failure_cause": rollout["failure_cause"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    aggregates = {}
    grouped = []
    for variant in (
        "persistence",
        "geometry_parameter_time_idw_k4",
        "nearest_training_trajectory",
        "pcno_baseline",
    ):
        selected_trajectories = [
            row for row in trajectory_rows if row["variant"] == variant
        ]
        selected_calls = [row for row in call_rows if row["variant"] == variant]
        aggregates[variant] = aggregate_variant(selected_trajectories, selected_calls)
        for group_id in sorted(
            {str(row["split_group_id"]) for row in selected_trajectories}
        ):
            group_trajectories = [
                row
                for row in selected_trajectories
                if str(row["split_group_id"]) == group_id
            ]
            group_keys = {str(row["trajectory"]) for row in group_trajectories}
            group_calls = [
                row for row in selected_calls if str(row["trajectory"]) in group_keys
            ]
            grouped.append(
                {
                    "variant": variant,
                    "split_group_id": group_id,
                    **aggregate_variant(group_trajectories, group_calls),
                }
            )
    cost = benchmark_cost(
        model,
        store,
        keys,
        step_stride=step_stride,
        device=device,
        amp=amp,
        latency_warmup=args.latency_warmup,
        latency_repeats=args.latency_repeats,
        throughput_batch_size=args.throughput_batch_size,
        throughput_repeats=args.throughput_repeats,
    )
    write_csv(args.output_dir / "trajectory_metrics.csv", trajectory_rows)
    write_csv(args.output_dir / "call_metrics.csv", call_rows)
    write_csv(args.output_dir / "endpoint_metrics.csv", endpoint_rows)
    endpoint_summary = endpoint_aggregates(endpoint_rows)
    write_csv(args.output_dir / "endpoint_aggregates.csv", endpoint_summary)
    write_csv(args.output_dir / "grouped_metrics.csv", grouped)
    summary = {
        "schema": SCHEMA,
        "status": "complete",
        "checkpoint": {
            "path_name": args.checkpoint.name,
            "sha256": sha256_file(args.checkpoint),
            "epoch": checkpoint.get("epoch"),
            "best_epoch": checkpoint.get("best_epoch"),
            "config_digest": checkpoint["config_digest"],
            "normalization_digest": checkpoint["normalization_digest"],
            "model_config": checkpoint["model_config"],
            "parameter_count": parameter_count(model),
            "selection_rule_state": checkpoint.get("best_selection"),
        },
        "data_contract": checkpoint["data_contract"],
        "reference_contract": {
            "family_id": family_manifest["family_id"],
            "family_manifest_digest": family_manifest["manifest_digest_sha256"],
            "physical_geometry": "validated volumes, faces, measures, unit normals, and owner orientation",
            "boundary_accounting": "recorded cumulative accepted-substep owner-oriented face impulses",
            "maximum_interval_balance_relative_l2": max(
                row["maximum_interval_balance_relative_l2"] for row in reference_checks
            ),
            "characteristic_travel": [
                {"trajectory": row["trajectory"], **row["characteristic_travel"]}
                for row in reference_checks
            ],
            "reference_checks": reference_checks,
        },
        "evaluation": {
            "split": args.split,
            "trajectory_keys": keys,
            "start_frame": args.start_frame,
            "num_steps": args.num_steps,
            "step_stride": step_stride,
            "saved_delta_t": float(store.manifest["dt"]),
            "physical_horizon": args.num_steps
            * step_stride
            * float(store.manifest["dt"]),
            "endpoint_calls": args.endpoint_calls,
            "device": str(device),
            "amp": amp,
            "boundary_mode": "model_all_nodes",
            "raw_recurrence": True,
            "inference_interventions": checkpoint["inference_interventions"],
            "coordinate_convention": store.manifest["coordinate_convention"],
            "physical_budget_interpretation": (
                "model state-total change compared with recorded reference boundary exchange; "
                "not a model-predicted face flux"
            ),
        },
        "aggregates": aggregates,
        "endpoint_aggregates": endpoint_summary,
        "grouped_parameter_ood": grouped,
        "cost": {
            **cost,
            "paired_coarse_cfd_error_cost": "not_evaluated_in_this_baseline_rollout",
        },
        "dataset_manifold_controls": {
            "contracts": control_contracts,
            "geometry_parameter_time_control": (
                "inverse-distance average of the four nearest train trajectories "
                "in train-range-normalized (vortex_epsilon, vortex_y) coordinates at the "
                "known physical time; no current state or validation target is used"
            ),
            "nearest_trajectory_control": (
                "nearest complete train trajectory in the same normalized parameter space"
            ),
            "cross_mesh_remapping_operator": "audited identity cell/node ordering",
            "oracle_remapping_error_floor": 0.0,
            "cost_contract": (
                "dataset lookups are diagnostic controls and are not assigned zero latency"
            ),
        },
        "trajectory_artifacts": artifact_contracts,
        "artifact_schema": {
            "trajectory_npz": (
                "initial state, held-out targets, raw PCNO and train-only control predictions, "
                "physical time, positions, "
                "graph, node types, physical volumes, mesh-to-graph map, oriented faces, "
                "reference face impulses, metric configuration and component scale, failure "
                "contract, checkpoint/config/data/geometry/normalization digests, boundary "
                "mode, and every intervention"
            ),
            "call_metrics_csv": "state error and physical total/boundary-exchange accounting by call",
            "endpoint_metrics_csv": "graph-native front, shock, and smooth-region diagnostics",
        },
        "line4_handoff": {
            "status": "baseline_evaluated_before_frozen_D013_method_routing",
            "line4_training_truth_authorized": None,
            "line4_front_candidate_available": None,
            "remaining_before_final_handoff": [
                "freeze the serious checkpoint and this rollout report",
                "complete D013 error classification on validation trajectories",
                "record the front-candidate boolean and causal reason",
            ],
        },
        "claim_boundary": {
            "verified": (
                "raw frozen-checkpoint state rollout, physical cell-total error, and mismatch "
                "to validated reference boundary exchange, plus train-only dataset-manifold "
                "controls on the selected family split"
            ),
            "not_verified": [
                "model-predicted physical face flux",
                "reference face-field uniqueness across solvers",
                "paired matched-cost coarse-CFD superiority",
                "stabilization-method benefit",
                "final Line-4 authorization before D013 routing is frozen",
            ],
        },
    }
    atomic_write_json(args.output_dir / "summary.json", summary)
    print(json.dumps({"summary": str(args.output_dir / "summary.json")}, indent=2))
    store.close()


if __name__ == "__main__":
    main()
