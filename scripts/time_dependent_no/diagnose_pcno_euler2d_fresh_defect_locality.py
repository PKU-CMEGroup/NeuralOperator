#!/usr/bin/env python3
"""Test whether D053's fresh ripple defect is causally localizable.

D054 is a zero-training necessary-condition test for one new representation
hypothesis: retain the frozen global residual PCNO, but factor its target into a
global component and a current-state shock-conditioned local detail.  The test
does not modify inference.  It asks whether the smooth-region high-pass part of
the teacher-forced defect is both sparse under a truth-informed 20% node oracle
and captured by a fixed, current-state-only four-hop pressure-jump halo.
"""

from __future__ import annotations

import argparse
from collections import deque
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
from scripts.time_dependent_no.diagnose_pcno_euler2d_ripples import (  # noqa: E402
    append_jsonl,
    build_model,
    load_checkpoint,
    model_call,
    select_device,
    sha256_file,
    write_json,
)
from utility.time_dependent_no.euler2d_metrics import shock_front_scores  # noqa: E402
from utility.time_dependent_no.pcno_euler2d import (  # noqa: E402
    PCNOEuler2DShardStore,
    reference_smooth_region_mask,
)
from utility.time_dependent_no.pcno_ripple_diagnostics import (  # noqa: E402
    RIPPLE_DIAGNOSTIC_SCHEMA,
    conservative_to_primitive_raw,
    node_highpass_field,
    normalized_node_weights,
    raw_admissibility_summary,
    spatial_correlation_summary,
)


FRESH_LOCALITY_SCHEMA = "pcno_euler2d_fresh_defect_locality_d054_v1"
EXPECTED_TRAJECTORY_COUNT = 6
EXPECTED_CALLS = tuple(range(1, 61))
LATE_CALLS = (30, 60)
REQUIRED_CASES = 5

# Frozen before inspecting D054 outputs.  The 20% support matches D044's local
# correction cap.  Seventy-percent oracle capture is a strong necessary
# compressibility condition; the causal halo must capture half the energy
# while touching no more than one quarter of the interior.
ORACLE_NODE_FRACTION = 0.20
ORACLE_ENERGY_CAPTURE_MIN = 0.70
CAUSAL_HALO_HOPS = 4
CAUSAL_HALO_SUPPORT_MAX = 0.25
CAUSAL_HALO_ENERGY_CAPTURE_MIN = 0.50
SHOCK_QUANTILE = 0.90
D053_SHOCK_DILATION_HOPS = 2
D053_NORM_REPLAY_RELATIVE_TOLERANCE = 2.0e-4


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--d053-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")


def build_graph_adjacency(
    num_nodes: int, edges: np.ndarray
) -> tuple[tuple[int, ...], ...]:
    """Build a deterministic undirected adjacency for repeated hop queries."""

    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")
    edge_index = np.asarray(edges, dtype=np.int64)
    if edge_index.ndim != 2 or edge_index.shape[1] != 2:
        raise ValueError("edges must have shape [E,2]")
    if edge_index.size and (
        int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes
    ):
        raise ValueError("edge index lies outside the node axis")
    neighbors: list[set[int]] = [set() for _ in range(num_nodes)]
    for left, right in edge_index:
        left_index = int(left)
        right_index = int(right)
        if left_index == right_index:
            continue
        neighbors[left_index].add(right_index)
        neighbors[right_index].add(left_index)
    return tuple(tuple(sorted(values)) for values in neighbors)


def graph_hop_distance(
    adjacency: Sequence[Sequence[int]], source_mask: np.ndarray
) -> np.ndarray:
    """Return unweighted graph distance from a nonempty source set."""

    selected = np.asarray(source_mask, dtype=bool)
    if selected.shape != (len(adjacency),):
        raise ValueError("source_mask must contain one value per graph node")
    if not np.any(selected):
        raise ValueError("source_mask must select at least one graph node")
    distance = np.full(len(adjacency), -1, dtype=np.int64)
    queue: deque[int] = deque()
    for source in np.flatnonzero(selected):
        index = int(source)
        distance[index] = 0
        queue.append(index)
    while queue:
        node = queue.popleft()
        candidate = int(distance[node]) + 1
        for neighbor in adjacency[node]:
            if distance[neighbor] < 0:
                distance[neighbor] = candidate
                queue.append(int(neighbor))
    return distance


def _top_fraction_mask(
    energy: np.ndarray, mask: np.ndarray, fraction: float
) -> np.ndarray:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0,1]")
    values = np.asarray(energy, dtype=np.float64)
    selected = np.asarray(mask, dtype=bool)
    if values.shape != selected.shape or values.ndim != 1:
        raise ValueError("energy and mask must share shape [N]")
    available = np.flatnonzero(selected)
    if available.size == 0:
        raise ValueError("mask must select at least one node")
    count = max(1, int(math.floor(fraction * available.size + 1.0e-12)))
    order = np.lexsort((available, -values[available]))
    result = np.zeros_like(selected)
    result[available[order[:count]]] = True
    return result


def _edge_neighbor_cosine(
    field: np.ndarray,
    edges: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
) -> float | None:
    values = np.asarray(field, dtype=np.float64)
    edge_index = np.asarray(edges, dtype=np.int64)
    mass = normalized_node_weights(weights, name="D054 neighbor cosine")
    selected = np.asarray(mask, dtype=bool)
    valid = selected[edge_index[:, 0]] & selected[edge_index[:, 1]]
    if not np.any(valid):
        return None
    left = edge_index[valid, 0]
    right = edge_index[valid, 1]
    edge_mass = 0.5 * (mass[left] + mass[right])
    numerator = float(
        np.sum(edge_mass * np.sum(values[left] * values[right], axis=-1))
    )
    left_energy = float(
        np.sum(edge_mass * np.sum(values[left] ** 2, axis=-1))
    )
    right_energy = float(
        np.sum(edge_mass * np.sum(values[right] ** 2, axis=-1))
    )
    denominator = math.sqrt(max(left_energy * right_energy, 0.0))
    return None if denominator <= 1.0e-30 else numerator / denominator


def fresh_defect_locality_metrics(
    highpass_field: np.ndarray,
    *,
    edges: np.ndarray,
    adjacency: Sequence[Sequence[int]],
    weights: np.ndarray,
    interior_mask: np.ndarray,
    smooth_mask: np.ndarray,
    current_shock_seed: np.ndarray,
    current_pressure_jump_score: np.ndarray | None = None,
) -> dict[str, Any]:
    """Measure sparse-oracle and causal-halo capture of one fresh defect."""

    values = np.asarray(highpass_field, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("highpass_field must have shape [N,C]")
    interior = np.asarray(interior_mask, dtype=bool)
    smooth = np.asarray(smooth_mask, dtype=bool)
    seed = np.asarray(current_shock_seed, dtype=bool)
    expected_shape = (values.shape[0],)
    if any(mask.shape != expected_shape for mask in (interior, smooth, seed)):
        raise ValueError("all node masks must match highpass_field")
    if not np.any(interior) or not np.any(smooth) or not np.any(seed):
        raise ValueError("interior, smooth, and current shock masks must be nonempty")
    if np.any(smooth & ~interior) or np.any(seed & ~interior):
        raise ValueError("smooth and shock masks must be subsets of the interior")

    mass = normalized_node_weights(weights, name="D054 locality")
    node_energy = mass * np.sum(values**2, axis=-1)
    smooth_energy = float(np.sum(node_energy[smooth]))
    if smooth_energy <= 1.0e-30:
        return {
            "status": "zero_smooth_highpass_energy",
            "smooth_node_count": int(np.count_nonzero(smooth)),
        }

    oracle_mask = _top_fraction_mask(node_energy, smooth, ORACLE_NODE_FRACTION)
    hop_distance = graph_hop_distance(adjacency, seed)
    causal_halo = interior & (hop_distance >= 0) & (hop_distance <= CAUSAL_HALO_HOPS)
    causal_smooth = causal_halo & smooth

    def capture(mask: np.ndarray) -> float:
        return float(np.sum(node_energy[mask]) / smooth_energy)

    rings: dict[str, dict[str, float | int]] = {}
    for name, lower, upper in (
        ("hop_0_to_2", 0, 2),
        ("hop_3_to_4", 3, 4),
        ("hop_5_to_8", 5, 8),
        ("beyond_hop_8", 9, None),
    ):
        ring = smooth & (hop_distance >= lower)
        if upper is not None:
            ring &= hop_distance <= upper
        rings[name] = {
            "node_count": int(np.count_nonzero(ring)),
            "smooth_node_fraction": float(
                np.count_nonzero(ring) / np.count_nonzero(smooth)
            ),
            "smooth_highpass_energy_fraction": capture(ring),
        }

    component_energy = np.sum(mass[smooth, None] * values[smooth] ** 2, axis=0)
    correlations: dict[str, Any] = {}
    if current_pressure_jump_score is not None:
        score = np.asarray(current_pressure_jump_score, dtype=np.float64)
        if score.shape != expected_shape:
            raise ValueError("current_pressure_jump_score must have shape [N]")
        correlations = spatial_correlation_summary(
            np.linalg.norm(values, axis=-1),
            {
                "current_pressure_jump_score": score,
                "current_shock_hop_distance": hop_distance.astype(np.float64),
            },
            mask=smooth,
        )

    oracle_capture = capture(oracle_mask)
    halo_capture = capture(causal_smooth)
    return {
        "status": "available",
        "smooth_node_count": int(np.count_nonzero(smooth)),
        "smooth_highpass_energy": smooth_energy,
        "component_energy_fraction": (
            component_energy / max(float(np.sum(component_energy)), 1.0e-30)
        ).tolist(),
        "truth_oracle": {
            "node_fraction_cap": ORACLE_NODE_FRACTION,
            "selected_node_count": int(np.count_nonzero(oracle_mask)),
            "selected_smooth_node_fraction": float(
                np.count_nonzero(oracle_mask) / np.count_nonzero(smooth)
            ),
            "smooth_highpass_energy_capture": oracle_capture,
        },
        "current_state_sensor": {
            "source": "current-state pressure-jump top decile; no target or rollout input",
            "shock_quantile": SHOCK_QUANTILE,
            "halo_hops": CAUSAL_HALO_HOPS,
            "seed_node_count": int(np.count_nonzero(seed)),
            "seed_fraction_interior": float(
                np.count_nonzero(seed) / np.count_nonzero(interior)
            ),
            "halo_node_count": int(np.count_nonzero(causal_halo)),
            "halo_fraction_interior": float(
                np.count_nonzero(causal_halo) / np.count_nonzero(interior)
            ),
            "halo_smooth_node_count": int(np.count_nonzero(causal_smooth)),
            "smooth_highpass_energy_capture": halo_capture,
            "truth_oracle_node_recall": float(
                np.count_nonzero(oracle_mask & causal_halo)
                / max(np.count_nonzero(oracle_mask), 1)
            ),
        },
        "distance_rings": rings,
        "smooth_neighbor_cosine": _edge_neighbor_cosine(
            values, edges, mass, smooth
        ),
        "spatial_correlations": correlations,
    }


def _median(values: Sequence[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return None if not finite else float(np.median(finite))


def fresh_defect_locality_selector(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the frozen six-case/two-late-call D054 decision contract."""

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
    available = all(row.get("locality", {}).get("status") == "available" for row in row_list)
    replay_errors = [
        max(
            float(row.get("d053_replay", {}).get("interior_fresh_norm_relative_error", math.inf)),
            float(row.get("d053_replay", {}).get("smooth_highpass_fresh_norm_relative_error", math.inf)),
        )
        for row in row_list
    ]
    checks = {
        "schema": bool(row_list)
        and all(row.get("schema") == FRESH_LOCALITY_SCHEMA for row in row_list),
        "exact_six_by_sixty_rows": (
            len(row_list) == EXPECTED_TRAJECTORY_COUNT * len(EXPECTED_CALLS)
            and len(trajectories) == EXPECTED_TRAJECTORY_COUNT
            and not duplicate_keys
            and all(calls == list(EXPECTED_CALLS) for calls in calls_by_trajectory.values())
        ),
        "all_locality_metrics_available": available,
        "no_smooth_mask_fallback": all(
            row.get("mask_contract", {}).get("smooth_fallback_to_interior") is False
            for row in row_list
        ),
        "all_teacher_predictions_admissible": all(
            row.get("teacher_admissibility", {}).get("all_admissible") is True
            for row in row_list
        ),
        "d053_fresh_norm_replay": bool(replay_errors)
        and max(replay_errors) <= D053_NORM_REPLAY_RELATIVE_TOLERANCE,
    }
    contract_complete = all(checks.values())

    late: dict[str, Any] = {}
    oracle_repeated = True
    causal_repeated = True
    for call in LATE_CALLS:
        selected = [row for row in row_list if int(row.get("call_index", -1)) == call]
        oracle_values = [
            float(row["locality"]["truth_oracle"]["smooth_highpass_energy_capture"])
            for row in selected
            if row.get("locality", {}).get("status") == "available"
        ]
        halo_values = [
            float(row["locality"]["current_state_sensor"]["smooth_highpass_energy_capture"])
            for row in selected
            if row.get("locality", {}).get("status") == "available"
        ]
        support_values = [
            float(row["locality"]["current_state_sensor"]["halo_fraction_interior"])
            for row in selected
            if row.get("locality", {}).get("status") == "available"
        ]
        oracle_count = sum(value >= ORACLE_ENERGY_CAPTURE_MIN for value in oracle_values)
        causal_count = sum(
            capture >= CAUSAL_HALO_ENERGY_CAPTURE_MIN
            and support <= CAUSAL_HALO_SUPPORT_MAX
            for capture, support in zip(halo_values, support_values, strict=True)
        )
        oracle_repeated &= len(selected) == EXPECTED_TRAJECTORY_COUNT and oracle_count >= REQUIRED_CASES
        causal_repeated &= len(selected) == EXPECTED_TRAJECTORY_COUNT and causal_count >= REQUIRED_CASES
        late[str(call)] = {
            "row_count": len(selected),
            "truth_oracle_passing_case_count": oracle_count,
            "causal_halo_passing_case_count": causal_count,
            "median_truth_oracle_energy_capture": _median(oracle_values),
            "median_causal_halo_energy_capture": _median(halo_values),
            "median_causal_halo_fraction_interior": _median(support_values),
        }

    if not contract_complete:
        classification = "incomplete_contract"
        route = "stop_without_training"
    elif oracle_repeated and causal_repeated:
        classification = "shock_conditioned_local_target_candidate"
        route = "authorize_one_matched_tiny_fit_contract_only"
    elif oracle_repeated:
        classification = "localized_but_not_causally_shock_localizable"
        route = "reject_simple_shock_gate_no_training"
    elif causal_repeated:
        classification = "sensor_aligned_but_not_20pct_sparse"
        route = "reject_bounded_local_target_no_training"
    else:
        classification = "diffuse_or_unresolved_fresh_defect"
        route = "reject_local_target_no_training"

    return {
        "version": "fresh_defect_locality_selector_d054_v1",
        "contract_complete": contract_complete,
        "contract_checks": checks,
        "duplicate_keys": [list(key) for key in duplicate_keys],
        "row_count": len(row_list),
        "trajectory_count": len(trajectories),
        "late_calls": late,
        "oracle_locality_repeated": oracle_repeated,
        "causal_locality_repeated": causal_repeated,
        "classification": classification,
        "route": route,
        "thresholds": {
            "truth_oracle_node_fraction": ORACLE_NODE_FRACTION,
            "truth_oracle_energy_capture_min": ORACLE_ENERGY_CAPTURE_MIN,
            "causal_halo_hops": CAUSAL_HALO_HOPS,
            "causal_halo_support_fraction_max": CAUSAL_HALO_SUPPORT_MAX,
            "causal_halo_energy_capture_min": CAUSAL_HALO_ENERGY_CAPTURE_MIN,
            "required_cases_per_late_call": REQUIRED_CASES,
            "d053_norm_replay_relative_tolerance": D053_NORM_REPLAY_RELATIVE_TOLERANCE,
        },
        "claim_boundary": (
            "necessary locality evidence for one target factorization on the frozen "
            "validation cohort; it is not learnability or rollout improvement"
        ),
    }


def _load_d053_rows(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["trajectory"]), int(row["call_index"]))
        if key in result:
            raise ValueError(f"duplicate D053 row: {key}")
        result[key] = row
    return result


def _relative_replay_error(value: float | None, reference: Any) -> float:
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
    if not source_summary_path.is_file() or not d053_summary_path.is_file():
        raise FileNotFoundError("D052 and D053 summary files are required")
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    d053_summary = json.loads(d053_summary_path.read_text(encoding="utf-8"))
    if source_summary.get("status") != "complete":
        raise ValueError("D054 requires a completed D052 source")
    d053_selector = d053_summary.get("selector", {})
    if (
        d053_summary.get("status") != "complete"
        or d053_selector.get("contract_complete") is not True
        or d053_selector.get("classification") != "mixed_or_split"
    ):
        raise ValueError("D054 requires the completed mixed/split D053 result")
    source_digest = sha256_file(source_summary_path)
    if d053_summary.get("source", {}).get("summary_sha256") != source_digest:
        raise ValueError("D052 source summary does not match the D053 binding")
    records = _source_records(source_summary)
    d053_rows_path = args.d053_dir / "error_sources.jsonl"
    d053_rows = _load_d053_rows(d053_rows_path)

    checkpoint = load_checkpoint(args.checkpoint)
    checkpoint_digest = sha256_file(args.checkpoint)
    if d053_summary.get("checkpoint", {}).get("sha256") != checkpoint_digest:
        raise ValueError("checkpoint SHA-256 differs from D053")
    if checkpoint.get("boundary_mode") != "model_all_nodes" or not bool(
        checkpoint.get("raw_recurrence")
    ):
        raise ValueError("D054 requires legal model-all-node raw recurrence")
    store = PCNOEuler2DShardStore(args.data_dir)
    if store.manifest_digest != checkpoint.get("data_manifest_digest"):
        raise ValueError("checkpoint and shard manifest digests differ")
    if store.manifest.get("weight_provenance") != "validated_physical_cell_volume_normalized":
        raise ValueError("D054 requires validated physical cell-volume weights")
    data_manifest_digest = store.manifest_digest
    model = build_model(checkpoint, device)
    component_scale = model.state_scale.detach().cpu().numpy().astype(np.float64).reshape(1, -1)

    args.output_dir.mkdir(parents=True)
    rows_path = args.output_dir / "fresh_defect_locality.jsonl"
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
                if _artifact_scalar(artifact, "config_digest") != str(
                    checkpoint["config_digest"]
                ):
                    raise ValueError("source trajectory and checkpoint config differ")
                valid_length = int(_artifact_scalar(artifact, "valid_length"))
                start_frame = int(_artifact_scalar(artifact, "start_frame"))
                step_stride = int(_artifact_scalar(artifact, "step_stride"))
                delta_t = float(_artifact_scalar(artifact, "delta_t"))
                arrays = {
                    name: np.array(artifact[name], copy=True)
                    for name in (
                        "positions",
                        "edges",
                        "node_type",
                        "physical_cell_volume_weights",
                        "reference_currents",
                        "targets",
                        "physical_target_times",
                    )
                }
            if step_stride != int(checkpoint["step_stride"]):
                raise ValueError("source trajectory and checkpoint strides differ")
            if arrays["reference_currents"].shape != arrays["targets"].shape:
                raise ValueError(f"source current/target shapes differ for {key}")
            expected_shape = (valid_length, arrays["positions"].shape[0], 4)
            if arrays["targets"].shape != expected_shape:
                raise ValueError(f"invalid source state shape for {key}")
            if arrays["physical_target_times"].shape != (valid_length,):
                raise ValueError(f"invalid physical-time shape for {key}")

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
            for offset in range(valid_length):
                call_index = offset + 1
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
                    dilation_hops=D053_SHOCK_DILATION_HOPS,
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

                fresh = (np.asarray(teacher_np, dtype=np.float64) - np.asarray(target_np, dtype=np.float64)) / component_scale
                fresh_highpass = node_highpass_field(fresh, arrays["edges"])
                current_primitive = conservative_to_primitive_raw(current_np, gamma=model.gamma)
                pressure_jump_score = shock_front_scores(
                    current_primitive, arrays["edges"], scalar_index=3
                )
                locality = fresh_defect_locality_metrics(
                    fresh_highpass,
                    edges=arrays["edges"],
                    adjacency=adjacency,
                    weights=weights,
                    interior_mask=interior,
                    smooth_mask=smooth,
                    current_shock_seed=current_shock_seed,
                    current_pressure_jump_score=pressure_jump_score,
                )

                d053_row = d053_rows.get((key, call_index))
                if d053_row is None:
                    raise ValueError(f"D053 row is missing for {(key, call_index)}")
                fresh_full_norm = _weighted_norm(fresh, weights, interior)
                fresh_highpass_norm = _weighted_norm(fresh_highpass, weights, smooth)
                row = {
                    "schema": FRESH_LOCALITY_SCHEMA,
                    "trajectory": key,
                    "call_index": call_index,
                    "physical_target_time": float(arrays["physical_target_times"][offset]),
                    "delta_t": delta_t,
                    "locality": locality,
                    "mask_contract": mask_contract,
                    "d053_replay": {
                        "interior_fresh_norm_relative_error": _relative_replay_error(
                            fresh_full_norm,
                            d053_row.get("regions", {}).get("interior_full", {}).get("fresh_defect_norm"),
                        ),
                        "smooth_highpass_fresh_norm_relative_error": _relative_replay_error(
                            fresh_highpass_norm,
                            d053_row.get("regions", {}).get("smooth_highpass", {}).get("fresh_defect_norm"),
                        ),
                    },
                    "teacher_admissibility": raw_admissibility_summary(teacher_np, gamma=model.gamma),
                    "claim_boundary": (
                        "truth selects only the 20% sparsity upper bound; the pressure-jump "
                        "sensor and its halo use the current model input only"
                    ),
                }
                rows.append(row)
                append_jsonl(rows_path, row)
            print(json.dumps({"trajectory": key, "rows": valid_length}, sort_keys=True), flush=True)
    finally:
        store.close()

    selector = fresh_defect_locality_selector(rows)
    summary = {
        "schema": FRESH_LOCALITY_SCHEMA,
        "status": "complete" if selector["contract_complete"] else "failed_contract",
        "source": {
            "d052_artifact_dir_name": args.source_dir.name,
            "d052_summary_sha256": source_digest,
            "d053_artifact_dir_name": args.d053_dir.name,
            "d053_summary_sha256": sha256_file(d053_summary_path),
            "d053_rows_sha256": sha256_file(d053_rows_path),
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
            "trajectory_count": len(records),
            "row_count": len(rows),
            "test_trajectory_access": [],
            "training_or_inference_intervention": False,
            "weights": "validated physical cell volume",
        },
        "hypothesis": {
            "name": "current-state shock-conditioned multiresolution residual target",
            "necessary_conditions": (
                "late fresh smooth-high-pass defect is strongly sparse under a 20% "
                "truth oracle and is captured by a bounded current-state shock halo"
            ),
            "method_if_passed": (
                "one matched tiny-fit contract may factor the conservative residual "
                "into a frozen-style global prediction and a zero-initialized, "
                "current-state-gated local detail; no serious run is authorized"
            ),
        },
        "selector": selector,
        "artifact_schema": {
            "fresh_defect_locality.jsonl": (
                "one scalar row per validation trajectory/call; no state arrays"
            )
        },
        "claim_boundary": {
            "verified": (
                "spatial concentration and current-state sensor coverage of D053's "
                "fresh teacher defect on the frozen validation cohort"
            ),
            "plausible": "one representation route only if every selector gate passes",
            "unsupported": (
                "learnability, autonomous improvement, strength-OOD behavior, physical "
                "flux prediction, or conservation by construction"
            ),
        },
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(selector, indent=2, sort_keys=True), flush=True)
    if not selector["contract_complete"]:
        raise RuntimeError("D054 failed its frozen diagnostic contract")


if __name__ == "__main__":
    main()
