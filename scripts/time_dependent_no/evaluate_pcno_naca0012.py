#!/usr/bin/env python3
"""Evaluate exactly the frozen three-seed NACA0012 PCNO R0 baseline."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import shutil
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.time_dependent_no.train_pcno_naca0012 import (
    EXPECTED_DATASET_FILES,
    FINAL_HASH_MANIFEST_SCHEMA,
    SOURCE_MANIFEST_SCHEMA,
    SOURCE_RECORDS_AT_IMPORT,
    TRAINING_SCHEMA,
    _load_dataset,
    _load_npy_snapshot,
    _load_self_hashed_snapshot,
    _runtime,
    _self_hashed,
    _write_json,
)
from utility.time_dependent_no.pcno_naca0012 import (
    BASELINE_CONTRACT_SHA256,
    NACANormalization,
    VerifiedNACAGeometry,
    build_naca_pcno,
    load_naca_baseline_contract,
    predict_normalized_residual,
    recurrent_step,
    validate_naca_model_config,
)
from utility.time_dependent_no.su2_restart_contract import sha256_file

EVALUATION_SCHEMA = "time_dependent_no.su2_naca0012_pcno_evaluation.v1"
CHECKPOINT_SCHEMA = "time_dependent_no.su2_naca0012_pcno_checkpoint.v1"
QUALIFICATION_SCHEMA = "time_dependent_no.su2_naca0012_native_replay_qualification.v1"
QUALIFICATION_SHA256 = (
    "8269354d4a021b921f6c1d99c4f0dd416896b62c23fb5f9beb77fab0f0e960d6"
)
SEEDS = (17, 29, 43)
FIELDS = ("Density", "Momentum_x", "Momentum_y", "Energy", "Nu_Tilde")
VIEWS = ("uniform_node", "near_body_wake_uniform_node", "physical_vertex_area")
HORIZONS = (1, 35, 104, 208)
EARLY_WINDOW = (1, 35)
LATE_WINDOW = (174, 208)
EXPECTED_TRAINING_FILES = {
    "best.pt",
    "config.json",
    "history.json",
    "input_manifest.json",
    "last.pt",
    "runtime_manifest.json",
    "status.json",
    "summary.json",
}


class EvaluationOutputError(RuntimeError):
    """An output packet could not be written after all inputs were validated."""


def _json_number(value: float) -> float | str:
    if math.isinf(value):
        return "Infinity" if value > 0.0 else "-Infinity"
    if math.isnan(value):
        return "NaN"
    return float(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        return _json_number(float(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _numeric_json_number(value: Any) -> float:
    if isinstance(value, bool):
        raise TypeError("boolean is not a numeric report value")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if value == "Infinity":
        return math.inf
    if value == "-Infinity":
        return -math.inf
    if value == "NaN":
        return math.nan
    raise ValueError("unsupported JSON numeric encoding")


def _same_json_number(left: Any, right: Any) -> bool:
    left_number = _numeric_json_number(left)
    right_number = _numeric_json_number(right)
    if math.isnan(left_number) or math.isnan(right_number):
        return math.isnan(left_number) and math.isnan(right_number)
    return left_number == right_number


def _nearest_rank(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("nearest-rank aggregation received no values")
    ordered = sorted(float(value) for value in values)
    rank = max(1, math.ceil(probability * len(ordered)))
    return ordered[rank - 1]


def _aggregate(values: Sequence[float]) -> dict[str, float | str]:
    if not values:
        raise ValueError("aggregation received no values")
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": _json_number(float(np.median(array))),
        "q90_nearest_rank": _json_number(_nearest_rank(values, 0.9)),
        "maximum": _json_number(float(np.max(array))),
    }


def _first_positive_step(rows: Sequence[Mapping[str, Any]], field: str) -> int | None:
    matches = [
        int(row["horizon"])
        for row in rows
        if math.isfinite(float(row[field])) and float(row[field]) > 0.0
    ]
    return min(matches) if matches else None


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(f"artifact is aliased: {path}")
    resolved = path.resolve()
    relative = resolved.relative_to(root.resolve()).as_posix()
    if not resolved.is_file():
        raise ValueError(f"artifact is absent or aliased: {path}")
    return {
        "relative_path": relative,
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _verify_training_packet(
    root: Path,
    dataset_manifest: Mapping[str, Any],
    dataset_packet_sha256: str,
    contract: Mapping[str, Any],
    geometry: VerifiedNACAGeometry,
    device: torch.device,
) -> tuple[
    int,
    torch.nn.Module,
    dict[str, Any],
    str,
    str,
    Path,
    dict[str, Any],
]:
    if root.is_symlink():
        raise ValueError("training packet is aliased")
    training_root = root.resolve()
    if not training_root.is_dir():
        raise ValueError("training packet is absent or aliased")
    final_path = training_root / "final_hash_manifest.json"
    if final_path.is_symlink() or not final_path.is_file():
        raise ValueError("training packet is incomplete")
    final_bytes = final_path.read_bytes()
    final_sha256 = sha256(final_bytes).hexdigest()
    final = json.loads(final_bytes)
    files = final.get("files", {}) if isinstance(final, dict) else {}
    observed = {path.name for path in training_root.iterdir()}
    if (
        final.get("schema") != FINAL_HASH_MANIFEST_SCHEMA
        or final.get("contract_sha256") != BASELINE_CONTRACT_SHA256
        or final.get("self_hash_excluded") is not True
        or final.get("prospective_opened") is not False
        or final.get("sealed_opened") is not False
        or observed != set(files) | {"final_hash_manifest.json"}
        or set(files) != EXPECTED_TRAINING_FILES
    ):
        raise ValueError("training final-hash manifest differs")
    for name, record in files.items():
        candidate = training_root / name
        if candidate.is_symlink():
            raise ValueError(f"training artifact is aliased: {name}")
        path = candidate.resolve()
        if path.parent != training_root or not path.is_file():
            raise ValueError(f"training artifact is absent or aliased: {name}")
        if (
            record.get("relative_path") != name
            or path.stat().st_size != record.get("bytes")
            or sha256_file(path) != record.get("sha256")
        ):
            raise ValueError(f"training artifact differs from final hashes: {name}")

    config = _load_self_hashed_snapshot(
        training_root,
        files["config.json"],
        "config.json",
        TRAINING_SCHEMA,
    )
    summary = _load_self_hashed_snapshot(
        training_root,
        files["summary.json"],
        "summary.json",
        TRAINING_SCHEMA,
    )
    inputs = _load_self_hashed_snapshot(
        training_root,
        files["input_manifest.json"],
        "input_manifest.json",
        "time_dependent_no.su2_naca0012_pcno_training_inputs.v1",
    )
    seed = summary.get("seed")
    if isinstance(seed, bool) or seed not in SEEDS:
        raise ValueError("training packet has an unsupported seed")
    if (
        config.get("seed") != seed
        or summary.get("status") != "complete"
        or summary.get("completed_epochs") != 100
        or summary.get("contract_sha256") != BASELINE_CONTRACT_SHA256
        or summary.get("dataset_manifest_payload_sha256")
        != dataset_manifest["canonical_payload_sha256"]
        or summary.get("dataset_final_hash_manifest_sha256") != dataset_packet_sha256
        or summary.get("source_set_sha256")
        != dataset_manifest["source_manifest"]["source_set_sha256"]
        or summary.get("checkpoint_selection_used_rollout") is not False
        or summary.get("prospective_opened") is not False
        or summary.get("sealed_opened") is not False
        or inputs.get("dataset_manifest_payload_sha256")
        != dataset_manifest["canonical_payload_sha256"]
        or inputs.get("dataset_final_hash_manifest_sha256") != dataset_packet_sha256
        or inputs.get("source_set_sha256")
        != dataset_manifest["source_manifest"]["source_set_sha256"]
        or config.get("epochs") != 100
        or config.get("dataset_final_hash_manifest_sha256") != dataset_packet_sha256
        or config.get("effective_batch_size") != 4
        or config.get("checkpoint_epochs") != list(range(5, 101, 5))
        or config.get("rollout_used_for_selection") is not False
        or config.get("production_device_binding")
        != inputs.get("production_device_binding")
    ):
        raise ValueError("training packet provenance or frozen configuration differs")

    best_record = files["best.pt"]
    best_bytes = (training_root / "best.pt").read_bytes()
    if (
        len(best_bytes) != best_record["bytes"]
        or sha256(best_bytes).hexdigest() != best_record["sha256"]
    ):
        raise ValueError("selected checkpoint bytes differ from final hashes")
    checkpoint = torch.load(
        io.BytesIO(best_bytes), map_location=device, weights_only=True
    )
    if (
        checkpoint.get("schema") != CHECKPOINT_SCHEMA
        or checkpoint.get("contract_sha256") != BASELINE_CONTRACT_SHA256
        or checkpoint.get("dataset_manifest_payload_sha256")
        != dataset_manifest["canonical_payload_sha256"]
        or checkpoint.get("dataset_final_hash_manifest_sha256") != dataset_packet_sha256
        or checkpoint.get("source_set_sha256")
        != dataset_manifest["source_manifest"]["source_set_sha256"]
        or checkpoint.get("seed") != seed
        or checkpoint.get("epoch") != summary.get("best_epoch")
        or checkpoint.get("best_epoch") != summary.get("best_epoch")
        or not _same_json_number(
            checkpoint.get("best_development_score"),
            summary.get("best_development_score"),
        )
        or checkpoint.get("model_only") is not True
    ):
        raise ValueError("selected checkpoint provenance differs")
    validate_naca_model_config(checkpoint["model_config"], contract, geometry)
    model = build_naca_pcno(contract, geometry).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    for name, record in files.items():
        candidate = training_root / name
        if candidate.is_symlink():
            raise ValueError(f"training artifact became aliased: {name}")
        if (
            not candidate.is_file()
            or candidate.stat().st_size != record.get("bytes")
            or sha256_file(candidate) != record.get("sha256")
        ):
            raise ValueError(f"training artifact changed while loading: {name}")
    if sha256_file(final_path) != final_sha256:
        raise ValueError("training final-hash manifest changed while loading")
    return (
        int(seed),
        model,
        summary,
        final_sha256,
        training_root.name,
        training_root,
        dict(files),
    )


def _reverify_packet_files(
    root: Path,
    files: Mapping[str, Mapping[str, Any]],
    final_sha256: str,
) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("verified packet became absent or aliased")
    if {path.name for path in root.iterdir()} != set(files) | {
        "final_hash_manifest.json"
    }:
        raise ValueError("verified packet gained or lost an artifact")
    for name, record in files.items():
        candidate = root / name
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError(
                f"verified packet artifact became absent or aliased: {name}"
            )
        payload = candidate.read_bytes()
        if (
            record.get("relative_path") != name
            or len(payload) != record.get("bytes")
            or sha256(payload).hexdigest() != record.get("sha256")
        ):
            raise ValueError(f"verified packet artifact changed: {name}")
    final_path = root / "final_hash_manifest.json"
    if final_path.is_symlink() or not final_path.is_file():
        raise ValueError("verified packet final manifest became absent or aliased")
    if sha256(final_path.read_bytes()).hexdigest() != final_sha256:
        raise ValueError("verified packet final manifest changed")


def _reverify_dataset_packet(root: Path, final_sha256: str) -> None:
    if root.is_symlink():
        raise ValueError("dataset packet became aliased")
    dataset_root = root.resolve()
    final_path = dataset_root / "final_hash_manifest.json"
    if final_path.is_symlink() or not final_path.is_file():
        raise ValueError("dataset final manifest became absent or aliased")
    payload = final_path.read_bytes()
    if sha256(payload).hexdigest() != final_sha256:
        raise ValueError("dataset final manifest changed")
    final = json.loads(payload)
    if (
        not isinstance(final, dict)
        or final.get("schema") != FINAL_HASH_MANIFEST_SCHEMA
        or set(final.get("files", {})) != EXPECTED_DATASET_FILES
    ):
        raise ValueError("dataset final manifest became invalid")
    _reverify_packet_files(dataset_root, final["files"], final_sha256)


def _reverify_live_source(source: Mapping[str, Any]) -> None:
    if source.get("files") != SOURCE_RECORDS_AT_IMPORT:
        raise ValueError("bound source differs from import-time execution bytes")
    records: dict[str, Any] = {}
    for relative, expected in source.get("files", {}).items():
        candidate = REPO_ROOT / relative
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError(f"bound source became absent or aliased: {relative}")
        payload = candidate.read_bytes()
        observed = {"bytes": len(payload), "sha256": sha256(payload).hexdigest()}
        if observed != expected:
            raise ValueError(f"bound source changed during evaluation: {relative}")
        records[relative] = observed
    if sha256(
        json.dumps(
            records, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest() != source.get("source_set_sha256"):
        raise ValueError("bound source-set digest changed during evaluation")


def _qualification(
    path: Path, contract: Mapping[str, Any]
) -> tuple[dict[str, Any], str]:
    if path.is_symlink():
        raise ValueError("native replay qualification is aliased")
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError("native replay qualification is absent or aliased")
    payload = resolved.read_bytes()
    observed_sha256 = sha256(payload).hexdigest()
    if observed_sha256 != QUALIFICATION_SHA256:
        raise ValueError("native replay qualification differs from its frozen hash")
    value = json.loads(payload)
    expected = contract["evaluation"]["native_replay_margin"][
        "qualified_native_replay_errors"
    ]
    views = value.get("canonical_target_comparison", {}).get("dynamic_error_views", {})
    if (
        not isinstance(value, dict)
        or value.get("schema") != QUALIFICATION_SCHEMA
        or value.get("native_execution", {}).get("status_c")
        != "execution_and_evaluation_succeeded"
        or value.get("native_execution", {}).get("status_d")
        != "execution_and_evaluation_succeeded"
        or value.get("native_execution", {})
        .get("output", {})
        .get("bitwise_deterministic")
        is not True
    ):
        raise ValueError("native replay qualification is not successful")
    for view, expected_error in expected.items():
        if views.get(view, {}).get("component_balanced_relative_l2") != expected_error:
            raise ValueError(f"qualified native replay error differs for {view}")
    return value, observed_sha256


def _view_weights(geometry: VerifiedNACAGeometry) -> dict[str, np.ndarray]:
    coordinates = geometry.native_coordinates
    count = geometry.num_nodes
    near = (
        (coordinates[:, 0] >= -1.0)
        & (coordinates[:, 0] <= 10.0)
        & (coordinates[:, 1] >= -5.0)
        & (coordinates[:, 1] <= 5.0)
    )
    if int(np.count_nonzero(near)) != 8392:
        raise ValueError("near-body/wake view differs from the qualified mesh")
    near_weights = np.zeros(count, dtype=np.float64)
    near_weights[near] = 1.0 / np.count_nonzero(near)
    weights = {
        "uniform_node": np.full(count, 1.0 / count, dtype=np.float64),
        "near_body_wake_uniform_node": near_weights,
        "physical_vertex_area": np.asarray(
            geometry.node_weights[:, 0], dtype=np.float64
        ),
    }
    for name, value in weights.items():
        if not np.isclose(np.sum(value), 1.0, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"view weights do not sum to one: {name}")
    return weights


def _field_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    normalization: NACANormalization,
) -> dict[str, Any]:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(
        target, dtype=np.float64
    )
    scale_sq = np.sum(
        weights[:, None] * np.square(error / normalization.state_scale), axis=0
    )
    e_field = np.sqrt(scale_sq)
    numerator = np.sum(weights[:, None] * np.square(error), axis=0)
    denominator = np.maximum(
        np.sum(weights[:, None] * np.square(target), axis=0),
        np.square(normalization.relative_l2_floor),
    )
    l_field = np.sqrt(numerator / denominator)
    return {
        "train_state_scale_per_field": e_field,
        "train_state_scale": float(np.sqrt(np.mean(np.square(e_field)))),
        "relative_l2_per_field": l_field,
        "relative_l2": float(np.sqrt(np.mean(np.square(l_field)))),
    }


def _normalized_residual_metrics(
    prediction: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, float]:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(
        target, dtype=np.float64
    )
    per_field = np.sqrt(np.sum(weights[:, None] * np.square(error), axis=0))
    return per_field, float(np.sqrt(np.mean(np.square(per_field))))


def _metric_columns(prefix: str, values: Sequence[float]) -> dict[str, float]:
    return {
        f"{prefix}_{field}": float(value)
        for field, value in zip(FIELDS, values, strict=True)
    }


def _one_step_method_view_summaries(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for method in ("pcno", "persistence"):
        view_summaries: dict[str, Any] = {}
        for view in VIEWS:
            selected = [
                row for row in rows if row["method"] == method and row["view"] == view
            ]
            if len(selected) != 238:
                raise ValueError(f"one-step summary lacks 238 rows for {method}/{view}")
            view_summaries[view] = {
                "transition_count": len(selected),
                "normalized_residual_rmse": _aggregate(
                    [float(row["normalized_residual_rmse"]) for row in selected]
                ),
                "normalized_residual_rmse_per_field": {
                    field: _aggregate(
                        [
                            float(row[f"normalized_residual_rmse_{field}"])
                            for row in selected
                        ]
                    )
                    for field in FIELDS
                },
                "next_state_relative_l2": _aggregate(
                    [float(row["next_state_relative_l2"]) for row in selected]
                ),
                "next_state_relative_l2_per_field": {
                    field: _aggregate(
                        [
                            float(row[f"next_state_relative_l2_{field}"])
                            for row in selected
                        ]
                    )
                    for field in FIELDS
                },
            }
        summaries[method] = {"views": view_summaries}
    return summaries


def _one_step_metrics(
    *,
    seed: int,
    model: torch.nn.Module,
    states: np.ndarray,
    frame_indices: np.ndarray,
    centers: Sequence[int],
    geometry: VerifiedNACAGeometry,
    normalization: NACANormalization,
    weights: Mapping[str, np.ndarray],
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    positions = {int(index): offset for offset, index in enumerate(frame_indices)}
    geometry_batch = geometry.expand(1, device)
    fourier = model.prepare_fourier_tensors(geometry_batch)  # type: ignore[attr-defined]
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for center in centers:
            previous = torch.as_tensor(
                np.array(states[positions[center - 1]], dtype=np.float32, copy=True),
                device=device,
            ).unsqueeze(0)
            current = torch.as_tensor(
                np.array(states[positions[center]], dtype=np.float32, copy=True),
                device=device,
            ).unsqueeze(0)
            target_state = np.asarray(states[positions[center + 1]], dtype=np.float64)
            target_residual = normalization.normalize_residual(
                np.subtract(target_state, states[positions[center]], dtype=np.float64)
            )
            predicted_residual_tensor = predict_normalized_residual(
                model,
                previous,
                current,
                geometry_batch,
                normalization,
                fourier_tensors=fourier,
            )
            predicted_state_tensor = current + normalization.decode_residual(
                predicted_residual_tensor
            )
            predicted_residual = (
                predicted_residual_tensor[0].detach().cpu().numpy().astype(np.float64)
            )
            predicted_state = (
                predicted_state_tensor[0].detach().cpu().numpy().astype(np.float64)
            )
            persistence_state = np.asarray(states[positions[center]], dtype=np.float64)
            for method, residual, state_prediction in (
                ("pcno", predicted_residual, predicted_state),
                (
                    "persistence",
                    np.zeros_like(target_residual),
                    persistence_state,
                ),
            ):
                for view in VIEWS:
                    per_field_r, balanced_r = _normalized_residual_metrics(
                        residual, target_residual, weights[view]
                    )
                    state_metrics = _field_metrics(
                        state_prediction,
                        target_state,
                        weights[view],
                        normalization,
                    )
                    rows.append(
                        {
                            "seed": seed,
                            "method": method,
                            "transition_center": center,
                            "view": view,
                            "normalized_residual_rmse": balanced_r,
                            **_metric_columns("normalized_residual_rmse", per_field_r),
                            "next_state_relative_l2": state_metrics["relative_l2"],
                            **_metric_columns(
                                "next_state_relative_l2",
                                state_metrics["relative_l2_per_field"],
                            ),
                        }
                    )
    uniform_pcno = [
        row for row in rows if row["method"] == "pcno" and row["view"] == "uniform_node"
    ]
    if len(uniform_pcno) != 238:
        raise ValueError("one-step evaluation lacks the dense development population")
    component_r = [float(row["normalized_residual_rmse"]) for row in uniform_pcno]
    component_l = [float(row["next_state_relative_l2"]) for row in uniform_pcno]
    field_medians = {
        field: float(
            np.median(
                [row[f"normalized_residual_rmse_{field}"] for row in uniform_pcno]
            )
        )
        for field in FIELDS
    }
    adequate = (
        float(np.median(component_r)) <= 0.5
        and all(value <= 0.75 for value in field_medians.values())
        and float(np.median(component_l)) <= 0.05
        and _nearest_rank(component_l, 0.9) <= 0.10
    )
    summary = {
        "normalized_residual_rmse": _aggregate(component_r),
        "normalized_residual_rmse_field_medians": _json_safe(field_medians),
        "next_state_relative_l2": _aggregate(component_l),
        "all_methods_views": _one_step_method_view_summaries(rows),
        "adequately_trained": adequate,
    }
    return rows, summary


def _airfoil_geometry(
    geometry: VerifiedNACAGeometry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    airfoil = geometry.boundary_one_hot[:, 1] == 1.0
    farfield = geometry.boundary_one_hot[:, 2] == 1.0
    counts: Counter[tuple[int, int]] = Counter()
    for element in geometry.elements[:, 1:]:
        for first, second in zip(element, np.roll(element, -1), strict=True):
            edge = tuple(sorted((int(first), int(second))))
            counts[edge] += 1
    edges = np.asarray(
        [
            edge
            for edge, count in counts.items()
            if count == 1 and airfoil[list(edge)].all()
        ],
        dtype=np.int64,
    )
    if edges.shape != (128, 2):
        raise ValueError("airfoil boundary-edge reconstruction differs")
    coordinates = geometry.native_coordinates
    centroid = np.mean(coordinates[airfoil], axis=0)
    normals = np.zeros((geometry.num_nodes, 2), dtype=np.float64)
    outward_edges = np.empty((edges.shape[0], 2), dtype=np.float64)
    lengths = np.empty(edges.shape[0], dtype=np.float64)
    for offset, (first, second) in enumerate(edges):
        delta = coordinates[second] - coordinates[first]
        length = float(np.linalg.norm(delta))
        normal = np.asarray([delta[1], -delta[0]], dtype=np.float64) / length
        midpoint = 0.5 * (coordinates[first] + coordinates[second])
        if float(np.dot(normal, midpoint - centroid)) < 0.0:
            normal = -normal
        normals[first] += length * normal
        normals[second] += length * normal
        outward_edges[offset] = normal
        lengths[offset] = length
    norms = np.linalg.norm(normals[airfoil], axis=1)
    if np.any(norms <= 0.0):
        raise ValueError("airfoil vertex normal is undefined")
    normals[airfoil] /= norms[:, None]
    return airfoil, farfield, edges, np.column_stack((outward_edges, lengths))


def _pressure_force(
    state: np.ndarray,
    edges: np.ndarray,
    edge_geometry: np.ndarray,
) -> tuple[float, float] | None:
    density = state[:, 0]
    if np.any(density <= 0.0):
        return None
    pressure = 0.4 * (
        state[:, 3]
        - (np.square(state[:, 1]) + np.square(state[:, 2])) / (2.0 * density)
    )
    if not np.all(np.isfinite(pressure)):
        return None
    normals = edge_geometry[:, :2]
    lengths = edge_geometry[:, 2]
    edge_pressure = 0.5 * (pressure[edges[:, 0]] + pressure[edges[:, 1]])
    force = -np.sum(edge_pressure[:, None] * normals * lengths[:, None], axis=0)
    angle = math.radians(17.0)
    drag_direction = np.asarray([math.cos(angle), math.sin(angle)])
    lift_direction = np.asarray([-math.sin(angle), math.cos(angle)])
    return float(np.dot(force, drag_direction)), float(np.dot(force, lift_direction))


def _force_scales(
    train_states: np.ndarray, edges: np.ndarray, edge_geometry: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(
        [
            _pressure_force(np.asarray(state, dtype=np.float64), edges, edge_geometry)
            for state in train_states
        ],
        dtype=np.float64,
    )
    if values.shape != (240, 2) or not np.all(np.isfinite(values)):
        raise ValueError("train-only pressure-force scale is invalid")
    rms = np.sqrt(np.mean(np.square(values), axis=0))
    scale = np.maximum(np.std(values, axis=0), 1.0e-8 * np.maximum(rms, 1.0))
    return values, scale


def _structure_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    geometry: VerifiedNACAGeometry,
    normalization: NACANormalization,
    airfoil: np.ndarray,
    farfield: np.ndarray,
    normals: np.ndarray,
    edges: np.ndarray,
    edge_geometry: np.ndarray,
    force_scale: np.ndarray,
) -> dict[str, Any]:
    airfoil_weights = np.full(
        np.count_nonzero(airfoil), 1.0 / np.count_nonzero(airfoil)
    )
    farfield_weights = np.full(
        np.count_nonzero(farfield), 1.0 / np.count_nonzero(farfield)
    )
    airfoil_error = _field_metrics(
        prediction[airfoil], target[airfoil], airfoil_weights, normalization
    )["train_state_scale"]
    farfield_error = _field_metrics(
        prediction[farfield], target[farfield], farfield_weights, normalization
    )["train_state_scale"]
    momentum_scale = math.sqrt(
        (normalization.state_scale[1] ** 2 + normalization.state_scale[2] ** 2) / 2.0
    )
    predicted_normal = np.sum(prediction[airfoil, 1:3] * normals[airfoil], axis=1)
    target_normal = np.sum(target[airfoil, 1:3] * normals[airfoil], axis=1)
    leakage = float(
        np.sqrt(np.mean(np.square((predicted_normal - target_normal) / momentum_scale)))
    )
    predicted_normal_rms = float(
        np.sqrt(np.mean(np.square(predicted_normal / momentum_scale)))
    )
    target_normal_rms = float(
        np.sqrt(np.mean(np.square(target_normal / momentum_scale)))
    )
    density = prediction[:, 0]
    positive_density = density > 0.0
    internal_energy = np.full(density.shape, np.nan, dtype=np.float64)
    internal_energy[positive_density] = prediction[positive_density, 3] - (
        np.square(prediction[positive_density, 1])
        + np.square(prediction[positive_density, 2])
    ) / (2.0 * density[positive_density])
    physical = geometry.node_weights[:, 0]
    integral_error = (
        np.abs(
            np.sum(physical[:, None] * prediction, axis=0)
            - np.sum(physical[:, None] * target, axis=0)
        )
        / normalization.state_scale
    )
    predicted_force = _pressure_force(prediction, edges, edge_geometry)
    target_force = _pressure_force(target, edges, edge_geometry)
    force_valid = predicted_force is not None and target_force is not None
    force_error = (
        np.abs(np.asarray(predicted_force) - np.asarray(target_force)) / force_scale
        if force_valid
        else np.asarray([math.nan, math.nan])
    )
    return {
        "airfoil_boundary_train_state_scale_error": airfoil_error,
        "farfield_boundary_train_state_scale_error": farfield_error,
        "wall_normal_momentum_leakage": leakage,
        "predicted_wall_normal_momentum_rms": predicted_normal_rms,
        "reference_wall_normal_momentum_rms": target_normal_rms,
        "density_nonpositive_fraction": float(np.mean(~positive_density)),
        "internal_energy_diagnostic_invalid_fraction": float(
            np.mean(~positive_density)
        ),
        "internal_energy_nonpositive_fraction_among_valid_density": float(
            np.mean(internal_energy[positive_density] <= 0.0)
        )
        if np.any(positive_density)
        else math.nan,
        "nu_tilde_minimum": float(np.min(prediction[:, 4])),
        "nu_tilde_negative_fraction": float(np.mean(prediction[:, 4] < 0.0)),
        **_metric_columns("volume_integral_error", integral_error),
        "volume_integral_error": float(np.sqrt(np.mean(np.square(integral_error)))),
        "pressure_force_valid": force_valid,
        "predicted_pressure_drag_proxy": predicted_force[0]
        if predicted_force is not None
        else math.nan,
        "predicted_pressure_lift_proxy": predicted_force[1]
        if predicted_force is not None
        else math.nan,
        "reference_pressure_drag_proxy": target_force[0]
        if target_force is not None
        else math.nan,
        "reference_pressure_lift_proxy": target_force[1]
        if target_force is not None
        else math.nan,
        "pressure_drag_proxy_normalized_error": float(force_error[0]),
        "pressure_lift_proxy_normalized_error": float(force_error[1]),
    }


def _train_template(
    prediction: np.ndarray,
    train_states: np.ndarray,
    train_indices: np.ndarray,
    weights: np.ndarray,
    normalization: NACANormalization,
    period: float,
    crossing: float,
) -> dict[str, Any]:
    differences = (train_states - prediction[None, ...]) / normalization.state_scale
    per_template = np.sqrt(
        np.mean(
            np.sum(weights[None, :, None] * np.square(differences), axis=1),
            axis=1,
        )
    )
    position = int(np.argmin(per_template))
    index = int(train_indices[position])
    return {
        "nearest_train_state_scale_distance": float(per_template[position]),
        "nearest_train_frame_index": index,
        "nearest_train_frame_phase": float(((index - crossing) / period) % 1.0),
    }


def _window_summary(
    values: Mapping[tuple[int, int, str, str], float],
    *,
    anchors: Sequence[int],
    method: str,
    view: str,
    first: int,
    last: int,
) -> dict[str, Any]:
    per_anchor: list[float] = []
    for anchor in anchors:
        series = [
            values[(anchor, horizon, view, method)]
            for horizon in range(first, last + 1)
        ]
        per_anchor.append(
            math.sqrt(float(np.mean(np.square(series))))
            if all(math.isfinite(item) for item in series)
            else math.inf
        )
    finite_only = [value for value in per_anchor if math.isfinite(value)]
    return {
        "per_anchor": {
            str(anchor): _json_number(value)
            for anchor, value in zip(anchors, per_anchor, strict=True)
        },
        "all_anchor_summary_failed_windows_are_infinity": _aggregate(per_anchor),
        "finite_anchor_only_summary": _aggregate(finite_only) if finite_only else None,
        "finite_anchor_count": len(finite_only),
    }


def _terminal_summary(
    errors: Mapping[tuple[int, int, str, str], float],
    relative_errors: Mapping[tuple[int, int, str, str], float],
    *,
    anchors: Sequence[int],
    method: str,
    view: str,
) -> dict[str, Any]:
    return {
        str(horizon): {
            "train_state_scale_error": _aggregate(
                [errors[(anchor, horizon, view, method)] for anchor in anchors]
            ),
            "component_balanced_relative_l2": _aggregate(
                [relative_errors[(anchor, horizon, view, method)] for anchor in anchors]
            ),
        }
        for horizon in HORIZONS
    }


def _admissibility_first_occurrence(
    structure_rows: Sequence[Mapping[str, Any]], anchors: Sequence[int]
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for method in ("pcno", "persistence"):
        method_records: dict[str, Any] = {}
        for anchor in anchors:
            rows = [
                row
                for row in structure_rows
                if row["method"] == method
                and row["anchor"] == anchor
                and row["finite"] is True
            ]
            method_records[str(anchor)] = {
                "density_nonpositive_first_step": _first_positive_step(
                    rows, "density_nonpositive_fraction"
                ),
                "internal_energy_nonpositive_among_valid_density_first_step": _first_positive_step(
                    rows,
                    "internal_energy_nonpositive_fraction_among_valid_density",
                ),
                "nu_tilde_negative_first_step": _first_positive_step(
                    rows, "nu_tilde_negative_fraction"
                ),
                "invalid_density_fraction_reported_separately": True,
            }
        summary[method] = method_records
    return summary


def _r0_decision(
    *, adequate_count: int, phenotype_pass: bool, margin_pass: bool
) -> str:
    if adequate_count != 3:
        return "R0_INCONCLUSIVE_INADEQUATE_TRAINING"
    if not phenotype_pass:
        return "R0_REJECT_NACA_AS_HERO_CORRECTION_NECESSITY_CASE"
    if not margin_pass:
        return "R0_INCONCLUSIVE_NATIVE_REPLAY_MARGIN_FAILED"
    return "R0_BASELINE_PHENOTYPE_SUPPORTED"


def _rollout_seed(
    *,
    seed: int,
    model: torch.nn.Module,
    development_states: np.ndarray,
    development_indices: np.ndarray,
    train_states: np.ndarray,
    train_indices: np.ndarray,
    anchors: Sequence[int],
    geometry: VerifiedNACAGeometry,
    normalization: NACANormalization,
    weights: Mapping[str, np.ndarray],
    contract: Mapping[str, Any],
    device: torch.device,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, np.ndarray],
    dict[str, Any],
]:
    positions = {int(index): offset for offset, index in enumerate(development_indices)}
    batch = len(anchors)
    previous = torch.as_tensor(
        np.stack(
            [development_states[positions[anchor - 1]] for anchor in anchors], axis=0
        ).astype(np.float32),
        device=device,
    )
    current = torch.as_tensor(
        np.stack(
            [development_states[positions[anchor]] for anchor in anchors], axis=0
        ).astype(np.float32),
        device=device,
    )
    geometry_batch = geometry.expand(batch, device)
    fourier = model.prepare_fourier_tensors(geometry_batch)  # type: ignore[attr-defined]
    active = torch.ones(batch, dtype=torch.bool, device=device)
    consecutive_finite = np.zeros(batch, dtype=np.int64)
    first_nonfinite: list[int | None] = [None] * batch
    predicted: dict[tuple[int, int], np.ndarray] = {}
    model.eval()
    with torch.no_grad():
        for horizon in range(1, 209):
            _, candidate = recurrent_step(
                model,
                previous,
                current,
                geometry_batch,
                normalization,
                fourier_tensors=fourier,
            )
            finite = torch.all(torch.isfinite(candidate), dim=(1, 2))
            newly_failed = active & ~finite
            for offset in (
                torch.nonzero(newly_failed, as_tuple=False).flatten().tolist()
            ):
                first_nonfinite[offset] = horizon
            retained = active & finite
            candidate_cpu = candidate.detach().cpu().numpy()
            for offset, anchor in enumerate(anchors):
                if bool(retained[offset]):
                    predicted[(anchor, horizon)] = np.asarray(
                        candidate_cpu[offset], dtype=np.float32
                    )
                    consecutive_finite[offset] += 1
            active = retained
            safe_candidate = torch.where(active[:, None, None], candidate, current)
            previous, current = current, safe_candidate

    airfoil, farfield, edges, edge_geometry = _airfoil_geometry(geometry)
    normals = np.zeros((geometry.num_nodes, 2), dtype=np.float64)
    centroid = np.mean(geometry.native_coordinates[airfoil], axis=0)
    for (first, second), record in zip(edges, edge_geometry, strict=True):
        normal = record[:2]
        length = record[2]
        midpoint = 0.5 * (
            geometry.native_coordinates[first] + geometry.native_coordinates[second]
        )
        if np.dot(normal, midpoint - centroid) < 0.0:
            raise ValueError("stored edge normal is not outward")
        normals[first] += normal * length
        normals[second] += normal * length
    normals[airfoil] /= np.linalg.norm(normals[airfoil], axis=1)[:, None]
    _, force_scale = _force_scales(train_states, edges, edge_geometry)
    period = float(contract["phase_population"]["measured_period_steps"])
    crossing = float(
        contract["phase_population"]["roles"]["train"][
            "source_crossing_output_coordinate"
        ]
    )

    rollout_rows: list[dict[str, Any]] = []
    structure_rows: list[dict[str, Any]] = []
    snapshots: dict[str, np.ndarray] = {}
    errors: dict[tuple[int, int, str, str], float] = {}
    relative_errors: dict[tuple[int, int, str, str], float] = {}
    for anchor_offset, anchor in enumerate(anchors):
        persistence = np.asarray(
            development_states[positions[anchor]], dtype=np.float64
        )
        for horizon in range(1, 209):
            target = np.asarray(
                development_states[positions[anchor + horizon]], dtype=np.float64
            )
            for method in ("pcno", "persistence"):
                state = (
                    predicted.get((anchor, horizon))
                    if method == "pcno"
                    else persistence
                )
                finite = state is not None and bool(np.all(np.isfinite(state)))
                for view in VIEWS:
                    if finite:
                        metrics = _field_metrics(
                            np.asarray(state, dtype=np.float64),
                            target,
                            weights[view],
                            normalization,
                        )
                        e_value = metrics["train_state_scale"]
                        l_value = metrics["relative_l2"]
                        row = {
                            "seed": seed,
                            "method": method,
                            "anchor": anchor,
                            "horizon": horizon,
                            "view": view,
                            "finite": True,
                            "train_state_scale_error": e_value,
                            "relative_l2": l_value,
                            **_metric_columns(
                                "train_state_scale_error",
                                metrics["train_state_scale_per_field"],
                            ),
                            **_metric_columns(
                                "relative_l2", metrics["relative_l2_per_field"]
                            ),
                        }
                        if horizon in HORIZONS:
                            template = _train_template(
                                np.asarray(state, dtype=np.float64),
                                train_states,
                                train_indices,
                                weights[view],
                                normalization,
                                period,
                                crossing,
                            )
                            template["time_matched_train_state_scale_error"] = e_value
                            template["time_minus_phase_aligned_error"] = (
                                e_value - template["nearest_train_state_scale_distance"]
                            )
                            row.update(template)
                        errors[(anchor, horizon, view, method)] = float(e_value)
                        relative_errors[(anchor, horizon, view, method)] = float(
                            l_value
                        )
                    else:
                        row = {
                            "seed": seed,
                            "method": method,
                            "anchor": anchor,
                            "horizon": horizon,
                            "view": view,
                            "finite": False,
                            "train_state_scale_error": math.inf,
                            "relative_l2": math.inf,
                        }
                        errors[(anchor, horizon, view, method)] = math.inf
                        relative_errors[(anchor, horizon, view, method)] = math.inf
                    rollout_rows.append(row)
                if finite:
                    structure_rows.append(
                        {
                            "seed": seed,
                            "method": method,
                            "anchor": anchor,
                            "horizon": horizon,
                            "finite": True,
                            **_structure_metrics(
                                np.asarray(state, dtype=np.float64),
                                target,
                                geometry,
                                normalization,
                                airfoil,
                                farfield,
                                normals,
                                edges,
                                edge_geometry,
                                force_scale,
                            ),
                        }
                    )
                else:
                    structure_rows.append(
                        {
                            "seed": seed,
                            "method": method,
                            "anchor": anchor,
                            "horizon": horizon,
                            "finite": False,
                        }
                    )
            if horizon in HORIZONS:
                reference_key = f"reference_anchor_{anchor}_h{horizon:03d}"
                snapshots.setdefault(reference_key, target)
                snapshots[f"pcno_seed_{seed}_anchor_{anchor}_h{horizon:03d}"] = (
                    np.asarray(predicted[(anchor, horizon)], dtype=np.float32)
                    if (anchor, horizon) in predicted
                    else np.full(target.shape, np.nan, dtype=np.float32)
                )

    methods: dict[str, Any] = {}
    for method in ("pcno", "persistence"):
        view_records: dict[str, Any] = {}
        for view in VIEWS:
            windows: dict[str, Any] = {}
            for label, (first, last) in (
                ("early_1_35", EARLY_WINDOW),
                ("late_174_208", LATE_WINDOW),
            ):
                windows[label] = {
                    "train_state_scale_error": _window_summary(
                        errors,
                        anchors=anchors,
                        method=method,
                        view=view,
                        first=first,
                        last=last,
                    ),
                    "component_balanced_relative_l2": _window_summary(
                        relative_errors,
                        anchors=anchors,
                        method=method,
                        view=view,
                        first=first,
                        last=last,
                    ),
                }
            view_records[view] = {
                "windows": windows,
                "terminal": _terminal_summary(
                    errors,
                    relative_errors,
                    anchors=anchors,
                    method=method,
                    view=view,
                ),
            }
        methods[method] = {"views": view_records}

    admissibility_first_occurrence = _admissibility_first_occurrence(
        structure_rows, anchors
    )

    completion = {
        str(horizon): {
            "finite_state_rate": float(
                np.sum(np.minimum(consecutive_finite, horizon)) / (batch * horizon)
            ),
            "full_rollout_completion_rate": float(
                np.mean(consecutive_finite >= horizon)
            ),
        }
        for horizon in HORIZONS
    }
    e1 = float(
        np.median([errors[(anchor, 1, "uniform_node", "pcno")] for anchor in anchors])
    )
    early_pcno = _numeric_json_number(
        methods["pcno"]["views"]["uniform_node"]["windows"]["early_1_35"][
            "train_state_scale_error"
        ]["all_anchor_summary_failed_windows_are_infinity"]["median"]
    )
    early_persistence = _numeric_json_number(
        methods["persistence"]["views"]["uniform_node"]["windows"]["early_1_35"][
            "train_state_scale_error"
        ]["all_anchor_summary_failed_windows_are_infinity"]["median"]
    )
    late_pcno = _numeric_json_number(
        methods["pcno"]["views"]["uniform_node"]["windows"]["late_174_208"][
            "train_state_scale_error"
        ]["all_anchor_summary_failed_windows_are_infinity"]["median"]
    )
    initially_accurate = (
        early_pcno < 0.15
        and early_pcno <= 0.80 * early_persistence
        and completion["35"]["full_rollout_completion_rate"] == 1.0
    )
    severe = (
        late_pcno >= max(0.25, 5.0 * e1)
        or completion["208"]["full_rollout_completion_rate"] < 0.80
    )
    summary = {
        "methods": methods,
        "finite_completion": completion,
        "first_nonfinite_step_by_anchor": {
            str(anchor): first_nonfinite[offset]
            for offset, anchor in enumerate(anchors)
        },
        "one_step_rollout_value_E1": e1,
        "admissibility_first_occurrence": admissibility_first_occurrence,
        "initially_accurate": initially_accurate,
        "severe_long_horizon": severe,
    }
    return rollout_rows, structure_rows, snapshots, summary


def _native_replay_margin(
    *,
    seed: int,
    model: torch.nn.Module,
    diagnostic_states: np.ndarray,
    geometry: VerifiedNACAGeometry,
    normalization: NACANormalization,
    weights: Mapping[str, np.ndarray],
    qualified_errors: Mapping[str, float],
    device: torch.device,
) -> dict[str, Any]:
    geometry_batch = geometry.expand(1, device)
    fourier = model.prepare_fourier_tensors(geometry_batch)  # type: ignore[attr-defined]
    previous = torch.as_tensor(
        np.array(diagnostic_states[0], dtype=np.float32, copy=True), device=device
    ).unsqueeze(0)
    current = torch.as_tensor(
        np.array(diagnostic_states[1], dtype=np.float32, copy=True), device=device
    ).unsqueeze(0)
    with torch.no_grad():
        _, prediction = recurrent_step(
            model,
            previous,
            current,
            geometry_batch,
            normalization,
            fourier_tensors=fourier,
        )
    predicted = prediction[0].cpu().numpy().astype(np.float64)
    target = np.asarray(diagnostic_states[2], dtype=np.float64)
    result: dict[str, Any] = {"seed": seed, "views": {}}
    for view in VIEWS:
        error = _field_metrics(predicted, target, weights[view], normalization)[
            "relative_l2"
        ]
        qualified = float(qualified_errors[view])
        result["views"][view] = {
            "pcno_component_balanced_relative_l2": error,
            "qualified_native_replay_component_balanced_relative_l2": qualified,
            "strict_quarter_margin_pass": qualified < 0.25 * error,
        }
    result["all_views_pass"] = all(
        record["strict_quarter_margin_pass"] for record in result["views"].values()
    )
    return result


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty metric table: {path.name}")
    columns: list[str] = []
    observed: set[str] = set()
    for row in rows:
        for key in row:
            if key not in observed:
                observed.add(key)
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _write_final_hash_manifest(output: Path) -> None:
    files = {
        path.name: _file_record(path, output)
        for path in sorted(output.iterdir(), key=lambda item: item.name)
        if path.name != "final_hash_manifest.json"
    }
    _write_json(
        output / "final_hash_manifest.json",
        {
            "schema": FINAL_HASH_MANIFEST_SCHEMA,
            "contract_sha256": BASELINE_CONTRACT_SHA256,
            "files": files,
            "self_hash_excluded": True,
            "prospective_opened": False,
            "sealed_opened": False,
        },
    )


def _write_staged_evaluation_packet(
    output: Path,
    writer: Callable[[Path], None],
    *,
    staging_suffix: str = ".staging",
) -> None:
    staging = output.with_name(f".{output.name}{staging_suffix}")
    if staging.exists() or staging.is_symlink():
        raise EvaluationOutputError("stale or aliased evaluation staging path exists")
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        staging.mkdir()
        writer(staging)
        os.replace(staging, output)
    except Exception as error:
        shutil.rmtree(staging, ignore_errors=True)
        if isinstance(error, EvaluationOutputError):
            raise
        raise EvaluationOutputError("evaluation output packet write failed") from error


def evaluate(arguments: argparse.Namespace) -> dict[str, Any]:
    if arguments.output_dir.is_symlink() or arguments.output_dir.parent.is_symlink():
        raise ValueError("evaluation output or its parent is aliased")
    output = arguments.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"evaluation output already exists: {output}")
    contract = load_naca_baseline_contract(arguments.contract)
    (
        dataset_manifest,
        geometry,
        normalization,
        roles,
        dataset_packet_sha256,
    ) = _load_dataset(arguments.dataset_dir, contract)
    _, qualification_sha256 = _qualification(arguments.replay_qualification, contract)
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if len(arguments.training_dir) != 3:
        raise ValueError("evaluation requires exactly three training directories")
    packets = [
        _verify_training_packet(
            path,
            dataset_manifest,
            dataset_packet_sha256,
            contract,
            geometry,
            device,
        )
        for path in arguments.training_dir
    ]
    seeds = [packet[0] for packet in packets]
    if sorted(seeds) != list(SEEDS) or len(set(seeds)) != 3:
        raise ValueError("training packets must contain exactly seeds 17, 29, and 43")
    packets.sort(key=lambda item: item[0])

    development_states, development_indices, development_dataset = roles["development"]
    train_states, train_indices, _ = roles["train"]
    centers = development_dataset.center_indices
    anchors = tuple(
        contract["phase_population"]["roles"]["development"]["anchor_input_indices"]
    )
    if len(anchors) != 8:
        raise ValueError("development role does not contain eight anchors")
    weights = _view_weights(geometry)

    diagnostic_record = dataset_manifest["diagnostic_sets"]["native_replay_margin"]
    diagnostic_indices = _load_npy_snapshot(
        arguments.dataset_dir,
        diagnostic_record["frame_indices"],
        "replay_margin_frame_indices.npy",
    )
    diagnostic_states = _load_npy_snapshot(
        arguments.dataset_dir,
        diagnostic_record["states"],
        "replay_margin_states.npy",
    )
    if (
        diagnostic_indices.dtype != np.int64
        or diagnostic_indices.tolist() != [497, 498, 499]
        or diagnostic_states.dtype != np.float64
        or diagnostic_states.shape != (3, geometry.num_nodes, 5)
    ):
        raise ValueError("diagnostic native-replay triplet differs")

    one_step_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    structure_rows: list[dict[str, Any]] = []
    snapshots: dict[str, np.ndarray] = {}
    seed_results: dict[str, Any] = {}
    qualified_errors = contract["evaluation"]["native_replay_margin"][
        "qualified_native_replay_errors"
    ]
    for seed, model, training_summary, _, _, _, _ in packets:
        seed_one_step_rows, one_step_summary = _one_step_metrics(
            seed=seed,
            model=model,
            states=development_states,
            frame_indices=development_indices,
            centers=centers,
            geometry=geometry,
            normalization=normalization,
            weights=weights,
            device=device,
        )
        seed_rollout_rows, seed_structure_rows, seed_snapshots, rollout_summary = (
            _rollout_seed(
                seed=seed,
                model=model,
                development_states=development_states,
                development_indices=development_indices,
                train_states=train_states,
                train_indices=train_indices,
                anchors=anchors,
                geometry=geometry,
                normalization=normalization,
                weights=weights,
                contract=contract,
                device=device,
            )
        )
        replay_margin = _native_replay_margin(
            seed=seed,
            model=model,
            diagnostic_states=diagnostic_states,
            geometry=geometry,
            normalization=normalization,
            weights=weights,
            qualified_errors=qualified_errors,
            device=device,
        )
        one_step_rows.extend(seed_one_step_rows)
        rollout_rows.extend(seed_rollout_rows)
        structure_rows.extend(seed_structure_rows)
        snapshots.update(seed_snapshots)
        seed_results[str(seed)] = {
            "selected_checkpoint": {
                "epoch": training_summary["best_epoch"],
                "development_selection_score": training_summary[
                    "best_development_score"
                ],
            },
            "one_step": one_step_summary,
            "rollout": rollout_summary,
            "native_replay_margin": replay_margin,
        }

    adequate_count = sum(
        result["one_step"]["adequately_trained"] for result in seed_results.values()
    )
    initial_and_severe_count = sum(
        result["rollout"]["initially_accurate"]
        and result["rollout"]["severe_long_horizon"]
        for result in seed_results.values()
    )
    margin_pass = all(
        result["native_replay_margin"]["all_views_pass"]
        for result in seed_results.values()
    )
    phenotype_pass = adequate_count == 3 and initial_and_severe_count >= 2
    decision = _r0_decision(
        adequate_count=adequate_count,
        phenotype_pass=phenotype_pass,
        margin_pass=margin_pass,
    )

    source = _load_self_hashed_snapshot(
        arguments.dataset_dir,
        dataset_manifest["source_manifest"]["file"],
        "source_manifest.json",
        SOURCE_MANIFEST_SCHEMA,
    )
    runtime_manifest = _self_hashed(
        {
            "schema": "time_dependent_no.su2_naca0012_pcno_runtime.v1",
            "runtime": _runtime(device),
        }
    )
    input_manifest = _self_hashed(
        {
            "schema": "time_dependent_no.su2_naca0012_pcno_evaluation_inputs.v1",
            "contract_sha256": BASELINE_CONTRACT_SHA256,
            "dataset_manifest_payload_sha256": dataset_manifest[
                "canonical_payload_sha256"
            ],
            "dataset_final_hash_manifest_sha256": dataset_packet_sha256,
            "source_set_sha256": dataset_manifest["source_manifest"][
                "source_set_sha256"
            ],
            "native_replay_qualification_file_sha256": qualification_sha256,
            "training_packets": [
                {
                    "seed": seed,
                    "directory_name": directory_name,
                    "final_hash_manifest_sha256": final_sha256,
                }
                for seed, _, _, final_sha256, directory_name, _, _ in packets
            ],
            "evaluated_population_role": "development",
            "rollout_horizon_steps": 208,
            "prospective_opened": False,
            "sealed_opened": False,
        }
    )
    result = _self_hashed(
        _json_safe(
            {
                "schema": EVALUATION_SCHEMA,
                "status": "complete",
                "classification": "SCIENTIFIC_RESULT",
                "contract_sha256": BASELINE_CONTRACT_SHA256,
                "dataset_final_hash_manifest_sha256": dataset_packet_sha256,
                "population_role": "development",
                "seeds": list(SEEDS),
                "seed_results": seed_results,
                "aggregate_decision": {
                    "adequately_trained_seed_count": adequate_count,
                    "initially_accurate_and_severe_seed_count": initial_and_severe_count,
                    "baseline_phenotype_pass": phenotype_pass,
                    "native_replay_margin_all_seeds_all_views_pass": margin_pass,
                    "decision": decision,
                    "post_outcome_horizon_extension_forbidden": True,
                },
                "aggregation": {
                    "anchors_are_phase_samples_not_independent_trajectories": True,
                    "deterministic_q90": "nearest_rank",
                    "failed_windows_assigned_positive_infinity": True,
                    "infinity_json_encoding": "Infinity",
                },
                "claim_boundary": {
                    "R0_development_baseline_evaluated": True,
                    "primary_PDE_promoted": False,
                    "corrective_mechanism_performance_claimed": False,
                    "prospective_opened": False,
                    "sealed_opened": False,
                },
            }
        )
    )

    _reverify_dataset_packet(arguments.dataset_dir, dataset_packet_sha256)
    _reverify_live_source(source)
    for _, _, _, final_sha256, _, training_root, files in packets:
        _reverify_packet_files(training_root, files, final_sha256)
    if (
        arguments.replay_qualification.is_symlink()
        or sha256(arguments.replay_qualification.read_bytes()).hexdigest()
        != qualification_sha256
    ):
        raise ValueError("native replay qualification changed during evaluation")

    def write_packet(staging: Path) -> None:
        _write_csv(staging / "one_step_metrics.csv", one_step_rows)
        _write_csv(staging / "rollout_metrics.csv", rollout_rows)
        _write_csv(staging / "rollout_structure.csv", structure_rows)
        np.savez_compressed(staging / "rollout_snapshots.npz", **snapshots)
        _write_json(staging / "source_manifest.json", source)
        _write_json(staging / "runtime_manifest.json", runtime_manifest)
        _write_json(staging / "input_manifest.json", input_manifest)
        _write_json(staging / "result.json", result)
        _write_final_hash_manifest(staging)

    _write_staged_evaluation_packet(output, write_packet)
    return result


def _failure_receipt(output: Path, classification: str, error: BaseException) -> None:
    if output.is_symlink() or output.parent.is_symlink():
        return
    output = output.resolve()
    if output.exists():
        return
    receipt = _self_hashed(
        {
            "schema": EVALUATION_SCHEMA,
            "status": "failed",
            "classification": classification,
            "contract_sha256": BASELINE_CONTRACT_SHA256,
            "error_type": type(error).__name__,
            "error": str(error),
            "R0_decision_made": False,
            "prospective_opened": False,
            "sealed_opened": False,
        }
    )

    def write_failure(staging: Path) -> None:
        _write_json(staging / "result.json", receipt)

    try:
        _write_staged_evaluation_packet(
            output, write_failure, staging_suffix=".failure-staging"
        )
    except EvaluationOutputError:
        return


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--replay-qualification", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = evaluate(arguments)
    except torch.cuda.OutOfMemoryError as error:
        _failure_receipt(arguments.output_dir, "INFRASTRUCTURE_FAILURE", error)
        print(
            f"NACA0012 PCNO evaluation infrastructure failure: {error}", file=sys.stderr
        )
        return 3
    except (OSError, TypeError, ValueError) as error:
        _failure_receipt(arguments.output_dir, "INVALID_ARTIFACT", error)
        print(f"NACA0012 PCNO evaluation invalid artifact: {error}", file=sys.stderr)
        return 2
    except RuntimeError as error:
        _failure_receipt(arguments.output_dir, "INFRASTRUCTURE_FAILURE", error)
        print(
            f"NACA0012 PCNO evaluation infrastructure failure: {error}", file=sys.stderr
        )
        return 3
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
