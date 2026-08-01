"""Resolution-translation utilities for the dynamic shock--vortex PCNO.

The functions in this module keep the physical problem fixed while rebuilding
its sampled state, finite-volume quadrature, node types, graph connectivity, and
least-squares differential weights on each requested Cartesian model grid.
They deliberately do not evolve a native-grid CFD solver.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from utility.time_dependent_no.pcno_artifacts import sha256_file
from utility.time_dependent_no.pcno_euler2d import (
    Euler2DNormalization,
    PCNOEuler2DResidual,
    PCNOEuler2DShardStore,
)
from utility.time_dependent_no.pcno_fv_geometry import (
    PCNOFiniteVolumeGeometry,
    build_pcno_finite_volume_geometry,
)
from utility.time_dependent_no.pcno_ripple_diagnostics import (
    conservative_to_primitive_raw,
)
from utility.time_dependent_no.pcno_runtime import (
    CHECKPOINT_SCHEMA_VERSION,
    build_checkpoint_model,
    checkpoint_model_node_type_input,
    load_checkpoint_payload,
    timed_model_call,
)
from utility.time_dependent_no.shock_vortex_coarse_cfd import (
    restrict_uniform_cell_averages,
)
from utility.time_dependent_no.shock_vortex_family import (
    REFERENCE_ARTIFACT_SCHEMA,
    config_for_family_case,
    family_case_provenance,
)
from utility.time_dependent_no.shock_vortex_fv import (
    BOUNDARY_TAG_NAMES,
    ShockVortexFVConfig,
    make_structured_fv_geometry,
    shock_vortex_initial_cell_averages,
)

Resolution = tuple[int, int]
MULTIRES_REFERENCE_SCHEMA = "shock_vortex_multires_restriction_reference_v1"
RESTRICTION_CROSSCHECK_ABS_TOLERANCE = 1.0e-12
PHYSICAL_WAVELENGTH_BANDS = ((0.05, 0.125), (0.125, 0.25))
SHOCK_WINDOW_HALF_WIDTH = 0.2
NodeTypeProtocol = Literal[
    "physical",
    "all_normal",
    "swapped_boundary_kinds",
    "training_band",
]
NODE_TYPE_PROTOCOLS: tuple[NodeTypeProtocol, ...] = (
    "physical",
    "all_normal",
    "swapped_boundary_kinds",
    "training_band",
)


def load_resolution_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = load_checkpoint_payload(path)
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
        raise ValueError("resolution transfer freezes model_all_nodes recurrence")
    if checkpoint["raw_recurrence"] is not True:
        raise ValueError("checkpoint does not declare raw recurrence")
    interventions = checkpoint["inference_interventions"]
    if not isinstance(interventions, Mapping):
        raise TypeError("checkpoint inference_interventions must be a mapping")
    if any(bool(value) for value in interventions.values()):
        raise ValueError("resolution transfer forbids checkpoint-time interventions")
    checkpoint_model_node_type_input(checkpoint)
    return dict(checkpoint)


def build_resolution_checkpoint_model(
    checkpoint: Mapping[str, Any],
    device: torch.device,
) -> tuple[PCNOEuler2DResidual, Euler2DNormalization]:
    return build_checkpoint_model(
        checkpoint,
        device,
        model_node_type_input=checkpoint_model_node_type_input(checkpoint),
    )


def conservative_admissibility_summary(
    state: np.ndarray,
    *,
    gamma: float,
) -> dict[str, Any]:
    values = np.asarray(state, dtype=np.float64)
    rho = values[:, 0]
    momentum_square = np.square(values[:, 1]) + np.square(values[:, 2])
    with np.errstate(divide="ignore", invalid="ignore"):
        internal_energy = values[:, 3] - 0.5 * momentum_square / rho
        pressure = (gamma - 1.0) * internal_energy
    finite = (
        np.isfinite(values).all(axis=-1)
        & np.isfinite(internal_energy)
        & np.isfinite(pressure)
    )
    admissible = finite & (rho > 0.0) & (internal_energy > 0.0) & (pressure > 0.0)
    return {
        "finite": bool(finite.all()),
        "admissible": bool(admissible.all()),
        "admissible_fraction": float(admissible.mean()),
        "minimum_density": float(np.nanmin(rho)),
        "minimum_internal_energy": float(np.nanmin(internal_energy)),
        "minimum_pressure": float(np.nanmin(pressure)),
    }


def checkpoint_step_stride(checkpoint: Mapping[str, Any]) -> int:
    declared = {
        int(value)
        for value in (
            checkpoint.get("step_stride"),
            checkpoint.get("training_args", {}).get("step_stride"),
            checkpoint.get("data_contract", {}).get("step_stride"),
        )
        if value is not None
    }
    if len(declared) != 1:
        raise ValueError(f"checkpoint step-stride declarations disagree: {declared}")
    stride = declared.pop()
    if stride < 1:
        raise ValueError("checkpoint step stride must be positive")
    return stride


def validate_resolution_rollout_contract(
    args: argparse.Namespace,
    checkpoint: Mapping[str, Any],
    manifest: Mapping[str, Any],
    store: PCNOEuler2DShardStore,
    resolutions: Sequence[Resolution],
    source_resolution: Resolution,
    training_resolution: Resolution,
) -> tuple[int, int, float, set[int]]:
    checkpoint_sha = sha256_file(args.checkpoint)
    if (
        args.expected_checkpoint_sha256 is not None
        and checkpoint_sha.lower() != args.expected_checkpoint_sha256.lower()
    ):
        raise ValueError("checkpoint SHA-256 mismatch")
    if (
        args.expected_normalization_digest is not None
        and checkpoint["normalization_digest"] != args.expected_normalization_digest
    ):
        raise ValueError("checkpoint normalization digest mismatch")
    if store.manifest_digest != checkpoint["data_manifest_digest"]:
        raise ValueError("checkpoint and retained data manifest digests differ")

    data_contract = checkpoint["data_contract"]
    if data_contract.get("source_family_id") != manifest["family_id"]:
        raise ValueError("checkpoint and family identifiers differ")
    if (
        data_contract.get("source_family_manifest_digest")
        != manifest["manifest_digest_sha256"]
    ):
        raise ValueError("checkpoint and family manifest digests differ")
    model_config = checkpoint["model_config"]
    if int(model_config["k_max"]) != args.expected_k_max:
        raise ValueError("checkpoint k_max differs from the frozen bandwidth")
    if tuple(float(v) for v in model_config["domain_lengths"]) != tuple(
        float(v) for v in args.expected_domain_lengths
    ):
        raise ValueError("checkpoint Fourier periods differ from the physical domain")
    if int(model_config["nmeasures"]) != 1:
        raise ValueError("maintained Euler wrapper must use exactly one measure")
    model_node_type_input = checkpoint_model_node_type_input(checkpoint)
    requested_protocols = list(dict.fromkeys(args.protocols))
    if model_node_type_input == "all_normal" and requested_protocols != ["all_normal"]:
        raise ValueError(
            "an all-normal-trained checkpoint must be evaluated with only the "
            "all_normal model-input protocol"
        )

    expected_source = tuple(
        int(v) for v in manifest["reference_fidelity"]["evolution_grid"]
    )
    expected_training = tuple(
        int(v) for v in manifest["reference_fidelity"]["stored_model_grid"]
    )
    if source_resolution != expected_source or training_resolution != expected_training:
        raise ValueError("source/training grids differ from the frozen family contract")
    if len(resolutions) < 2 or len(set(resolutions)) != len(resolutions):
        raise ValueError("at least two unique target resolutions are required")
    if training_resolution not in resolutions:
        raise ValueError("evaluation resolutions must include the native training grid")
    ordered = sorted(resolutions, key=lambda value: value[0] * value[1])
    for resolution in ordered:
        if source_resolution[0] % resolution[0] or source_resolution[1] % resolution[1]:
            raise ValueError("every target grid must divide the common source grid")
    for coarse, fine in pairwise(ordered):
        if fine[0] % coarse[0] or fine[1] % coarse[1]:
            raise ValueError("successive target grids must be nested")

    stride = checkpoint_step_stride(checkpoint)
    if args.expected_step_stride is not None and stride != args.expected_step_stride:
        raise ValueError("checkpoint step stride differs from the expected stride")
    base_dt = float(data_contract["time_contract"]["delta_t"])
    saved_calls = int(data_contract["time_contract"]["saved_calls"])
    maximum_calls = saved_calls // stride
    rollout_calls = maximum_calls if args.rollout_calls is None else args.rollout_calls
    if rollout_calls < 1 or rollout_calls > maximum_calls:
        raise ValueError("rollout calls exceed the checkpoint-bound reference horizon")
    physical_dt = stride * base_dt
    endpoint_calls: set[int] = set()
    for physical_time in args.endpoint_physical_times:
        call = round(float(physical_time) / physical_dt)
        if (
            call < 1
            or call > rollout_calls
            or not math.isclose(
                call * physical_dt, float(physical_time), rel_tol=0.0, abs_tol=1.0e-12
            )
        ):
            raise ValueError(
                "endpoint times must align with model calls inside the horizon"
            )
        endpoint_calls.add(call)
    if not 0.0 < args.shock_quantile < 1.0:
        raise ValueError("shock quantile must lie strictly between zero and one")
    if args.repeat_forward < 1:
        raise ValueError("repeat-forward must be positive")
    widths = np.asarray(args.boundary_band_widths, dtype=np.float64)
    if np.any(~np.isfinite(widths)) or np.any(widths <= 0.0):
        raise ValueError("boundary-band widths must be positive and finite")
    return stride, rollout_calls, physical_dt, endpoint_calls


def _array_comparison(rebuilt: np.ndarray, stored: np.ndarray) -> dict[str, Any]:
    rebuilt_array = np.asarray(rebuilt)
    stored_array = np.asarray(stored)
    if rebuilt_array.shape != stored_array.shape:
        raise ValueError("rebuilt and stored arrays have different shapes")
    cast = rebuilt_array.astype(stored_array.dtype, copy=False)
    integer = np.issubdtype(stored_array.dtype, np.integer)
    return {
        "shape": list(stored_array.shape),
        "stored_dtype": str(stored_array.dtype),
        "exact_after_stored_dtype_cast": bool(np.array_equal(cast, stored_array)),
        "maximum_absolute_difference": (
            None
            if integer
            else float(
                np.max(
                    np.abs(
                        rebuilt_array.astype(np.float64)
                        - stored_array.astype(np.float64)
                    )
                )
            )
        ),
    }


def native_geometry_audit(
    geometry: Any,
    store: PCNOEuler2DShardStore,
    case_id: str,
) -> dict[str, Any]:
    stored = store.geometry_numpy(case_id)
    rows = {}
    for name in (
        "nodes",
        "node_measures",
        "node_weights",
        "node_rhos",
        "directed_edges",
        "edge_gradient_weights",
        "node_type",
    ):
        rows[name] = _array_comparison(getattr(geometry, name), stored[name])
    failed = [
        name for name, row in rows.items() if not row["exact_after_stored_dtype_cast"]
    ]
    if failed:
        raise ValueError(f"regenerated native geometry differs from shards: {failed}")
    return rows


def load_resolution_reference(
    family_root: Path,
    multires_reference_root: Path | None,
    store: PCNOEuler2DShardStore,
    manifest: Mapping[str, Any],
    case_id: str,
    *,
    training_resolution: Resolution,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and bind frozen 250 truth, then optionally promote finer truth."""

    path = family_root / case_id / "reference.npz"
    expected_sha = store.entry(case_id).get("source_reference_sha256")
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise ValueError(f"reference digest mismatch for {case_id}")
    expected_provenance = family_case_provenance(manifest, case_id)
    names = (
        "schema",
        "family_contract_json",
        "conservative_states",
        "physical_times",
        "cell_centers",
        "cell_volume",
        "interval_boundary_exchange",
    )
    with np.load(path, allow_pickle=False) as artifact:
        missing = sorted(set(names) - set(artifact.files))
        if missing:
            raise ValueError(f"reference {case_id} is missing arrays: {missing}")
        if artifact["schema"].item() != REFERENCE_ARTIFACT_SCHEMA:
            raise ValueError(f"reference schema mismatch for {case_id}")
        provenance = json.loads(artifact["family_contract_json"].item())
        if provenance != expected_provenance:
            raise ValueError(f"reference provenance mismatch for {case_id}")
        reference = {
            name: np.array(artifact[name], copy=True)
            for name in names
            if name not in {"schema", "family_contract_json"}
        }

    expected_nodes = training_resolution[0] * training_resolution[1]
    states = reference["conservative_states"]
    if states.ndim != 3 or states.shape[1:] != (expected_nodes, 4):
        raise ValueError(f"reference state shape mismatch for {case_id}")
    if reference["physical_times"].shape != (states.shape[0],):
        raise ValueError(f"reference time shape mismatch for {case_id}")
    if reference["cell_centers"].shape != (expected_nodes, 2):
        raise ValueError(f"reference node shape mismatch for {case_id}")
    if reference["cell_volume"].shape != (expected_nodes,):
        raise ValueError(f"reference volume shape mismatch for {case_id}")
    if reference["interval_boundary_exchange"].shape != (states.shape[0] - 1, 4):
        raise ValueError(f"reference boundary-exchange shape mismatch for {case_id}")
    shard_states = np.asarray(store.states(case_id), dtype=np.float64)
    if not np.allclose(shard_states, states, rtol=1.0e-6, atol=1.0e-7):
        raise ValueError(f"reference and checkpoint-bound shards differ for {case_id}")
    reference["retained_resolution"] = training_resolution
    check: dict[str, Any] = {
        "case_id": case_id,
        "frozen_training_reference_sha256": actual_sha,
        "active_reference_artifact_sha256": actual_sha,
        "retained_resolution": resolution_label(training_resolution),
        "state_dtype": str(states.dtype),
        "state_shape": list(states.shape),
    }
    if multires_reference_root is None:
        check["restriction_crosscheck_max_abs"] = 0.0
        return reference, check

    multires_case_root = multires_reference_root / case_id
    multires_path = multires_case_root / "reference.npz"
    multires_summary_path = multires_case_root / "summary.json"
    multires_summary = json.loads(multires_summary_path.read_text(encoding="utf-8"))
    multires_sha = sha256_file(multires_path)
    if multires_summary.get("status") != "passed":
        raise ValueError(f"multires reference did not pass for {case_id}")
    if multires_summary.get("reference_artifact_sha256") != multires_sha:
        raise ValueError(f"multires reference digest mismatch for {case_id}")
    names = (
        "schema",
        "family_contract_json",
        "config_json",
        "conservative_states",
        "physical_times",
        "interval_boundary_exchange",
        "source_resolution",
        "retained_resolution",
        "training_resolution",
        "frozen_training_reference_sha256",
    )
    with np.load(multires_path, allow_pickle=False) as artifact:
        missing = sorted(set(names) - set(artifact.files))
        if missing:
            raise ValueError(f"multires reference is missing arrays: {missing}")
        if artifact["schema"].item() != MULTIRES_REFERENCE_SCHEMA:
            raise ValueError(f"multires reference schema mismatch for {case_id}")
        if json.loads(artifact["family_contract_json"].item()) != expected_provenance:
            raise ValueError(f"multires reference provenance mismatch for {case_id}")
        retained_resolution = tuple(
            int(value) for value in artifact["retained_resolution"].tolist()
        )
        source_resolution = tuple(
            int(value) for value in artifact["source_resolution"].tolist()
        )
        artifact_training_resolution = tuple(
            int(value) for value in artifact["training_resolution"].tolist()
        )
        expected_source_resolution = tuple(
            int(value) for value in manifest["reference_fidelity"]["evolution_grid"]
        )
        if source_resolution != expected_source_resolution:
            raise ValueError(f"multires evolution grid mismatch for {case_id}")
        if artifact_training_resolution != training_resolution:
            raise ValueError(f"multires training grid mismatch for {case_id}")
        if artifact["frozen_training_reference_sha256"].item() != actual_sha:
            raise ValueError(
                f"multires source-reference binding mismatch for {case_id}"
            )
        expected_config = config_for_family_case(manifest, case_id).to_dict()
        expected_config["coarse_nx"] = retained_resolution[0]
        expected_config["coarse_ny"] = retained_resolution[1]
        expected_config["restriction_x"] = (
            expected_config["nx"] // retained_resolution[0]
        )
        expected_config["restriction_y"] = (
            expected_config["ny"] // retained_resolution[1]
        )
        if json.loads(artifact["config_json"].item()) != expected_config:
            raise ValueError(f"multires solver configuration mismatch for {case_id}")
        multires_reference = {
            "conservative_states": np.array(artifact["conservative_states"], copy=True),
            "physical_times": np.array(artifact["physical_times"], copy=True),
            "interval_boundary_exchange": np.array(
                artifact["interval_boundary_exchange"], copy=True
            ),
            "retained_resolution": retained_resolution,
        }

    multires_states = multires_reference["conservative_states"]
    expected_retained_nodes = retained_resolution[0] * retained_resolution[1]
    if multires_states.shape != (states.shape[0], expected_retained_nodes, 4):
        raise ValueError(f"multires state shape mismatch for {case_id}")
    if not np.array_equal(
        multires_reference["physical_times"], reference["physical_times"]
    ):
        raise ValueError(f"multires physical times mismatch for {case_id}")
    if (
        multires_reference["interval_boundary_exchange"].shape
        != reference["interval_boundary_exchange"].shape
    ):
        raise ValueError(f"multires boundary-exchange shape mismatch for {case_id}")
    restricted = restrict_nested_state(
        multires_states,
        fine_resolution=retained_resolution,
        coarse_resolution=training_resolution,
    )
    restriction_crosscheck_max_abs = float(np.max(np.abs(restricted - states)))
    if restriction_crosscheck_max_abs > RESTRICTION_CROSSCHECK_ABS_TOLERANCE:
        raise ValueError(
            f"multires reference does not reproduce frozen training truth for {case_id}: "
            f"{restriction_crosscheck_max_abs:.6e}"
        )
    check.update(
        {
            "active_reference_artifact_sha256": multires_sha,
            "retained_resolution": resolution_label(retained_resolution),
            "state_dtype": str(multires_states.dtype),
            "state_shape": list(multires_states.shape),
            "restriction_crosscheck_max_abs": restriction_crosscheck_max_abs,
        }
    )
    return multires_reference, check


def reference_at_resolution(
    reference_state: np.ndarray,
    *,
    reference_resolution: Resolution,
    target_resolution: Resolution,
) -> np.ndarray | None:
    if target_resolution == reference_resolution:
        return np.asarray(reference_state, dtype=np.float64)
    if (
        reference_resolution[0] % target_resolution[0]
        or reference_resolution[1] % target_resolution[1]
    ):
        return None
    return restrict_nested_state(
        reference_state,
        fine_resolution=reference_resolution,
        coarse_resolution=target_resolution,
    )


def as_model_state(state: np.ndarray) -> np.ndarray:
    """Round explicitly to the float32 state seen by the checkpoint."""

    return np.asarray(state, dtype=np.float32).astype(np.float64)


def predict_resolution_sample(
    model: PCNOEuler2DResidual,
    sample: Mapping[str, torch.Tensor],
    state: np.ndarray,
    *,
    device: torch.device,
    amp: str,
    repeats: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    current = torch.as_tensor(
        np.asarray(state, dtype=np.float32), dtype=torch.float32, device=device
    ).unsqueeze(0)
    prediction, timing = timed_model_call(
        model,
        sample,
        current,
        device=device,
        amp=amp,
        repeats=repeats,
    )
    del current
    return as_model_state(prediction), timing


def pressure_profile_shock_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    resolution: Resolution,
    x_min: float,
    x_max: float,
    gamma: float,
    shock_center_x: float | None = None,
    shock_window_half_width: float = SHOCK_WINDOW_HALF_WIDTH,
    relative_threshold: float = 0.25,
) -> dict[str, float | int | str | None]:
    """Return a local y-averaged pressure-jump shock proxy in physical units."""

    nx, ny = resolution
    dx = (x_max - x_min) / nx
    center_x = (
        0.5 * (x_min + x_max) if shock_center_x is None else float(shock_center_x)
    )
    if (
        not x_min < center_x < x_max
        or not math.isfinite(shock_window_half_width)
        or shock_window_half_width <= 0.0
    ):
        raise ValueError("invalid physical shock window")

    def summarize(state: np.ndarray) -> dict[str, float | int | None]:
        primitive = conservative_to_primitive_raw(state, gamma=gamma).reshape(ny, nx, 4)
        profile = np.mean(primitive[..., 3], axis=0)
        jump = np.abs(np.diff(profile))
        face_x = x_min + dx * np.arange(1, nx, dtype=np.float64)
        eligible = np.flatnonzero(np.abs(face_x - center_x) <= shock_window_half_width)
        if eligible.size == 0:
            raise ValueError("shock window contains no grid faces")
        peak = int(eligible[np.argmax(jump[eligible])])
        maximum = float(jump[peak])
        if not math.isfinite(maximum) or maximum <= 0.0:
            return {
                "position": None,
                "thickness_cells": 0,
                "thickness_physical": 0.0,
                "strength": maximum,
            }
        threshold = relative_threshold * maximum
        eligible_start = int(eligible[0])
        eligible_end = int(eligible[-1])
        left = peak
        right = peak
        while left > eligible_start and jump[left - 1] >= threshold:
            left -= 1
        while right < eligible_end and jump[right + 1] >= threshold:
            right += 1
        active = np.arange(left, right + 1, dtype=np.int64)
        position = float(np.dot(jump[active], face_x[active]) / np.sum(jump[active]))
        thickness_cells = int(active.size)
        return {
            "position": position,
            "thickness_cells": thickness_cells,
            "thickness_physical": float(thickness_cells * dx),
            "strength": maximum,
        }

    predicted = summarize(prediction)
    reference = summarize(target)
    position_error = (
        abs(float(predicted["position"]) - float(reference["position"]))
        if predicted["position"] is not None and reference["position"] is not None
        else None
    )
    thickness_ratio = (
        float(predicted["thickness_physical"]) / float(reference["thickness_physical"])
        if float(reference["thickness_physical"]) > 0.0
        else None
    )
    strength_ratio = (
        float(predicted["strength"]) / float(reference["strength"])
        if float(reference["strength"]) > 0.0
        else None
    )
    return {
        "shock_profile_contract": (
            "y-mean pressure adjacent-x-jump; contiguous peak component; "
            "threshold_0.25; fixed physical window"
        ),
        "shock_window_center_x": center_x,
        "shock_window_half_width": float(shock_window_half_width),
        "prediction_shock_position_x": predicted["position"],
        "reference_shock_position_x": reference["position"],
        "shock_position_absolute_error": position_error,
        "prediction_shock_thickness_cells": predicted["thickness_cells"],
        "reference_shock_thickness_cells": reference["thickness_cells"],
        "prediction_shock_thickness_physical": predicted["thickness_physical"],
        "reference_shock_thickness_physical": reference["thickness_physical"],
        "pressure_profile_shock_thickness_ratio": thickness_ratio,
        "pressure_profile_shock_strength_ratio": strength_ratio,
    }


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [
        float(row[key])
        for row in rows
        if row.get(key) is not None and math.isfinite(float(row[key]))
    ]
    return float(np.mean(values)) if values else None


def commutator_row(
    *,
    case_id: str,
    case_split: str,
    protocol: str,
    kind: str,
    coarse: Resolution,
    fine: Resolution,
    call: int,
    input_frame: int,
    output_frame: int,
    input_physical_time: float,
    output_physical_time: float,
    reference_gap: float | None,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    prediction_gap = float(metrics["prediction_commutator_scaled_rms"])
    return {
        "case_id": case_id,
        "case_split": case_split,
        "node_type_protocol": protocol,
        "commutator_kind": kind,
        "coarse_resolution": resolution_label(coarse),
        "fine_resolution": resolution_label(fine),
        "call": call,
        "input_call": call - 1,
        "input_frame": input_frame,
        "output_frame": output_frame,
        "input_physical_time": input_physical_time,
        "physical_time": output_physical_time,
        "reference_discretization_gap_scaled_rms": reference_gap,
        "model_inconsistency_excess_scaled_rms": (
            max(prediction_gap - reference_gap, 0.0)
            if reference_gap is not None
            else None
        ),
        **metrics,
    }


def aggregate_resolution_rollout(
    state_rows: Sequence[Mapping[str, Any]],
    commutator_rows: Sequence[Mapping[str, Any]],
    completion_rows: Sequence[Mapping[str, Any]],
    *,
    teacher_state_rows: Sequence[Mapping[str, Any]] = (),
    final_call: int,
) -> dict[str, Any]:
    final_states = [row for row in state_rows if int(row["call"]) == final_call]
    state_groups = sorted(
        {
            (
                str(row["node_type_protocol"]),
                str(row["resolution"]),
                str(row["case_split"]),
            )
            for row in completion_rows
        }
    )
    states = []
    for protocol, resolution, split in state_groups:
        selected_states = [
            row
            for row in final_states
            if row["node_type_protocol"] == protocol
            and row["resolution"] == resolution
            and row["case_split"] == split
        ]
        selected_completion = [
            row
            for row in completion_rows
            if row["node_type_protocol"] == protocol
            and row["resolution"] == resolution
            and row["case_split"] == split
        ]
        states.append(
            {
                "node_type_protocol": protocol,
                "resolution": resolution,
                "case_split": split,
                "case_count": len(selected_completion),
                "final_state_case_count": len(selected_states),
                "completion_fraction": float(
                    np.mean(
                        [bool(row["full_completion"]) for row in selected_completion]
                    )
                ),
                "admissible_fraction_of_cases": float(
                    np.mean(
                        [
                            bool(row["all_completed_calls_admissible"])
                            for row in selected_completion
                        ]
                    )
                ),
                "mean_scaled_relative_l2_physical_volume": _mean(
                    selected_states, "scaled_relative_l2_physical_volume"
                ),
            }
        )

    final_teacher_states = [
        row for row in teacher_state_rows if int(row["call"]) == final_call
    ]
    teacher_groups = sorted(
        {
            (
                str(row["node_type_protocol"]),
                str(row["resolution"]),
                str(row["case_split"]),
            )
            for row in final_teacher_states
        }
    )
    teacher_states = []
    for protocol, resolution, split in teacher_groups:
        selected = [
            row
            for row in final_teacher_states
            if row["node_type_protocol"] == protocol
            and row["resolution"] == resolution
            and row["case_split"] == split
        ]
        teacher_states.append(
            {
                "node_type_protocol": protocol,
                "resolution": resolution,
                "case_split": split,
                "case_count": len(selected),
                "finite_fraction": float(
                    np.mean([bool(row["finite"]) for row in selected])
                ),
                "admissible_fraction": float(
                    np.mean([bool(row["admissible"]) for row in selected])
                ),
                "mean_scaled_relative_l2_physical_volume": _mean(
                    selected, "scaled_relative_l2_physical_volume"
                ),
            }
        )

    final_commutators = [
        row for row in commutator_rows if int(row["call"]) == final_call
    ]
    commutator_groups = sorted(
        {
            (
                str(row["node_type_protocol"]),
                str(row["commutator_kind"]),
                str(row["coarse_resolution"]),
                str(row["fine_resolution"]),
                str(row["case_split"]),
            )
            for row in final_commutators
        }
    )
    commutators = []
    for protocol, kind, coarse, fine, split in commutator_groups:
        selected = [
            row
            for row in final_commutators
            if row["node_type_protocol"] == protocol
            and row["commutator_kind"] == kind
            and row["coarse_resolution"] == coarse
            and row["fine_resolution"] == fine
            and row["case_split"] == split
        ]
        commutators.append(
            {
                "node_type_protocol": protocol,
                "commutator_kind": kind,
                "coarse_resolution": coarse,
                "fine_resolution": fine,
                "case_split": split,
                "case_count": len(selected),
                "mean_prediction_commutator_relative_l2": _mean(
                    selected, "prediction_commutator_relative_l2"
                ),
                "mean_update_commutator_relative_to_fine_update": _mean(
                    selected, "update_commutator_relative_to_fine_update"
                ),
            }
        )
    return {
        "final_state": states,
        "final_teacher_forced_state": teacher_states,
        "final_commutator": commutators,
    }


def parse_resolution(value: str) -> Resolution:
    """Parse an ``NXxNY`` model-grid declaration."""

    pieces = str(value).lower().split("x")
    if len(pieces) != 2:
        raise ValueError(f"resolution must have form NXxNY, got {value!r}")
    try:
        nx, ny = (int(piece) for piece in pieces)
    except ValueError as exc:
        raise ValueError(f"resolution must have form NXxNY, got {value!r}") from exc
    if nx < 2 or ny < 2:
        raise ValueError("both resolution dimensions must be at least two")
    return nx, ny


def resolution_label(resolution: Resolution) -> str:
    nx, ny = resolution
    return f"{int(nx)}x{int(ny)}"


def config_for_model_grid(
    base: ShockVortexFVConfig,
    resolution: Resolution,
) -> ShockVortexFVConfig:
    """Return the same physical case sampled directly on one model grid."""

    nx, ny = resolution
    return replace(
        base,
        nx=int(nx),
        ny=int(ny),
        coarse_nx=int(nx),
        coarse_ny=int(ny),
    ).validated()


def build_resolution_geometry(
    base: ShockVortexFVConfig,
    resolution: Resolution,
) -> tuple[ShockVortexFVConfig, PCNOFiniteVolumeGeometry]:
    """Regenerate FV quadrature, graph edges, types, and gradient weights."""

    config = config_for_model_grid(base, resolution)
    finite_volume = make_structured_fv_geometry(config)
    geometry = build_pcno_finite_volume_geometry(
        cell_centers=finite_volume.cell_centers,
        cell_volume=finite_volume.cell_volume,
        face_owner=finite_volume.face_owner,
        face_neighbor=finite_volume.face_neighbor,
        face_boundary_tag=finite_volume.face_boundary_tag,
        boundary_tag_names=BOUNDARY_TAG_NAMES,
    )
    return config, geometry


def initial_state_for_model_grid(
    base: ShockVortexFVConfig,
    resolution: Resolution,
    *,
    dtype: torch.dtype = torch.float64,
    quadrature_order: int | None = None,
) -> np.ndarray:
    """Sample the physical initial condition as conservative cell averages."""

    config = config_for_model_grid(base, resolution)
    state = shock_vortex_initial_cell_averages(
        config,
        device="cpu",
        dtype=dtype,
        quadrature_order=quadrature_order,
    )
    return state.detach().cpu().numpy().reshape(config.nx * config.ny, 4)


def initial_states_from_common_source(
    base: ShockVortexFVConfig,
    resolutions: Sequence[Resolution],
    *,
    source_resolution: Resolution,
    dtype: torch.dtype = torch.float64,
    quadrature_order: int | None = None,
) -> tuple[np.ndarray, dict[Resolution, np.ndarray]]:
    """Sample once on a common fine grid, then conservatively restrict.

    This is the finite-volume sampling contract for a restriction-consistent
    physical map.  Every returned component is a cell average of a conserved
    variable; primitive variables are never interpolated or restricted.
    """

    source_nx, source_ny = source_resolution
    requested = list(resolutions)
    if not requested:
        raise ValueError("at least one target resolution is required")
    if len(set(requested)) != len(requested):
        raise ValueError("target resolutions must be unique")
    for target_nx, target_ny in requested:
        if source_nx % target_nx or source_ny % target_ny:
            raise ValueError(
                "every target grid must divide the common source grid exactly"
            )

    source = initial_state_for_model_grid(
        base,
        source_resolution,
        dtype=dtype,
        quadrature_order=quadrature_order,
    )
    states = {}
    for resolution in requested:
        if resolution == source_resolution:
            states[resolution] = np.array(source, copy=True)
        else:
            states[resolution] = restrict_nested_state(
                source,
                fine_resolution=source_resolution,
                coarse_resolution=resolution,
            )
    return source, states


def node_types_for_protocol(
    geometry: PCNOFiniteVolumeGeometry,
    config: ShockVortexFVConfig,
    protocol: NodeTypeProtocol,
    *,
    training_resolution: Resolution,
) -> np.ndarray:
    """Return dynamic-family node types under one declared intervention.

    ``physical`` regenerates the one-cell boundary-touch descriptor on the
    target mesh. ``all_normal`` removes the descriptor.
    ``swapped_boundary_kinds`` preserves the tagged set but exchanges the
    y-symmetry-only and x-extrapolation-only categories (codes 1 and 2);
    corner code 3 is unchanged. ``training_band`` is a binary
    cell-intersection diagnostic that approximates a boundary strip one
    training cell wide. It is exact only when the target mesh resolves that
    width and is not a replacement boundary policy.
    """

    if protocol not in NODE_TYPE_PROTOCOLS:
        raise ValueError(f"unsupported node-type protocol: {protocol}")
    if geometry.nodes.shape != (config.nx * config.ny, 2):
        raise ValueError("geometry and model-grid configuration disagree")
    if protocol == "physical":
        return np.array(geometry.node_type, dtype=np.int64, copy=True)
    if protocol == "all_normal":
        return np.zeros(config.nx * config.ny, dtype=np.int64)
    if protocol == "swapped_boundary_kinds":
        physical = np.asarray(geometry.node_type, dtype=np.int64)
        swapped = np.array(physical, copy=True)
        swapped[physical == 1] = 2
        swapped[physical == 2] = 1
        return swapped

    training_nx, training_ny = training_resolution
    if training_nx < 1 or training_ny < 1:
        raise ValueError("training resolution must be positive")
    fixed_x_width = (config.x_max - config.x_min) / training_nx
    fixed_y_width = (config.y_max - config.y_min) / training_ny
    target_dx = (config.x_max - config.x_min) / config.nx
    target_dy = (config.y_max - config.y_min) / config.ny
    x = geometry.nodes[:, 0]
    y = geometry.nodes[:, 1]
    left_edge = x - 0.5 * target_dx
    right_edge = x + 0.5 * target_dx
    bottom_edge = y - 0.5 * target_dy
    top_edge = y + 0.5 * target_dy
    tolerance = (
        32.0
        * np.finfo(np.float64).eps
        * max(
            abs(config.x_min),
            abs(config.x_max),
            abs(config.y_min),
            abs(config.y_max),
            1.0,
        )
    )
    touches_x_band = (left_edge < config.x_min + fixed_x_width - tolerance) | (
        right_edge > config.x_max - fixed_x_width + tolerance
    )
    touches_y_band = (bottom_edge < config.y_min + fixed_y_width - tolerance) | (
        top_edge > config.y_max - fixed_y_width + tolerance
    )
    return 2 * touches_x_band.astype(np.int64) + touches_y_band.astype(np.int64)


def node_type_scaling_summary(
    geometry: PCNOFiniteVolumeGeometry,
    config: ShockVortexFVConfig,
    node_type: np.ndarray,
) -> dict[str, Any]:
    """Summarize raw counts, physical mass, and tagged physical widths."""

    codes = np.asarray(node_type, dtype=np.int64)
    if codes.shape != (config.nx * config.ny,):
        raise ValueError("node_type must contain one code per model-grid cell")
    if np.any((codes < 0) | (codes > 3)):
        raise ValueError("dynamic-family node types must lie in [0, 3]")
    volumes = np.asarray(geometry.node_measures[:, 0], dtype=np.float64)
    normalized_weights = np.asarray(geometry.node_weights[:, 0], dtype=np.float64)
    tagged = codes != 0
    touches_x = (codes == 2) | (codes == 3)
    touches_y = (codes == 1) | (codes == 3)
    dx = (config.x_max - config.x_min) / config.nx
    dy = (config.y_max - config.y_min) / config.ny
    edge_delta = (
        geometry.nodes[geometry.edges[:, 1]] - geometry.nodes[geometry.edges[:, 0]]
    )
    edge_length = np.linalg.norm(edge_delta, axis=-1)

    def tagged_width(mask: np.ndarray, coordinate: np.ndarray, *, axis: str) -> float:
        if not bool(mask.any()):
            return 0.0
        if axis == "x":
            distance = np.minimum(
                coordinate - config.x_min,
                config.x_max - coordinate,
            )
            return float(np.max(distance[mask]) + 0.5 * dx)
        distance = np.minimum(
            coordinate - config.y_min,
            config.y_max - coordinate,
        )
        return float(np.max(distance[mask]) + 0.5 * dy)

    counts = {str(code): int(np.count_nonzero(codes == code)) for code in range(4)}
    physical_mass = {
        str(code): float(volumes[codes == code].sum()) for code in range(4)
    }
    normalized_mass = {
        str(code): float(normalized_weights[codes == code].sum()) for code in range(4)
    }
    return {
        "nx": int(config.nx),
        "ny": int(config.ny),
        "num_nodes": int(codes.size),
        "dx": float(dx),
        "dy": float(dy),
        "counts": counts,
        "fractions": {
            code: count / float(codes.size) for code, count in counts.items()
        },
        "physical_mass_by_type": physical_mass,
        "normalized_mass_by_type": normalized_mass,
        "tagged_count": int(tagged.sum()),
        "tagged_fraction": float(tagged.mean()),
        "tagged_physical_volume": float(volumes[tagged].sum()),
        "tagged_normalized_volume_mass": float(normalized_weights[tagged].sum()),
        "x_tagged_width_per_side": tagged_width(
            touches_x, geometry.nodes[:, 0], axis="x"
        ),
        "y_tagged_width_per_side": tagged_width(
            touches_y, geometry.nodes[:, 1], axis="y"
        ),
        "minimum_one_hop_edge_length": float(edge_length.min()),
        "maximum_one_hop_edge_length": float(edge_length.max()),
        "differential_gradient_stencil_hops": 1,
        "differential_postgradient_smoothing_hops": 2,
        "differential_input_support_hops_upper_bound": 3,
        "differential_input_support_physical_upper_bound": float(
            3.0 * edge_length.max()
        ),
    }


def make_model_sample(
    geometry: PCNOFiniteVolumeGeometry,
    node_type: np.ndarray,
    *,
    mach: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Materialize one unpadded, homogeneous-resolution PCNO geometry batch."""

    codes = np.asarray(node_type, dtype=np.int64)
    if codes.shape != (geometry.nodes.shape[0],):
        raise ValueError("node_type must contain one code per geometry node")

    def batched(value: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        copied = np.array(value, copy=True)
        return torch.as_tensor(copied, dtype=dtype, device=device).unsqueeze(0)

    num_nodes = geometry.nodes.shape[0]
    return {
        "node_mask": torch.ones((1, num_nodes, 1), dtype=torch.float32, device=device),
        "nodes": batched(geometry.nodes, torch.float32),
        "node_measures": batched(geometry.node_measures, torch.float32),
        "node_weights": batched(geometry.node_weights, torch.float32),
        "node_rhos": batched(geometry.node_rhos, torch.float32),
        "directed_edges": batched(geometry.directed_edges, torch.int64),
        "edge_gradient_weights": batched(geometry.edge_gradient_weights, torch.float32),
        "node_type": batched(codes, torch.int64),
        "mach": torch.tensor([float(mach)], dtype=torch.float32, device=device),
    }


def restrict_nested_state(
    state: np.ndarray,
    *,
    fine_resolution: Resolution,
    coarse_resolution: Resolution,
) -> np.ndarray:
    """Conservatively restrict flattened fine-grid cell averages."""

    fine_nx, fine_ny = fine_resolution
    coarse_nx, coarse_ny = coarse_resolution
    return restrict_uniform_cell_averages(
        np.asarray(state),
        target_nx=fine_nx,
        target_ny=fine_ny,
        coarse_nx=coarse_nx,
        coarse_ny=coarse_ny,
    )


def fixed_boundary_distance_mask(
    nodes: np.ndarray,
    config: ShockVortexFVConfig,
    width: float,
) -> np.ndarray:
    """Select cell centers within a fixed physical distance of the boundary."""

    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("boundary-band width must be positive and finite")
    positions = np.asarray(nodes, dtype=np.float64)
    distance = np.minimum.reduce(
        (
            positions[:, 0] - config.x_min,
            config.x_max - positions[:, 0],
            positions[:, 1] - config.y_min,
            config.y_max - positions[:, 1],
        )
    )
    return distance <= float(width)


def weighted_scaled_rms(
    value: np.ndarray,
    *,
    volumes: np.ndarray,
    component_scale: Sequence[float] | np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Physical-volume RMS after fixed per-component scaling."""

    array = np.asarray(value, dtype=np.float64)
    weights = np.asarray(volumes, dtype=np.float64).reshape(-1)
    scale = np.asarray(component_scale, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != weights.size:
        raise ValueError("value and volumes must align on the node axis")
    if scale.shape != (array.shape[1],) or np.any(scale <= 0.0):
        raise ValueError("component_scale must be positive and match components")
    selected = np.ones(weights.size, dtype=bool) if mask is None else np.asarray(mask)
    if selected.shape != (weights.size,) or selected.dtype != np.bool_:
        raise ValueError("mask must be a boolean vector on the node axis")
    selected_mass = float(weights[selected].sum())
    if selected_mass <= 0.0:
        raise ValueError("metric mask has zero physical volume")
    squared = np.square(array[selected] / scale[None, :]).sum(axis=-1)
    return float(np.sqrt(np.dot(weights[selected], squared) / selected_mass))


def weighted_scaled_relative_l2(
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    volumes: np.ndarray,
    component_scale: Sequence[float] | np.ndarray,
    mask: np.ndarray | None = None,
    denominator_epsilon: float = 1.0e-30,
) -> float | None:
    """Physical-volume component-scaled relative L2 on an optional region."""

    prediction_array = np.asarray(prediction, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    if prediction_array.shape != reference_array.shape:
        raise ValueError("prediction and reference shapes must match")
    weights = np.asarray(volumes, dtype=np.float64).reshape(-1)
    scale = np.asarray(component_scale, dtype=np.float64)
    selected = np.ones(weights.size, dtype=bool) if mask is None else np.asarray(mask)
    if selected.shape != (weights.size,) or selected.dtype != np.bool_:
        raise ValueError("mask must be a boolean vector on the node axis")
    error = (prediction_array[selected] - reference_array[selected]) / scale
    baseline = reference_array[selected] / scale
    numerator = float(np.einsum("n,nc,nc->", weights[selected], error, error))
    denominator = float(np.einsum("n,nc,nc->", weights[selected], baseline, baseline))
    if denominator <= denominator_epsilon:
        return None
    return float(np.sqrt(numerator / denominator))


def physical_wavelength_band_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    resolution: Resolution,
    domain_lengths: tuple[float, float],
    component_scale: Sequence[float] | np.ndarray,
    wavelength_min: float,
    wavelength_max: float,
    denominator_epsilon: float = 1.0e-30,
) -> dict[str, float | int | None]:
    """Measure regular-grid spectral error in one fixed physical wavelength band.

    The band is declared in physical units and therefore does not move with a
    grid's Nyquist frequency. Inputs are row-major cell-centered fields with
    components on the final axis.
    """

    nx, ny = resolution
    lx, ly = (float(value) for value in domain_lengths)
    if (
        nx < 2
        or ny < 2
        or lx <= 0.0
        or ly <= 0.0
        or not np.isfinite(wavelength_min)
        or not np.isfinite(wavelength_max)
        or wavelength_min <= 0.0
        or wavelength_max <= wavelength_min
    ):
        raise ValueError("invalid grid, domain, or physical wavelength band")
    prediction_array = np.asarray(prediction, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    if (
        prediction_array.ndim != 2
        or prediction_array.shape[0] != nx * ny
        or reference_array.shape != prediction_array.shape
    ):
        raise ValueError("prediction/reference must be flattened row-major grid fields")
    scale = np.asarray(component_scale, dtype=np.float64)
    if scale.shape != (prediction_array.shape[-1],) or np.any(scale <= 0.0):
        raise ValueError("component_scale must be positive and match components")

    scaled_prediction = prediction_array.reshape(ny, nx, -1) / scale.reshape(1, 1, -1)
    scaled_reference = reference_array.reshape(ny, nx, -1) / scale.reshape(1, 1, -1)
    error_spectrum = np.fft.rfftn(
        scaled_prediction - scaled_reference,
        axes=(0, 1),
        norm="ortho",
    )
    reference_spectrum = np.fft.rfftn(
        scaled_reference,
        axes=(0, 1),
        norm="ortho",
    )
    frequency_x = np.fft.rfftfreq(nx, d=lx / nx)
    frequency_y = np.fft.fftfreq(ny, d=ly / ny)
    radial_frequency = np.hypot(
        frequency_y[:, None],
        frequency_x[None, :],
    )
    lower_frequency = 1.0 / wavelength_max
    upper_frequency = 1.0 / wavelength_min
    mask = (radial_frequency >= lower_frequency) & (radial_frequency < upper_frequency)
    mode_count = int(np.count_nonzero(mask))
    if mode_count == 0:
        raise ValueError("the physical wavelength band contains no Fourier modes")
    error_energy = float(np.sum(np.abs(error_spectrum[mask]) ** 2))
    reference_energy = float(np.sum(np.abs(reference_spectrum[mask]) ** 2))
    return {
        "wavelength_min": float(wavelength_min),
        "wavelength_max": float(wavelength_max),
        "mode_count": mode_count,
        "error_spectral_l2": float(np.sqrt(error_energy)),
        "reference_spectral_l2": float(np.sqrt(reference_energy)),
        "relative_spectral_l2": (
            float(np.sqrt(error_energy / reference_energy))
            if reference_energy > denominator_epsilon
            else None
        ),
    }


def commutator_metrics(
    *,
    coarse_current: np.ndarray,
    coarse_prediction: np.ndarray,
    fine_current: np.ndarray,
    fine_prediction: np.ndarray,
    coarse_resolution: Resolution,
    fine_resolution: Resolution,
    coarse_volumes: np.ndarray,
    state_scale: Sequence[float] | np.ndarray,
    residual_scale: Sequence[float] | np.ndarray,
) -> dict[str, float | None]:
    """Measure state and learned-update commutation after exact restriction."""

    restricted_current = restrict_nested_state(
        fine_current,
        fine_resolution=fine_resolution,
        coarse_resolution=coarse_resolution,
    )
    restricted_prediction = restrict_nested_state(
        fine_prediction,
        fine_resolution=fine_resolution,
        coarse_resolution=coarse_resolution,
    )
    coarse_update = np.asarray(coarse_prediction) - np.asarray(coarse_current)
    restricted_fine_update = restrict_nested_state(
        np.asarray(fine_prediction) - np.asarray(fine_current),
        fine_resolution=fine_resolution,
        coarse_resolution=coarse_resolution,
    )
    input_difference = np.asarray(coarse_current) - restricted_current
    prediction_difference = np.asarray(coarse_prediction) - restricted_prediction
    update_difference = coarse_update - restricted_fine_update
    input_rms = weighted_scaled_rms(
        input_difference,
        volumes=coarse_volumes,
        component_scale=state_scale,
    )
    prediction_rms = weighted_scaled_rms(
        prediction_difference,
        volumes=coarse_volumes,
        component_scale=state_scale,
    )
    update_rms = weighted_scaled_rms(
        update_difference,
        volumes=coarse_volumes,
        component_scale=residual_scale,
    )
    restricted_update_rms = weighted_scaled_rms(
        restricted_fine_update,
        volumes=coarse_volumes,
        component_scale=residual_scale,
    )
    return {
        "input_restriction_gap_scaled_rms": input_rms,
        "input_restriction_gap_relative_l2": weighted_scaled_relative_l2(
            coarse_current,
            restricted_current,
            volumes=coarse_volumes,
            component_scale=state_scale,
        ),
        "prediction_commutator_scaled_rms": prediction_rms,
        "prediction_commutator_relative_l2": weighted_scaled_relative_l2(
            coarse_prediction,
            restricted_prediction,
            volumes=coarse_volumes,
            component_scale=state_scale,
        ),
        "prediction_commutator_minus_input_gap_scaled_rms": max(
            prediction_rms - input_rms, 0.0
        ),
        "update_commutator_scaled_rms": update_rms,
        "restricted_fine_update_scaled_rms": restricted_update_rms,
        "update_commutator_relative_to_fine_update": (
            update_rms / restricted_update_rms
            if restricted_update_rms > 1.0e-30
            else None
        ),
    }
