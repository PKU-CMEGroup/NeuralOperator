#!/usr/bin/env python3
"""Evaluate one frozen dynamic PCNO through multiresolution raw rollouts.

The primary initialization is sampled once as conservative cell averages on a
common fine grid and then block-averaged to every model grid. With an optional
multiresolution reference root, evolved targets are likewise exact conservative
restrictions of one common high-fidelity evolution, including a grid finer than
the checkpoint's training grid.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no.evaluate_pcno_resolution_transfer import (
    atomic_write_json,
    build_model,
    checkpoint_model_node_type_input,
    conservative_admissibility_summary,
    load_checkpoint,
    select_device,
    sha256_file,
    timed_model_call,
    write_csv,
)
from utility.time_dependent_no.euler2d_metrics import shock_centroid
from utility.time_dependent_no.pcno_euler2d import (
    PCNOEuler2DResidual,
    PCNOEuler2DShardStore,
    parameter_count,
)
from utility.time_dependent_no.pcno_resolution_transfer import (
    MULTIRES_REFERENCE_SCHEMA,
    NODE_TYPE_PROTOCOLS,
    Resolution,
    build_resolution_geometry,
    commutator_metrics,
    fixed_boundary_distance_mask,
    initial_state_for_model_grid,
    initial_states_from_common_source,
    make_model_sample,
    node_type_scaling_summary,
    node_types_for_protocol,
    parse_resolution,
    physical_wavelength_band_metrics,
    resolution_label,
    restrict_nested_state,
    weighted_scaled_relative_l2,
    weighted_scaled_rms,
)
from utility.time_dependent_no.pcno_ripple_diagnostics import (
    conservative_to_primitive_raw,
)
from utility.time_dependent_no.shock_vortex_family import (
    REFERENCE_ARTIFACT_SCHEMA,
    config_for_family_case,
    family_case_provenance,
    load_shock_vortex_family_manifest,
)
from utility.time_dependent_no.shock_vortex_metrics import (
    endpoint_metrics,
    physical_call_metrics,
)

SCHEMA = "pcno_resolution_rollout_r1_v2"
RESTRICTION_CROSSCHECK_ABS_TOLERANCE = 1.0e-12
PHYSICAL_WAVELENGTH_BANDS = ((0.05, 0.125), (0.125, 0.25))
SHOCK_WINDOW_HALF_WIDTH = 0.2
DEFAULT_CASE_IDS = (
    "sv_e00_y04",
    "sv_e06_y04",
    "sv_e11_y04",
    "sv_e00_y00",
    "sv_e06_y00",
    "sv_e11_y00",
)
DEFAULT_RESOLUTIONS = ("125x50", "250x100", "500x200")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family-root", type=Path, required=True)
    parser.add_argument(
        "--multires-reference-root",
        type=Path,
        help=(
            "Optional case-root containing finer retained references from one "
            "common high-fidelity evolution."
        ),
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-ids", nargs="+", default=list(DEFAULT_CASE_IDS))
    parser.add_argument(
        "--allowed-splits",
        nargs="+",
        choices=("train", "validation"),
        default=("train", "validation"),
    )
    parser.add_argument("--resolutions", nargs="+", default=list(DEFAULT_RESOLUTIONS))
    parser.add_argument("--source-resolution", default="1000x400")
    parser.add_argument("--training-resolution", default="250x100")
    parser.add_argument(
        "--protocols",
        nargs="+",
        choices=NODE_TYPE_PROTOCOLS,
        default=("physical",),
    )
    parser.add_argument("--rollout-calls", type=int)
    parser.add_argument(
        "--endpoint-physical-times", type=float, nargs="+", default=(0.1, 0.3, 0.6)
    )
    parser.add_argument(
        "--boundary-band-widths", type=float, nargs="+", default=(0.02, 0.05)
    )
    parser.add_argument("--quadrature-order", type=int)
    parser.add_argument("--shock-quantile", type=float, default=0.9)
    parser.add_argument("--repeat-forward", type=int, default=2)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--expected-normalization-digest")
    parser.add_argument("--expected-step-stride", type=int)
    parser.add_argument("--expected-k-max", type=int, default=8)
    parser.add_argument(
        "--expected-domain-lengths", type=float, nargs=2, default=(2.0, 1.0)
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", choices=("none", "bf16", "fp16"), default="none")
    return parser.parse_args(argv)


def _git_head() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _source_hashes() -> dict[str, str]:
    paths = (
        Path("pcno/pcno.py"),
        Path("utility/time_dependent_no/pcno_euler2d.py"),
        Path("utility/time_dependent_no/pcno_fv_geometry.py"),
        Path("utility/time_dependent_no/pcno_resolution_transfer.py"),
        Path("utility/time_dependent_no/shock_vortex_family.py"),
        Path("utility/time_dependent_no/shock_vortex_metrics.py"),
        Path(
            "scripts/time_dependent_no/generate_pcno_shock_vortex_multires_reference.py"
        ),
        Path("scripts/time_dependent_no/evaluate_pcno_resolution_rollout.py"),
    )
    return {str(path): sha256_file(ROOT / path) for path in paths}


def _step_stride(checkpoint: Mapping[str, Any]) -> int:
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


def _validate_contract(
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

    stride = _step_stride(checkpoint)
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


def _native_geometry_audit(
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


def _load_reference(
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


def _reference_at_resolution(
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


def _as_model_state(state: np.ndarray) -> np.ndarray:
    """Round explicitly to the float32 state seen by the checkpoint."""

    return np.asarray(state, dtype=np.float32).astype(np.float64)


def _predict(
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
    return _as_model_state(prediction), timing


def _pressure_profile_shock_metrics(
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


def _expected_vortex_center(
    manifest: Mapping[str, Any],
    store: PCNOEuler2DShardStore,
    case_id: str,
    physical_time: float,
) -> tuple[float, float]:
    config = manifest["reference_config"]
    upstream_speed = float(config["shock_mach"]) * math.sqrt(float(config["gamma"]))
    arrival = (float(config["shock_x"]) - float(config["vortex_x"])) / upstream_speed
    expected_x = (
        float(config["vortex_x"]) + upstream_speed * physical_time
        if physical_time <= arrival
        else float(config["shock_x"])
        + float(config["right_u"]) * (physical_time - arrival)
    )
    return expected_x, float(store.entry(case_id)["parameters"]["vortex_y"])


def _translation_rows(
    case_id: str,
    common_states: Mapping[Resolution, np.ndarray],
    direct_states: Mapping[Resolution, np.ndarray],
    reference: Mapping[str, np.ndarray],
    *,
    resolutions: Sequence[Resolution],
    reference_resolution: Resolution,
    geometry_by_resolution: Mapping[Resolution, Any],
    state_scale: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for resolution in resolutions:
        common = np.asarray(common_states[resolution], dtype=np.float64)
        direct = np.asarray(direct_states[resolution], dtype=np.float64)
        geometry = geometry_by_resolution[resolution]
        delta = direct - common
        target = _reference_at_resolution(
            reference["conservative_states"][0],
            reference_resolution=reference_resolution,
            target_resolution=resolution,
        )
        row = {
            "case_id": case_id,
            "resolution": resolution_label(resolution),
            "direct_vs_common_max_abs": float(np.max(np.abs(delta))),
            "direct_vs_common_scaled_rms": weighted_scaled_rms(
                delta,
                volumes=geometry.node_measures,
                component_scale=state_scale,
            ),
            "reference_available": target is not None,
        }
        if target is not None:
            reference_delta = _as_model_state(common) - target
            row.update(
                {
                    "model_input_vs_reference_t0_max_abs": float(
                        np.max(np.abs(reference_delta))
                    ),
                    "model_input_vs_reference_t0_scaled_rms": weighted_scaled_rms(
                        reference_delta,
                        volumes=geometry.node_measures,
                        component_scale=state_scale,
                    ),
                }
            )
        rows.append(row)
    return rows


def _state_row(
    *,
    case_id: str,
    case_split: str,
    protocol: str,
    resolution: Resolution,
    call: int,
    physical_time: float,
    state: np.ndarray,
    reference_state: np.ndarray | None,
    volumes: np.ndarray,
    state_scale: np.ndarray,
    gamma: float,
    timing: Mapping[str, Any] | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "case_id": case_id,
        "case_split": case_split,
        "node_type_protocol": protocol,
        "resolution": resolution_label(resolution),
        "call": call,
        "physical_time": physical_time,
        "num_nodes": int(state.shape[0]),
        "reference_available": reference_state is not None,
        "state_scaled_rms": weighted_scaled_rms(
            state, volumes=volumes, component_scale=state_scale
        ),
        **conservative_admissibility_summary(state, gamma=gamma),
    }
    if reference_state is not None:
        error = state - reference_state
        row.update(
            {
                "error_scaled_rms": weighted_scaled_rms(
                    error, volumes=volumes, component_scale=state_scale
                ),
                "scaled_relative_l2_physical_volume": weighted_scaled_relative_l2(
                    state,
                    reference_state,
                    volumes=volumes,
                    component_scale=state_scale,
                ),
            }
        )
    if timing is not None:
        row.update(timing)
    return row


def _regional_error_rows(
    *,
    case_id: str,
    case_split: str,
    protocol: str,
    resolution: Resolution,
    call: int,
    physical_time: float,
    state: np.ndarray,
    reference_state: np.ndarray,
    geometry: Any,
    config: Any,
    physical_node_type: np.ndarray,
    boundary_band_widths: Sequence[float],
    state_scale: np.ndarray,
) -> list[dict[str, Any]]:
    regions: dict[str, np.ndarray] = {
        "boundary_touching_cells": np.asarray(physical_node_type) != 0,
    }
    for width in boundary_band_widths:
        regions[f"boundary_distance_le_{float(width):g}"] = (
            fixed_boundary_distance_mask(geometry.nodes, config, float(width))
        )
    rows = []
    for name, mask in regions.items():
        rows.append(
            {
                "case_id": case_id,
                "case_split": case_split,
                "node_type_protocol": protocol,
                "resolution": resolution_label(resolution),
                "call": call,
                "physical_time": physical_time,
                "region": name,
                "region_node_count": int(mask.sum()),
                "region_physical_volume": float(
                    np.asarray(geometry.node_measures).reshape(-1)[mask].sum()
                ),
                "error_scaled_rms": weighted_scaled_rms(
                    state - reference_state,
                    volumes=geometry.node_measures,
                    component_scale=state_scale,
                    mask=mask,
                ),
                "scaled_relative_l2_physical_volume": weighted_scaled_relative_l2(
                    state,
                    reference_state,
                    volumes=geometry.node_measures,
                    component_scale=state_scale,
                    mask=mask,
                ),
            }
        )
    return rows


def _structure_row(
    *,
    case_id: str,
    case_split: str,
    protocol: str,
    resolution: Resolution,
    call: int,
    physical_time: float,
    state: np.ndarray,
    reference_state: np.ndarray,
    initial_state: np.ndarray,
    cumulative_boundary_exchange: np.ndarray,
    geometry: Any,
    config: Any,
    manifest: Mapping[str, Any],
    store: PCNOEuler2DShardStore,
    state_scale: np.ndarray,
    gamma: float,
    shock_quantile: float,
) -> dict[str, Any]:
    prediction_primitive = conservative_to_primitive_raw(state, gamma=gamma)
    reference_primitive = conservative_to_primitive_raw(reference_state, gamma=gamma)
    prediction_centroid = np.asarray(
        shock_centroid(prediction_primitive, geometry.nodes, geometry.edges)
    ).reshape(-1)
    reference_centroid = np.asarray(
        shock_centroid(reference_primitive, geometry.nodes, geometry.edges)
    ).reshape(-1)
    spectral_fields: dict[str, Any] = {}
    domain_lengths = (
        float(config.x_max - config.x_min),
        float(config.y_max - config.y_min),
    )
    for band_index, (wavelength_min, wavelength_max) in enumerate(
        PHYSICAL_WAVELENGTH_BANDS, start=1
    ):
        metrics = physical_wavelength_band_metrics(
            state,
            reference_state,
            resolution=resolution,
            domain_lengths=domain_lengths,
            component_scale=state_scale,
            wavelength_min=wavelength_min,
            wavelength_max=wavelength_max,
        )
        for name, value in metrics.items():
            spectral_fields[f"physical_wavelength_band_{band_index}_{name}"] = value
    return {
        "case_id": case_id,
        "case_split": case_split,
        "node_type_protocol": protocol,
        "resolution": resolution_label(resolution),
        "call": call,
        "physical_time": physical_time,
        "prediction_shock_centroid": prediction_centroid,
        "reference_shock_centroid": reference_centroid,
        "shock_centroid_distance_physical": float(
            np.linalg.norm(prediction_centroid - reference_centroid)
        ),
        **_pressure_profile_shock_metrics(
            state,
            reference_state,
            resolution=resolution,
            x_min=config.x_min,
            x_max=config.x_max,
            gamma=gamma,
            shock_center_x=config.shock_x,
        ),
        **endpoint_metrics(
            state,
            reference_state,
            positions=geometry.nodes,
            edges=geometry.edges,
            volumes=np.asarray(geometry.node_measures).reshape(-1),
            component_scale=state_scale,
            gamma=gamma,
            shock_quantile=shock_quantile,
            vortex_center=_expected_vortex_center(
                manifest, store, case_id, physical_time
            ),
        ),
        **physical_call_metrics(
            state,
            reference_state,
            initial_state,
            volumes=np.asarray(geometry.node_measures).reshape(-1),
            reference_cumulative_boundary_exchange=cumulative_boundary_exchange,
            component_scale=state_scale,
        ),
        **spectral_fields,
    }


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [
        float(row[key])
        for row in rows
        if row.get(key) is not None and math.isfinite(float(row[key]))
    ]
    return float(np.mean(values)) if values else None


def _commutator_row(
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


def _aggregate(
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


def _architecture_audit(
    model: PCNOEuler2DResidual,
    geometry_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    prefactors = [
        float(layer.gw1.detach().float().cpu()) for layer in model.backbone.gws
    ]
    resolution_rows = []
    physical_rows = [
        row for row in geometry_rows if row["node_type_protocol"] == "physical"
    ]
    for row in physical_rows:
        dx = float(row["dx"])
        dy = float(row["dy"])
        resolution_rows.append(
            {
                "resolution": row["resolution"],
                "three_hop_physical_support_upper_bound": row[
                    "differential_input_support_physical_upper_bound"
                ],
                "softsign_boundary_jump_proxy_x_by_layer": [
                    abs(value) / dx / (1.0 + abs(value) / dx) for value in prefactors
                ],
                "softsign_boundary_jump_proxy_y_by_layer": [
                    abs(value) / dy / (1.0 + abs(value) / dy) for value in prefactors
                ],
            }
        )
    return {
        "paper_equation_alignment": {
            "integral": (
                "paper Eq. 13-15 uniform density: node_weights are rho(x)*cell_volume "
                "and sum to one"
            ),
            "gradient": "paper Eq. 16-19 least-squares edge pseudoinverse",
            "maintained_differences_from_paper_eq_20_21": (
                "each gradient is averaged for two fixed graph hops before a learned "
                "scalar prefactor and SoftSign; the paper states direct SoftSign(gradient)"
            ),
            "fourier": (
                "the maintained checkpoint fixes physical periods and modes; the paper "
                "describes length scales L as learnable"
            ),
        },
        "learned_gradient_softsign_prefactors": prefactors,
        "resolution_scaling": resolution_rows,
        "evaluation_padding_contract": (
            "batch size one, unpadded nodes and edges; heterogeneous padded batching is "
            "not used by this evaluator"
        ),
    }


def _run_experiment(
    args: argparse.Namespace,
    checkpoint: Mapping[str, Any],
    manifest: Mapping[str, Any],
    store: PCNOEuler2DShardStore,
    *,
    resolutions: Sequence[Resolution],
    source_resolution: Resolution,
    training_resolution: Resolution,
    stride: int,
    rollout_calls: int,
    physical_dt: float,
    endpoint_calls: set[int],
) -> dict[str, Any]:
    started_wall = perf_counter()
    protocols = list(dict.fromkeys(args.protocols))
    ordered_resolutions = sorted(resolutions, key=lambda value: value[0] * value[1])
    allowed_splits = set(args.allowed_splits)
    case_provenance: dict[str, dict[str, Any]] = {}
    case_configs = {}
    for case_id in args.case_ids:
        provenance = family_case_provenance(manifest, case_id)
        if provenance["split"] not in allowed_splits:
            raise ValueError(
                f"case {case_id} belongs to disallowed split {provenance['split']}"
            )
        if case_id not in store.keys:
            raise ValueError(f"case {case_id} is absent from checkpoint-bound shards")
        case_provenance[case_id] = provenance
        case_configs[case_id] = config_for_family_case(manifest, case_id)
    if not case_configs:
        raise ValueError("at least one physical case is required")

    device = select_device(args.device)
    if args.amp != "none" and device.type != "cuda":
        raise ValueError("mixed precision evaluation requires CUDA")
    model, normalization = build_model(checkpoint, device)
    first_config = next(iter(case_configs.values()))
    physical_lengths = (
        first_config.x_max - first_config.x_min,
        first_config.y_max - first_config.y_min,
    )
    if physical_lengths != tuple(float(v) for v in args.expected_domain_lengths):
        raise ValueError("physical domain and Fourier periods differ")

    geometry_by_resolution = {}
    config_by_resolution = {}
    physical_type_by_resolution = {}
    sample_by_resolution_protocol = {}
    geometry_rows: list[dict[str, Any]] = []
    for resolution in ordered_resolutions:
        build_started = perf_counter()
        config, geometry = build_resolution_geometry(first_config, resolution)
        build_seconds = perf_counter() - build_started
        geometry_by_resolution[resolution] = geometry
        config_by_resolution[resolution] = config
        physical_type = node_types_for_protocol(
            geometry,
            config,
            "physical",
            training_resolution=training_resolution,
        )
        physical_type_by_resolution[resolution] = physical_type
        for protocol in protocols:
            node_type = node_types_for_protocol(
                geometry,
                config,
                protocol,
                training_resolution=training_resolution,
            )
            sample_by_resolution_protocol[(resolution, protocol)] = make_model_sample(
                geometry,
                node_type,
                mach=first_config.shock_mach,
                device=device,
            )
            geometry_rows.append(
                {
                    "resolution": resolution_label(resolution),
                    "node_type_protocol": protocol,
                    "num_directed_edges": int(geometry.directed_edges.shape[0]),
                    "minimum_stencil_singular_value": geometry.minimum_stencil_singular_value,
                    "maximum_stencil_condition_number": geometry.maximum_stencil_condition_number,
                    "maximum_coordinate_gradient_error": geometry.maximum_coordinate_gradient_error,
                    "geometry_build_seconds": build_seconds,
                    **node_type_scaling_summary(geometry, config, node_type),
                }
            )
        print(
            f"built {resolution_label(resolution)} nodes={geometry.nodes.shape[0]} "
            f"edges={geometry.directed_edges.shape[0]} seconds={build_seconds:.3f}",
            flush=True,
        )

    native_geometry_audit = _native_geometry_audit(
        geometry_by_resolution[training_resolution], store, args.case_ids[0]
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)

    translation_rows: list[dict[str, Any]] = []
    reference_check_rows: list[dict[str, Any]] = []
    reference_gap_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    regional_rows: list[dict[str, Any]] = []
    structure_rows: list[dict[str, Any]] = []
    teacher_state_rows: list[dict[str, Any]] = []
    commutator_rows: list[dict[str, Any]] = []
    completion_rows: list[dict[str, Any]] = []
    reference_resolutions_seen: set[Resolution] = set()
    execution = {
        "logical_model_calls": 0,
        "actual_forward_passes_including_repeats": 0,
        "total_forward_seconds": 0.0,
        "maximum_repeat_abs_difference": 0.0,
        "maximum_peak_gpu_memory_bytes": 0,
    }

    def account(timing: Mapping[str, Any]) -> None:
        execution["logical_model_calls"] += 1
        execution["actual_forward_passes_including_repeats"] += len(
            timing["forward_seconds"]
        )
        execution["total_forward_seconds"] += float(sum(timing["forward_seconds"]))
        repeat = float(timing["repeat_max_abs"])
        if math.isfinite(repeat):
            execution["maximum_repeat_abs_difference"] = max(
                float(execution["maximum_repeat_abs_difference"]), repeat
            )
        peak = timing.get("peak_gpu_memory_bytes")
        if peak is not None:
            execution["maximum_peak_gpu_memory_bytes"] = max(
                int(execution["maximum_peak_gpu_memory_bytes"]), int(peak)
            )

    for case_index, case_id in enumerate(args.case_ids, start=1):
        provenance = case_provenance[case_id]
        case_split = str(provenance["split"])
        case_config = case_configs[case_id]
        print(
            f"case {case_index}/{len(args.case_ids)} {case_id} split={case_split}: "
            "loading reference and translating t0",
            flush=True,
        )
        reference, reference_check = _load_reference(
            args.family_root,
            args.multires_reference_root,
            store,
            manifest,
            case_id,
            training_resolution=training_resolution,
        )
        reference_resolution = tuple(
            int(value) for value in reference["retained_resolution"]
        )
        reference_resolutions_seen.add(reference_resolution)
        for resolution in ordered_resolutions:
            if (
                reference_resolution[0] % resolution[0]
                or reference_resolution[1] % resolution[1]
            ):
                raise ValueError(
                    "active retained truth cannot restrict to every evaluation grid"
                )
        reference_check_rows.append(reference_check)
        _, common_states = initial_states_from_common_source(
            case_config,
            ordered_resolutions,
            source_resolution=source_resolution,
            dtype=torch.float64,
            quadrature_order=args.quadrature_order,
        )
        direct_states = {
            resolution: initial_state_for_model_grid(
                case_config,
                resolution,
                dtype=torch.float64,
                quadrature_order=args.quadrature_order,
            )
            for resolution in ordered_resolutions
        }
        stored_t0 = np.asarray(store.states(case_id)[0])
        reconstructed_native = np.asarray(
            common_states[training_resolution], dtype=np.float32
        )
        if not np.array_equal(reconstructed_native, stored_t0):
            raise ValueError(
                f"common-source native t0 does not reproduce stored float32 state: {case_id}"
            )
        translation_rows.extend(
            _translation_rows(
                case_id,
                common_states,
                direct_states,
                reference,
                resolutions=ordered_resolutions,
                reference_resolution=reference_resolution,
                geometry_by_resolution=geometry_by_resolution,
                state_scale=normalization.state_scale,
            )
        )

        reference_gap_lookup: dict[tuple[Resolution, Resolution, int], float] = {}
        for call in range(rollout_calls + 1):
            frame = call * stride
            physical_time = call * physical_dt
            if not math.isclose(
                float(reference["physical_times"][frame]),
                physical_time,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError(f"reference time mismatch for {case_id} call {call}")
            for coarse, fine in pairwise(ordered_resolutions):
                coarse_reference = _reference_at_resolution(
                    reference["conservative_states"][frame],
                    reference_resolution=reference_resolution,
                    target_resolution=coarse,
                )
                fine_reference = _reference_at_resolution(
                    reference["conservative_states"][frame],
                    reference_resolution=reference_resolution,
                    target_resolution=fine,
                )
                if coarse_reference is None or fine_reference is None:
                    continue
                restricted = restrict_nested_state(
                    fine_reference,
                    fine_resolution=fine,
                    coarse_resolution=coarse,
                )
                gap = weighted_scaled_rms(
                    coarse_reference - restricted,
                    volumes=geometry_by_resolution[coarse].node_measures,
                    component_scale=normalization.state_scale,
                )
                reference_gap_lookup[(coarse, fine, call)] = gap
                reference_gap_rows.append(
                    {
                        "case_id": case_id,
                        "case_split": case_split,
                        "coarse_resolution": resolution_label(coarse),
                        "fine_resolution": resolution_label(fine),
                        "call": call,
                        "physical_time": physical_time,
                        "reference_discretization_gap_scaled_rms": gap,
                        "reference_discretization_gap_relative_l2": weighted_scaled_relative_l2(
                            coarse_reference,
                            restricted,
                            volumes=geometry_by_resolution[coarse].node_measures,
                            component_scale=normalization.state_scale,
                        ),
                    }
                )

        for protocol in protocols:
            free_states = {
                resolution: _as_model_state(common_states[resolution])
                for resolution in ordered_resolutions
            }
            initial_states = {
                resolution: np.array(state, copy=True)
                for resolution, state in free_states.items()
            }
            active_by_resolution = {
                resolution: True for resolution in ordered_resolutions
            }
            all_calls_admissible_by_resolution = {
                resolution: True for resolution in ordered_resolutions
            }
            completed_calls_by_resolution = {
                resolution: 0 for resolution in ordered_resolutions
            }
            first_nonfinite_call_by_resolution = {
                resolution: None for resolution in ordered_resolutions
            }

            for resolution in ordered_resolutions:
                reference_t0 = _reference_at_resolution(
                    reference["conservative_states"][0],
                    reference_resolution=reference_resolution,
                    target_resolution=resolution,
                )
                geometry = geometry_by_resolution[resolution]
                state_rows.append(
                    _state_row(
                        case_id=case_id,
                        case_split=case_split,
                        protocol=protocol,
                        resolution=resolution,
                        call=0,
                        physical_time=0.0,
                        state=free_states[resolution],
                        reference_state=reference_t0,
                        volumes=geometry.node_measures,
                        state_scale=normalization.state_scale,
                        gamma=normalization.gamma,
                        timing=None,
                    )
                )
                if reference_t0 is not None:
                    regional_rows.extend(
                        _regional_error_rows(
                            case_id=case_id,
                            case_split=case_split,
                            protocol=protocol,
                            resolution=resolution,
                            call=0,
                            physical_time=0.0,
                            state=free_states[resolution],
                            reference_state=reference_t0,
                            geometry=geometry,
                            config=config_by_resolution[resolution],
                            physical_node_type=physical_type_by_resolution[resolution],
                            boundary_band_widths=args.boundary_band_widths,
                            state_scale=normalization.state_scale,
                        )
                    )

            for call in range(1, rollout_calls + 1):
                physical_time = call * physical_dt
                frame = call * stride
                input_physical_time = (call - 1) * physical_dt
                input_frame = (call - 1) * stride
                previous_states = free_states
                active_at_start = dict(active_by_resolution)
                next_states = dict(previous_states)
                timing_by_resolution = {}
                finite_by_resolution = {}
                for resolution in ordered_resolutions:
                    if not active_at_start[resolution]:
                        continue
                    prediction, timing = _predict(
                        model,
                        sample_by_resolution_protocol[(resolution, protocol)],
                        previous_states[resolution],
                        device=device,
                        amp=args.amp,
                        repeats=args.repeat_forward if call == 1 else 1,
                    )
                    account(timing)
                    next_states[resolution] = prediction
                    timing_by_resolution[resolution] = timing
                    finite = bool(np.isfinite(prediction).all())
                    finite_by_resolution[resolution] = finite
                    if not finite:
                        active_by_resolution[resolution] = False

                teacher_currents = {}
                teacher_predictions = {}
                teacher_finite_by_resolution = {}
                for resolution in ordered_resolutions:
                    reference_input = _reference_at_resolution(
                        reference["conservative_states"][input_frame],
                        reference_resolution=reference_resolution,
                        target_resolution=resolution,
                    )
                    reference_target = _reference_at_resolution(
                        reference["conservative_states"][frame],
                        reference_resolution=reference_resolution,
                        target_resolution=resolution,
                    )
                    if reference_input is None or reference_target is None:
                        raise ValueError(
                            "teacher-forced reference is unavailable on an "
                            "evaluation grid"
                        )
                    teacher_current = _as_model_state(reference_input)
                    teacher_prediction, teacher_timing = _predict(
                        model,
                        sample_by_resolution_protocol[(resolution, protocol)],
                        teacher_current,
                        device=device,
                        amp=args.amp,
                        repeats=1,
                    )
                    account(teacher_timing)
                    teacher_currents[resolution] = teacher_current
                    teacher_predictions[resolution] = teacher_prediction
                    teacher_finite_by_resolution[resolution] = bool(
                        np.isfinite(teacher_prediction).all()
                    )
                    teacher_row = _state_row(
                        case_id=case_id,
                        case_split=case_split,
                        protocol=protocol,
                        resolution=resolution,
                        call=call,
                        physical_time=physical_time,
                        state=teacher_prediction,
                        reference_state=reference_target,
                        volumes=geometry_by_resolution[resolution].node_measures,
                        state_scale=normalization.state_scale,
                        gamma=normalization.gamma,
                        timing=teacher_timing,
                    )
                    teacher_row.update(
                        {
                            "input_call": call - 1,
                            "input_frame": input_frame,
                            "output_frame": frame,
                            "input_physical_time": input_physical_time,
                        }
                    )
                    teacher_state_rows.append(teacher_row)

                for coarse, fine in pairwise(ordered_resolutions):
                    pair_is_finite = (
                        active_at_start[coarse]
                        and active_at_start[fine]
                        and finite_by_resolution.get(coarse, False)
                        and finite_by_resolution.get(fine, False)
                    )
                    if pair_is_finite:
                        free_metrics = commutator_metrics(
                            coarse_current=previous_states[coarse],
                            coarse_prediction=next_states[coarse],
                            fine_current=previous_states[fine],
                            fine_prediction=next_states[fine],
                            coarse_resolution=coarse,
                            fine_resolution=fine,
                            coarse_volumes=geometry_by_resolution[coarse].node_measures,
                            state_scale=normalization.state_scale,
                            residual_scale=normalization.residual_scale,
                        )
                        reference_gap = reference_gap_lookup.get((coarse, fine, call))
                        commutator_rows.append(
                            _commutator_row(
                                case_id=case_id,
                                case_split=case_split,
                                protocol=protocol,
                                kind="free_rollout",
                                coarse=coarse,
                                fine=fine,
                                call=call,
                                input_frame=input_frame,
                                output_frame=frame,
                                input_physical_time=input_physical_time,
                                output_physical_time=physical_time,
                                reference_gap=reference_gap,
                                metrics=free_metrics,
                            )
                        )

                        local_current = _as_model_state(
                            restrict_nested_state(
                                previous_states[fine],
                                fine_resolution=fine,
                                coarse_resolution=coarse,
                            )
                        )
                        local_prediction, local_timing = _predict(
                            model,
                            sample_by_resolution_protocol[(coarse, protocol)],
                            local_current,
                            device=device,
                            amp=args.amp,
                            repeats=1,
                        )
                        account(local_timing)
                        if bool(np.isfinite(local_prediction).all()):
                            local_metrics = commutator_metrics(
                                coarse_current=local_current,
                                coarse_prediction=local_prediction,
                                fine_current=previous_states[fine],
                                fine_prediction=next_states[fine],
                                coarse_resolution=coarse,
                                fine_resolution=fine,
                                coarse_volumes=geometry_by_resolution[
                                    coarse
                                ].node_measures,
                                state_scale=normalization.state_scale,
                                residual_scale=normalization.residual_scale,
                            )
                            commutator_rows.append(
                                _commutator_row(
                                    case_id=case_id,
                                    case_split=case_split,
                                    protocol=protocol,
                                    kind="local_restricted_fine_input",
                                    coarse=coarse,
                                    fine=fine,
                                    call=call,
                                    input_frame=input_frame,
                                    output_frame=frame,
                                    input_physical_time=input_physical_time,
                                    output_physical_time=physical_time,
                                    reference_gap=reference_gap,
                                    metrics=local_metrics,
                                )
                            )

                    teacher_pair_is_finite = (
                        teacher_finite_by_resolution[coarse]
                        and teacher_finite_by_resolution[fine]
                    )
                    if teacher_pair_is_finite:
                        teacher_metrics = commutator_metrics(
                            coarse_current=teacher_currents[coarse],
                            coarse_prediction=teacher_predictions[coarse],
                            fine_current=teacher_currents[fine],
                            fine_prediction=teacher_predictions[fine],
                            coarse_resolution=coarse,
                            fine_resolution=fine,
                            coarse_volumes=geometry_by_resolution[coarse].node_measures,
                            state_scale=normalization.state_scale,
                            residual_scale=normalization.residual_scale,
                        )
                        reference_gap = reference_gap_lookup.get((coarse, fine, call))
                        commutator_rows.append(
                            _commutator_row(
                                case_id=case_id,
                                case_split=case_split,
                                protocol=protocol,
                                kind="teacher_forced_exact_reference_input",
                                coarse=coarse,
                                fine=fine,
                                call=call,
                                input_frame=input_frame,
                                output_frame=frame,
                                input_physical_time=input_physical_time,
                                output_physical_time=physical_time,
                                reference_gap=reference_gap,
                                metrics=teacher_metrics,
                            )
                        )

                free_states = next_states
                for resolution in ordered_resolutions:
                    if not active_at_start[resolution]:
                        continue
                    geometry = geometry_by_resolution[resolution]
                    target = _reference_at_resolution(
                        reference["conservative_states"][frame],
                        reference_resolution=reference_resolution,
                        target_resolution=resolution,
                    )
                    state_row = _state_row(
                        case_id=case_id,
                        case_split=case_split,
                        protocol=protocol,
                        resolution=resolution,
                        call=call,
                        physical_time=physical_time,
                        state=free_states[resolution],
                        reference_state=target,
                        volumes=geometry.node_measures,
                        state_scale=normalization.state_scale,
                        gamma=normalization.gamma,
                        timing=timing_by_resolution[resolution],
                    )
                    state_rows.append(state_row)
                    finite = finite_by_resolution[resolution]
                    if finite:
                        completed_calls_by_resolution[resolution] = call
                        all_calls_admissible_by_resolution[resolution] = (
                            all_calls_admissible_by_resolution[resolution]
                            and bool(state_row["admissible"])
                        )
                    else:
                        all_calls_admissible_by_resolution[resolution] = False
                        first_nonfinite_call_by_resolution[resolution] = call
                    if target is not None and finite:
                        regional_rows.extend(
                            _regional_error_rows(
                                case_id=case_id,
                                case_split=case_split,
                                protocol=protocol,
                                resolution=resolution,
                                call=call,
                                physical_time=physical_time,
                                state=free_states[resolution],
                                reference_state=target,
                                geometry=geometry,
                                config=config_by_resolution[resolution],
                                physical_node_type=physical_type_by_resolution[
                                    resolution
                                ],
                                boundary_band_widths=args.boundary_band_widths,
                                state_scale=normalization.state_scale,
                            )
                        )
                        if call in endpoint_calls:
                            cumulative_exchange = np.sum(
                                reference["interval_boundary_exchange"][:frame], axis=0
                            )
                            structure_rows.append(
                                _structure_row(
                                    case_id=case_id,
                                    case_split=case_split,
                                    protocol=protocol,
                                    resolution=resolution,
                                    call=call,
                                    physical_time=physical_time,
                                    state=free_states[resolution],
                                    reference_state=target,
                                    initial_state=initial_states[resolution],
                                    cumulative_boundary_exchange=cumulative_exchange,
                                    geometry=geometry,
                                    config=config_by_resolution[resolution],
                                    manifest=manifest,
                                    store=store,
                                    state_scale=normalization.state_scale,
                                    gamma=normalization.gamma,
                                    shock_quantile=args.shock_quantile,
                                )
                            )
                if call == 1 or call in endpoint_calls:
                    print(
                        f"rollout case={case_id} types={protocol} call={call}/"
                        f"{rollout_calls} t={physical_time:.3f}",
                        flush=True,
                    )

            for resolution in ordered_resolutions:
                completed_calls = completed_calls_by_resolution[resolution]
                completion_rows.append(
                    {
                        "case_id": case_id,
                        "case_split": case_split,
                        "node_type_protocol": protocol,
                        "resolution": resolution_label(resolution),
                        "requested_calls": rollout_calls,
                        "completed_finite_calls": completed_calls,
                        "full_completion": completed_calls == rollout_calls,
                        "all_completed_calls_admissible": (
                            all_calls_admissible_by_resolution[resolution]
                        ),
                        "first_nonfinite_call": (
                            first_nonfinite_call_by_resolution[resolution]
                        ),
                    }
                )

        del reference, common_states, direct_states
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if len(reference_resolutions_seen) != 1:
        raise ValueError(
            "all cases must use one common retained-reference resolution contract"
        )
    active_reference_resolution = next(iter(reference_resolutions_seen))

    write_csv(output_dir / "geometry.csv", geometry_rows)
    write_csv(output_dir / "translation_preflight.csv", translation_rows)
    write_csv(output_dir / "reference_checks.csv", reference_check_rows)
    write_csv(output_dir / "reference_gaps.csv", reference_gap_rows)
    write_csv(output_dir / "state_metrics.csv", state_rows)
    write_csv(output_dir / "regional_errors.csv", regional_rows)
    write_csv(output_dir / "structure_metrics.csv", structure_rows)
    write_csv(output_dir / "teacher_forced_state_metrics.csv", teacher_state_rows)
    write_csv(output_dir / "commutators.csv", commutator_rows)
    write_csv(output_dir / "completion.csv", completion_rows)

    checkpoint_sha = sha256_file(args.checkpoint)
    source_hashes = _source_hashes()
    table_files = sorted(output_dir.glob("*.csv"))
    elapsed_wall = perf_counter() - started_wall
    execution.update(
        {
            "device": str(device),
            "amp": args.amp,
            "repeat_forward_scope": "first free call for every case/grid/protocol",
            "parameter_count": parameter_count(model),
            "teacher_forced_state_rows": len(teacher_state_rows),
            "wall_seconds_before_summary_write": elapsed_wall,
            "csv_storage_bytes": int(sum(path.stat().st_size for path in table_files)),
        }
    )
    if device.type != "cuda":
        execution["maximum_peak_gpu_memory_bytes"] = None
    return {
        "schema": SCHEMA,
        "status": "complete",
        "scientific_scope": {
            "primary_test": (
                "raw recurrent rollout of one frozen parameter vector on common-fine "
                "conservative representations"
            ),
            "teacher_forced_diagnostic": (
                "at every aligned map call, independently restrict the exact common "
                "reference input to each grid, apply the same frozen model, and compare "
                "coarse prediction with exact restriction of fine prediction"
            ),
            "restriction_consistent_reference_resolutions": [
                resolution_label(resolution)
                for resolution in ordered_resolutions
                if active_reference_resolution[0] % resolution[0] == 0
                and active_reference_resolution[1] % resolution[1] == 0
            ],
            "commutator_only_resolutions": [
                resolution_label(resolution)
                for resolution in ordered_resolutions
                if active_reference_resolution[0] % resolution[0] != 0
                or active_reference_resolution[1] % resolution[1] != 0
            ],
            "claim_boundary": (
                "Finite tests can support zero-shot discretization generalization only for "
                "the named PDE/geometry/boundary/timestep family and bound sampling, "
                "quadrature, normalization, and evaluator. They do not prove that PCNO is "
                "an operator or establish native-solver, PDE-family, geometry, or timestep transfer."
            ),
        },
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": checkpoint_sha,
            "normalization_digest": checkpoint["normalization_digest"],
            "data_manifest_digest": checkpoint["data_manifest_digest"],
            "config_digest": checkpoint["config_digest"],
            "boundary_mode": checkpoint["boundary_mode"],
            "raw_recurrence": checkpoint["raw_recurrence"],
            "model_config": checkpoint["model_config"],
            "model_node_type_input": checkpoint_model_node_type_input(checkpoint),
            "data_contract": checkpoint["data_contract"],
        },
        "source": {"git_head": _git_head(), "file_sha256": source_hashes},
        "family": {
            "root": str(args.family_root),
            "manifest_digest_sha256": manifest["manifest_digest_sha256"],
            "cases": [case_provenance[case_id] for case_id in args.case_ids],
        },
        "translation_contract": {
            "state": (
                "cell averages of conservative [rho,rho_u,rho_v,E], sampled once on "
                "the common source grid and block-averaged; no primitive interpolation"
            ),
            "source_resolution": resolution_label(source_resolution),
            "retained_reference_resolution": resolution_label(
                active_reference_resolution
            ),
            "training_resolution": resolution_label(training_resolution),
            "evaluation_resolutions": [
                resolution_label(value) for value in ordered_resolutions
            ],
            "geometry": (
                "centroids, volumes, boundary types, directed graph, and least-squares "
                "gradient weights regenerated independently on every grid"
            ),
            "node_type_protocols": protocols,
            "Fourier_periods": list(args.expected_domain_lengths),
            "k_max": args.expected_k_max,
            "normalization": "immutable checkpoint normalization",
            "reference_provenance": (
                "common high-fidelity evolution with exact conservative block "
                "restriction"
            ),
            "boundary_policy": "model_all_nodes raw recurrence",
            "teacher_forced_input": (
                "exact common-reference conservative state at the preceding aligned "
                "frame, independently restricted to each grid and rounded to float32 "
                "at the model boundary"
            ),
            "step_stride": stride,
            "physical_delta_t_per_model_call": physical_dt,
            "rollout_calls": rollout_calls,
            "physical_horizon": rollout_calls * physical_dt,
        },
        "native_geometry_audit": native_geometry_audit,
        "architecture_audit": _architecture_audit(model, geometry_rows),
        "execution": execution,
        "completion": completion_rows,
        "aggregate": _aggregate(
            state_rows,
            commutator_rows,
            completion_rows,
            teacher_state_rows=teacher_state_rows,
            final_call=rollout_calls,
        ),
        "artifacts": {
            path.name: {"bytes": path.stat().st_size} for path in table_files
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    resolutions = [parse_resolution(value) for value in args.resolutions]
    source_resolution = parse_resolution(args.source_resolution)
    training_resolution = parse_resolution(args.training_resolution)
    manifest_path = args.family_root / "family_manifest.json"
    manifest = load_shock_vortex_family_manifest(manifest_path)
    checkpoint = load_checkpoint(args.checkpoint)
    store = PCNOEuler2DShardStore(args.data_dir, max_cached_trajectories=2)
    try:
        stride, rollout_calls, physical_dt, endpoint_calls = _validate_contract(
            args,
            checkpoint,
            manifest,
            store,
            resolutions,
            source_resolution,
            training_resolution,
        )
        print(
            "resolution rollout:",
            f"device={args.device}",
            f"amp={args.amp}",
            f"cases={len(args.case_ids)}",
            f"calls={rollout_calls}",
            f"grids={','.join(resolution_label(value) for value in resolutions)}",
            flush=True,
        )
        summary = _run_experiment(
            args,
            checkpoint,
            manifest,
            store,
            resolutions=resolutions,
            source_resolution=source_resolution,
            training_resolution=training_resolution,
            stride=stride,
            rollout_calls=rollout_calls,
            physical_dt=physical_dt,
            endpoint_calls=endpoint_calls,
        )
        atomic_write_json(args.output_dir / "summary.json", summary)
    finally:
        store.close()
    print(f"wrote {args.output_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
