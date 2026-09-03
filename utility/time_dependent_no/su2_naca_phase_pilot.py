"""Frozen solver-only phase-population analysis for the SU2 NACA trajectory."""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from functools import cache
from hashlib import sha256
from itertools import pairwise
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from utility.time_dependent_no.su2_naca_trajectory import (
    NACA_HISTORY_FIELDS,
    NACA_TRAJECTORY_HISTORY_FILENAME,
    NACA_TRAJECTORY_OUTPUT_STEM,
    NACA_TRAJECTORY_RECEIPT_FILENAME,
    NACA_TRAJECTORY_STORAGE_FILENAME,
    NACA_TRAJECTORY_STORAGE_SCHEMA,
    load_verified_trajectory_receipt,
)
from utility.time_dependent_no.su2_native_replay import (
    NACA_NATIVE_RESTART_FIELDS,
    lumped_vertex_areas,
)
from utility.time_dependent_no.su2_restart_contract import (
    NACA_DYNAMIC_FIELDS,
    parse_su2_mesh,
    read_su2_binary_restart,
    sha256_file,
)

NACA_PHASE_PILOT_CONTRACT_SCHEMA = "time_dependent_no.su2_naca0012_phase_pilot.v1"
NACA_PHASE_PILOT_CONTRACT_SHA256 = (
    "f442065172beb014ebcd4f3823abe1e2764fb2b74bbad348a8dd94d6b4c903f5"
)
NACA_PHASE_PILOT_ANALYSIS_SCHEMA = (
    "time_dependent_no.su2_naca0012_phase_pilot_analysis.v1"
)
NACA_PHASE_PILOT_RECEIPT_FILENAME = "phase_pilot_analysis.json"

PHASE_PILOT_OUTCOMES = (
    "PHASE_ONLY_SUPPORTED",
    "INSUFFICIENT_CYCLES",
    "TRANSIENT_NOT_SETTLED",
    "MODULATED_LOW_DIMENSIONAL",
    "NO_COHERENT_PERIOD",
    "INVALID_ARTIFACT",
)
_GATED_VIEWS = ("uniform_node", "near_body_wake_uniform_node")
_ALL_VIEWS = (*_GATED_VIEWS, "physical_vertex_area")
_CLAIM_BOUNDARY = {
    "solver_execution_authorized_by_this_analysis": False,
    "phase_population_claimed": False,
    "independent_trajectory_claimed": False,
    "pcno_training_claimed": False,
    "sealed_access_claimed": False,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _file_record(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"required regular file is absent or aliased: {path}")
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _verified_self_hashed_json(path: Path, schema: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"required JSON is absent or aliased: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != schema:
        raise ValueError(f"{path}: unsupported schema")
    observed = payload.pop("canonical_payload_sha256", None)
    expected = sha256(_canonical_bytes(payload)).hexdigest()
    if observed != expected:
        raise ValueError(f"{path}: canonical payload SHA256 differs")
    payload["canonical_payload_sha256"] = observed
    return payload


def _write_atomic_self_hashed_json(
    path: Path, payload: Mapping[str, Any]
) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(path)
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise ValueError("analysis receipt parent is absent or aliased")
    rendered = dict(payload)
    rendered["canonical_payload_sha256"] = sha256(
        _canonical_bytes(rendered)
    ).hexdigest()
    temporary = path.with_name(f".{path.name}.{uuid4()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(rendered, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return rendered


def _source_inventory() -> dict[str, dict[str, Any]]:
    repository = Path(__file__).resolve().parents[2]
    sources = (
        Path(__file__).resolve(),
        repository / "scripts/time_dependent_no/analyze_su2_naca0012_phase_pilot.py",
        repository / "utility/time_dependent_no/su2_naca_trajectory.py",
        repository / "utility/time_dependent_no/su2_native_replay.py",
        repository / "utility/time_dependent_no/su2_restart_contract.py",
    )
    return {str(path.relative_to(repository)): _file_record(path) for path in sources}


def _load_contract(path: Path) -> dict[str, Any]:
    record = _file_record(path)
    if record["sha256"] != NACA_PHASE_PILOT_CONTRACT_SHA256:
        raise ValueError("phase-pilot contract differs from the frozen SHA256")
    contract = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(contract, dict)
        or contract.get("schema") != NACA_PHASE_PILOT_CONTRACT_SCHEMA
    ):
        raise ValueError("unsupported phase-pilot contract schema")
    if contract["input_contract"]["dynamic_fields_by_name"] != list(
        NACA_DYNAMIC_FIELDS
    ):
        raise ValueError("phase-pilot dynamic-field contract differs")
    if contract["five_field_recurrence"]["weight_views"] != list(_ALL_VIEWS):
        raise ValueError("phase-pilot metric views differ")
    if contract["five_field_recurrence"]["gated_views"] != list(_GATED_VIEWS):
        raise ValueError("phase-pilot gated views differ")
    if tuple(contract["outcomes"]) != PHASE_PILOT_OUTCOMES:
        raise ValueError("phase-pilot outcomes differ")
    return contract


def _same_file_record(
    live: Mapping[str, Any], expected: Mapping[str, Any], label: str
) -> None:
    for key in ("file", "bytes", "sha256"):
        if live.get(key) != expected.get(key):
            raise ValueError(f"{label} differs from its bound file record")


def _parse_history(
    path: Path, indices: Sequence[int]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="strict", newline="") as handle:
        reader = csv.reader(handle, skipinitialspace=True)
        try:
            header = tuple(item.strip() for item in next(reader))
        except StopIteration as error:
            raise ValueError("trajectory history is empty") from error
        if header != NACA_HISTORY_FIELDS:
            raise ValueError("trajectory history has an unsupported field schema")
        columns = {name: [] for name in header}
        for line_number, row in enumerate(reader, start=2):
            values = [item.strip() for item in row]
            if len(values) != len(header):
                raise ValueError(f"history row {line_number} has the wrong width")
            for name, value in zip(header, values, strict=True):
                try:
                    number = float(value)
                except ValueError as error:
                    raise ValueError(
                        f"history row {line_number} is nonnumeric"
                    ) from error
                if not math.isfinite(number):
                    raise ValueError(f"history row {line_number} is nonfinite")
                columns[name].append(number)
    observed = np.asarray(columns["Time_Iter"], dtype=np.float64)
    inner = np.asarray(columns["Inner_Iter"], dtype=np.float64)
    if not np.all(observed == np.floor(observed)) or not np.all(
        inner == np.floor(inner)
    ):
        raise ValueError("history iteration columns must be integers")
    if observed.astype(np.int64).tolist() != list(indices):
        raise ValueError("history indices differ from the restart prefix")
    if np.any(inner < 0.0):
        raise ValueError("history contains a negative inner iteration")
    signals = {
        key: np.asarray(columns[key], dtype=np.float64)
        for key in ("CL", "CD", "CFx", "CFy")
    }
    final = {
        key: int(columns[key][-1])
        if key in ("Time_Iter", "Inner_Iter")
        else columns[key][-1]
        for key in ("Time_Iter", "Inner_Iter", "CD", "CL", "CFx", "CFy")
    }
    record = {
        **_file_record(path),
        "schema": list(header),
        "row_count": len(indices),
        "first_time_iter": indices[0],
        "last_time_iter": indices[-1],
        "final_row": final,
    }
    return signals, record


def _load_verified_packet(
    *, receipt_path: Path, mesh_path: Path, history_path: Path, state_path: Path
) -> tuple[np.memmap, np.ndarray, dict[str, Any], dict[str, Any]]:
    if receipt_path.parent.is_symlink():
        raise ValueError("trajectory packet directory is aliased")
    packet_root = receipt_path.parent.resolve()
    if receipt_path.name != NACA_TRAJECTORY_RECEIPT_FILENAME:
        raise ValueError("trajectory receipt has an unexpected filename")
    if history_path.name != NACA_TRAJECTORY_HISTORY_FILENAME:
        raise ValueError("trajectory history has an unexpected filename")
    if (
        mesh_path.resolve().parent != packet_root
        or history_path.resolve().parent != packet_root
    ):
        raise ValueError("mesh and history must be inside the trajectory packet")
    receipt = load_verified_trajectory_receipt(receipt_path)
    if (
        receipt.get("status") != "execution_and_validation_succeeded"
        or receipt.get("trajectory_validated") is not True
        or receipt.get("execution_succeeded") is not True
    ):
        raise ValueError("trajectory receipt is not successfully validated")
    if any(
        receipt.get(key) is not False
        for key in (
            "trajectory_population_claimed",
            "id_population_claimed",
            "boundary_and_force_gate_complete",
        )
    ):
        raise ValueError("trajectory receipt exceeds the allowed claim boundary")

    trajectory_contract = receipt.get("trajectory_contract")
    if not isinstance(trajectory_contract, dict):
        raise TypeError("trajectory receipt lacks its output contract")
    first = int(trajectory_contract.get("generated_first_index", -1))
    last = int(trajectory_contract.get("generated_last_index", -1))
    count = int(trajectory_contract.get("generated_count", -1))
    if (
        first != 499
        or last < 1999
        or last > 6499
        or (last - 1999) % 500 != 0
        or count != last - first + 1
        or trajectory_contract.get("output_every_time_step") is not True
        or trajectory_contract.get("output_stem") != NACA_TRAJECTORY_OUTPUT_STEM
        or trajectory_contract.get("restart_iter") != 499
        or trajectory_contract.get("final_time_iter") != last + 1
        or trajectory_contract.get("history_wrt_freq_inner") != 0
        or trajectory_contract.get("one_final_inner_history_row_per_time_iter")
        is not True
    ):
        raise ValueError(
            "trajectory receipt does not describe an allowed accumulated prefix"
        )
    indices = list(range(first, last + 1))

    storage_binding = receipt.get("storage_manifest")
    if not isinstance(storage_binding, dict):
        raise TypeError("trajectory receipt lacks a storage-manifest binding")
    storage_path = packet_root / NACA_TRAJECTORY_STORAGE_FILENAME
    _same_file_record(_file_record(storage_path), storage_binding, "storage manifest")
    storage = _verified_self_hashed_json(storage_path, NACA_TRAJECTORY_STORAGE_SCHEMA)
    if storage.get("status") != "validated":
        raise ValueError("trajectory storage manifest is not validated")
    output_contract = storage.get("output_contract")
    expected_output_contract = {
        "native_fields": list(NACA_NATIVE_RESTART_FIELDS),
        "dynamic_fields": list(NACA_DYNAMIC_FIELDS),
        "first_index": first,
        "last_index": last,
        "count": count,
    }
    if output_contract != expected_output_contract:
        raise ValueError("storage output contract differs from the frozen interface")
    records = storage.get("files")
    if not isinstance(records, list) or len(records) != count:
        raise ValueError("storage manifest has the wrong restart record count")
    observed_files = sorted(packet_root.glob(f"{NACA_TRAJECTORY_OUTPUT_STEM}_*.dat"))
    expected_names = [
        f"{NACA_TRAJECTORY_OUTPUT_STEM}_{index:05d}.dat" for index in indices
    ]
    if [path.name for path in observed_files] != expected_names:
        raise ValueError(
            "live trajectory restart set differs from the accumulated prefix"
        )

    case_contract = receipt.get("prepared_case", {}).get("case_contract", {})
    mesh_binding = case_contract.get("staged_inputs", {}).get("mesh")
    if not isinstance(mesh_binding, dict):
        raise TypeError("trajectory receipt lacks its staged mesh binding")
    _same_file_record(_file_record(mesh_path), mesh_binding, "mesh")
    mesh = parse_su2_mesh(mesh_path)
    if mesh.dimension != 2:
        raise ValueError("phase pilot requires the frozen two-dimensional mesh")
    history_signals, history_record = _parse_history(history_path, indices)
    _same_file_record(history_record, receipt.get("history", {}), "history receipt")
    if history_record != storage.get("history"):
        raise ValueError("history record differs between receipt and storage manifest")

    coordinate_indices = [NACA_NATIVE_RESTART_FIELDS.index(name) for name in ("x", "y")]
    dynamic_indices = [
        NACA_NATIVE_RESTART_FIELDS.index(name) for name in NACA_DYNAMIC_FIELDS
    ]
    states = np.memmap(
        state_path,
        mode="w+",
        dtype=np.float64,
        shape=(count, mesh.num_points, len(NACA_DYNAMIC_FIELDS)),
    )
    aggregate_core: list[dict[str, Any]] = []
    total_bytes = 0
    for offset, (index, record, path) in enumerate(
        zip(indices, records, observed_files, strict=True)
    ):
        expected_name = expected_names[offset]
        if (
            not isinstance(record, dict)
            or record.get("index") != index
            or record.get("file") != expected_name
        ):
            raise ValueError("storage restart records are not the exact ordered prefix")
        live = _file_record(path)
        _same_file_record(live, record, f"restart {index}")
        restart = read_su2_binary_restart(path)
        if restart.fields != NACA_NATIVE_RESTART_FIELDS or list(
            restart.header
        ) != record.get("header"):
            raise ValueError(f"restart {index} has an unsupported native schema")
        if restart.num_points != mesh.num_points:
            raise ValueError(f"restart {index} point count differs from the mesh")
        coordinate_error = float(
            np.max(np.abs(restart.values[:, coordinate_indices] - mesh.points))
        )
        if coordinate_error > 1.0e-12 or coordinate_error != record.get(
            "coordinate_max_abs_error"
        ):
            raise ValueError(
                f"restart {index} coordinates differ from the mesh or manifest"
            )
        dynamic = restart.values[:, dynamic_indices]
        if not np.all(np.isfinite(dynamic)):
            raise ValueError(f"restart {index} contains nonfinite evolved fields")
        minimum = [float(value) for value in np.min(dynamic, axis=0)]
        maximum = [float(value) for value in np.max(dynamic, axis=0)]
        if minimum != record.get("dynamic_minimum") or maximum != record.get(
            "dynamic_maximum"
        ):
            raise ValueError(f"restart {index} ranges differ from the storage manifest")
        states[offset] = dynamic
        aggregate_core.append({"index": index, **live})
        total_bytes += int(live["bytes"])
    states.flush()

    aggregate = storage.get("aggregate")
    expected_aggregate = {
        "file_count": count,
        "first_index": first,
        "last_index": last,
        "total_bytes": total_bytes,
        "total_gibibytes": total_bytes / (1024**3),
        "ordered_file_records_sha256": sha256(
            _canonical_bytes({"files": aggregate_core})
        ).hexdigest(),
        "five_evolved_float64_bytes": count * mesh.num_points * 5 * 8,
        "bdf2_distinct_frame_semantics": {
            "staged_history_indices": [497, 498],
            "generated_indices": indices,
            "distinct_frames": count + 2,
            "transition_level_frame_tripling": False,
        },
    }
    if aggregate != expected_aggregate:
        raise ValueError("storage aggregate differs from live restart artifacts")
    if receipt.get("storage_aggregate") != aggregate:
        raise ValueError("trajectory receipt and storage aggregate differ")
    packet = {
        "trajectory_receipt": _file_record(receipt_path),
        "trajectory_receipt_payload_sha256": receipt["canonical_payload_sha256"],
        "storage_manifest": _file_record(storage_path),
        "storage_manifest_payload_sha256": storage["canonical_payload_sha256"],
        "mesh": _file_record(mesh_path),
        "history": _file_record(history_path),
        "stage0_authority_binding": receipt.get("stage0_authority_binding"),
        "first_output_index": first,
        "last_output_index": last,
        "output_count": count,
        "ordered_restart_records_sha256": aggregate["ordered_file_records_sha256"],
    }
    weights = np.vstack(
        (
            np.ones(mesh.num_points, dtype=np.float64),
            (
                (mesh.points[:, 0] >= -1.0)
                & (mesh.points[:, 0] <= 10.0)
                & (mesh.points[:, 1] >= -5.0)
                & (mesh.points[:, 1] <= 5.0)
            ).astype(np.float64),
            lumped_vertex_areas(mesh),
        )
    )
    if np.count_nonzero(weights[1]) == 0:
        raise ValueError("near-body/wake diagnostic contains no mesh nodes")
    return states, weights, history_signals, packet


def _verify_packet_end(packet_root: Path, packet: Mapping[str, Any]) -> None:
    """Re-hash the full consumed packet after analysis to close the bracket."""

    for key in ("trajectory_receipt", "storage_manifest", "mesh", "history"):
        binding = packet.get(key)
        if not isinstance(binding, dict) or not isinstance(binding.get("file"), str):
            raise TypeError(f"analysis packet lacks its {key} end binding")
        _same_file_record(
            _file_record(packet_root / binding["file"]), binding, f"end-bracket {key}"
        )
    storage_binding = packet["storage_manifest"]
    storage = _verified_self_hashed_json(
        packet_root / storage_binding["file"], NACA_TRAJECTORY_STORAGE_SCHEMA
    )
    if storage.get("canonical_payload_sha256") != packet.get(
        "storage_manifest_payload_sha256"
    ):
        raise ValueError("storage payload changed during phase analysis")
    records = storage.get("files")
    if not isinstance(records, list) or len(records) != packet.get("output_count"):
        raise ValueError("storage restart records changed during phase analysis")
    aggregate_core: list[dict[str, Any]] = []
    expected_names: list[str] = []
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("storage restart record changed type during phase analysis")
        index = int(record.get("index", -1))
        expected_name = f"{NACA_TRAJECTORY_OUTPUT_STEM}_{index:05d}.dat"
        if record.get("file") != expected_name:
            raise ValueError("storage restart filename changed during phase analysis")
        live = _file_record(packet_root / expected_name)
        _same_file_record(live, record, f"end-bracket restart {index}")
        aggregate_core.append({"index": index, **live})
        expected_names.append(expected_name)
    live_names = [
        path.name
        for path in sorted(packet_root.glob(f"{NACA_TRAJECTORY_OUTPUT_STEM}_*.dat"))
    ]
    if live_names != expected_names:
        raise ValueError("restart set changed during phase analysis")
    ordered_digest = sha256(_canonical_bytes({"files": aggregate_core})).hexdigest()
    if ordered_digest != packet.get("ordered_restart_records_sha256"):
        raise ValueError("ordered restart aggregate changed during phase analysis")


def _affine_detrend(signal: np.ndarray) -> np.ndarray:
    x = np.arange(len(signal), dtype=np.float64)
    design = np.column_stack((np.ones(len(signal)), x))
    return signal - design @ np.linalg.lstsq(design, signal, rcond=None)[0]


def _spectral_period(
    signal: np.ndarray, lag_min: int, lag_max: int
) -> tuple[float | None, float]:
    detrended = _affine_detrend(signal)
    power = np.abs(np.fft.rfft(detrended)) ** 2
    frequencies = np.fft.rfftfreq(len(detrended))
    admissible = (
        (frequencies > 0.0)
        & (1.0 / np.maximum(frequencies, 1e-300) >= lag_min)
        & (1.0 / np.maximum(frequencies, 1e-300) <= lag_max)
    )
    candidates = np.flatnonzero(admissible)
    if len(candidates) == 0 or float(np.max(power[candidates])) <= 0.0:
        return None, 0.0
    peak = int(candidates[int(np.argmax(power[candidates]))])
    return float(1.0 / frequencies[peak]), float(power[peak] / np.sum(power[1:]))


def _estimate_period(signal: np.ndarray, contract: Mapping[str, Any]) -> dict[str, Any]:
    trailing_start = len(signal) // 4
    trailing = np.asarray(signal[trailing_start:], dtype=np.float64)
    lag_min = int(contract["period_estimation"]["lag_search_minimum"])
    lag_max = len(trailing) // 4
    result: dict[str, Any] = {
        "trailing_start_offset": trailing_start,
        "trailing_sample_count": len(trailing),
        "lag_search_minimum": lag_min,
        "lag_search_maximum": lag_max,
        "coherent": False,
        "boundary_limited": lag_max < lag_min,
        "unresolved_evidence": lag_max < lag_min,
    }
    if lag_max < lag_min or float(np.std(trailing)) == 0.0:
        result["failure"] = "insufficient_nonconstant_CL_for_period_search"
        result["unresolved_evidence"] = lag_max < lag_min
        return result
    detrended = _affine_detrend(trailing)
    full_power = np.abs(np.fft.rfft(detrended)) ** 2
    full_frequencies = np.fft.rfftfreq(len(detrended))
    unrestricted_peak = int(np.argmax(full_power[1:])) + 1
    unrestricted_period = float(1.0 / full_frequencies[unrestricted_peak])
    result["unrestricted_spectral_peak_period_steps"] = unrestricted_period
    acf = np.empty(lag_max + 1, dtype=np.float64)
    acf[0] = 1.0
    for lag in range(1, lag_max + 1):
        left = detrended[:-lag]
        right = detrended[lag:]
        denominator = math.sqrt(float(np.dot(left, left) * np.dot(right, right)))
        acf[lag] = (
            float(np.dot(left, right) / denominator) if denominator > 0.0 else 0.0
        )
    zero_crossings = np.flatnonzero((acf[:-1] > 0.0) & (acf[1:] <= 0.0))
    acf_period: int | None = None
    if len(zero_crossings):
        for lag in range(max(lag_min, int(zero_crossings[0]) + 2), lag_max):
            if acf[lag] >= 0.8 and acf[lag] >= acf[lag - 1] and acf[lag] > acf[lag + 1]:
                acf_period = lag
                break
    spectral_period, spectral_fraction = _spectral_period(trailing, lag_min, lag_max)
    result.update(
        {
            "period_acf_steps": acf_period,
            "acf_at_candidate": float(acf[acf_period]) if acf_period else None,
            "period_spectral_steps": spectral_period,
            "spectral_peak_power_fraction": spectral_fraction,
        }
    )
    if acf_period is None or spectral_period is None:
        result["boundary_limited"] = bool(unrestricted_period >= 0.9 * lag_max)
        result["failure"] = "missing_ACF_or_spectral_period_candidate"
        result["unresolved_evidence"] = True
        return result
    agreement = abs(acf_period - spectral_period) / acf_period
    result["acf_spectral_relative_difference"] = agreement
    if agreement > 0.1:
        result["boundary_limited"] = max(acf_period, spectral_period) >= 0.9 * lag_max
        result["unresolved_evidence"] = result["boundary_limited"]
        result["failure"] = "ACF_and_spectral_periods_disagree"
        return result

    target_width = acf_period / 50.0
    lower = max(3, math.floor(target_width))
    if lower % 2 == 0:
        lower -= 1
    upper = max(3, lower + 2)
    width = upper if abs(upper - target_width) <= abs(target_width - lower) else lower
    padded = np.pad(trailing, (width // 2, width // 2), mode="edge")
    smoothed = np.convolve(padded, np.ones(width) / width, mode="valid")
    level = float(np.median(smoothed))
    crossings: list[float] = []
    for offset in np.flatnonzero((smoothed[:-1] < level) & (smoothed[1:] >= level)):
        delta = smoothed[offset + 1] - smoothed[offset]
        fraction = (level - smoothed[offset]) / delta if delta != 0.0 else 0.0
        crossings.append(trailing_start + float(offset) + float(fraction))
    intervals = np.diff(crossings)
    if len(intervals) < 2:
        result["failure"] = "too_few_admissible_CL_crossings"
        result["unresolved_evidence"] = True
        return result
    if np.any((intervals < 0.5 * acf_period) | (intervals > 1.5 * acf_period)):
        result["failure"] = "CL_crossing_separation_outside_frozen_range"
        return result
    period = float(np.median(intervals))
    mad_ratio = float(np.median(np.abs(intervals - period)) / period)
    period_vs_acf = abs(period - acf_period) / acf_period
    result.update(
        {
            "smoothing_width": width,
            "crossing_offsets": crossings,
            "admissible_crossing_interval_count": len(intervals),
            "period_steps": period,
            "period_mad_over_median": mad_ratio,
            "period_vs_acf_relative_difference": period_vs_acf,
        }
    )
    if mad_ratio > 0.02 or period_vs_acf > 0.1:
        result["failure"] = "CL_crossing_period_is_not_coherent"
        return result
    result["coherent"] = True
    return result


def _weighted_rms(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted RMS over phase and node axes, retaining the field axis."""

    if values.ndim != 3 or weights.shape != (values.shape[1],):
        raise ValueError("weighted recurrence inputs have incompatible shapes")
    selected = weights > 0.0
    if not np.any(selected):
        raise ValueError("metric view contains no positive weights")
    energy = np.einsum(
        "knf,n,knf->f", values[:, selected], weights[selected], values[:, selected]
    )
    return np.sqrt(energy / (values.shape[0] * float(np.sum(weights[selected]))))


def _resample_cycle(
    array: np.ndarray, start: float, stop: float, samples: int
) -> np.ndarray:
    positions = start + (stop - start) * np.arange(samples, dtype=np.float64) / samples
    lower = np.floor(positions).astype(np.int64)
    upper = np.minimum(lower + 1, len(array) - 1)
    fraction = positions - lower
    shape = (samples,) + (1,) * (array.ndim - 1)
    return array[lower] * (1.0 - fraction.reshape(shape)) + array[
        upper
    ] * fraction.reshape(shape)


def _scalar_bank(
    states: np.ndarray, history: Mapping[str, np.ndarray]
) -> tuple[np.ndarray, list[str]]:
    means = np.empty((len(states), states.shape[2]), dtype=np.float64)
    rms = np.empty_like(means)
    for start in range(0, len(states), 16):
        chunk = np.asarray(states[start : start + 16])
        chunk_mean = np.mean(chunk, axis=1)
        means[start : start + len(chunk)] = chunk_mean
        rms[start : start + len(chunk)] = np.sqrt(
            np.mean((chunk - chunk_mean[:, None, :]) ** 2, axis=1)
        )
    bank = np.column_stack((history["CL"], history["CD"], means, rms))
    names = (
        ["CL", "CD"]
        + [f"mean_{name}" for name in NACA_DYNAMIC_FIELDS]
        + [f"rms_{name}" for name in NACA_DYNAMIC_FIELDS]
    )
    return bank, names


@cache
def _theil_sen_pairs(length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left, right = np.triu_indices(length, 1)
    return left, right, (right - left).astype(np.float64)


def _stationarity(
    states: np.ndarray,
    history: Mapping[str, np.ndarray],
    crossings: Sequence[float],
    period: float,
) -> dict[str, Any]:
    boundaries = np.asarray(crossings, dtype=np.float64)
    intervals = np.diff(boundaries)
    keep = (intervals >= 0.5 * period) & (intervals <= 1.5 * period)
    cycles = [(boundaries[i], boundaries[i + 1]) for i in np.flatnonzero(keep)]
    scalar, names = _scalar_bank(states, history)
    templates = np.asarray([_resample_cycle(scalar, a, b, 64) for a, b in cycles])
    candidates: list[dict[str, Any]] = []
    for burn in range(max(0, len(cycles) - 7)):
        suffix = templates[burn:]
        if len(suffix) < 8:
            continue
        split = len(suffix) // 2
        first = np.mean(suffix[:split], axis=0)
        second = np.mean(suffix[split:], axis=0)
        combined = np.mean(suffix, axis=0)
        within = np.sqrt(np.mean((combined - np.mean(combined, axis=0)) ** 2, axis=0))
        full_rms = np.sqrt(np.mean(scalar * scalar, axis=0))
        floor = 1.0e-8 * full_rms
        normalizer = np.maximum(within, floor)
        floor_hits = normalizer == floor
        safe = np.where(normalizer > 0.0, normalizer, 1.0)
        shift = np.sqrt(np.mean((first - second) ** 2, axis=0)) / safe
        cycle_means = np.mean(suffix, axis=1)
        cycle_amplitudes = np.sqrt(
            np.mean(
                (suffix - cycle_means[:, None, :]) ** 2,
                axis=1,
            )
        )
        left, right, separation = _theil_sen_pairs(len(suffix))
        mean_slopes = (cycle_means[right] - cycle_means[left]) / separation[:, None]
        amplitude_slopes = (
            cycle_amplitudes[right] - cycle_amplitudes[left]
        ) / separation[:, None]
        span = len(suffix) - 1
        mean_drift = np.abs(np.median(mean_slopes, axis=0)) * span / safe
        amplitude_drift = np.abs(np.median(amplitude_slopes, axis=0)) * span / safe
        drift = np.maximum(mean_drift, amplitude_drift)
        suffix_intervals = np.asarray([b - a for a, b in cycles[burn:]])
        median_interval = float(np.median(suffix_intervals))
        period_mad = float(
            np.median(np.abs(suffix_intervals - median_interval)) / median_interval
        )
        maximum_shift = float(np.max(shift))
        maximum_drift = float(np.max(drift))
        score = max(maximum_shift / 0.05, maximum_drift / 0.05, period_mad / 0.02)
        passed = bool(
            np.all(shift <= 0.05) and np.all(drift <= 0.05) and period_mad <= 0.02
        )
        candidates.append(
            {
                "burn_cycle": burn,
                "burn_offset": cycles[burn][0],
                "complete_cycles_after_burn": len(suffix),
                "template_shift_by_signal": dict(
                    zip(names, shift.tolist(), strict=True)
                ),
                "theil_sen_total_drift_by_signal": dict(
                    zip(names, drift.tolist(), strict=True)
                ),
                "period_mad_over_median": period_mad,
                "maximum_normalized_template_shift": maximum_shift,
                "maximum_normalized_theil_sen_total_drift": maximum_drift,
                "normalizer_floor_hit_signals": [
                    name for name, hit in zip(names, floor_hits, strict=True) if hit
                ],
                "gate_score": score,
                "passed": passed,
            }
        )
        if passed:
            return {
                "passed": True,
                "selected": candidates[-1],
                "cycles": cycles[burn:],
                "candidate_count": len(candidates),
            }
    tail = candidates[-3:]
    tracked = (
        "maximum_normalized_template_shift",
        "maximum_normalized_theil_sen_total_drift",
        "period_mad_over_median",
    )
    improving = bool(
        len(tail) >= 3
        and all(
            all(right[key] <= left[key] for key in tracked)
            for left, right in pairwise(tail)
        )
        and any(tail[-1][key] < tail[0][key] for key in tracked)
    )
    return {
        "passed": False,
        "available_complete_cycles": len(cycles),
        "candidate_count": len(candidates),
        "last_candidate": candidates[-1] if candidates else None,
        "late_candidate_summaries": tail,
        "late_metrics_improve_monotonically": improving,
    }


def _alias_scan(states: np.ndarray, period: float) -> dict[str, Any]:
    start = len(states) // 4
    tail = states[start:]
    node_chunk = 128
    variance_sum = np.zeros(tail.shape[2], dtype=np.float64)
    for node_start in range(0, tail.shape[1], node_chunk):
        chunk = np.asarray(tail[:, node_start : node_start + node_chunk, :])
        centered = chunk - np.mean(chunk, axis=0, keepdims=True)
        variance_sum += np.sum(np.mean(centered * centered, axis=0), axis=0)
    variation = np.sqrt(variance_sum / tail.shape[1])
    zero_variation = variation == 0.0
    scales = np.where(zero_variation, 1.0, variation)
    normalization = {
        name: {
            "uniform_node_RMS_temporal_variation": float(variation[field]),
            "zero_variation_contributes_zero": bool(zero_variation[field]),
        }
        for field, name in enumerate(NACA_DYNAMIC_FIELDS)
    }
    low = max(1, math.ceil(0.8 * period))
    high = min(len(tail) - 1, math.floor(1.2 * period))
    lags = np.arange(low, high + 1, dtype=np.int64)
    component_error = np.zeros((len(lags), tail.shape[2]), dtype=np.float64)
    nfft = 1 << (2 * len(tail) - 1).bit_length()
    for field in range(tail.shape[2]):
        if zero_variation[field]:
            continue
        energy_by_time = np.zeros(len(tail), dtype=np.float64)
        correlation = np.zeros(high + 1, dtype=np.float64)
        for node_start in range(0, tail.shape[1], node_chunk):
            chunk = (
                np.asarray(tail[:, node_start : node_start + node_chunk, field])
                / scales[field]
            )
            energy_by_time += np.sum(chunk * chunk, axis=1)
            transform = np.fft.rfft(chunk, n=nfft, axis=0)
            correlation += np.sum(
                np.fft.irfft(transform.conj() * transform, n=nfft, axis=0)[: high + 1],
                axis=1,
            )
        prefix = np.concatenate(([0.0], np.cumsum(energy_by_time)))
        pair_count = len(tail) - lags
        squared = (
            prefix[pair_count] + prefix[-1] - prefix[lags] - 2.0 * correlation[lags]
        )
        squared = np.maximum(squared, 0.0)
        component_error[:, field] = np.sqrt(squared / (pair_count * tail.shape[1]))
    balanced = np.sqrt(np.mean(component_error * component_error, axis=1))
    values = {
        str(int(lag)): float(value) for lag, value in zip(lags, balanced, strict=True)
    }
    component_values = {
        name: {
            str(int(lag)): float(value)
            for lag, value in zip(lags, component_error[:, field], strict=True)
        }
        for field, name in enumerate(NACA_DYNAMIC_FIELDS)
    }
    if not values:
        return {
            "passed": False,
            "failure": "empty_state_alias_scan",
            "normalization": normalization,
            "component_scan": component_values,
            "scan": values,
        }
    best = min((float(value), int(lag)) for lag, value in values.items())[1]
    distance = abs(best - period) / period
    return {
        "passed": distance <= 0.05,
        "period_minimizer_steps": best,
        "minimizer_relative_distance_from_period": distance,
        "all_available_tail_pairs_per_lag": True,
        "normalization": normalization,
        "component_scan": component_values,
        "scan": values,
    }


def _cycle_recurrence(
    states: np.ndarray, cycles: Sequence[tuple[float, float]], weights: np.ndarray
) -> dict[str, Any]:
    count = len(cycles)
    template = np.zeros((64, states.shape[1], states.shape[2]), dtype=np.float64)
    for start, stop in cycles:
        template += _resample_cycle(states, start, stop, 64) / count
    views: dict[str, Any] = {}
    for view_index, view in enumerate(_ALL_VIEWS):
        weight = weights[view_index]
        orbit_mean = np.mean(template, axis=0, keepdims=True)
        orbit_amplitude = _weighted_rms(template - orbit_mean, weight)
        state_rms = _weighted_rms(template, weight)
        use_state_rms = orbit_amplitude / np.maximum(state_rms, 1e-300) < 1.0e-6
        normalizer = np.where(use_state_rms, state_rms, orbit_amplitude)
        normalizer = np.where(normalizer > 0.0, normalizer, 1.0)
        adjacent_components: list[np.ndarray] = []
        adjacent_native: list[np.ndarray] = []
        template_components: list[float] = []
        previous: np.ndarray | None = None
        for start, stop in cycles:
            current = _resample_cycle(states, start, stop, 64)
            residual = _weighted_rms(current - template, weight) / normalizer
            template_components.append(float(np.sqrt(np.mean(residual * residual))))
            if previous is not None:
                native = _weighted_rms(current - previous, weight)
                adjacent_native.append(native)
                adjacent_components.append(native / normalizer)
            previous = current
        components = np.asarray(adjacent_components)
        native = np.asarray(adjacent_native)
        balanced = np.sqrt(np.mean(components * components, axis=1))
        q90_fields = np.quantile(components, 0.9, axis=0, method="linear")
        metrics = {
            "normalization": {
                name: {
                    "orbit_amplitude": float(orbit_amplitude[j]),
                    "state_rms": float(state_rms[j]),
                    "used_state_rms_fallback": bool(use_state_rms[j]),
                    "selected": float(normalizer[j]),
                }
                for j, name in enumerate(NACA_DYNAMIC_FIELDS)
            },
            "adjacent_native_unit_numerator": {
                name: native[:, j].tolist()
                for j, name in enumerate(NACA_DYNAMIC_FIELDS)
            },
            "median_component_balanced": float(np.median(balanced)),
            "q90_component_balanced": float(
                np.quantile(balanced, 0.9, method="linear")
            ),
            "q90_each_field": dict(
                zip(NACA_DYNAMIC_FIELDS, q90_fields.tolist(), strict=True)
            ),
            "q90_cycle_to_template": float(
                np.quantile(template_components, 0.9, method="linear")
            ),
        }
        metrics["passed"] = bool(
            metrics["median_component_balanced"] <= 0.05
            and metrics["q90_component_balanced"] <= 0.1
            and np.all(q90_fields <= 0.15)
            and metrics["q90_cycle_to_template"] <= 0.1
        )
        views[view] = metrics
    return {
        "views": views,
        "gated_views": list(_GATED_VIEWS),
        "physical_vertex_area_can_qualify_alone": False,
        "passed": all(views[view]["passed"] for view in _GATED_VIEWS),
    }


def _cd_harmonic_diagnostic(cd: np.ndarray, period: float) -> dict[str, Any]:
    trailing = np.asarray(cd[len(cd) // 4 :], dtype=np.float64)
    candidate, power_fraction = _spectral_period(
        trailing, max(2, int(0.35 * period)), max(3, int(1.2 * period))
    )
    half_period_harmonic = bool(
        candidate is not None and abs(candidate - 0.5 * period) / period <= 0.1
    )
    return {
        "period_spectral_steps": candidate,
        "spectral_peak_power_fraction": power_fraction,
        "half_period_harmonic_detected": half_period_harmonic,
        "role": "reported_only_CL_remains_the_phase_clock",
        "can_change_CL_period": False,
        "passed": True,
    }


def _extension_request(
    *, last_index: int, period: float | None, reason: str
) -> dict[str, Any] | None:
    if last_index >= 6499:
        return None
    if period is None:
        append = 1500
        rule = "unresolved_or_boundary_limited_append_1500"
    else:
        raw = max(1000, math.ceil(4.0 * period))
        append = math.ceil(raw / 500.0) * 500
        rule = "known_period_append_max_1000_or_four_periods_rounded_to_500"
    requested = min(6499, last_index + append)
    return {
        "reason": reason,
        "rule": rule,
        "current_final_output_index": last_index,
        "append_steps": requested - last_index,
        "requested_final_output_index": requested,
        "same_accumulated_prefix_required": True,
        "model_blind": True,
    }


def _analyze_arrays(
    *,
    states: np.ndarray,
    weights: np.ndarray,
    history: Mapping[str, np.ndarray],
    contract: Mapping[str, Any],
    last_index: int,
) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
    period = _estimate_period(history["CL"], contract)
    diagnostics: dict[str, Any] = {"period": period}
    if not period["coherent"]:
        extendable = bool(
            (period["boundary_limited"] or period["unresolved_evidence"])
            and last_index < 6499
        )
        outcome = "INSUFFICIENT_CYCLES" if extendable else "NO_COHERENT_PERIOD"
        extension = (
            _extension_request(
                last_index=last_index, period=None, reason=period["failure"]
            )
            if extendable
            else None
        )
        return outcome, diagnostics, extension
    period_steps = float(period["period_steps"])
    cd_diagnostic = _cd_harmonic_diagnostic(history["CD"], period_steps)
    alias = _alias_scan(states, period_steps)
    diagnostics.update(
        {
            "CD_harmonic_diagnostic": cd_diagnostic,
            "state_recurrence_alias_scan": alias,
        }
    )
    if not alias["passed"]:
        return "NO_COHERENT_PERIOD", diagnostics, None
    stationarity = _stationarity(
        states, history, period["crossing_offsets"], period_steps
    )
    diagnostics["stationarity"] = {
        key: value for key, value in stationarity.items() if key != "cycles"
    }
    if not stationarity["passed"]:
        available = int(stationarity["available_complete_cycles"])
        if available < 8:
            extension = _extension_request(
                last_index=last_index,
                period=period_steps,
                reason="fewer_than_eight_complete_postburn_cycles",
            )
            return "INSUFFICIENT_CYCLES", diagnostics, extension
        if stationarity["late_metrics_improve_monotonically"]:
            extension = _extension_request(
                last_index=last_index,
                period=period_steps,
                reason="late_stationarity_metrics_improve_monotonically",
            )
            return "TRANSIENT_NOT_SETTLED", diagnostics, extension
        return "MODULATED_LOW_DIMENSIONAL", diagnostics, None
    cycles = stationarity["cycles"]
    recurrence = _cycle_recurrence(states, cycles, weights)
    diagnostics["five_field_recurrence"] = recurrence
    if not recurrence["passed"]:
        return "MODULATED_LOW_DIMENSIONAL", diagnostics, None
    return "PHASE_ONLY_SUPPORTED", diagnostics, None


def analyze_su2_naca0012_phase_pilot(
    *,
    contract_path: str | Path,
    trajectory_receipt_path: str | Path,
    mesh_path: str | Path,
    history_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Validate one accumulated solver prefix and atomically emit its decision."""

    paths = {
        "contract": Path(contract_path),
        "trajectory_receipt": Path(trajectory_receipt_path),
        "mesh": Path(mesh_path),
        "history": Path(history_path),
        "output": Path(output_path),
    }
    for name, path in paths.items():
        if not path.is_absolute():
            raise ValueError(f"{name}_path must be absolute")
    if paths["output"].parent.is_symlink():
        raise ValueError("output_path parent must not be aliased")
    output = paths["output"].resolve()
    if output.name != NACA_PHASE_PILOT_RECEIPT_FILENAME:
        raise ValueError(f"output_path must end in {NACA_PHASE_PILOT_RECEIPT_FILENAME}")
    if output.exists():
        raise FileExistsError(output)

    started = _utc_now()
    error: str | None = None
    packet: dict[str, Any] | None = None
    diagnostics: dict[str, Any] = {}
    extension: dict[str, Any] | None = None
    outcome = "INVALID_ARTIFACT"
    source_at_start: dict[str, dict[str, Any]] | None = None
    source_at_end: dict[str, Any] | None = None
    contract_record: dict[str, Any] | None = None
    try:
        for name in ("contract", "trajectory_receipt", "mesh", "history"):
            _file_record(paths[name])
        source_at_start = _source_inventory()
        contract = _load_contract(paths["contract"].resolve())
        contract_record = _file_record(paths["contract"].resolve())
        with tempfile.TemporaryDirectory(
            prefix=".naca_phase_pilot_", dir=output.parent
        ) as temporary:
            state_path = Path(temporary) / "states.float64.memmap"
            states: np.memmap | None = None
            try:
                states, weights, history, packet = _load_verified_packet(
                    receipt_path=paths["trajectory_receipt"].resolve(),
                    mesh_path=paths["mesh"].resolve(),
                    history_path=paths["history"].resolve(),
                    state_path=state_path,
                )
                outcome, diagnostics, extension = _analyze_arrays(
                    states=states,
                    weights=weights,
                    history=history,
                    contract=contract,
                    last_index=int(packet["last_output_index"]),
                )
            finally:
                if states is not None:
                    states.flush()
                    states._mmap.close()  # Close mapping before Windows removes the file.
        _verify_packet_end(paths["trajectory_receipt"].resolve().parent, packet)
        if _file_record(paths["contract"].resolve()) != contract_record:
            raise ValueError("phase-pilot contract changed during analysis")
    except Exception as caught:  # noqa: BLE001 - invalid evidence gets a receipt
        error = f"{type(caught).__name__}: {caught}"
        outcome = "INVALID_ARTIFACT"
        diagnostics = {}
        extension = None
    if source_at_start is not None:
        provenance_error: str | None = None
        try:
            source_at_end = _source_inventory()
        except Exception as caught:  # noqa: BLE001 - receipt preserves the failure
            provenance_error = f"{type(caught).__name__}: {caught}"
            source_at_end = {"inventory_error": provenance_error}
        else:
            if source_at_end != source_at_start:
                provenance_error = (
                    "ValueError: phase-pilot source changed during analysis"
                )
        if provenance_error is not None:
            error = f"{error}; {provenance_error}" if error else provenance_error
            outcome = "INVALID_ARTIFACT"
            diagnostics = {}
            extension = None
    receipt = {
        "schema": NACA_PHASE_PILOT_ANALYSIS_SCHEMA,
        "created_at_utc": _utc_now(),
        "analysis_started_at_utc": started,
        "outcome": outcome,
        "error": error,
        "contract": contract_record,
        "packet": packet,
        "provenance_bracket": {
            "source_at_start": source_at_start,
            "source_at_end": source_at_end,
        },
        "runtime_identity": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "byteorder": sys.byteorder,
        },
        "diagnostics": diagnostics,
        "extension_request": extension,
        "metric_view_roles": {
            "uniform_node": "primary_gated",
            "near_body_wake_uniform_node": "required_gated_diagnostic",
            "physical_vertex_area": "required_reported_secondary_not_sufficient_alone",
        },
        "native_solver_executed": False,
        "PCNO_inputs_consumed": False,
        **_CLAIM_BOUNDARY,
    }
    return _write_atomic_self_hashed_json(output, receipt)


def load_verified_phase_pilot_receipt(path: str | Path) -> dict[str, Any]:
    """Load an analysis receipt and verify its schema and self hash."""

    return _verified_self_hashed_json(Path(path), NACA_PHASE_PILOT_ANALYSIS_SCHEMA)
