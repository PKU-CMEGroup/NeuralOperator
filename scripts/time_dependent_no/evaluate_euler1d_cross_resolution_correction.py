"""Evaluate W26-L5-E1D-P0 on the frozen D030 common-source family.

Calibration and held-out evaluation are separate CLI phases.  The evaluator
constructs every 128/256/512 input from one 512-cell reference state and never
uses independently diverged rollouts for the teacher-forced gate.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__:
    from scripts.time_dependent_no.evaluate_euler1d_flow_map_frontier import (
        _predict_model_batch_state,
        load_frozen_residual_checkpoint,
        pressure_front_top2_metrics_np,
    )
else:
    from evaluate_euler1d_flow_map_frontier import (  # type: ignore[no-redef]
        _predict_model_batch_state,
        load_frozen_residual_checkpoint,
        pressure_front_top2_metrics_np,
    )

from utility.time_dependent_no.euler1d_cross_resolution_correction import (
    DENOMINATOR_FLOOR,
    MAXIMUM_RELATIVE_CORRECTION,
    PHYSICAL_COMPONENT_SCALES,
    CommonSourceStates,
    CosineProjector1D,
    Euler1DResolutionContract,
    NativeIncrementBasis1D,
    ScalarFitRecord,
    ScoreRecord,
    build_cosine_projector,
    component_integrals,
    corrections_for_coefficient,
    crossfit_case_groups,
    fit_case_first_scalar,
    mapped_increment_basis,
    prolong_piecewise_constant,
    relation_from_fit_records,
    restrict_cell_averages,
    score_records,
    validate_record_inventory,
    weighted_scaled_inner,
    weighted_scaled_rms,
)
from utility.time_dependent_no.euler1d_data import (
    Euler1DNPZ,
    conservative_to_primitive_np,
    load_euler1d_npz,
    primitive_to_conservative_np,
)

SCHEMA = "w26_l5_e1d_coefficient_transfer_v1"
CONTRACT = Euler1DResolutionContract()
ALL_CALLS = tuple(range(100))
EARLY_CALLS = tuple(range(50))
LATE_CALLS = tuple(range(50, 100))
PRIMARY_MODES = tuple(range(1, 8))
VIEW_MODES = {
    "constant": (0,),
    "low_k1_k7": PRIMARY_MODES,
    "transition_k8_k31": tuple(range(8, 32)),
    "local_k32_k255": tuple(range(32, 256)),
}
COMPONENT_NAMES = ("density", "momentum", "energy")
FIXED_COEFFICIENT = -0.5
NO_HARM_RATIO = 1.05
MINIMUM_LOW_SKILL = 0.05
MINIMUM_CASE_WINS = 12
MAXIMUM_RELATIVE_IQR = 0.5
SOURCE_PATHS = (
    "docs/time_dependent_no/W26_L5_EULER1D_COEFFICIENT_TRANSFER_PREREGISTRATION.md",
    "utility/time_dependent_no/euler1d_cross_resolution_correction.py",
    "scripts/time_dependent_no/evaluate_euler1d_cross_resolution_correction.py",
    "tests/time_dependent_no/test_euler1d_cross_resolution_correction.py",
    "scripts/time_dependent_no/evaluate_euler1d_flow_map_frontier.py",
    "scripts/time_dependent_no/train_euler1d_target_ladder.py",
    "utility/time_dependent_no/euler1d.py",
    "utility/time_dependent_no/euler1d_data.py",
    "utility/time_dependent_no/euler1d_models.py",
    "utility/time_dependent_no/euler1d_targets.py",
    "utility/time_dependent_no/fv.py",
    "baselines/fno.py",
)
EXPECTED_HASHES = {
    "fine_data": "3a4bcb41079480b88d998a9f503537b6e729e92166fd16b88219c92c3011957d",
    "coarse_data": "dcfc9f181082a0dba9f92032306364586ba7b870be3e7fd46e8e17378adf0230",
    "native_data": "a80c2dfb23632eaeedf402513f5a5acdff3b062540986ee27219917f423c9479",
    "checkpoint": "5040219ecc2a53d1e5db560b5261cc202cad983e3c87db35852ae178095efc8d",
    "family_contract": "bb17c9e953e44f3a484f406e2858a1f40e89e242745370e2529a6136720ccddb",
}
EXPECTED_VALIDATION_CASES = (
    17,
    32,
    45,
    104,
    108,
    113,
    123,
    141,
    174,
    259,
    321,
    370,
    409,
    440,
    479,
    507,
)
EXPECTED_TEST_CASES = (
    21,
    55,
    114,
    133,
    139,
    153,
    183,
    190,
    196,
    234,
    238,
    276,
    303,
    348,
    418,
    504,
)


@dataclass(frozen=True)
class CollectedSnapshot:
    case_id: str
    numeric_case_id: int
    input_call: int
    record: ScalarFitRecord
    basis: NativeIncrementBasis1D
    current_native: np.ndarray
    truth_next_native: np.ndarray
    raw_next_native: np.ndarray
    coarse_error_on_native: np.ndarray
    native_error: np.ndarray
    fine_error_on_native: np.ndarray
    shock_mask: np.ndarray
    x_native: np.ndarray
    gamma: float


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if hasattr(value, "__dataclass_fields__"):
        return _json_ready(asdict(value))
    return value


def _canonical_sha256(payload: dict[str, Any], *, omit: tuple[str, ...] = ()) -> str:
    value = {key: item for key, item in payload.items() if key not in omit}
    encoded = json.dumps(
        _json_ready(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _with_payload_sha256(payload: dict[str, Any]) -> dict[str, Any]:
    ready = _json_ready(payload)
    ready["payload_sha256"] = _canonical_sha256(ready, omit=("payload_sha256",))
    return ready


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _source_manifest() -> dict[str, str]:
    root = _repository_root()
    missing = [path for path in SOURCE_PATHS if not (root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"source manifest is missing files: {missing}")
    return {path: _sha256_file(root / path) for path in SOURCE_PATHS}


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _load_and_validate_inputs(
    args: argparse.Namespace,
) -> tuple[dict[str, Euler1DNPZ], dict[str, str], dict[str, Any]]:
    paths = {
        "coarse_data": args.coarse_data,
        "native_data": args.native_data,
        "fine_data": args.fine_data,
        "checkpoint": args.checkpoint,
        "family_contract": args.family_contract,
    }
    actual_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    mismatches = {
        name: {"actual": actual_hashes[name], "expected": expected}
        for name, expected in EXPECTED_HASHES.items()
        if actual_hashes.get(name) != expected
    }
    if mismatches:
        raise ValueError(f"D030 input hash mismatch: {mismatches}")
    sources = {
        "coarse": load_euler1d_npz(args.coarse_data),
        "native": load_euler1d_npz(args.native_data),
        "fine": load_euler1d_npz(args.fine_data),
    }
    expected_cells = {
        "coarse": CONTRACT.coarse_cells,
        "native": CONTRACT.native_cells,
        "fine": CONTRACT.fine_cells,
    }
    for name, source in sources.items():
        if source.num_cells != expected_cells[name]:
            raise ValueError(f"{name} source has the wrong cell count")
        if source.num_cases != 512 or source.num_frames != 101:
            raise ValueError(f"{name} source violates the D030 case/frame contract")
    fine = sources["fine"]
    identity_checks = {
        "saved_times_exact": all(
            np.array_equal(source.t, fine.t) for source in sources.values()
        ),
        "left_boundary_exact": all(
            np.array_equal(source.left_states, fine.left_states)
            for source in sources.values()
        ),
        "right_boundary_exact": all(
            np.array_equal(source.right_states, fine.right_states)
            for source in sources.values()
        ),
        "gamma_exact": all(source.gamma == fine.gamma for source in sources.values()),
        "component_scales_exact": np.array_equal(
            PHYSICAL_COMPONENT_SCALES,
            np.asarray((1.0, 1.0, 2.5), dtype=np.float64),
        ),
    }
    if not all(identity_checks.values()):
        raise ValueError(f"D030 physical identity check failed: {identity_checks}")
    family_contract = json.loads(args.family_contract.read_text(encoding="utf-8"))
    if family_contract.get("status") != "ok":
        raise ValueError("D030 family contract is not marked ok")
    return (
        sources,
        actual_hashes,
        {
            "identity_checks": identity_checks,
            "family_contract_status": family_contract.get("status"),
            "family_contract_maximum_relative_error": family_contract.get(
                "maximum_relative_error"
            ),
            "family_contract_local_relative_error_diagnostic_max": family_contract.get(
                "local_relative_error_diagnostic_max"
            ),
        },
    )


def _validate_checkpoint_split(checkpoint: dict[str, Any]) -> None:
    validation = tuple(int(value) for value in checkpoint.get("val_cases", ()))
    test = tuple(int(value) for value in checkpoint.get("test_cases", ()))
    if validation != EXPECTED_VALIDATION_CASES or test != EXPECTED_TEST_CASES:
        raise ValueError("checkpoint split differs from the frozen D030 split")
    args = checkpoint.get("args", {})
    expected = {
        "fno_width": 64,
        "fno_modes": 24,
        "fno_layers": 4,
        "step_stride": 1,
        "input_coordinates": "conservative",
        "input_normalization": "fixed_physical",
        "loss_coordinates": "conservative",
        "loss_normalization": "fixed_physical",
        "recurrent_coordinates": "conservative",
        "target_supervision": "state",
    }
    mismatches = {
        key: {"actual": args.get(key), "expected": value}
        for key, value in expected.items()
        if args.get(key) != value
    }
    if mismatches:
        raise ValueError(f"checkpoint contract mismatch: {mismatches}")


def _shock_mask(truth_primitive: np.ndarray) -> np.ndarray:
    pressure = np.asarray(truth_primitive, dtype=np.float64)[:, 2]
    jumps = np.abs(np.diff(pressure))
    selected: list[int] = []
    for face in np.argsort(-jumps, kind="stable"):
        face_id = int(face)
        if all(abs(face_id - previous) > 4 for previous in selected):
            selected.append(face_id)
            if len(selected) == 2:
                break
    if len(selected) != 2:
        raise ValueError("could not identify two separated pressure fronts")
    cells = np.arange(pressure.size, dtype=np.int64)
    mask = np.zeros(pressure.size, dtype=np.bool_)
    for face in selected:
        distance = np.minimum(np.abs(cells - face), np.abs(cells - (face + 1)))
        mask |= distance <= 4
    return mask


def _synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _collect_snapshots(
    sources: dict[str, Euler1DNPZ],
    case_ids: tuple[int, ...],
    model: torch.nn.Module,
    adapter: torch.nn.Module,
    device: torch.device,
    primary_projector: CosineProjector1D,
) -> tuple[list[CollectedSnapshot], dict[str, Any]]:
    ids = np.asarray(case_ids, dtype=np.int64)
    fine_source = sources["fine"]
    gamma = fine_source.gamma
    native_volumes = np.full(CONTRACT.native_cells, 1.0 / CONTRACT.native_cells)
    maxima = {
        "common_source_nesting_abs": 0.0,
        "serialized_coarse_state_abs": 0.0,
        "serialized_native_state_abs": 0.0,
        "serialized_coarse_increment_abs": 0.0,
        "serialized_native_increment_abs": 0.0,
        "mapped_increment_integral_abs": 0.0,
        "projection_integral_abs": 0.0,
        "post_fp32_nesting_abs": 0.0,
        "repeat_prediction_abs": 0.0,
    }
    snapshots: list[CollectedSnapshot] = []
    forward_seconds = 0.0
    actual_model_calls = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    wall_start = time.perf_counter()
    for input_call in ALL_CALLS:
        fine_current = primitive_to_conservative_np(
            np.asarray(fine_source.data[ids, input_call], dtype=np.float64), gamma
        )
        fine_truth_next = primitive_to_conservative_np(
            np.asarray(fine_source.data[ids, input_call + 1], dtype=np.float64), gamma
        )
        native_current = restrict_cell_averages(fine_current, CONTRACT.native_cells)
        coarse_current = restrict_cell_averages(fine_current, CONTRACT.coarse_cells)
        native_truth_next = restrict_cell_averages(
            fine_truth_next, CONTRACT.native_cells
        )
        coarse_truth_next = restrict_cell_averages(
            fine_truth_next, CONTRACT.coarse_cells
        )
        nested_coarse = restrict_cell_averages(native_current, CONTRACT.coarse_cells)
        maxima["common_source_nesting_abs"] = max(
            maxima["common_source_nesting_abs"],
            float(np.max(np.abs(nested_coarse - coarse_current))),
        )
        serialized_native = primitive_to_conservative_np(
            np.asarray(sources["native"].data[ids, input_call], dtype=np.float64),
            gamma,
        )
        serialized_coarse = primitive_to_conservative_np(
            np.asarray(sources["coarse"].data[ids, input_call], dtype=np.float64),
            gamma,
        )
        serialized_native_next = primitive_to_conservative_np(
            np.asarray(sources["native"].data[ids, input_call + 1], dtype=np.float64),
            gamma,
        )
        serialized_coarse_next = primitive_to_conservative_np(
            np.asarray(sources["coarse"].data[ids, input_call + 1], dtype=np.float64),
            gamma,
        )
        maxima["serialized_native_state_abs"] = max(
            maxima["serialized_native_state_abs"],
            float(np.max(np.abs(serialized_native - native_current))),
        )
        maxima["serialized_coarse_state_abs"] = max(
            maxima["serialized_coarse_state_abs"],
            float(np.max(np.abs(serialized_coarse - coarse_current))),
        )
        maxima["serialized_native_increment_abs"] = max(
            maxima["serialized_native_increment_abs"],
            float(
                np.max(
                    np.abs(
                        (serialized_native_next - serialized_native)
                        - (native_truth_next - native_current)
                    )
                )
            ),
        )
        maxima["serialized_coarse_increment_abs"] = max(
            maxima["serialized_coarse_increment_abs"],
            float(
                np.max(
                    np.abs(
                        (serialized_coarse_next - serialized_coarse)
                        - (coarse_truth_next - coarse_current)
                    )
                )
            ),
        )

        common_states = {
            "coarse": coarse_current,
            "native": native_current,
            "fine": fine_current,
        }
        model_states: dict[str, np.ndarray] = {}
        predictions: dict[str, np.ndarray] = {}
        for name in ("coarse", "native", "fine"):
            primitive = np.asarray(
                conservative_to_primitive_np(common_states[name], gamma),
                dtype=np.float32,
            )
            model_states[name] = np.asarray(
                primitive_to_conservative_np(primitive, gamma), dtype=np.float64
            )
            _synchronize_if_needed(device)
            started = time.perf_counter()
            _primitive_prediction, conservative_prediction = _predict_model_batch_state(
                sources[name],
                ids,
                primitive,
                input_call,
                1,
                model,
                adapter,
                device,
            )
            _synchronize_if_needed(device)
            forward_seconds += time.perf_counter() - started
            actual_model_calls += 1
            predictions[name] = conservative_prediction
            if input_call == 0:
                _repeat_primitive, repeat_conservative = _predict_model_batch_state(
                    sources[name],
                    ids,
                    primitive,
                    input_call,
                    1,
                    model,
                    adapter,
                    device,
                )
                actual_model_calls += 1
                maxima["repeat_prediction_abs"] = max(
                    maxima["repeat_prediction_abs"],
                    float(
                        np.max(np.abs(repeat_conservative - conservative_prediction))
                    ),
                )
        maxima["post_fp32_nesting_abs"] = max(
            maxima["post_fp32_nesting_abs"],
            float(
                np.max(
                    np.abs(
                        restrict_cell_averages(
                            model_states["fine"], CONTRACT.native_cells
                        )
                        - model_states["native"]
                    )
                )
            ),
            float(
                np.max(
                    np.abs(
                        restrict_cell_averages(
                            model_states["native"], CONTRACT.coarse_cells
                        )
                        - model_states["coarse"]
                    )
                )
            ),
        )

        for row, numeric_case_id in enumerate(case_ids):
            state_triplet = CommonSourceStates(
                coarse=model_states["coarse"][row],
                native=model_states["native"][row],
                fine=model_states["fine"][row],
            )
            predicted_triplet = CommonSourceStates(
                coarse=predictions["coarse"][row],
                native=predictions["native"][row],
                fine=predictions["fine"][row],
            )
            basis = mapped_increment_basis(
                state_triplet,
                predicted_triplet,
                CONTRACT,
            )
            fine_integral = component_integrals(
                predictions["fine"][row] - model_states["fine"][row],
                volumes=np.full(CONTRACT.fine_cells, 1.0 / CONTRACT.fine_cells),
                component_scale=PHYSICAL_COMPONENT_SCALES,
            )
            coarse_integral = component_integrals(
                predictions["coarse"][row] - model_states["coarse"][row],
                volumes=np.full(CONTRACT.coarse_cells, 1.0 / CONTRACT.coarse_cells),
                component_scale=PHYSICAL_COMPONENT_SCALES,
            )
            mapped_fine_integral = component_integrals(
                basis.fine_on_native,
                volumes=native_volumes,
                component_scale=PHYSICAL_COMPONENT_SCALES,
            )
            mapped_coarse_integral = component_integrals(
                basis.coarse_on_native,
                volumes=native_volumes,
                component_scale=PHYSICAL_COMPONENT_SCALES,
            )
            maxima["mapped_increment_integral_abs"] = max(
                maxima["mapped_increment_integral_abs"],
                float(np.max(np.abs(fine_integral - mapped_fine_integral))),
                float(np.max(np.abs(coarse_integral - mapped_coarse_integral))),
            )
            projected_feature = primary_projector.project(
                basis.fine_minus_native,
                active_modes=PRIMARY_MODES,
                component_scale=PHYSICAL_COMPONENT_SCALES,
            )
            maxima["projection_integral_abs"] = max(
                maxima["projection_integral_abs"],
                float(
                    np.max(
                        np.abs(
                            component_integrals(
                                projected_feature,
                                volumes=native_volumes,
                                component_scale=PHYSICAL_COMPONENT_SCALES,
                            )
                        )
                    )
                ),
            )
            truth = native_truth_next[row]
            raw = predictions["native"][row]
            target = truth - raw
            coarse_error = prolong_piecewise_constant(
                predictions["coarse"][row] - coarse_truth_next[row],
                CONTRACT.native_cells,
            )
            fine_error = restrict_cell_averages(
                predictions["fine"][row] - fine_truth_next[row],
                CONTRACT.native_cells,
            )
            truth_primitive = conservative_to_primitive_np(truth, gamma)
            case_id = f"case_{numeric_case_id:03d}"
            record = ScalarFitRecord(
                case_id=case_id,
                input_call=input_call,
                feature=projected_feature,
                target_correction=target,
                native_increment=basis.native_increment,
                volumes=native_volumes,
                component_scale=PHYSICAL_COMPONENT_SCALES,
            )
            snapshots.append(
                CollectedSnapshot(
                    case_id=case_id,
                    numeric_case_id=numeric_case_id,
                    input_call=input_call,
                    record=record,
                    basis=basis,
                    current_native=model_states["native"][row],
                    truth_next_native=truth,
                    raw_next_native=raw,
                    coarse_error_on_native=coarse_error,
                    native_error=raw - truth,
                    fine_error_on_native=fine_error,
                    shock_mask=_shock_mask(truth_primitive),
                    x_native=np.asarray(
                        sources["native"].x[numeric_case_id], dtype=np.float64
                    ),
                    gamma=gamma,
                )
            )
    wall_seconds = time.perf_counter() - wall_start
    expected_records = len(case_ids) * len(ALL_CALLS)
    if len(snapshots) != expected_records:
        raise AssertionError("snapshot inventory is incomplete")
    validate_record_inventory(
        [snapshot.record for snapshot in snapshots],
        expected_case_ids=tuple(f"case_{case_id:03d}" for case_id in case_ids),
        expected_input_calls=ALL_CALLS,
    )
    return snapshots, {
        "snapshot_count": len(snapshots),
        "logical_model_calls": 3 * len(ALL_CALLS),
        "actual_model_calls": actual_model_calls,
        "forward_seconds": forward_seconds,
        "wall_seconds": wall_seconds,
        "peak_cuda_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "maxima": maxima,
    }


def _score_view(
    records: list[ScoreRecord],
    snapshots_by_key: dict[tuple[str, int], CollectedSnapshot],
    target_band_cache: dict[tuple[str, int], dict[str, np.ndarray]],
    view: str,
) -> dict[str, Any]:
    transformed = []
    for record in records:
        target = np.asarray(record.target_correction, dtype=np.float64)
        correction = np.asarray(record.correction, dtype=np.float64)
        volumes = np.asarray(record.volumes, dtype=np.float64)
        scale = np.asarray(record.component_scale, dtype=np.float64)
        snapshot = snapshots_by_key[(record.case_id, int(record.input_call))]
        if view in VIEW_MODES:
            target = target_band_cache[(record.case_id, int(record.input_call))][view]
            if view != "low_k1_k7":
                correction = np.zeros_like(correction)
        elif view.startswith("component_"):
            component = COMPONENT_NAMES.index(view.removeprefix("component_"))
            target = target[:, component : component + 1]
            correction = correction[:, component : component + 1]
            scale = scale[component : component + 1]
        elif view == "boundary":
            mask = np.zeros(CONTRACT.native_cells, dtype=np.bool_)
            mask[:8] = True
            mask[-8:] = True
            target, correction, volumes = target[mask], correction[mask], volumes[mask]
        elif view in ("shock_local", "smooth"):
            mask = snapshot.shock_mask
            if view == "smooth":
                mask = ~mask
            target, correction, volumes = target[mask], correction[mask], volumes[mask]
        elif view != "full":
            raise ValueError(f"unknown score view: {view}")
        transformed.append(
            ScoreRecord(
                case_id=record.case_id,
                input_call=int(record.input_call),
                target_correction=target,
                correction=correction,
                volumes=volumes,
                component_scale=scale,
            )
        )
    return score_records(transformed)


def _target_band_cache(
    snapshots: list[CollectedSnapshot],
    projector: CosineProjector1D,
) -> dict[tuple[str, int], dict[str, np.ndarray]]:
    cache = {}
    for snapshot in snapshots:
        coordinates = projector.coordinates(
            snapshot.record.target_correction,
            component_scale=PHYSICAL_COMPONENT_SCALES,
        )
        bands = {}
        for view, modes in VIEW_MODES.items():
            indices = np.asarray(modes, dtype=np.int64)
            weighted = projector.q_matrix[:, indices] @ coordinates[indices]
            bands[view] = (
                weighted
                / projector.square_root_mass[:, None]
                * PHYSICAL_COMPONENT_SCALES[None, :]
            )
        cache[(snapshot.case_id, snapshot.input_call)] = bands
    return cache


def _case_first_control(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    by_case: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        by_case.setdefault(row["case_id"], []).append(
            (float(row[f"raw_{key}"]), float(row[f"corrected_{key}"]))
        )
    raw_case = []
    corrected_case = []
    case_rows = []
    for case_id, values in sorted(by_case.items()):
        array = np.asarray(values, dtype=np.float64)
        raw_rms = float(np.sqrt(np.mean(np.square(array[:, 0]))))
        corrected_rms = float(np.sqrt(np.mean(np.square(array[:, 1]))))
        if raw_rms > DENOMINATOR_FLOOR:
            ratio = corrected_rms / raw_rms
            status = "ok"
        elif corrected_rms <= DENOMINATOR_FLOOR:
            ratio = 1.0
            status = "exact_zero_no_change"
        else:
            ratio = None
            status = "small_raw_nonzero_corrected"
        raw_case.append(raw_rms * raw_rms)
        corrected_case.append(corrected_rms * corrected_rms)
        case_rows.append(
            {
                "case_id": case_id,
                "raw_rms": raw_rms,
                "corrected_rms": corrected_rms,
                "ratio": ratio,
                "status": status,
            }
        )
    raw = float(np.sqrt(np.mean(raw_case)))
    corrected = float(np.sqrt(np.mean(corrected_case)))
    if raw > DENOMINATOR_FLOOR:
        ratio = corrected / raw
        status = "ok"
    elif corrected <= DENOMINATOR_FLOOR:
        ratio = 1.0
        status = "exact_zero_no_change"
    else:
        ratio = None
        status = "small_raw_nonzero_corrected"
    return {
        "key": key,
        "raw_rms": raw,
        "corrected_rms": corrected,
        "ratio": ratio,
        "status": status,
        "case_rows": case_rows,
    }


def _physical_controls(
    snapshots: list[CollectedSnapshot],
    score_rows: list[ScoreRecord],
) -> tuple[dict[str, Any], dict[str, Any]]:
    snapshot_by_key = {
        (snapshot.case_id, snapshot.input_call): snapshot for snapshot in snapshots
    }
    rows = []
    proposal_rows = []
    for score_row in score_rows:
        snapshot = snapshot_by_key[(score_row.case_id, int(score_row.input_call))]
        corrected = snapshot.raw_next_native + score_row.correction
        raw_primitive = conservative_to_primitive_np(
            snapshot.raw_next_native, snapshot.gamma
        )
        corrected_primitive = conservative_to_primitive_np(corrected, snapshot.gamma)
        truth_primitive = conservative_to_primitive_np(
            snapshot.truth_next_native, snapshot.gamma
        )
        raw_front = pressure_front_top2_metrics_np(
            raw_primitive, truth_primitive, snapshot.x_native
        )
        corrected_front = pressure_front_top2_metrics_np(
            corrected_primitive, truth_primitive, snapshot.x_native
        )
        raw_integral = component_integrals(
            snapshot.raw_next_native - snapshot.truth_next_native,
            volumes=score_row.volumes,
            component_scale=score_row.component_scale,
        )
        corrected_integral = component_integrals(
            corrected - snapshot.truth_next_native,
            volumes=score_row.volumes,
            component_scale=score_row.component_scale,
        )
        row = {
            "case_id": snapshot.case_id,
            "input_call": snapshot.input_call,
        }
        for key in (
            "position_assignment_mae",
            "strength_relative_l1",
            "primary_to_truth_top2_mae",
        ):
            row[f"raw_front_{key}"] = float(np.asarray(raw_front[key]))
            row[f"corrected_front_{key}"] = float(np.asarray(corrected_front[key]))
        for component, name in enumerate(COMPONENT_NAMES):
            row[f"raw_integral_{name}"] = float(abs(raw_integral[component]))
            row[f"corrected_integral_{name}"] = float(
                abs(corrected_integral[component])
            )
        rows.append(row)
        pressure = corrected_primitive[:, 2]
        density = corrected_primitive[:, 0]
        proposal_rows.append(
            {
                "case_id": snapshot.case_id,
                "input_call": snapshot.input_call,
                "finite": bool(np.isfinite(corrected_primitive).all()),
                "admissible": bool(
                    np.isfinite(corrected_primitive).all()
                    and np.all(density > 0.0)
                    and np.all(pressure > 0.0)
                ),
                "minimum_density": float(np.min(density)),
                "minimum_pressure": float(np.min(pressure)),
            }
        )
    controls = {}
    for key in (
        "front_position_assignment_mae",
        "front_strength_relative_l1",
        "front_primary_to_truth_top2_mae",
        "integral_density",
        "integral_momentum",
        "integral_energy",
    ):
        controls[key] = _case_first_control(rows, key)
    return controls, {
        "all_finite": all(row["finite"] for row in proposal_rows),
        "all_admissible": all(row["admissible"] for row in proposal_rows),
        "minimum_density": min(row["minimum_density"] for row in proposal_rows),
        "minimum_pressure": min(row["minimum_pressure"] for row in proposal_rows),
        "rows": proposal_rows,
    }


def _candidate_evidence(
    snapshots: list[CollectedSnapshot],
    coefficient: float,
    target_band_cache: dict[tuple[str, int], dict[str, np.ndarray]],
) -> dict[str, Any]:
    fit_records = [snapshot.record for snapshot in snapshots]
    score_rows, audits = corrections_for_coefficient(fit_records, coefficient)
    snapshots_by_key = {
        (snapshot.case_id, snapshot.input_call): snapshot for snapshot in snapshots
    }
    views = (
        "full",
        "low_k1_k7",
        "transition_k8_k31",
        "local_k32_k255",
        "component_density",
        "component_momentum",
        "component_energy",
        "boundary",
        "shock_local",
        "smooth",
    )
    view_scores = {
        view: _score_view(score_rows, snapshots_by_key, target_band_cache, view)
        for view in views
    }
    early_keys = {
        (row.case_id, row.input_call)
        for row in fit_records
        if row.input_call in EARLY_CALLS
    }
    early_scores = [
        row for row in score_rows if (row.case_id, row.input_call) in early_keys
    ]
    late_scores = [
        row for row in score_rows if (row.case_id, row.input_call) not in early_keys
    ]
    early_low = _score_view(
        early_scores, snapshots_by_key, target_band_cache, "low_k1_k7"
    )
    late_low = _score_view(
        late_scores, snapshots_by_key, target_band_cache, "low_k1_k7"
    )
    full_cases = {row["case_id"]: row for row in view_scores["full"]["case_scores"]}
    low_cases = {row["case_id"]: row for row in view_scores["low_k1_k7"]["case_scores"]}
    case_wins = sum(
        full_cases[case_id]["skill_status"] == "ok"
        and low_cases[case_id]["skill_status"] == "ok"
        and float(full_cases[case_id]["skill_vs_zero"]) >= 0.0
        and float(low_cases[case_id]["skill_vs_zero"]) > 0.0
        for case_id in sorted(full_cases)
    )
    physical_controls, proposals = _physical_controls(snapshots, score_rows)
    view_no_harm = {}
    for view, score in view_scores.items():
        population_pass = (
            score["rms_ratio_vs_zero"] is not None
            and float(score["rms_ratio_vs_zero"]) <= NO_HARM_RATIO
        )
        case_pass = all(
            row["rms_ratio_vs_zero"] is not None
            and float(row["rms_ratio_vs_zero"]) <= NO_HARM_RATIO
            for row in score["case_scores"]
        )
        view_no_harm[view] = population_pass and case_pass
    physical_no_harm = {
        key: row["status"] in ("ok", "exact_zero_no_change")
        and row["ratio"] is not None
        and float(row["ratio"]) <= NO_HARM_RATIO
        and all(
            case_row["status"] in ("ok", "exact_zero_no_change")
            and case_row["ratio"] is not None
            and float(case_row["ratio"]) <= NO_HARM_RATIO
            for case_row in row["case_rows"]
        )
        for key, row in physical_controls.items()
    }
    audit_checks = {
        "all_status_ok": all(row["status"] == "ok" for row in audits),
        "maximum_component_integral_abs": max(
            float(row["maximum_component_integral_abs"]) for row in audits
        ),
        "cap_activation_count": sum(bool(row["cap_active"]) for row in audits),
        "maximum_correction_to_native_increment": max(
            float(row["correction_to_native_increment"])
            for row in audits
            if row["correction_to_native_increment"] is not None
        ),
    }
    checks = {
        "full_skill_nonnegative": view_scores["full"]["skill_status"] == "ok"
        and float(view_scores["full"]["skill_vs_zero"]) >= 0.0,
        "low_skill_at_least_0p05": view_scores["low_k1_k7"]["skill_status"] == "ok"
        and float(view_scores["low_k1_k7"]["skill_vs_zero"]) >= MINIMUM_LOW_SKILL,
        "minimum_twelve_case_wins": case_wins >= MINIMUM_CASE_WINS,
        "both_half_horizons_positive": early_low["skill_status"] == "ok"
        and late_low["skill_status"] == "ok"
        and float(early_low["skill_vs_zero"]) > 0.0
        and float(late_low["skill_vs_zero"]) > 0.0,
        "all_view_controls_no_harm": all(view_no_harm.values()),
        "all_physical_controls_no_harm": all(physical_no_harm.values()),
        "all_proposals_finite_admissible": proposals["all_finite"]
        and proposals["all_admissible"],
        "all_audits_resolved": audit_checks["all_status_ok"],
        "correction_integral_closure": audit_checks["maximum_component_integral_abs"]
        <= 1.0e-12,
    }
    return {
        "coefficient": coefficient,
        "checks": checks,
        "eligible": all(checks.values()),
        "case_win_count": case_wins,
        "view_scores": view_scores,
        "early_low_score": early_low,
        "late_low_score": late_low,
        "view_no_harm": view_no_harm,
        "physical_controls": physical_controls,
        "physical_no_harm": physical_no_harm,
        "proposals": proposals,
        "audit": audit_checks,
    }


def _fit_record_with(
    snapshot: CollectedSnapshot,
    feature: np.ndarray,
    target: np.ndarray,
    *,
    component: int | None = None,
    mask: np.ndarray | None = None,
) -> ScalarFitRecord:
    volumes = np.asarray(snapshot.record.volumes)
    scale = np.asarray(snapshot.record.component_scale)
    native = np.asarray(snapshot.record.native_increment)
    values = np.asarray(feature)
    labels = np.asarray(target)
    if mask is not None:
        values, labels, native, volumes = (
            values[mask],
            labels[mask],
            native[mask],
            volumes[mask],
        )
    if component is not None:
        values = values[:, component : component + 1]
        labels = labels[:, component : component + 1]
        native = native[:, component : component + 1]
        scale = scale[component : component + 1]
    return ScalarFitRecord(
        case_id=snapshot.case_id,
        input_call=snapshot.input_call,
        feature=values,
        target_correction=labels,
        native_increment=native,
        volumes=volumes,
        component_scale=scale,
    )


def _two_feature_fit(
    snapshots: list[CollectedSnapshot],
    projector: CosineProjector1D,
) -> dict[str, Any]:
    by_case: dict[str, list[tuple[np.ndarray, np.ndarray, float]]] = {}
    for snapshot in snapshots:
        fine = snapshot.record.feature
        coarse = projector.project(
            snapshot.basis.native_minus_coarse,
            active_modes=PRIMARY_MODES,
            component_scale=PHYSICAL_COMPONENT_SCALES,
        )
        target = projector.project(
            snapshot.record.target_correction,
            active_modes=PRIMARY_MODES,
            component_scale=PHYSICAL_COMPONENT_SCALES,
        )
        gram = np.asarray(
            (
                (
                    weighted_scaled_inner(
                        coarse,
                        coarse,
                        volumes=snapshot.record.volumes,
                        component_scale=PHYSICAL_COMPONENT_SCALES,
                    ),
                    weighted_scaled_inner(
                        coarse,
                        fine,
                        volumes=snapshot.record.volumes,
                        component_scale=PHYSICAL_COMPONENT_SCALES,
                    ),
                ),
                (
                    weighted_scaled_inner(
                        fine,
                        coarse,
                        volumes=snapshot.record.volumes,
                        component_scale=PHYSICAL_COMPONENT_SCALES,
                    ),
                    weighted_scaled_inner(
                        fine,
                        fine,
                        volumes=snapshot.record.volumes,
                        component_scale=PHYSICAL_COMPONENT_SCALES,
                    ),
                ),
            )
        )
        cross = np.asarray(
            (
                weighted_scaled_inner(
                    coarse,
                    target,
                    volumes=snapshot.record.volumes,
                    component_scale=PHYSICAL_COMPONENT_SCALES,
                ),
                weighted_scaled_inner(
                    fine,
                    target,
                    volumes=snapshot.record.volumes,
                    component_scale=PHYSICAL_COMPONENT_SCALES,
                ),
            )
        )
        target_square = weighted_scaled_inner(
            target,
            target,
            volumes=snapshot.record.volumes,
            component_scale=PHYSICAL_COMPONENT_SCALES,
        )
        by_case.setdefault(snapshot.case_id, []).append((gram, cross, target_square))
    gram = np.mean(
        [np.mean([row[0] for row in values], axis=0) for values in by_case.values()],
        axis=0,
    )
    cross = np.mean(
        [np.mean([row[1] for row in values], axis=0) for values in by_case.values()],
        axis=0,
    )
    target_square = float(
        np.mean([np.mean([row[2] for row in values]) for values in by_case.values()])
    )
    eigenvalues = np.linalg.eigvalsh(gram)
    status = "ok" if float(eigenvalues[0]) > DENOMINATOR_FLOOR**2 else "rank_deficient"
    coefficients = np.linalg.solve(gram, cross) if status == "ok" else None
    corrected_sse = (
        float(
            target_square
            - 2.0 * coefficients @ cross
            + coefficients @ gram @ coefficients
        )
        if coefficients is not None
        else None
    )
    fine_coefficient = (
        float(cross[1] / gram[1, 1])
        if float(np.sqrt(max(gram[1, 1], 0.0))) > DENOMINATOR_FLOOR
        else None
    )
    fine_sse = (
        float(
            target_square
            - 2.0 * fine_coefficient * cross[1]
            + fine_coefficient * fine_coefficient * gram[1, 1]
        )
        if fine_coefficient is not None
        else None
    )
    return {
        "gram": gram.tolist(),
        "cross": cross.tolist(),
        "target_square": target_square,
        "eigenvalues": eigenvalues.tolist(),
        "condition_number": (
            float(eigenvalues[-1] / eigenvalues[0]) if status == "ok" else None
        ),
        "coefficients_coarse_fine": (
            coefficients.tolist() if coefficients is not None else None
        ),
        "skill_vs_zero": (
            float(1.0 - corrected_sse / target_square)
            if corrected_sse is not None and target_square > DENOMINATOR_FLOOR**2
            else None
        ),
        "fine_only_coefficient": fine_coefficient,
        "fine_only_skill_vs_zero": (
            float(1.0 - fine_sse / target_square)
            if fine_sse is not None and target_square > DENOMINATOR_FLOOR**2
            else None
        ),
        "coarse_incremental_skill": (
            float((fine_sse - corrected_sse) / target_square)
            if fine_sse is not None
            and corrected_sse is not None
            and target_square > DENOMINATOR_FLOOR**2
            else None
        ),
        "status": status,
    }


def _singular_summary(matrix: np.ndarray) -> dict[str, Any]:
    values = np.asarray(matrix, dtype=np.float64)
    singular_values = np.linalg.svd(values, compute_uv=False)
    energy = np.square(singular_values)
    total = float(energy.sum())
    fractions = (
        energy / total if total > DENOMINATOR_FLOOR**2 else np.zeros_like(energy)
    )
    cumulative = np.cumsum(fractions)
    rank95 = int(np.searchsorted(cumulative, 0.95) + 1) if total > 0.0 else None
    tolerance = (
        max(values.shape)
        * np.finfo(np.float64).eps
        * (float(singular_values[0]) if singular_values.size else 0.0)
    )
    return {
        "shape": list(values.shape),
        "rank": int(np.count_nonzero(singular_values > tolerance)),
        "rank95": rank95,
        "first_energy_fraction": float(fractions[0]) if fractions.size else None,
        "first_two_energy_fraction": (
            float(fractions[:2].sum()) if fractions.size >= 2 else None
        ),
        "first_six_energy_fraction": (
            float(fractions[:6].sum()) if fractions.size >= 6 else None
        ),
        "first_sixteen_energy_fraction": (
            float(fractions[:16].sum()) if fractions.size >= 16 else None
        ),
        "leading_singular_values": singular_values[:16].tolist(),
    }


def _temporal_cosines(
    snapshots: list[CollectedSnapshot],
    selector: str,
) -> dict[str, Any]:
    by_case: dict[str, list[CollectedSnapshot]] = {}
    for snapshot in snapshots:
        by_case.setdefault(snapshot.case_id, []).append(snapshot)
    rows = []
    for case_id, values in sorted(by_case.items()):
        ordered = sorted(values, key=lambda row: row.input_call)
        cosines = []
        for first, second in itertools.pairwise(ordered):
            if selector == "fine_discrepancy":
                left, right = first.record.feature, second.record.feature
            elif selector == "target":
                left, right = (
                    first.record.target_correction,
                    second.record.target_correction,
                )
            else:
                raise ValueError(selector)
            left_norm = weighted_scaled_rms(
                left,
                volumes=first.record.volumes,
                component_scale=first.record.component_scale,
            )
            right_norm = weighted_scaled_rms(
                right,
                volumes=second.record.volumes,
                component_scale=second.record.component_scale,
            )
            cosine = None
            if left_norm > DENOMINATOR_FLOOR and right_norm > DENOMINATOR_FLOOR:
                cosine = weighted_scaled_inner(
                    left,
                    right,
                    volumes=first.record.volumes,
                    component_scale=first.record.component_scale,
                ) / (left_norm * right_norm)
                cosines.append(float(cosine))
            rows.append(
                {
                    "case_id": case_id,
                    "first_call": first.input_call,
                    "second_call": second.input_call,
                    "cosine": cosine,
                    "status": "ok" if cosine is not None else "small_denominator",
                }
            )
    return {
        "median_consecutive_cosine": float(np.median(cosines)) if cosines else None,
        "status": "ok" if len(cosines) == len(rows) else "unresolved_denominator",
        "row_count": len(rows),
    }


def _structure_diagnostics(
    snapshots: list[CollectedSnapshot],
    full_projector: CosineProjector1D,
) -> dict[str, Any]:
    modal_cache = {}
    for snapshot in snapshots:
        key = (snapshot.case_id, snapshot.input_call)
        modal_cache[key] = {
            "fine": full_projector.coordinates(
                snapshot.basis.fine_minus_native,
                component_scale=PHYSICAL_COMPONENT_SCALES,
            ),
            "coarse": full_projector.coordinates(
                snapshot.basis.native_minus_coarse,
                component_scale=PHYSICAL_COMPONENT_SCALES,
            ),
            "target": full_projector.coordinates(
                snapshot.record.target_correction,
                component_scale=PHYSICAL_COMPONENT_SCALES,
            ),
        }

    def reconstruct_modes(
        coordinates: np.ndarray, modes: tuple[int, ...]
    ) -> np.ndarray:
        indices = np.asarray(modes, dtype=np.int64)
        weighted = full_projector.q_matrix[:, indices] @ coordinates[indices]
        return (
            weighted
            / full_projector.square_root_mass[:, None]
            * PHYSICAL_COMPONENT_SCALES[None, :]
        )

    scopes = {
        "all": snapshots,
        "calls_0_49": [row for row in snapshots if row.input_call in EARLY_CALLS],
        "calls_50_99": [row for row in snapshots if row.input_call in LATE_CALLS],
    }
    relations = {}
    for scope_name, scope in scopes.items():
        target_relations = {}
        for feature_name in ("fine_full", "fine_low", "coarse_full", "coarse_low"):
            records = []
            for snapshot in scope:
                if feature_name.startswith("fine"):
                    feature = snapshot.basis.fine_minus_native
                else:
                    feature = snapshot.basis.native_minus_coarse
                target = snapshot.record.target_correction
                if feature_name.endswith("low"):
                    key = (snapshot.case_id, snapshot.input_call)
                    feature_key = (
                        "fine" if feature_name.startswith("fine") else "coarse"
                    )
                    feature = reconstruct_modes(
                        modal_cache[key][feature_key], PRIMARY_MODES
                    )
                    target = reconstruct_modes(
                        modal_cache[key]["target"], PRIMARY_MODES
                    )
                records.append(_fit_record_with(snapshot, feature, target))
            target_relations[feature_name] = relation_from_fit_records(records)
        error_relations = {}
        for pair_name, feature_getter in (
            ("coarse_error_to_native_error", lambda row: row.coarse_error_on_native),
            ("fine_error_to_native_error", lambda row: row.fine_error_on_native),
            ("coarse_error_to_fine_error", lambda row: row.coarse_error_on_native),
        ):
            records = []
            for snapshot in scope:
                target = (
                    snapshot.fine_error_on_native
                    if pair_name == "coarse_error_to_fine_error"
                    else snapshot.native_error
                )
                records.append(
                    _fit_record_with(snapshot, feature_getter(snapshot), target)
                )
            error_relations[pair_name] = relation_from_fit_records(records)
        relations[scope_name] = {
            "correction_target_relations": target_relations,
            "mapped_prediction_error_relations": error_relations,
        }

    components = {}
    for component, name in enumerate(COMPONENT_NAMES):
        records = [
            _fit_record_with(
                snapshot,
                snapshot.record.feature,
                snapshot.record.target_correction,
                component=component,
            )
            for snapshot in snapshots
        ]
        components[name] = relation_from_fit_records(records)

    regions = {}
    for name in ("shock_local", "smooth"):
        records = []
        for snapshot in snapshots:
            mask = (
                snapshot.shock_mask if name == "shock_local" else ~snapshot.shock_mask
            )
            records.append(
                _fit_record_with(
                    snapshot,
                    snapshot.record.feature,
                    snapshot.record.target_correction,
                    mask=mask,
                )
            )
        regions[name] = relation_from_fit_records(records)

    modes = {}
    for mode in range(32):
        records = []
        for snapshot in snapshots:
            key = (snapshot.case_id, snapshot.input_call)
            feature = reconstruct_modes(modal_cache[key]["fine"], (mode,))
            target = reconstruct_modes(modal_cache[key]["target"], (mode,))
            records.append(_fit_record_with(snapshot, feature, target))
        relation = relation_from_fit_records(records)
        modes[str(mode)] = {
            "fit": relation["fit"],
            "oracle_skill": (
                None
                if relation["uncapped_oracle_score"] is None
                else relation["uncapped_oracle_score"]["skill_vs_zero"]
            ),
            "median_case_cosine": relation["unit_feature_score"]["median_case_cosine"],
            "signed_metrics_status": relation["unit_feature_score"][
                "signed_metrics_status"
            ],
        }

    target_matrix = []
    rank1_fractions = []
    volumes = np.asarray(snapshots[0].record.volumes)
    root_mass = np.sqrt(volumes)[:, None]
    for snapshot in snapshots:
        target_matrix.append(
            (
                root_mass
                * (
                    snapshot.record.target_correction
                    / PHYSICAL_COMPONENT_SCALES[None, :]
                )
            ).reshape(-1)
        )
        triplet = np.stack(
            (
                snapshot.coarse_error_on_native,
                snapshot.native_error,
                snapshot.fine_error_on_native,
            ),
            axis=0,
        )
        scaled = (
            root_mass[None] * (triplet / PHYSICAL_COMPONENT_SCALES[None, None, :])
        ).reshape(3, -1)
        singular = np.linalg.svd(scaled, compute_uv=False)
        energy = np.square(singular)
        rank1_fractions.append(float(energy[0] / energy.sum()))

    per_case_oracles = {}
    for case_id in sorted({snapshot.case_id for snapshot in snapshots}):
        fit = fit_case_first_scalar(
            [snapshot.record for snapshot in snapshots if snapshot.case_id == case_id]
        )
        per_case_oracles[case_id] = asdict(fit)

    return {
        "relations": relations,
        "component_relations": components,
        "region_relations": regions,
        "mode_relations_k0_k31": modes,
        "two_feature_low_mode_fit": _two_feature_fit(snapshots, full_projector),
        "target_singular_energy": _singular_summary(np.asarray(target_matrix)),
        "mapped_error_triplet_rank1_energy": {
            "median": float(np.median(rank1_fractions)),
            "minimum": float(np.min(rank1_fractions)),
            "maximum": float(np.max(rank1_fractions)),
        },
        "temporal": {
            "fine_discrepancy": _temporal_cosines(snapshots, "fine_discrepancy"),
            "target": _temporal_cosines(snapshots, "target"),
        },
        "per_case_oracle_coefficients": per_case_oracles,
        "oracle_deployable": False,
    }


def _common_checks(collection: dict[str, Any]) -> dict[str, bool]:
    maxima = collection["maxima"]
    return {
        "common_source_nesting": maxima["common_source_nesting_abs"] <= 1.0e-12,
        "serialized_coarse_state": maxima["serialized_coarse_state_abs"] <= 1.0e-6,
        "serialized_native_state": maxima["serialized_native_state_abs"] <= 1.0e-6,
        "serialized_coarse_increment": maxima["serialized_coarse_increment_abs"]
        <= 1.0e-6,
        "serialized_native_increment": maxima["serialized_native_increment_abs"]
        <= 1.0e-6,
        "mapped_increment_integral": maxima["mapped_increment_integral_abs"] <= 1.0e-12,
        "projection_integral": maxima["projection_integral_abs"] <= 1.0e-12,
        "post_fp32_nesting": maxima["post_fp32_nesting_abs"] <= 1.0e-6,
        "repeat_prediction": maxima["repeat_prediction_abs"] <= 1.0e-6,
    }


def _calibration_groups(case_ids: tuple[int, ...]) -> dict[str, tuple[str, ...]]:
    ordered = tuple(sorted(case_ids))
    return {
        f"fold_{fold}": tuple(
            f"case_{case_id:03d}" for case_id in ordered[4 * fold : 4 * fold + 4]
        )
        for fold in range(4)
    }


def _coefficient_stability(
    full_fit: Any,
    early_fit: Any,
    late_fit: Any,
    crossfit: dict[str, Any],
) -> dict[str, Any]:
    coefficients = [
        full_fit.coefficient,
        early_fit.coefficient,
        late_fit.coefficient,
        *crossfit["fold_coefficients"],
    ]
    resolved = all(value is not None and np.isfinite(value) for value in coefficients)
    nonzero_sign = (
        resolved
        and abs(float(full_fit.coefficient)) > DENOMINATOR_FLOOR
        and all(
            np.sign(float(value)) == np.sign(float(full_fit.coefficient))
            for value in coefficients[1:]
        )
    )
    checks = {
        "full_fit_resolved": full_fit.status == "ok"
        and full_fit.coefficient is not None,
        "early_fit_resolved": early_fit.status == "ok"
        and early_fit.coefficient is not None,
        "late_fit_resolved": late_fit.status == "ok"
        and late_fit.coefficient is not None,
        "crossfit_resolved": crossfit["status"] == "ok",
        "same_nonzero_sign": bool(nonzero_sign),
        "relative_iqr_at_most_0p5": crossfit["relative_iqr"] is not None
        and float(crossfit["relative_iqr"]) <= MAXIMUM_RELATIVE_IQR,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "full_fit": asdict(full_fit),
        "early_fit": asdict(early_fit),
        "late_fit": asdict(late_fit),
        "crossfit": crossfit,
    }


def _run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    source_manifest = _source_manifest()
    _sources, input_hashes, input_contract = _load_and_validate_inputs(args)
    device = _resolve_device(args.device)
    model, _adapter, _normalizer, checkpoint = load_frozen_residual_checkpoint(
        args.checkpoint, device
    )
    _validate_checkpoint_split(checkpoint)
    volumes = np.full(CONTRACT.native_cells, 1.0 / CONTRACT.native_cells)
    primary = build_cosine_projector(CONTRACT.native_cells, rank=8, volumes=volumes)
    full = build_cosine_projector(
        CONTRACT.native_cells, rank=CONTRACT.native_cells, volumes=volumes
    )
    checks = {
        "input_hashes_exact": input_hashes == EXPECTED_HASHES,
        "physical_identity": all(input_contract["identity_checks"].values()),
        "checkpoint_split_exact": tuple(checkpoint["val_cases"])
        == EXPECTED_VALIDATION_CASES
        and tuple(checkpoint["test_cases"]) == EXPECTED_TEST_CASES,
        "parameter_count_exact": sum(
            parameter.numel() for parameter in model.parameters()
        )
        == 316_419,
        "primary_projector_rank": primary.rank == 8,
        "full_projector_rank": full.rank == CONTRACT.native_cells,
        "primary_projector_orthogonal": primary.maximum_orthogonality_error <= 1.0e-12,
        "full_projector_orthogonal": full.maximum_orthogonality_error <= 1.0e-12,
        "validation_targets_closed": True,
        "test_targets_closed": True,
    }
    return _with_payload_sha256(
        {
            "schema": SCHEMA,
            "phase": "preflight",
            "status": "passed" if all(checks.values()) else "failed",
            "checks": checks,
            "source_manifest": source_manifest,
            "input_hashes": input_hashes,
            "input_contract": input_contract,
            "device": str(device),
            "target_access": {
                "validation_targets_scored": False,
                "test_targets_scored": False,
            },
            "checkpoint": {
                "parameter_count": checkpoint.get("parameter_count"),
                "best_epoch": checkpoint.get("best_epoch"),
                "validation_cases": checkpoint.get("val_cases"),
                "test_cases": checkpoint.get("test_cases"),
            },
        }
    )


def _run_calibration(args: argparse.Namespace) -> dict[str, Any]:
    source_manifest = _source_manifest()
    sources, input_hashes, input_contract = _load_and_validate_inputs(args)
    device = _resolve_device(args.device)
    model, adapter, _normalizer, checkpoint = load_frozen_residual_checkpoint(
        args.checkpoint, device
    )
    _validate_checkpoint_split(checkpoint)
    volumes = np.full(CONTRACT.native_cells, 1.0 / CONTRACT.native_cells)
    primary_projector = build_cosine_projector(
        CONTRACT.native_cells, rank=8, volumes=volumes
    )
    full_projector = build_cosine_projector(
        CONTRACT.native_cells, rank=CONTRACT.native_cells, volumes=volumes
    )
    snapshots, collection = _collect_snapshots(
        sources,
        EXPECTED_VALIDATION_CASES,
        model,
        adapter,
        device,
        primary_projector,
    )
    target_bands = _target_band_cache(snapshots, full_projector)
    records = [snapshot.record for snapshot in snapshots]
    full_fit = fit_case_first_scalar(records)
    early_fit = fit_case_first_scalar(
        [record for record in records if record.input_call in EARLY_CALLS]
    )
    late_fit = fit_case_first_scalar(
        [record for record in records if record.input_call in LATE_CALLS]
    )
    projected_crossfit_records = [
        replace(
            record,
            target_correction=primary_projector.project(
                record.target_correction,
                active_modes=PRIMARY_MODES,
                component_scale=PHYSICAL_COMPONENT_SCALES,
            ),
        )
        for record in records
    ]
    crossfit = crossfit_case_groups(
        projected_crossfit_records,
        expected_groups=_calibration_groups(EXPECTED_VALIDATION_CASES),
        expected_input_calls=ALL_CALLS,
    )
    stability = _coefficient_stability(full_fit, early_fit, late_fit, crossfit)
    coefficients = {
        "vortex_fixed": FIXED_COEFFICIENT,
        "family_scalar": full_fit.coefficient,
    }
    candidate_rows = {}
    for name, coefficient in coefficients.items():
        if coefficient is None:
            candidate_rows[name] = {
                "coefficient": None,
                "eligible": False,
                "checks": {"fit_resolved": False},
            }
            continue
        row = _candidate_evidence(snapshots, float(coefficient), target_bands)
        if name == "family_scalar":
            row["checks"]["coefficient_stability"] = stability["passed"]
            row["eligible"] = all(row["checks"].values())
        candidate_rows[name] = row
    common_checks = _common_checks(collection)
    eligible = [
        name
        for name in ("vortex_fixed", "family_scalar")
        if candidate_rows[name]["eligible"] and all(common_checks.values())
    ]
    selected = "zero"
    selected_coefficient = 0.0
    if eligible:
        selected = min(
            eligible,
            key=lambda name: (
                float(
                    candidate_rows[name]["view_scores"]["low_k1_k7"][
                        "rms_ratio_vs_zero"
                    ]
                ),
                0 if name == "vortex_fixed" else 1,
            ),
        )
        selected_coefficient = float(coefficients[selected])
    candidate_manifest = {
        "schema": SCHEMA,
        "selected_candidate": selected,
        "selected_coefficient": selected_coefficient,
        "input_hashes": input_hashes,
        "source_manifest": source_manifest,
        "case_ids": list(EXPECTED_VALIDATION_CASES),
        "input_calls": list(ALL_CALLS),
        "projection_modes": list(PRIMARY_MODES),
        "maximum_relative_correction": MAXIMUM_RELATIVE_CORRECTION,
    }
    candidate_manifest["candidate_sha256"] = _canonical_sha256(candidate_manifest)
    payload = {
        "schema": SCHEMA,
        "phase": "calibration",
        "status": "qualified" if selected != "zero" else "stopped_zero_selected",
        "population_status": "historically_open_validation_calibration",
        "evaluation_targets_scored": False,
        "contract": {
            "resolutions": asdict(CONTRACT),
            "component_scales": PHYSICAL_COMPONENT_SCALES.tolist(),
            "calls": list(ALL_CALLS),
            "primary_modes": list(PRIMARY_MODES),
            "fixed_coefficient": FIXED_COEFFICIENT,
            "cap": MAXIMUM_RELATIVE_CORRECTION,
        },
        "source_manifest": source_manifest,
        "input_hashes": input_hashes,
        "input_contract": input_contract,
        "checkpoint": {
            "model": checkpoint.get("model"),
            "target": checkpoint.get("target"),
            "parameter_count": checkpoint.get("parameter_count"),
            "best_epoch": checkpoint.get("best_epoch"),
            "validation_cases": list(EXPECTED_VALIDATION_CASES),
            "test_cases": list(EXPECTED_TEST_CASES),
        },
        "device": str(device),
        "collection": collection,
        "common_source_checks": common_checks,
        "coefficient_stability": stability,
        "candidates": candidate_rows,
        "selection": {
            "eligible_candidates": eligible,
            "selected_candidate": selected,
            "selected_coefficient": selected_coefficient,
            "selection_metric": "aggregate_low_k1_k7_rms_ratio",
            "tie_preference": "vortex_fixed",
        },
        "candidate_manifest": candidate_manifest,
        "structure_diagnostics": _structure_diagnostics(snapshots, full_projector),
        "claim_boundary": (
            "Calibration on the checkpoint validation cases; no D030 test correction "
            "score is computed and no recurrent run is authorized by this payload alone."
        ),
    }
    return _with_payload_sha256(payload)


def _load_calibration(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("phase") != "calibration":
        raise ValueError("calibration payload has the wrong schema or phase")
    expected = payload.get("payload_sha256")
    actual = _canonical_sha256(payload, omit=("payload_sha256",))
    if expected != actual:
        raise ValueError("calibration payload SHA-256 does not verify")
    manifest = payload.get("candidate_manifest", {})
    manifest_expected = manifest.get("candidate_sha256")
    manifest_actual = _canonical_sha256(manifest, omit=("candidate_sha256",))
    if manifest_expected != manifest_actual:
        raise ValueError("candidate manifest SHA-256 does not verify")
    if payload.get("status") != "qualified":
        raise ValueError("calibration did not qualify a nonzero candidate")
    return payload


def _run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    if args.calibration_json is None:
        raise ValueError("evaluation requires --calibration-json")
    calibration = _load_calibration(args.calibration_json)
    source_manifest = _source_manifest()
    sources, input_hashes, input_contract = _load_and_validate_inputs(args)
    if source_manifest != calibration["source_manifest"]:
        raise ValueError("source manifest differs from calibration")
    if input_hashes != calibration["input_hashes"]:
        raise ValueError("D030 input hashes differ from calibration")
    device = _resolve_device(args.device)
    model, adapter, _normalizer, checkpoint = load_frozen_residual_checkpoint(
        args.checkpoint, device
    )
    _validate_checkpoint_split(checkpoint)
    volumes = np.full(CONTRACT.native_cells, 1.0 / CONTRACT.native_cells)
    primary_projector = build_cosine_projector(
        CONTRACT.native_cells, rank=8, volumes=volumes
    )
    full_projector = build_cosine_projector(
        CONTRACT.native_cells, rank=CONTRACT.native_cells, volumes=volumes
    )
    snapshots, collection = _collect_snapshots(
        sources,
        EXPECTED_TEST_CASES,
        model,
        adapter,
        device,
        primary_projector,
    )
    target_bands = _target_band_cache(snapshots, full_projector)
    local_coefficient = calibration["coefficient_stability"]["full_fit"]["coefficient"]
    coefficients = {
        "vortex_fixed": FIXED_COEFFICIENT,
        "family_scalar": float(local_coefficient),
    }
    candidate_rows = {
        name: _candidate_evidence(snapshots, coefficient, target_bands)
        for name, coefficient in coefficients.items()
    }
    selected = calibration["selection"]["selected_candidate"]
    selected_coefficient = float(calibration["selection"]["selected_coefficient"])
    if selected not in candidate_rows or selected_coefficient != coefficients[selected]:
        raise ValueError("frozen selection differs from evaluation inventory")
    common_checks = _common_checks(collection)
    selected_gate = {
        **candidate_rows[selected]["checks"],
        "all_common_source_checks": all(common_checks.values()),
        "calibration_qualified": calibration["status"] == "qualified",
        "calibration_coefficient_stability": selected != "family_scalar"
        or calibration["coefficient_stability"]["passed"],
        "candidate_manifest_verified": True,
    }
    fixed_transfer_gate = {
        **candidate_rows["vortex_fixed"]["checks"],
        "all_common_source_checks": all(common_checks.values()),
    }
    recurrence_authorized = all(selected_gate.values())
    fixed_transfers = all(fixed_transfer_gate.values())
    if fixed_transfers:
        transfer_classification = "fixed_coefficient_transfers_on_historical_1d_family"
    elif recurrence_authorized and selected == "family_scalar":
        transfer_classification = "protocol_transfers_but_fixed_coefficient_does_not"
    else:
        transfer_classification = "registered_low_mode_scalar_does_not_transfer"
    payload = {
        "schema": SCHEMA,
        "phase": "evaluation",
        "status": "qualified" if recurrence_authorized else "teacher_gate_failed",
        "population_status": "historically_open_disjoint_d030_test",
        "calibration_payload_sha256": calibration["payload_sha256"],
        "candidate_manifest": calibration["candidate_manifest"],
        "source_manifest": source_manifest,
        "input_hashes": input_hashes,
        "input_contract": input_contract,
        "device": str(device),
        "collection": collection,
        "common_source_checks": common_checks,
        "selected_candidate": selected,
        "selected_coefficient": selected_coefficient,
        "selected_teacher_gate": selected_gate,
        "recurrence_authorized": recurrence_authorized,
        "fixed_coefficient_transfer_gate": fixed_transfer_gate,
        "fixed_coefficient_transfers": fixed_transfers,
        "transfer_classification": transfer_classification,
        "candidates": candidate_rows,
        "structure_diagnostics": _structure_diagnostics(snapshots, full_projector),
        "claim_boundary": (
            "This is disjoint for correction fitting but historically open D030 "
            "evidence. A per-case oracle is nondeployable, and no recurrent result "
            "is included in this teacher-forced payload."
        ),
    }
    return _with_payload_sha256(payload)


def synthetic_summary(seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    contract = Euler1DResolutionContract(4, 8, 16)
    fine = rng.normal(size=(16, 3))
    nested = restrict_cell_averages(
        restrict_cell_averages(fine, contract.native_cells),
        contract.coarse_cells,
    )
    direct = restrict_cell_averages(fine, contract.coarse_cells)
    volumes = np.full(contract.native_cells, 1.0 / contract.native_cells)
    projector = build_cosine_projector(contract.native_cells, rank=8, volumes=volumes)
    x = (
        np.arange(contract.native_cells, dtype=np.float64) + 0.5
    ) / contract.native_cells
    records = []
    for case in range(8):
        for call in range(2):
            feature = (0.1 + 0.01 * case + 0.02 * call) * np.column_stack(
                (
                    np.cos(np.pi * x),
                    np.cos(2.0 * np.pi * x),
                    np.cos(3.0 * np.pi * x),
                )
            )
            feature = projector.project(
                feature,
                active_modes=PRIMARY_MODES,
                component_scale=PHYSICAL_COMPONENT_SCALES,
            )
            records.append(
                ScalarFitRecord(
                    case_id=f"case_{case}",
                    input_call=call,
                    feature=feature,
                    target_correction=-0.4 * feature,
                    native_increment=np.ones_like(feature),
                    volumes=volumes,
                    component_scale=PHYSICAL_COMPONENT_SCALES,
                )
            )
    groups = {
        "fold_0": tuple(f"case_{case}" for case in range(4)),
        "fold_1": tuple(f"case_{case}" for case in range(4, 8)),
    }
    fit = fit_case_first_scalar(records)
    crossfit = crossfit_case_groups(
        records, expected_groups=groups, expected_input_calls=(0, 1)
    )
    corrections, audits = corrections_for_coefficient(records, float(fit.coefficient))
    score = score_records(corrections)
    checks = {
        "nested_restriction": float(np.max(np.abs(nested - direct))) <= 1.0e-15,
        "fit_recovers_scalar": abs(float(fit.coefficient) + 0.4) <= 1.0e-12,
        "crossfit_complete": crossfit["status"] == "ok",
        "score_closes": abs(float(score["skill_vs_zero"]) - 1.0) <= 1.0e-12,
        "mean_neutral": max(
            float(row["maximum_component_integral_abs"]) for row in audits
        )
        <= 1.0e-12,
        "datasets_closed": True,
        "checkpoint_closed": True,
    }
    return _with_payload_sha256(
        {
            "schema": SCHEMA,
            "phase": "dry_run",
            "status": "passed" if all(checks.values()) else "failed",
            "checks": checks,
            "fit": asdict(fit),
            "crossfit": crossfit,
            "score": score,
        }
    )


def _parser() -> argparse.ArgumentParser:
    root = Path("artifacts/time_dependent_no/d030_restriction_consistent_f64_20260716")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("dry-run", "preflight", "calibration", "evaluation"),
        required=True,
    )
    parser.add_argument(
        "--coarse-data",
        type=Path,
        default=root / "family/restriction_nx128_from_nx512.npz",
    )
    parser.add_argument(
        "--native-data",
        type=Path,
        default=root / "family/restriction_nx256_from_nx512.npz",
    )
    parser.add_argument(
        "--fine-data",
        type=Path,
        default=root / "data/euler1d_exact_f64_nx512_seed2401.npz",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=root / "shared_final/checkpoint.pt"
    )
    parser.add_argument(
        "--family-contract", type=Path, default=root / "family/contract.json"
    )
    parser.add_argument("--calibration-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.phase == "dry-run":
        payload = synthetic_summary(args.seed)
        output_name = "dry_run.json"
    elif args.phase == "preflight":
        payload = _run_preflight(args)
        output_name = "preflight.json"
    elif args.phase == "calibration":
        payload = _run_calibration(args)
        output_name = "calibration.json"
    else:
        payload = _run_evaluation(args)
        output_name = "evaluation.json"
    output_path = args.output_dir / output_name
    _write_json(output_path, payload)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "phase": payload["phase"],
                "status": payload["status"],
                "payload_sha256": payload["payload_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
