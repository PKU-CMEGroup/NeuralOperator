#!/usr/bin/env python3
"""Pair the frozen D044 PCNO with a bounded coarse-CFD error--cost ladder.

The audit reads only D044 validation artifacts.  Four nested coarse grids are
screened on the frozen D013 six-case cohort; at most two grids are extended to
all 24 validation trajectories by a deterministic cost/error rule.  The 27
strength-OOD test cases remain sealed.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import fields, replace
import hashlib
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

from utility.time_dependent_no.pcno_ripple_diagnostics import (  # noqa: E402
    raw_admissibility_summary,
    weighted_relative_l2_numpy,
)
from utility.time_dependent_no.shock_vortex_coarse_cfd import (  # noqa: E402
    prolong_piecewise_constant,
    remapping_oracle,
    restrict_uniform_cell_averages,
    run_coarse_cfd_rollout,
)
from utility.time_dependent_no.shock_vortex_fv import (  # noqa: E402
    ShockVortexFVConfig,
)
from utility.time_dependent_no.shock_vortex_metrics import (  # noqa: E402
    endpoint_metrics,
    physical_call_metrics,
)

SCHEMA = "pcno_shock_vortex_coarse_cfd_error_cost_v1"
SOURCE_SCHEMA = "pcno_shock_vortex_physical_baseline_v1"
EXPECTED_CHECKPOINT_SHA256 = (
    "c5e468c7045bf5ff8ccdd5f222c19bd63dab15af54b0461a58e26af5e17f678f"
)
DEFAULT_GRIDS = ("25x10", "50x20", "125x50", "250x100")
DEFAULT_PILOT_KEYS = (
    "sv_e00_y00",
    "sv_e00_y08",
    "sv_e06_y00",
    "sv_e06_y08",
    "sv_e11_y00",
    "sv_e11_y08",
)
DEFAULT_ENDPOINT_CALLS = (1, 5, 10, 20, 40, 60)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d044-evaluation-dir", type=Path, required=True)
    parser.add_argument("--timing-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--candidate-grids", nargs="+", default=DEFAULT_GRIDS)
    parser.add_argument(
        "--pilot-trajectory-keys", nargs="+", default=DEFAULT_PILOT_KEYS
    )
    parser.add_argument(
        "--endpoint-calls", nargs="+", type=int, default=DEFAULT_ENDPOINT_CALLS
    )
    parser.add_argument("--expected-trajectory-count", type=int, default=24)
    parser.add_argument(
        "--expected-checkpoint-sha256", default=EXPECTED_CHECKPOINT_SHA256
    )
    parser.add_argument("--max-full-grids", type=int, default=2)
    parser.add_argument("--timing-p95-median-ratio-max", type=float, default=2.0)
    parser.add_argument("--matched-cost-ratio-max", type=float, default=2.0)
    parser.add_argument("--matched-error-ratio-max", type=float, default=1.25)
    parser.add_argument("--own-balance-tolerance", type=float, default=5.0e-5)
    parser.add_argument("--case-bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260722)
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
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(value), indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path.name}")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for name in row:
            if name not in seen:
                seen.add(name)
                fieldnames.append(name)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    name: (
                        json.dumps(json_safe(row.get(name)))
                        if isinstance(row.get(name), (dict, list, tuple, np.ndarray))
                        else json_safe(row.get(name))
                    )
                    for name in fieldnames
                }
            )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _select_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _parse_grid(value: str) -> tuple[int, int]:
    pieces = value.lower().split("x")
    if len(pieces) != 2:
        raise ValueError(f"grid must use NXxNY syntax: {value}")
    nx, ny = (int(piece) for piece in pieces)
    if nx < 6 or ny < 6:
        raise ValueError(
            "every WENO5 coarse grid must have at least six cells per axis"
        )
    return nx, ny


def _grid_name(grid: tuple[int, int]) -> str:
    return f"{grid[0]}x{grid[1]}"


def _scalar(array: np.ndarray) -> Any:
    return np.asarray(array).reshape(()).item()


def _ratio_distance(first: float, second: float) -> float:
    if first <= 0.0 or second <= 0.0:
        raise ValueError("matching quantities must be positive")
    return abs(math.log(first / second))


def _symmetric_ratio(first: float, second: float) -> float:
    return max(first / second, second / first)


def validate_inputs(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    if args.expected_trajectory_count < 1:
        raise ValueError("expected trajectory count must be positive")
    if args.max_full_grids < 1 or args.max_full_grids > 2:
        raise ValueError("this bounded audit permits one or two full-validation grids")
    if args.case_bootstrap_repetitions < 1:
        raise ValueError("case bootstrap repetitions must be positive")
    for name in (
        "timing_p95_median_ratio_max",
        "matched_cost_ratio_max",
        "matched_error_ratio_max",
        "own_balance_tolerance",
    ):
        if float(getattr(args, name)) <= 0.0:
            raise ValueError(f"{name} must be positive")

    grids = tuple(_parse_grid(value) for value in args.candidate_grids)
    if len(grids) < 2 or len(set(grids)) != len(grids):
        raise ValueError("candidate grids must contain at least two unique entries")
    source_summary_path = args.d044_evaluation_dir / "summary.json"
    source = json.loads(source_summary_path.read_text(encoding="utf-8"))
    timing = json.loads(args.timing_summary.read_text(encoding="utf-8"))
    if source.get("schema") != SOURCE_SCHEMA or source.get("status") != "complete":
        raise ValueError("D044 evaluation summary is not the completed frozen schema")
    evaluation = source.get("evaluation", {})
    if (
        evaluation.get("split") != "validation"
        or evaluation.get("raw_recurrence") is not True
    ):
        raise ValueError("D044 source must be the raw validation rollout")
    if any(
        bool(value) for value in evaluation.get("inference_interventions", {}).values()
    ):
        raise ValueError("D044 source declares an inference intervention")
    keys = tuple(str(value) for value in evaluation.get("trajectory_keys", ()))
    if len(keys) != args.expected_trajectory_count or len(set(keys)) != len(keys):
        raise ValueError("D044 validation trajectory count or uniqueness changed")
    pilot_keys = tuple(str(value) for value in args.pilot_trajectory_keys)
    if len(set(pilot_keys)) != len(pilot_keys) or not set(pilot_keys).issubset(keys):
        raise ValueError("pilot trajectory keys must be unique D044 validation cases")
    num_steps = int(evaluation.get("num_steps", 0))
    endpoint_calls = tuple(sorted(set(int(value) for value in args.endpoint_calls)))
    if not endpoint_calls or endpoint_calls[0] < 1 or endpoint_calls[-1] > num_steps:
        raise ValueError("endpoint calls must lie within the frozen rollout")

    checkpoint = source.get("checkpoint", {})
    if checkpoint.get("sha256") != args.expected_checkpoint_sha256:
        raise ValueError("D044 checkpoint SHA-256 changed")
    if timing.get("schema") != SOURCE_SCHEMA or timing.get("status") != "complete":
        raise ValueError("PCNO timing calibration is not a completed baseline artifact")
    timing_checkpoint = timing.get("checkpoint", {})
    for field in ("sha256", "config_digest", "normalization_digest"):
        if timing_checkpoint.get(field) != checkpoint.get(field):
            raise ValueError(f"PCNO timing calibration {field} differs from D044")
    timing_evaluation = timing.get("evaluation", {})
    if (
        timing_evaluation.get("split") != "validation"
        or timing_evaluation.get("device") != evaluation.get("device")
        or timing_evaluation.get("amp") != evaluation.get("amp")
        or timing_evaluation.get("raw_recurrence") is not True
    ):
        raise ValueError("PCNO timing calibration changed the inference contract")
    if any(
        bool(value)
        for value in timing_evaluation.get("inference_interventions", {}).values()
    ):
        raise ValueError("PCNO timing calibration declares an intervention")
    cost = timing.get("cost", {})
    median = float(cost.get("batch1_seconds_median", 0.0))
    p95 = float(cost.get("batch1_seconds_p95", 0.0))
    if median <= 0.0 or p95 <= 0.0 or p95 < median:
        raise ValueError("PCNO timing calibration contains invalid latency")

    target_nx = target_ny = None
    artifact_map: dict[str, dict[str, Any]] = {}
    for item in source.get("trajectory_artifacts", ()):
        artifact_map[str(item["trajectory"])] = dict(item)
    if set(artifact_map) != set(keys):
        raise ValueError("D044 trajectory artifact registry is incomplete")
    first_path = args.d044_evaluation_dir / artifact_map[keys[0]]["artifact"]
    if sha256_file(first_path) != artifact_map[keys[0]]["sha256"]:
        raise ValueError("first D044 trajectory artifact digest changed")
    with np.load(first_path, allow_pickle=False) as artifact:
        config = json.loads(str(_scalar(artifact["reference_config_json"])))
        target_nx = int(config["coarse_nx"])
        target_ny = int(config["coarse_ny"])
    for nx, ny in grids:
        if target_nx % nx or target_ny % ny:
            raise ValueError(f"candidate grid {nx}x{ny} is not nested in D044")

    source_files = {
        name: args.d044_evaluation_dir / name
        for name in (
            "summary.json",
            "call_metrics.csv",
            "endpoint_metrics.csv",
            "trajectory_metrics.csv",
        )
    }
    if not all(path.is_file() for path in source_files.values()):
        raise FileNotFoundError("D044 source tables are incomplete")
    return {
        "source": source,
        "timing": timing,
        "keys": keys,
        "pilot_keys": pilot_keys,
        "grids": grids,
        "endpoint_calls": endpoint_calls,
        "num_steps": num_steps,
        "target_nx": target_nx,
        "target_ny": target_ny,
        "artifact_map": artifact_map,
        "source_files": source_files,
        "pcno_horizon_seconds": num_steps * median,
        "pcno_horizon_p95_envelope_seconds": num_steps * p95,
        "pcno_timing_p95_median_ratio": p95 / median,
    }


def load_case(
    source_dir: Path,
    key: str,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    registry = contract["artifact_map"][key]
    path = source_dir / registry["artifact"]
    digest = sha256_file(path)
    if digest != registry["sha256"]:
        raise ValueError(f"D044 trajectory artifact digest changed for {key}")
    names = (
        "schema",
        "trajectory",
        "split",
        "parameters_json",
        "reference_config_json",
        "gamma",
        "state_component_scale",
        "shock_quantile",
        "physical_times",
        "initial_state",
        "targets",
        "predictions",
        "positions",
        "edges",
        "physical_cell_volumes",
        "mesh_cell_to_graph_node",
        "reference_interval_boundary_exchange",
        "checkpoint_sha256",
        "checkpoint_config_digest",
        "data_manifest_digest",
        "geometry_digest",
        "normalization_digest",
    )
    with np.load(path, allow_pickle=False) as artifact:
        missing = sorted(set(names) - set(artifact.files))
        if missing:
            raise ValueError(f"D044 trajectory {key} is missing fields: {missing}")
        result = {name: np.array(artifact[name], copy=True) for name in names}
    if str(_scalar(result["schema"])) != SOURCE_SCHEMA:
        raise ValueError(f"D044 trajectory schema changed for {key}")
    if (
        str(_scalar(result["trajectory"])) != key
        or str(_scalar(result["split"])) != "validation"
    ):
        raise ValueError(f"D044 trajectory identity changed for {key}")
    checkpoint = contract["source"]["checkpoint"]
    scalar_contract = {
        "checkpoint_sha256": checkpoint["sha256"],
        "checkpoint_config_digest": checkpoint["config_digest"],
        "normalization_digest": checkpoint["normalization_digest"],
    }
    for name, expected in scalar_contract.items():
        if str(_scalar(result[name])) != expected:
            raise ValueError(f"D044 trajectory {key} changed {name}")
    mapping = np.asarray(result["mesh_cell_to_graph_node"], dtype=np.int64)
    if not np.array_equal(mapping, np.arange(mapping.size)):
        raise ValueError(f"D044 mesh-to-graph identity map failed for {key}")
    target_cells = int(contract["target_nx"]) * int(contract["target_ny"])
    volumes = np.asarray(result["physical_cell_volumes"], dtype=np.float64)
    if (
        volumes.shape != (target_cells,)
        or not np.all(np.isfinite(volumes))
        or np.any(volumes <= 0.0)
        or not np.allclose(volumes, volumes[0], rtol=0.0, atol=1.0e-14)
    ):
        raise ValueError(f"D044 trajectory {key} lacks the frozen uniform FV volumes")
    if np.asarray(result["initial_state"]).shape != (target_cells, 4):
        raise ValueError(f"D044 trajectory {key} initial-state shape changed")
    if np.asarray(result["targets"]).shape[:2] != (
        int(contract["num_steps"]),
        target_cells,
    ):
        raise ValueError(f"D044 trajectory {key} target shape changed")
    result["path"] = path
    result["sha256"] = digest
    result["parameters"] = json.loads(str(_scalar(result["parameters_json"])))
    result["reference_config"] = json.loads(
        str(_scalar(result["reference_config_json"]))
    )
    return result


def make_coarse_config(
    case: Mapping[str, Any],
    grid: tuple[int, int],
    num_steps: int,
) -> ShockVortexFVConfig:
    allowed = {field.name for field in fields(ShockVortexFVConfig)}
    payload = {
        name: value
        for name, value in case["reference_config"].items()
        if name in allowed
    }
    payload.update(case["parameters"])
    times = np.asarray(case["physical_times"], dtype=np.float64)[: num_steps + 1]
    payload.update(
        nx=grid[0],
        ny=grid[1],
        coarse_nx=grid[0],
        coarse_ny=grid[1],
        t_final=float(times[-1]),
        output_times=tuple(float(value) for value in times),
    )
    return ShockVortexFVConfig(**payload).validated()


def select_full_grids(
    pilot_aggregates: Sequence[Mapping[str, Any]],
    *,
    pcno_seconds: float,
    pcno_error: float,
    maximum: int,
) -> dict[str, Any]:
    if not pilot_aggregates:
        raise ValueError("pilot aggregates are empty")
    ordered = sorted(pilot_aggregates, key=lambda row: int(row["cells"]))
    cost_ranked = sorted(
        ordered,
        key=lambda row: (
            _ratio_distance(float(row["median_core_seconds"]), pcno_seconds),
            int(row["cells"]),
        ),
    )
    error_ranked = sorted(
        ordered,
        key=lambda row: (
            _ratio_distance(float(row["mean_horizon_error"]), pcno_error),
            int(row["cells"]),
        ),
    )
    selected = [str(cost_ranked[0]["grid"])]
    if str(error_ranked[0]["grid"]) not in selected and len(selected) < maximum:
        selected.append(str(error_ranked[0]["grid"]))
    if len(selected) < maximum:
        combined = sorted(
            ordered,
            key=lambda row: (
                _ratio_distance(float(row["median_core_seconds"]), pcno_seconds)
                + _ratio_distance(float(row["mean_horizon_error"]), pcno_error),
                int(row["cells"]),
            ),
        )
        for row in combined:
            if str(row["grid"]) not in selected:
                selected.append(str(row["grid"]))
                break
    return {
        "selected_grids": selected,
        "pilot_cost_match_grid": str(cost_ranked[0]["grid"]),
        "pilot_error_match_grid": str(error_ranked[0]["grid"]),
        "rule": (
            "closest pilot median core time and closest pilot mean H60 error in "
            "absolute log ratio; if identical, add the remaining grid minimizing "
            "the sum of those two distances; at most two grids"
        ),
    }


def paired_case_bootstrap(
    differences: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size < 1 or not np.all(np.isfinite(values)):
        raise ValueError("paired differences must be a finite nonempty vector")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(repetitions, values.size))
    means = np.mean(values[indices], axis=1)
    return {
        "mean_cfd_minus_pcno": float(np.mean(values)),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "pcno_lower_error_fraction": float(np.mean(values > 0.0)),
    }


def _mean(rows: Sequence[Mapping[str, Any]], name: str) -> float:
    values = [float(row[name]) for row in rows]
    if not values:
        raise ValueError(f"cannot aggregate empty metric: {name}")
    return float(np.mean(values))


def warm_up_grid(
    case: Mapping[str, Any],
    grid: tuple[int, int],
    contract: Mapping[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    config = make_coarse_config(case, grid, contract["num_steps"])
    warm_config = replace(
        config,
        t_final=float(config.output_times[1]),
        output_times=(float(config.output_times[0]), float(config.output_times[1])),
    ).validated()
    native_initial = restrict_uniform_cell_averages(
        case["initial_state"],
        target_nx=contract["target_nx"],
        target_ny=contract["target_ny"],
        coarse_nx=grid[0],
        coarse_ny=grid[1],
    )
    run_coarse_cfd_rollout(
        native_initial,
        warm_config,
        device=device,
        dtype=dtype,
    )


def evaluate_case(
    case: Mapping[str, Any],
    grid: tuple[int, int],
    contract: Mapping[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
    stage: str,
    trajectory_dir: Path,
) -> dict[str, Any]:
    key = str(_scalar(case["trajectory"]))
    target_nx = int(contract["target_nx"])
    target_ny = int(contract["target_ny"])
    num_steps = int(contract["num_steps"])
    config = make_coarse_config(case, grid, num_steps)
    native_initial = restrict_uniform_cell_averages(
        case["initial_state"],
        target_nx=target_nx,
        target_ny=target_ny,
        coarse_nx=grid[0],
        coarse_ny=grid[1],
    )
    runner_started = perf_counter()
    rollout = run_coarse_cfd_rollout(
        native_initial,
        config,
        device=device,
        dtype=dtype,
    )
    runner_seconds = perf_counter() - runner_started
    if rollout.states.shape != (num_steps + 1, grid[0] * grid[1], 4):
        raise ValueError(
            f"coarse CFD output shape changed for {key} on {_grid_name(grid)}"
        )

    targets = np.asarray(case["targets"], dtype=np.float64)[:num_steps]
    if targets.shape != (num_steps, target_nx * target_ny, 4):
        raise ValueError(f"D044 target shape changed for {key}")
    prolonged = prolong_piecewise_constant(
        rollout.states,
        target_nx=target_nx,
        target_ny=target_ny,
        coarse_nx=grid[0],
        coarse_ny=grid[1],
    )
    oracle = remapping_oracle(
        targets,
        target_nx=target_nx,
        target_ny=target_ny,
        coarse_nx=grid[0],
        coarse_ny=grid[1],
    )
    initial = np.asarray(case["initial_state"], dtype=np.float64)
    coarse_initial = prolonged[0]
    volumes = np.asarray(case["physical_cell_volumes"], dtype=np.float64)
    positions = np.asarray(case["positions"], dtype=np.float64)
    edges = np.asarray(case["edges"], dtype=np.int64)
    scale = np.asarray(case["state_component_scale"], dtype=np.float64)
    gamma = float(_scalar(case["gamma"]))
    shock_quantile = float(_scalar(case["shock_quantile"]))
    physical_times = np.asarray(case["physical_times"], dtype=np.float64)
    reference_interval_exchange = np.asarray(
        case["reference_interval_boundary_exchange"], dtype=np.float64
    )[:num_steps]
    if reference_interval_exchange.shape != (num_steps, 4):
        raise ValueError(f"D044 reference boundary exchange shape changed for {key}")
    reference_cumulative = np.cumsum(reference_interval_exchange, axis=0)
    coarse_cumulative = np.cumsum(rollout.interval_boundary_exchange, axis=0)
    admissibility = raw_admissibility_summary(prolonged[1:].reshape(-1, 4), gamma=gamma)

    reference_config = case["reference_config"]
    upstream_speed = float(reference_config["shock_mach"]) * math.sqrt(
        float(reference_config["gamma"])
    )
    shock_arrival_time = (
        float(reference_config["shock_x"]) - float(reference_config["vortex_x"])
    ) / upstream_speed
    vortex_y = float(case["parameters"]["vortex_y"])

    call_rows: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []
    for call in range(1, num_steps + 1):
        prediction = prolonged[call]
        target = targets[call - 1]
        floor = oracle[call - 1]
        reference_physical = physical_call_metrics(
            prediction,
            target,
            initial,
            volumes=volumes,
            reference_cumulative_boundary_exchange=reference_cumulative[call - 1],
            component_scale=scale,
        )
        own_physical = physical_call_metrics(
            prediction,
            prediction,
            coarse_initial,
            volumes=volumes,
            reference_cumulative_boundary_exchange=coarse_cumulative[call - 1],
            component_scale=scale,
        )
        call_rows.append(
            {
                "trajectory": key,
                "grid": _grid_name(grid),
                "stage": stage,
                "call": call,
                "physical_time": float(physical_times[call] - physical_times[0]),
                "scaled_relative_l2_physical_volume": weighted_relative_l2_numpy(
                    prediction, target, volumes, scale
                ),
                "oracle_remapping_error_floor": weighted_relative_l2_numpy(
                    floor, target, volumes, scale
                ),
                "physical_total_component_scaled_rmse": reference_physical[
                    "physical_total_component_scaled_rmse"
                ],
                "reference_boundary_balance_component_scaled_rmse": reference_physical[
                    "prediction_reference_balance_component_scaled_rmse"
                ],
                "own_boundary_balance_component_scaled_rmse": own_physical[
                    "prediction_reference_balance_component_scaled_rmse"
                ],
                "reference_outward_boundary_exchange": reference_physical[
                    "reference_outward_boundary_exchange"
                ],
                "coarse_cfd_outward_boundary_exchange": coarse_cumulative[call - 1],
            }
        )
        if call in contract["endpoint_calls"]:
            absolute_time = float(physical_times[call])
            expected_vortex_x = (
                float(reference_config["vortex_x"]) + upstream_speed * absolute_time
                if absolute_time <= shock_arrival_time
                else float(reference_config["shock_x"])
                + float(reference_config["right_u"])
                * (absolute_time - shock_arrival_time)
            )
            predicted_metrics = endpoint_metrics(
                prediction,
                target,
                positions=positions,
                edges=edges,
                volumes=volumes,
                component_scale=scale,
                gamma=gamma,
                shock_quantile=shock_quantile,
                vortex_center=(expected_vortex_x, vortex_y),
            )
            oracle_metrics = endpoint_metrics(
                floor,
                target,
                positions=positions,
                edges=edges,
                volumes=volumes,
                component_scale=scale,
                gamma=gamma,
                shock_quantile=shock_quantile,
                vortex_center=(expected_vortex_x, vortex_y),
            )
            endpoint_rows.append(
                {
                    "trajectory": key,
                    "grid": _grid_name(grid),
                    "stage": stage,
                    "call": call,
                    **predicted_metrics,
                    **{
                        f"oracle_{name}": value
                        for name, value in oracle_metrics.items()
                    },
                }
            )

    endpoint_indices = np.asarray(contract["endpoint_calls"], dtype=np.int64)
    native_endpoint_targets = restrict_uniform_cell_averages(
        targets[endpoint_indices - 1],
        target_nx=target_nx,
        target_ny=target_ny,
        coarse_nx=grid[0],
        coarse_ny=grid[1],
    )
    artifact_name = f"coarse_cfd_{_grid_name(grid)}_{key}.npz"
    artifact_path = trajectory_dir / artifact_name
    np.savez_compressed(
        artifact_path,
        schema=np.asarray(SCHEMA),
        trajectory=np.asarray(key),
        grid=np.asarray(_grid_name(grid)),
        stage=np.asarray(stage),
        parameters_json=np.asarray(json.dumps(case["parameters"], sort_keys=True)),
        endpoint_calls=endpoint_indices,
        endpoint_physical_times=physical_times[endpoint_indices],
        native_initial_state=native_initial.astype(np.float32),
        native_endpoint_predictions=rollout.states[endpoint_indices].astype(np.float32),
        native_endpoint_restricted_targets=native_endpoint_targets.astype(np.float32),
        interval_outward_boundary_exchange=rollout.interval_boundary_exchange,
        source_d044_trajectory_artifact=np.asarray(case["path"].name),
        source_d044_trajectory_sha256=np.asarray(case["sha256"]),
        checkpoint_sha256=np.asarray(str(_scalar(case["checkpoint_sha256"]))),
        checkpoint_config_digest=np.asarray(
            str(_scalar(case["checkpoint_config_digest"]))
        ),
        data_manifest_digest=np.asarray(str(_scalar(case["data_manifest_digest"]))),
        geometry_digest=np.asarray(str(_scalar(case["geometry_digest"]))),
        normalization_digest=np.asarray(str(_scalar(case["normalization_digest"]))),
        remapping_operator=np.asarray(
            "uniform conservative block restriction plus piecewise-constant prolongation"
        ),
        boundary_mode=np.asarray(
            "solver-native linear x extrapolation and reflecting y symmetry"
        ),
        future_reference_boundary_values=np.asarray(False),
        clipping_or_state_floors=np.asarray(False),
        core_seconds=np.asarray(rollout.core_seconds, dtype=np.float64),
    )
    artifact_contract = {
        "trajectory": key,
        "grid": _grid_name(grid),
        "artifact": f"trajectories/{artifact_name}",
        "sha256": sha256_file(artifact_path),
        "source_d044_trajectory_sha256": case["sha256"],
    }
    trajectory_row = {
        "trajectory": key,
        "grid": _grid_name(grid),
        "stage": stage,
        "parameters": case["parameters"],
        "cells": grid[0] * grid[1],
        "completed": True,
        "all_finite": bool(admissibility["all_finite"]),
        "all_admissible": bool(admissibility["all_admissible"]),
        "minimum_density": rollout.minimum_density,
        "minimum_pressure": rollout.minimum_pressure,
        "accepted_steps": rollout.accepted_steps,
        "rejected_attempts": rollout.rejected_attempts,
        "face_reconstruction_fallbacks": rollout.face_reconstruction_fallbacks,
        "core_seconds": rollout.core_seconds,
        "runner_seconds_including_host_export": runner_seconds,
        "maximum_own_boundary_balance_component_scaled_rmse": max(
            float(row["own_boundary_balance_component_scaled_rmse"])
            for row in call_rows
        ),
    }
    return {
        "trajectory_row": trajectory_row,
        "call_rows": call_rows,
        "endpoint_rows": endpoint_rows,
        "artifact_contract": artifact_contract,
    }


def aggregate_grid(
    grid_name: str,
    keys: Sequence[str],
    trajectory_rows: Sequence[Mapping[str, Any]],
    call_rows: Sequence[Mapping[str, Any]],
    *,
    num_steps: int,
    scope: str,
) -> dict[str, Any]:
    selected_keys = set(keys)
    trajectories = [
        row
        for row in trajectory_rows
        if row["grid"] == grid_name and row["trajectory"] in selected_keys
    ]
    horizon = [
        row
        for row in call_rows
        if row["grid"] == grid_name
        and row["trajectory"] in selected_keys
        and int(row["call"]) == num_steps
    ]
    if len(trajectories) != len(selected_keys) or len(horizon) != len(selected_keys):
        raise ValueError(f"grid {grid_name} lacks complete {scope} rows")
    core = np.asarray([float(row["core_seconds"]) for row in trajectories])
    return {
        "grid": grid_name,
        "scope": scope,
        "trajectories": len(selected_keys),
        "cells": int(trajectories[0]["cells"]),
        "mean_horizon_error": _mean(horizon, "scaled_relative_l2_physical_volume"),
        "mean_horizon_remapping_floor": _mean(horizon, "oracle_remapping_error_floor"),
        "mean_horizon_reference_budget_error": _mean(
            horizon, "reference_boundary_balance_component_scaled_rmse"
        ),
        "maximum_own_boundary_balance_component_scaled_rmse": max(
            float(row["maximum_own_boundary_balance_component_scaled_rmse"])
            for row in trajectories
        ),
        "median_core_seconds": float(np.median(core)),
        "mean_core_seconds": float(np.mean(core)),
        "p95_core_seconds": float(np.quantile(core, 0.95)),
        "sequential_trajectories_per_second": float(len(core) / np.sum(core)),
        "accepted_steps_mean": _mean(trajectories, "accepted_steps"),
        "all_completed": all(bool(row["completed"]) for row in trajectories),
        "all_finite": all(bool(row["all_finite"]) for row in trajectories),
        "all_admissible": all(bool(row["all_admissible"]) for row in trajectories),
    }


def aggregate_endpoints(
    grid_name: str,
    keys: Sequence[str],
    endpoint_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    selected_keys = set(keys)
    rows = [
        row
        for row in endpoint_rows
        if row["grid"] == grid_name and row["trajectory"] in selected_keys
    ]
    metric_names = (
        "front_iou",
        "front_centroid_distance",
        "front_symmetric_chamfer",
        "shock_thickness_log_error",
        "shock_strength_log_error",
        "vortex_core_density_relative_error",
        "smooth_region_scaled_relative_l2",
        "smooth_region_graph_highpass_energy",
        "oracle_front_iou",
        "oracle_front_symmetric_chamfer",
        "oracle_shock_thickness_log_error",
        "oracle_shock_strength_log_error",
        "oracle_smooth_region_scaled_relative_l2",
    )
    result: list[dict[str, Any]] = []
    for call in sorted({int(row["call"]) for row in rows}):
        group = [row for row in rows if int(row["call"]) == call]
        aggregate: dict[str, Any] = {
            "grid": grid_name,
            "call": call,
            "trajectories": len(group),
        }
        for name in metric_names:
            values = [float(row[name]) for row in group if row.get(name) is not None]
            aggregate[f"mean_{name}"] = float(np.mean(values)) if values else None
        result.append(aggregate)
    return result


def load_pcno_point(contract: Mapping[str, Any]) -> dict[str, Any]:
    source = contract["source"]
    num_steps = int(contract["num_steps"])
    keys = set(contract["keys"])
    call_rows = read_csv(contract["source_files"]["call_metrics.csv"])
    horizon_rows = [
        row
        for row in call_rows
        if row["variant"] == "pcno_baseline"
        and int(row["call"]) == num_steps
        and row["trajectory"] in keys
    ]
    if len(horizon_rows) != len(keys):
        raise ValueError("D044 PCNO horizon table is incomplete")
    per_case_error = {
        row["trajectory"]: float(row["scaled_relative_l2_physical_volume"])
        for row in horizon_rows
    }
    mean_error = float(np.mean(list(per_case_error.values())))
    frozen_error = float(
        source["aggregates"]["pcno_baseline"]["common_endpoint_mean_relative_l2"]
    )
    if not np.isclose(mean_error, frozen_error, rtol=0.0, atol=1.0e-12):
        raise ValueError("D044 PCNO horizon error does not reproduce its summary")
    trajectory_rows = read_csv(contract["source_files"]["trajectory_metrics.csv"])
    observed = [
        float(row["total_forward_seconds"])
        for row in trajectory_rows
        if row["variant"] == "pcno_baseline" and row["trajectory"] in keys
    ]
    if len(observed) != len(keys):
        raise ValueError("D044 PCNO trajectory timing table is incomplete")
    timing_cost = contract["timing"]["cost"]
    source_cost = contract["source"]["cost"]
    batch_throughput = float(source_cost["throughput_samples_per_second"])
    return {
        "mean_horizon_error": mean_error,
        "pilot_mean_horizon_error": float(
            np.mean([per_case_error[key] for key in contract["pilot_keys"]])
        ),
        "per_case_horizon_error": per_case_error,
        "calibrated_horizon_seconds": float(contract["pcno_horizon_seconds"]),
        "calibrated_horizon_p95_envelope_seconds": float(
            contract["pcno_horizon_p95_envelope_seconds"]
        ),
        "timing_p95_median_ratio": float(contract["pcno_timing_p95_median_ratio"]),
        "frozen_observed_mean_forward_seconds": float(np.mean(observed)),
        "frozen_observed_median_forward_seconds": float(np.median(observed)),
        "amortized_horizon_trajectories_per_second": batch_throughput / num_steps,
        "batch1_contract": timing_cost["contract"],
        "timing_batch_size": int(source_cost["throughput_batch_size"]),
    }


def compare_grid(
    aggregate: Mapping[str, Any],
    call_rows: Sequence[Mapping[str, Any]],
    pcno: Mapping[str, Any],
    contract: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    grid_name = str(aggregate["grid"])
    num_steps = int(contract["num_steps"])
    cfd_by_case = {
        str(row["trajectory"]): float(row["scaled_relative_l2_physical_volume"])
        for row in call_rows
        if row["grid"] == grid_name
        and int(row["call"]) == num_steps
        and row["trajectory"] in set(contract["keys"])
    }
    if set(cfd_by_case) != set(pcno["per_case_horizon_error"]):
        raise ValueError(f"paired error table is incomplete for {grid_name}")
    differences = np.asarray(
        [
            cfd_by_case[key] - pcno["per_case_horizon_error"][key]
            for key in contract["keys"]
        ]
    )
    cost_ratio = _symmetric_ratio(
        float(aggregate["median_core_seconds"]),
        float(pcno["calibrated_horizon_seconds"]),
    )
    error_ratio = _symmetric_ratio(
        float(aggregate["mean_horizon_error"]), float(pcno["mean_horizon_error"])
    )
    pcno_cost = float(pcno["calibrated_horizon_seconds"])
    cfd_cost = float(aggregate["median_core_seconds"])
    pcno_error = float(pcno["mean_horizon_error"])
    cfd_error = float(aggregate["mean_horizon_error"])
    return {
        "grid": grid_name,
        "cfd_mean_horizon_error": cfd_error,
        "pcno_mean_horizon_error": pcno_error,
        "error_symmetric_ratio": error_ratio,
        "within_matched_error_ratio": error_ratio <= args.matched_error_ratio_max,
        "cfd_median_core_seconds": cfd_cost,
        "pcno_calibrated_horizon_seconds": pcno_cost,
        "cost_symmetric_ratio": cost_ratio,
        "within_matched_cost_ratio": cost_ratio <= args.matched_cost_ratio_max,
        "pcno_pareto_dominates": bool(
            pcno_cost <= cfd_cost
            and pcno_error <= cfd_error
            and (pcno_cost < cfd_cost or pcno_error < cfd_error)
        ),
        "coarse_cfd_pareto_dominates": bool(
            cfd_cost <= pcno_cost
            and cfd_error <= pcno_error
            and (cfd_cost < pcno_cost or cfd_error < pcno_error)
        ),
        **paired_case_bootstrap(
            differences,
            repetitions=args.case_bootstrap_repetitions,
            seed=args.bootstrap_seed,
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    contract = validate_inputs(args)
    device = _select_device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    pcno = load_pcno_point(contract)
    args.output_dir.mkdir(parents=True)
    trajectory_dir = args.output_dir / "trajectories"
    trajectory_dir.mkdir()

    trajectory_rows: list[dict[str, Any]] = []
    call_rows: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []
    artifact_contracts: list[dict[str, Any]] = []
    pilot_cases = {
        key: load_case(args.d044_evaluation_dir, key, contract)
        for key in contract["pilot_keys"]
    }
    for grid in contract["grids"]:
        warm_up_grid(
            pilot_cases[contract["pilot_keys"][0]],
            grid,
            contract,
            device=device,
            dtype=dtype,
        )
        for key in contract["pilot_keys"]:
            result = evaluate_case(
                pilot_cases[key],
                grid,
                contract,
                device=device,
                dtype=dtype,
                stage="pilot",
                trajectory_dir=trajectory_dir,
            )
            trajectory_rows.append(result["trajectory_row"])
            call_rows.extend(result["call_rows"])
            endpoint_rows.extend(result["endpoint_rows"])
            artifact_contracts.append(result["artifact_contract"])

    pilot_aggregates = [
        aggregate_grid(
            _grid_name(grid),
            contract["pilot_keys"],
            trajectory_rows,
            call_rows,
            num_steps=contract["num_steps"],
            scope="pilot",
        )
        for grid in contract["grids"]
    ]
    selection = select_full_grids(
        pilot_aggregates,
        pcno_seconds=pcno["calibrated_horizon_seconds"],
        pcno_error=pcno["pilot_mean_horizon_error"],
        maximum=args.max_full_grids,
    )
    selected_names = set(selection["selected_grids"])
    selected_grids = [
        grid for grid in contract["grids"] if _grid_name(grid) in selected_names
    ]
    remaining_keys = [
        key for key in contract["keys"] if key not in set(contract["pilot_keys"])
    ]
    for key in remaining_keys:
        case = load_case(args.d044_evaluation_dir, key, contract)
        for grid in selected_grids:
            result = evaluate_case(
                case,
                grid,
                contract,
                device=device,
                dtype=dtype,
                stage="full_extension",
                trajectory_dir=trajectory_dir,
            )
            trajectory_rows.append(result["trajectory_row"])
            call_rows.extend(result["call_rows"])
            endpoint_rows.extend(result["endpoint_rows"])
            artifact_contracts.append(result["artifact_contract"])

    for rows in (trajectory_rows, call_rows, endpoint_rows):
        for row in rows:
            row["full_validation_member"] = row["grid"] in selected_names
    full_aggregates = [
        aggregate_grid(
            _grid_name(grid),
            contract["keys"],
            trajectory_rows,
            call_rows,
            num_steps=contract["num_steps"],
            scope="full_validation",
        )
        for grid in selected_grids
    ]
    endpoint_aggregates = [
        row
        for grid in selected_grids
        for row in aggregate_endpoints(
            _grid_name(grid), contract["keys"], endpoint_rows
        )
    ]
    comparisons = [
        compare_grid(aggregate, call_rows, pcno, contract, args)
        for aggregate in full_aggregates
    ]
    matched_cost = min(
        comparisons, key=lambda row: (float(row["cost_symmetric_ratio"]), row["grid"])
    )
    matched_error = min(
        comparisons,
        key=lambda row: (float(row["error_symmetric_ratio"]), row["grid"]),
    )
    timing_stable = pcno["timing_p95_median_ratio"] <= args.timing_p95_median_ratio_max
    selected_all_admissible = all(
        bool(row["all_admissible"] and row["all_finite"]) for row in full_aggregates
    )
    selected_balance_closed = all(
        float(row["maximum_own_boundary_balance_component_scaled_rmse"])
        <= args.own_balance_tolerance
        for row in full_aggregates
    )
    gates = {
        "source_validation_only_and_digest_bound": True,
        "test_split_accessed": False,
        "pcno_timing_stable": timing_stable,
        "selected_grids_complete_and_admissible": selected_all_admissible,
        "selected_grids_own_boundary_balance_closed": selected_balance_closed,
        "observed_matched_cost_member": bool(matched_cost["within_matched_cost_ratio"]),
        "observed_matched_error_member": bool(
            matched_error["within_matched_error_ratio"]
        ),
    }

    tables = {
        "trajectory_metrics.csv": trajectory_rows,
        "call_metrics.csv": call_rows,
        "endpoint_metrics.csv": endpoint_rows,
        "grid_aggregates.csv": [*pilot_aggregates, *full_aggregates],
        "endpoint_aggregates.csv": endpoint_aggregates,
        "paired_comparisons.csv": comparisons,
    }
    for name, rows in tables.items():
        write_csv(args.output_dir / name, rows)
    table_sha256 = {name: sha256_file(args.output_dir / name) for name in tables}
    source_sha256 = {
        name: sha256_file(path) for name, path in contract["source_files"].items()
    }
    summary = {
        "schema": SCHEMA,
        "status": "complete",
        "source_contract": {
            "d044_schema": contract["source"]["schema"],
            "d044_summary_sha256": source_sha256["summary.json"],
            "d044_table_sha256": source_sha256,
            "checkpoint": contract["source"]["checkpoint"],
            "evaluation": contract["source"]["evaluation"],
            "timing_summary_name": args.timing_summary.name,
            "timing_summary_sha256": sha256_file(args.timing_summary),
        },
        "evaluation_contract": {
            "split": "validation",
            "trajectory_keys": list(contract["keys"]),
            "pilot_trajectory_keys": list(contract["pilot_keys"]),
            "strength_ood_test_accessed": False,
            "candidate_grids": [_grid_name(grid) for grid in contract["grids"]],
            "selected_full_validation_grids": selection["selected_grids"],
            "physical_horizon": contract["source"]["evaluation"]["physical_horizon"],
            "num_steps": contract["num_steps"],
            "endpoint_calls": list(contract["endpoint_calls"]),
            "device": str(device),
            "dtype": args.dtype,
            "coarse_solver": "WENO5-HLLC-SSPRK3 with adaptive CFL and raw accepted states",
            "boundary_mode": "linear x extrapolation and reflecting y symmetry",
            "future_reference_boundary_values": False,
            "clipping_floors_limiters_or_smoothing": False,
        },
        "cost_contract": {
            "pcno": pcno,
            "coarse_cfd": (
                "synchronized native-grid evolution including CFL reductions, "
                "WENO5-HLLC-SSPRK3, retries, retained save states, and the solver's "
                "own boundary sums; excludes initialization, host export, remapping, "
                "metrics, and artifact I/O"
            ),
            "coarse_cfd_throughput": (
                "sequential trajectories per second; the current reference implementation "
                "does not batch independent CFD cases"
            ),
        },
        "remapping_contract": {
            "operator": (
                "exact uniform block average from 250x100 to the nested coarse grid, "
                "then conservative piecewise-constant prolongation to 250x100"
            ),
            "oracle_floor": (
                "the D044 target passed through that same restrict-prolong operator; "
                "it is an information/remapping floor, not a CFD prediction"
            ),
        },
        "selection": selection,
        "pilot_aggregates": pilot_aggregates,
        "full_validation_aggregates": full_aggregates,
        "paired_comparisons": comparisons,
        "matched_cost_observed_member": matched_cost,
        "matched_error_observed_member": matched_error,
        "gates": gates,
        "strict_error_cost_claim_valid": bool(
            timing_stable
            and selected_all_admissible
            and selected_balance_closed
            and matched_cost["within_matched_cost_ratio"]
        ),
        "claim_boundary": {
            "verified": (
                "paired validation error and measured implementation cost against the "
                "frozen D044 PCNO, with explicit conservative remapping floors and the "
                "measured coarse-solver own-boundary balance residual"
            ),
            "not_verified": [
                "production-CFD performance",
                "batched coarse-CFD throughput",
                "strength-OOD test performance",
                "neural physical face flux or conservation by construction",
                "method or architecture superiority beyond the frozen implementations",
                "seed uncertainty; the paired interval resamples validation cases only",
            ],
        },
        "artifact_contract": {
            "trajectory_npz": (
                "native initial state, endpoint predictions and restricted targets, own "
                "boundary exchange, timing, parameters, source hashes, remapping and "
                "boundary contracts; full D044 targets remain in the digest-bound source"
            ),
            "trajectory_artifacts": artifact_contracts,
        },
        "row_counts": {name: len(rows) for name, rows in tables.items()},
        "table_sha256": table_sha256,
        "entrypoint_sha256": sha256_file(Path(__file__).resolve()),
        "utility_sha256": {
            "coarse_cfd": sha256_file(
                ROOT / "utility/time_dependent_no/shock_vortex_coarse_cfd.py"
            ),
            "finite_volume_solver": sha256_file(
                ROOT / "utility/time_dependent_no/shock_vortex_fv.py"
            ),
            "shared_metrics": sha256_file(
                ROOT / "utility/time_dependent_no/shock_vortex_metrics.py"
            ),
        },
    }
    atomic_write_json(args.output_dir / "summary.json", summary)
    print(json.dumps({"summary": str(args.output_dir / "summary.json")}, indent=2))


if __name__ == "__main__":
    main()
