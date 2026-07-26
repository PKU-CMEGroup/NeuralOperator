#!/usr/bin/env python3
"""Run the D061 frozen stride-1/stride-2 rollout headroom diagnostic.

The two input rollouts are legal, raw, independently recurrent PCNO forecasts
on the same validation trajectories. D061 aligns the stride-1 forecast at even
saved frames with stride 2, evaluates a fixed 50/50 blend, and selects a
truth-informed scalar convex blend from one frozen alpha grid. The oracle is a
representational upper bound, not an autonomous solver. No checkpoint is
executed and no model is trained by this entry point.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no.evaluate_pcno_shock_vortex_baseline import (  # noqa: E402
    SCHEMA as BASELINE_SCHEMA,
    atomic_write_json,
    endpoint_metrics,
    json_safe,
    physical_call_metrics,
    sha256_file,
    write_csv,
)
from utility.time_dependent_no.pcno_ripple_diagnostics import (  # noqa: E402
    raw_admissibility_summary,
    spatial_correlation_summary,
    weighted_relative_l2_numpy,
)

SCHEMA = "pcno_shock_vortex_multirate_headroom_v1"
EXPERIMENT_ID = "D061"
DEFAULT_TRAJECTORY_KEYS = (
    "sv_e00_y00",
    "sv_e00_y08",
    "sv_e06_y00",
    "sv_e06_y08",
    "sv_e11_y00",
    "sv_e11_y08",
)
DEFAULT_CALLS = (1, 5, 15, 30)
ALPHA_GRID = np.linspace(0.0, 1.0, 21, dtype=np.float64)
MIN_MEDIAN_STATE_REDUCTION = 0.10
MIN_MEDIAN_STRIDE2_HIGHPASS_REDUCTION = 0.20
MIN_JOINT_NONWORSE_CASES = 5
MIN_ANTI_SMEARING_CASES = 5
MAX_ANTI_SMEARING_WORSENING = 0.05
MIN_SENSOR_SPEARMAN = 0.50
MIN_SENSOR_TOP20_CAPTURE = 0.50
ANTI_SMEARING_METRICS = (
    "front_centroid_distance",
    "front_symmetric_chamfer",
    "shock_strength_log_error",
    "shock_thickness_log_error",
    "vortex_core_density_relative_error",
    "prediction_reference_balance_component_scaled_rmse",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stride1-dir", type=Path, required=True)
    parser.add_argument("--stride2-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--trajectory-keys",
        nargs="+",
        default=list(DEFAULT_TRAJECTORY_KEYS),
        help="Defaults to the frozen six-case D013 validation cohort.",
    )
    parser.add_argument("--calls", type=int, nargs="+", default=DEFAULT_CALLS)
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _scalar(value: np.ndarray) -> Any:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError("expected a scalar artifact field")
    return array.item()


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), sort_keys=True, allow_nan=False))
            handle.write("\n")
    os.replace(temporary, path)


def _validated_summary(
    directory: Path, *, expected_stride: int, keys: Sequence[str]
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]], dict[str, Any]]:
    summary_path = directory / "summary.json"
    summary = _load_json(summary_path)
    evaluation = summary.get("evaluation", {})
    if summary.get("schema") != BASELINE_SCHEMA or summary.get("status") != "complete":
        raise ValueError(f"incomplete physical baseline summary: {summary_path}")
    if evaluation.get("split") != "validation":
        raise ValueError("D061 accepts validation rollouts only")
    if int(evaluation.get("step_stride", -1)) != expected_stride:
        raise ValueError(f"expected a stride-{expected_stride} source")
    if evaluation.get("boundary_mode") != "model_all_nodes":
        raise ValueError("source rollout does not use the legal boundary contract")
    if evaluation.get("raw_recurrence") is not True:
        raise ValueError("source rollout is not raw recurrence")
    interventions = evaluation.get("inference_interventions", {})
    if not isinstance(interventions, dict) or any(
        bool(value) for value in interventions.values()
    ):
        raise ValueError("source rollout contains an inference intervention")
    declared = set(map(str, evaluation.get("trajectory_keys", [])))
    missing = sorted(set(keys) - declared)
    if missing:
        raise ValueError(f"source summary lacks requested validation cases: {missing}")
    artifacts = summary.get("trajectory_artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("source trajectory artifact index is missing")
    by_key = {
        str(row["trajectory"]): row
        for row in artifacts
        if isinstance(row, dict) and "trajectory" in row
    }
    missing = sorted(set(keys) - set(by_key))
    if missing:
        raise ValueError(f"source artifact index lacks cases: {missing}")
    source = {
        "directory_name": directory.name,
        "summary_sha256": sha256_file(summary_path),
        "checkpoint_sha256": summary["checkpoint"]["sha256"],
        "checkpoint_config_digest": summary["checkpoint"]["config_digest"],
        "normalization_digest": summary["checkpoint"]["normalization_digest"],
        "data_manifest_digest": summary["data_contract"]["data_manifest_digest"],
        "step_stride": expected_stride,
        "num_steps": int(evaluation["num_steps"]),
        "physical_horizon": float(evaluation["physical_horizon"]),
    }
    return summary, by_key, source


REQUIRED_FIELDS = {
    "schema",
    "trajectory",
    "split",
    "parameters_json",
    "reference_config_json",
    "gamma",
    "state_component_scale",
    "shock_quantile",
    "physical_times",
    "physical_delta_t",
    "initial_state",
    "targets",
    "predictions",
    "positions",
    "edges",
    "node_type",
    "physical_cell_volumes",
    "mesh_cell_to_graph_node",
    "face_centers",
    "face_measures",
    "face_normals",
    "face_owner",
    "face_neighbor",
    "face_boundary_tag",
    "coordinate_convention",
    "face_orientation_convention",
    "reference_interval_boundary_exchange",
    "valid_length",
    "failure_cause",
    "checkpoint_sha256",
    "checkpoint_config_digest",
    "data_manifest_digest",
    "geometry_digest",
    "normalization_digest",
    "boundary_mode",
    "raw_recurrence",
    "inference_interventions_json",
}


def _load_trajectory(
    directory: Path,
    *,
    key: str,
    index_row: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    path = directory / str(index_row["artifact"])
    if sha256_file(path) != index_row.get("sha256"):
        raise ValueError(f"trajectory digest mismatch for {key}")
    with np.load(path, allow_pickle=False) as artifact:
        missing = sorted(REQUIRED_FIELDS - set(artifact.files))
        if missing:
            raise ValueError(f"trajectory {key} is missing fields: {missing}")
        result = {name: np.array(artifact[name], copy=True) for name in artifact.files}
    if _scalar(result["schema"]) != BASELINE_SCHEMA:
        raise ValueError(f"unexpected trajectory schema for {key}")
    if _scalar(result["trajectory"]) != key or _scalar(result["split"]) != "validation":
        raise ValueError(f"trajectory identity or split mismatch for {key}")
    if _scalar(result["checkpoint_sha256"]) != source["checkpoint_sha256"]:
        raise ValueError(f"checkpoint digest mismatch for {key}")
    if (
        _scalar(result["checkpoint_config_digest"])
        != source["checkpoint_config_digest"]
    ):
        raise ValueError(f"configuration digest mismatch for {key}")
    if _scalar(result["data_manifest_digest"]) != source["data_manifest_digest"]:
        raise ValueError(f"data manifest digest mismatch for {key}")
    if _scalar(result["boundary_mode"]) != "model_all_nodes":
        raise ValueError(f"illegal boundary mode for {key}")
    if bool(_scalar(result["raw_recurrence"])) is not True:
        raise ValueError(f"non-raw recurrence for {key}")
    interventions = json.loads(str(_scalar(result["inference_interventions_json"])))
    if any(bool(value) for value in interventions.values()):
        raise ValueError(f"trajectory {key} contains an inference intervention")
    return result


def _assert_close(
    name: str, left: np.ndarray, right: np.ndarray, *, atol: float
) -> float:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape:
        raise ValueError(
            f"{name} shapes differ: {left_array.shape} vs {right_array.shape}"
        )
    difference = float(
        np.max(np.abs(left_array.astype(np.float64) - right_array.astype(np.float64)))
    )
    if difference > atol:
        raise ValueError(f"{name} differs by {difference:.6g}, above {atol:.6g}")
    return difference


def validate_trajectory_pair(
    stride1: Mapping[str, np.ndarray],
    stride2: Mapping[str, np.ndarray],
    *,
    required_stride2_calls: int,
) -> dict[str, float]:
    """Validate aligned state, geometry, normalization, and recurrence contracts."""

    if int(_scalar(stride1["valid_length"])) < 2 * required_stride2_calls:
        raise ValueError("stride-1 rollout does not reach the requested horizon")
    if int(_scalar(stride2["valid_length"])) < required_stride2_calls:
        raise ValueError("stride-2 rollout does not reach the requested horizon")
    if (
        _scalar(stride1["failure_cause"]) != "completed"
        or _scalar(stride2["failure_cause"]) != "completed"
    ):
        raise ValueError("D061 requires completed parent rollouts")
    checks: dict[str, float] = {}
    for name, tolerance in (
        ("initial_state", 1.0e-7),
        ("positions", 1.0e-7),
        ("physical_cell_volumes", 1.0e-10),
        ("face_centers", 1.0e-12),
        ("face_measures", 1.0e-12),
        ("face_normals", 1.0e-12),
        ("physical_times", 1.0e-12),
        ("reference_interval_boundary_exchange", 1.0e-12),
        ("state_component_scale", 1.0e-10),
    ):
        checks[f"maximum_{name}_difference"] = _assert_close(
            name, stride1[name], stride2[name], atol=tolerance
        )
    for name in (
        "edges",
        "node_type",
        "mesh_cell_to_graph_node",
        "face_owner",
        "face_neighbor",
        "face_boundary_tag",
    ):
        if not np.array_equal(stride1[name], stride2[name]):
            raise ValueError(f"integer geometry field {name} differs")
    for name in (
        "coordinate_convention",
        "face_orientation_convention",
        "geometry_digest",
    ):
        if _scalar(stride1[name]) != _scalar(stride2[name]):
            raise ValueError(f"contract field {name} differs")
    targets1 = np.asarray(stride1["targets"][: 2 * required_stride2_calls])
    targets2 = np.asarray(stride2["targets"][:required_stride2_calls])
    checks["maximum_even_frame_target_difference"] = _assert_close(
        "even-frame targets", targets1[1::2], targets2, atol=1.0e-7
    )
    return checks


def convex_blend(direct: np.ndarray, composed: np.ndarray, alpha: float) -> np.ndarray:
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0,1]")
    direct_array = np.asarray(direct, dtype=np.float64)
    composed_array = np.asarray(composed, dtype=np.float64)
    if direct_array.shape != composed_array.shape:
        raise ValueError("parent proposals must have identical shapes")
    return (1.0 - alpha) * direct_array + alpha * composed_array


def select_oracle_blend(
    direct: np.ndarray,
    composed: np.ndarray,
    target: np.ndarray,
    *,
    volumes: np.ndarray,
    component_scale: np.ndarray,
    gamma: float,
    alpha_grid: np.ndarray = ALPHA_GRID,
) -> dict[str, Any]:
    """Select one admissible global blend by held-out state error."""

    grid = np.asarray(alpha_grid, dtype=np.float64)
    if grid.ndim != 1 or grid.size < 2 or np.any(np.diff(grid) <= 0.0):
        raise ValueError("alpha grid must be a strictly increasing vector")
    if grid[0] != 0.0 or grid[-1] != 1.0:
        raise ValueError("alpha grid must contain both parent endpoints")
    rows = []
    predictions = []
    for alpha in grid:
        prediction = convex_blend(direct, composed, float(alpha))
        admissibility = raw_admissibility_summary(prediction, gamma=gamma)
        error = weighted_relative_l2_numpy(prediction, target, volumes, component_scale)
        rows.append(
            {
                "alpha": float(alpha),
                "state_error": error,
                "raw_admissible": bool(admissibility["all_admissible"]),
                "all_finite": bool(admissibility["all_finite"]),
            }
        )
        predictions.append(prediction)
    eligible = [
        index
        for index, row in enumerate(rows)
        if row["raw_admissible"] and row["all_finite"]
    ]
    if not eligible:
        raise ValueError("no admissible convex blend exists")
    selected = min(eligible, key=lambda index: (rows[index]["state_error"], index))
    return {
        "alpha": rows[selected]["alpha"],
        "prediction": predictions[selected],
        "state_error": rows[selected]["state_error"],
        "curve": rows,
    }


def _relative_reduction(candidate: float, baseline: float) -> float:
    if not math.isfinite(candidate) or not math.isfinite(baseline) or baseline < 0.0:
        raise ValueError("relative reduction requires finite nonnegative metrics")
    tolerance = 1.0e-15
    if baseline <= tolerance:
        return 0.0 if candidate <= tolerance else -math.inf
    return (baseline - candidate) / baseline


def _top_fraction_capture(
    score: np.ndarray,
    error_amplitude: np.ndarray,
    volumes: np.ndarray,
    *,
    fraction: float = 0.20,
) -> float:
    values = np.asarray(score, dtype=np.float64).reshape(-1)
    error = np.asarray(error_amplitude, dtype=np.float64).reshape(-1)
    weights = np.asarray(volumes, dtype=np.float64).reshape(-1)
    if values.shape != error.shape or values.shape != weights.shape:
        raise ValueError("sensor, error, and volume arrays must align")
    count = max(1, int(math.ceil(fraction * values.size)))
    selected = np.argsort(values, kind="mergesort")[-count:]
    energy = weights * error**2
    return float(np.sum(energy[selected]) / max(float(np.sum(energy)), 1.0e-30))


def proposal_sensor_metrics(
    direct: np.ndarray,
    composed: np.ndarray,
    target: np.ndarray,
    *,
    component_scale: np.ndarray,
    volumes: np.ndarray,
) -> dict[str, Any]:
    scale = np.asarray(component_scale, dtype=np.float64).reshape(1, -1)
    disagreement = np.linalg.norm((direct - composed) / scale, axis=-1)
    direct_error = np.linalg.norm((direct - target) / scale, axis=-1)
    composed_error = np.linalg.norm((composed - target) / scale, axis=-1)
    better_parent_error = np.minimum(direct_error, composed_error)
    correlation = spatial_correlation_summary(
        better_parent_error,
        {"direct_composed_disagreement": disagreement},
    )["direct_composed_disagreement"]
    return {
        "correlation_with_better_parent_error": correlation,
        "top20_disagreement_capture_of_better_parent_error_energy": (
            _top_fraction_capture(disagreement, better_parent_error, volumes)
        ),
    }


def _variant_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    initial: np.ndarray,
    *,
    positions: np.ndarray,
    edges: np.ndarray,
    volumes: np.ndarray,
    component_scale: np.ndarray,
    gamma: float,
    shock_quantile: float,
    vortex_center: tuple[float, float],
    cumulative_boundary_exchange: np.ndarray,
) -> dict[str, Any]:
    endpoint = endpoint_metrics(
        prediction,
        target,
        positions=positions,
        edges=edges,
        volumes=volumes,
        component_scale=component_scale,
        gamma=gamma,
        shock_quantile=shock_quantile,
        vortex_center=vortex_center,
    )
    physical = physical_call_metrics(
        prediction,
        target,
        initial,
        volumes=volumes,
        reference_cumulative_boundary_exchange=cumulative_boundary_exchange,
        component_scale=component_scale,
    )
    admissibility = raw_admissibility_summary(prediction, gamma=gamma)
    return {
        "scaled_relative_l2_physical_volume": weighted_relative_l2_numpy(
            prediction, target, volumes, component_scale
        ),
        **endpoint,
        "smooth_region_graph_highpass_rms": math.sqrt(
            endpoint["smooth_region_graph_highpass_energy"]
        ),
        **physical,
        "admissibility": admissibility,
    }


def anti_smearing_acceptance(
    direct: Mapping[str, Any],
    composed: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    max_relative_worsening: float = MAX_ANTI_SMEARING_WORSENING,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for name in ANTI_SMEARING_METRICS:
        values = (direct.get(name), composed.get(name), candidate.get(name))
        if any(value is None or not math.isfinite(float(value)) for value in values):
            report[name] = {
                "accepted": False,
                "best_parent": None,
                "candidate": None,
            }
            continue
        best = min(float(values[0]), float(values[1]))
        proposed = float(values[2])
        tolerance = 1.0e-12 * max(1.0, best)
        report[name] = {
            "accepted": proposed <= (1.0 + max_relative_worsening) * best + tolerance,
            "best_parent": best,
            "candidate": proposed,
        }
    return {
        "passed": set(report) == set(ANTI_SMEARING_METRICS)
        and all(item["accepted"] for item in report.values()),
        "metrics": report,
    }


def promotion_decision(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_trajectories: int,
    expected_calls: Sequence[int] = DEFAULT_CALLS,
) -> dict[str, Any]:
    expected = expected_trajectories * len(expected_calls)
    h60_call = max(map(int, expected_calls))
    h60 = [row for row in rows if int(row["call"]) == h60_call]
    all_rows = len(rows) == expected and len(h60) == expected_trajectories
    all_admissible = all_rows and all(
        bool(row["all_variants_raw_admissible"]) for row in rows
    )
    state_reductions = [
        float(row["oracle_state_reduction_vs_better_parent"]) for row in h60
    ]
    highpass_reductions = [
        float(row["oracle_highpass_reduction_vs_stride2"]) for row in h60
    ]
    median_state = float(np.median(state_reductions)) if state_reductions else None
    median_highpass = (
        float(np.median(highpass_reductions)) if highpass_reductions else None
    )
    joint_count = sum(
        bool(row["oracle_state_and_highpass_nonworse_than_both"]) for row in h60
    )
    anti_count = sum(bool(row["oracle_anti_smearing"]["passed"]) for row in h60)
    gates = {
        "all_expected_rows_available": all_rows,
        "all_parent_equal_and_oracle_states_raw_admissible": all_admissible,
        "h60_median_state_reduction_vs_better_parent_at_least_0p10": (
            median_state is not None and median_state >= MIN_MEDIAN_STATE_REDUCTION
        ),
        "h60_median_highpass_reduction_vs_stride2_at_least_0p20": (
            median_highpass is not None
            and median_highpass >= MIN_MEDIAN_STRIDE2_HIGHPASS_REDUCTION
        ),
        "h60_state_and_highpass_nonworse_than_both_in_at_least_5_of_6": (
            joint_count >= MIN_JOINT_NONWORSE_CASES
        ),
        "h60_anti_smearing_envelope_passes_in_at_least_5_of_6": (
            anti_count >= MIN_ANTI_SMEARING_CASES
        ),
    }
    passed = all(gates.values())
    late = [row for row in rows if int(row["call"]) >= 15]
    spearman = [
        float(
            row["proposal_sensor"]["correlation_with_better_parent_error"]["spearman"]
        )
        for row in late
        if row["proposal_sensor"]["correlation_with_better_parent_error"]["spearman"]
        is not None
    ]
    capture = [
        float(
            row["proposal_sensor"][
                "top20_disagreement_capture_of_better_parent_error_energy"
            ]
        )
        for row in late
    ]
    median_spearman = float(np.median(spearman)) if spearman else None
    median_capture = float(np.median(capture)) if capture else None
    autonomous_indicator = {
        "median_late_spearman": median_spearman,
        "median_late_top20_error_energy_capture": median_capture,
        "passed": (
            median_spearman is not None
            and median_spearman >= MIN_SENSOR_SPEARMAN
            and median_capture is not None
            and median_capture >= MIN_SENSOR_TOP20_CAPTURE
        ),
        "claim_boundary": (
            "descriptive target-free disagreement localization only; it does not "
            "select the truth-informed oracle alpha"
        ),
    }
    return {
        "passed": passed,
        "decision": (
            "authorize_one_shared_backbone_multirate_tiny_fit_only"
            if passed
            else "reject_two_rate_blend_as_next_stabilization_branch"
        ),
        "expected_rows": expected,
        "observed_rows": len(rows),
        "h60_rows": len(h60),
        "h60_median_state_reduction_vs_better_parent": median_state,
        "h60_median_highpass_reduction_vs_stride2": median_highpass,
        "h60_joint_nonworse_case_count": joint_count,
        "h60_anti_smearing_case_count": anti_count,
        "gates": gates,
        "autonomous_indicator": autonomous_indicator,
    }


def _vortex_center(
    reference_config: Mapping[str, Any],
    parameters: Mapping[str, Any],
    *,
    absolute_time: float,
) -> tuple[float, float]:
    gamma = float(reference_config["gamma"])
    upstream_speed = float(reference_config["shock_mach"]) * math.sqrt(gamma)
    shock_arrival = (
        float(reference_config["shock_x"]) - float(reference_config["vortex_x"])
    ) / upstream_speed
    if absolute_time <= shock_arrival:
        x_position = (
            float(reference_config["vortex_x"]) + upstream_speed * absolute_time
        )
    else:
        x_position = float(reference_config["shock_x"]) + float(
            reference_config["right_u"]
        ) * (absolute_time - shock_arrival)
    return x_position, float(parameters["vortex_y"])


def _cost_summary(
    stride1: Mapping[str, Any], stride2: Mapping[str, Any]
) -> dict[str, Any]:
    call1 = float(stride1["cost"]["batch1_seconds_median"])
    call2 = float(stride2["cost"]["batch1_seconds_median"])
    h60_stride1 = 60.0 * call1
    h60_stride2 = 30.0 * call2
    combined = h60_stride1 + h60_stride2
    return {
        "contract": (
            "sum of the two source evaluators' synchronized median batch-1 call "
            "times; excludes D061 offline metric and artifact cost"
        ),
        "stride1_h60_seconds_estimate": h60_stride1,
        "stride2_h60_seconds_estimate": h60_stride2,
        "two_checkpoint_blend_h60_seconds_estimate": combined,
        "two_checkpoint_ratio_to_stride1": combined / h60_stride1,
        "two_checkpoint_ratio_to_stride2": combined / h60_stride2,
        "shared_backbone_cost": "not_measured_or_claimed",
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    keys = [str(key) for key in args.trajectory_keys]
    calls = sorted(set(int(call) for call in args.calls))
    if keys != list(DEFAULT_TRAJECTORY_KEYS):
        raise ValueError("D061 requires the frozen six-case D013 trajectory order")
    if tuple(calls) != DEFAULT_CALLS:
        raise ValueError("D061 requires calls 1, 5, 15, and 30")
    stride1_summary, stride1_index, stride1_source = _validated_summary(
        args.stride1_dir, expected_stride=1, keys=keys
    )
    stride2_summary, stride2_index, stride2_source = _validated_summary(
        args.stride2_dir, expected_stride=2, keys=keys
    )
    if stride1_source["data_manifest_digest"] != stride2_source["data_manifest_digest"]:
        raise ValueError("parent rollouts use different dataset manifests")
    if not math.isclose(
        stride1_source["physical_horizon"],
        stride2_source["physical_horizon"],
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("parent rollouts have different physical horizons")

    args.output_dir.mkdir(parents=True)
    trajectory_dir = args.output_dir / "trajectories"
    trajectory_dir.mkdir()
    rows: list[dict[str, Any]] = []
    variant_rows: list[dict[str, Any]] = []
    contracts: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []

    for key in keys:
        stride1 = _load_trajectory(
            args.stride1_dir,
            key=key,
            index_row=stride1_index[key],
            source=stride1_source,
        )
        stride2 = _load_trajectory(
            args.stride2_dir,
            key=key,
            index_row=stride2_index[key],
            source=stride2_source,
        )
        pair_checks = validate_trajectory_pair(
            stride1, stride2, required_stride2_calls=max(calls)
        )
        contracts.append({"trajectory": key, **pair_checks})

        initial = np.asarray(stride2["initial_state"], dtype=np.float64)
        targets = np.asarray(stride2["targets"], dtype=np.float64)
        direct_rollout = np.asarray(stride2["predictions"], dtype=np.float64)
        stride1_rollout = np.asarray(stride1["predictions"], dtype=np.float64)
        composed_rollout = stride1_rollout[1::2]
        positions = np.asarray(stride2["positions"], dtype=np.float64)
        edges = np.asarray(stride2["edges"], dtype=np.int64)
        volumes = np.asarray(stride2["physical_cell_volumes"], dtype=np.float64)
        component_scale = np.asarray(stride2["state_component_scale"], dtype=np.float64)
        physical_times = np.asarray(stride2["physical_times"], dtype=np.float64)
        interval_exchange = np.asarray(
            stride2["reference_interval_boundary_exchange"], dtype=np.float64
        )
        gamma = float(_scalar(stride2["gamma"]))
        shock_quantile = float(_scalar(stride2["shock_quantile"]))
        reference_config = json.loads(str(_scalar(stride2["reference_config_json"])))
        parameters = json.loads(str(_scalar(stride2["parameters_json"])))

        saved_targets = []
        saved_direct = []
        saved_composed = []
        saved_equal = []
        saved_oracle = []
        saved_direct_current = []
        saved_composed_current = []
        saved_alphas = []
        for call in calls:
            target = targets[call - 1]
            direct = direct_rollout[call - 1]
            composed = composed_rollout[call - 1]
            equal = convex_blend(direct, composed, 0.5)
            oracle = select_oracle_blend(
                direct,
                composed,
                target,
                volumes=volumes,
                component_scale=component_scale,
                gamma=gamma,
            )
            oracle_prediction = np.asarray(oracle["prediction"], dtype=np.float64)
            saved_stop = 2 * call
            cumulative_exchange = np.sum(interval_exchange[:saved_stop], axis=0)
            center = _vortex_center(
                reference_config,
                parameters,
                absolute_time=float(physical_times[saved_stop]),
            )
            predictions = {
                "direct_stride2": direct,
                "composed_stride1": composed,
                "equal_blend": equal,
                "oracle_blend": oracle_prediction,
            }
            metrics = {
                name: _variant_metrics(
                    prediction,
                    target,
                    initial,
                    positions=positions,
                    edges=edges,
                    volumes=volumes,
                    component_scale=component_scale,
                    gamma=gamma,
                    shock_quantile=shock_quantile,
                    vortex_center=center,
                    cumulative_boundary_exchange=cumulative_exchange,
                )
                for name, prediction in predictions.items()
            }
            for variant, values in metrics.items():
                variant_rows.append(
                    {
                        "trajectory": key,
                        "call": call,
                        "physical_time": float(
                            physical_times[saved_stop] - physical_times[0]
                        ),
                        "variant": variant,
                        "oracle_alpha": (
                            oracle["alpha"] if variant == "oracle_blend" else None
                        ),
                        **values,
                    }
                )
            direct_state = float(
                metrics["direct_stride2"]["scaled_relative_l2_physical_volume"]
            )
            composed_state = float(
                metrics["composed_stride1"]["scaled_relative_l2_physical_volume"]
            )
            oracle_state = float(
                metrics["oracle_blend"]["scaled_relative_l2_physical_volume"]
            )
            direct_highpass = float(
                metrics["direct_stride2"]["smooth_region_graph_highpass_rms"]
            )
            composed_highpass = float(
                metrics["composed_stride1"]["smooth_region_graph_highpass_rms"]
            )
            oracle_highpass = float(
                metrics["oracle_blend"]["smooth_region_graph_highpass_rms"]
            )
            anti_smearing = anti_smearing_acceptance(
                metrics["direct_stride2"],
                metrics["composed_stride1"],
                metrics["oracle_blend"],
            )
            sensor = proposal_sensor_metrics(
                direct,
                composed,
                target,
                component_scale=component_scale,
                volumes=volumes,
            )
            admissible = all(
                bool(values["admissibility"]["all_finite"])
                and bool(values["admissibility"]["all_admissible"])
                for values in metrics.values()
            )
            rows.append(
                {
                    "trajectory": key,
                    "call": call,
                    "physical_time": float(
                        physical_times[saved_stop] - physical_times[0]
                    ),
                    "oracle_alpha": float(oracle["alpha"]),
                    "oracle_alpha_curve": oracle["curve"],
                    "all_variants_raw_admissible": admissible,
                    "oracle_state_reduction_vs_better_parent": (
                        _relative_reduction(
                            oracle_state, min(direct_state, composed_state)
                        )
                    ),
                    "oracle_highpass_reduction_vs_stride2": (
                        _relative_reduction(oracle_highpass, direct_highpass)
                    ),
                    "oracle_highpass_reduction_vs_better_parent": (
                        _relative_reduction(
                            oracle_highpass,
                            min(direct_highpass, composed_highpass),
                        )
                    ),
                    "oracle_state_and_highpass_nonworse_than_both": (
                        oracle_state <= min(direct_state, composed_state) + 1.0e-12
                        and oracle_highpass
                        <= min(direct_highpass, composed_highpass) + 1.0e-12
                    ),
                    "oracle_anti_smearing": anti_smearing,
                    "proposal_sensor": sensor,
                    "metrics": metrics,
                }
            )
            saved_targets.append(target.astype(np.float32))
            saved_direct.append(direct.astype(np.float32))
            saved_composed.append(composed.astype(np.float32))
            saved_equal.append(equal.astype(np.float32))
            saved_oracle.append(oracle_prediction.astype(np.float32))
            saved_direct_current.append(
                (initial if call == 1 else direct_rollout[call - 2]).astype(np.float32)
            )
            saved_composed_current.append(
                (initial if call == 1 else stride1_rollout[2 * call - 3]).astype(
                    np.float32
                )
            )
            saved_alphas.append(float(oracle["alpha"]))

        artifact_path = trajectory_dir / f"trajectory_{key}.npz"
        np.savez_compressed(
            artifact_path,
            schema=np.asarray(SCHEMA),
            experiment_id=np.asarray(EXPERIMENT_ID),
            trajectory=np.asarray(key),
            split=np.asarray("validation"),
            parameters_json=stride2["parameters_json"],
            mach=np.asarray(float(reference_config["shock_mach"]), dtype=np.float64),
            calls=np.asarray(calls, dtype=np.int64),
            physical_times=physical_times[np.asarray(calls) * 2],
            physical_delta_t=np.asarray(
                2.0 * float(stride2_summary["evaluation"]["saved_delta_t"])
            ),
            initial_state=initial.astype(np.float32),
            direct_stride2_current=np.asarray(saved_direct_current),
            composed_stride1_current=np.asarray(saved_composed_current),
            targets=np.asarray(saved_targets),
            direct_stride2_predictions=np.asarray(saved_direct),
            composed_stride1_predictions=np.asarray(saved_composed),
            equal_blend_predictions=np.asarray(saved_equal),
            oracle_blend_predictions=np.asarray(saved_oracle),
            oracle_alpha=np.asarray(saved_alphas, dtype=np.float64),
            alpha_grid=ALPHA_GRID,
            positions=stride2["positions"],
            edges=stride2["edges"],
            node_type=stride2["node_type"],
            physical_cell_volumes=stride2["physical_cell_volumes"],
            mesh_cell_to_graph_node=stride2["mesh_cell_to_graph_node"],
            face_centers=stride2["face_centers"],
            face_measures=stride2["face_measures"],
            face_normals=stride2["face_normals"],
            face_owner=stride2["face_owner"],
            face_neighbor=stride2["face_neighbor"],
            face_boundary_tag=stride2["face_boundary_tag"],
            coordinate_convention=stride2["coordinate_convention"],
            face_orientation_convention=stride2["face_orientation_convention"],
            reference_interval_boundary_exchange=stride2[
                "reference_interval_boundary_exchange"
            ],
            state_component_scale=stride2["state_component_scale"],
            shock_quantile=stride2["shock_quantile"],
            stride1_valid_length=stride1["valid_length"],
            stride2_valid_length=stride2["valid_length"],
            stride1_failure_cause=stride1["failure_cause"],
            stride2_failure_cause=stride2["failure_cause"],
            stride1_checkpoint_sha256=np.asarray(stride1_source["checkpoint_sha256"]),
            stride2_checkpoint_sha256=np.asarray(stride2_source["checkpoint_sha256"]),
            stride1_config_digest=np.asarray(
                stride1_source["checkpoint_config_digest"]
            ),
            stride2_config_digest=np.asarray(
                stride2_source["checkpoint_config_digest"]
            ),
            data_manifest_digest=stride2["data_manifest_digest"],
            geometry_digest=stride2["geometry_digest"],
            boundary_mode=np.asarray("model_all_nodes"),
            raw_recurrence=np.asarray(True),
            inference_interventions_json=stride2["inference_interventions_json"],
        )
        artifacts.append(
            {
                "trajectory": key,
                "artifact": f"trajectories/{artifact_path.name}",
                "sha256": sha256_file(artifact_path),
                "stride1_source_artifact_sha256": stride1_index[key]["sha256"],
                "stride2_source_artifact_sha256": stride2_index[key]["sha256"],
            }
        )

    decision = promotion_decision(
        rows, expected_trajectories=len(keys), expected_calls=calls
    )
    _write_jsonl(args.output_dir / "proposal_rows.jsonl", rows)
    write_csv(args.output_dir / "variant_metrics.csv", variant_rows)
    summary = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "complete",
        "source_contract": {
            "stride1": stride1_source,
            "stride2": stride2_source,
            "trajectory_pair_checks": contracts,
            "same_validation_split_and_physical_horizon": True,
        },
        "evaluation": {
            "split": "validation",
            "trajectory_keys": keys,
            "calls": calls,
            "physical_frames": [2 * call for call in calls],
            "physical_horizon": stride2_source["physical_horizon"],
            "alpha_grid": ALPHA_GRID.tolist(),
            "oracle_selector": (
                "minimum physical-volume state error among raw-admissible global "
                "convex blends; target-informed and unavailable autonomously"
            ),
            "legal_control": ("fixed alpha=0.5 blend of two legal raw rollouts"),
            "test_access": False,
            "checkpoint_execution": False,
            "training": False,
        },
        "promotion": decision,
        "cost": _cost_summary(stride1_summary, stride2_summary),
        "trajectory_artifacts": artifacts,
        "artifact_schema": {
            "trajectory_npz": (
                "initial and parent current states, matched targets, direct/composed/"
                "equal/oracle predictions, physical time, graph and validated FV "
                "geometry, diagnostic weights, parent validity/failure fields, and "
                "checkpoint/config/data/geometry/boundary provenance"
            ),
            "proposal_rows_jsonl": (
                "oracle alpha curve, state/high-pass headroom, anti-smearing "
                "envelope, target-free disagreement localization, and full "
                "variant metrics"
            ),
            "variant_metrics_csv": (
                "state, shock, vortex, smooth-region, admissibility, physical-total, "
                "and reference-boundary-exchange diagnostics by parent/blend"
            ),
        },
        "claim_boundary": {
            "verified": (
                "offline behavior of two frozen raw validation rollouts and scalar "
                "convex combinations under the validated finite-volume geometry "
                "contract"
            ),
            "not_verified": [
                "an autonomous oracle-alpha selector",
                "a one-backbone multirate architecture",
                "lower inference cost than either parent",
                "model-predicted physical face flux or conservation",
                "strength-OOD or test performance",
            ],
        },
    }
    atomic_write_json(args.output_dir / "summary.json", summary)
    print(
        json.dumps(
            {
                "summary": str(args.output_dir / "summary.json"),
                "decision": decision["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
