#!/usr/bin/env python3
"""Analyze and render hash-verified D080-B local-correction results."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utility.time_dependent_no.pcno_artifacts import (
    atomic_write_json_with_paths as write_json,
)
from utility.time_dependent_no.pcno_artifacts import (
    git_state,
    jsonable_args,
    sha256_file,
    write_csv_with_paths,
)

SCHEMA = "pcno_local_correction_pilot_visualization_v1"
RESULT_IDENTITY = (
    "pcno_local_correction_pilot_d080b_v1",
    "d080b_deterministic_local_correction_pilot",
)
COMPONENTS = ("rho", "rho_u", "rho_v", "energy")
VISUAL_ARMS = (
    "zero",
    "persistent",
    "combined_shock_isotropic",
    "combined_shock_normal",
    "combined_vortex_isotropic",
)
COMBINED_ARMS = VISUAL_ARMS[2:]
LOCAL_ONLY_ARMS = {
    "combined_shock_isotropic": "local_shock_isotropic",
    "combined_shock_normal": "local_shock_normal",
    "combined_vortex_isotropic": "local_vortex_isotropic",
}
ARM_LABELS = {
    "persistent": "persistent rank-8",
    "combined_shock_isotropic": "shock isotropic",
    "combined_shock_normal": "shock normal",
    "combined_vortex_isotropic": "vortex isotropic",
}
COLORS = {
    "persistent": "#333333",
    "combined_shock_isotropic": "#0072B2",
    "combined_shock_normal": "#D55E00",
    "combined_vortex_isotropic": "#009E73",
}
CASE_COLORS = {
    "e00": "#0072B2",
    "e06": "#D55E00",
    "e11": "#009E73",
}
REQUIRED_TABLES = {
    "case_contracts.csv",
    "evaluation_call_metrics.csv",
    "evaluation_case_summary.csv",
    "evaluation_comparisons.csv",
    "lag_correlations.csv",
    "local_correction_audit.csv",
    "pod_summaries.csv",
    "sequence_summaries.csv",
    "sequence_time_metrics.csv",
    "visual_payload_inventory.csv",
}
CONTROL_METRICS = (
    ("component_residual_rms__rho", "residual rho"),
    ("component_residual_rms__rho_u", "residual rho_u"),
    ("component_residual_rms__rho_v", "residual rho_v"),
    ("component_residual_rms__energy", "residual energy"),
    ("endpoint_state__shock", "shock endpoint"),
    ("endpoint_state__vortex", "vortex endpoint"),
    ("endpoint_smooth_highpass", "smooth high-pass"),
    ("endpoint_state__boundary_distance_le_0.05", "boundary <= 0.05"),
    ("physical_volume_integral_state_scale_final_abs__rho", "final integral rho"),
    (
        "physical_volume_integral_state_scale_final_abs__rho_u",
        "final integral rho_u",
    ),
    (
        "physical_volume_integral_state_scale_final_abs__rho_v",
        "final integral rho_v",
    ),
    (
        "physical_volume_integral_state_scale_final_abs__energy",
        "final integral energy",
    ),
)
ANIMATION_COLUMNS = (
    ("true_increment", "true residual", "increment"),
    ("persistent_prediction", "persistent prediction", "increment"),
    ("combined_prediction", "combined prediction", "increment"),
    ("local_correction", "local correction", "correction"),
    ("persistent_defect", "persistent residual error", "defect"),
    ("combined_defect", "combined residual error", "defect"),
    ("combined_cumulative", "combined cumulative defect", "cumulative"),
    ("combined_growth", "combined signed error growth", "growth"),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--command", choices=("plots", "animations", "all"), default="all"
    )
    parser.add_argument("--case-ids", nargs="*")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=70)
    parser.add_argument("--frame-stride", type=int, default=1)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.fps < 1 or args.dpi < 30 or args.frame_stride < 1:
        raise ValueError("fps, dpi, and frame-stride must be positive")
    return args


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _scalar(value: np.ndarray) -> Any:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError("expected scalar payload field")
    return array.item()


def _contained_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(
            f"result path escapes results directory: {relative}"
        ) from error
    return path


def _verify_results(results_dir: Path) -> tuple[dict[str, Any], list[Path]]:
    summary_path = results_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    identity = (summary.get("schema"), summary.get("experiment_contract"))
    if identity != RESULT_IDENTITY:
        raise ValueError(f"unsupported local-correction result: {identity}")
    if not (
        summary.get("status") == "complete"
        and summary.get("contract_checks_passed") is True
        and summary.get("scientific_interpretation_allowed") is True
    ):
        raise ValueError("D080-B result is not eligible for scientific rendering")
    if summary.get("family") != "dynamic_fv":
        raise ValueError("D080-B visualization is bound to the dynamic FV family")
    if summary.get("boundary_policy") != (
        "model_all_nodes raw recurrence; base policy unchanged"
    ):
        raise ValueError("D080-B physical boundary policy changed")

    output_hashes = summary.get("output_hashes")
    if not isinstance(output_hashes, dict) or not output_hashes:
        raise ValueError("summary contains no output hashes")
    missing = sorted(REQUIRED_TABLES - set(output_hashes))
    if missing:
        raise ValueError(f"output hashes omit required tables: {missing}")
    for relative, expected in output_hashes.items():
        path = _contained_path(results_dir, str(relative))
        if not path.is_file() or sha256_file(path) != str(expected):
            raise ValueError(f"output digest mismatch: {relative}")

    inventory = _read_rows(results_dir / "visual_payload_inventory.csv")
    inventory_paths = {row["relative_path"] for row in inventory}
    hash_paths = {
        str(relative)
        for relative in output_hashes
        if str(relative).startswith("visual_payloads/")
    }
    disk_paths = {
        path.relative_to(results_dir).as_posix()
        for path in (results_dir / "visual_payloads").glob("*.npz")
    }
    if inventory_paths != hash_paths or inventory_paths != disk_paths:
        raise ValueError("visual payload inventories disagree")
    for row in inventory:
        relative = row["relative_path"]
        if row.get("sha256") != output_hashes.get(relative):
            raise ValueError(f"visual payload hash binding failed: {relative}")
        if int(row["arm_count"]) != len(VISUAL_ARMS):
            raise ValueError(f"visual arm inventory changed: {relative}")
    return summary, [
        _contained_path(results_dir, relative) for relative in sorted(hash_paths)
    ]


def _load_payload(path: Path, summary: Mapping[str, Any]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        payload = {name: np.asarray(source[name]) for name in source.files}
    if str(_scalar(payload["schema"])) != str(summary["schema"]):
        raise ValueError("visual payload schema differs from summary")

    nodes = np.asarray(payload["nodes"], dtype=np.float64)
    weights = np.asarray(payload["weights"], dtype=np.float64)
    node_type = np.asarray(payload["node_type"], dtype=np.int64)
    times = np.asarray(payload["physical_times"], dtype=np.float64)
    state_scale = np.asarray(payload["state_scale"], dtype=np.float64)
    residual_scale = np.asarray(payload["residual_scale"], dtype=np.float64)
    truth = np.asarray(payload["true_increment"], dtype=np.float64)
    reference = np.asarray(payload["reference_states"], dtype=np.float64)
    calls = truth.shape[0]
    field_shape = (calls, nodes.shape[0], len(COMPONENTS))
    state_shape = (calls + 1, nodes.shape[0], len(COMPONENTS))
    if (
        nodes.ndim != 2
        or nodes.shape[1] != 2
        or weights.shape != (nodes.shape[0],)
        or node_type.shape != (nodes.shape[0],)
        or set(np.unique(node_type)) - {0, 1, 2, 3}
        or times.shape != (calls + 1,)
        or state_scale.shape != (len(COMPONENTS),)
        or residual_scale.shape != (len(COMPONENTS),)
        or reference.shape != state_shape
        or truth.shape != field_shape
        or np.any(weights <= 0.0)
        or np.any(state_scale <= 0.0)
        or np.any(residual_scale <= 0.0)
        or not np.all(np.isfinite(nodes))
        or not np.all(np.isfinite(weights))
        or not np.all(np.isfinite(times))
        or np.any(np.diff(times) <= 0.0)
    ):
        raise ValueError("invalid visual payload geometry, clock, or scales")

    for arm in VISUAL_ARMS:
        states = np.asarray(payload[f"states__{arm}"], dtype=np.float64)
        base = np.asarray(payload[f"base_increment__{arm}"], dtype=np.float64)
        persistent = np.asarray(
            payload[f"persistent_correction__{arm}"], dtype=np.float64
        )
        local = np.asarray(payload[f"local_correction__{arm}"], dtype=np.float64)
        defect = np.asarray(payload[f"defect__{arm}"], dtype=np.float64)
        cumulative = np.asarray(payload[f"cumulative_defect__{arm}"], dtype=np.float64)
        if states.shape != state_shape:
            raise ValueError(f"invalid stored state shape for {arm}")
        if any(
            field.shape != field_shape
            for field in (base, persistent, local, defect, cumulative)
        ):
            raise ValueError(f"invalid stored field shape for {arm}")
        if not all(
            np.all(np.isfinite(field))
            for field in (states, base, persistent, local, defect, cumulative)
        ):
            raise ValueError(f"non-finite visual field for {arm}")
        prediction = base + persistent + local
        if not np.allclose(prediction - truth, defect, rtol=0.0, atol=5e-8):
            raise ValueError(f"predicted-increment identity failed for {arm}")
        if not np.allclose(np.cumsum(defect, axis=0), cumulative, rtol=0.0, atol=2e-8):
            raise ValueError(f"cumulative-defect replay failed for {arm}")
        if not np.allclose(states[1:] - reference[1:], cumulative, rtol=0.0, atol=3e-7):
            raise ValueError(f"state-error replay failed for {arm}")
        if not np.allclose(states[1:] - states[:-1], prediction, rtol=0.0, atol=6e-7):
            raise ValueError(f"recurrence replay failed for {arm}")
    return payload


def _float(row: Mapping[str, str], key: str) -> float:
    value = float(row[key])
    if not np.isfinite(value):
        raise ValueError(f"non-finite {key}")
    return value


def _median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot aggregate an empty case population")
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _quantile(values: Sequence[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def _case_metric_rows(results_dir: Path) -> list[dict[str, Any]]:
    sequences = {
        (row["case_id"], row["arm"]): row
        for row in _read_rows(results_dir / "sequence_summaries.csv")
    }
    result = []
    for row in _read_rows(results_dir / "evaluation_comparisons.csv"):
        comparison = row["comparison"]
        if not comparison.endswith("_vs_persistent"):
            continue
        arm = row["numerator_candidate"]
        if arm not in COMBINED_ARMS:
            continue
        case_id = row["case_id"]
        controls = json.loads(row["control_ratios_json"])
        arm_sequence = sequences[(case_id, arm)]
        persistent_sequence = sequences[(case_id, "persistent")]
        record: dict[str, Any] = {
            "case_id": case_id,
            "arm": arm,
            "endpoint_state_ratio": _float(row, "endpoint_state_ratio"),
            "residual_rms_ratio": _float(row, "residual_rms_ratio"),
            "cumulative_defect_ratio": (
                _float(arm_sequence, "net_relative_to_truth_change")
                / _float(persistent_sequence, "net_relative_to_truth_change")
            ),
            "maximum_control_ratio": _float(row, "maximum_control_ratio"),
        }
        for key, _ in CONTROL_METRICS:
            record[key] = float(controls[key])
        result.append(record)
    expected = 6 * len(COMBINED_ARMS)
    if len(result) != expected:
        raise ValueError(f"expected {expected} combined/persistent case rows")
    return sorted(result, key=lambda row: (row["arm"], row["case_id"]))


def _temporal_rows(
    call_rows: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    index = {(row["case_id"], row["arm"], int(row["call"])): row for row in call_rows}
    cases = sorted({case_id for case_id, arm, _ in index if arm == "persistent"})
    case_rows: list[dict[str, Any]] = []
    for arm in COMBINED_ARMS:
        calls = sorted(
            call
            for case_id, observed_arm, call in index
            if case_id == cases[0] and observed_arm == arm
        )
        for case_id in cases:
            for call in calls:
                numerator = index[(case_id, arm, call)]
                denominator = index[(case_id, "persistent", call)]
                case_rows.append(
                    {
                        "case_id": case_id,
                        "arm": arm,
                        "call": call,
                        "physical_time": _float(numerator, "physical_time"),
                        "state_error_ratio": _float(numerator, "state_error_rms")
                        / _float(denominator, "state_error_rms"),
                        "instant_defect_ratio": _float(numerator, "instant_defect_rms")
                        / _float(denominator, "instant_defect_rms"),
                        "cumulative_defect_ratio": _float(
                            numerator, "cumulative_defect_rms"
                        )
                        / _float(denominator, "cumulative_defect_rms"),
                        "state_error_numerator": _float(numerator, "state_error_rms"),
                        "state_error_denominator": _float(
                            denominator, "state_error_rms"
                        ),
                        "instant_defect_numerator": _float(
                            numerator, "instant_defect_rms"
                        ),
                        "instant_defect_denominator": _float(
                            denominator, "instant_defect_rms"
                        ),
                        "cumulative_defect_numerator": _float(
                            numerator, "cumulative_defect_rms"
                        ),
                        "cumulative_defect_denominator": _float(
                            denominator, "cumulative_defect_rms"
                        ),
                    }
                )
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        grouped[(row["arm"], row["call"])].append(row)
    aggregate_rows = []
    for (arm, call), rows in sorted(grouped.items()):
        record: dict[str, Any] = {
            "arm": arm,
            "call": call,
            "physical_time": rows[0]["physical_time"],
            "case_count": len(rows),
        }
        for metric in (
            "state_error_ratio",
            "instant_defect_ratio",
            "cumulative_defect_ratio",
        ):
            values = [float(row[metric]) for row in rows]
            record[f"median__{metric}"] = _median(values)
            record[f"q25__{metric}"] = _quantile(values, 0.25)
            record[f"q75__{metric}"] = _quantile(values, 0.75)
            record[f"nonworse_count__{metric}"] = sum(value <= 1.0 for value in values)
        aggregate_rows.append(record)
    return case_rows, aggregate_rows


def _component_region_rows(
    case_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for arm in COMBINED_ARMS:
        arm_rows = [row for row in case_rows if row["arm"] == arm]
        for key, label in CONTROL_METRICS:
            values = [float(row[key]) for row in arm_rows]
            result.append(
                {
                    "arm": arm,
                    "metric": key,
                    "label": label,
                    "case_count": len(values),
                    "median_ratio": _median(values),
                    "q25_ratio": _quantile(values, 0.25),
                    "q75_ratio": _quantile(values, 0.75),
                    "nonworse_count": sum(value <= 1.0 for value in values),
                    "maximum_ratio": max(values),
                }
            )
    return result


def _structure_rows(results_dir: Path) -> list[dict[str, Any]]:
    sequences = {
        (row["case_id"], row["arm"]): row
        for row in _read_rows(results_dir / "sequence_summaries.csv")
    }
    pods = {
        (row["case_id"], row["arm"], row["centering"]): row
        for row in _read_rows(results_dir / "pod_summaries.csv")
    }
    lags = {
        (row["case_id"], row["arm"], int(row["lag"])): _float(row, "correlation")
        for row in _read_rows(results_dir / "lag_correlations.csv")
    }
    calls = {
        (row["case_id"], row["arm"], int(row["call"])): row
        for row in _read_rows(results_dir / "evaluation_call_metrics.csv")
    }
    cases = sorted({case_id for case_id, arm in sequences if arm == "persistent"})
    result = []
    for arm in COMBINED_ARMS:
        local_arm = LOCAL_ONLY_ARMS[arm]
        coherence_ratios = []
        pod_first_three_deltas = []
        pod_mode95_deltas = []
        local_energy_ratios = []
        local_cosines = []
        for case_id in cases:
            coherence_ratios.append(
                _float(sequences[(case_id, arm)], "temporal_coherence")
                / _float(sequences[(case_id, "persistent")], "temporal_coherence")
            )
            pod_first_three_deltas.append(
                _float(
                    pods[(case_id, arm, "uncentered")],
                    "first_three_energy_fraction",
                )
                - _float(
                    pods[(case_id, "persistent", "uncentered")],
                    "first_three_energy_fraction",
                )
            )
            pod_mode95_deltas.append(
                _float(pods[(case_id, arm, "uncentered")], "modes_for_95_percent")
                - _float(
                    pods[(case_id, "persistent", "uncentered")],
                    "modes_for_95_percent",
                )
            )
            case_calls = sorted(
                call
                for observed_case, observed_arm, call in calls
                if observed_case == case_id and observed_arm == local_arm
            )
            energy_values = []
            cosine_values = []
            for call in case_calls:
                row = calls[(case_id, local_arm, call)]
                base = _float(row, "base_defect_energy_same_input")
                energy_values.append(
                    1.0 + _float(row, "corrected_minus_base_energy") / base
                )
                if row["correction_base_cosine_valid"] == "True":
                    cosine_values.append(_float(row, "correction_base_cosine"))
            local_energy_ratios.append(_median(energy_values))
            local_cosines.append(_median(cosine_values))
        result.extend(
            [
                {
                    "arm": arm,
                    "metric": "temporal_coherence_ratio",
                    "median": _median(coherence_ratios),
                    "q25": _quantile(coherence_ratios, 0.25),
                    "q75": _quantile(coherence_ratios, 0.75),
                },
                {
                    "arm": arm,
                    "metric": "pod_uncentered_first_three_delta",
                    "median": _median(pod_first_three_deltas),
                    "q25": _quantile(pod_first_three_deltas, 0.25),
                    "q75": _quantile(pod_first_three_deltas, 0.75),
                },
                {
                    "arm": arm,
                    "metric": "pod_uncentered_modes95_delta",
                    "median": _median(pod_mode95_deltas),
                    "q25": _quantile(pod_mode95_deltas, 0.25),
                    "q75": _quantile(pod_mode95_deltas, 0.75),
                },
                {
                    "arm": arm,
                    "metric": "local_only_same_input_energy_ratio",
                    "median": _median(local_energy_ratios),
                    "q25": _quantile(local_energy_ratios, 0.25),
                    "q75": _quantile(local_energy_ratios, 0.75),
                },
                {
                    "arm": arm,
                    "metric": "local_only_correction_base_cosine",
                    "median": _median(local_cosines),
                    "q25": _quantile(local_cosines, 0.25),
                    "q75": _quantile(local_cosines, 0.75),
                },
            ]
        )
        for lag in sorted(
            observed_lag
            for observed_case, observed_arm, observed_lag in lags
            if observed_case == cases[0] and observed_arm == arm
        ):
            values = [lags[(case_id, arm, lag)] for case_id in cases]
            baseline = [lags[(case_id, "persistent", lag)] for case_id in cases]
            deltas = [
                value - base for value, base in zip(values, baseline, strict=True)
            ]
            result.append(
                {
                    "arm": arm,
                    "metric": f"lag_correlation__{lag}",
                    "median": _median(values),
                    "q25": _quantile(values, 0.25),
                    "q75": _quantile(values, 0.75),
                    "persistent_median": _median(baseline),
                    "median_delta": _median(deltas),
                }
            )
    return result


def _local_audit_rows(results_dir: Path) -> list[dict[str, Any]]:
    contracts = {
        row["case_id"]: sum(json.loads(row["node_type_counts"]).values())
        for row in _read_rows(results_dir / "case_contracts.csv")
        if row["case_id"].startswith(("sv_e00", "sv_e06", "sv_e11"))
    }
    rows = _read_rows(results_dir / "local_correction_audit.csv")
    result = []
    for arm in COMBINED_ARMS:
        for case_id in sorted(contracts):
            selected = [
                row for row in rows if row["arm"] == arm and row["case_id"] == case_id
            ]
            caps = [_float(row, "norm_cap") for row in selected]
            applied = [_float(row, "applied_relative_norm") for row in selected]
            result.append(
                {
                    "case_id": case_id,
                    "arm": arm,
                    "calls": len(selected),
                    "median_active_node_fraction": _median(
                        [
                            int(row["active_node_count"]) / contracts[case_id]
                            for row in selected
                        ]
                    ),
                    "median_eligible_edge_count": _median(
                        [int(row["eligible_edge_count"]) for row in selected]
                    ),
                    "norm_cap": caps[0],
                    "cap_hit_fraction": float(
                        np.mean(
                            np.isclose(
                                np.asarray(applied),
                                np.asarray(caps),
                                rtol=0.0,
                                atol=1e-12,
                            )
                        )
                    ),
                    "maximum_cap_excess": max(
                        _float(row, "cap_excess") for row in selected
                    ),
                    "maximum_local_mean_closure_scaled": max(
                        _float(row, "maximum_local_mean_closure_scaled")
                        for row in selected
                    ),
                    "maximum_non_type0_local_correction": max(
                        _float(row, "maximum_non_type0_local_correction")
                        for row in selected
                    ),
                }
            )
    return result


def _analysis_summary(
    summary: Mapping[str, Any],
    case_rows: Sequence[Mapping[str, Any]],
    temporal_rows: Sequence[Mapping[str, Any]],
    component_rows: Sequence[Mapping[str, Any]],
    structure_rows: Sequence[Mapping[str, Any]],
    local_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_temporal = {(row["arm"], row["call"]): row for row in temporal_rows}
    by_structure = {(row["arm"], row["metric"]): row for row in structure_rows}
    arms = {}
    for arm in COMBINED_ARMS:
        cases = [row for row in case_rows if row["arm"] == arm]
        calls = sorted(
            call for observed_arm, call in by_temporal if observed_arm == arm
        )
        state = [
            by_temporal[(arm, call)]["median__state_error_ratio"] for call in calls
        ]
        cumulative = [
            by_temporal[(arm, call)]["median__cumulative_defect_ratio"]
            for call in calls
        ]
        local = [row for row in local_rows if row["arm"] == arm]
        target_key = (
            "endpoint_state__vortex"
            if arm == "combined_vortex_isotropic"
            else "endpoint_state__shock"
        )
        arms[arm] = {
            "median_endpoint_state_ratio": _median(
                [float(row["endpoint_state_ratio"]) for row in cases]
            ),
            "median_residual_rms_ratio": _median(
                [float(row["residual_rms_ratio"]) for row in cases]
            ),
            "median_cumulative_defect_ratio": _median(
                [float(row["cumulative_defect_ratio"]) for row in cases]
            ),
            "median_target_region_ratio": _median(
                [float(row[target_key]) for row in cases]
            ),
            "endpoint_nonworse_count": sum(
                float(row["endpoint_state_ratio"]) <= 1.0 for row in cases
            ),
            "maximum_control_ratio": max(
                float(row["maximum_control_ratio"]) for row in cases
            ),
            "minimum_temporal_median_state_ratio": min(state),
            "minimum_temporal_median_state_call": calls[int(np.argmin(state))],
            "last_call_with_median_state_improvement": max(
                (call for call, value in zip(calls, state, strict=True) if value < 1.0),
                default=None,
            ),
            "last_call_with_median_cumulative_improvement": max(
                (
                    call
                    for call, value in zip(calls, cumulative, strict=True)
                    if value < 1.0
                ),
                default=None,
            ),
            "temporal_coherence_ratio": by_structure[(arm, "temporal_coherence_ratio")][
                "median"
            ],
            "pod_first_three_energy_fraction_delta": by_structure[
                (arm, "pod_uncentered_first_three_delta")
            ]["median"],
            "local_only_same_input_energy_ratio": by_structure[
                (arm, "local_only_same_input_energy_ratio")
            ]["median"],
            "local_only_correction_base_cosine": by_structure[
                (arm, "local_only_correction_base_cosine")
            ]["median"],
            "median_active_node_fraction": _median(
                [float(row["median_active_node_fraction"]) for row in local]
            ),
            "median_cap_hit_fraction": _median(
                [float(row["cap_hit_fraction"]) for row in local]
            ),
        }
    return {
        "schema": SCHEMA,
        "source_result_schema": summary["schema"],
        "source_experiment_contract": summary["experiment_contract"],
        "source_status": summary["status"],
        "source_contract_checks_passed": summary["contract_checks_passed"],
        "family": summary["family"],
        "boundary_policy": summary["boundary_policy"],
        "population": summary["population"],
        "promotion": summary["promotion"],
        "aggregation": (
            "all ratios are formed within case and call before six-case summaries; "
            "no nodewise values are pooled across cases or meshes"
        ),
        "metric_contract": {
            "endpoint_state_ratio": (
                "state-scale physical-volume RMS at H30, combined / persistent"
            ),
            "residual_rms_ratio": (
                "residual-scale physical-volume RMS over 30 calls, combined / "
                "persistent"
            ),
            "instant_defect_ratio": (
                "residual-scale ||delta_n||, combined / persistent, each on its "
                "own recurrent inputs"
            ),
            "cumulative_defect_ratio": (
                "residual-scale ||sum_{j<=n} delta_j||, combined / persistent"
            ),
            "temporal_coherence": "||sum delta|| / sum ||delta||",
            "signed_growth_density": (
                "(2 e_n delta_n + delta_n^2) / D_r^2 componentwise"
            ),
            "visual_fields": (
                "all residual, defect, correction, and cumulative fields are divided "
                "by frozen component residual scales D_r"
            ),
        },
        "arms": arms,
        "interpretation": {
            "promotion": "no D080-B combined arm passes the registered gates",
            "shock": (
                "shock corrections are transiently beneficial but reverse late, "
                "slightly increasing defect coherence and low-rank concentration"
            ),
            "vortex": (
                "vortex correction is near-neutral in aggregate and heterogeneous "
                "across vortex placement"
            ),
            "mechanism": (
                "fixed causal local dissipation does not isolate the persistent "
                "large-scale recurrent defect; cap saturation and late loss of "
                "alignment dominate the bounded pilot"
            ),
        },
        "claim_boundary": summary["claim_boundary"],
    }


def _save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{stem}.pdf", output_dir / f"{stem}.png"]
    figure.savefig(paths[0], bbox_inches="tight")
    figure.savefig(paths[1], dpi=300, bbox_inches="tight")
    plt.close(figure)
    return paths


def _case_style(case_id: str) -> tuple[str, str]:
    strength = case_id.split("_")[1]
    placement = case_id.split("_")[2]
    return CASE_COLORS.get(strength, "#777777"), "-" if placement == "y00" else "--"


def plot_temporal_ratios(
    case_rows: Sequence[Mapping[str, Any]],
    aggregate_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> list[Path]:
    metrics = (
        ("state_error_ratio", "state-scale rollout error"),
        ("instant_defect_ratio", "instant residual-scale defect"),
        ("cumulative_defect_ratio", "cumulative residual-scale defect"),
    )
    figure, axes = plt.subplots(3, 3, figsize=(14.2, 10.5), sharex=True)
    for row_index, arm in enumerate(COMBINED_ARMS):
        selected = [row for row in case_rows if row["arm"] == arm]
        aggregate = [row for row in aggregate_rows if row["arm"] == arm]
        for column, (metric, title) in enumerate(metrics):
            axis = axes[row_index, column]
            for case_id in sorted({row["case_id"] for row in selected}):
                case = [row for row in selected if row["case_id"] == case_id]
                color, style = _case_style(case_id)
                axis.plot(
                    [row["physical_time"] for row in case],
                    [row[metric] for row in case],
                    color=color,
                    linestyle=style,
                    alpha=0.34,
                    linewidth=0.9,
                )
            axis.plot(
                [row["physical_time"] for row in aggregate],
                [row[f"median__{metric}"] for row in aggregate],
                color=COLORS[arm],
                linewidth=2.2,
                label="case-first median",
            )
            axis.fill_between(
                [row["physical_time"] for row in aggregate],
                [row[f"q25__{metric}"] for row in aggregate],
                [row[f"q75__{metric}"] for row in aggregate],
                color=COLORS[arm],
                alpha=0.14,
                linewidth=0.0,
            )
            axis.axhline(1.0, color="#222222", linestyle=":", linewidth=1.0)
            axis.grid(True, alpha=0.25)
            if row_index == 0:
                axis.set_title(title)
            if column == 0:
                axis.set_ylabel(f"{ARM_LABELS[arm]}\ncombined / persistent")
            if row_index == len(COMBINED_ARMS) - 1:
                axis.set_xlabel("physical time")
    figure.suptitle(
        "D080-B local corrections help transiently, then lose alignment",
        fontsize=14,
    )
    figure.text(
        0.5,
        0.005,
        "Thin lines: six cases (solid y00, dashed y08); band: interquartile range. "
        "Ratios are formed per case before aggregation.",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.025, 1, 0.97))
    return _save_figure(figure, output_dir, "temporal_correction_ratios")


def _ratio_norm(values: np.ndarray) -> TwoSlopeNorm:
    deviation = max(float(np.max(np.abs(values - 1.0))), 1e-6)
    return TwoSlopeNorm(vmin=1.0 - deviation, vcenter=1.0, vmax=1.0 + deviation)


def plot_case_matrix(
    case_rows: Sequence[Mapping[str, Any]], output_dir: Path
) -> list[Path]:
    metrics = (
        ("endpoint_state_ratio", "endpoint"),
        ("residual_rms_ratio", "residual RMS"),
        ("cumulative_defect_ratio", "cumulative"),
        ("target", "target region"),
        ("endpoint_smooth_highpass", "smooth HP"),
        ("maximum_control_ratio", "worst control"),
    )
    arrays = []
    for arm in COMBINED_ARMS:
        selected = [row for row in case_rows if row["arm"] == arm]
        target = (
            "endpoint_state__vortex"
            if arm == "combined_vortex_isotropic"
            else "endpoint_state__shock"
        )
        arrays.append(
            np.asarray(
                [
                    [
                        float(row[target] if key == "target" else row[key])
                        for key, _ in metrics
                    ]
                    for row in selected
                ]
            )
        )
    all_values = np.concatenate([array.ravel() for array in arrays])
    norm = _ratio_norm(all_values)
    figure, axes = plt.subplots(3, 1, figsize=(10.8, 10.5), constrained_layout=True)
    for axis, arm, values in zip(axes, COMBINED_ARMS, arrays, strict=True):
        selected = [row for row in case_rows if row["arm"] == arm]
        image = axis.imshow(values, cmap="RdBu_r", norm=norm, aspect="auto")
        axis.set_xticks(range(len(metrics)), [label for _, label in metrics])
        axis.set_yticks(range(len(selected)), [row["case_id"] for row in selected])
        axis.set_title(ARM_LABELS[arm])
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                axis.text(
                    j, i, f"{values[i, j]:.4f}", ha="center", va="center", fontsize=8
                )
    figure.colorbar(
        image,
        ax=list(axes),
        shrink=0.75,
        label="combined / persistent (exact common scale; 1 is neutral)",
    )
    figure.suptitle("D080-B per-case efficacy and no-harm matrix", fontsize=14)
    return _save_figure(figure, output_dir, "case_efficacy_matrix")


def plot_component_regions(
    rows: Sequence[Mapping[str, Any]], output_dir: Path
) -> list[Path]:
    labels = [label for _, label in CONTROL_METRICS]
    values = np.asarray(
        [
            [
                next(
                    float(row["median_ratio"])
                    for row in rows
                    if row["arm"] == arm and row["label"] == label
                )
                for label in labels
            ]
            for arm in COMBINED_ARMS
        ]
    )
    wins = np.asarray(
        [
            [
                next(
                    int(row["nonworse_count"])
                    for row in rows
                    if row["arm"] == arm and row["label"] == label
                )
                for label in labels
            ]
            for arm in COMBINED_ARMS
        ]
    )
    figure, axis = plt.subplots(figsize=(15.2, 4.4), constrained_layout=True)
    image = axis.imshow(values, cmap="RdBu_r", norm=_ratio_norm(values), aspect="auto")
    axis.set_xticks(range(len(labels)), labels, rotation=32, ha="right")
    axis.set_yticks(
        range(len(COMBINED_ARMS)), [ARM_LABELS[arm] for arm in COMBINED_ARMS]
    )
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            axis.text(
                j,
                i,
                f"{values[i, j]:.4f}\n{wins[i, j]}/6",
                ha="center",
                va="center",
                fontsize=7,
            )
    figure.colorbar(image, ax=axis, label="case-first median ratio")
    axis.set_title("Component, region, and global-integral effects (ratio; wins / 6)")
    return _save_figure(figure, output_dir, "component_region_ratios")


def plot_structure(
    results_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> list[Path]:
    index = {(row["arm"], row["metric"]): row for row in rows}
    lag_rows = [row for row in rows if row["metric"].startswith("lag_correlation__")]
    pods = _read_rows(results_dir / "pod_summaries.csv")
    sequences = _read_rows(results_dir / "sequence_summaries.csv")
    figure, axes = plt.subplots(2, 2, figsize=(11.8, 8.4), constrained_layout=True)
    labels = ["persistent", *[ARM_LABELS[arm] for arm in COMBINED_ARMS]]
    pod_values = []
    coherence_values = []
    for arm in ("persistent", *COMBINED_ARMS):
        pod_values.append(
            _median(
                [
                    _float(row, "first_three_energy_fraction")
                    for row in pods
                    if row["arm"] == arm and row["centering"] == "uncentered"
                ]
            )
        )
        coherence_values.append(
            _median(
                [
                    _float(row, "temporal_coherence")
                    for row in sequences
                    if row["arm"] == arm
                ]
            )
        )
    colors = [COLORS["persistent"], *[COLORS[arm] for arm in COMBINED_ARMS]]
    axes[0, 0].bar(range(4), pod_values, color=colors)
    axes[0, 0].set_xticks(range(4), labels, rotation=20, ha="right")
    axes[0, 0].set_ylabel("energy fraction")
    axes[0, 0].set_title("uncentered POD: first three modes")
    axes[0, 1].bar(range(4), coherence_values, color=colors)
    axes[0, 1].set_xticks(range(4), labels, rotation=20, ha="right")
    axes[0, 1].set_ylabel("kappa")
    axes[0, 1].set_title("temporal coherence")
    for arm in COMBINED_ARMS:
        selected = sorted(
            (row for row in lag_rows if row["arm"] == arm),
            key=lambda row: int(str(row["metric"]).split("__")[1]),
        )
        axes[1, 0].plot(
            [int(str(row["metric"]).split("__")[1]) for row in selected],
            [float(row["median_delta"]) for row in selected],
            label=ARM_LABELS[arm],
            color=COLORS[arm],
        )
    axes[1, 0].axhline(0.0, color="#222222", linestyle=":", linewidth=1.0)
    axes[1, 0].set_xlabel("lag")
    axes[1, 0].set_ylabel("median correlation change")
    axes[1, 0].set_title("defect lag correlation change")
    axes[1, 0].legend(fontsize=8)
    x = np.arange(len(COMBINED_ARMS))
    energy = [
        float(index[(arm, "local_only_same_input_energy_ratio")]["median"])
        for arm in COMBINED_ARMS
    ]
    cosine = [
        float(index[(arm, "local_only_correction_base_cosine")]["median"])
        for arm in COMBINED_ARMS
    ]
    axes[1, 1].bar(x - 0.18, energy, width=0.36, color="#0072B2", label="energy ratio")
    axes[1, 1].axhline(1.0, color="#0072B2", linestyle=":", linewidth=1.0)
    twin = axes[1, 1].twinx()
    twin.bar(x + 0.18, cosine, width=0.36, color="#D55E00", label="cosine")
    twin.axhline(0.0, color="#D55E00", linestyle=":", linewidth=1.0)
    axes[1, 1].set_xticks(
        x, [ARM_LABELS[arm] for arm in COMBINED_ARMS], rotation=20, ha="right"
    )
    axes[1, 1].set_ylabel("local-only same-input energy ratio", color="#0072B2")
    twin.set_ylabel("correction/base-defect cosine", color="#D55E00")
    axes[1, 1].set_title("intrinsic local direction (case-first medians)")
    for axis in axes.flat:
        axis.grid(True, axis="y", alpha=0.25)
    figure.suptitle(
        "D080-B residual structure: local filtering does not remove the persistent mode"
    )
    return _save_figure(figure, output_dir, "temporal_rank_structure")


def _structured_shape(resolution: str, count: int) -> tuple[int, int] | None:
    try:
        nx, ny = (int(value) for value in resolution.lower().split("x"))
    except (TypeError, ValueError):
        return None
    return (nx, ny) if nx * ny == count else None


def _structured_extent(nodes: np.ndarray, shape: tuple[int, int]) -> tuple[float, ...]:
    nx, ny = shape
    grid = nodes.reshape(ny, nx, 2)
    dx = float(np.median(np.diff(grid[0, :, 0]))) if nx > 1 else 1.0
    dy = float(np.median(np.diff(grid[:, 0, 1]))) if ny > 1 else 1.0
    return (
        float(grid[0, 0, 0] - 0.5 * dx),
        float(grid[0, -1, 0] + 0.5 * dx),
        float(grid[0, 0, 1] - 0.5 * dy),
        float(grid[-1, 0, 1] + 0.5 * dy),
    )


def _frame_indices(count: int, stride: int) -> tuple[int, ...]:
    values = list(range(0, count, stride))
    if values[-1] != count - 1:
        values.append(count - 1)
    return tuple(values)


def _prediction(payload: Mapping[str, np.ndarray], arm: str) -> np.ndarray:
    return (
        np.asarray(payload[f"base_increment__{arm}"], dtype=np.float64)
        + np.asarray(payload[f"persistent_correction__{arm}"], dtype=np.float64)
        + np.asarray(payload[f"local_correction__{arm}"], dtype=np.float64)
    )


def _growth(
    defect: np.ndarray, cumulative: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    previous = np.concatenate((np.zeros_like(cumulative[:1]), cumulative[:-1]), axis=0)
    return (2.0 * previous * defect + np.square(defect)) / np.square(
        scale[None, None, :]
    )


def _animation_fields(
    payload: Mapping[str, np.ndarray], arm: str
) -> dict[str, np.ndarray]:
    scale = np.asarray(payload["residual_scale"], dtype=np.float64)[None, None, :]
    truth = np.asarray(payload["true_increment"], dtype=np.float64)
    persistent_defect = np.asarray(payload["defect__persistent"], dtype=np.float64)
    combined_defect = np.asarray(payload[f"defect__{arm}"], dtype=np.float64)
    combined_cumulative = np.asarray(
        payload[f"cumulative_defect__{arm}"], dtype=np.float64
    )
    combined_growth = _growth(combined_defect, combined_cumulative, scale[0, 0])
    return {
        "true_increment": truth / scale,
        "persistent_prediction": _prediction(payload, "persistent") / scale,
        "combined_prediction": _prediction(payload, arm) / scale,
        "local_correction": np.asarray(
            payload[f"local_correction__{arm}"], dtype=np.float64
        )
        / scale,
        "persistent_defect": persistent_defect / scale,
        "combined_defect": combined_defect / scale,
        "combined_cumulative": combined_cumulative / scale,
        "combined_growth": combined_growth,
    }


def _population_limits(
    payload_paths: Sequence[Path],
    summary: Mapping[str, Any],
    case_ids: set[str] | None,
) -> tuple[dict[str, np.ndarray], list[Path]]:
    limits = {
        group: np.zeros(len(COMPONENTS), dtype=np.float64)
        for group in ("increment", "correction", "defect", "cumulative", "growth")
    }
    selected_paths = []
    observed = set()
    for path in payload_paths:
        payload = _load_payload(path, summary)
        case_id = str(_scalar(payload["case_id"]))
        observed.add(case_id)
        if case_ids is not None and case_id not in case_ids:
            continue
        selected_paths.append(path)
        for arm in COMBINED_ARMS:
            fields = _animation_fields(payload, arm)
            for name, _, group in ANIMATION_COLUMNS:
                limits[group] = np.maximum(
                    limits[group], np.max(np.abs(fields[name]), axis=(0, 1))
                )
    if case_ids is not None and case_ids - observed:
        raise ValueError(
            f"requested visualization cases are absent: {case_ids - observed}"
        )
    if not selected_paths:
        raise ValueError("no visual payloads selected")
    for values in limits.values():
        values[values == 0.0] = 1.0
    return limits, selected_paths


def _set_artist(
    artist: Any, structured: tuple[int, int] | None, values: np.ndarray
) -> None:
    if structured is None:
        artist.set_array(values)
    else:
        nx, ny = structured
        artist.set_data(values.reshape(ny, nx))


def animate_payload(
    payload_path: Path,
    output_dir: Path,
    summary: Mapping[str, Any],
    limits: Mapping[str, np.ndarray],
    *,
    fps: int,
    dpi: int,
    frame_stride: int,
) -> tuple[list[Path], list[dict[str, Any]]]:
    payload = _load_payload(payload_path, summary)
    nodes = np.asarray(payload["nodes"], dtype=np.float64)
    case_id = str(_scalar(payload["case_id"]))
    resolution = str(_scalar(payload["resolution"]))
    times = np.asarray(payload["physical_times"], dtype=np.float64)[1:]
    shape = _structured_shape(resolution, nodes.shape[0])
    extent = None if shape is None else _structured_extent(nodes, shape)
    frames = _frame_indices(len(times), frame_stride)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    saturation_rows = []
    for component, component_name in enumerate(COMPONENTS):
        arm_fields = {arm: _animation_fields(payload, arm) for arm in COMBINED_ARMS}
        figure, axes = plt.subplots(
            len(COMBINED_ARMS),
            len(ANIMATION_COLUMNS),
            figsize=(23.5, 8.8),
            constrained_layout=True,
        )
        artists: list[tuple[Any, str, str]] = []
        colorbar_artists: dict[str, Any] = {}
        for row_index, arm in enumerate(COMBINED_ARMS):
            fields = arm_fields[arm]
            for column, (name, label, group) in enumerate(ANIMATION_COLUMNS):
                axis = axes[row_index, column]
                values = fields[name][..., component]
                limit = float(limits[group][component])
                if shape is None:
                    artist = axis.scatter(
                        nodes[:, 0],
                        nodes[:, 1],
                        c=values[frames[0]],
                        s=2.0,
                        cmap="RdBu_r",
                        vmin=-limit,
                        vmax=limit,
                        linewidths=0.0,
                    )
                else:
                    nx, ny = shape
                    artist = axis.imshow(
                        values[frames[0]].reshape(ny, nx),
                        origin="lower",
                        extent=extent,
                        interpolation="nearest",
                        aspect="equal",
                        cmap="RdBu_r",
                        vmin=-limit,
                        vmax=limit,
                    )
                if row_index == 0:
                    axis.set_title(label, fontsize=8)
                if column == 0:
                    axis.set_ylabel(ARM_LABELS[arm], fontsize=8)
                axis.set_xticks([])
                axis.set_yticks([])
                artists.append((artist, arm, name))
                colorbar_artists.setdefault(group, artist)
                saturation_rows.append(
                    {
                        "case_id": case_id,
                        "resolution": resolution,
                        "component": component_name,
                        "arm": arm,
                        "field": name,
                        "scale_group": group,
                        "fixed_limit": limit,
                        "maximum_absolute_scaled_value": float(np.max(np.abs(values))),
                        "fraction_outside_limit": float(
                            np.mean(np.abs(values) > limit)
                        ),
                    }
                )
        group_columns = {
            "increment": (0, 1, 2),
            "correction": (3,),
            "defect": (4, 5),
            "cumulative": (6,),
            "growth": (7,),
        }
        for group, columns in group_columns.items():
            group_axes = [axes[row, column] for row in range(3) for column in columns]
            figure.colorbar(
                colorbar_artists[group],
                ax=group_axes,
                shrink=0.58,
                pad=0.01,
                label=group,
            )
        title = figure.suptitle("")
        figure.text(
            0.5,
            0.005,
            "All fields use frozen component D_r scaling. Exact selected-population "
            "maxima are fixed over the rollout; no clipping or per-frame normalization.",
            ha="center",
            fontsize=8,
        )

        def update(
            frame: int,
            *,
            bound_artists: list[tuple[Any, str, str]] = artists,
            bound_shape: tuple[int, int] | None = shape,
            bound_fields: Mapping[str, Mapping[str, np.ndarray]] = arm_fields,
            bound_component: int = component,
            bound_title: Any = title,
            bound_component_name: str = component_name,
        ) -> list[Any]:
            for artist, arm, name in bound_artists:
                _set_artist(
                    artist,
                    bound_shape,
                    bound_fields[arm][name][frame, :, bound_component],
                )
            bound_title.set_text(
                f"dynamic FV {case_id} {resolution} {bound_component_name} "
                f"t={times[frame]:.4g}"
            )
            return [artist for artist, _, _ in bound_artists] + [bound_title]

        movie = animation.FuncAnimation(
            figure, update, frames=frames, interval=1000 / fps, blit=False
        )
        stem = f"dynamic_fv_{case_id}_{resolution}_{component_name}_local_comparison"
        gif_path = output_dir / f"{stem}.gif"
        final_path = output_dir / f"{stem}_final.png"
        movie.save(gif_path, writer=animation.PillowWriter(fps=fps), dpi=dpi)
        update(frames[-1])
        figure.savefig(final_path, dpi=300, bbox_inches="tight")
        plt.close(figure)
        generated.extend((gif_path, final_path))
    return generated, saturation_rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    summary, payload_paths = _verify_results(args.results_dir)
    args.output_dir.mkdir(parents=True)
    generated: list[Path] = []

    case_rows = _case_metric_rows(args.results_dir)
    temporal_case_rows, temporal_aggregate_rows = _temporal_rows(
        _read_rows(args.results_dir / "evaluation_call_metrics.csv")
    )
    component_rows = _component_region_rows(case_rows)
    structure_rows = _structure_rows(args.results_dir)
    local_rows = _local_audit_rows(args.results_dir)
    analysis = _analysis_summary(
        summary,
        case_rows,
        temporal_aggregate_rows,
        component_rows,
        structure_rows,
        local_rows,
    )
    table_payloads = (
        ("case_metrics.csv", case_rows),
        ("temporal_case_ratios.csv", temporal_case_rows),
        ("temporal_aggregate.csv", temporal_aggregate_rows),
        ("component_region_summary.csv", component_rows),
        ("structure_summary.csv", structure_rows),
        ("local_audit_summary.csv", local_rows),
    )
    for name, rows in table_payloads:
        path = args.output_dir / name
        write_csv_with_paths(path, rows)
        generated.append(path)
    analysis_path = args.output_dir / "analysis_summary.json"
    write_json(analysis_path, analysis)
    generated.append(analysis_path)

    if args.command in {"plots", "all"}:
        figure_dir = args.output_dir / "figures"
        generated.extend(
            plot_temporal_ratios(
                temporal_case_rows, temporal_aggregate_rows, figure_dir
            )
        )
        generated.extend(plot_case_matrix(case_rows, figure_dir))
        generated.extend(plot_component_regions(component_rows, figure_dir))
        generated.extend(plot_structure(args.results_dir, structure_rows, figure_dir))

    limits = None
    selected_payload_paths: list[Path] = []
    saturation_rows: list[dict[str, Any]] = []
    if args.command in {"animations", "all"}:
        selected_cases = None if not args.case_ids else set(args.case_ids)
        limits, selected_payload_paths = _population_limits(
            payload_paths, summary, selected_cases
        )
        animation_dir = args.output_dir / "animations"
        for payload_path in selected_payload_paths:
            paths, rows = animate_payload(
                payload_path,
                animation_dir,
                summary,
                limits,
                fps=args.fps,
                dpi=args.dpi,
                frame_stride=args.frame_stride,
            )
            generated.extend(paths)
            saturation_rows.extend(rows)
        saturation_path = args.output_dir / "visual_scale_saturation.csv"
        write_csv_with_paths(saturation_path, saturation_rows)
        generated.append(saturation_path)

    output_hashes = {
        path.relative_to(args.output_dir).as_posix(): sha256_file(path)
        for path in sorted(generated)
    }
    manifest = {
        "schema": SCHEMA,
        "args": jsonable_args(args),
        "input_summary_sha256": sha256_file(args.results_dir / "summary.json"),
        "input_output_hashes_verified": True,
        "source_result_identity": list(RESULT_IDENTITY),
        "boundary_policy": summary["boundary_policy"],
        "aggregation": analysis["aggregation"],
        "visual_scale_contract": {
            "field_scaling": analysis["metric_contract"]["visual_fields"],
            "signed_growth": analysis["metric_contract"]["signed_growth_density"],
            "population_exact_max_by_group_and_component": (
                None
                if limits is None
                else {key: value.tolist() for key, value in limits.items()}
            ),
            "percentile_clipping": False,
            "per_frame_normalization": False,
            "saturation_row_count": len(saturation_rows),
        },
        "selected_visual_payloads": [
            path.relative_to(args.results_dir.resolve()).as_posix()
            for path in selected_payload_paths
        ],
        "output_hashes": output_hashes,
        "source_sha256": sha256_file(Path(__file__)),
        "git": git_state(),
    }
    write_json(args.output_dir / "manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    manifest = run(parse_args(argv))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
