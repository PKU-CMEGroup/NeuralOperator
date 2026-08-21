#!/usr/bin/env python3
"""Validate and compare paired W26-L2 bump full/no-gradient artifacts.

This script performs no checkpoint evaluation.  It consumes the two maintained
native-rollout evaluator directories, the two maintained fresh/propagated
decomposition directories, and the two training directories for each seed.  It
requires the frozen contracts and renders only registered open-validation
metrics with common axes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no.decompose_pcno_b1_frozen_rollout_error import (
    ADAPTER_PATH as DECOMPOSITION_ADAPTER_PATH,
)
from scripts.time_dependent_no.decompose_pcno_b1_frozen_rollout_error import (
    DECOMPOSER_PATH,
)
from scripts.time_dependent_no.decompose_pcno_b1_frozen_rollout_error import (
    SCHEMA as DECOMPOSITION_ADAPTER_SCHEMA,
)
from scripts.time_dependent_no.decompose_pcno_b1_frozen_rollout_error import (
    SUMMARY_KEY as DECOMPOSITION_ADAPTER_SUMMARY_KEY,
)
from scripts.time_dependent_no.train_pcno_bump_gradient_ablation import (
    B1_BOUNDARY_POLICY_SET_SHA256,
    B1_DATA_MANIFEST_SHA256,
    B1_FROZEN_SOURCE_SHA256,
    B1_SOURCE_SET_SHA256,
    B1_VALIDATION_KEYS,
    DIFFERENTIAL_BRANCH_MODES,
    REGISTERED_COMPLETED_PASSES,
    REGISTERED_OPTIMIZER_STEPS,
    REGISTERED_PRESENTATIONS,
    REGISTERED_SEEDS,
    checkpoint_differential_branch_mode,
    extension_source_hashes,
    load_checkpoint_for_ablation,
    mapping_sha256,
    sha256_file,
    write_json,
)
from scripts.time_dependent_no.train_pcno_bump_gradient_ablation import (
    SCHEMA as TRAINING_SCHEMA,
)

SCHEMA = "w26_l2_bump_gradient_ablation_analysis_v1"
ENDPOINT_CALLS = (1, 20, 40, 60, 79)
FIXED_VISUAL_CASES = ("16", "187")
COLORS = {"full": "#355c7d", "no_gradient": "#c06c84"}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"nonfinite metric value: {value!r}")
    return result


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty metric population")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _ratio(candidate: float, baseline: float) -> float:
    if (
        not math.isfinite(candidate)
        or not math.isfinite(baseline)
        or candidate < 0.0
        or baseline < 0.0
    ):
        raise ValueError("paired metric ratio requires finite nonnegative values")
    if baseline <= 1.0e-12:
        return 1.0 if candidate <= 1.0e-12 else candidate / 1.0e-12
    return candidate / baseline


def _paired_relative_change(candidate: float, baseline: float) -> float:
    return _ratio(candidate, baseline) - 1.0


def _checkpoint_path(training_dir: Path) -> Path:
    for name in ("best.pt", "last.pt"):
        path = training_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"training directory has no checkpoint: {training_dir}")


def verify_training_directory(
    training_dir: Path,
    *,
    mode: str,
    seed: int,
) -> dict[str, Any]:
    contract = _read_json(training_dir / "ablation_contract.json")
    if contract.get("schema") != TRAINING_SCHEMA:
        raise ValueError("training contract schema changed")
    checks = {
        "status_completed": contract.get("status") == "completed",
        "science_result_eligible": contract.get("science_result_eligible") is True,
        "mode_matches": contract.get("mode") == mode,
        "seed_matches": int(contract.get("seed", -1)) == seed,
        "data_manifest_matches": (
            contract.get("data_manifest_sha256") == B1_DATA_MANIFEST_SHA256
        ),
        "boundary_policy_set_matches": (
            contract.get("boundary_policy_set_sha256") == B1_BOUNDARY_POLICY_SET_SHA256
        ),
        "base_source_set_matches": (
            contract.get("base_source_set_sha256") == B1_SOURCE_SET_SHA256
        ),
        "base_source_files_match": (
            contract.get("base_source_sha256") == B1_FROZEN_SOURCE_SHA256
        ),
        "extension_source_files_match": (
            contract.get("extension_source_sha256") == extension_source_hashes()
        ),
        "matched_config_self_hashes": (
            contract.get("matched_config_sha256")
            == mapping_sha256(contract.get("matched_config", {}))
        ),
        "completed_passes_match": (
            int(contract.get("completed_passes", -1)) == REGISTERED_COMPLETED_PASSES
        ),
        "presentations_match": (
            int(contract.get("actual_presentations", -1)) == REGISTERED_PRESENTATIONS
        ),
        "optimizer_steps_match": (
            int(contract.get("actual_optimizer_steps", -1))
            == REGISTERED_OPTIMIZER_STEPS
        ),
        "stream_hash_count_matches": len(
            contract.get("presentation_stream_sha256_by_pass", [])
        )
        == REGISTERED_COMPLETED_PASSES,
        "holdout_remained_closed": (
            contract.get("holdout_or_sealed_population_accessed") is False
        ),
    }
    checkpoint_path = _checkpoint_path(training_dir)
    checkpoint = load_checkpoint_for_ablation(checkpoint_path, device="cpu")
    checkpoint_contract = checkpoint["differential_branch_contract"]
    checkpoint_streams = checkpoint_contract.get(
        "presentation_stream_sha256_by_pass", []
    )
    checks.update(
        {
            "checkpoint_mode_matches": (
                checkpoint_differential_branch_mode(checkpoint) == mode
            ),
            "checkpoint_base_source_matches": (
                checkpoint_contract.get("base_source_set_sha256")
                == B1_SOURCE_SET_SHA256
            ),
            "checkpoint_data_matches": (
                checkpoint_contract.get("data_manifest_sha256")
                == B1_DATA_MANIFEST_SHA256
            ),
            "checkpoint_boundary_matches": (
                checkpoint_contract.get("boundary_policy_set_sha256")
                == B1_BOUNDARY_POLICY_SET_SHA256
            ),
            "checkpoint_config_matches": (
                checkpoint_contract.get("matched_config_sha256")
                == contract.get("matched_config_sha256")
            ),
            "checkpoint_stream_is_registered_prefix": (
                checkpoint_streams
                == contract["presentation_stream_sha256_by_pass"][
                    : len(checkpoint_streams)
                ]
            ),
        }
    )
    if not all(checks.values()):
        raise ValueError(
            f"training contract checks failed for {mode}/s{seed}: {checks}"
        )
    return {
        "contract": contract,
        "checkpoint": {
            "path": checkpoint_path,
            "sha256": sha256_file(checkpoint_path),
        },
        "checks": checks,
    }


def verify_paired_training(
    full: Mapping[str, Any], no_gradient: Mapping[str, Any]
) -> dict[str, Any]:
    full_contract = full["contract"]
    no_gradient_contract = no_gradient["contract"]
    checks = {
        "matched_config": (
            full_contract["matched_config_sha256"]
            == no_gradient_contract["matched_config_sha256"]
        ),
        "presentation_streams": (
            full_contract["presentation_stream_sha256_by_pass"]
            == no_gradient_contract["presentation_stream_sha256_by_pass"]
        ),
        "base_source": (
            full_contract["base_source_sha256"]
            == no_gradient_contract["base_source_sha256"]
        ),
        "extension_source": (
            full_contract["extension_source_sha256"]
            == no_gradient_contract["extension_source_sha256"]
        ),
    }
    full_checkpoint = load_checkpoint_for_ablation(
        full["checkpoint"]["path"], device="cpu"
    )
    no_gradient_checkpoint = load_checkpoint_for_ablation(
        no_gradient["checkpoint"]["path"], device="cpu"
    )
    full_initial = full_checkpoint["differential_branch_contract"]
    no_gradient_initial = no_gradient_checkpoint["differential_branch_contract"]
    checks["initial_full_state"] = (
        full_initial["initial_full_state_sha256"]
        == no_gradient_initial["initial_full_state_sha256"]
    )
    checks["initial_nondifferential_state"] = (
        full_initial["initial_nondifferential_state_sha256"]
        == no_gradient_initial["initial_nondifferential_state_sha256"]
    )
    if not all(checks.values()):
        raise ValueError(f"paired training closure failed: {checks}")
    return checks


def verify_rollout_directory(
    rollout_dir: Path,
    *,
    checkpoint_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]]]:
    summary = _read_json(rollout_dir / "summary.json")
    checkpoint = summary.get("checkpoint", {})
    if checkpoint.get("sha256") != checkpoint_sha256:
        raise ValueError("rollout summary checkpoint hash does not match training")
    if checkpoint.get("data_manifest_digest") != B1_DATA_MANIFEST_SHA256:
        raise ValueError("rollout summary data manifest changed")
    evaluation = summary.get("evaluation", {})
    keys = tuple(str(value) for value in evaluation.get("trajectory_keys", ()))
    if set(keys) != set(B1_VALIDATION_KEYS) or len(keys) != len(B1_VALIDATION_KEYS):
        raise ValueError(
            "rollout evaluator does not contain the exact open validation set"
        )
    aggregate = summary.get("aggregates", {}).get("pcno_baseline", {})
    if int(aggregate.get("trajectories", -1)) != len(B1_VALIDATION_KEYS):
        raise ValueError("rollout aggregate trajectory count changed")
    call_rows = [
        row
        for row in _read_csv(rollout_dir / "call_metrics.csv")
        if row["variant"] == "pcno_baseline"
    ]
    endpoint_rows = [
        row
        for row in _read_csv(rollout_dir / "endpoint_metrics.csv")
        if row["variant"] == "pcno_baseline"
    ]
    required_pairs = {
        (trajectory, str(call))
        for trajectory in B1_VALIDATION_KEYS
        for call in ENDPOINT_CALLS
    }
    call_pairs = {(row["trajectory"], row["call_index"]) for row in call_rows}
    endpoint_pairs = {(row["trajectory"], row["call_index"]) for row in endpoint_rows}
    if not required_pairs.issubset(call_pairs):
        raise ValueError("call rows do not span every registered case and endpoint")
    if not required_pairs.issubset(endpoint_pairs):
        raise ValueError("endpoint rows do not span every registered case and endpoint")
    return summary, call_rows, endpoint_rows


def verify_decomposition_directory(
    decomposition_dir: Path,
    *,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    summary = _read_json(decomposition_dir / "summary.json")
    if summary.get("contract_complete") is not True:
        raise ValueError("fresh/propagated decomposition contract is incomplete")
    if summary.get("checkpoint", {}).get("sha256") != checkpoint_sha256:
        raise ValueError("decomposition checkpoint hash does not match training")
    compatibility = summary.get(DECOMPOSITION_ADAPTER_SUMMARY_KEY)
    if not isinstance(compatibility, Mapping):
        raise TypeError("decomposition lacks its frozen-B1 compatibility record")
    decomposition_sources = compatibility.get("decomposition_source_sha256")
    compatibility_checks = {
        "schema": compatibility.get("schema") == DECOMPOSITION_ADAPTER_SCHEMA,
        "contract_complete": compatibility.get("contract_complete") is True,
        "base_source_set": (
            compatibility.get("base_source_set_sha256") == B1_SOURCE_SET_SHA256
        ),
        "base_source_files": (
            compatibility.get("base_source_sha256") == B1_FROZEN_SOURCE_SHA256
        ),
        "extension_source_files": (
            compatibility.get("extension_source_sha256") == extension_source_hashes()
        ),
        "source_record_is_mapping": isinstance(decomposition_sources, Mapping),
        "source_record_matches_summary": (
            isinstance(decomposition_sources, Mapping)
            and summary.get("source_files") == decomposition_sources
        ),
        "source_record_self_hashes": (
            isinstance(decomposition_sources, Mapping)
            and compatibility.get("decomposition_source_set_sha256")
            == mapping_sha256(decomposition_sources)
        ),
        "adapter_recorded": (
            isinstance(decomposition_sources, Mapping)
            and DECOMPOSITION_ADAPTER_PATH in decomposition_sources
        ),
        "maintained_decomposer_recorded": (
            isinstance(decomposition_sources, Mapping)
            and DECOMPOSER_PATH in decomposition_sources
        ),
        "maintained_math_unchanged": (
            compatibility.get("maintained_decomposition_math_unchanged") is True
        ),
    }
    if not all(compatibility_checks.values()):
        raise ValueError(
            "frozen-B1 decomposition compatibility checks failed: "
            f"{compatibility_checks}"
        )
    checks = summary.get("contract_checks", {})
    if not checks or not all(value is True for value in checks.values()):
        raise ValueError("fresh/propagated decomposition checks failed")
    records = summary.get("trajectories")
    if not isinstance(records, list):
        raise TypeError("decomposition trajectories must be a record list")
    keys = [str(record.get("trajectory")) for record in records]
    if len(keys) != len(B1_VALIDATION_KEYS) or set(keys) != set(B1_VALIDATION_KEYS):
        raise ValueError("decomposition trajectory count changed")
    return summary


def _metric_by_call(rows: Sequence[Mapping[str, str]], column: str) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        call = int(row["call_index"])
        if call in ENDPOINT_CALLS:
            grouped[call].append(_float(row[column]))
    missing = sorted(set(ENDPOINT_CALLS) - set(grouped))
    if missing:
        raise ValueError(f"metric {column} lacks calls {missing}")
    return {call: _mean(grouped[call]) for call in ENDPOINT_CALLS}


def _endpoint_metric(
    rows: Sequence[Mapping[str, str]], column: str
) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        call = int(row["call_index"])
        if call in ENDPOINT_CALLS:
            grouped[call].append(_float(row[column]))
    missing = sorted(set(ENDPOINT_CALLS) - set(grouped))
    if missing:
        raise ValueError(f"metric {column} lacks calls {missing}")
    return {call: _mean(grouped[call]) for call in ENDPOINT_CALLS}


def _decomposition_mean(
    summary: Mapping[str, Any], call: int, region: str, metric: str
) -> float:
    return _float(summary["endpoints"][str(call)]["regions"][region][metric]["mean"])


def _arm_metrics(
    rollout_summary: Mapping[str, Any],
    call_rows: Sequence[Mapping[str, str]],
    endpoint_rows: Sequence[Mapping[str, str]],
    decomposition: Mapping[str, Any],
) -> dict[str, Any]:
    aggregate = rollout_summary["aggregates"]["pcno_baseline"]
    result = {
        "completion_rate": _float(aggregate["completion_rate"]),
        "mean_survival_fraction": _float(aggregate["mean_survival_fraction"]),
        "one_step_entry_relative_l2": _float(
            aggregate["one_step_entry_gate_mean_relative_l2_proxy"]
        ),
        "state_relative_l2_by_call": _metric_by_call(
            call_rows, "scaled_relative_l2_reconstructed_weight_proxy"
        ),
        "normal_pressure_rmse_by_call": _metric_by_call(call_rows, "normal_pres_rmse"),
        "boundary_pressure_rmse_by_call": _metric_by_call(
            call_rows, "boundary_pres_rmse"
        ),
        "smooth_highpass_by_call": _endpoint_metric(
            endpoint_rows, "smooth_highpass_energy_reconstructed_weight_proxy"
        ),
        "front_iou_by_call": _endpoint_metric(endpoint_rows, "front_iou"),
        "front_chamfer_by_call": _endpoint_metric(
            endpoint_rows, "front_symmetric_chamfer"
        ),
        "front_position_by_call": _endpoint_metric(
            endpoint_rows, "front_centroid_distance"
        ),
        "shock_thickness_log_error_by_call": _endpoint_metric(
            endpoint_rows, "shock_thickness_log_error"
        ),
        "shock_strength_log_error_by_call": _endpoint_metric(
            endpoint_rows, "shock_strength_log_error"
        ),
        "decomposition_h79": {},
    }
    for region in (
        "normal_nodes_full",
        "boundary_nodes_full",
        "front_support_full",
        "smooth_highpass",
    ):
        result["decomposition_h79"][region] = {
            metric: _decomposition_mean(decomposition, 79, region, metric)
            for metric in (
                "total_norm",
                "propagated_norm",
                "fresh_defect_norm",
                "cross_energy_fraction_of_total",
            )
        }
    return result


def _seed_decision(
    full: Mapping[str, Any], no_gradient: Mapping[str, Any]
) -> dict[str, Any]:
    ratios = {
        "h79_state": _ratio(
            no_gradient["state_relative_l2_by_call"][79],
            full["state_relative_l2_by_call"][79],
        ),
        "h79_smooth_highpass": _ratio(
            no_gradient["smooth_highpass_by_call"][79],
            full["smooth_highpass_by_call"][79],
        ),
        "h79_front_chamfer": _ratio(
            no_gradient["front_chamfer_by_call"][79],
            full["front_chamfer_by_call"][79],
        ),
        "h79_front_position": _ratio(
            no_gradient["front_position_by_call"][79],
            full["front_position_by_call"][79],
        ),
        "h79_thickness_log_error": _ratio(
            no_gradient["shock_thickness_log_error_by_call"][79],
            full["shock_thickness_log_error_by_call"][79],
        ),
        "h79_strength_log_error": _ratio(
            no_gradient["shock_strength_log_error_by_call"][79],
            full["shock_strength_log_error_by_call"][79],
        ),
        "fresh_normal": _ratio(
            no_gradient["decomposition_h79"]["normal_nodes_full"]["fresh_defect_norm"],
            full["decomposition_h79"]["normal_nodes_full"]["fresh_defect_norm"],
        ),
        "propagated_normal": _ratio(
            no_gradient["decomposition_h79"]["normal_nodes_full"]["propagated_norm"],
            full["decomposition_h79"]["normal_nodes_full"]["propagated_norm"],
        ),
    }
    controls = {
        "completion_not_lower": (
            no_gradient["completion_rate"] >= full["completion_rate"]
        ),
        "survival_not_lower": (
            no_gradient["mean_survival_fraction"]
            >= full["mean_survival_fraction"] - 1.0e-12
        ),
        "no_harm_h20_state": _ratio(
            no_gradient["state_relative_l2_by_call"][20],
            full["state_relative_l2_by_call"][20],
        )
        <= 1.05,
        "no_harm_boundary_pressure": _ratio(
            no_gradient["boundary_pressure_rmse_by_call"][79],
            full["boundary_pressure_rmse_by_call"][79],
        )
        <= 1.05,
        "no_harm_front_iou": (
            full["front_iou_by_call"][79] > 0.0
            and no_gradient["front_iou_by_call"][79] > 0.0
            and no_gradient["front_iou_by_call"][79]
            >= 0.95 * full["front_iou_by_call"][79]
        ),
        "no_harm_front_chamfer": ratios["h79_front_chamfer"] <= 1.05,
        "no_harm_thickness": ratios["h79_thickness_log_error"] <= 1.05,
        "no_harm_strength": ratios["h79_strength_log_error"] <= 1.05,
        "no_harm_smooth_highpass": ratios["h79_smooth_highpass"] <= 1.05,
    }
    primary = ratios["h79_state"] <= 0.95
    return {
        "ratios_no_gradient_over_full": ratios,
        "primary_h79_state_improves_at_least_5pct": primary,
        "controls": controls,
        "all_controls_pass": all(controls.values()),
        "seed_pass": primary and all(controls.values()),
    }


def _configure_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_figure(fig: Any, output_stem: Path) -> list[Path]:
    outputs = []
    for suffix in (".png", ".pdf"):
        path = output_stem.with_suffix(suffix)
        fig.savefig(path, dpi=220 if suffix == ".png" else None, bbox_inches="tight")
        outputs.append(path)
    return outputs


def render_summary_figures(
    seed_payloads: Sequence[Mapping[str, Any]], output_dir: Path
) -> list[Path]:
    plt = _configure_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    calls = np.asarray(ENDPOINT_CALLS)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True)
    panels = (
        ("state_relative_l2_by_call", "state proxy relative L2", False),
        ("smooth_highpass_by_call", "smooth-region graph high-pass", True),
        ("front_chamfer_by_call", "front symmetric Chamfer", False),
        ("shock_thickness_log_error_by_call", "shock thickness log error", False),
    )
    for axis, (metric, ylabel, log_scale) in zip(axes.flat, panels, strict=True):
        for mode in DIFFERENTIAL_BRANCH_MODES:
            curves = np.asarray(
                [
                    [payload[mode][metric][int(call)] for call in calls]
                    for payload in seed_payloads
                ],
                dtype=np.float64,
            )
            median = np.median(curves, axis=0)
            low = np.min(curves, axis=0)
            high = np.max(curves, axis=0)
            axis.plot(calls, median, marker="o", color=COLORS[mode], label=mode)
            axis.fill_between(calls, low, high, color=COLORS[mode], alpha=0.18)
        if log_scale:
            axis.set_yscale("log")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel("autoregressive call")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("W26-L2 bump gradient ablation: paired open validation")
    outputs.extend(_save_figure(fig, output_dir / "paired_rollout_metrics"))
    plt.close(fig)

    regions = (
        "normal_nodes_full",
        "boundary_nodes_full",
        "front_support_full",
        "smooth_highpass",
    )
    metrics = ("fresh_defect_norm", "propagated_norm")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=False)
    width = 0.36
    x = np.arange(len(regions))
    for axis, metric in zip(axes, metrics, strict=True):
        for offset, mode in ((-width / 2, "full"), (width / 2, "no_gradient")):
            values = []
            low = []
            high = []
            for region in regions:
                observed = np.asarray(
                    [
                        payload[mode]["decomposition_h79"][region][metric]
                        for payload in seed_payloads
                    ],
                    dtype=np.float64,
                )
                values.append(float(np.median(observed)))
                low.append(float(np.median(observed) - np.min(observed)))
                high.append(float(np.max(observed) - np.median(observed)))
            axis.bar(
                x + offset,
                values,
                width,
                yerr=np.asarray([low, high]),
                color=COLORS[mode],
                label=mode,
                capsize=3,
            )
        axis.set_title(metric.replace("_", " "))
        axis.set_xticks(
            x, [value.replace("_full", "") for value in regions], rotation=20
        )
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("weighted scaled norm at call 79")
    axes[0].legend(frameon=False)
    fig.suptitle("Fresh versus propagated error, exact additive decomposition")
    outputs.extend(_save_figure(fig, output_dir / "fresh_propagated_h79"))
    plt.close(fig)
    return outputs


def _pressure(conservative: np.ndarray, *, gamma: float = 1.4) -> np.ndarray:
    value = np.asarray(conservative, dtype=np.float64)
    rho = value[..., 0]
    if np.any(rho <= 0.0) or not np.isfinite(value).all():
        raise ValueError("pressure visualization requires finite positive density")
    kinetic = 0.5 * (np.square(value[..., 1]) + np.square(value[..., 2])) / rho
    return (gamma - 1.0) * (value[..., 3] - kinetic)


def _load_trajectory_artifact(
    path: Path,
    *,
    trajectory: str,
    checkpoint_sha256: str,
) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    required = {
        "trajectory_key",
        "initial_conservative",
        "reference_targets_conservative",
        "pcno_baseline_predictions_conservative",
        "positions",
        "checkpoint_sha256",
        "test_manifest_digest",
        "training_manifest_digest",
        "baseline_valid_length",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"trajectory artifact is missing arrays: {missing}")
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    if str(arrays["trajectory_key"].item()) != trajectory:
        raise ValueError("trajectory artifact key changed")
    if str(arrays["checkpoint_sha256"].item()) != checkpoint_sha256:
        raise ValueError("trajectory artifact checkpoint hash changed")
    for name in ("test_manifest_digest", "training_manifest_digest"):
        if str(arrays[name].item()) != B1_DATA_MANIFEST_SHA256:
            raise ValueError(f"trajectory artifact {name} changed")
    return arrays


def _residual_fields(
    arrays: Mapping[str, np.ndarray], calls: Sequence[int]
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]:
    initial = np.asarray(arrays["initial_conservative"], dtype=np.float64)
    targets = np.asarray(arrays["reference_targets_conservative"], dtype=np.float64)
    predictions = np.asarray(
        arrays["pcno_baseline_predictions_conservative"], dtype=np.float64
    )
    valid_length = int(arrays["baseline_valid_length"].item())
    true: dict[int, np.ndarray] = {}
    predicted: dict[int, np.ndarray] = {}
    error: dict[int, np.ndarray] = {}
    initial_pressure = _pressure(initial)
    target_pressure = _pressure(targets)
    prediction_pressure = _pressure(predictions[:valid_length])
    for call in calls:
        true_previous = initial_pressure if call == 1 else target_pressure[call - 2]
        true[call] = target_pressure[call - 1] - true_previous
        if call <= valid_length:
            predicted_previous = (
                initial_pressure if call == 1 else prediction_pressure[call - 2]
            )
            predicted[call] = prediction_pressure[call - 1] - predicted_previous
            error[call] = predicted[call] - true[call]
    return true, predicted, error


def render_fixed_case_residual_figures(
    input_root: Path,
    output_dir: Path,
    *,
    seed: int,
    provenance: Mapping[str, Any],
) -> list[Path]:
    """Render fixed-case true/predicted residual and residual-error maps."""

    plt = _configure_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for trajectory in FIXED_VISUAL_CASES:
        artifacts: dict[str, dict[str, np.ndarray]] = {}
        for mode in DIFFERENTIAL_BRANCH_MODES:
            path = (
                _run_paths(input_root, seed, mode)["rollout"]
                / "trajectories"
                / f"trajectory_{trajectory}.npz"
            )
            artifacts[mode] = _load_trajectory_artifact(
                path,
                trajectory=trajectory,
                checkpoint_sha256=provenance[str(seed)][mode]["checkpoint_sha256"],
            )
        for name in (
            "positions",
            "initial_conservative",
            "reference_targets_conservative",
        ):
            if not np.array_equal(
                artifacts["full"][name], artifacts["no_gradient"][name]
            ):
                raise ValueError(f"paired trajectory artifacts disagree on {name}")

        positions = np.asarray(artifacts["full"]["positions"], dtype=np.float64)
        true, full_predicted, full_error = _residual_fields(
            artifacts["full"], ENDPOINT_CALLS
        )
        no_grad_true, no_grad_predicted, no_grad_error = _residual_fields(
            artifacts["no_gradient"], ENDPOINT_CALLS
        )
        for call in ENDPOINT_CALLS:
            if not np.array_equal(true[call], no_grad_true[call]):
                raise ValueError("paired residual artifacts disagree on exact truth")
        physical_values = [true[call] for call in ENDPOINT_CALLS]
        physical_values.extend(full_predicted.values())
        physical_values.extend(no_grad_predicted.values())
        error_values = [*full_error.values(), *no_grad_error.values()]
        physical_limit = float(
            np.quantile(np.abs(np.concatenate(physical_values)), 0.995)
        )
        error_limit = (
            float(np.quantile(np.abs(np.concatenate(error_values)), 0.995))
            if error_values
            else physical_limit
        )
        physical_limit = max(physical_limit, np.finfo(float).tiny)
        error_limit = max(error_limit, np.finfo(float).tiny)

        row_fields = (
            ("true residual", true, physical_limit),
            ("full predicted", full_predicted, physical_limit),
            ("full error", full_error, error_limit),
            ("no-gradient predicted", no_grad_predicted, physical_limit),
            ("no-gradient error", no_grad_error, error_limit),
        )
        fig, axes = plt.subplots(
            len(row_fields),
            len(ENDPOINT_CALLS),
            figsize=(15.5, 10.5),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        colorbars: dict[str, Any] = {}
        for row_index, (label, fields, limit) in enumerate(row_fields):
            for column_index, call in enumerate(ENDPOINT_CALLS):
                axis = axes[row_index, column_index]
                if call in fields:
                    artist = axis.scatter(
                        positions[:, 0],
                        positions[:, 1],
                        c=fields[call],
                        s=0.45,
                        cmap="coolwarm",
                        vmin=-limit,
                        vmax=limit,
                        linewidths=0.0,
                        rasterized=True,
                    )
                    colorbars["error" if "error" in label else "physical"] = artist
                else:
                    axis.text(
                        0.5,
                        0.5,
                        "no admissible retained prediction",
                        ha="center",
                        va="center",
                        transform=axis.transAxes,
                        fontsize=8,
                    )
                if row_index == 0:
                    axis.set_title(f"call {call}")
                if column_index == 0:
                    axis.set_ylabel(label)
                axis.set_aspect("equal")
                axis.set_xticks([])
                axis.set_yticks([])
        if "physical" in colorbars:
            fig.colorbar(
                colorbars["physical"],
                ax=axes[[0, 1, 3], :].ravel().tolist(),
                shrink=0.72,
                label="pressure increment",
            )
        if "error" in colorbars:
            fig.colorbar(
                colorbars["error"],
                ax=axes[[2, 4], :].ravel().tolist(),
                shrink=0.72,
                label="pressure-increment error",
            )
        fig.suptitle(
            f"Trajectory {trajectory}, seed {seed}: exact vs autoregressive pressure residual",
            fontsize=13,
        )
        outputs.extend(
            _save_figure(
                fig,
                output_dir / f"residual_maps_seed_{seed}_trajectory_{trajectory}",
            )
        )
        plt.close(fig)
    return outputs


def _run_paths(root: Path, seed: int, mode: str) -> dict[str, Path]:
    base = root / f"seed_{seed}" / mode
    return {
        "training": base / "training",
        "rollout": base / "rollout_open_validation",
        "decomposition": base / "decomposition_open_validation",
    }


def run_analysis(
    input_root: Path,
    output_dir: Path,
    *,
    seeds: Sequence[int] = REGISTERED_SEEDS,
    render: bool = True,
) -> dict[str, Any]:
    requested_seeds = tuple(int(seed) for seed in seeds)
    if not requested_seeds:
        raise ValueError("at least one registered seed is required")
    if len(set(requested_seeds)) != len(requested_seeds):
        raise ValueError("analysis seeds must be unique")
    if not set(requested_seeds).issubset(REGISTERED_SEEDS):
        raise ValueError(f"analysis seeds must lie in {REGISTERED_SEEDS}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_payloads = []
    provenance: dict[str, Any] = {}
    for seed in requested_seeds:
        arms: dict[str, Any] = {}
        training: dict[str, Any] = {}
        provenance[str(seed)] = {}
        for mode in DIFFERENTIAL_BRANCH_MODES:
            paths = _run_paths(input_root, int(seed), mode)
            trained = verify_training_directory(
                paths["training"], mode=mode, seed=int(seed)
            )
            rollout_summary, call_rows, endpoint_rows = verify_rollout_directory(
                paths["rollout"],
                checkpoint_sha256=trained["checkpoint"]["sha256"],
            )
            decomposition = verify_decomposition_directory(
                paths["decomposition"],
                checkpoint_sha256=trained["checkpoint"]["sha256"],
            )
            training[mode] = trained
            arms[mode] = _arm_metrics(
                rollout_summary, call_rows, endpoint_rows, decomposition
            )
            provenance[str(seed)][mode] = {
                "training_contract_sha256": sha256_file(
                    paths["training"] / "ablation_contract.json"
                ),
                "checkpoint_sha256": trained["checkpoint"]["sha256"],
                "rollout_summary_sha256": sha256_file(
                    paths["rollout"] / "summary.json"
                ),
                "decomposition_summary_sha256": sha256_file(
                    paths["decomposition"] / "summary.json"
                ),
            }
        pairing = verify_paired_training(training["full"], training["no_gradient"])
        seed_payloads.append(
            {
                "seed": int(seed),
                **arms,
                "pairing_checks": pairing,
                "decision": _seed_decision(arms["full"], arms["no_gradient"]),
            }
        )

    state_changes = [
        _paired_relative_change(
            payload["no_gradient"]["state_relative_l2_by_call"][79],
            payload["full"]["state_relative_l2_by_call"][79],
        )
        for payload in seed_payloads
    ]
    seed_passes = [bool(payload["decision"]["seed_pass"]) for payload in seed_payloads]
    primary_seed_wins = [
        bool(payload["decision"]["primary_h79_state_improves_at_least_5pct"])
        for payload in seed_payloads
    ]
    result = {
        "schema": SCHEMA,
        "status": "completed",
        "science_result": set(requested_seeds) == set(REGISTERED_SEEDS),
        "population": "open_validation_30_only",
        "seeds": list(requested_seeds),
        "fixed_visual_cases": list(FIXED_VISUAL_CASES),
        "seed_results": seed_payloads,
        "aggregate_decision": {
            "primary_seed_win_count": sum(primary_seed_wins),
            "joint_seed_pass_count": sum(seed_passes),
            "median_h79_state_relative_change": float(np.median(state_changes)),
            "primary_pass": sum(primary_seed_wins) >= 2
            and float(np.median(state_changes)) <= -0.05,
            "joint_pass": sum(seed_passes) >= 2,
            "registered_open_validation_architecture_result": (sum(seed_passes) >= 2),
            "interpretation_if_primary_only": (
                "optimization or recurrence benefit with an unresolved structure/safety tradeoff"
            ),
            "interpretation_if_joint": (
                "trained no-gradient architecture improves the matched bump rollout on the "
                "registered open population; sealed generalization remains untested"
            ),
        },
        "claim_boundary": {
            "bump_weights_are_proxy_only": True,
            "physical_conservation_claim": False,
            "d041_is_historical_not_causal_control": True,
            "test_or_holdout_opened": False,
            "general_gradient_harm_claim": False,
            "strict_gibbs_claim": False,
        },
        "provenance": provenance,
    }
    write_json(output_dir / "analysis.json", result)
    figure_paths = render_summary_figures(seed_payloads, output_dir) if render else []
    if render and int(REGISTERED_SEEDS[0]) in requested_seeds:
        figure_paths.extend(
            render_fixed_case_residual_figures(
                input_root,
                output_dir,
                seed=int(REGISTERED_SEEDS[0]),
                provenance=provenance,
            )
        )
    outputs = [output_dir / "analysis.json", *figure_paths]
    manifest = {
        "schema": SCHEMA,
        "analysis_sha256": sha256_file(output_dir / "analysis.json"),
        "output_sha256": {path.name: sha256_file(path) for path in outputs},
    }
    manifest["manifest_payload_sha256"] = mapping_sha256(manifest)
    write_json(output_dir / "manifest.json", manifest)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(REGISTERED_SEEDS))
    parser.add_argument("--no-render", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_analysis(
        args.input_root,
        args.output_dir,
        seeds=args.seeds,
        render=not args.no_render,
    )
    print(json.dumps(result["aggregate_decision"], sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
