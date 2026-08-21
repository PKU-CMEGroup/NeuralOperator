"""Probe PlanarDet recurrence with one exact chemistry or density pulse.

The frozen baseline proposal at a registered pulse call is preserved and scored.
Only for the next recurrent input, one selected field group is replaced by the
released next-frame truth.  Every later state is again a raw model proposal.
This is an oracle hybrid-state diagnostic on one open validation trajectory,
not a deployable correction method.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.time_dependent_no import (
    evaluate_realm_planardet_group_feedback as feedback,
)
from scripts.time_dependent_no.evaluate_realm_planardet_pcno import (
    DEVICE_NAME,
    DTYPE,
    _configure_determinism,
    _is_within,
    _load_json,
    _prepare_output_directory,
    _runtime_manifest,
    _structure_summary,
    _validate_best_checkpoint,
    _validate_closed_training_tree,
    _validate_signed_payload,
    _view_summary,
    _write_json,
    _write_npy,
)
from scripts.time_dependent_no.evaluate_realm_planardet_pmax_projection import (
    BASELINE_BEST_STEP,
    BASELINE_CHECKPOINT_SHA256,
    BASELINE_EVALUATION_FILES,
    BASELINE_FINAL_MANIFEST_SHA256,
    BASELINE_RESULT_PAYLOAD_SHA256,
    BASELINE_RESULT_SHA256,
    BASELINE_RUN_ID,
    BASELINE_SOURCE_DIGEST,
    EXPECTED_ARRAY_SHAPE,
    _validate_baseline_evaluation,
    _validate_startup_environment,
)
from utility.time_dependent_no.realm_benchmark import (
    canonical_json_sha256,
    predict_one_call,
)
from utility.time_dependent_no.realm_pcno import (
    RealmPCNOConfig,
    RealmRegularGridPCNO,
    build_realm_regular_grid_geometry,
)
from utility.time_dependent_no.realm_planardet import (
    CANONICAL_COORDINATE_ORDER,
    DOMAIN_LENGTHS_XY,
    PLANARDET_OPEN_MANIFEST_SHA256,
    PLANARDET_TRAIN_GROUPS,
    PLANARDET_VAL_GROUPS,
    load_planardet_metadata,
    sha256_file,
    validate_local_open_tree,
    validate_planardet_open_manifest,
)
from utility.time_dependent_no.realm_planardet_artifacts import (
    EXECUTABLE_ENTRYPOINTS,
    build_source_manifest,
)
from utility.time_dependent_no.realm_planardet_runtime import (
    CHANNELS,
    EXPECTED_PARAMETER_COUNTS,
    FC_DIM,
    MODE_COUNTS_XY,
    VALIDATION_HORIZON,
    load_normalized_trajectories,
    load_normalizer_bundle,
    planardet_boundary_mask,
    validate_frozen_controls,
    validation_control_summary,
)

EXPERIMENT_ID = "w26_l4_pd0_group_pulse_g1_d092r1_step950_20260817a"
PREREGISTRATION_SCHEMA = "w26_l4_planardet_group_pulse_preregistration_v1"
DIAGNOSTIC_SOURCE_SCHEMA = "w26_l4_planardet_group_pulse_source_v1"
RUNTIME_SCHEMA = "w26_l4_planardet_group_pulse_cuda_runtime_v1"
RESULT_SCHEMA = "w26_l4_planardet_group_pulse_evaluation_v1"
FINAL_MANIFEST_SCHEMA = "w26_l4_planardet_group_pulse_final_hash_manifest_v1"
SUMMARY_SCHEMA = "w26_l4_planardet_group_pulse_summary_v1"

G0B_EXPERIMENT_ID = "w26_l4_pd0_group_feedback_g0b_d092r1_step950_20260817a"
G0B_RESULT_SHA256 = "a05e44a23529e7883802315c47b7fa41c3476c37a6e57106e187e8516c0260a3"
G0B_RESULT_PAYLOAD_SHA256 = (
    "a4e4dbf841321acde666ca882a7dd47b49d6de645de7d45904affb6290dd1259"
)
G0B_FINAL_MANIFEST_SHA256 = (
    "385b5f7fea016f3704ccb4e96d2f0dee8558d5681585c087ee4032f25a592ce6"
)
G0B_ERROR_ACCUMULATION_CSV_SHA256 = (
    "19bdc1d8feaeacc512c801d0bbd084a6c99ff333113bd7a544bb8bf7c11c6237"
)
G0B_PRIMARY_RATIOS = {
    "chem": 0.4840323189437652,
    "rho": 0.3443464568435626,
}

PULSE_GROUPS = ("chem", "rho")
PULSE_CALLS = (4, 12, 32)
ARM_SPECS = tuple(
    (group, pulse_call) for pulse_call in PULSE_CALLS for group in PULSE_GROUPS
)
COMMON_DOWNSTREAM_GROUPS = ("T", "u")
EXPECTED_SOURCE_SUM_FACTOR = VALIDATION_HORIZON

PULSE_SELECTION = {
    4: {
        "rationale": (
            "first call where the continuous chemistry-feedback primary ratio "
            "enters the frozen materially-helpful band"
        ),
        "baseline_all_group_npe": 0.27066630125045776,
        "continuous_chem_primary_ratio": 0.8537846922949964,
        "continuous_rho_primary_ratio": 0.7803301276401942,
    },
    12: {
        "rationale": "first call where baseline all-group NPE is at least 1.0",
        "baseline_all_group_npe": 1.0902049541473389,
        "continuous_chem_primary_ratio": 0.42825282496268396,
        "continuous_rho_primary_ratio": 0.43286246539240913,
    },
    32: {
        "rationale": (
            "fixed late high-drift pulse near two-thirds horizon with at least "
            "17 subsequent calls"
        ),
        "baseline_all_group_npe": 2.249277114868164,
        "continuous_chem_primary_ratio": 0.48705256445051875,
        "continuous_rho_primary_ratio": 0.31540046324048243,
    },
}


def arm_id(group: str, pulse_call: int) -> str:
    if group not in PULSE_GROUPS or pulse_call not in PULSE_CALLS:
        raise ValueError("unknown pulse arm")
    return f"{group}_call{pulse_call}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preregister = subparsers.add_parser(
        "preregister", description="Write the immutable pulse preregistration."
    )
    preregister.add_argument("--output", type=Path, required=True)

    evaluate = subparsers.add_parser(
        "evaluate", description="Execute the frozen CUDA pulse diagnostic."
    )
    evaluate.add_argument("--training-dir", type=Path, required=True)
    evaluate.add_argument("--baseline-evaluation-dir", type=Path, required=True)
    evaluate.add_argument("--group-feedback-evaluation-dir", type=Path, required=True)
    evaluate.add_argument("--preregistration", type=Path, required=True)
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--data-root", type=Path, required=True)
    evaluate.add_argument("--normalizer-arrays", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    return parser


def build_diagnostic_source_manifest() -> dict[str, Any]:
    relatives = (
        "scripts/time_dependent_no/evaluate_realm_planardet_group_pulse.py",
        "scripts/time_dependent_no/evaluate_realm_planardet_group_feedback.py",
        "scripts/time_dependent_no/evaluate_realm_planardet_pmax_projection.py",
    )
    files = []
    for relative in relatives:
        path = REPO_ROOT.joinpath(*relative.split("/"))
        files.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload: dict[str, Any] = {
        "schema": DIAGNOSTIC_SOURCE_SCHEMA,
        "baseline_training_source_manifest_digest": BASELINE_SOURCE_DIGEST,
        "files": files,
    }
    payload["canonical_payload_sha256"] = canonical_json_sha256(payload)
    return payload


def build_preregistration() -> dict[str, Any]:
    source = build_diagnostic_source_manifest()
    payload: dict[str, Any] = {
        "schema": PREREGISTRATION_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "frozen_before_gpu_execution",
        "baseline": {
            "run_id": BASELINE_RUN_ID,
            "best_step": BASELINE_BEST_STEP,
            "checkpoint_sha256": BASELINE_CHECKPOINT_SHA256,
            "training_source_manifest_digest": BASELINE_SOURCE_DIGEST,
            "evaluation_result_sha256": BASELINE_RESULT_SHA256,
            "evaluation_result_payload_sha256": BASELINE_RESULT_PAYLOAD_SHA256,
            "evaluation_final_manifest_sha256": BASELINE_FINAL_MANIFEST_SHA256,
            "free_array_sha256": BASELINE_EVALUATION_FILES[
                "free_prediction_normalized.npy"
            ],
        },
        "selection_evidence": {
            "experiment_id": G0B_EXPERIMENT_ID,
            "result_sha256": G0B_RESULT_SHA256,
            "result_payload_sha256": G0B_RESULT_PAYLOAD_SHA256,
            "final_manifest_sha256": G0B_FINAL_MANIFEST_SHA256,
            "error_accumulation_csv_sha256": G0B_ERROR_ACCUMULATION_CSV_SHA256,
            "continuous_primary_ratios": G0B_PRIMARY_RATIOS,
            "pulse_calls": {str(call): PULSE_SELECTION[call] for call in PULSE_CALLS},
        },
        "diagnostic_source_manifest_digest": source["canonical_payload_sha256"],
        "population": "one released open-validation PlanarDet trajectory",
        "arms": {
            arm_id(group, pulse_call): {
                "group": group,
                "channels": list(feedback.GROUP_CHANNELS[group]),
                "pulse_call": pulse_call,
                "scored_pulse_state": "verified unintervened raw baseline proposal",
                "accepted_pulse_state": (
                    "replace only the selected group with released next-frame "
                    "normalized truth for the following recurrent input"
                ),
                "later_recurrence": "raw model proposals with no further truth input",
            }
            for group, pulse_call in ARM_SPECS
        },
        "primary_causal_readout": (
            "for each arm, mean grouped normalized prediction error on common "
            "untouched T+u over calls after the pulse, divided by the same calls "
            "and groups in the unintervened baseline"
        ),
        "directional_readout": {
            "chem_to_rho": "post-pulse rho error ratio for a chemistry pulse",
            "rho_to_chem": "post-pulse chemistry error ratio for a density pulse",
            "routing": (
                "both <=0.90 supports bidirectional model-state error coupling; "
                "only one <=0.90 supports that direction at the registered pulse; "
                "neither rejects a material single-pulse partner reduction"
            ),
        },
        "temporal_windows": {
            "immediate": "lag 1",
            "early": "lags 1--4",
            "middle": "lags 5--12",
            "late": "lags 13 through the remaining horizon",
        },
        "secondary_readouts": [
            "post-pulse partner, own-group, and all untouched non-pMax error ratios",
            "per-lag common-downstream, partner, own-group, and all-group error ratios",
            "full-horizon all-group mean and official-release horizon sum",
            "decoded correlation, admissibility, boundedness, boundary, and structure diagnostics",
        ],
        "metric_contract": {
            "realm_npe_mean": "mean over 49 calls of the sum of five group MSEs",
            "realm_npe_sum_source": (
                "sum over 49 calls of the five group MSEs, matching released "
                "REALM train_rollout.py"
            ),
            "required_sum_to_mean_factor": EXPECTED_SOURCE_SUM_FACTOR,
        },
        "effect_bands": {
            "materially_helpful_ratio_at_most": 0.90,
            "materially_harmful_ratio_at_least": 1.10,
            "near_null_ratio_interval": [0.95, 1.05],
            "otherwise": "small_or_inconclusive",
        },
        "gates": {
            "one_shared_model_call_1_must_replay_baseline_bitwise": True,
            "every_arm_prefix_through_pulse_must_replay_baseline_bitwise": True,
            "pulse_must_change_at_least_one_selected_element": True,
            "accepted_pulse_state_must_equal_truth_only_on_selected_group": True,
            "exactly_one_truth_injection_per_arm": True,
            "all_six_arms_must_complete_finite_H49": True,
            "source_sum_must_equal_49_times_horizon_mean_within_float_tolerance": True,
        },
        "runtime": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8 before Python startup"},
        "sealed_test_object_opened": False,
        "anti_claims": [
            "no new training, checkpoint selection, architecture, or data",
            "an oracle hybrid-state pulse is not a deployable correction method",
            "one checkpoint and one trajectory cannot establish seed or condition robustness",
            "a pulse response cannot by itself distinguish architecture, exposure, objective, optimization, or data causes",
            "late pulses begin from the baseline deployed state even when that decoded state is inadmissible",
            "no sealed-test, state-of-the-art, physical-conservation, or REALM-wide claim",
        ],
    }
    payload["canonical_payload_sha256"] = canonical_json_sha256(payload)
    return payload


def _write_preregistration(path: Path) -> dict[str, Any]:
    if path.exists() or path.is_symlink():
        raise ValueError("pulse preregistration output already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_preregistration()
    _write_json(path, payload)
    return {
        "experiment_id": EXPERIMENT_ID,
        "preregistration": str(path),
        "canonical_payload_sha256": payload["canonical_payload_sha256"],
        "file_sha256": sha256_file(path),
    }


def _validate_preregistration(payload: Mapping[str, Any]) -> None:
    if payload != build_preregistration():
        raise ValueError("group-pulse preregistration differs from frozen contract")


def _validate_g0b_evidence(directory: Path) -> Mapping[str, Any]:
    result_path = directory / "result.json"
    final_path = directory / "final_hash_manifest.json"
    if sha256_file(result_path) != G0B_RESULT_SHA256:
        raise ValueError("G0b result file hash differs")
    if sha256_file(final_path) != G0B_FINAL_MANIFEST_SHA256:
        raise ValueError("G0b final manifest file hash differs")
    result = _load_json(result_path)
    _validate_signed_payload(result, name="G0b result")
    final_manifest = _load_json(final_path)
    if (
        result.get("schema") != feedback.RESULT_SCHEMA
        or result.get("experiment_id") != G0B_EXPERIMENT_ID
        or result.get("canonical_payload_sha256") != G0B_RESULT_PAYLOAD_SHA256
        or result.get("diagnostic_interpretation_allowed") is not True
        or result.get("test_object_opened") is not False
        or final_manifest.get("files", {}).get("result.json") != G0B_RESULT_SHA256
        or final_manifest.get("test_object_opened") is not False
    ):
        raise ValueError("G0b result contract differs")
    arms = result.get("arms")
    if not isinstance(arms, Mapping):
        raise TypeError("G0b result arms must be a mapping")
    for group, expected in G0B_PRIMARY_RATIOS.items():
        actual = arms[group]["comparison_to_unintervened_free_baseline"][
            "primary_untouched_non_pMax_groups"
        ]["ratio"]
        if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1.0e-15):
            raise ValueError("G0b selection evidence differs")
    return result


def _validate_pulse_inputs(
    baseline_free: np.ndarray,
    truth_normalized: torch.Tensor,
    *,
    pulse_call: int,
) -> None:
    if pulse_call not in PULSE_CALLS:
        raise ValueError("pulse call differs from frozen schedule")
    if (
        baseline_free.ndim != 4
        or baseline_free.shape[0] != VALIDATION_HORIZON
        or baseline_free.shape[1] != CHANNELS
        or baseline_free.dtype != np.dtype("float32")
    ):
        raise ValueError("baseline pulse array must be float32 [49,13,y,x]")
    if (
        truth_normalized.device.type != "cpu"
        or truth_normalized.dtype != DTYPE
        or truth_normalized.ndim != 4
        or truth_normalized.shape[0] != VALIDATION_HORIZON + 1
        or truth_normalized.shape[1] != CHANNELS
        or tuple(truth_normalized.shape[2:]) != tuple(baseline_free.shape[2:])
        or not bool(torch.isfinite(truth_normalized).all())
    ):
        raise ValueError("pulse truth must be finite CPU float32 [50,13,y,x]")


def _prepare_pulse_prefix(
    baseline_free: np.ndarray,
    truth_normalized: torch.Tensor,
    *,
    group: str,
    pulse_call: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    _validate_pulse_inputs(
        baseline_free,
        truth_normalized,
        pulse_call=pulse_call,
    )
    if group not in PULSE_GROUPS:
        raise ValueError("pulse group differs from frozen arms")

    spatial_shape = tuple(int(value) for value in baseline_free.shape[2:])
    predictions = torch.empty(
        (VALIDATION_HORIZON, CHANNELS, *spatial_shape), dtype=DTYPE, device="cpu"
    )
    recurrent_inputs = torch.empty_like(predictions)
    baseline_prefix = torch.from_numpy(np.array(baseline_free[:pulse_call], copy=True))
    predictions[:pulse_call].copy_(baseline_prefix)
    recurrent_inputs[0].copy_(truth_normalized[0])
    if pulse_call > 1:
        recurrent_inputs[1:pulse_call].copy_(predictions[: pulse_call - 1])

    raw_pulse = predictions[pulse_call - 1].unsqueeze(0)
    accepted, stats = feedback.inject_truth_feedback(
        raw_pulse,
        truth_normalized[pulse_call],
        group=group,
    )
    recurrent_inputs[pulse_call].copy_(accepted[0])
    prefix_exact = bool(torch.equal(predictions[:pulse_call], baseline_prefix))
    stats = {
        "pulse_call": pulse_call,
        "baseline_prefix_bitwise_exact": prefix_exact,
        "truth_injection_count": 1,
        **stats,
    }
    if not prefix_exact or int(stats["changed_selected_elements"]) <= 0:
        raise RuntimeError("pulse prefix or activity closure failed")
    return predictions, recurrent_inputs, accepted, stats


def _predict_pulse_recurrence(
    model: RealmRegularGridPCNO,
    truth_normalized: torch.Tensor,
    coordinates: torch.Tensor,
    baseline_free: np.ndarray,
    *,
    group: str,
    pulse_call: int,
    device: torch.device,
    predictor: Callable[..., torch.Tensor] = predict_one_call,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    int | None,
    dict[str, Any],
    list[float],
]:
    predictions, recurrent_inputs, accepted, activity = _prepare_pulse_prefix(
        baseline_free,
        truth_normalized,
        group=group,
        pulse_call=pulse_call,
    )
    current = accepted.to(device=device, dtype=DTYPE)
    nonfinite_call: int | None = None
    call_seconds: list[float] = []
    model.eval()
    with torch.no_grad():
        for frame in range(pulse_call, VALIDATION_HORIZON):
            started = time.monotonic()
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else nullcontext()
            )
            with autocast:
                raw = predictor(
                    model,
                    current,
                    coordinates,
                    parameterization="residual",
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            call_seconds.append(time.monotonic() - started)
            if not bool(torch.isfinite(raw).all()):
                nonfinite_call = frame + 1
                predictions = predictions[:frame]
                recurrent_inputs = recurrent_inputs[:frame]
                break
            raw_cpu = raw[0].to(device="cpu", dtype=DTYPE)
            predictions[frame].copy_(raw_cpu)
            if frame + 1 < VALIDATION_HORIZON:
                recurrent_inputs[frame + 1].copy_(raw_cpu)
            current = raw
    return predictions, recurrent_inputs, nonfinite_call, activity, call_seconds


def _group_series(summary: Mapping[str, Any], groups: Sequence[str]) -> list[float]:
    if not groups or len(groups) != len(set(groups)):
        raise ValueError("metric groups must be nonempty and unique")
    grouped = summary.get("npe_group_by_call")
    if not isinstance(grouped, Mapping) or set(grouped) != set(
        feedback.ALL_METRIC_GROUPS
    ):
        raise ValueError("summary lacks the frozen PlanarDet metric groups")
    histories = []
    for group in groups:
        if group not in feedback.ALL_METRIC_GROUPS:
            raise ValueError(f"unknown metric group: {group}")
        history = grouped[group]
        if not isinstance(history, list) or not history:
            raise ValueError("metric histories must be nonempty lists")
        histories.append(history)
    if len({len(history) for history in histories}) != 1:
        raise ValueError("metric histories must have equal length")
    result = []
    for values in zip(*histories, strict=True):
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in values
        ):
            raise ValueError("metric histories must contain finite numbers")
        result.append(sum(float(value) for value in values))
    return result


def mean_grouped_npe_window(
    summary: Mapping[str, Any],
    groups: Sequence[str],
    *,
    start_call: int,
    end_call: int,
) -> float:
    values = _group_series(summary, groups)
    if not (1 <= start_call <= end_call <= len(values)):
        raise ValueError("metric window is outside the available call range")
    selected = values[start_call - 1 : end_call]
    return sum(selected) / len(selected)


def _metric_delta(baseline: float, pulse: float) -> dict[str, float]:
    if not math.isfinite(baseline) or baseline <= 0.0 or not math.isfinite(pulse):
        raise ValueError("pulse comparison requires a positive finite baseline")
    return {
        "baseline": baseline,
        "pulse": pulse,
        "ratio": pulse / baseline,
        "delta": pulse - baseline,
    }


def _window_delta(
    baseline: Mapping[str, Any],
    pulse: Mapping[str, Any],
    groups: Sequence[str],
    *,
    start_call: int,
    end_call: int,
) -> dict[str, Any]:
    return {
        "groups": list(groups),
        "start_call": start_call,
        "end_call": end_call,
        **_metric_delta(
            mean_grouped_npe_window(
                baseline,
                groups,
                start_call=start_call,
                end_call=end_call,
            ),
            mean_grouped_npe_window(
                pulse,
                groups,
                start_call=start_call,
                end_call=end_call,
            ),
        ),
    }


def compare_pulse_summary(
    baseline: Mapping[str, Any],
    pulse: Mapping[str, Any],
    *,
    group: str,
    pulse_call: int,
) -> dict[str, Any]:
    if group not in PULSE_GROUPS or pulse_call not in PULSE_CALLS:
        raise ValueError("unknown pulse comparison arm")
    if (
        int(baseline.get("call_count", 0)) != VALIDATION_HORIZON
        or int(pulse.get("call_count", 0)) != VALIDATION_HORIZON
    ):
        raise ValueError("pulse comparison requires complete H49 summaries")
    start_call = pulse_call + 1
    partner = "rho" if group == "chem" else "chem"
    untouched_non_pmax = tuple(name for name in feedback.ARM_ORDER if name != group)
    windows = {
        "immediate": (start_call, start_call),
        "early": (start_call, min(pulse_call + 4, VALIDATION_HORIZON)),
        "middle": (pulse_call + 5, min(pulse_call + 12, VALIDATION_HORIZON)),
        "late": (pulse_call + 13, VALIDATION_HORIZON),
    }
    temporal = {
        name: _window_delta(
            baseline,
            pulse,
            COMMON_DOWNSTREAM_GROUPS,
            start_call=first,
            end_call=last,
        )
        for name, (first, last) in windows.items()
    }
    primary = _window_delta(
        baseline,
        pulse,
        COMMON_DOWNSTREAM_GROUPS,
        start_call=start_call,
        end_call=VALIDATION_HORIZON,
    )
    comparison = {
        "primary_postpulse_common_downstream": primary,
        "primary_effect_classification": feedback.classify_effect(
            primary["ratio"], complete=True
        ),
        "postpulse_partner_group": {
            "direction": f"{group}_to_{partner}",
            **_window_delta(
                baseline,
                pulse,
                (partner,),
                start_call=start_call,
                end_call=VALIDATION_HORIZON,
            ),
        },
        "postpulse_own_group": _window_delta(
            baseline,
            pulse,
            (group,),
            start_call=start_call,
            end_call=VALIDATION_HORIZON,
        ),
        "postpulse_all_untouched_non_pMax_groups": _window_delta(
            baseline,
            pulse,
            untouched_non_pmax,
            start_call=start_call,
            end_call=VALIDATION_HORIZON,
        ),
        "temporal_common_downstream": temporal,
        "all_group_realm_npe_mean": _metric_delta(
            float(baseline["realm_npe_mean"]),
            float(pulse["realm_npe_mean"]),
        ),
        "official_release_horizon_sum": _metric_delta(
            float(baseline["realm_npe_sum_source"]),
            float(pulse["realm_npe_sum_source"]),
        ),
        "decoded_correlation_case_first": {
            "baseline": baseline["decoded_correlation_case_first"],
            "pulse": pulse["decoded_correlation_case_first"],
        },
        "admissible_call_count": {
            "baseline": baseline["admissible_call_count"],
            "pulse": pulse["admissible_call_count"],
        },
        "bounded_call_count": {
            "baseline": baseline["bounded_call_count"],
            "pulse": pulse["bounded_call_count"],
        },
    }
    partner_ratio = comparison["postpulse_partner_group"]["ratio"]
    comparison["partner_effect_classification"] = feedback.classify_effect(
        partner_ratio, complete=True
    )
    return comparison


def build_lag_readout(
    baseline: Mapping[str, Any],
    pulse: Mapping[str, Any],
    *,
    group: str,
    pulse_call: int,
) -> list[dict[str, Any]]:
    partner = "rho" if group == "chem" else "chem"
    baseline_common = _group_series(baseline, COMMON_DOWNSTREAM_GROUPS)
    pulse_common = _group_series(pulse, COMMON_DOWNSTREAM_GROUPS)
    baseline_partner = _group_series(baseline, (partner,))
    pulse_partner = _group_series(pulse, (partner,))
    baseline_own = _group_series(baseline, (group,))
    pulse_own = _group_series(pulse, (group,))
    baseline_all = _group_series(baseline, feedback.ALL_METRIC_GROUPS)
    pulse_all = _group_series(pulse, feedback.ALL_METRIC_GROUPS)
    rows = []
    for call in range(pulse_call + 1, VALIDATION_HORIZON + 1):
        index = call - 1
        rows.append(
            {
                "call": call,
                "lag": call - pulse_call,
                "common_downstream_ratio": pulse_common[index] / baseline_common[index],
                "partner_ratio": pulse_partner[index] / baseline_partner[index],
                "own_group_ratio": pulse_own[index] / baseline_own[index],
                "all_group_ratio": pulse_all[index] / baseline_all[index],
            }
        )
    return rows


def classify_directionality(
    chem_to_rho_ratio: float,
    rho_to_chem_ratio: float,
    *,
    complete: bool,
) -> str:
    if not complete:
        return "not_interpretable"
    chem_helpful = chem_to_rho_ratio <= 0.90
    rho_helpful = rho_to_chem_ratio <= 0.90
    if chem_helpful and rho_helpful:
        return "bidirectional_material_partner_reduction"
    if chem_helpful:
        return "chem_to_rho_only_material_partner_reduction"
    if rho_helpful:
        return "rho_to_chem_only_material_partner_reduction"
    return "no_material_single_pulse_partner_reduction"


def _source_sum_identity(summary: Mapping[str, Any]) -> bool:
    return math.isclose(
        float(summary["realm_npe_sum_source"]),
        EXPECTED_SOURCE_SUM_FACTOR * float(summary["realm_npe_mean"]),
        rel_tol=2.0e-6,
        abs_tol=2.0e-6,
    )


def _probe_model_call_1(
    model: RealmRegularGridPCNO,
    initial: torch.Tensor,
    coordinates: torch.Tensor,
    baseline_free: np.ndarray,
    *,
    device: torch.device,
) -> None:
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        proposal = predict_one_call(
            model,
            initial.unsqueeze(0).to(device=device, dtype=DTYPE),
            coordinates,
            parameterization="residual",
        )
    torch.cuda.synchronize(device)
    actual = proposal[0].to(device="cpu", dtype=DTYPE)
    expected = torch.from_numpy(np.array(baseline_free[0], copy=True))
    if not torch.equal(actual, expected):
        raise RuntimeError("shared model call 1 differs from frozen baseline")


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    _validate_startup_environment()
    training_source = build_source_manifest(
        REPO_ROOT, entrypoints=EXECUTABLE_ENTRYPOINTS
    )
    diagnostic_source = build_diagnostic_source_manifest()
    preregistration = json.loads(args.preregistration.read_text(encoding="utf-8"))
    _validate_preregistration(preregistration)
    g0b_result = _validate_g0b_evidence(args.group_feedback_evaluation_dir)
    contract, training_status, training_inputs, training_config = (
        _validate_closed_training_tree(
            args.training_dir,
            current_source_manifest=training_source,
        )
    )
    baseline_result = _validate_baseline_evaluation(
        args.baseline_evaluation_dir,
        current_source_manifest=training_source,
    )
    if contract.run_id != BASELINE_RUN_ID:
        raise ValueError("closed training run differs from pulse baseline")

    manifest_payload, repository, revision, entries = feedback._load_open_manifest(
        args.manifest,
        expected_payload_sha256=contract.open_manifest_payload_sha256,
    )
    manifest_summary = validate_planardet_open_manifest(repository, revision, entries)
    inventory = validate_local_open_tree(args.data_root, entries)
    metadata = load_planardet_metadata(args.data_root / "data" / "data.npz")
    if (
        manifest_summary["manifest_sha256"] != PLANARDET_OPEN_MANIFEST_SHA256
        or metadata.train_groups != PLANARDET_TRAIN_GROUPS
        or metadata.val_groups != PLANARDET_VAL_GROUPS
        or training_inputs.get("open_manifest_payload_sha256")
        != canonical_json_sha256(manifest_payload)
        or training_inputs.get("normalizer_arrays_sha256")
        != contract.normalizer_arrays_sha256
        or training_inputs.get("test_object_opened") is not False
    ):
        raise ValueError("pulse inputs differ from closed training inputs")
    bundle = load_normalizer_bundle(
        args.normalizer_arrays,
        expected_sha256=contract.normalizer_arrays_sha256,
        expected_coordinates_yx=metadata.canonical_coords_yx,
    )
    normalizer = bundle.normalizer(1, dtype=DTYPE)
    validation_normalized, validation_native_batched = load_normalized_trajectories(
        args.data_root,
        split="val",
        groups=PLANARDET_VAL_GROUPS,
        normalizer=normalizer,
        retain_native=True,
    )
    if validation_native_batched is None:
        raise RuntimeError("validation native truth was not retained")
    controls = validation_control_summary(validation_normalized[0])
    validate_frozen_controls(controls, contract)
    if controls != baseline_result["controls"] or controls != g0b_result["controls"]:
        raise ValueError("pulse controls differ from frozen evaluations")

    checkpoint = torch.load(
        args.training_dir / "best.pt", map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, Mapping):
        raise TypeError("retained checkpoint root must be a mapping")
    if sha256_file(args.training_dir / "best.pt") != BASELINE_CHECKPOINT_SHA256:
        raise ValueError("retained checkpoint file hash differs")
    model_state = _validate_best_checkpoint(
        checkpoint,
        contract=contract,
        status=training_status,
        input_manifest=training_inputs,
        config=training_config,
        training_dir=args.training_dir,
        bundle=bundle,
    )

    _configure_determinism(contract.seed)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("group-pulse evaluation requires one visible CUDA device")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("group-pulse evaluation requires bfloat16 autocast support")
    device = torch.device(DEVICE_NAME)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    runtime = {
        key: value
        for key, value in _runtime_manifest(device, seed=contract.seed).items()
        if key != "canonical_payload_sha256"
    }
    runtime.update(
        {
            "schema": RUNTIME_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "diagnostic_source_manifest_digest": diagnostic_source[
                "canonical_payload_sha256"
            ],
        }
    )
    runtime["canonical_payload_sha256"] = canonical_json_sha256(runtime)

    geometry = build_realm_regular_grid_geometry(
        metadata.canonical_coords_yx,
        domain_lengths_xy=DOMAIN_LENGTHS_XY,
        released_coordinate_order=CANONICAL_COORDINATE_ORDER,
    )
    model = RealmRegularGridPCNO(
        config=RealmPCNOConfig(
            channels=CHANNELS,
            mode_counts_xy=MODE_COUNTS_XY,
            layers=(contract.width,) * 5,
            fc_dim=FC_DIM,
            zero_initialize_head=True,
        ),
        geometry=geometry,
    ).to(device=device, dtype=DTYPE)
    model.load_state_dict(model_state, strict=True)
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if parameter_count != EXPECTED_PARAMETER_COUNTS[contract.width]:
        raise RuntimeError("group-pulse PCNO parameter count differs")
    coordinates = bundle.canonical_coordinates_yx.unsqueeze(0).to(
        device=device, dtype=DTYPE
    )
    validation_truth = validation_normalized[0]
    validation_native = validation_native_batched[0]
    boundary_mask = planardet_boundary_mask(
        torch.from_numpy(metadata.x.copy()), torch.from_numpy(metadata.y.copy())
    )
    baseline_free = np.load(
        args.baseline_evaluation_dir / "free_prediction_normalized.npy",
        allow_pickle=False,
        mmap_mode="r",
    )
    if baseline_free.shape != EXPECTED_ARRAY_SHAPE or baseline_free.dtype != np.dtype(
        "float32"
    ):
        raise ValueError("baseline free array contract differs")
    _probe_model_call_1(
        model,
        validation_truth[0],
        coordinates,
        baseline_free,
        device=device,
    )

    for frozen_dir in (
        args.baseline_evaluation_dir,
        args.group_feedback_evaluation_dir,
    ):
        if _is_within(args.output_dir, frozen_dir) or _is_within(
            frozen_dir, args.output_dir
        ):
            raise ValueError("pulse output and frozen evaluations must be disjoint")
    _prepare_output_directory(
        args.output_dir,
        training_dir=args.training_dir,
        data_root=args.data_root,
    )
    _write_json(args.output_dir / "training_source_manifest.json", training_source)
    _write_json(args.output_dir / "diagnostic_source_manifest.json", diagnostic_source)
    _write_json(args.output_dir / "runtime_manifest.json", runtime)
    _write_json(args.output_dir / "preregistration.json", preregistration)

    baseline_summary = baseline_result["views"]["free_recurrence"]["summary"]
    if not _source_sum_identity(baseline_summary):
        raise ValueError("baseline mean/source-sum metric identity differs")
    arms: dict[str, Any] = {}
    arrays: dict[str, Any] = {}
    global_started = time.monotonic()
    for group, pulse_call in ARM_SPECS:
        current_arm_id = arm_id(group, pulse_call)
        predictions, recurrent_inputs, nonfinite_call, activity, call_seconds = (
            _predict_pulse_recurrence(
                model,
                validation_truth,
                coordinates,
                baseline_free,
                group=group,
                pulse_call=pulse_call,
                device=device,
            )
        )
        array_path = (
            args.output_dir
            / f"raw_pulse_{group}_call{pulse_call}_prediction_normalized.npy"
        )
        _write_npy(array_path, predictions)
        arrays[array_path.name] = {
            "sha256": sha256_file(array_path),
            "shape": list(predictions.shape),
            "dtype": str(predictions.numpy().dtype),
        }
        current_native = normalizer.decode(
            recurrent_inputs, inverse_domain_policy="nan"
        )
        summary, decoded = _view_summary(
            predictions,
            validation_truth[1 : predictions.shape[0] + 1],
            validation_native[1 : predictions.shape[0] + 1],
            current_native,
            bundle=bundle,
            boundary_mask=boundary_mask,
        )
        if summary is None:
            raise RuntimeError(f"{current_arm_id} has no finite raw call")
        complete = bool(
            predictions.shape[0] == VALIDATION_HORIZON
            and summary["all_normalized_finite"]
            and summary["all_decoded_finite"]
        )
        source_sum_identity = _source_sum_identity(summary)
        closure = {
            "baseline_prefix_through_pulse_bitwise_exact": activity[
                "baseline_prefix_bitwise_exact"
            ],
            "pulse_changed_selected_elements": int(
                activity["changed_selected_elements"]
            )
            > 0,
            "accepted_selected_group_equals_truth_exactly": activity[
                "selected_group_truth_exact"
            ],
            "accepted_other_channels_bitwise_unchanged": activity[
                "other_channels_bitwise_unchanged"
            ],
            "raw_pulse_proposal_bitwise_unchanged": activity[
                "raw_proposal_bitwise_unchanged"
            ],
            "exactly_one_truth_injection": activity["truth_injection_count"] == 1,
            "complete_finite_H49": complete,
            "official_source_sum_identity": source_sum_identity,
            "prediction_array_persisted": True,
        }
        closure["all_gates_pass"] = all(closure.values())
        comparison = (
            compare_pulse_summary(
                baseline_summary,
                summary,
                group=group,
                pulse_call=pulse_call,
            )
            if complete
            else {"primary_effect_classification": "not_interpretable"}
        )
        lag_readout = (
            build_lag_readout(
                baseline_summary,
                summary,
                group=group,
                pulse_call=pulse_call,
            )
            if complete
            else []
        )
        if not closure["all_gates_pass"]:
            comparison["primary_effect_classification"] = "not_interpretable"
        arms[current_arm_id] = {
            "group": group,
            "channels": list(feedback.GROUP_CHANNELS[group]),
            "pulse_call": pulse_call,
            "aggregation": (
                "verified baseline prefix through pulse; raw pulse scored; one "
                "selected truth group used in the next input; raw recurrence thereafter"
            ),
            "valid_length": predictions.shape[0],
            "nonfinite_proposal_call": nonfinite_call,
            "modeled_suffix_call_seconds": call_seconds,
            "raw_summary": summary,
            "pulse_activity": activity,
            "comparison_to_unintervened_free_baseline": comparison,
            "postpulse_lag_readout": lag_readout,
            "structure": _structure_summary(
                decoded,
                validation_native,
                metadata_times=metadata.times,
                x=metadata.x,
                y=metadata.y,
            ),
            "closure": closure,
        }
        del predictions, recurrent_inputs, current_native, decoded
        gc.collect()

    global_closure = {
        "closed_training_verified": True,
        "frozen_training_source_verified": True,
        "diagnostic_source_verified": True,
        "preregistration_verified": True,
        "baseline_evaluation_verified": True,
        "G0b_selection_evidence_verified": True,
        "open_tree_verified": len(inventory) == len(entries),
        "shared_model_call_1_replays_baseline_bitwise": True,
        "all_six_arms_present": set(arms)
        == {arm_id(group, call) for group, call in ARM_SPECS},
        "all_arm_gates_pass": all(
            bool(arm["closure"]["all_gates_pass"]) for arm in arms.values()
        ),
        "prediction_arrays_persisted": set(arrays)
        == {
            f"raw_pulse_{group}_call{call}_prediction_normalized.npy"
            for group, call in ARM_SPECS
        },
        "test_object_opened": False,
    }
    global_closure["all_gates_pass"] = (
        all(
            value
            for key, value in global_closure.items()
            if key != "test_object_opened"
        )
        and global_closure["test_object_opened"] is False
    )
    directionality_by_pulse = {}
    for pulse_call in PULSE_CALLS:
        chem_arm = arms[arm_id("chem", pulse_call)]
        rho_arm = arms[arm_id("rho", pulse_call)]
        complete = bool(
            chem_arm["closure"]["all_gates_pass"]
            and rho_arm["closure"]["all_gates_pass"]
        )
        chem_ratio = (
            chem_arm["comparison_to_unintervened_free_baseline"][
                "postpulse_partner_group"
            ]["ratio"]
            if complete
            else None
        )
        rho_ratio = (
            rho_arm["comparison_to_unintervened_free_baseline"][
                "postpulse_partner_group"
            ]["ratio"]
            if complete
            else None
        )
        directionality_by_pulse[str(pulse_call)] = {
            "chem_to_rho_ratio": chem_ratio,
            "rho_to_chem_ratio": rho_ratio,
            "classification": classify_directionality(
                float(chem_ratio) if chem_ratio is not None else math.nan,
                float(rho_ratio) if rho_ratio is not None else math.nan,
                complete=complete,
            ),
        }

    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "baseline_run_id": BASELINE_RUN_ID,
        "checkpoint": {
            "relative_path": "best.pt",
            "sha256": BASELINE_CHECKPOINT_SHA256,
            "completed_step": BASELINE_BEST_STEP,
            "model_state_sha256": checkpoint["model_state_sha256"],
        },
        "bindings": {
            "training_source_manifest_digest": BASELINE_SOURCE_DIGEST,
            "diagnostic_source_manifest_digest": diagnostic_source[
                "canonical_payload_sha256"
            ],
            "preregistration_payload_digest": preregistration[
                "canonical_payload_sha256"
            ],
            "preregistration_file_sha256": sha256_file(args.preregistration),
            "baseline_result_sha256": BASELINE_RESULT_SHA256,
            "baseline_final_manifest_sha256": BASELINE_FINAL_MANIFEST_SHA256,
            "G0b_result_sha256": G0B_RESULT_SHA256,
            "G0b_final_manifest_sha256": G0B_FINAL_MANIFEST_SHA256,
            "normalizer_arrays_sha256": bundle.source_sha256,
            "open_manifest_payload_sha256": canonical_json_sha256(manifest_payload),
        },
        "controls": controls,
        "pulse_calls": list(PULSE_CALLS),
        "pulse_groups": list(PULSE_GROUPS),
        "baseline_free_summary": baseline_summary,
        "arms": arms,
        "directionality_by_pulse": directionality_by_pulse,
        "evaluator_closure": global_closure,
        "diagnostic_interpretation_allowed": global_closure["all_gates_pass"],
        "prediction_arrays": arrays,
        "timing_seconds": {
            "evaluation": time.monotonic() - global_started,
            "by_arm": {
                current_arm_id: sum(
                    float(value)
                    for value in arms[current_arm_id]["modeled_suffix_call_seconds"]
                )
                for current_arm_id in arms
            },
        },
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "parameter_count": parameter_count,
        "test_object_opened": False,
        "anti_claims": preregistration["anti_claims"],
    }
    result["canonical_payload_sha256"] = canonical_json_sha256(result)
    _write_json(args.output_dir / "result.json", result)
    hashed_files = tuple(arrays) + (
        "result.json",
        "training_source_manifest.json",
        "diagnostic_source_manifest.json",
        "runtime_manifest.json",
        "preregistration.json",
    )
    final_manifest = {
        "schema": FINAL_MANIFEST_SCHEMA,
        "files": {name: sha256_file(args.output_dir / name) for name in hashed_files},
        "self_hash_excluded": True,
        "test_object_opened": False,
    }
    _write_json(args.output_dir / "final_hash_manifest.json", final_manifest)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "evaluator_closure_pass": global_closure["all_gates_pass"],
        "diagnostic_interpretation_allowed": result[
            "diagnostic_interpretation_allowed"
        ],
        "primary_ratio_by_arm": {
            current_arm_id: arms[current_arm_id][
                "comparison_to_unintervened_free_baseline"
            ]
            .get("primary_postpulse_common_downstream", {})
            .get("ratio")
            for current_arm_id in arms
        },
        "partner_ratio_by_arm": {
            current_arm_id: arms[current_arm_id][
                "comparison_to_unintervened_free_baseline"
            ]
            .get("postpulse_partner_group", {})
            .get("ratio")
            for current_arm_id in arms
        },
        "directionality_by_pulse": directionality_by_pulse,
        "result_sha256": sha256_file(args.output_dir / "result.json"),
        "final_hash_manifest_sha256": sha256_file(
            args.output_dir / "final_hash_manifest.json"
        ),
        "test_object_opened": False,
    }
    _write_json(args.output_dir / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preregister":
        summary = _write_preregistration(args.output)
    elif args.command == "evaluate":
        summary = run_evaluation(args)
    else:  # pragma: no cover - argparse enforces the closed command set.
        raise AssertionError("unreachable command")
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
