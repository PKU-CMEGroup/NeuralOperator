"""Localize PlanarDet PCNO recurrence error with oracle field-group feedback.

The learned D092-R1 step-950 proposal is always stored and scored before any
intervention.  For recurrence only, one registered field group in the accepted
state is replaced by the released next-frame truth.  This is an oracle causal
diagnostic on the single open validation trajectory, not a deployable method.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
    parse_manifest_payload,
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

EXPERIMENT_ID = "w26_l4_pd0_group_feedback_g0b_d092r1_step950_20260817a"
PREREGISTRATION_SCHEMA = "w26_l4_planardet_group_feedback_preregistration_v2"
DIAGNOSTIC_SOURCE_SCHEMA = "w26_l4_planardet_group_feedback_source_v2"
RUNTIME_SCHEMA = "w26_l4_planardet_group_feedback_cuda_runtime_v2"
RESULT_SCHEMA = "w26_l4_planardet_group_feedback_evaluation_v2"
FINAL_MANIFEST_SCHEMA = "w26_l4_planardet_group_feedback_final_hash_manifest_v2"
SUMMARY_SCHEMA = "w26_l4_planardet_group_feedback_summary_v2"

FAILED_G0_EXPERIMENT_ID = "w26_l4_pd0_group_feedback_g0_d092r1_step950_20260817a"
FAILED_G0_PREREGISTRATION_PAYLOAD_SHA256 = (
    "70de092200818f4345e6ed44c406e336c4a212f0a6b008964e3931b3895e7a7c"
)
FAILED_G0_DIAGNOSTIC_SOURCE_DIGEST = (
    "b766c9aae979cc74e275a25dbc6fec976d6d5d88eb891d13e908053fe67660c1"
)
FAILED_G0_LOG_SHA256 = (
    "80b87f543536b97f5d3ade0183ebc2290297d7d4927662812848ee818265c9dc"
)
FAILED_G0_EXIT_SHA256 = (
    "4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865"
)

GROUP_CHANNELS: dict[str, tuple[int, ...]] = {
    "chem": tuple(range(8)),
    "T": (8,),
    "rho": (9,),
    "u": (10, 11),
}
ALL_METRIC_GROUPS = ("chem", "T", "rho", "u", "p")
ARM_ORDER = ("chem", "T", "rho", "u")
MATERIAL_EFFECT_RATIO = 0.10
NEAR_NULL_RATIO = 0.05
EXPECTED_SOURCE_SUM_FACTOR = VALIDATION_HORIZON


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--baseline-evaluation-dir", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--normalizer-arrays", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def build_diagnostic_source_manifest() -> dict[str, Any]:
    relatives = (
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
        "corrects_failed_launch": {
            "experiment_id": FAILED_G0_EXPERIMENT_ID,
            "preregistration_payload_sha256": (
                FAILED_G0_PREREGISTRATION_PAYLOAD_SHA256
            ),
            "diagnostic_source_manifest_digest": (FAILED_G0_DIAGNOSTIC_SOURCE_DIGEST),
            "log_sha256": FAILED_G0_LOG_SHA256,
            "exit_sha256": FAILED_G0_EXIT_SHA256,
            "failure": (
                "the evaluator passed raw JSON text rather than the decoded mapping "
                "to parse_manifest_payload and stopped before release-data loading, "
                "checkpoint loading, learned calls, or output-directory creation"
            ),
            "scientific_interpretation_allowed": False,
            "sealed_test_object_opened": False,
        },
        "diagnostic_source_manifest_digest": source["canonical_payload_sha256"],
        "population": "one released open-validation PlanarDet trajectory",
        "arms": {
            group: {
                "channels": list(GROUP_CHANNELS[group]),
                "accepted_recurrent_state": (
                    "copy the released next-frame normalized truth into exactly "
                    f"the {group} channels after storing the raw learned proposal"
                ),
                "scored_state": "raw learned proposal before truth feedback",
            }
            for group in ARM_ORDER
        },
        "excluded_arm": {
            "group": "p",
            "reason": (
                "the separate frozen P0b causal pMax projection already produced "
                "a near-null non-pMax recurrence result"
            ),
        },
        "primary_causal_readout": (
            "for each arm, horizon-mean grouped normalized prediction error over "
            "the untouched non-pMax groups, divided by the same subset in the "
            "unintervened baseline"
        ),
        "secondary_readouts": [
            "same untouched-group readout including pMax",
            "raw own-group error after feedback recurrence",
            "all-group horizon mean and official-release horizon sum",
            "decoded correlation, admissibility, boundedness, boundary, and structure diagnostics",
        ],
        "metric_contract": {
            "realm_npe_mean": "mean over 49 calls of the sum of five group MSEs",
            "realm_npe_sum_source": (
                "sum over 49 calls of the five group MSEs, matching the released "
                "REALM train_rollout.py implementation"
            ),
            "required_sum_to_mean_factor": EXPECTED_SOURCE_SUM_FACTOR,
        },
        "effect_bands": {
            "materially_helpful_ratio_at_most": 1.0 - MATERIAL_EFFECT_RATIO,
            "materially_harmful_ratio_at_least": 1.0 + MATERIAL_EFFECT_RATIO,
            "near_null_ratio_interval": [
                1.0 - NEAR_NULL_RATIO,
                1.0 + NEAR_NULL_RATIO,
            ],
            "otherwise": "small_or_inconclusive",
        },
        "gates": {
            "raw_call_1_must_match_baseline_bitwise_in_every_arm": True,
            "raw_proposals_must_be_scored_before_feedback": True,
            "accepted_state_must_equal_truth_on_only_the_registered_group": True,
            "all_four_arms_must_complete_finite_H49": True,
            "source_sum_must_equal_49_times_horizon_mean_within_float_tolerance": True,
        },
        "runtime": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8 before Python startup"},
        "sealed_test_object_opened": False,
        "anti_claims": [
            "no new training, checkpoint selection, architecture, or data",
            "oracle truth feedback is not a deployable prediction method",
            "no sealed-test, seed-robustness, architecture-level, or REALM-wide claim",
            "an own-group score change is secondary and cannot by itself localize cross-group recurrence",
            "this diagnostic cannot establish physical conservation",
        ],
    }
    payload["canonical_payload_sha256"] = canonical_json_sha256(payload)
    return payload


def _validate_preregistration(payload: Mapping[str, Any]) -> None:
    if payload != build_preregistration():
        raise ValueError("group-feedback preregistration differs from frozen contract")


def _load_open_manifest(
    path: Path, *, expected_payload_sha256: str
) -> tuple[Mapping[str, Any], str, str, tuple[Any, ...]]:
    payload = _load_json(path)
    if canonical_json_sha256(payload) != expected_payload_sha256:
        raise ValueError("open manifest payload differs from preregistration")
    repository, revision, entries = parse_manifest_payload(payload)
    return payload, repository, revision, entries


def inject_truth_feedback(
    raw_proposal: torch.Tensor,
    truth_next: torch.Tensor,
    *,
    group: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return the accepted recurrent state without mutating the raw proposal."""

    if group not in GROUP_CHANNELS:
        raise ValueError(f"unknown feedback group: {group}")
    expected_truth_shape = tuple(raw_proposal.shape[1:])
    if (
        raw_proposal.device.type != "cpu"
        or truth_next.device.type != "cpu"
        or raw_proposal.dtype != DTYPE
        or truth_next.dtype != DTYPE
        or raw_proposal.ndim != 4
        or raw_proposal.shape[0] != 1
        or raw_proposal.shape[1] != CHANNELS
        or tuple(truth_next.shape) != expected_truth_shape
    ):
        raise ValueError(
            "feedback requires CPU float32 raw [1,13,y,x] and truth [13,y,x]"
        )
    if not bool(torch.isfinite(raw_proposal).all()) or not bool(
        torch.isfinite(truth_next).all()
    ):
        raise ValueError("feedback states must be finite")

    channels = GROUP_CHANNELS[group]
    untouched = tuple(index for index in range(CHANNELS) if index not in channels)
    raw_snapshot = raw_proposal.clone()
    accepted = raw_proposal.clone()
    selected = list(channels)
    untouched_list = list(untouched)
    accepted[:, selected] = truth_next[selected].unsqueeze(0)
    changed = accepted[:, selected] != raw_snapshot[:, selected]
    stats = {
        "group": group,
        "channels": list(channels),
        "selected_group_truth_exact": bool(
            torch.equal(accepted[:, selected], truth_next[selected].unsqueeze(0))
        ),
        "other_channels_bitwise_unchanged": bool(
            torch.equal(accepted[:, untouched_list], raw_snapshot[:, untouched_list])
        ),
        "raw_proposal_bitwise_unchanged": bool(torch.equal(raw_proposal, raw_snapshot)),
        "changed_selected_elements": int(changed.sum().item()),
        "selected_elements": int(changed.numel()),
    }
    if not all(
        stats[key]
        for key in (
            "selected_group_truth_exact",
            "other_channels_bitwise_unchanged",
            "raw_proposal_bitwise_unchanged",
        )
    ):
        raise RuntimeError("truth-feedback bitwise closure failed")
    return accepted, stats


def mean_grouped_npe(summary: Mapping[str, Any], groups: Sequence[str]) -> float:
    if not groups or len(set(groups)) != len(groups):
        raise ValueError("metric groups must be nonempty and unique")
    grouped = summary.get("npe_group_by_call")
    if not isinstance(grouped, Mapping) or set(grouped) != set(ALL_METRIC_GROUPS):
        raise ValueError("view summary lacks the frozen PlanarDet metric groups")
    histories = []
    for group in groups:
        if group not in ALL_METRIC_GROUPS:
            raise ValueError(f"unknown metric group: {group}")
        values = grouped[group]
        if not isinstance(values, list) or not values:
            raise ValueError("group histories must be nonempty lists")
        histories.append(values)
    if len({len(values) for values in histories}) != 1:
        raise ValueError("group histories must have equal length")
    per_call = []
    for items in zip(*histories, strict=True):
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in items
        ):
            raise ValueError("group histories must contain finite numeric values")
        per_call.append(sum(float(value) for value in items))
    return sum(per_call) / len(per_call)


def classify_effect(ratio: float, *, complete: bool) -> str:
    if not complete or not math.isfinite(ratio) or ratio < 0.0:
        return "not_interpretable"
    if ratio <= 1.0 - MATERIAL_EFFECT_RATIO:
        return "materially_helpful"
    if ratio >= 1.0 + MATERIAL_EFFECT_RATIO:
        return "materially_harmful"
    if 1.0 - NEAR_NULL_RATIO <= ratio <= 1.0 + NEAR_NULL_RATIO:
        return "near_null"
    return "small_or_inconclusive"


def _metric_delta(baseline: float, feedback: float) -> dict[str, float]:
    if not math.isfinite(baseline) or baseline <= 0.0 or not math.isfinite(feedback):
        raise ValueError(
            "metric comparison requires finite values and positive baseline"
        )
    return {
        "baseline": baseline,
        "feedback": feedback,
        "ratio": feedback / baseline,
        "delta": feedback - baseline,
    }


def compare_feedback_summary(
    baseline: Mapping[str, Any],
    feedback: Mapping[str, Any],
    *,
    group: str,
) -> dict[str, Any]:
    if group not in GROUP_CHANNELS:
        raise ValueError(f"unknown feedback group: {group}")
    primary_groups = tuple(name for name in ARM_ORDER if name != group)
    secondary_groups = primary_groups + ("p",)
    comparison = {
        "primary_untouched_non_pMax_groups": {
            "groups": list(primary_groups),
            **_metric_delta(
                mean_grouped_npe(baseline, primary_groups),
                mean_grouped_npe(feedback, primary_groups),
            ),
        },
        "secondary_untouched_groups_including_pMax": {
            "groups": list(secondary_groups),
            **_metric_delta(
                mean_grouped_npe(baseline, secondary_groups),
                mean_grouped_npe(feedback, secondary_groups),
            ),
        },
        "raw_own_group": {
            "group": group,
            **_metric_delta(
                mean_grouped_npe(baseline, (group,)),
                mean_grouped_npe(feedback, (group,)),
            ),
        },
        "all_group_realm_npe_mean": _metric_delta(
            float(baseline["realm_npe_mean"]),
            float(feedback["realm_npe_mean"]),
        ),
        "official_release_horizon_sum": _metric_delta(
            float(baseline["realm_npe_sum_source"]),
            float(feedback["realm_npe_sum_source"]),
        ),
        "decoded_correlation_case_first": {
            "baseline": baseline["decoded_correlation_case_first"],
            "feedback": feedback["decoded_correlation_case_first"],
        },
        "admissible_call_count": {
            "baseline": baseline["admissible_call_count"],
            "feedback": feedback["admissible_call_count"],
        },
        "bounded_call_count": {
            "baseline": baseline["bounded_call_count"],
            "feedback": feedback["bounded_call_count"],
        },
    }
    comparison["primary_effect_classification"] = classify_effect(
        comparison["primary_untouched_non_pMax_groups"]["ratio"],
        complete=int(feedback["call_count"]) == VALIDATION_HORIZON,
    )
    return comparison


def _predict_feedback_recurrence(
    model: RealmRegularGridPCNO,
    initial: torch.Tensor,
    truth_normalized: torch.Tensor,
    coordinates: torch.Tensor,
    baseline_free: np.ndarray,
    *,
    group: str,
    device: torch.device,
) -> tuple[torch.Tensor, int | None, list[dict[str, Any]], list[float]]:
    predictions = torch.empty(
        (VALIDATION_HORIZON, *initial.shape), dtype=DTYPE, device="cpu"
    )
    current = initial.unsqueeze(0).to(device)
    nonfinite_call: int | None = None
    records: list[dict[str, Any]] = []
    call_seconds: list[float] = []
    model.eval()
    with torch.no_grad():
        for frame in range(VALIDATION_HORIZON):
            started = time.monotonic()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                raw = predict_one_call(
                    model,
                    current,
                    coordinates,
                    parameterization="residual",
                )
            if not bool(torch.isfinite(raw).all()):
                nonfinite_call = frame + 1
                predictions = predictions[:frame]
                break
            raw_cpu = raw[0].to(device="cpu", dtype=DTYPE)
            if frame == 0:
                expected = torch.from_numpy(np.array(baseline_free[0], copy=True))
                if not torch.equal(raw_cpu, expected):
                    raise RuntimeError(f"{group} raw call 1 differs from baseline")
            predictions[frame].copy_(raw_cpu)
            accepted, stats = inject_truth_feedback(
                raw_cpu.unsqueeze(0), truth_normalized[frame + 1], group=group
            )
            torch.cuda.synchronize(device)
            call_seconds.append(time.monotonic() - started)
            records.append({"call": frame + 1, **stats})
            current = accepted.to(device=device, dtype=DTYPE)
    return predictions, nonfinite_call, records, call_seconds


def _aggregate_feedback(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = sum(int(record["selected_elements"]) for record in records)
    changed = sum(int(record["changed_selected_elements"]) for record in records)
    return {
        "call_count": len(records),
        "selected_element_calls": total,
        "changed_selected_element_calls": changed,
        "changed_selected_fraction": changed / total if total else 0.0,
        "all_selected_groups_truth_exact": all(
            bool(record["selected_group_truth_exact"]) for record in records
        ),
        "all_other_channels_bitwise_unchanged": all(
            bool(record["other_channels_bitwise_unchanged"]) for record in records
        ),
        "all_raw_proposals_bitwise_unchanged": all(
            bool(record["raw_proposal_bitwise_unchanged"]) for record in records
        ),
        "by_call": list(records),
    }


def _source_sum_identity(summary: Mapping[str, Any]) -> bool:
    mean = float(summary["realm_npe_mean"])
    source_sum = float(summary["realm_npe_sum_source"])
    return math.isclose(
        source_sum,
        EXPECTED_SOURCE_SUM_FACTOR * mean,
        rel_tol=2.0e-6,
        abs_tol=2.0e-6,
    )


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    _validate_startup_environment()
    training_source = build_source_manifest(
        REPO_ROOT, entrypoints=EXECUTABLE_ENTRYPOINTS
    )
    diagnostic_source = build_diagnostic_source_manifest()
    preregistration = json.loads(args.preregistration.read_text(encoding="utf-8"))
    _validate_preregistration(preregistration)
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
        raise ValueError("closed training run differs from feedback baseline")

    manifest_payload, repository, revision, entries = _load_open_manifest(
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
        raise ValueError("feedback inputs differ from closed training inputs")
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
    if controls != baseline_result["controls"]:
        raise ValueError("feedback controls differ from baseline evaluation")

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
        raise RuntimeError("group-feedback evaluation requires one visible CUDA device")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "group-feedback evaluation requires bfloat16 autocast support"
        )
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
        raise RuntimeError("group-feedback PCNO parameter count differs")
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

    if _is_within(args.output_dir, args.baseline_evaluation_dir) or _is_within(
        args.baseline_evaluation_dir, args.output_dir
    ):
        raise ValueError("feedback output and baseline evaluation must be disjoint")
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
    for group in ARM_ORDER:
        predictions, nonfinite_call, records, call_seconds = (
            _predict_feedback_recurrence(
                model,
                validation_truth[0],
                validation_truth,
                coordinates,
                baseline_free,
                group=group,
                device=device,
            )
        )
        array_path = args.output_dir / f"raw_feedback_{group}_prediction_normalized.npy"
        _write_npy(array_path, predictions)
        arrays[array_path.name] = {
            "sha256": sha256_file(array_path),
            "shape": list(predictions.shape),
            "dtype": str(predictions.numpy().dtype),
        }
        decoded_for_current = normalizer.decode(
            predictions, inverse_domain_policy="nan"
        )
        current_native = torch.cat(
            (validation_native[:1], decoded_for_current[:-1]), dim=0
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
            raise RuntimeError(f"{group} feedback recurrence has no finite raw call")
        activity = _aggregate_feedback(records)
        complete = bool(
            predictions.shape[0] == VALIDATION_HORIZON
            and summary["all_normalized_finite"]
            and summary["all_decoded_finite"]
        )
        source_sum_identity = _source_sum_identity(summary)
        closure = {
            "raw_call_1_replays_baseline_bitwise": len(records) >= 1,
            "raw_proposals_scored_before_feedback": True,
            "accepted_selected_group_equals_truth_exactly": activity[
                "all_selected_groups_truth_exact"
            ],
            "accepted_other_channels_bitwise_unchanged": activity[
                "all_other_channels_bitwise_unchanged"
            ],
            "raw_proposals_bitwise_unchanged_by_injection": activity[
                "all_raw_proposals_bitwise_unchanged"
            ],
            "complete_finite_H49": complete,
            "official_source_sum_identity": source_sum_identity,
            "prediction_array_persisted": True,
        }
        closure["all_gates_pass"] = all(closure.values())
        comparison = compare_feedback_summary(baseline_summary, summary, group=group)
        if not closure["all_gates_pass"]:
            comparison["primary_effect_classification"] = "not_interpretable"
        arms[group] = {
            "group": group,
            "channels": list(GROUP_CHANNELS[group]),
            "aggregation": (
                "one trajectory from released frame zero; raw proposals scored, "
                "registered truth group used only in next recurrent input"
            ),
            "valid_length": predictions.shape[0],
            "nonfinite_proposal_call": nonfinite_call,
            "call_seconds": call_seconds,
            "raw_summary": summary,
            "feedback_activity": activity,
            "comparison_to_unintervened_free_baseline": comparison,
            "structure": _structure_summary(
                decoded,
                validation_native,
                metadata_times=metadata.times,
                x=metadata.x,
                y=metadata.y,
            ),
            "closure": closure,
        }
        del predictions, decoded_for_current, current_native, decoded
        gc.collect()

    global_closure = {
        "closed_training_verified": True,
        "frozen_training_source_verified": True,
        "diagnostic_source_verified": True,
        "preregistration_verified": True,
        "baseline_evaluation_verified": True,
        "open_tree_verified": len(inventory) == len(entries),
        "all_four_arms_present": set(arms) == set(ARM_ORDER),
        "all_arm_gates_pass": all(
            bool(arm["closure"]["all_gates_pass"]) for arm in arms.values()
        ),
        "prediction_arrays_persisted": set(arrays)
        == {f"raw_feedback_{group}_prediction_normalized.npy" for group in ARM_ORDER},
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
            "normalizer_arrays_sha256": bundle.source_sha256,
            "open_manifest_payload_sha256": canonical_json_sha256(manifest_payload),
        },
        "controls": controls,
        "baseline_free_summary": baseline_summary,
        "arms": arms,
        "evaluator_closure": global_closure,
        "diagnostic_interpretation_allowed": global_closure["all_gates_pass"],
        "prediction_arrays": arrays,
        "timing_seconds": {
            "evaluation": time.monotonic() - global_started,
            "by_arm": {
                group: sum(float(value) for value in arms[group]["call_seconds"])
                for group in ARM_ORDER
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
        "primary_effect_classification_by_arm": {
            group: arms[group]["comparison_to_unintervened_free_baseline"][
                "primary_effect_classification"
            ]
            for group in ARM_ORDER
        },
        "primary_ratio_by_arm": {
            group: arms[group]["comparison_to_unintervened_free_baseline"][
                "primary_untouched_non_pMax_groups"
            ]["ratio"]
            for group in ARM_ORDER
        },
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
    summary = run_evaluation(args)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
