#!/usr/bin/env python3
"""D076: qualify a short-window response controller for native PCNO correction."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.time_dependent_no.evaluate_pcno_native_residual_correction as parent
from scripts.time_dependent_no.evaluate_pcno_node_type_interventions import CaseData
from utility.time_dependent_no.pcno_artifacts import (
    atomic_write_json_with_paths as write_json,
)
from utility.time_dependent_no.pcno_artifacts import (
    git_head,
    git_state,
    jsonable_args,
    runtime_environment,
    sha256_file,
    write_csv_with_paths,
)
from utility.time_dependent_no.pcno_euler2d import PCNOEuler2DShardStore
from utility.time_dependent_no.pcno_residual_structure import cosine
from utility.time_dependent_no.pcno_resolution_transfer import (
    build_resolution_checkpoint_model,
    load_resolution_checkpoint,
)
from utility.time_dependent_no.pcno_runtime import select_device

SCHEMA = "pcno_response_gain_controller_diagnostic_v1"
EXPERIMENT_CONTRACT = "d076_response_probe"
PROBE_CALLS = 5
PROBE_K = 3
FEATURE_SCALE_FLOOR = 1.0e-12
VORTEX_DENOMINATOR_FLOOR = 1.0e-14
CALIBRATION_REQUIRED_NONZERO = 12
CALIBRATION_ENDPOINT_LIMIT = 0.95
CALIBRATION_RESIDUAL_LIMIT = 1.0
EVALUATION_ENDPOINT_LIMIT = 0.95
STATIC_ENDPOINT_LIMIT = 0.98
STATIC_RESIDUAL_LIMIT = 1.0
PROBE_ABSOLUTE_REPLAY_LIMIT = 2.0e-5
PROBE_RELATIVE_REPLAY_LIMIT = 1.0e-7
DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
POSITIVE_GAINS = (0.125, 0.25, 0.5, 1.0)
FEATURE_NAMES = (
    "initial_vortex_amplitude",
    "initial_vortex_centroid_y",
    "response_state_rms",
    "response_integral__rho",
    "response_integral__rho_u",
    "response_integral__rho_v",
    "response_integral__energy",
    "response_amplification",
    "response_correction_cosine",
)


@dataclass(frozen=True)
class ProbeRollout:
    """Target-free recurrent prefix used only for response-based gain choice."""

    candidate: parent.Candidate
    complete: bool
    states: np.ndarray
    corrections: np.ndarray
    admissibility_rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ResponseExperimentSpec:
    """Identity and fold behavior for one immutable response-controller run."""

    experiment_id: str
    schema: str
    experiment_contract: str
    description: str
    smoke_calibration_case_count: int
    group_builder: Callable[[Sequence[CaseData]], Mapping[str, str]] | None
    group_contract: str | None
    require_group_amplitude_match: bool
    require_highest_group_abstention: bool
    null_not_run_evaluation: bool
    method_claim: str
    extra_source_paths: tuple[Path, ...]
    deterministic_algorithms: bool = False
    evaluation_runner: Callable[..., dict[str, Any]] | None = None


D076_SPEC = ResponseExperimentSpec(
    experiment_id="D076",
    schema=SCHEMA,
    experiment_contract=EXPERIMENT_CONTRACT,
    description=__doc__ or "D076 response controller",
    smoke_calibration_case_count=4,
    group_builder=None,
    group_contract=None,
    require_group_amplitude_match=False,
    require_highest_group_abstention=False,
    null_not_run_evaluation=False,
    method_claim=(
        "offline rank-8 bias with a five-call target-free response selector; "
        "not data assimilation"
    ),
    extra_source_paths=(),
)


def _candidate_inventory(*, smoke: bool) -> tuple[parent.Candidate, ...]:
    gains = (0.125, 0.5) if smoke else POSITIVE_GAINS
    return (
        parent.Candidate(key="zero", rank=0, gain=0.0),
        *(
            parent.Candidate(
                key=f"rank8_gain{parent._gain_key(gain)}_raw",
                rank=8,
                gain=gain,
            )
            for gain in gains
        ),
    )


def parse_args_for_experiment(
    argv: Sequence[str] | None,
    spec: ResponseExperimentSpec,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=spec.description)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization-json", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--family-root", type=Path, required=True)
    parser.add_argument("--multires-reference-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--amp", choices=("none",), default="none")
    parser.add_argument("--shock-quantile", type=float, default=0.9)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-normalization-sha256", required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--expected-data-manifest-digest", required=True)
    parser.add_argument("--expected-family-manifest-sha256", required=True)
    parser.add_argument("--expected-source-base-git-head", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument(
        "--expected-source", action="append", default=[], metavar="PATH=SHA256"
    )
    parser.add_argument("--visualization-cases", nargs="*", default=())
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.shock_quantile != parent.FROZEN_SHOCK_QUANTILE:
        raise ValueError(f"{spec.experiment_id} freezes --shock-quantile=0.9")
    source_head = args.expected_source_base_git_head.lower()
    if len(source_head) != 40 or any(
        character not in "0123456789abcdef" for character in source_head
    ):
        raise ValueError(
            f"{spec.experiment_id} source base must be one 40-digit hexadecimal commit"
        )
    required = parent.REQUIRED_ARTIFACTS["dynamic_fv"]
    observed = {
        "checkpoint": args.expected_checkpoint_sha256.lower(),
        "normalization": args.expected_normalization_sha256.lower(),
        "split": args.expected_split_sha256.lower(),
        "data_manifest": args.expected_data_manifest_digest.lower(),
        "family_manifest": args.expected_family_manifest_sha256.lower(),
    }
    if observed != required:
        raise ValueError(f"{spec.experiment_id} artifact contract changed: {observed}")

    args.family = "dynamic_fv"
    args.experiment_contract = spec.experiment_contract
    args.deterministic_algorithms = spec.deterministic_algorithms
    args.cublas_workspace_config = (
        DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG
        if spec.deterministic_algorithms
        else None
    )
    args.rollout_calls = 2 if args.smoke else 30
    args.probe_calls = 1 if args.smoke else PROBE_CALLS
    args.resolutions = ("250x100",)
    args.training_resolution = "250x100"
    args.bump_replay_root = None
    args.bump_replay_cases = ()
    args.bump_replay_calls = 20
    args.bump_replay_absolute_limit = None
    args.bump_replay_relative_limit = None
    return args


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return parse_args_for_experiment(argv, D076_SPEC)


def _rollout_probe(
    model: torch.nn.Module,
    case: CaseData,
    candidate: parent.Candidate,
    *,
    initial_state: np.ndarray,
    bias_sequence: np.ndarray | None,
    probe_calls: int,
) -> ProbeRollout:
    """Run a prefix without accepting any truth state after the supplied input."""

    initial = np.asarray(initial_state, dtype=np.float64)
    expected_shape = (case.nodes.shape[0], len(parent.COMPONENTS))
    if initial.shape != expected_shape or not np.isfinite(initial).all():
        raise ValueError("probe initial state differs from the native case shape")
    if probe_calls < 1:
        raise ValueError("probe_calls must be positive")
    if candidate.is_zero:
        if bias_sequence is not None:
            raise ValueError("zero probe must not receive a bias sequence")
    elif (
        bias_sequence is None
        or bias_sequence.ndim != 3
        or bias_sequence.shape[0] < probe_calls
        or bias_sequence.shape[1:] != expected_shape
    ):
        raise ValueError("positive probe bias does not cover the frozen prefix")

    current = np.array(initial, copy=True)
    states = [current.copy()]
    corrections = []
    rows = []
    for call in range(1, probe_calls + 1):
        base_prediction = parent._predict(model, case, current)
        base_summary = parent._safe_admissibility_summary(
            base_prediction, gamma=case.gamma
        )
        if not base_summary["finite"]:
            rows.append(
                {
                    "call": call,
                    "accepted": False,
                    "failure_stage": "base_prediction_nonfinite",
                    **base_summary,
                }
            )
            break
        correction = (
            np.zeros_like(base_prediction)
            if candidate.is_zero
            else -candidate.gain * bias_sequence[call - 1]
        )
        prediction = base_prediction + correction
        summary = parent._safe_admissibility_summary(prediction, gamma=case.gamma)
        accepted = bool(summary["finite"] and summary["admissible"])
        rows.append(
            {
                "call": call,
                "accepted": accepted,
                "failure_stage": None if accepted else "proposal_inadmissible",
                **summary,
            }
        )
        if not summary["finite"]:
            break
        states.append(np.asarray(prediction, dtype=np.float64))
        corrections.append(np.asarray(correction, dtype=np.float64))
        if not accepted:
            break
        current = prediction
    return ProbeRollout(
        candidate=candidate,
        complete=bool(
            len(corrections) == probe_calls and all(r["accepted"] for r in rows)
        ),
        states=np.asarray(states, dtype=np.float64),
        corrections=np.asarray(corrections, dtype=np.float64),
        admissibility_rows=tuple(rows),
    )


def _initial_vortex_features(
    case: CaseData, initial_state: np.ndarray
) -> tuple[float, float]:
    values = np.asarray(initial_state, dtype=np.float64)
    if values.shape != (case.nodes.shape[0], len(parent.COMPONENTS)):
        raise ValueError("initial vortex feature state has the wrong shape")
    interior = np.asarray(case.physical_node_type == 0, dtype=bool)
    if not np.any(interior):
        raise ValueError("D076 requires nonempty dynamic type-0 support")
    weights = np.asarray(case.weights[interior], dtype=np.float64)
    momentum = values[interior, parent.COMPONENTS.index("rho_v")]
    mass = float(weights.sum())
    mean = float(np.dot(weights, momentum) / mass)
    centered_square = np.square(momentum - mean)
    denominator = float(np.dot(weights, centered_square))
    if not np.isfinite(denominator) or denominator <= VORTEX_DENOMINATOR_FLOOR:
        raise ValueError("initial transverse-momentum variance is unresolved")
    amplitude = float(
        np.sqrt(denominator / mass) / case.state_scale[parent.COMPONENTS.index("rho_v")]
    )
    centroid = float(
        np.dot(weights * centered_square, case.nodes[interior, 1]) / denominator
    )
    if not np.isfinite(amplitude) or not np.isfinite(centroid):
        raise ValueError("initial vortex features are nonfinite")
    return amplitude, centroid


def _response_probe_features(
    case: CaseData,
    baseline: ProbeRollout | parent.RolloutResult,
    corrected: ProbeRollout | parent.RolloutResult,
    *,
    probe_calls: int,
) -> dict[str, float]:
    """Return the exact target-free feature vector frozen by D076."""

    if corrected.candidate.is_zero:
        raise ValueError("response features require a positive-gain arm")
    if not baseline.complete or not corrected.complete:
        raise ValueError("response features require complete probe prefixes")
    if (
        baseline.states.shape[0] <= probe_calls
        or corrected.states.shape[0] <= probe_calls
        or corrected.corrections.shape[0] < probe_calls
    ):
        raise ValueError("response feature payload is shorter than the probe horizon")
    if not np.array_equal(baseline.states[0], corrected.states[0]):
        raise ValueError("response probes do not share an exact initial state")

    amplitude, centroid = _initial_vortex_features(case, baseline.states[0])
    response = corrected.states[probe_calls] - baseline.states[probe_calls]
    direct = corrected.corrections[:probe_calls].sum(axis=0)
    response_norm = parent._weighted_rms(
        response,
        weights=case.weights,
        scale=case.state_scale,
    )
    direct_norm = parent._weighted_rms(
        direct,
        weights=case.weights,
        scale=case.state_scale,
    )
    if (
        response_norm is None
        or direct_norm is None
        or direct_norm <= FEATURE_SCALE_FLOOR
    ):
        raise ValueError("D076 response or direct-correction scale is unresolved")
    response_integrals = (
        np.einsum("n,nc->c", case.weights, response, optimize=True) / case.state_scale
    )
    response_cosine = cosine(
        response,
        direct,
        volumes=case.weights,
        component_scale=case.state_scale,
    )
    if response_cosine is None:
        raise ValueError("D076 response/correction cosine is unresolved")
    result = {
        "initial_vortex_amplitude": amplitude,
        "initial_vortex_centroid_y": centroid,
        "response_state_rms": float(response_norm),
        **{
            f"response_integral__{name}": float(response_integrals[index])
            for index, name in enumerate(parent.COMPONENTS)
        },
        "response_amplification": float(
            response_norm / (direct_norm + FEATURE_SCALE_FLOOR)
        ),
        "response_correction_cosine": float(response_cosine),
    }
    if set(result) != set(FEATURE_NAMES) or not all(
        np.isfinite(value) for value in result.values()
    ):
        raise AssertionError("D076 response feature inventory is invalid")
    return result


def _policy_statistics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) < PROBE_K:
        raise ValueError("response policy has fewer than three training rows")
    matrix = np.asarray(
        [[float(row[name]) for name in FEATURE_NAMES] for row in rows],
        dtype=np.float64,
    )
    if not np.isfinite(matrix).all():
        raise ValueError("response policy features are nonfinite")
    mean = matrix.mean(axis=0)
    observed_scale = matrix.std(axis=0)
    scale = np.maximum(observed_scale, FEATURE_SCALE_FLOOR)
    return {
        "feature_mean": mean,
        "feature_scale": scale,
        "zero_variance_features": [
            FEATURE_NAMES[index]
            for index, value in enumerate(observed_scale)
            if value < FEATURE_SCALE_FLOOR
        ],
        "maximum_initial_vortex_amplitude": float(
            max(float(row["initial_vortex_amplitude"]) for row in rows)
        ),
        "training_case_ids": [str(row["case_id"]) for row in rows],
    }


def _frozen_policy_record(
    table_rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[parent.Candidate],
    *,
    schema: str = SCHEMA,
) -> dict[str, Any]:
    by_candidate = {}
    for candidate in candidates:
        if candidate.is_zero:
            continue
        rows = [row for row in table_rows if row["candidate"] == candidate.key]
        stats = _policy_statistics(rows)
        by_candidate[candidate.key] = {
            "gain": candidate.gain,
            "feature_mean": stats["feature_mean"].tolist(),
            "feature_scale": stats["feature_scale"].tolist(),
            "zero_variance_features": stats["zero_variance_features"],
            "maximum_initial_vortex_amplitude": stats[
                "maximum_initial_vortex_amplitude"
            ],
            "training_case_ids": stats["training_case_ids"],
        }
    return {
        "schema": schema,
        "feature_names": list(FEATURE_NAMES),
        "neighbors": PROBE_K,
        "feature_scale_floor": FEATURE_SCALE_FLOOR,
        "upper_amplitude_extrapolation": "reject_positive_gain",
        "lower_amplitude_extrapolation": "allowed_adaptive_open_pilot",
        "candidates": by_candidate,
    }


def _select_response_gain(
    query_rows: Mapping[str, Mapping[str, Any]],
    table_rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[parent.Candidate],
    *,
    excluded_case_id: str | None = None,
    excluded_case_ids: Sequence[str] | None = None,
    frozen_policy: Mapping[str, Any] | None = None,
) -> tuple[parent.Candidate, list[dict[str, Any]], list[dict[str, Any]]]:
    """Choose one gain using only query features and calibration outcome labels."""

    if excluded_case_id is not None and excluded_case_ids is not None:
        raise ValueError("specify one response-policy exclusion form")
    excluded = (
        {str(excluded_case_id)}
        if excluded_case_id is not None
        else {str(value) for value in excluded_case_ids or ()}
    )
    if frozen_policy is not None and excluded:
        raise ValueError("frozen evaluation policy cannot exclude a calibration case")
    choice_rows = []
    neighbor_rows = []
    eligible = []
    for candidate in candidates:
        if candidate.is_zero:
            continue
        if candidate.key not in query_rows:
            raise ValueError(f"query lacks response features for {candidate.key}")
        training = [
            row
            for row in table_rows
            if row["candidate"] == candidate.key and str(row["case_id"]) not in excluded
        ]
        if len(training) < PROBE_K:
            raise ValueError("response policy fold has fewer than three neighbors")
        if frozen_policy is None:
            stats = _policy_statistics(training)
            mean = np.asarray(stats["feature_mean"], dtype=np.float64)
            scale = np.asarray(stats["feature_scale"], dtype=np.float64)
            maximum_amplitude = stats["maximum_initial_vortex_amplitude"]
        else:
            if tuple(frozen_policy.get("feature_names", ())) != FEATURE_NAMES:
                raise ValueError("frozen response-policy feature inventory changed")
            record = frozen_policy["candidates"][candidate.key]
            if set(record["training_case_ids"]) != {
                str(row["case_id"]) for row in training
            }:
                raise ValueError("frozen response-policy case inventory changed")
            mean = np.asarray(record["feature_mean"], dtype=np.float64)
            scale = np.asarray(record["feature_scale"], dtype=np.float64)
            maximum_amplitude = float(record["maximum_initial_vortex_amplitude"])
        query = np.asarray(
            [float(query_rows[candidate.key][name]) for name in FEATURE_NAMES],
            dtype=np.float64,
        )
        matrix = np.asarray(
            [[float(row[name]) for name in FEATURE_NAMES] for row in training],
            dtype=np.float64,
        )
        standardized_matrix = (matrix - mean[None, :]) / scale[None, :]
        standardized_query = (query - mean) / scale
        distances = np.linalg.norm(
            standardized_matrix - standardized_query[None, :], axis=1
        )
        order = sorted(
            range(len(training)),
            key=lambda index: (
                float(distances[index]),
                str(training[index]["case_id"]),
            ),
        )
        neighbors = [training[index] for index in order[:PROBE_K]]
        upper_support_pass = bool(
            query_rows[candidate.key]["initial_vortex_amplitude"]
            <= maximum_amplitude + FEATURE_SCALE_FLOOR
        )
        neighbors_safe = bool(all(bool(row["safe"]) for row in neighbors))
        candidate_eligible = bool(upper_support_pass and neighbors_safe)
        predicted_endpoint = float(
            np.median([float(row["endpoint_state_ratio"]) for row in neighbors])
        )
        choice_rows.append(
            {
                "candidate": candidate.key,
                "gain": candidate.gain,
                "eligible": candidate_eligible,
                "upper_amplitude_support_pass": upper_support_pass,
                "neighbors_safe": neighbors_safe,
                "predicted_endpoint_ratio": predicted_endpoint,
                "maximum_training_initial_vortex_amplitude": maximum_amplitude,
                "query_initial_vortex_amplitude": float(
                    query_rows[candidate.key]["initial_vortex_amplitude"]
                ),
            }
        )
        for rank, index in enumerate(order[:PROBE_K], start=1):
            row = training[index]
            neighbor_rows.append(
                {
                    "candidate": candidate.key,
                    "gain": candidate.gain,
                    "neighbor_rank": rank,
                    "neighbor_case_id": str(row["case_id"]),
                    "distance": float(distances[index]),
                    "neighbor_safe": bool(row["safe"]),
                    "neighbor_endpoint_state_ratio": float(row["endpoint_state_ratio"]),
                }
            )
        if candidate_eligible:
            eligible.append(
                (predicted_endpoint, candidate.gain, candidate.key, candidate)
            )
    selected = (
        min(eligible, key=lambda item: item[:3])[-1] if eligible else candidates[0]
    )
    return selected, choice_rows, neighbor_rows


def _crossfit_response_calibration(
    model: torch.nn.Module,
    cases: Sequence[CaseData],
    candidates: Sequence[parent.Candidate],
    *,
    shock_quantile: float,
    probe_calls: int,
    smoke: bool,
    group_by_case: Mapping[str, str] | None = None,
    require_group_amplitude_match: bool = False,
    require_highest_group_abstention: bool = False,
) -> dict[str, Any]:
    case_by_id = {case.case_id: case for case in cases}
    if len(case_by_id) != len(cases):
        raise ValueError("D076 calibration cases are not unique")
    case_ids = tuple(sorted(case_by_id))
    if len(case_ids) <= PROBE_K:
        raise ValueError("D076 calibration requires at least four cases")
    grouped = group_by_case is not None
    if grouped:
        group_id_by_case = {
            str(case_id): str(group_id)
            for case_id, group_id in dict(group_by_case).items()
        }
        if set(group_id_by_case) != set(case_ids):
            raise ValueError("grouped calibration does not cover the exact cases")
        if any(not value for value in group_id_by_case.values()):
            raise ValueError("grouped calibration has an empty group identifier")
    else:
        group_id_by_case = {case_id: case_id for case_id in case_ids}
    members_by_group: dict[str, tuple[str, ...]] = {}
    for group_id in sorted(set(group_id_by_case.values())):
        members_by_group[group_id] = tuple(
            case_id for case_id in case_ids if group_id_by_case[case_id] == group_id
        )
    excluded_by_case = {
        case_id: members_by_group[group_id_by_case[case_id]] for case_id in case_ids
    }
    training_ids_by_case = {
        case_id: tuple(
            value for value in case_ids if value not in set(excluded_by_case[case_id])
        )
        for case_id in case_ids
    }
    if any(len(values) < PROBE_K for values in training_ids_by_case.values()):
        raise ValueError("grouped response fold has fewer than three neighbors")

    group_inventory_rows = []
    amplitude_match_pass = True
    if grouped:
        initial_amplitude_by_case = {
            case_id: _initial_vortex_features(
                case_by_id[case_id],
                case_by_id[case_id].reference_states[0],
            )[0]
            for case_id in case_ids
        }
        for group_id, member_ids in members_by_group.items():
            strengths = []
            positions = []
            for case_id in member_ids:
                case = case_by_id[case_id]
                provenance = case.provenance
                parameters = provenance.get("parameters", {})
                if provenance.get("case_id") != case_id:
                    raise ValueError("grouped case provenance does not bind case_id")
                try:
                    strengths.append(float(parameters["vortex_epsilon"]))
                    positions.append(float(parameters["vortex_y"]))
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        "grouped case provenance lacks vortex parameters"
                    ) from error
            strength_match = bool(
                np.isfinite(strengths).all()
                and np.allclose(strengths, strengths[0], rtol=1.0e-12, atol=0.0)
            )
            amplitudes = [initial_amplitude_by_case[value] for value in member_ids]
            group_amplitude_match = bool(
                np.isfinite(amplitudes).all()
                and np.allclose(amplitudes, amplitudes[0], rtol=1.0e-10, atol=1.0e-12)
            )
            amplitude_match_pass = bool(amplitude_match_pass and group_amplitude_match)
            if not strength_match:
                raise ValueError("one response group mixes physical strengths")
            group_inventory_rows.append(
                {
                    "group_id": group_id,
                    "case_count": len(member_ids),
                    "case_ids_json": json.dumps(member_ids),
                    "vortex_epsilon": strengths[0],
                    "vortex_y_values_json": json.dumps(sorted(positions)),
                    "initial_vortex_amplitude_min": min(amplitudes),
                    "initial_vortex_amplitude_max": max(amplitudes),
                    "initial_vortex_amplitude_match": group_amplitude_match,
                }
            )
        if require_group_amplitude_match and not amplitude_match_pass:
            raise ValueError(
                "paired physical-strength amplitudes differ beyond tolerance"
            )

    fold_inventory_rows = []
    if grouped:
        for case_id in case_ids:
            excluded_ids = excluded_by_case[case_id]
            training_ids = training_ids_by_case[case_id]
            training_groups = tuple(
                sorted({group_id_by_case[value] for value in training_ids})
            )
            query_group_id = group_id_by_case[case_id]
            fold_inventory_rows.append(
                {
                    "query_case_id": case_id,
                    "query_group_id": query_group_id,
                    "excluded_case_ids_json": json.dumps(excluded_ids),
                    "coefficient_training_case_ids_json": json.dumps(training_ids),
                    "coefficient_training_group_ids_json": json.dumps(training_groups),
                    "policy_training_case_ids_json": json.dumps(training_ids),
                    "query_group_absent_from_coefficient_training": (
                        query_group_id not in training_groups
                    ),
                    "query_group_absent_from_policy_training": (
                        all(
                            group_id_by_case[value] != query_group_id
                            for value in training_ids
                        )
                    ),
                }
            )
    coefficients_by_case = {
        case_id: parent._fit_case_coefficients(model, case_by_id[case_id], ranks=(8,))
        for case_id in case_ids
    }
    frozen_by_training_ids: dict[tuple[str, ...], np.ndarray] = {}
    compact: dict[str, dict[str, parent.SelectorRolloutSummary]] = {
        candidate.key: {} for candidate in candidates
    }
    feature_rows = []
    for case_id in case_ids:
        case = case_by_id[case_id]
        baseline = parent._rollout_candidate(
            model,
            case,
            candidates[0],
            bias_sequence=None,
            shock_quantile=shock_quantile,
        )
        compact["zero"][case_id] = parent._selector_rollout_summary(baseline)
        training_ids = training_ids_by_case[case_id]
        if training_ids not in frozen_by_training_ids:
            frozen_by_training_ids[training_ids] = parent.mean_calibration_coefficients(
                np.asarray(
                    [coefficients_by_case[value][8] for value in training_ids],
                    dtype=np.float64,
                )
            )
        frozen = frozen_by_training_ids[training_ids]
        _, bias = parent._bias_sequence(case, frozen, rank=8)
        for candidate in candidates[1:]:
            result = parent._rollout_candidate(
                model,
                case,
                candidate,
                bias_sequence=bias,
                shock_quantile=shock_quantile,
            )
            compact[candidate.key][case_id] = parent._selector_rollout_summary(result)
            feature = {
                "case_id": case_id,
                "candidate": candidate.key,
                "gain": candidate.gain,
                **_response_probe_features(
                    case,
                    baseline,
                    result,
                    probe_calls=probe_calls,
                ),
            }
            if grouped:
                feature.update(
                    {
                        "query_group_id": group_id_by_case[case_id],
                        "excluded_case_ids_json": json.dumps(excluded_by_case[case_id]),
                        "coefficient_training_case_ids_json": json.dumps(training_ids),
                    }
                )
            feature_rows.append(feature)
    (
        static_selected,
        selector_rows,
        selector_summary_rows,
        selector_metric_rows,
    ) = parent.select_candidate(
        candidates,
        case_ids,
        compact,
        family="dynamic_fv",
    )
    selector_by_key = {
        (str(row["case_id"]), str(row["candidate"])): row for row in selector_rows
    }
    if len(selector_by_key) != len(selector_rows):
        raise ValueError("D076 calibration selector rows are not unique")
    response_table = []
    for feature in feature_rows:
        outcome = selector_by_key[(feature["case_id"], feature["candidate"])]
        response_table.append(
            {
                **feature,
                "safe": bool(outcome["eligible"]),
                "complete": bool(outcome["complete"]),
                "endpoint_state_ratio": float(outcome["endpoint_state_ratio"]),
                "residual_rms_ratio": float(outcome["residual_rms_ratio"]),
                "maximum_control_ratio": float(outcome["maximum_control_ratio"]),
            }
        )

    selection_rows = []
    choice_rows = []
    neighbor_rows = []
    for case_id in case_ids:
        query = {
            row["candidate"]: row for row in response_table if row["case_id"] == case_id
        }
        if grouped:
            selected, choices, neighbors = _select_response_gain(
                query,
                response_table,
                candidates,
                excluded_case_ids=excluded_by_case[case_id],
            )
        else:
            selected, choices, neighbors = _select_response_gain(
                query,
                response_table,
                candidates,
                excluded_case_id=case_id,
            )
        for row in choices:
            prefix = {"query_case_id": case_id}
            if grouped:
                prefix.update(
                    {
                        "query_group_id": group_id_by_case[case_id],
                        "excluded_case_ids_json": json.dumps(excluded_by_case[case_id]),
                        "policy_training_case_ids_json": json.dumps(
                            training_ids_by_case[case_id]
                        ),
                    }
                )
            choice_rows.append({**prefix, **row})
        for row in neighbors:
            prefix = {"query_case_id": case_id}
            if grouped:
                neighbor_case_id = str(row["neighbor_case_id"])
                prefix.update(
                    {
                        "query_group_id": group_id_by_case[case_id],
                        "excluded_case_ids_json": json.dumps(excluded_by_case[case_id]),
                        "neighbor_group_id": group_id_by_case[neighbor_case_id],
                    }
                )
            neighbor_rows.append({**prefix, **row})
        if selected.is_zero:
            endpoint_ratio = 1.0
            residual_ratio = 1.0
            maximum_control_ratio = 1.0
            safe = True
        else:
            outcome = selector_by_key[(case_id, selected.key)]
            endpoint_ratio = float(outcome["endpoint_state_ratio"])
            residual_ratio = float(outcome["residual_rms_ratio"])
            maximum_control_ratio = float(outcome["maximum_control_ratio"])
            safe = bool(outcome["eligible"])
        selection = {
            "case_id": case_id,
            "selected_candidate": selected.key,
            "selected_gain": selected.gain,
            "selected_nonzero": not selected.is_zero,
            "actual_safe": safe,
            "actual_endpoint_state_ratio": endpoint_ratio,
            "actual_residual_rms_ratio": residual_ratio,
            "actual_maximum_control_ratio": maximum_control_ratio,
        }
        if grouped:
            selection.update(
                {
                    "query_group_id": group_id_by_case[case_id],
                    "excluded_case_ids_json": json.dumps(excluded_by_case[case_id]),
                }
            )
        selection_rows.append(selection)
    endpoint_values = np.asarray(
        [row["actual_endpoint_state_ratio"] for row in selection_rows],
        dtype=np.float64,
    )
    residual_values = np.asarray(
        [row["actual_residual_rms_ratio"] for row in selection_rows],
        dtype=np.float64,
    )
    safety_pass = bool(all(row["actual_safe"] for row in selection_rows))
    nonzero_count = sum(bool(row["selected_nonzero"]) for row in selection_rows)
    endpoint_pass = bool(np.median(endpoint_values) <= CALIBRATION_ENDPOINT_LIMIT)
    residual_pass = bool(np.median(residual_values) <= CALIBRATION_RESIDUAL_LIMIT)
    nonzero_pass = bool(nonzero_count >= CALIBRATION_REQUIRED_NONZERO)
    fold_exclusion_pass = bool(
        not grouped
        or all(
            row["query_group_absent_from_coefficient_training"]
            and row["query_group_absent_from_policy_training"]
            for row in fold_inventory_rows
        )
    )
    highest_group_id = None
    highest_group_abstention_pass = True
    if grouped:
        highest_group_id = max(
            group_inventory_rows,
            key=lambda row: (float(row["vortex_epsilon"]), str(row["group_id"])),
        )["group_id"]
        highest_group_abstention_pass = bool(
            all(
                not row["selected_nonzero"]
                for row in selection_rows
                if row["query_group_id"] == highest_group_id
            )
        )
    group_contract_pass = bool(
        fold_exclusion_pass
        and (not require_group_amplitude_match or amplitude_match_pass)
        and (not require_highest_group_abstention or highest_group_abstention_pass)
    )
    scientific_pass = bool(
        safety_pass
        and nonzero_pass
        and endpoint_pass
        and residual_pass
        and group_contract_pass
    )
    matrix_complete = bool(all(row["complete"] for row in selector_rows))
    qualification = {
        "passed": scientific_pass,
        "target_loading_authorized": bool(
            scientific_pass or (smoke and matrix_complete and group_contract_pass)
        ),
        "smoke_bypass": bool(
            smoke and not scientific_pass and matrix_complete and group_contract_pass
        ),
        "all_selected_safe": safety_pass,
        "nonzero_case_count": nonzero_count,
        "required_nonzero_case_count": CALIBRATION_REQUIRED_NONZERO,
        "median_endpoint_state_ratio": float(np.median(endpoint_values)),
        "endpoint_limit": CALIBRATION_ENDPOINT_LIMIT,
        "median_residual_rms_ratio": float(np.median(residual_values)),
        "residual_limit": CALIBRATION_RESIDUAL_LIMIT,
        "candidate_matrix_complete": matrix_complete,
    }
    group_contract_checks = {
        "applicable": grouped,
        "passed": group_contract_pass,
        "group_count": len(members_by_group) if grouped else None,
        "group_amplitude_match_pass": amplitude_match_pass if grouped else None,
        "fold_exclusion_pass": fold_exclusion_pass if grouped else None,
        "highest_group_id": highest_group_id,
        "highest_group_abstention_required": (
            require_highest_group_abstention if grouped else None
        ),
        "highest_group_abstention_pass": (
            highest_group_abstention_pass if grouped else None
        ),
    }
    if grouped:
        qualification["group_contract_pass"] = group_contract_pass
        qualification["highest_group_id"] = highest_group_id
        qualification["highest_group_abstention_pass"] = highest_group_abstention_pass
    frozen_coefficients = parent.mean_calibration_coefficients(
        np.asarray(
            [coefficients_by_case[value][8] for value in case_ids],
            dtype=np.float64,
        )
    )
    return {
        "static_selected": static_selected,
        "coefficients_by_case": coefficients_by_case,
        "frozen_coefficients": frozen_coefficients,
        "selector_rows": selector_rows,
        "selector_summary_rows": selector_summary_rows,
        "selector_metric_rows": selector_metric_rows,
        "response_table_rows": response_table,
        "controller_selection_rows": selection_rows,
        "controller_choice_rows": choice_rows,
        "controller_neighbor_rows": neighbor_rows,
        "group_inventory_rows": group_inventory_rows,
        "fold_inventory_rows": fold_inventory_rows,
        "group_contract_checks": group_contract_checks,
        "qualification": qualification,
    }


def _comparison_row(
    case_id: str,
    numerator: parent.RolloutResult,
    denominator: parent.RolloutResult,
    *,
    comparison: str,
) -> dict[str, Any]:
    if not numerator.complete or not denominator.complete:
        raise ValueError(f"{comparison} requires complete equal-horizon rollouts")
    numerator_controls = numerator.summary.get("controls", {})
    denominator_controls = denominator.summary.get("controls", {})
    required_controls = set(parent.expected_control_keys("dynamic_fv"))
    if (
        set(numerator_controls) != required_controls
        or set(denominator_controls) != required_controls
    ):
        raise ValueError("D076 comparison control inventory changed")
    endpoint_ratio = parent._required_ratio(
        numerator.summary.get("final_state_error"),
        denominator.summary.get("final_state_error"),
    )
    residual_ratio = parent._required_ratio(
        numerator.summary.get("residual_rms"),
        denominator.summary.get("residual_rms"),
    )
    control_ratios = {
        name: parent._required_ratio(
            numerator_controls[name], denominator_controls[name]
        )
        for name in sorted(required_controls)
    }
    return {
        "case_id": case_id,
        "comparison": comparison,
        "numerator_candidate": numerator.candidate.key,
        "denominator_candidate": denominator.candidate.key,
        "endpoint_state_numerator": numerator.summary["final_state_error"],
        "endpoint_state_denominator": denominator.summary["final_state_error"],
        "endpoint_state_ratio": endpoint_ratio,
        "residual_rms_numerator": numerator.summary["residual_rms"],
        "residual_rms_denominator": denominator.summary["residual_rms"],
        "residual_rms_ratio": residual_ratio,
        "maximum_control_ratio": max(control_ratios.values()),
        "control_ratios_json": json.dumps(control_ratios, sort_keys=True),
    }


def _promotion(
    comparison_rows: Sequence[Mapping[str, Any]],
    selection_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    baseline = [
        row for row in comparison_rows if row["comparison"] == "selected_vs_zero"
    ]
    static = [
        row for row in comparison_rows if row["comparison"] == "selected_vs_static"
    ]
    if len(baseline) != len(selection_rows) or len(static) != len(selection_rows):
        raise ValueError("D076 promotion comparison inventory is incomplete")
    endpoint = np.asarray(
        [float(row["endpoint_state_ratio"]) for row in baseline], dtype=np.float64
    )
    residual = np.asarray(
        [float(row["residual_rms_ratio"]) for row in baseline], dtype=np.float64
    )
    control = np.asarray(
        [float(row["maximum_control_ratio"]) for row in baseline], dtype=np.float64
    )
    static_endpoint = np.asarray(
        [float(row["endpoint_state_ratio"]) for row in static], dtype=np.float64
    )
    static_residual = np.asarray(
        [float(row["residual_rms_ratio"]) for row in static], dtype=np.float64
    )
    nonzero_count = sum(float(row["selected_gain"]) > 0.0 for row in selection_rows)
    gates = {
        "all_endpoint_no_harm": bool(np.all(endpoint <= 1.0)),
        "all_residual_no_harm": bool(np.all(residual <= 1.0)),
        "all_controls_no_harm": bool(np.all(control <= parent.CONTROL_NO_HARM_LIMIT)),
        "nonzero_case_count": nonzero_count,
        "nonzero_case_count_pass": bool(nonzero_count >= 4),
        "median_endpoint_state_ratio": float(np.median(endpoint)),
        "median_endpoint_pass": bool(np.median(endpoint) <= EVALUATION_ENDPOINT_LIMIT),
        "median_residual_rms_ratio": float(np.median(residual)),
        "selected_static_median_endpoint_ratio": float(np.median(static_endpoint)),
        "selected_static_endpoint_pass": bool(
            np.median(static_endpoint) <= STATIC_ENDPOINT_LIMIT
        ),
        "selected_static_median_residual_ratio": float(np.median(static_residual)),
        "selected_static_residual_pass": bool(
            np.median(static_residual) <= STATIC_RESIDUAL_LIMIT
        ),
        "maximum_control_ratio": float(np.max(control)),
    }
    gates["passed"] = bool(
        gates["all_endpoint_no_harm"]
        and gates["all_residual_no_harm"]
        and gates["all_controls_no_harm"]
        and gates["nonzero_case_count_pass"]
        and gates["median_endpoint_pass"]
        and gates["selected_static_endpoint_pass"]
        and gates["selected_static_residual_pass"]
    )
    gates["adaptive_open_population"] = True
    return gates


def _probe_replay(
    probe: ProbeRollout,
    rollout: parent.RolloutResult,
    *,
    probe_calls: int,
) -> dict[str, Any]:
    if not probe.complete or not rollout.complete:
        raise ValueError("D076 prefix replay requires complete arms")
    left = np.asarray(probe.states[: probe_calls + 1], dtype=np.float64)
    right = np.asarray(rollout.states[: probe_calls + 1], dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("D076 prefix replay shape changed")
    maximum_absolute = float(np.max(np.abs(left - right)))
    denominator = max(float(np.linalg.norm(left)), FEATURE_SCALE_FLOOR)
    relative_l2 = float(np.linalg.norm(left - right) / denominator)
    return {
        "maximum_absolute": maximum_absolute,
        "relative_l2": relative_l2,
        "absolute_limit": PROBE_ABSOLUTE_REPLAY_LIMIT,
        "relative_limit": PROBE_RELATIVE_REPLAY_LIMIT,
        "passed": bool(
            maximum_absolute <= PROBE_ABSOLUTE_REPLAY_LIMIT
            and relative_l2 <= PROBE_RELATIVE_REPLAY_LIMIT
        ),
    }


def _save_visual_payload(
    path: Path,
    case: CaseData,
    baseline: parent.RolloutResult,
    static: parent.RolloutResult,
    selected: parent.RolloutResult,
    *,
    schema: str = SCHEMA,
) -> dict[str, Any]:
    record = parent._save_visual_payload(path, case, baseline, selected)
    with np.load(path, allow_pickle=False) as payload:
        values = {key: np.array(payload[key], copy=True) for key in payload.files}
    static_errors = static.states[1:] - case.reference_states[1:]
    values.update(
        {
            "schema": np.asarray(schema),
            "static_candidate": np.asarray(static.candidate.key),
            "static_gain": np.asarray(static.candidate.gain, dtype=np.float64),
            "static_defect": static.defects.astype(np.float32),
            "static_cumulative_error": static_errors.astype(np.float32),
            "selected_minus_zero_state": (
                selected.states[1:] - baseline.states[1:]
            ).astype(np.float32),
            "selected_minus_static_state": (
                selected.states[1:] - static.states[1:]
            ).astype(np.float32),
        }
    )
    np.savez_compressed(path, **values)
    return {
        **record,
        "static_candidate": static.candidate.key,
        "selected_gain": selected.candidate.gain,
    }


def _maximum_absolute(
    rows: Sequence[Mapping[str, Any]], field: str, *, empty: float = float("inf")
) -> float:
    if not rows:
        return empty
    values = [row.get(field) for row in rows]
    if any(value is None or not np.isfinite(value) for value in values):
        return float("inf")
    return max(abs(float(value)) for value in values)


def _evaluation_inventory_checks(
    outputs: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    evaluation_ids: Sequence[str],
    candidates: Sequence[parent.Candidate],
    selection_rows: Sequence[Mapping[str, Any]],
    rollout_calls: int,
    visual_case_ids: Sequence[str],
) -> dict[str, bool]:
    """Require exact case-first diagnostic inventories before interpretation."""

    case_ids = tuple(sorted(str(value) for value in evaluation_ids))
    calls = tuple(range(1, rollout_calls + 1))
    arms = ("baseline", "static", "selected")
    components = parent.COMPONENTS
    selection_by_case = {
        str(row["case_id"]): bool(row["selected_nonzero"]) for row in selection_rows
    }
    selection_inventory_pass = bool(
        len(selection_by_case) == len(selection_rows)
        and set(selection_by_case) == set(case_ids)
    )
    positive_keys = tuple(
        candidate.key for candidate in candidates if not candidate.is_zero
    )
    expected_probe_keys = {
        (case_id, candidate) for case_id in case_ids for candidate in positive_keys
    }
    expected_neighbor_keys = {
        (case_id, candidate, rank)
        for case_id in case_ids
        for candidate in positive_keys
        for rank in range(1, PROBE_K + 1)
    }
    expected_arm_call_keys = {
        (case_id, arm, call) for case_id in case_ids for arm in arms for call in calls
    }
    expected_integral_keys = {
        (*key, component) for key in expected_arm_call_keys for component in components
    }
    sequence_keys = set()
    for case_id in case_ids:
        sequence_keys.update(
            {
                (case_id, "baseline", "corrected_defect_on_own_recurrent_inputs"),
                (case_id, "static", "corrected_defect_on_own_recurrent_inputs"),
                (case_id, "static", "base_defect_on_same_corrected_inputs"),
                (case_id, "selected", "corrected_defect_on_own_recurrent_inputs"),
            }
        )
        if selection_by_case.get(case_id):
            sequence_keys.add(
                (case_id, "selected", "base_defect_on_same_corrected_inputs")
            )
    expected_sequence_time_keys = {
        (*key, call) for key in sequence_keys for call in calls
    }
    expected_lag_keys = {
        (*key, lag)
        for key in sequence_keys
        for lag in range(1, min(10, rollout_calls - 1) + 1)
    }
    expected_pod_keys = {
        (*key, centering)
        for key in sequence_keys
        for centering in ("uncentered", "centered")
    }
    base_budget_fields = (
        "base_defect_same_corrected_input",
        "correction",
        "corrected_defect",
        "cumulative_correction",
        "state_error",
    )
    modal_budget_fields = (
        "correction_constant_mode",
        "correction_nonconstant_modes",
    )
    expected_budget_keys = {
        (case_id, arm, call, field, component)
        for case_id in case_ids
        for arm in arms
        for call in calls
        for field in base_budget_fields
        for component in components
    }
    expected_budget_keys.update(
        {
            (case_id, "static", call, field, component)
            for case_id in case_ids
            for call in calls
            for field in modal_budget_fields
            for component in components
        }
    )
    expected_budget_keys.update(
        {
            (case_id, "selected", call, field, component)
            for case_id in case_ids
            if selection_by_case.get(case_id)
            for call in calls
            for field in modal_budget_fields
            for component in components
        }
    )
    expected_projection_keys = {
        (case_id, "static", call) for case_id in case_ids for call in calls
    }
    expected_projection_keys.update(
        {
            (case_id, "selected", call)
            for case_id in case_ids
            if selection_by_case.get(case_id)
            for call in calls
        }
    )
    expected_projection_component_keys = {
        (*key, component)
        for key in expected_projection_keys
        for component in components
    }
    inventory = {
        "evaluation_selection_inventory_pass": selection_inventory_pass,
        "evaluation_probe_feature_inventory_pass": parent._row_inventory_exact(
            outputs["evaluation_probe_features.csv"],
            expected_probe_keys,
            ("case_id", "candidate"),
        ),
        "evaluation_choice_inventory_pass": parent._row_inventory_exact(
            outputs["evaluation_controller_choices.csv"],
            expected_probe_keys,
            ("case_id", "candidate"),
        ),
        "evaluation_neighbor_inventory_pass": parent._row_inventory_exact(
            outputs["evaluation_controller_neighbors.csv"],
            expected_neighbor_keys,
            ("case_id", "candidate", "neighbor_rank"),
        ),
        "evaluation_case_summary_inventory_pass": parent._row_inventory_exact(
            outputs["evaluation_case_summary.csv"],
            {(case_id, arm) for case_id in case_ids for arm in arms},
            ("case_id", "arm"),
        ),
        "evaluation_comparison_inventory_pass": parent._row_inventory_exact(
            outputs["evaluation_comparisons.csv"],
            {
                (case_id, comparison)
                for case_id in case_ids
                for comparison in (
                    "selected_vs_zero",
                    "selected_vs_static",
                    "static_vs_zero",
                )
            },
            ("case_id", "comparison"),
        ),
        "evaluation_replay_inventory_pass": parent._row_inventory_exact(
            outputs["evaluation_probe_replay.csv"],
            {(case_id,) for case_id in case_ids},
            ("case_id",),
        ),
        "evaluation_call_inventory_pass": parent._row_inventory_exact(
            outputs["evaluation_call_metrics.csv"],
            expected_arm_call_keys,
            ("case_id", "arm", "call"),
        ),
        "completion_inventory_pass": (
            parent._row_inventory_exact(
                outputs["completion.csv"],
                expected_arm_call_keys,
                ("case_id", "arm", "call"),
            )
            and all(bool(row["accepted"]) for row in outputs["completion.csv"])
        ),
        "correction_integral_inventory_pass": parent._row_inventory_exact(
            outputs["correction_integral_audit.csv"],
            expected_integral_keys,
            ("case_id", "arm", "call", "component"),
        ),
        "sequence_inventory_pass": parent._row_inventory_exact(
            outputs["sequence_summaries.csv"],
            sequence_keys,
            ("case_id", "arm", "defect_kind"),
        ),
        "sequence_time_inventory_pass": parent._row_inventory_exact(
            outputs["sequence_time_metrics.csv"],
            expected_sequence_time_keys,
            ("case_id", "arm", "defect_kind", "step"),
        ),
        "lag_inventory_pass": parent._row_inventory_exact(
            outputs["lag_correlations.csv"],
            expected_lag_keys,
            ("case_id", "arm", "defect_kind", "lag"),
        ),
        "pod_inventory_pass": parent._row_inventory_exact(
            outputs["pod_summaries.csv"],
            expected_pod_keys,
            ("case_id", "arm", "defect_kind", "centering"),
        ),
        "budget_inventory_pass": parent._row_inventory_exact(
            outputs["signed_component_budgets.csv"],
            expected_budget_keys,
            ("case_id", "arm", "call", "field", "component"),
        ),
        "projection_inventory_pass": parent._row_inventory_exact(
            outputs["projection_metrics.csv"],
            expected_projection_keys,
            ("case_id", "arm", "call"),
        ),
        "projection_component_inventory_pass": parent._row_inventory_exact(
            outputs["projection_component_metrics.csv"],
            expected_projection_component_keys,
            ("case_id", "arm", "call", "component"),
        ),
        "projection_status_inventory_pass": parent._row_inventory_exact(
            outputs["projection_status.csv"],
            {(case_id,) for case_id in case_ids},
            ("case_id",),
        ),
        "visual_inventory_pass": parent._row_inventory_exact(
            outputs["visual_payload_inventory.csv"],
            {(str(case_id),) for case_id in visual_case_ids},
            ("case_id",),
        ),
    }
    inventory["row_inventories_pass"] = bool(all(inventory.values()))
    return inventory


def _write_summary(
    args: argparse.Namespace,
    *,
    spec: ResponseExperimentSpec,
    started: float,
    device: torch.device,
    provenance: dict[str, Any],
    calibration_ids: Sequence[str],
    evaluation_ids: Sequence[str],
    selector_record: Mapping[str, Any],
    selector_sha: str,
    qualification: Mapping[str, Any],
    phase_rows: Sequence[Mapping[str, Any]],
    calibration_hook: Mapping[str, Any],
    evaluation_hook: Mapping[str, Any] | None,
    reference_binding_pass: bool,
    core_contract_pass: bool,
    promotion: Mapping[str, Any],
    checks: Mapping[str, Any],
    row_counts: Mapping[str, int],
) -> dict[str, Any]:
    loaded_source_hashes = parent._loaded_project_source_hashes()
    for source_path in (Path(__file__), *spec.extra_source_paths):
        resolved = source_path.resolve()
        relative = str(resolved.relative_to(ROOT)).replace("\\", "/")
        loaded_source_hashes[relative] = sha256_file(resolved)
    provenance["loaded_project_source_sha256"] = loaded_source_hashes
    output_hashes = {
        str(path.relative_to(args.output_dir)).replace("\\", "/"): sha256_file(path)
        for path in sorted(args.output_dir.rglob("*"))
        if path.is_file() and path.name != "summary.json"
    }
    evaluated = bool(evaluation_hook is not None)
    if args.smoke:
        status = "smoke_complete" if core_contract_pass else "smoke_failed"
    elif not qualification["passed"]:
        status = (
            "calibration_not_qualified"
            if core_contract_pass and not evaluated
            else "failed_contract"
        )
    else:
        status = "complete" if core_contract_pass else "failed_contract"
    summary = {
        "schema": spec.schema,
        "experiment_id": spec.experiment_id,
        "experiment_contract": spec.experiment_contract,
        "status": status,
        "contract_checks_passed": core_contract_pass,
        "scientific_interpretation_allowed": bool(
            core_contract_pass and not args.smoke
        ),
        "family": "dynamic_fv",
        "args": jsonable_args(args),
        "population": {
            "split": "validation",
            "calibration_case_ids": list(calibration_ids),
            "evaluation_case_ids": list(evaluation_ids) if evaluated else [],
            "conditional_evaluation_case_ids": list(evaluation_ids),
            "evaluation_targets_loaded": evaluated,
            "target_loading_authorized": bool(
                qualification["target_loading_authorized"]
            ),
            "sealed_populations_accessed": False,
            "adaptive_reuse_of_open_validation": True,
            "adaptive_reuse_of_d075_evaluation": True,
        },
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "boundary_policy": "model_all_nodes raw recurrence; base policy unchanged",
        "selector": selector_record,
        "selector_sha256": selector_sha,
        "calibration_qualification": dict(qualification),
        "promotion": dict(promotion),
        "checks": {
            **checks,
            "reference_binding_pass": reference_binding_pass,
            "core_contract_pass": core_contract_pass,
        },
        "claim_boundary": {
            "method": spec.method_claim,
            "confirmation": "adaptive open-validation development evidence only",
            "dynamic": "audited physical volumes; base PCNO is not conservative",
            "bump": "not loaded; separate immutable D041 replay contract required",
            "resolution": "native 250x100 only",
            "boundary": "representation fixed; no boundary-condition change",
        },
        "provenance": provenance,
        "calibration_hook_equivalence": dict(calibration_hook),
        "evaluation_hook_equivalence": (
            None if evaluation_hook is None else dict(evaluation_hook)
        ),
        "phase_order": list(phase_rows),
        "runtime": {
            **runtime_environment(device),
            "deterministic_algorithms_requested": spec.deterministic_algorithms,
            "deterministic_algorithms_enabled": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "git": git_state(),
        "row_counts": dict(row_counts),
        "elapsed_seconds": perf_counter() - started,
        "output_hashes": output_hashes,
    }
    write_json(args.output_dir / "summary.json", summary)
    return summary


def _initial_evaluation_checks(
    spec: ResponseExperimentSpec,
) -> dict[str, Any]:
    if spec.null_not_run_evaluation:
        return {
            "evaluation_ran": False,
            "evaluation_choice_bundle_revalidated": None,
            "evaluation_inventory_pass": None,
            "all_evaluation_complete": None,
            "probe_prefix_replay_pass": None,
            "closure_pass": None,
            "maximum_probe_prefix_absolute": None,
            "maximum_probe_prefix_relative_l2": None,
            "maximum_recurrence_closure_rms": None,
            "maximum_growth_closure_absolute": None,
            "maximum_same_input_pointwise_closure": None,
        }
    return {
        "evaluation_choice_bundle_revalidated": False,
        "evaluation_inventory_pass": True,
        "all_evaluation_complete": True,
        "probe_prefix_replay_pass": True,
        "closure_pass": True,
        "maximum_probe_prefix_absolute": 0.0,
        "maximum_probe_prefix_relative_l2": 0.0,
        "maximum_recurrence_closure_rms": 0.0,
        "maximum_growth_closure_absolute": 0.0,
        "maximum_same_input_pointwise_closure": 0.0,
    }


def run(
    args: argparse.Namespace,
    *,
    spec: ResponseExperimentSpec = D076_SPEC,
) -> dict[str, Any]:
    started = perf_counter()
    if spec.deterministic_algorithms:
        if torch.cuda.is_initialized():
            raise RuntimeError(
                "deterministic cuBLAS workspace must be set before CUDA initialization"
            )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = (
            DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG
        )
    torch.use_deterministic_algorithms(spec.deterministic_algorithms)
    if spec.deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
    runtime_contract_pass = bool(
        torch.are_deterministic_algorithms_enabled()
        == spec.deterministic_algorithms
        and (
            not spec.deterministic_algorithms
            or not torch.backends.cudnn.benchmark
        )
        and (
            not spec.deterministic_algorithms
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG")
            == DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG
        )
    )
    device = select_device(args.device)
    split = json.loads(args.split_json.read_text(encoding="utf-8"))
    validation_ids = tuple(str(value) for value in split.get("val_keys", ()))
    expected_validation = set(parent.DYNAMIC_CALIBRATION_CASES) | set(
        parent.DYNAMIC_EVALUATION_CASES
    )
    if set(validation_ids) != expected_validation:
        raise ValueError(
            f"{spec.experiment_id} open-validation split inventory changed"
        )
    calibration_ids = (
        parent.DYNAMIC_CALIBRATION_CASES[: spec.smoke_calibration_case_count]
        if args.smoke
        else parent.DYNAMIC_CALIBRATION_CASES
    )
    evaluation_ids = (
        parent.DYNAMIC_EVALUATION_CASES[:1]
        if args.smoke
        else parent.DYNAMIC_EVALUATION_CASES
    )
    candidates = _candidate_inventory(smoke=args.smoke)
    store = PCNOEuler2DShardStore(args.data_dir, max_cached_trajectories=2)
    checkpoint = load_resolution_checkpoint(args.checkpoint)
    model, _ = build_resolution_checkpoint_model(checkpoint, device)
    model.eval()
    provenance = parent._verify_common_provenance(args, checkpoint, store)
    if git_head() != args.expected_source_base_git_head.lower():
        raise ValueError(
            f"active repository HEAD differs from the frozen {spec.experiment_id} base"
        )

    phase_rows: list[dict[str, Any]] = []
    case_contract_rows = []
    calibration_reference_checks: list[dict[str, Any]] = []
    evaluation_reference_checks: list[dict[str, Any]] = []
    evaluation_hook = None
    evaluation_choice_bundle = None
    evaluation_outputs: dict[str, list[dict[str, Any]]] = {}
    evaluation_checks = _initial_evaluation_checks(spec)
    try:
        calibration_cases, calibration_reference_checks = parent._load_cases(
            args,
            checkpoint,
            store,
            calibration_ids,
            device=device,
        )
        case_contract_rows.extend(
            parent._validate_case_contract(case) for case in calibration_cases
        )
        phase_rows.append(
            {
                "phase": "calibration_targets_loaded",
                "case_count": len(calibration_cases),
                "evaluation_targets_loaded": False,
            }
        )
        calibration_hook = parent._run_hook_equivalence(
            model, calibration_cases[0], device=device
        )
        group_by_case = (
            None
            if spec.group_builder is None
            else spec.group_builder(calibration_cases)
        )
        calibration = _crossfit_response_calibration(
            model,
            calibration_cases,
            candidates,
            shock_quantile=args.shock_quantile,
            probe_calls=args.probe_calls,
            smoke=args.smoke,
            group_by_case=group_by_case,
            require_group_amplitude_match=spec.require_group_amplitude_match,
            require_highest_group_abstention=(spec.require_highest_group_abstention),
        )
        qualification = calibration["qualification"]
        frozen_coefficients = calibration["frozen_coefficients"]
        frozen_policy = _frozen_policy_record(
            calibration["response_table_rows"],
            candidates,
            schema=spec.schema,
        )

        args.output_dir.mkdir(parents=True)
        (args.output_dir / "visual_payloads").mkdir()
        calibration_coefficients_path = (
            args.output_dir / "calibration_coefficients_by_case.npz"
        )
        np.savez_compressed(
            calibration_coefficients_path,
            schema=np.asarray(spec.schema),
            **{
                f"case__{case_id}__rank__8": by_rank[8]
                for case_id, by_rank in sorted(
                    calibration["coefficients_by_case"].items()
                )
            },
        )
        frozen_coefficients_path = args.output_dir / "frozen_rank8_coefficients.npz"
        np.savez_compressed(
            frozen_coefficients_path,
            schema=np.asarray(spec.schema),
            rank=np.asarray(8, dtype=np.int64),
            coefficients=np.asarray(frozen_coefficients, dtype=np.float64),
        )
        calibration_files = {
            "selector_rows.csv": calibration["selector_rows"],
            "selector_summary.csv": calibration["selector_summary_rows"],
            "selector_metrics.csv": calibration["selector_metric_rows"],
            "calibration_probe_features.csv": calibration["response_table_rows"],
            "calibration_controller_selections.csv": calibration[
                "controller_selection_rows"
            ],
            "calibration_controller_choices.csv": calibration["controller_choice_rows"],
            "calibration_controller_neighbors.csv": calibration[
                "controller_neighbor_rows"
            ],
            "calibration_reference_checks.csv": calibration_reference_checks,
        }
        if spec.group_builder is not None:
            calibration_files.update(
                {
                    "calibration_group_inventory.csv": calibration[
                        "group_inventory_rows"
                    ],
                    "calibration_group_fold_inventory.csv": calibration[
                        "fold_inventory_rows"
                    ],
                }
            )
        for name, rows in calibration_files.items():
            write_csv_with_paths(args.output_dir / name, rows)
        frozen_policy_path = args.output_dir / "frozen_response_policy.json"
        write_json(frozen_policy_path, frozen_policy)
        selector_artifacts = {
            name: sha256_file(args.output_dir / name) for name in calibration_files
        }
        selector_artifacts.update(
            {
                calibration_coefficients_path.name: sha256_file(
                    calibration_coefficients_path
                ),
                frozen_coefficients_path.name: sha256_file(frozen_coefficients_path),
                frozen_policy_path.name: sha256_file(frozen_policy_path),
            }
        )
        selector_record = {
            "schema": spec.schema,
            "experiment_contract": spec.experiment_contract,
            "smoke": args.smoke,
            "candidate_inventory": [candidate.__dict__ for candidate in candidates],
            "probe_calls": args.probe_calls,
            "response_policy": frozen_policy,
            "calibration_qualification": qualification,
            "static_complete_case_selection": calibration["static_selected"].__dict__,
            "evaluation_targets_loaded_before_freeze": False,
            "artifact_sha256": selector_artifacts,
        }
        if spec.group_builder is not None:
            selector_record["crossfit_grouping"] = {
                "contract": spec.group_contract,
                "coefficient_and_policy_exclusion": True,
                "checks": calibration["group_contract_checks"],
            }
        selector_path = args.output_dir / "selector.json"
        write_json(selector_path, selector_record)
        selector_sha = sha256_file(selector_path)
        phase_rows.append(
            {
                "phase": "selector_frozen",
                "case_count": len(calibration_ids),
                "evaluation_targets_loaded": False,
                "selector_sha256": selector_sha,
                "calibration_qualified": qualification["passed"],
                "target_loading_authorized": qualification["target_loading_authorized"],
            }
        )
        parent._assert_selector_frozen_before_evaluation(
            selector_path,
            selector_sha,
            selector_artifacts,
            phase_rows,
        )
        selector_bundle_revalidated = True

        if qualification["target_loading_authorized"]:
            evaluation_runner = spec.evaluation_runner or _conditional_evaluation
            evaluation = evaluation_runner(
                args,
                spec=spec,
                model=model,
                checkpoint=checkpoint,
                store=store,
                device=device,
                evaluation_ids=evaluation_ids,
                candidates=candidates,
                frozen_coefficients=frozen_coefficients,
                frozen_policy=frozen_policy,
                response_table_rows=calibration["response_table_rows"],
                selector_sha=selector_sha,
                phase_rows=phase_rows,
            )
            evaluation_hook = evaluation["hook"]
            evaluation_reference_checks = evaluation["reference_checks"]
            case_contract_rows.extend(evaluation["case_contract_rows"])
            evaluation_outputs = evaluation["output_rows"]
            evaluation_checks = evaluation["checks"]
            if spec.null_not_run_evaluation:
                evaluation_checks = {"evaluation_ran": True, **evaluation_checks}
            evaluation_choice_bundle = evaluation["choice_bundle"]
            promotion = evaluation["promotion"]
        else:
            promotion = {
                "passed": False,
                "status": "not_evaluated_calibration_not_qualified",
                "adaptive_open_population": True,
            }
    finally:
        store.close()

    output_rows = {
        **evaluation_outputs,
        "phase_order.csv": phase_rows,
        "case_contracts.csv": case_contract_rows,
    }
    for name, rows in output_rows.items():
        write_csv_with_paths(args.output_dir / name, rows)
    if evaluation_choice_bundle is not None and (
        sha256_file(evaluation_choice_bundle["record_path"])
        != evaluation_choice_bundle["record_sha"]
        or any(
            sha256_file(args.output_dir / name) != digest
            for name, digest in evaluation_choice_bundle["artifact_sha256"].items()
        )
    ):
        raise ValueError(
            f"{spec.experiment_id} evaluation choice bundle changed after final write"
        )
    parent._assert_selector_bundle_unchanged(
        selector_path,
        selector_sha,
        selector_artifacts,
    )
    selector_bundle_revalidated = True

    evaluated = evaluation_hook is not None
    reference_binding_pass = bool(
        len(calibration_reference_checks) == len(calibration_ids)
        and (not evaluated or len(evaluation_reference_checks) == len(evaluation_ids))
    )
    calibration_inventory_pass = bool(
        len(calibration["response_table_rows"])
        == len(calibration_ids) * (len(candidates) - 1)
        and len(calibration["controller_selection_rows"]) == len(calibration_ids)
    )
    group_inventory_pass = bool(
        spec.group_builder is None
        or (
            calibration["group_contract_checks"]["passed"]
            and len(calibration["fold_inventory_rows"]) == len(calibration_ids)
            and len(calibration["group_inventory_rows"]) * 2 == len(calibration_ids)
        )
    )
    core_contract_pass = bool(
        runtime_contract_pass
        and calibration_hook["passed"]
        and (evaluation_hook is None or evaluation_hook["passed"])
        and reference_binding_pass
        and selector_bundle_revalidated
        and calibration_inventory_pass
        and group_inventory_pass
        and (
            not evaluated
            or all(
                bool(evaluation_checks[name])
                for name in (
                    "evaluation_inventory_pass",
                    "all_evaluation_complete",
                    "probe_prefix_replay_pass",
                    "closure_pass",
                )
            )
        )
        and (not evaluated or evaluation_checks["evaluation_choice_bundle_revalidated"])
    )
    checks = {
        "runtime_determinism_contract_pass": runtime_contract_pass,
        "calibration_hook_equivalence_pass": calibration_hook["passed"],
        "evaluation_hook_equivalence_pass": (
            None if evaluation_hook is None else evaluation_hook["passed"]
        ),
        "selector_bundle_revalidated": selector_bundle_revalidated,
        "calibration_inventory_pass": calibration_inventory_pass,
        **evaluation_checks,
    }
    if spec.group_builder is not None:
        checks.update(
            {
                "group_inventory_pass": group_inventory_pass,
                "group_contract_checks": calibration["group_contract_checks"],
            }
        )
    row_counts = {
        **{name: len(rows) for name, rows in calibration_files.items()},
        **{name: len(rows) for name, rows in output_rows.items()},
    }
    return _write_summary(
        args,
        spec=spec,
        started=started,
        device=device,
        provenance=provenance,
        calibration_ids=calibration_ids,
        evaluation_ids=evaluation_ids,
        selector_record=selector_record,
        selector_sha=selector_sha,
        qualification=qualification,
        phase_rows=phase_rows,
        calibration_hook=calibration_hook,
        evaluation_hook=evaluation_hook,
        reference_binding_pass=reference_binding_pass,
        core_contract_pass=core_contract_pass,
        promotion=promotion,
        checks=checks,
        row_counts=row_counts,
    )


def _freeze_evaluation_probe_choices(
    args: argparse.Namespace,
    *,
    spec: ResponseExperimentSpec,
    model: torch.nn.Module,
    checkpoint: Mapping[str, Any],
    store: PCNOEuler2DShardStore,
    device: torch.device,
    evaluation_ids: Sequence[str],
    candidates: Sequence[parent.Candidate],
    frozen_coefficients: np.ndarray,
    frozen_policy: Mapping[str, Any],
    response_table_rows: Sequence[Mapping[str, Any]],
    selector_sha: str,
    phase_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    cases, reference_checks = parent._load_cases(
        args,
        checkpoint,
        store,
        evaluation_ids,
        device=device,
    )
    case_contract_rows = [parent._validate_case_contract(case) for case in cases]
    phase_rows.append(
        {
            "phase": "evaluation_targets_loaded",
            "case_count": len(cases),
            "evaluation_targets_loaded": True,
            "frozen_selector_sha256": selector_sha,
        }
    )
    hook = parent._run_hook_equivalence(model, cases[0], device=device)
    feature_rows = []
    choice_rows = []
    neighbor_rows = []
    selection_rows = []
    selected_by_case: dict[str, parent.Candidate] = {}
    probes_by_case: dict[str, dict[str, ProbeRollout]] = {}
    bias_by_case: dict[str, np.ndarray] = {}
    for case in cases:
        _, bias = parent._bias_sequence(case, frozen_coefficients, rank=8)
        bias_by_case[case.case_id] = bias
        baseline_probe = _rollout_probe(
            model,
            case,
            candidates[0],
            initial_state=case.reference_states[0],
            bias_sequence=None,
            probe_calls=args.probe_calls,
        )
        probes = {"zero": baseline_probe}
        query = {}
        for candidate in candidates[1:]:
            probe = _rollout_probe(
                model,
                case,
                candidate,
                initial_state=case.reference_states[0],
                bias_sequence=bias,
                probe_calls=args.probe_calls,
            )
            probes[candidate.key] = probe
            features = {
                "case_id": case.case_id,
                "candidate": candidate.key,
                "gain": candidate.gain,
                **_response_probe_features(
                    case,
                    baseline_probe,
                    probe,
                    probe_calls=args.probe_calls,
                ),
            }
            query[candidate.key] = features
            feature_rows.append(features)
        selected, choices, neighbors = _select_response_gain(
            query,
            response_table_rows,
            candidates,
            frozen_policy=frozen_policy,
        )
        selected_by_case[case.case_id] = selected
        probes_by_case[case.case_id] = probes
        selected_choice = next(
            (row for row in choices if row["candidate"] == selected.key),
            None,
        )
        selection_rows.append(
            {
                "case_id": case.case_id,
                "selected_candidate": selected.key,
                "selected_gain": selected.gain,
                "selected_nonzero": not selected.is_zero,
                "selection_reason": (
                    "eligible_minimum_predicted_endpoint"
                    if selected_choice is not None
                    else "zero_fallback"
                ),
                "predicted_endpoint_ratio": (
                    None
                    if selected_choice is None
                    else selected_choice["predicted_endpoint_ratio"]
                ),
            }
        )
        choice_rows.extend({"case_id": case.case_id, **row} for row in choices)
        neighbor_rows.extend({"case_id": case.case_id, **row} for row in neighbors)
    output_rows = {
        "evaluation_probe_features.csv": feature_rows,
        "evaluation_controller_choices.csv": choice_rows,
        "evaluation_controller_neighbors.csv": neighbor_rows,
        "evaluation_gain_selections.csv": selection_rows,
    }
    for name, rows in output_rows.items():
        write_csv_with_paths(args.output_dir / name, rows)
    output_hashes = {name: sha256_file(args.output_dir / name) for name in output_rows}
    record = {
        "schema": spec.schema,
        "selector_sha256": selector_sha,
        "artifact_sha256": output_hashes,
        "future_target_metric_used": False,
        "selections": selection_rows,
    }
    record_path = args.output_dir / "evaluation_probe_selector.json"
    write_json(record_path, record)
    record_sha = sha256_file(record_path)
    phase_rows.append(
        {
            "phase": "evaluation_probe_choices_frozen",
            "case_count": len(cases),
            "evaluation_targets_loaded": True,
            "future_target_metric_used": False,
            "choice_sha256": record_sha,
        }
    )
    return {
        "cases": cases,
        "hook": hook,
        "reference_checks": reference_checks,
        "case_contract_rows": case_contract_rows,
        "output_rows": output_rows,
        "selected_by_case": selected_by_case,
        "probes_by_case": probes_by_case,
        "bias_by_case": bias_by_case,
        "selection_rows": selection_rows,
        "record_path": record_path,
        "record_sha": record_sha,
        "artifact_sha256": output_hashes,
    }


def _append_arm_diagnostics(
    outputs: dict[str, list[dict[str, Any]]],
    case: CaseData,
    *,
    arm: str,
    result: parent.RolloutResult,
    basis: np.ndarray | None,
    applied_coefficients: np.ndarray | None,
) -> None:
    outputs["evaluation_case_summary.csv"].append(
        parent._case_summary_row(case, arm, result)
    )
    outputs["correction_integral_audit.csv"].extend(
        parent._correction_integral_rows(
            case,
            result,
            arm=arm,
            policy_adjustment=np.zeros_like(result.corrections),
        )
    )
    diagnostics = parent._same_input_rows(
        case,
        result,
        arm=arm,
        basis=basis,
        applied_coefficients=applied_coefficients,
    )
    outputs["evaluation_call_metrics.csv"].extend(diagnostics["call_rows"])
    outputs["projection_metrics.csv"].extend(diagnostics["projection_rows"])
    outputs["projection_component_metrics.csv"].extend(
        diagnostics["projection_component_rows"]
    )
    outputs["signed_component_budgets.csv"].extend(diagnostics["budget_rows"])
    outputs["sequence_summaries.csv"].extend(diagnostics["sequence_summary_rows"])
    outputs["sequence_time_metrics.csv"].extend(diagnostics["sequence_time_rows"])
    outputs["lag_correlations.csv"].extend(diagnostics["lag_rows"])
    outputs["pod_summaries.csv"].extend(diagnostics["pod_rows"])
    for row in result.admissibility_rows:
        outputs["completion.csv"].append(
            {
                "family": case.family,
                "case_id": case.case_id,
                "resolution": case.resolution_name,
                "arm": arm,
                **row,
            }
        )


def _evaluate_frozen_choices(
    args: argparse.Namespace,
    *,
    spec: ResponseExperimentSpec,
    model: torch.nn.Module,
    device: torch.device,
    candidates: Sequence[parent.Candidate],
    frozen_coefficients: np.ndarray,
    probe_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    static_candidate = next(
        candidate for candidate in candidates if candidate.gain == 0.125
    )
    outputs: dict[str, list[dict[str, Any]]] = {
        "evaluation_case_summary.csv": [],
        "evaluation_comparisons.csv": [],
        "evaluation_probe_replay.csv": [],
        "evaluation_call_metrics.csv": [],
        "projection_metrics.csv": [],
        "projection_component_metrics.csv": [],
        "projection_status.csv": [],
        "signed_component_budgets.csv": [],
        "sequence_summaries.csv": [],
        "sequence_time_metrics.csv": [],
        "lag_correlations.csv": [],
        "pod_summaries.csv": [],
        "completion.csv": [],
        "correction_integral_audit.csv": [],
        "visual_payload_inventory.csv": [],
    }
    all_results = []
    selection_rows = probe_bundle["selection_rows"]
    default_visual_ids = {
        probe_bundle["cases"][0].case_id,
        probe_bundle["cases"][len(probe_bundle["cases"]) // 2].case_id,
        probe_bundle["cases"][-1].case_id,
    }
    visual_ids = (
        set(args.visualization_cases)
        if args.visualization_cases
        else default_visual_ids
    )
    if not visual_ids <= {case.case_id for case in probe_bundle["cases"]}:
        raise ValueError(
            f"{spec.experiment_id} visualization case lies outside evaluation"
        )
    for index, case in enumerate(probe_bundle["cases"], start=1):
        bias = probe_bundle["bias_by_case"][case.case_id]
        baseline = parent._rollout_candidate(
            model,
            case,
            candidates[0],
            bias_sequence=None,
            shock_quantile=args.shock_quantile,
        )
        static = parent._rollout_candidate(
            model,
            case,
            static_candidate,
            bias_sequence=bias,
            shock_quantile=args.shock_quantile,
        )
        selected_candidate = probe_bundle["selected_by_case"][case.case_id]
        if selected_candidate.is_zero:
            selected = baseline
        elif selected_candidate.key == static_candidate.key:
            selected = static
        else:
            selected = parent._rollout_candidate(
                model,
                case,
                selected_candidate,
                bias_sequence=bias,
                shock_quantile=args.shock_quantile,
            )
        all_results.extend((baseline, static, selected))
        outputs["projection_status.csv"].append(
            {
                "family": case.family,
                "case_id": case.case_id,
                "resolution": case.resolution_name,
                "selected_candidate": selected_candidate.key,
                "status": (
                    "not_applicable_zero"
                    if selected_candidate.is_zero
                    else "applied_complete"
                    if selected.complete
                    else "failed_incomplete"
                ),
            }
        )
        replay = _probe_replay(
            probe_bundle["probes_by_case"][case.case_id][selected_candidate.key],
            selected,
            probe_calls=args.probe_calls,
        )
        outputs["evaluation_probe_replay.csv"].append(
            {
                "case_id": case.case_id,
                "selected_candidate": selected_candidate.key,
                "selected_gain": selected_candidate.gain,
                **replay,
            }
        )
        outputs["evaluation_comparisons.csv"].extend(
            (
                _comparison_row(
                    case.case_id,
                    selected,
                    baseline,
                    comparison="selected_vs_zero",
                ),
                _comparison_row(
                    case.case_id,
                    selected,
                    static,
                    comparison="selected_vs_static",
                ),
                _comparison_row(
                    case.case_id,
                    static,
                    baseline,
                    comparison="static_vs_zero",
                ),
            )
        )
        basis, _ = parent._bias_sequence(case, frozen_coefficients, rank=8)
        for arm, result in (
            ("baseline", baseline),
            ("static", static),
            ("selected", selected),
        ):
            _append_arm_diagnostics(
                outputs,
                case,
                arm=arm,
                result=result,
                basis=None if result.candidate.is_zero else basis,
                applied_coefficients=(
                    None
                    if result.candidate.is_zero
                    else -result.candidate.gain * frozen_coefficients
                ),
            )
        if case.case_id in visual_ids:
            visual_path = (
                args.output_dir
                / "visual_payloads"
                / f"dynamic_fv_{case.case_id}_{case.resolution_name}.npz"
            )
            visual = _save_visual_payload(
                visual_path,
                case,
                baseline,
                static,
                selected,
                schema=spec.schema,
            )
            visual["relative_path"] = str(
                visual_path.relative_to(args.output_dir)
            ).replace("\\", "/")
            visual["sha256"] = sha256_file(visual_path)
            outputs["visual_payload_inventory.csv"].append(visual)
        print(
            f"evaluated {index}/{len(probe_bundle['cases'])} "
            f"{case.case_id} {case.resolution_name}",
            flush=True,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {
        "output_rows": outputs,
        "promotion": _promotion(outputs["evaluation_comparisons.csv"], selection_rows),
        "all_results_complete": bool(all(result.complete for result in all_results)),
        "visual_case_ids": tuple(sorted(visual_ids)),
    }


def _conditional_evaluation(
    args: argparse.Namespace,
    *,
    spec: ResponseExperimentSpec,
    model: torch.nn.Module,
    checkpoint: Mapping[str, Any],
    store: PCNOEuler2DShardStore,
    device: torch.device,
    evaluation_ids: Sequence[str],
    candidates: Sequence[parent.Candidate],
    frozen_coefficients: np.ndarray,
    frozen_policy: Mapping[str, Any],
    response_table_rows: Sequence[Mapping[str, Any]],
    selector_sha: str,
    phase_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    probe = _freeze_evaluation_probe_choices(
        args,
        spec=spec,
        model=model,
        checkpoint=checkpoint,
        store=store,
        device=device,
        evaluation_ids=evaluation_ids,
        candidates=candidates,
        frozen_coefficients=frozen_coefficients,
        frozen_policy=frozen_policy,
        response_table_rows=response_table_rows,
        selector_sha=selector_sha,
        phase_rows=phase_rows,
    )
    evaluated = _evaluate_frozen_choices(
        args,
        spec=spec,
        model=model,
        device=device,
        candidates=candidates,
        frozen_coefficients=frozen_coefficients,
        probe_bundle=probe,
    )
    if sha256_file(probe["record_path"]) != probe["record_sha"] or any(
        sha256_file(args.output_dir / name) != digest
        for name, digest in probe["artifact_sha256"].items()
    ):
        raise ValueError(
            f"{spec.experiment_id} evaluation choice bundle changed during H30"
        )
    output_rows = {
        **probe["output_rows"],
        **evaluated["output_rows"],
        "evaluation_reference_checks.csv": probe["reference_checks"],
    }
    call_rows = output_rows["evaluation_call_metrics.csv"]
    replay_rows = output_rows["evaluation_probe_replay.csv"]
    inventories = _evaluation_inventory_checks(
        output_rows,
        evaluation_ids=evaluation_ids,
        candidates=candidates,
        selection_rows=probe["selection_rows"],
        rollout_calls=args.rollout_calls,
        visual_case_ids=evaluated["visual_case_ids"],
    )
    prefix_replay_pass = bool(all(bool(row["passed"]) for row in replay_rows))
    maximum_recurrence = _maximum_absolute(
        call_rows, "recurrence_closure_rms", empty=0.0
    )
    maximum_growth = _maximum_absolute(call_rows, "growth_closure", empty=0.0)
    maximum_same_input_energy = _maximum_absolute(
        call_rows, "same_input_energy_closure", empty=0.0
    )
    maximum_same_input = _maximum_absolute(
        call_rows, "same_input_pointwise_closure_max", empty=0.0
    )
    projection_rows = output_rows["projection_metrics.csv"]
    projection_component_rows = output_rows["projection_component_metrics.csv"]
    maximum_projection_inner = _maximum_absolute(
        projection_rows, "maximum_parallel_orthogonal_inner"
    )
    maximum_projection_closure = _maximum_absolute(
        projection_rows, "maximum_energy_closure"
    )
    maximum_projection_reconstruction = _maximum_absolute(
        projection_rows, "maximum_reconstruction_error"
    )
    maximum_non_type0_correction = _maximum_absolute(
        projection_rows, "maximum_non_type0_correction"
    )
    maximum_coefficient_reconstruction = _maximum_absolute(
        projection_rows, "maximum_coefficient_reconstruction_error"
    )
    maximum_orthogonal_partition_change = _maximum_absolute(
        projection_rows, "maximum_orthogonal_partition_change"
    )
    maximum_excluded_partition_change = _maximum_absolute(
        projection_rows, "maximum_excluded_partition_change"
    )
    maximum_correction_projection_leakage = max(
        _maximum_absolute(projection_rows, "correction_orthogonal_energy"),
        _maximum_absolute(projection_rows, "correction_excluded_non_type0_energy"),
    )
    maximum_component_projection_inner = max(
        _maximum_absolute(
            projection_component_rows, "before_parallel_orthogonal_inner"
        ),
        _maximum_absolute(projection_component_rows, "after_parallel_orthogonal_inner"),
        _maximum_absolute(
            projection_component_rows, "correction_parallel_orthogonal_inner"
        ),
    )
    maximum_component_energy_closure = max(
        _maximum_absolute(projection_component_rows, "before_energy_closure"),
        _maximum_absolute(projection_component_rows, "after_energy_closure"),
        _maximum_absolute(projection_component_rows, "correction_energy_closure"),
    )
    maximum_constant_nonconstant_energy_closure = _maximum_absolute(
        projection_component_rows,
        "constant_nonconstant_energy_closure",
    )
    visual_rows = output_rows["visual_payload_inventory.csv"]
    maximum_visual_cumulative_closure = _maximum_absolute(
        visual_rows, "maximum_cumulative_closure"
    )
    maximum_visual_growth_replay = _maximum_absolute(
        visual_rows, "maximum_signed_growth_replay"
    )
    visual_hash_pass = bool(
        all(
            sha256_file(args.output_dir / str(row["relative_path"]))
            == str(row["sha256"])
            for row in visual_rows
        )
    )
    integral_rows = output_rows["correction_integral_audit.csv"]
    correction_integral_contract_pass = bool(
        all(bool(row["integral_closure_pass"]) for row in integral_rows)
    )
    required_integral_rows = [
        row for row in integral_rows if bool(row["integral_neutralization_required"])
    ]
    maximum_required_integral_closure = (
        max(
            abs(float(row["residual_scaled_physical_volume_mean"]))
            for row in required_integral_rows
        )
        if required_integral_rows
        else 0.0
    )
    closure_pass = bool(
        maximum_recurrence <= PROBE_ABSOLUTE_REPLAY_LIMIT
        and maximum_growth <= 2.0e-5
        and maximum_same_input_energy <= 1.0e-10
        and maximum_same_input <= 1.0e-12
        and maximum_projection_inner <= 1.0e-10
        and maximum_component_projection_inner <= 1.0e-10
        and maximum_projection_closure <= 1.0e-10
        and maximum_component_energy_closure <= 1.0e-10
        and maximum_constant_nonconstant_energy_closure <= 1.0e-10
        and maximum_projection_reconstruction <= 1.0e-10
        and maximum_non_type0_correction <= 1.0e-12
        and maximum_coefficient_reconstruction <= 1.0e-10
        and maximum_orthogonal_partition_change <= 1.0e-10
        and maximum_excluded_partition_change <= 1.0e-12
        and maximum_correction_projection_leakage <= 1.0e-20
        and maximum_visual_cumulative_closure <= 2.0e-5
        and maximum_visual_growth_replay <= 1.0e-10
        and visual_hash_pass
        and correction_integral_contract_pass
        and maximum_required_integral_closure <= parent.INTEGRAL_CLOSURE_LIMIT
    )
    closure_metrics = {
        "maximum_recurrence_closure_rms": maximum_recurrence,
        "maximum_growth_closure_absolute": maximum_growth,
        "maximum_same_input_energy_closure": maximum_same_input_energy,
        "maximum_same_input_pointwise_closure": maximum_same_input,
        "maximum_projection_inner": maximum_projection_inner,
        "maximum_component_projection_inner": maximum_component_projection_inner,
        "maximum_projection_closure": maximum_projection_closure,
        "maximum_component_energy_closure": maximum_component_energy_closure,
        "maximum_constant_nonconstant_energy_closure": (
            maximum_constant_nonconstant_energy_closure
        ),
        "maximum_projection_reconstruction": maximum_projection_reconstruction,
        "maximum_non_type0_correction": maximum_non_type0_correction,
        "maximum_coefficient_reconstruction": maximum_coefficient_reconstruction,
        "maximum_orthogonal_partition_change": maximum_orthogonal_partition_change,
        "maximum_excluded_partition_change": maximum_excluded_partition_change,
        "maximum_correction_projection_leakage": (
            maximum_correction_projection_leakage
        ),
        "maximum_visual_cumulative_closure": maximum_visual_cumulative_closure,
        "maximum_visual_growth_replay": maximum_visual_growth_replay,
        "maximum_required_integral_closure": maximum_required_integral_closure,
    }
    return {
        "hook": probe["hook"],
        "reference_checks": probe["reference_checks"],
        "case_contract_rows": probe["case_contract_rows"],
        "output_rows": output_rows,
        "promotion": evaluated["promotion"],
        "choice_bundle": {
            "record_path": probe["record_path"],
            "record_sha": probe["record_sha"],
            "artifact_sha256": probe["artifact_sha256"],
        },
        "checks": {
            "evaluation_choice_bundle_revalidated": True,
            "evaluation_inventory_pass": inventories["row_inventories_pass"],
            "evaluation_inventory_details": inventories,
            "all_evaluation_complete": evaluated["all_results_complete"],
            "probe_prefix_replay_pass": prefix_replay_pass,
            "closure_pass": closure_pass,
            "correction_integral_contract_pass": (correction_integral_contract_pass),
            "visual_payload_hash_pass": visual_hash_pass,
            "closure_metrics": closure_metrics,
            "maximum_probe_prefix_absolute": _maximum_absolute(
                replay_rows, "maximum_absolute", empty=0.0
            ),
            "maximum_probe_prefix_relative_l2": _maximum_absolute(
                replay_rows, "relative_l2", empty=0.0
            ),
            **closure_metrics,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    summary = run(parse_args(argv))
    print(
        f"D076 dynamic_fv status={summary['status']} "
        f"calibration_qualified={summary['calibration_qualification']['passed']} "
        f"promotion={summary['promotion'].get('passed')}"
    )
    return 0 if summary["contract_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
