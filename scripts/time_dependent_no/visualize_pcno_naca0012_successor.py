#!/usr/bin/env python3
"""Replay and render the frozen NACA corrective-successor comparison.

The qualitative selection is fixed before successor results are inspected:
development anchor 1234, seeds 17/29/43, and every deployed map (the four
learned arms plus CLEAN with PATH_PROJECTION).  Replay runs beside the verified
packets; rendering consumes only the resulting portable, hash-bound state and
density archive.  No prospective or sealed role is supported.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import shutil
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import torch

MODULE_PATH = Path(__file__)
if MODULE_PATH.is_symlink():
    raise RuntimeError("successor visualization source is aliased")
REPO_ROOT = MODULE_PATH.resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _source_record(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    if path.read_bytes() != payload:
        raise RuntimeError("successor visualization source changed while reading")
    return {"bytes": len(payload), "sha256": sha256(payload).hexdigest()}


VISUALIZATION_SOURCE_AT_IMPORT = _source_record(MODULE_PATH.resolve())

from scripts.time_dependent_no.evaluate_pcno_naca0012_successor import (
    EXPERIMENT_ID,
    LEARNED_ARMS,
    SCIENTIFIC_OUTPUT_FILES,
    SEEDS,
    ClosedPacket,
    ProtectedPopulationError,
    TrainingPacket,
    _canonical_sha256,
    _file_record,
    _file_sha256,
    _instantiate_model,
    _load_calibration,
    _load_dataset,
    _load_successor_contract,
    _load_training_packets,
    _make_staging_directory,
    _numeric,
    _packet_json,
    _rename_directory_no_replace,
    _reverify_packet,
    _reverify_source,
    _source_snapshot,
    _validate_population,
    _verify_closed_packet,
    _verify_record,
    _write_json,
)
from scripts.time_dependent_no.evaluate_pcno_naca0012_successor import (
    HORIZONS as EVALUATOR_HORIZONS,
)
from scripts.time_dependent_no.evaluate_pcno_naca0012_successor import (
    OUTPUT_FILES as EVALUATION_OUTPUT_FILES,
)
from utility.time_dependent_no.pcno_naca0012 import recurrent_step
from utility.time_dependent_no.pcno_naca0012_successor import (
    TrainPathProjector,
    corrected_recurrent_step,
)

if _source_record(MODULE_PATH.resolve()) != VISUALIZATION_SOURCE_AT_IMPORT:
    raise RuntimeError("successor visualization source changed during bootstrap")

REPLAY_SCHEMA = "time_dependent_no.naca_corrective_successor_visualization_replay.v2"
RENDER_SCHEMA = "time_dependent_no.naca_corrective_successor_visualization.v2"
REPLAY_FILE = "rollout_replay.npz"
REPLAY_MANIFEST_FILE = "replay_manifest.json"
RENDER_MANIFEST_FILE = "visualization_manifest.json"

ANCHOR = 1234
DEPLOYMENTS = (*LEARNED_ARMS, "PATH_PROJECTION")
FULL_HORIZONS = tuple(range(209))
STATIC_HORIZONS = (35, 104, 208)
FIELD = "Density"
FIELD_INDEX = 0
# This gate compares only the first recurrent transition.  Float32 CUDA kernels
# are not bitwise deterministic across independent runs, so later differences
# are diagnostics rather than implementation-equivalence tests.  Keep these
# constants isolated: the final values are fixed from the registered all-arm h1
# diagnostic, not from amplified long-horizon discrepancies.
FIRST_TRANSITION_RTOL = 1.0e-5
FIRST_TRANSITION_ATOL = 1.0e-4
X_LIMITS = (-1.0, 10.0)
Y_LIMITS = (-5.0, 5.0)

DISPLAY_NAMES = {
    "CLEAN": "Clean",
    "IID_RECOVERY": "IID recovery",
    "ERROR_SUBSPACE_RECOVERY": "Error-subspace recovery",
    "DETACHED_PUSHFORWARD": "Detached pushforward",
    "PATH_PROJECTION": "Path projection",
}
METHOD_COLORS = {
    "CLEAN": "#000000",
    "IID_RECOVERY": "#0072B2",
    "ERROR_SUBSPACE_RECOVERY": "#E69F00",
    "DETACHED_PUSHFORWARD": "#009E73",
    "PATH_PROJECTION": "#CC79A7",
}
SEED_COLORS = ("#0072B2", "#D55E00", "#009E73")
SEED_MARKERS = ("o", "s", "^")

QUALITATIVE_CONTRACT = {
    "population_role": "development",
    "anchor": ANCHOR,
    "seeds": list(SEEDS),
    "deployments": list(DEPLOYMENTS),
    "path_projection_parent": "CLEAN",
    "field": FIELD,
    "spatial_representation": "flat_native_quadrilateral_vertex_mean",
    "interpolation": "none",
    "rollout_horizons_inclusive": [0, 208],
    "registered_snapshot_checks": list(EVALUATOR_HORIZONS),
    "first_transition_implementation_gate": {
        "horizon": 1,
        "rtol": FIRST_TRANSITION_RTOL,
        "atol": FIRST_TRANSITION_ATOL,
        "scope": "float32_cuda_independent_rerun",
        "rationale": (
            "tight simple pair above the registered all-arm h1 max absolute "
            "difference 9.5367431640625e-05; never applied as a later-horizon gate"
        ),
    },
    "later_registered_checks_are_diagnostic_only": True,
    "paper_static_horizons": list(STATIC_HORIZONS),
    "paper_static_source": "exact_evaluator_registered_snapshots",
    "continuous_qualitative_source": "independent_cuda_rerun",
    "paper_panels": [
        "raw_density",
        "signed_normalized_density_error",
        "density_error_trace",
        "response_rollout_and_path_accuracy_tradeoffs",
    ],
    "prospective_opened": False,
    "sealed_opened": False,
}
QUALITATIVE_CONTRACT_SHA256 = _canonical_sha256(QUALITATIVE_CONTRACT)


@dataclass(frozen=True)
class ReplayBundle:
    """Validated portable data used by all successor visualizations."""

    anchor: int
    horizons: np.ndarray
    frame_indices: np.ndarray
    seeds: np.ndarray
    deployments: tuple[str, ...]
    coordinates: np.ndarray
    quads: np.ndarray
    airfoil_mask: np.ndarray
    reference_density: np.ndarray
    predicted_density: np.ndarray
    valid_mask: np.ndarray
    registered_horizons: np.ndarray
    exact_reference_density: np.ndarray
    exact_predicted_density: np.ndarray
    exact_valid_mask: np.ndarray
    density_state_scale: float
    density_error_rmse: np.ndarray
    response_gain: np.ndarray
    rollout_auc: np.ndarray
    late_state_error: np.ndarray
    late_path_distance: np.ndarray
    late_projected_path_discrepancy: np.ndarray
    manifest: Mapping[str, Any]


def _reverify_visualization_source() -> None:
    if _source_record(MODULE_PATH.resolve()) != VISUALIZATION_SOURCE_AT_IMPORT:
        raise ValueError("successor visualization source changed during execution")


def _claim_boundary() -> dict[str, Any]:
    return {
        "visualization_only_replay": True,
        "population_role": "development",
        "non_cherry_picked_anchor": ANCHOR,
        "all_registered_seeds_shown": True,
        "all_deployed_maps_shown": True,
        "raw_and_corrected_path_recurrences_separated": True,
        "path_projection_panel_uses_corrected_recurrent_feedback": True,
        "static_pdfs_use_exact_evaluator_snapshots": True,
        "animations_and_traces_use_independent_qualitative_replay": True,
        "long_horizon_cuda_rerun_equality_claimed": False,
        "trusted_displaced_state_response_claimed": False,
        "manifold_drift_established_by_visualization": False,
        "prospective_opened": False,
        "sealed_opened": False,
    }


def _array_sha256(array: np.ndarray) -> str:
    """Hash one already-validated array without dtype or value conversion."""

    return sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _is_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_training_records(records: Any) -> None:
    if not isinstance(records, list) or len(records) != len(LEARNED_ARMS) * len(SEEDS):
        raise ValueError("visualization training-packet inventory differs")
    expected_keys = [
        (deployment, seed) for deployment in LEARNED_ARMS for seed in SEEDS
    ]
    observed_keys: list[tuple[str, int]] = []
    source_sets: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {
            "arm",
            "seed",
            "final_hash_manifest_sha256",
            "checkpoint_sha256",
            "source_set_sha256",
        }:
            raise ValueError("visualization training-packet record differs")
        arm = record["arm"]
        seed = record["seed"]
        if (
            not isinstance(arm, str)
            or isinstance(seed, bool)
            or not isinstance(seed, int)
        ):
            raise TypeError("visualization training-packet identity is invalid")
        if not all(
            _is_digest(record[key])
            for key in (
                "final_hash_manifest_sha256",
                "checkpoint_sha256",
                "source_set_sha256",
            )
        ):
            raise ValueError("visualization training-packet digest is invalid")
        observed_keys.append((arm, seed))
        source_sets.add(record["source_set_sha256"])
    if observed_keys != expected_keys:
        raise ValueError("visualization training packets do not form the fixed grid")
    if len(source_sets) != 1:
        raise ValueError("visualization training packets do not share one source set")


def _snapshot_check_record(
    *,
    deployment: str,
    seed: int,
    horizon: int,
    observed: np.ndarray | None,
    expected: np.ndarray | None,
) -> dict[str, Any]:
    """Describe one rerun/evaluator comparison and enforce only the h1 gate."""

    gate_role = (
        "first_transition_implementation_gate" if horizon == 1 else "diagnostic_only"
    )
    comparison_available = observed is not None and expected is not None
    record: dict[str, Any] = {
        "deployment": deployment,
        "seed": seed,
        "anchor": ANCHOR,
        "horizon": horizon,
        "gate_role": gate_role,
        "finite_in_evaluation": expected is not None,
        "finite_in_qualitative_replay": observed is not None,
        "comparison_available": comparison_available,
        "prediction_bitwise_equal": None,
        "prediction_max_abs_difference": None,
        "prediction_max_relative_difference": None,
        "prediction_max_tolerance_ratio": None,
        "prediction_element_count": None,
        "prediction_mismatch_count": None,
        "prediction_tolerance_pass": None,
        "prediction_rtol": FIRST_TRANSITION_RTOL,
        "prediction_atol": FIRST_TRANSITION_ATOL,
    }
    if comparison_available:
        assert observed is not None and expected is not None
        observed_array = np.asarray(observed)
        expected_array = np.asarray(expected)
        if (
            observed_array.shape != expected_array.shape
            or observed_array.dtype != np.float32
            or expected_array.dtype != np.float32
            or not np.all(np.isfinite(observed_array))
            or not np.all(np.isfinite(expected_array))
        ):
            raise ValueError("registered prediction snapshot array contract differs")
        difference = np.abs(
            observed_array.astype(np.float64) - expected_array.astype(np.float64)
        )
        tolerance = FIRST_TRANSITION_ATOL + FIRST_TRANSITION_RTOL * np.abs(
            expected_array.astype(np.float64)
        )
        within_tolerance = difference <= tolerance
        record.update(
            {
                "prediction_bitwise_equal": bool(
                    np.array_equal(observed_array, expected_array)
                ),
                "prediction_max_abs_difference": float(np.max(difference)),
                "prediction_max_relative_difference": float(
                    np.max(
                        difference
                        / np.maximum(
                            np.abs(expected_array.astype(np.float64)),
                            np.finfo(np.float32).tiny,
                        )
                    )
                ),
                "prediction_max_tolerance_ratio": float(np.max(difference / tolerance)),
                "prediction_element_count": int(expected_array.size),
                "prediction_mismatch_count": int(np.count_nonzero(~within_tolerance)),
                "prediction_tolerance_pass": bool(np.all(within_tolerance)),
            }
        )
    if horizon == 1:
        if not comparison_available:
            raise ValueError(
                f"first-transition finite-state presence differs: {deployment}:{seed}:h1"
            )
        if record["prediction_tolerance_pass"] is not True:
            raise ValueError(
                "first-transition implementation-consistency gate failed: "
                f"{deployment}:{seed}:h1"
            )
    return record


def _snapshot_check_summary(checks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    first = [check for check in checks if check.get("horizon") == 1]
    later = [check for check in checks if check.get("horizon") != 1]
    return {
        "total_check_count": len(checks),
        "first_transition_gate_count": len(first),
        "first_transition_gate_all_passed": all(
            check.get("comparison_available") is True
            and check.get("prediction_tolerance_pass") is True
            for check in first
        ),
        "later_diagnostic_count": len(later),
        "later_comparison_available_count": sum(
            check.get("comparison_available") is True for check in later
        ),
        "later_presence_mismatch_count": sum(
            check.get("finite_in_evaluation")
            != check.get("finite_in_qualitative_replay")
            for check in later
        ),
        "later_tolerance_failure_count": sum(
            check.get("prediction_tolerance_pass") is False for check in later
        ),
    }


def _validate_replay_bookkeeping(
    manifest: Mapping[str, Any],
    valid_mask: np.ndarray,
    *,
    evaluator_prediction_states: np.ndarray,
    evaluator_valid_mask: np.ndarray,
    qualitative_prediction_states: np.ndarray,
    qualitative_valid_mask: np.ndarray,
) -> None:
    first_nonfinite = manifest.get("first_nonfinite_horizon")
    expected_trace_keys = {
        f"{deployment}:{seed}" for deployment in DEPLOYMENTS for seed in SEEDS
    }
    if (
        not isinstance(first_nonfinite, Mapping)
        or set(first_nonfinite) != expected_trace_keys
    ):
        raise ValueError("visualization nonfinite-trace inventory differs")
    for method_offset, deployment in enumerate(DEPLOYMENTS):
        for seed_offset, seed in enumerate(SEEDS):
            trace = valid_mask[method_offset, seed_offset]
            invalid = np.flatnonzero(~trace)
            expected = int(invalid[0]) if invalid.size else None
            observed = first_nonfinite[f"{deployment}:{seed}"]
            if expected is None:
                matches = observed is None
            else:
                matches = (
                    not isinstance(observed, bool)
                    and isinstance(observed, int)
                    and 1 <= observed <= FULL_HORIZONS[-1]
                    and observed == expected
                )
            if not matches:
                raise ValueError("visualization first-nonfinite record differs")

    checks = manifest.get("registered_snapshot_replay_checks")
    if not isinstance(checks, list) or len(checks) != (
        len(DEPLOYMENTS) * len(SEEDS) * len(EVALUATOR_HORIZONS)
    ):
        raise ValueError("visualization snapshot-check inventory differs")
    expected_identities = [
        (deployment, seed, horizon)
        for deployment in DEPLOYMENTS
        for seed in SEEDS
        for horizon in EVALUATOR_HORIZONS
    ]
    observed_identities: list[tuple[str, int, int]] = []
    expected_checks: list[dict[str, Any]] = []
    for method_offset, deployment in enumerate(DEPLOYMENTS):
        for seed_offset, seed in enumerate(SEEDS):
            for horizon_offset, horizon in enumerate(EVALUATOR_HORIZONS):
                evaluator_finite = bool(
                    evaluator_valid_mask[method_offset, seed_offset, horizon_offset]
                )
                qualitative_finite = bool(
                    qualitative_valid_mask[method_offset, seed_offset, horizon_offset]
                )
                expected_checks.append(
                    _snapshot_check_record(
                        deployment=deployment,
                        seed=seed,
                        horizon=horizon,
                        observed=(
                            qualitative_prediction_states[
                                method_offset, seed_offset, horizon_offset
                            ]
                            if qualitative_finite
                            else None
                        ),
                        expected=(
                            evaluator_prediction_states[
                                method_offset, seed_offset, horizon_offset
                            ]
                            if evaluator_finite
                            else None
                        ),
                    )
                )
    for check, expected_check in zip(checks, expected_checks, strict=True):
        if not isinstance(check, Mapping):
            raise TypeError("visualization snapshot check is not an object")
        identity = (check.get("deployment"), check.get("seed"), check.get("horizon"))
        if identity != (
            expected_check["deployment"],
            expected_check["seed"],
            expected_check["horizon"],
        ):
            raise ValueError("visualization snapshot-check identity differs")
        observed_identities.append(identity)  # type: ignore[arg-type]
        if dict(check) != expected_check:
            raise ValueError("visualization snapshot diagnostic record differs")
    if observed_identities != expected_identities:
        raise ValueError("visualization snapshot checks do not form the fixed grid")
    expected_summary = _snapshot_check_summary(expected_checks)
    if (
        expected_summary["total_check_count"] != 75
        or expected_summary["first_transition_gate_count"] != 15
        or expected_summary["first_transition_gate_all_passed"] is not True
        or manifest.get("registered_snapshot_check_summary") != expected_summary
    ):
        raise ValueError("visualization snapshot-check summary differs")


def _reverify_file_record(path: Path, root: Path, expected: Mapping[str, Any]) -> None:
    if _file_record(path, root) != dict(expected):
        raise ValueError(f"visualization input changed while rendering: {path.name}")


def _paths_overlap(left: Path, right: Path) -> bool:
    left_forms = {
        Path(os.path.abspath(os.fspath(left))),
        left.resolve(strict=False),
    }
    right_forms = {
        Path(os.path.abspath(os.fspath(right))),
        right.resolve(strict=False),
    }
    return any(
        left_form == right_form
        or left_form in right_form.parents
        or right_form in left_form.parents
        for left_form in left_forms
        for right_form in right_forms
    )


def _validate_output_disjoint(
    output: Path, inputs: Sequence[tuple[str, Path]], *, label: str
) -> None:
    for input_label, input_path in inputs:
        if _paths_overlap(output, input_path):
            raise ValueError(f"{label} overlaps {input_label}")


def _primary_variant_key(deployment: str, seed: int) -> str:
    if deployment == "PATH_PROJECTION":
        return f"PATH_PROJECTION:{seed}:path_projection:corrected"
    return f"{deployment}:{seed}:raw:raw"


def _extract_summary_arrays(result: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Extract the preregistered mediator/path summaries without selecting values."""

    offline = result.get("offline_diagnostics")
    methods = result.get("method_summaries")
    if not isinstance(offline, Mapping) or not isinstance(methods, Mapping):
        raise TypeError("evaluation result lacks diagnostic or method summaries")
    response = np.full((len(DEPLOYMENTS), len(SEEDS)), np.nan, dtype=np.float64)
    rollout_auc = np.empty_like(response)
    late_error = np.empty_like(response)
    late_path = np.empty_like(response)
    late_projected = np.empty_like(response)
    for method_offset, deployment in enumerate(DEPLOYMENTS):
        for seed_offset, seed in enumerate(SEEDS):
            summary = methods.get(_primary_variant_key(deployment, seed))
            if not isinstance(summary, Mapping):
                raise TypeError(
                    f"evaluation result lacks primary variant {deployment}:{seed}"
                )
            if (
                summary.get("arm") != deployment
                or summary.get("seed") != seed
                or summary.get("state_stage")
                != ("corrected" if deployment == "PATH_PROJECTION" else "raw")
            ):
                raise ValueError("evaluation primary-variant identity differs")
            values = (
                ("normalized_state_error_auc", rollout_auc),
                ("late_window_normalized_state_error_median", late_error),
                ("late_window_path_distance_median", late_path),
                (
                    "late_window_projected_path_discrepancy_median",
                    late_projected,
                ),
            )
            for key, destination in values:
                aggregate = summary.get(key)
                if not isinstance(aggregate, Mapping) or aggregate.get("count") != 8:
                    raise ValueError(f"evaluation summary aggregate differs: {key}")
                value = _numeric(aggregate.get("median"))
                if math.isnan(value) or value < 0.0:
                    raise ValueError(f"evaluation summary is invalid: {key}")
                destination[method_offset, seed_offset] = value
            if deployment != "PATH_PROJECTION":
                diagnostic = offline.get(f"{deployment}:{seed}")
                if not isinstance(diagnostic, Mapping):
                    raise ValueError("evaluation offline diagnostic identity differs")
                aggregate = diagnostic.get("one_prefix_response_gain")
                if not isinstance(aggregate, Mapping) or aggregate.get("count") != 8:
                    raise ValueError("one-prefix response aggregate differs")
                value = _numeric(aggregate.get("median"))
                if math.isnan(value) or value < 0.0:
                    raise ValueError("one-prefix response gain is invalid")
                response[method_offset, seed_offset] = value
    return {
        "response_gain": response,
        "rollout_auc": rollout_auc,
        "late_state_error": late_error,
        "late_path_distance": late_path,
        "late_projected_path_discrepancy": late_projected,
    }


def _verify_evaluation_packet(
    root: Path,
    *,
    successor: Mapping[str, Any],
    successor_sha256: str,
    dataset_sha256: str,
    dataset_payload_sha256: str,
    calibration_packet: ClosedPacket,
    calibration: Mapping[str, Any],
    training_packets: Sequence[TrainingPacket],
    evaluator_source: Mapping[str, Any],
) -> tuple[ClosedPacket, dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    schemas = successor["packet_schemas"]
    packet = _verify_closed_packet(
        root,
        final_schema=schemas["final_hash_manifest"],
        expected_files=EVALUATION_OUTPUT_FILES,
        label="successor development evaluation packet",
    )
    scientific = {name: packet.files[name] for name in sorted(SCIENTIFIC_OUTPUT_FILES)}
    if (
        packet.final.get("experiment_id") != EXPERIMENT_ID
        or packet.final.get("successor_contract_sha256") != successor_sha256
        or packet.final.get("population_role") != "development"
        or packet.final.get("scientific_files_sha256") != _canonical_sha256(scientific)
        or packet.final.get("runtime_manifest_excluded_from_scientific_files_sha256")
        is not True
        or packet.final.get("prospective_opened") is not False
        or packet.final.get("sealed_opened") is not False
    ):
        raise ValueError("evaluation final manifest violates the open-role contract")

    inputs = _packet_json(
        packet,
        "input_manifest.json",
        schema=schemas["evaluation_inputs"],
        label="evaluation input manifest",
    )
    result = _packet_json(
        packet,
        "result.json",
        schema=schemas["evaluation"],
        label="evaluation result",
    )
    expected_training = [
        {
            "arm": training.arm,
            "seed": training.seed,
            "final_hash_manifest_sha256": training.packet.final_file_sha256,
            "checkpoint_sha256": training.checkpoint_record["sha256"],
            "source_set_sha256": training.summary["source_set_sha256"],
        }
        for training in training_packets
    ]
    expected_inputs = {
        "experiment_id": EXPERIMENT_ID,
        "successor_contract_sha256": successor_sha256,
        "preregistration_sha256": successor["preregistration_sha256"],
        "inherited_r0_contract_sha256": successor["inherited_r0_contract_sha256"],
        "dataset_final_hash_manifest_sha256": dataset_sha256,
        "dataset_manifest_payload_sha256": dataset_payload_sha256,
        "calibration_final_hash_manifest_sha256": (
            calibration_packet.final_file_sha256
        ),
        "calibration_payload_sha256": calibration["canonical_payload_sha256"],
        "training_packets": expected_training,
        "evaluator_source_set_sha256": evaluator_source["source_set_sha256"],
        "population_role": "development",
        "anchors": successor["evaluation"]["development_anchors"],
        "horizons": list(EVALUATOR_HORIZONS),
        "full_trace_inclusive": [1, 208],
        "late_window_inclusive": [174, 208],
        "prospective_opened": False,
        "sealed_opened": False,
    }
    for key, expected in expected_inputs.items():
        if inputs.get(key) != expected:
            raise ValueError(f"evaluation input binding differs: {key}")
    if inputs.get("production_authority_set_sha256") != packet.final.get(
        "production_authority_set_sha256"
    ) or inputs.get("production_authorities") != packet.final.get(
        "production_authorities"
    ):
        raise ValueError("evaluation production-authority binding differs")

    claim = result.get("claim_boundary")
    if (
        result.get("experiment_id") != EXPERIMENT_ID
        or result.get("status") != "complete"
        or result.get("classification") != "SCIENTIFIC_RESULT"
        or result.get("population_role") != "development"
        or result.get("successor_contract_sha256") != successor_sha256
        or result.get("input_manifest_payload_sha256")
        != inputs["canonical_payload_sha256"]
        or result.get("production_authority_set_sha256")
        != packet.final.get("production_authority_set_sha256")
        or result.get("production_authorities")
        != packet.final.get("production_authorities")
        or result.get("seeds") != list(SEEDS)
        or result.get("learned_arms") != list(LEARNED_ARMS)
        or result.get("operational_correctors") != ["identity", "path_projection"]
        or not isinstance(claim, Mapping)
        or claim.get("raw_and_corrected_path_recurrences_separated") is not True
        or claim.get("prospective_opened") is not False
        or claim.get("sealed_opened") is not False
    ):
        raise ValueError("evaluation result violates the successor claim boundary")

    _, snapshot_bytes = _verify_record(
        packet.root, packet.files["rollout_snapshots.npz"], "rollout_snapshots.npz"
    )
    with np.load(io.BytesIO(snapshot_bytes), allow_pickle=False) as archive:
        snapshots = {name: np.array(archive[name], copy=True) for name in archive.files}
    for horizon in EVALUATOR_HORIZONS:
        reference_key = f"reference_a{ANCHOR}_h{horizon}"
        if reference_key not in snapshots:
            raise ValueError("evaluation snapshots lack a frozen reference state")
    summary_arrays = _extract_summary_arrays(result)
    _reverify_packet(packet)
    return packet, result, snapshots, summary_arrays


def _rollout_deployment(
    *,
    deployment: str,
    model: torch.nn.Module,
    previous: torch.Tensor,
    current: torch.Tensor,
    selected_offset: int,
    geometry: Any,
    normalization: Any,
    projector: TrainPathProjector | None,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray], int | None]:
    """Reproduce evaluator recurrence and retain one anchor's density trace."""

    if deployment not in DEPLOYMENTS:
        raise ValueError("unsupported deployed map")
    if deployment == "PATH_PROJECTION" and projector is None:
        raise ValueError("PATH_PROJECTION requires the train-only projector")
    batch_size, node_count, _ = current.shape
    if not 0 <= selected_offset < batch_size:
        raise ValueError("selected anchor offset is invalid")
    density = np.full((len(FULL_HORIZONS), node_count), np.nan, dtype=np.float32)
    valid = np.zeros(len(FULL_HORIZONS), dtype=np.bool_)
    density[0] = current[selected_offset, :, FIELD_INDEX].detach().cpu().numpy()
    valid[0] = True
    checkpoints: dict[int, np.ndarray] = {}
    active = torch.ones(batch_size, dtype=torch.bool, device=device)
    geometry_batch = geometry.expand(batch_size, device)
    fourier = model.prepare_fourier_tensors(geometry_batch)  # type: ignore[attr-defined]
    device_projector = (
        projector.to(device, dtype=torch.float32) if projector is not None else None
    )
    first_nonfinite: int | None = None
    model.eval()
    with torch.no_grad():
        for horizon in FULL_HORIZONS[1:]:
            if deployment == "PATH_PROJECTION":
                retained_masks: list[torch.Tensor] = []

                def apply_projection(
                    raw_state: torch.Tensor,
                    *,
                    active_now: torch.Tensor = active,
                    current_now: torch.Tensor = current,
                    masks: list[torch.Tensor] = retained_masks,
                ) -> torch.Tensor:
                    raw_finite = torch.all(torch.isfinite(raw_state), dim=(1, 2))
                    retained_now = active_now & raw_finite
                    safe_query = torch.where(
                        retained_now[:, None, None], raw_state, current_now
                    )
                    assert device_projector is not None
                    projection = device_projector.project(safe_query)
                    masks.append(retained_now)
                    return torch.where(
                        retained_now[:, None, None],
                        projection.corrected_state,
                        current_now,
                    )

                step = corrected_recurrent_step(
                    model,
                    previous,
                    current,
                    geometry_batch,
                    normalization,
                    apply_projection,
                    fourier_tensors=fourier,
                )
                if len(retained_masks) != 1:
                    raise AssertionError("PATH_PROJECTION corrector call count differs")
                retained = retained_masks[0]
                raw_finite = torch.all(torch.isfinite(step.raw_next_state), dim=(1, 2))
                if not torch.equal(retained, active & raw_finite):
                    raise AssertionError("PATH_PROJECTION finite mask changed")
                candidate = step.corrected_next_state
                corrected_finite = torch.all(torch.isfinite(candidate), dim=(1, 2))
                if torch.any(retained & ~corrected_finite):
                    raise FloatingPointError(
                        "PATH_PROJECTION produced a nonfinite corrected state"
                    )
                recurrent_previous = step.recurrent_previous
            else:
                recurrent_previous, candidate = recurrent_step(
                    model,
                    previous,
                    current,
                    geometry_batch,
                    normalization,
                    fourier_tensors=fourier,
                )
                retained = active & torch.all(torch.isfinite(candidate), dim=(1, 2))

            selected_valid = bool(retained[selected_offset])
            if selected_valid:
                selected = candidate[selected_offset].detach().cpu().numpy()
                density[horizon] = np.asarray(
                    selected[:, FIELD_INDEX], dtype=np.float32
                )
                valid[horizon] = True
                if horizon in EVALUATOR_HORIZONS:
                    checkpoints[horizon] = np.asarray(selected, dtype=np.float32)
            elif bool(active[selected_offset]) and first_nonfinite is None:
                first_nonfinite = horizon
            active = retained
            if deployment == "PATH_PROJECTION":
                previous, current = recurrent_previous, candidate
            else:
                safe_candidate = torch.where(active[:, None, None], candidate, current)
                previous, current = recurrent_previous, safe_candidate
    return density, valid, checkpoints, first_nonfinite


def _snapshot_key(deployment: str, seed: int, horizon: int) -> str:
    if deployment == "PATH_PROJECTION":
        return f"path_projection_seed{seed}_corrected_a{ANCHOR}_h{horizon}"
    return f"{deployment.lower()}_seed{seed}_raw_a{ANCHOR}_h{horizon}"


def _extract_exact_registered_snapshots(
    snapshots: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pack the evaluator's immutable registered snapshots without conversion."""

    reference_states: list[np.ndarray] = []
    state_shape: tuple[int, ...] | None = None
    for horizon in EVALUATOR_HORIZONS:
        reference = np.asarray(snapshots[f"reference_a{ANCHOR}_h{horizon}"])
        if (
            reference.dtype != np.float64
            or reference.ndim != 2
            or reference.shape[1] <= FIELD_INDEX
            or not np.all(np.isfinite(reference))
        ):
            raise ValueError("evaluation reference snapshot contract differs")
        if state_shape is None:
            state_shape = reference.shape
        elif reference.shape != state_shape:
            raise ValueError("evaluation reference snapshot shape differs")
        reference_states.append(reference)
    assert state_shape is not None
    prediction_states = np.full(
        (len(DEPLOYMENTS), len(SEEDS), len(EVALUATOR_HORIZONS), *state_shape),
        np.nan,
        dtype=np.float32,
    )
    valid = np.zeros(
        (len(DEPLOYMENTS), len(SEEDS), len(EVALUATOR_HORIZONS)), dtype=np.bool_
    )
    for method_offset, deployment in enumerate(DEPLOYMENTS):
        for seed_offset, seed in enumerate(SEEDS):
            for horizon_offset, horizon in enumerate(EVALUATOR_HORIZONS):
                prediction = snapshots.get(_snapshot_key(deployment, seed, horizon))
                if prediction is None:
                    continue
                prediction_array = np.asarray(prediction)
                if (
                    prediction_array.dtype != np.float32
                    or prediction_array.shape != state_shape
                    or not np.all(np.isfinite(prediction_array))
                ):
                    raise ValueError("evaluation prediction snapshot contract differs")
                prediction_states[method_offset, seed_offset, horizon_offset] = (
                    prediction_array
                )
                valid[method_offset, seed_offset, horizon_offset] = True
    reference_density = np.stack(reference_states)[:, :, FIELD_INDEX]
    prediction_density = prediction_states[..., FIELD_INDEX].copy()
    return reference_density, prediction_density, prediction_states, valid


def _retain_qualitative_registered_trace(
    destination: np.ndarray,
    valid: np.ndarray,
    *,
    deployment: str,
    seed: int,
    checkpoints: Mapping[int, np.ndarray],
) -> None:
    method_offset = DEPLOYMENTS.index(deployment)
    seed_offset = SEEDS.index(seed)
    for horizon_offset, horizon in enumerate(EVALUATOR_HORIZONS):
        state = checkpoints.get(horizon)
        if state is None:
            continue
        state_array = np.asarray(state)
        if (
            state_array.dtype != np.float32
            or state_array.shape != destination.shape[-2:]
            or not np.all(np.isfinite(state_array))
        ):
            raise ValueError("qualitative registered snapshot contract differs")
        destination[method_offset, seed_offset, horizon_offset] = state_array
        valid[method_offset, seed_offset, horizon_offset] = True


def _snapshot_replay_checks(
    *,
    deployment: str,
    seed: int,
    checkpoints: Mapping[int, np.ndarray],
    snapshots: Mapping[str, np.ndarray],
    references: Mapping[int, np.ndarray],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for horizon in EVALUATOR_HORIZONS:
        expected_reference = np.asarray(snapshots[f"reference_a{ANCHOR}_h{horizon}"])
        observed_reference = np.asarray(references[horizon])
        if not np.array_equal(observed_reference, expected_reference):
            raise ValueError("dataset reference differs from evaluation snapshot")
        key = _snapshot_key(deployment, seed, horizon)
        observed = checkpoints.get(horizon)
        expected = snapshots.get(key)
        checks.append(
            _snapshot_check_record(
                deployment=deployment,
                seed=seed,
                horizon=horizon,
                observed=observed,
                expected=None if expected is None else np.asarray(expected),
            )
        )
    return checks


def _reverify_replay_inputs(
    *,
    arguments: argparse.Namespace,
    successor: Mapping[str, Any],
    successor_sha256: str,
    baseline: Mapping[str, Any],
    dataset_root: Path,
    dataset_sha256: str,
    dataset_payload_sha256: str,
    evaluator_source: Mapping[str, Any],
    calibration_packet: ClosedPacket,
    evaluation_packet: ClosedPacket,
    training_packets: Sequence[TrainingPacket],
) -> None:
    _reverify_visualization_source()
    _reverify_source(evaluator_source)
    if _file_sha256(arguments.successor_contract) != successor_sha256:
        raise ValueError("successor contract changed during visualization replay")
    if _file_sha256(arguments.contract) != successor["inherited_r0_contract_sha256"]:
        raise ValueError("R0 contract changed during visualization replay")
    if _file_sha256(arguments.preregistration) != successor["preregistration_sha256"]:
        raise ValueError("preregistration changed during visualization replay")
    observed_manifest, _, _, _, observed_dataset_sha256 = _load_dataset(
        dataset_root, baseline
    )
    if (
        observed_dataset_sha256 != dataset_sha256
        or observed_manifest.get("canonical_payload_sha256") != dataset_payload_sha256
    ):
        raise ValueError("dataset packet changed during visualization replay")
    _reverify_packet(calibration_packet)
    _reverify_packet(evaluation_packet)
    for packet in training_packets:
        _reverify_packet(packet.packet)


def replay(arguments: argparse.Namespace) -> dict[str, Any]:
    """Replay all frozen deployed maps on one predeclared development anchor."""

    if arguments.population_role != "development":
        raise ProtectedPopulationError(
            "successor visualization refuses prospective/sealed roles before inspecting scientific inputs"
        )
    replay_inputs = [
        ("R0 contract", arguments.contract),
        ("successor contract", arguments.successor_contract),
        ("successor preregistration", arguments.preregistration),
        ("dataset packet", arguments.dataset_dir),
        ("calibration packet", arguments.calibration_dir),
        ("development evaluation", arguments.evaluation_dir),
    ]
    replay_inputs.extend(("training packet", path) for path in arguments.training_dir)
    _validate_output_disjoint(
        arguments.output_dir,
        replay_inputs,
        label="visualization replay output",
    )
    _reverify_visualization_source()
    evaluator_source = _source_snapshot()
    _reverify_source(evaluator_source)
    output = arguments.output_dir.resolve()
    if arguments.output_dir.is_symlink() or arguments.output_dir.parent.is_symlink():
        raise ValueError("visualization output or its parent is aliased")
    if output.exists():
        raise FileExistsError(f"visualization replay output already exists: {output}")

    successor, successor_sha256, baseline = _load_successor_contract(
        arguments.successor_contract, arguments.contract, arguments.preregistration
    )
    dataset_root = arguments.dataset_dir.resolve()
    dataset_manifest, geometry, normalization, roles, dataset_sha256 = _load_dataset(
        dataset_root, baseline
    )
    if dataset_sha256 != successor["dataset"]["final_hash_manifest_sha256"]:
        raise ValueError("dataset packet differs from the successor contract")
    train_states, train_indices, _ = roles["train"]
    development_states, development_indices, _ = roles["development"]
    _validate_population(
        train_states,
        train_indices,
        role="train",
        expected_frames=range(955, 1195),
        node_count=geometry.num_nodes,
    )
    _validate_population(
        development_states,
        development_indices,
        role="development",
        expected_frames=range(1233, 1473),
        node_count=geometry.num_nodes,
    )
    calibration_packet, calibration, projector = _load_calibration(
        arguments.calibration_dir,
        successor=successor,
        successor_sha256=successor_sha256,
        dataset_sha256=dataset_sha256,
        normalization=normalization,
        train_states=train_states,
        train_indices=train_indices,
    )
    training_packets = _load_training_packets(
        arguments.training_dir,
        successor=successor,
        successor_sha256=successor_sha256,
        dataset_sha256=dataset_sha256,
        dataset_payload_sha256=dataset_manifest["canonical_payload_sha256"],
        calibration_sha256=calibration_packet.final_file_sha256,
        calibration_payload_sha256=calibration["canonical_payload_sha256"],
        baseline=baseline,
        geometry=geometry,
    )
    evaluation_packet, _, evaluation_snapshots, summary_arrays = (
        _verify_evaluation_packet(
            arguments.evaluation_dir,
            successor=successor,
            successor_sha256=successor_sha256,
            dataset_sha256=dataset_sha256,
            dataset_payload_sha256=dataset_manifest["canonical_payload_sha256"],
            calibration_packet=calibration_packet,
            calibration=calibration,
            training_packets=training_packets,
            evaluator_source=evaluator_source,
        )
    )

    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    anchors = tuple(successor["evaluation"]["development_anchors"])
    if ANCHOR != anchors[0] or set(DEPLOYMENTS) != {
        *LEARNED_ARMS,
        "PATH_PROJECTION",
    }:
        raise ValueError(
            "frozen qualitative selection differs from the successor contract"
        )
    positions = {int(index): offset for offset, index in enumerate(development_indices)}
    if any(
        required not in positions
        for anchor in anchors
        for required in (anchor - 1, anchor, anchor + FULL_HORIZONS[-1])
    ):
        raise ValueError("development role does not contain the frozen replay")
    selected_offset = anchors.index(ANCHOR)
    previous_numpy = np.stack(
        [development_states[positions[anchor - 1]] for anchor in anchors]
    ).astype(np.float32)
    current_numpy = np.stack(
        [development_states[positions[anchor]] for anchor in anchors]
    ).astype(np.float32)
    frame_indices = np.asarray(
        [ANCHOR + horizon for horizon in FULL_HORIZONS], dtype=np.int64
    )
    references = {
        horizon: np.asarray(
            development_states[positions[ANCHOR + horizon]], dtype=np.float64
        )
        for horizon in EVALUATOR_HORIZONS
    }
    reference_density = np.stack(
        [
            development_states[positions[int(index)], :, FIELD_INDEX]
            for index in frame_indices
        ]
    ).astype(np.float32)
    (
        exact_reference_density,
        exact_predicted_density,
        evaluator_prediction_states,
        exact_valid,
    ) = _extract_exact_registered_snapshots(evaluation_snapshots)

    predicted = np.full(
        (
            len(DEPLOYMENTS),
            len(SEEDS),
            len(FULL_HORIZONS),
            geometry.num_nodes,
        ),
        np.nan,
        dtype=np.float32,
    )
    valid = np.zeros((len(DEPLOYMENTS), len(SEEDS), len(FULL_HORIZONS)), dtype=np.bool_)
    qualitative_prediction_states = np.full_like(
        evaluator_prediction_states, np.nan, dtype=np.float32
    )
    qualitative_registered_valid = np.zeros_like(exact_valid)
    first_nonfinite: dict[str, int | None] = {}
    replay_checks: list[dict[str, Any]] = []
    training_records: list[dict[str, Any]] = []
    for packet in training_packets:
        model = _instantiate_model(
            packet, baseline=baseline, geometry=geometry, device=device
        )
        method_offset = DEPLOYMENTS.index(packet.arm)
        seed_offset = SEEDS.index(packet.seed)
        density, mask, checkpoints, failed = _rollout_deployment(
            deployment=packet.arm,
            model=model,
            previous=torch.as_tensor(previous_numpy, device=device),
            current=torch.as_tensor(current_numpy, device=device),
            selected_offset=selected_offset,
            geometry=geometry,
            normalization=normalization,
            projector=None,
            device=device,
        )
        predicted[method_offset, seed_offset] = density
        valid[method_offset, seed_offset] = mask
        first_nonfinite[f"{packet.arm}:{packet.seed}"] = failed
        _retain_qualitative_registered_trace(
            qualitative_prediction_states,
            qualitative_registered_valid,
            deployment=packet.arm,
            seed=packet.seed,
            checkpoints=checkpoints,
        )
        replay_checks.extend(
            _snapshot_replay_checks(
                deployment=packet.arm,
                seed=packet.seed,
                checkpoints=checkpoints,
                snapshots=evaluation_snapshots,
                references=references,
            )
        )
        if packet.arm == "CLEAN":
            path_offset = DEPLOYMENTS.index("PATH_PROJECTION")
            density, mask, checkpoints, failed = _rollout_deployment(
                deployment="PATH_PROJECTION",
                model=model,
                previous=torch.as_tensor(previous_numpy, device=device),
                current=torch.as_tensor(current_numpy, device=device),
                selected_offset=selected_offset,
                geometry=geometry,
                normalization=normalization,
                projector=projector,
                device=device,
            )
            predicted[path_offset, seed_offset] = density
            valid[path_offset, seed_offset] = mask
            first_nonfinite[f"PATH_PROJECTION:{packet.seed}"] = failed
            _retain_qualitative_registered_trace(
                qualitative_prediction_states,
                qualitative_registered_valid,
                deployment="PATH_PROJECTION",
                seed=packet.seed,
                checkpoints=checkpoints,
            )
            replay_checks.extend(
                _snapshot_replay_checks(
                    deployment="PATH_PROJECTION",
                    seed=packet.seed,
                    checkpoints=checkpoints,
                    snapshots=evaluation_snapshots,
                    references=references,
                )
            )
        training_records.append(
            {
                "arm": packet.arm,
                "seed": packet.seed,
                "final_hash_manifest_sha256": packet.packet.final_file_sha256,
                "checkpoint_sha256": packet.checkpoint_record["sha256"],
                "source_set_sha256": packet.summary["source_set_sha256"],
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        _reverify_packet(packet.packet)

    replay_checks.sort(
        key=lambda check: (
            DEPLOYMENTS.index(str(check["deployment"])),
            SEEDS.index(int(check["seed"])),
            EVALUATOR_HORIZONS.index(int(check["horizon"])),
        )
    )
    if not np.all(valid[:, :, 0]):
        raise AssertionError("every replay must retain the common initial state")
    scale = float(np.asarray(normalization.state_scale)[FIELD_INDEX])
    signed = (predicted.astype(np.float64) - reference_density[None, None]) / scale
    errors = np.sqrt(np.mean(np.square(signed), axis=3))
    errors[~valid] = np.nan
    coordinates = np.asarray(geometry.native_coordinates, dtype=np.float64)
    quads = np.asarray(geometry.elements[:, 1:], dtype=np.int64)
    airfoil_mask = np.asarray(geometry.boundary_one_hot[:, 1], dtype=np.uint8)

    _reverify_replay_inputs(
        arguments=arguments,
        successor=successor,
        successor_sha256=successor_sha256,
        baseline=baseline,
        dataset_root=dataset_root,
        dataset_sha256=dataset_sha256,
        dataset_payload_sha256=dataset_manifest["canonical_payload_sha256"],
        evaluator_source=evaluator_source,
        calibration_packet=calibration_packet,
        evaluation_packet=evaluation_packet,
        training_packets=training_packets,
    )

    staging: Path | None = None
    owned = False
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.parent.is_symlink() or not output.parent.is_dir():
            raise ValueError("visualization replay output parent is absent or aliased")
        staging = _make_staging_directory(
            output.parent, prefix=f".{output.name}.replay-staging-"
        )
        owned = True
        np.savez_compressed(
            staging / REPLAY_FILE,
            schema=np.asarray(REPLAY_SCHEMA),
            qualitative_contract_sha256=np.asarray(QUALITATIVE_CONTRACT_SHA256),
            anchor=np.asarray(ANCHOR, dtype=np.int64),
            horizons=np.asarray(FULL_HORIZONS, dtype=np.int64),
            frame_indices=frame_indices,
            seeds=np.asarray(SEEDS, dtype=np.int64),
            deployments=np.asarray(DEPLOYMENTS),
            coordinates=coordinates,
            quads=quads,
            airfoil_mask=airfoil_mask,
            reference_density=reference_density,
            predicted_density=predicted,
            valid_mask=valid,
            registered_horizons=np.asarray(EVALUATOR_HORIZONS, dtype=np.int64),
            exact_reference_density=exact_reference_density,
            exact_predicted_density=exact_predicted_density,
            exact_valid_mask=exact_valid,
            evaluator_prediction_states=evaluator_prediction_states,
            qualitative_registered_prediction_states=(qualitative_prediction_states),
            qualitative_registered_valid_mask=qualitative_registered_valid,
            density_state_scale=np.asarray(scale, dtype=np.float64),
            density_error_rmse=errors,
            **summary_arrays,
        )
        replay_record = _file_record(staging / REPLAY_FILE, staging)
        manifest = {
            "schema": REPLAY_SCHEMA,
            "status": "complete",
            "classification": "VISUALIZATION_ONLY_REPLAY",
            "experiment_id": EXPERIMENT_ID,
            "qualitative_contract": QUALITATIVE_CONTRACT,
            "qualitative_contract_sha256": QUALITATIVE_CONTRACT_SHA256,
            "population_role": "development",
            "anchor": ANCHOR,
            "seeds": list(SEEDS),
            "deployments": list(DEPLOYMENTS),
            "field": FIELD,
            "prospective_opened": False,
            "sealed_opened": False,
            "successor_contract_sha256": successor_sha256,
            "inherited_r0_contract_sha256": successor["inherited_r0_contract_sha256"],
            "preregistration_sha256": successor["preregistration_sha256"],
            "dataset_final_hash_manifest_sha256": dataset_sha256,
            "dataset_manifest_payload_sha256": dataset_manifest[
                "canonical_payload_sha256"
            ],
            "calibration_final_hash_manifest_sha256": (
                calibration_packet.final_file_sha256
            ),
            "calibration_payload_sha256": calibration["canonical_payload_sha256"],
            "evaluation_final_hash_manifest_sha256": (
                evaluation_packet.final_file_sha256
            ),
            "training_packets": training_records,
            "source": {
                "visualization_script": dict(VISUALIZATION_SOURCE_AT_IMPORT),
                "evaluator_source_set_sha256": evaluator_source["source_set_sha256"],
            },
            "first_nonfinite_horizon": first_nonfinite,
            "registered_snapshot_replay_checks": replay_checks,
            "registered_snapshot_check_summary": _snapshot_check_summary(replay_checks),
            "exact_static_snapshot_provenance": {
                "source": "verified_evaluation_rollout_snapshots",
                "evaluation_final_hash_manifest_sha256": (
                    evaluation_packet.final_file_sha256
                ),
                "evaluation_snapshot_file": dict(
                    evaluation_packet.files["rollout_snapshots.npz"]
                ),
                "registered_horizons": list(EVALUATOR_HORIZONS),
                "field": FIELD,
                "path_projection_state_stage": "corrected",
                "continuous_qualitative_replay_used_for_static_pdfs": False,
                "array_sha256": {
                    "registered_horizons": _array_sha256(
                        np.asarray(EVALUATOR_HORIZONS, dtype=np.int64)
                    ),
                    "exact_reference_density": _array_sha256(exact_reference_density),
                    "exact_predicted_density": _array_sha256(exact_predicted_density),
                    "exact_valid_mask": _array_sha256(exact_valid),
                    "evaluator_prediction_states": _array_sha256(
                        evaluator_prediction_states
                    ),
                },
            },
            "replay_file": replay_record,
            "arrays": {
                "reference_density": list(reference_density.shape),
                "predicted_density": list(predicted.shape),
                "valid_mask": list(valid.shape),
                "registered_horizons": [len(EVALUATOR_HORIZONS)],
                "exact_reference_density": list(exact_reference_density.shape),
                "exact_predicted_density": list(exact_predicted_density.shape),
                "exact_valid_mask": list(exact_valid.shape),
                "evaluator_prediction_states": list(evaluator_prediction_states.shape),
                "qualitative_registered_prediction_states": list(
                    qualitative_prediction_states.shape
                ),
                "qualitative_registered_valid_mask": list(
                    qualitative_registered_valid.shape
                ),
                "density_error_rmse": list(errors.shape),
                "coordinates": list(coordinates.shape),
                "quads": list(quads.shape),
            },
            "claim_boundary": _claim_boundary(),
        }
        _validate_training_records(training_records)
        _validate_replay_bookkeeping(
            manifest,
            valid,
            evaluator_prediction_states=evaluator_prediction_states,
            evaluator_valid_mask=exact_valid,
            qualitative_prediction_states=qualitative_prediction_states,
            qualitative_valid_mask=qualitative_registered_valid,
        )
        manifest["canonical_payload_sha256"] = _canonical_sha256(manifest)
        _write_json(staging / REPLAY_MANIFEST_FILE, manifest)
        _reverify_replay_inputs(
            arguments=arguments,
            successor=successor,
            successor_sha256=successor_sha256,
            baseline=baseline,
            dataset_root=dataset_root,
            dataset_sha256=dataset_sha256,
            dataset_payload_sha256=dataset_manifest["canonical_payload_sha256"],
            evaluator_source=evaluator_source,
            calibration_packet=calibration_packet,
            evaluation_packet=evaluation_packet,
            training_packets=training_packets,
        )
        staged_bundle = load_replay(staging)
        if staged_bundle.manifest != manifest:
            raise ValueError("staged replay manifest differs after validation")
        _rename_directory_no_replace(staging, output)
        owned = False
    except Exception:
        if owned and staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def load_replay(
    root: Path, *, expected_visualization_source_sha256: str | None = None
) -> ReplayBundle:
    """Load a replay after verifying its manifest, file hash, and array contract."""

    expected_source_sha256 = (
        VISUALIZATION_SOURCE_AT_IMPORT["sha256"]
        if expected_visualization_source_sha256 is None
        else expected_visualization_source_sha256
    )
    if not _is_digest(expected_source_sha256):
        raise ValueError("expected replay-producer source digest is invalid")
    if root.is_symlink():
        raise ValueError("visualization replay directory is aliased")
    replay_root = root.resolve()
    if not replay_root.is_dir():
        raise ValueError("visualization replay directory is absent")
    expected_inventory = {REPLAY_MANIFEST_FILE, REPLAY_FILE}
    entries = list(replay_root.iterdir())
    if (
        {entry.name for entry in entries} != expected_inventory
        or len(entries) != len(expected_inventory)
        or any(entry.is_symlink() or not entry.is_file() for entry in entries)
    ):
        raise ValueError("visualization replay inventory differs")
    manifest_path = replay_root / REPLAY_MANIFEST_FILE
    replay_path = replay_root / REPLAY_FILE
    manifest_record = _file_record(manifest_path, replay_root)
    manifest_payload = manifest_path.read_bytes()
    if manifest_path.read_bytes() != manifest_payload:
        raise ValueError("visualization replay manifest changed while reading")
    manifest = json.loads(manifest_payload)
    if not isinstance(manifest, dict) or manifest.get("schema") != REPLAY_SCHEMA:
        raise ValueError("unsupported successor visualization replay manifest")
    observed = manifest.pop("canonical_payload_sha256", None)
    expected = _canonical_sha256(manifest)
    manifest["canonical_payload_sha256"] = observed
    if observed != expected:
        raise ValueError("visualization replay manifest self-hash differs")
    if (
        manifest.get("status") != "complete"
        or manifest.get("classification") != "VISUALIZATION_ONLY_REPLAY"
        or manifest.get("experiment_id") != EXPERIMENT_ID
        or manifest.get("qualitative_contract") != QUALITATIVE_CONTRACT
        or manifest.get("qualitative_contract_sha256") != QUALITATIVE_CONTRACT_SHA256
        or manifest.get("population_role") != "development"
        or manifest.get("anchor") != ANCHOR
        or manifest.get("seeds") != list(SEEDS)
        or manifest.get("deployments") != list(DEPLOYMENTS)
        or manifest.get("field") != FIELD
        or manifest.get("prospective_opened") is not False
        or manifest.get("sealed_opened") is not False
        or manifest.get("claim_boundary") != _claim_boundary()
    ):
        raise ValueError("visualization replay selection or claim boundary differs")
    for key in (
        "successor_contract_sha256",
        "inherited_r0_contract_sha256",
        "preregistration_sha256",
        "dataset_final_hash_manifest_sha256",
        "dataset_manifest_payload_sha256",
        "calibration_final_hash_manifest_sha256",
        "calibration_payload_sha256",
        "evaluation_final_hash_manifest_sha256",
    ):
        if not _is_digest(manifest.get(key)):
            raise ValueError(f"visualization upstream digest is invalid: {key}")
    source = manifest.get("source")
    replay_visualization_source = (
        source.get("visualization_script") if isinstance(source, Mapping) else None
    )
    if (
        not isinstance(source, Mapping)
        or set(source) != {"visualization_script", "evaluator_source_set_sha256"}
        or not isinstance(replay_visualization_source, Mapping)
        or set(replay_visualization_source) != {"bytes", "sha256"}
        or isinstance(replay_visualization_source.get("bytes"), bool)
        or not isinstance(replay_visualization_source.get("bytes"), int)
        or replay_visualization_source["bytes"] <= 0
        or not _is_digest(replay_visualization_source.get("sha256"))
        or replay_visualization_source["sha256"] != expected_source_sha256
        or not _is_digest(source.get("evaluator_source_set_sha256"))
    ):
        raise ValueError("visualization source closure differs")
    _validate_training_records(manifest.get("training_packets"))
    record = manifest.get("replay_file")
    if (
        not isinstance(record, Mapping)
        or record.get("relative_path") != REPLAY_FILE
        or not replay_path.is_file()
        or replay_path.stat().st_size != record.get("bytes")
        or _file_sha256(replay_path) != record.get("sha256")
    ):
        raise ValueError("visualization replay file differs from its manifest")

    with np.load(replay_path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    expected_names = {
        "schema",
        "qualitative_contract_sha256",
        "anchor",
        "horizons",
        "frame_indices",
        "seeds",
        "deployments",
        "coordinates",
        "quads",
        "airfoil_mask",
        "reference_density",
        "predicted_density",
        "valid_mask",
        "registered_horizons",
        "exact_reference_density",
        "exact_predicted_density",
        "exact_valid_mask",
        "evaluator_prediction_states",
        "qualitative_registered_prediction_states",
        "qualitative_registered_valid_mask",
        "density_state_scale",
        "density_error_rmse",
        "response_gain",
        "rollout_auc",
        "late_state_error",
        "late_path_distance",
        "late_projected_path_discrepancy",
    }
    if set(arrays) != expected_names:
        raise ValueError("visualization replay NPZ inventory differs")
    anchor = int(np.asarray(arrays["anchor"]).item())
    horizons = np.asarray(arrays["horizons"])
    frame_indices = np.asarray(arrays["frame_indices"])
    seeds = np.asarray(arrays["seeds"])
    deployments_array = np.asarray(arrays["deployments"])
    deployments = tuple(str(value) for value in deployments_array.tolist())
    coordinates = np.asarray(arrays["coordinates"])
    quads = np.asarray(arrays["quads"])
    airfoil_mask = np.asarray(arrays["airfoil_mask"])
    reference = np.asarray(arrays["reference_density"])
    predicted = np.asarray(arrays["predicted_density"])
    valid = np.asarray(arrays["valid_mask"])
    registered_horizons = np.asarray(arrays["registered_horizons"])
    exact_reference = np.asarray(arrays["exact_reference_density"])
    exact_predicted = np.asarray(arrays["exact_predicted_density"])
    exact_valid = np.asarray(arrays["exact_valid_mask"])
    evaluator_states = np.asarray(arrays["evaluator_prediction_states"])
    qualitative_states = np.asarray(arrays["qualitative_registered_prediction_states"])
    qualitative_registered_valid = np.asarray(
        arrays["qualitative_registered_valid_mask"]
    )
    scale = float(np.asarray(arrays["density_state_scale"]).item())
    errors = np.asarray(arrays["density_error_rmse"])
    summary_shapes = {
        "response_gain": (len(DEPLOYMENTS), len(SEEDS)),
        "rollout_auc": (len(DEPLOYMENTS), len(SEEDS)),
        "late_state_error": (len(DEPLOYMENTS), len(SEEDS)),
        "late_path_distance": (len(DEPLOYMENTS), len(SEEDS)),
        "late_projected_path_discrepancy": (len(DEPLOYMENTS), len(SEEDS)),
    }
    summaries = {name: np.asarray(arrays[name]) for name in summary_shapes}
    node_count = coordinates.shape[0] if coordinates.ndim == 2 else -1
    if (
        np.asarray(arrays["schema"]).item() != REPLAY_SCHEMA
        or np.asarray(arrays["qualitative_contract_sha256"]).item()
        != QUALITATIVE_CONTRACT_SHA256
        or anchor != ANCHOR
        or horizons.dtype != np.int64
        or not np.array_equal(horizons, np.asarray(FULL_HORIZONS, dtype=np.int64))
        or frame_indices.dtype != np.int64
        or not np.array_equal(frame_indices, ANCHOR + horizons)
        or seeds.dtype != np.int64
        or not np.array_equal(seeds, np.asarray(SEEDS, dtype=np.int64))
        or deployments_array.dtype.kind != "U"
        or deployments != DEPLOYMENTS
        or coordinates.dtype != np.float64
        or coordinates.shape != (node_count, 2)
        or not np.all(np.isfinite(coordinates))
        or quads.dtype != np.int64
        or quads.ndim != 2
        or quads.shape[1] != 4
        or np.any(quads < 0)
        or np.any(quads >= node_count)
        or airfoil_mask.dtype != np.uint8
        or airfoil_mask.shape != (node_count,)
        or not np.all((airfoil_mask == 0) | (airfoil_mask == 1))
        or np.count_nonzero(airfoil_mask) == 0
        or reference.dtype != np.float32
        or reference.shape != (len(FULL_HORIZONS), node_count)
        or not np.all(np.isfinite(reference))
        or predicted.dtype != np.float32
        or predicted.shape
        != (len(DEPLOYMENTS), len(SEEDS), len(FULL_HORIZONS), node_count)
        or valid.dtype != np.bool_
        or valid.shape != (len(DEPLOYMENTS), len(SEEDS), len(FULL_HORIZONS))
        or not np.all(valid[:, :, 0])
        or registered_horizons.dtype != np.int64
        or not np.array_equal(
            registered_horizons, np.asarray(EVALUATOR_HORIZONS, dtype=np.int64)
        )
        or exact_reference.dtype != np.float64
        or exact_reference.shape != (len(EVALUATOR_HORIZONS), node_count)
        or not np.all(np.isfinite(exact_reference))
        or exact_predicted.dtype != np.float32
        or exact_predicted.shape
        != (len(DEPLOYMENTS), len(SEEDS), len(EVALUATOR_HORIZONS), node_count)
        or exact_valid.dtype != np.bool_
        or exact_valid.shape != (len(DEPLOYMENTS), len(SEEDS), len(EVALUATOR_HORIZONS))
        or evaluator_states.dtype != np.float32
        or evaluator_states.ndim != 5
        or evaluator_states.shape[:4]
        != (len(DEPLOYMENTS), len(SEEDS), len(EVALUATOR_HORIZONS), node_count)
        or evaluator_states.shape[4] <= FIELD_INDEX
        or qualitative_states.dtype != np.float32
        or qualitative_states.shape != evaluator_states.shape
        or qualitative_registered_valid.dtype != np.bool_
        or qualitative_registered_valid.shape != exact_valid.shape
        or errors.dtype != np.float64
        or errors.shape != (len(DEPLOYMENTS), len(SEEDS), len(FULL_HORIZONS))
        or not np.isfinite(scale)
        or scale <= 0.0
    ):
        raise ValueError("visualization replay array contract differs")
    if (
        np.any(np.isfinite(exact_predicted[~exact_valid]))
        or np.any(~np.isfinite(exact_predicted[exact_valid]))
        or np.any(np.isfinite(evaluator_states[~exact_valid]))
        or np.any(~np.isfinite(evaluator_states[exact_valid]))
        or np.any(np.isfinite(qualitative_states[~qualitative_registered_valid]))
        or np.any(~np.isfinite(qualitative_states[qualitative_registered_valid]))
        or not np.array_equal(
            exact_predicted,
            evaluator_states[..., FIELD_INDEX],
            equal_nan=True,
        )
        or not np.array_equal(
            qualitative_registered_valid,
            valid[:, :, registered_horizons],
        )
        or not np.array_equal(
            qualitative_states[..., FIELD_INDEX],
            predicted[:, :, registered_horizons],
            equal_nan=True,
        )
        or not np.array_equal(
            exact_reference.astype(np.float32), reference[registered_horizons]
        )
    ):
        raise ValueError("registered evaluator/qualitative snapshot arrays differ")
    for name, shape in summary_shapes.items():
        array = summaries[name]
        if array.dtype != np.float64 or array.shape != shape:
            raise ValueError(f"visualization diagnostic summary differs: {name}")
        if name == "response_gain":
            if not np.all(np.isnan(array[-1])) or np.any(np.isnan(array[:-1])):
                raise ValueError("response-gain availability differs by method")
        elif np.any(np.isnan(array)):
            raise ValueError(f"visualization diagnostic contains NaN: {name}")
        finite = array[np.isfinite(array)]
        if np.any(finite < 0.0):
            raise ValueError(f"visualization diagnostic is negative: {name}")
    if np.any(np.isfinite(predicted[~valid])) or np.any(~np.isfinite(predicted[valid])):
        raise ValueError("prediction finiteness does not match the valid mask")
    for trace in valid.reshape(-1, len(FULL_HORIZONS)):
        if np.any(np.diff(trace.astype(np.int8)) > 0):
            raise ValueError("a nonfinite replay trace became valid again")
    recomputed = np.sqrt(
        np.mean(
            np.square((predicted.astype(np.float64) - reference[None, None]) / scale),
            axis=3,
        )
    )
    recomputed[~valid] = np.nan
    if not np.allclose(errors, recomputed, rtol=1.0e-13, atol=1.0e-15, equal_nan=True):
        raise ValueError("stored density-error traces differ from replay states")
    expected_shapes = {
        "reference_density": list(reference.shape),
        "predicted_density": list(predicted.shape),
        "valid_mask": list(valid.shape),
        "registered_horizons": list(registered_horizons.shape),
        "exact_reference_density": list(exact_reference.shape),
        "exact_predicted_density": list(exact_predicted.shape),
        "exact_valid_mask": list(exact_valid.shape),
        "evaluator_prediction_states": list(evaluator_states.shape),
        "qualitative_registered_prediction_states": list(qualitative_states.shape),
        "qualitative_registered_valid_mask": list(qualitative_registered_valid.shape),
        "density_error_rmse": list(errors.shape),
        "coordinates": list(coordinates.shape),
        "quads": list(quads.shape),
    }
    if manifest.get("arrays") != expected_shapes:
        raise ValueError("visualization replay shape inventory differs")
    provenance = manifest.get("exact_static_snapshot_provenance")
    if not isinstance(provenance, Mapping) or set(provenance) != {
        "source",
        "evaluation_final_hash_manifest_sha256",
        "evaluation_snapshot_file",
        "registered_horizons",
        "field",
        "path_projection_state_stage",
        "continuous_qualitative_replay_used_for_static_pdfs",
        "array_sha256",
    }:
        raise ValueError("exact static-snapshot provenance differs")
    snapshot_file = provenance.get("evaluation_snapshot_file")
    expected_array_hashes = {
        "registered_horizons": _array_sha256(registered_horizons),
        "exact_reference_density": _array_sha256(exact_reference),
        "exact_predicted_density": _array_sha256(exact_predicted),
        "exact_valid_mask": _array_sha256(exact_valid),
        "evaluator_prediction_states": _array_sha256(evaluator_states),
    }
    if (
        provenance.get("source") != "verified_evaluation_rollout_snapshots"
        or provenance.get("evaluation_final_hash_manifest_sha256")
        != manifest["evaluation_final_hash_manifest_sha256"]
        or not isinstance(snapshot_file, Mapping)
        or snapshot_file.get("relative_path") != "rollout_snapshots.npz"
        or isinstance(snapshot_file.get("bytes"), bool)
        or not isinstance(snapshot_file.get("bytes"), int)
        or snapshot_file["bytes"] <= 0
        or not _is_digest(snapshot_file.get("sha256"))
        or provenance.get("registered_horizons") != list(EVALUATOR_HORIZONS)
        or provenance.get("field") != FIELD
        or provenance.get("path_projection_state_stage") != "corrected"
        or provenance.get("continuous_qualitative_replay_used_for_static_pdfs")
        is not False
        or provenance.get("array_sha256") != expected_array_hashes
    ):
        raise ValueError("exact static-snapshot provenance binding differs")
    _validate_replay_bookkeeping(
        manifest,
        valid,
        evaluator_prediction_states=evaluator_states,
        evaluator_valid_mask=exact_valid,
        qualitative_prediction_states=qualitative_states,
        qualitative_valid_mask=qualitative_registered_valid,
    )
    entries = list(replay_root.iterdir())
    if (
        {entry.name for entry in entries} != expected_inventory
        or len(entries) != len(expected_inventory)
        or any(entry.is_symlink() or not entry.is_file() for entry in entries)
    ):
        raise ValueError("visualization replay inventory changed while loading")
    _reverify_file_record(manifest_path, replay_root, manifest_record)
    _reverify_file_record(replay_path, replay_root, record)
    return ReplayBundle(
        anchor=anchor,
        horizons=horizons,
        frame_indices=frame_indices,
        seeds=seeds,
        deployments=deployments,
        coordinates=coordinates,
        quads=quads,
        airfoil_mask=airfoil_mask,
        reference_density=reference,
        predicted_density=predicted,
        valid_mask=valid,
        registered_horizons=registered_horizons,
        exact_reference_density=exact_reference,
        exact_predicted_density=exact_predicted,
        exact_valid_mask=exact_valid,
        density_state_scale=scale,
        density_error_rmse=errors,
        response_gain=summaries["response_gain"],
        rollout_auc=summaries["rollout_auc"],
        late_state_error=summaries["late_state_error"],
        late_path_distance=summaries["late_path_distance"],
        late_projected_path_discrepancy=summaries["late_projected_path_discrepancy"],
        manifest=manifest,
    )


def _visible_quads(bundle: ReplayBundle) -> np.ndarray:
    vertices = bundle.coordinates[bundle.quads]
    overlaps = (
        (np.max(vertices[:, :, 0], axis=1) >= X_LIMITS[0])
        & (np.min(vertices[:, :, 0], axis=1) <= X_LIMITS[1])
        & (np.max(vertices[:, :, 1], axis=1) >= Y_LIMITS[0])
        & (np.min(vertices[:, :, 1], axis=1) <= Y_LIMITS[1])
    )
    selected = bundle.quads[overlaps]
    if selected.size == 0:
        raise ValueError("fixed near-body/wake view contains no native quads")
    return selected


def _quad_face_values(node_values: np.ndarray, quads: np.ndarray) -> np.ndarray:
    values = np.asarray(node_values)
    if values.ndim != 1 or np.any(quads < 0) or np.any(quads >= values.size):
        raise ValueError("node values and native quads do not align")
    return np.mean(values[quads], axis=1)


def _exact_static_density(
    bundle: ReplayBundle, *, horizon: int, seed_offset: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return only evaluator-owned density fields for one static paper row."""

    if horizon not in STATIC_HORIZONS:
        raise ValueError("static density horizon is not predeclared")
    if not 0 <= seed_offset < len(SEEDS):
        raise ValueError("static density seed offset is invalid")
    matches = np.flatnonzero(bundle.registered_horizons == horizon)
    if matches.size != 1:
        raise ValueError("static density horizon lacks one evaluator snapshot")
    horizon_offset = int(matches[0])
    return (
        bundle.exact_reference_density[horizon_offset],
        bundle.exact_predicted_density[:, seed_offset, horizon_offset],
        bundle.exact_valid_mask[:, seed_offset, horizon_offset],
    )


def color_limits(
    bundle: ReplayBundle, quads: np.ndarray
) -> dict[str, tuple[float, float]]:
    """Use one fixed scale over all methods, seeds, and displayed horizons."""

    nodes = np.unique(quads)
    valid_prediction = np.where(
        bundle.valid_mask[:, :, :, None],
        bundle.predicted_density,
        np.nan,
    )
    density_min = float(
        min(
            np.min(bundle.reference_density[:, nodes]),
            np.nanmin(valid_prediction[:, :, :, nodes]),
        )
    )
    density_max = float(
        max(
            np.max(bundle.reference_density[:, nodes]),
            np.nanmax(valid_prediction[:, :, :, nodes]),
        )
    )
    signed = (
        valid_prediction[:, :, :, nodes].astype(np.float64)
        - bundle.reference_density[None, None, :, nodes]
    ) / bundle.density_state_scale
    error_max = max(float(np.nanmax(np.abs(signed))), 1.0e-12)
    curve_max = max(float(np.nanmax(bundle.density_error_rmse)), 1.0e-12)
    if not density_max > density_min:
        density_max = density_min + 1.0e-12
    return {
        "density": (density_min, density_max),
        "signed_error": (-error_max, error_max),
        "error_curve": (0.0, 1.05 * curve_max),
    }


def static_color_limits(
    bundle: ReplayBundle, quads: np.ndarray
) -> dict[str, tuple[float, float]]:
    """Fix static scales using exact evaluator snapshots and no rerun values."""

    nodes = np.unique(quads)
    offsets = [
        int(np.flatnonzero(bundle.registered_horizons == horizon)[0])
        for horizon in STATIC_HORIZONS
    ]
    reference = bundle.exact_reference_density[offsets][:, nodes]
    prediction = np.where(
        bundle.exact_valid_mask[:, :, offsets, None],
        bundle.exact_predicted_density[:, :, offsets][:, :, :, nodes],
        np.nan,
    )
    finite_prediction = prediction[np.isfinite(prediction)]
    density_min = float(np.min(reference))
    density_max = float(np.max(reference))
    if finite_prediction.size:
        density_min = min(density_min, float(np.min(finite_prediction)))
        density_max = max(density_max, float(np.max(finite_prediction)))
    signed = (
        prediction.astype(np.float64) - reference[None, None]
    ) / bundle.density_state_scale
    finite_signed = np.abs(signed[np.isfinite(signed)])
    error_max = (
        max(float(np.max(finite_signed)), 1.0e-12) if finite_signed.size else 1.0e-12
    )
    if not density_max > density_min:
        density_max = density_min + 1.0e-12
    return {
        "density": (density_min, density_max),
        "signed_error": (-error_max, error_max),
    }


def animation_horizons(frame_stride: int) -> tuple[int, ...]:
    if isinstance(frame_stride, bool) or frame_stride < 1:
        raise ValueError("frame stride must be a positive integer")
    selected = list(range(FULL_HORIZONS[0], FULL_HORIZONS[-1] + 1, frame_stride))
    if selected[-1] != FULL_HORIZONS[-1]:
        selected.append(FULL_HORIZONS[-1])
    return tuple(selected)


def _plot_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _error_cmap_and_norm(error_max: float):
    from matplotlib.colors import LinearSegmentedColormap, SymLogNorm

    linthresh = max(0.02 * error_max, 1.0e-12)
    cmap = LinearSegmentedColormap.from_list(
        "okabe_ito_blue_white_orange", ("#0072B2", "#FFFFFF", "#E69F00")
    )
    norm = SymLogNorm(
        linthresh=linthresh,
        linscale=1.0,
        vmin=-error_max,
        vmax=error_max,
        base=10,
    )
    return cmap, norm, linthresh


def _airfoil_outline(bundle: ReplayBundle) -> np.ndarray:
    airfoil = bundle.coordinates[bundle.airfoil_mask.astype(bool)]
    centroid = np.mean(airfoil, axis=0)
    angles = np.arctan2(airfoil[:, 1] - centroid[1], airfoil[:, 0] - centroid[0])
    ordered = airfoil[np.argsort(angles)]
    return np.vstack((ordered, ordered[0]))


def _format_field_axis(axis: Any, outline: np.ndarray) -> None:
    axis.set_xlim(*X_LIMITS)
    axis.set_ylim(*Y_LIMITS)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.plot(outline[:, 0], outline[:, 1], color="#1A1A1A", linewidth=0.65)


def _density_snapshot(
    bundle: ReplayBundle,
    *,
    horizon: int,
    quads: np.ndarray,
    limits: Mapping[str, tuple[float, float]],
):
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import Normalize

    vertices = bundle.coordinates[quads]
    outline = _airfoil_outline(bundle)
    norm = Normalize(*limits["density"])
    figure, axes = plt.subplots(
        len(SEEDS), len(DEPLOYMENTS) + 1, figsize=(13.2, 6.0), constrained_layout=True
    )
    for seed_offset, seed in enumerate(SEEDS):
        reference, predictions, _ = _exact_static_density(
            bundle, horizon=horizon, seed_offset=seed_offset
        )
        values = [
            reference,
            *predictions,
        ]
        labels = ["Reference", *(DISPLAY_NAMES[name] for name in DEPLOYMENTS)]
        for column, (field, label) in enumerate(zip(values, labels, strict=True)):
            axis = axes[seed_offset, column]
            artist = PolyCollection(
                vertices,
                array=np.ma.masked_invalid(_quad_face_values(field, quads)),
                cmap="viridis",
                norm=norm,
                edgecolors="none",
                antialiased=False,
                rasterized=True,
            )
            axis.add_collection(artist)
            _format_field_axis(axis, outline)
            if seed_offset == 0:
                axis.set_title(label)
            if column == 0:
                axis.set_ylabel(f"seed {seed}")
    figure.colorbar(
        ScalarMappable(norm=norm, cmap="viridis"),
        ax=list(axes.flat),
        label=r"Density $\rho$",
        fraction=0.014,
        pad=0.012,
    )
    return figure


def _error_snapshot(
    bundle: ReplayBundle,
    *,
    horizon: int,
    quads: np.ndarray,
    limits: Mapping[str, tuple[float, float]],
):
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import PolyCollection

    vertices = bundle.coordinates[quads]
    outline = _airfoil_outline(bundle)
    error_max = limits["signed_error"][1]
    cmap, norm, _ = _error_cmap_and_norm(error_max)
    figure, axes = plt.subplots(
        len(SEEDS), len(DEPLOYMENTS), figsize=(11.5, 6.0), constrained_layout=True
    )
    for seed_offset, seed in enumerate(SEEDS):
        reference, predictions, _ = _exact_static_density(
            bundle, horizon=horizon, seed_offset=seed_offset
        )
        for method_offset, deployment in enumerate(DEPLOYMENTS):
            axis = axes[seed_offset, method_offset]
            error = (
                predictions[method_offset] - reference
            ) / bundle.density_state_scale
            artist = PolyCollection(
                vertices,
                array=np.ma.masked_invalid(_quad_face_values(error, quads)),
                cmap=cmap,
                norm=norm,
                edgecolors="none",
                antialiased=False,
                rasterized=True,
            )
            axis.add_collection(artist)
            _format_field_axis(axis, outline)
            if seed_offset == 0:
                axis.set_title(DISPLAY_NAMES[deployment])
            if method_offset == 0:
                axis.set_ylabel(f"seed {seed}")
    figure.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=list(axes.flat),
        label=r"Signed normalized density error $(\hat\rho-\rho)/s_\rho$",
        fraction=0.016,
        pad=0.012,
    )
    return figure


def _error_trace_panel(bundle: ReplayBundle):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        1, len(DEPLOYMENTS), figsize=(12.0, 2.6), sharex=True, sharey=True
    )
    for method_offset, deployment in enumerate(DEPLOYMENTS):
        axis = axes[method_offset]
        for seed_offset, (seed, color) in enumerate(
            zip(SEEDS, SEED_COLORS, strict=True)
        ):
            axis.plot(
                bundle.horizons[1:],
                bundle.density_error_rmse[method_offset, seed_offset, 1:],
                color=color,
                linewidth=1.25,
                label=f"seed {seed}",
            )
        axis.set_yscale("log")
        axis.set_xlim(1, FULL_HORIZONS[-1])
        axis.grid(color="#D9D9D9", linewidth=0.5, alpha=0.75)
        axis.set_title(DISPLAY_NAMES[deployment])
        axis.set_xlabel("Horizon step")
    axes[0].set_ylabel(r"Density RMSE $\|\hat\rho-\rho\|/s_\rho$")
    axes[0].legend(loc="best", frameon=False)
    figure.tight_layout()
    return figure


def _tradeoff_panel(bundle: ReplayBundle):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure, axes = plt.subplots(1, 3, figsize=(10.4, 3.0))
    omitted: dict[str, list[str]] = {
        "response_rollout": [],
        "path_state_accuracy": [],
        "path_phase_accuracy": [],
    }
    for method_offset, deployment in enumerate(DEPLOYMENTS):
        for seed_offset, seed in enumerate(SEEDS):
            color = METHOD_COLORS[deployment]
            marker = SEED_MARKERS[seed_offset]
            if deployment != "PATH_PROJECTION":
                x_value = bundle.response_gain[method_offset, seed_offset]
                y_value = bundle.rollout_auc[method_offset, seed_offset]
                if np.isfinite(x_value) and np.isfinite(y_value):
                    axes[0].scatter(
                        x_value,
                        y_value,
                        color=color,
                        marker=marker,
                        s=32,
                        edgecolors="white",
                        linewidths=0.4,
                    )
                else:
                    omitted["response_rollout"].append(f"{deployment}:{seed}")
            x_value = bundle.late_path_distance[method_offset, seed_offset]
            y_value = bundle.late_state_error[method_offset, seed_offset]
            if np.isfinite(x_value) and np.isfinite(y_value):
                axes[1].scatter(
                    x_value,
                    y_value,
                    color=color,
                    marker=marker,
                    s=32,
                    edgecolors="white",
                    linewidths=0.4,
                )
            else:
                omitted["path_state_accuracy"].append(f"{deployment}:{seed}")
            y_value = bundle.late_projected_path_discrepancy[method_offset, seed_offset]
            if np.isfinite(x_value) and np.isfinite(y_value):
                axes[2].scatter(
                    x_value,
                    y_value,
                    color=color,
                    marker=marker,
                    s=32,
                    edgecolors="white",
                    linewidths=0.4,
                )
            else:
                omitted["path_phase_accuracy"].append(f"{deployment}:{seed}")
    axes[0].set_xlabel("Median one-prefix response gain")
    axes[0].set_ylabel("Median normalized rollout-error AUC")
    axes[1].set_xlabel("Median late-window path distance")
    axes[1].set_ylabel("Median late-window state error")
    axes[2].set_xlabel("Median late-window path distance")
    axes[2].set_ylabel("Median projected-path discrepancy")
    for label, axis in zip(("(a)", "(b)", "(c)"), axes, strict=True):
        axis.text(-0.14, 1.03, label, transform=axis.transAxes, fontweight="bold")
        axis.grid(color="#D9D9D9", linewidth=0.5, alpha=0.75)
    for axis, key in zip(axes, omitted, strict=True):
        if omitted[key]:
            axis.text(
                0.02,
                0.02,
                f"nonfinite pairs omitted: {len(omitted[key])}",
                transform=axis.transAxes,
                fontsize=7,
            )
    method_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=METHOD_COLORS[method],
            label=DISPLAY_NAMES[method],
        )
        for method in DEPLOYMENTS
    ]
    seed_handles = [
        Line2D(
            [],
            [],
            marker=marker,
            linestyle="",
            color="#555555",
            label=f"seed {seed}",
        )
        for seed, marker in zip(SEEDS, SEED_MARKERS, strict=True)
    ]
    figure.legend(
        handles=[*method_handles, *seed_handles],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=4,
        frameon=False,
    )
    figure.tight_layout(rect=(0.0, 0.16, 1.0, 1.0))
    return figure, omitted


def _animation_figure(
    bundle: ReplayBundle,
    *,
    seed_offset: int,
    quads: np.ndarray,
    limits: Mapping[str, tuple[float, float]],
):
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import Normalize

    vertices = bundle.coordinates[quads]
    outline = _airfoil_outline(bundle)
    norm = Normalize(*limits["density"])
    figure, axes = plt.subplots(2, 3, figsize=(10.4, 6.0), constrained_layout=True)
    labels = ["Reference", *(DISPLAY_NAMES[name] for name in DEPLOYMENTS)]
    artists = []
    invalid_labels = []
    for axis, label in zip(axes.flat, labels, strict=True):
        artist = PolyCollection(
            vertices,
            array=np.zeros(quads.shape[0]),
            cmap="viridis",
            norm=norm,
            edgecolors="none",
            antialiased=False,
        )
        axis.add_collection(artist)
        axis.set_title(label)
        _format_field_axis(axis, outline)
        invalid = axis.text(
            0.5,
            0.5,
            "nonfinite",
            transform=axis.transAxes,
            ha="center",
            va="center",
            color="#1A1A1A",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
        )
        invalid.set_visible(False)
        artists.append(artist)
        invalid_labels.append(invalid)
    figure.colorbar(
        ScalarMappable(norm=norm, cmap="viridis"),
        ax=list(axes.flat),
        label=r"Density $\rho$",
        fraction=0.018,
        pad=0.012,
    )
    frame_label = figure.text(0.985, 0.015, "", ha="right", va="bottom")

    def update(horizon: int):
        fields = [
            bundle.reference_density[horizon],
            *bundle.predicted_density[:, seed_offset, horizon],
        ]
        validity = [True, *bundle.valid_mask[:, seed_offset, horizon].tolist()]
        for artist, invalid, field, is_valid in zip(
            artists, invalid_labels, fields, validity, strict=True
        ):
            artist.set_array(
                np.ma.masked_invalid(_quad_face_values(field, quads))
                if is_valid
                else np.ma.masked_all(quads.shape[0])
            )
            invalid.set_visible(not is_valid)
        frame_label.set_text(f"h = {horizon:03d}")
        return [*artists, *invalid_labels, frame_label]

    update(0)
    return figure, update


def _verify_staged_render_packet(
    staging: Path,
    manifest: Mapping[str, Any],
    *,
    replay_manifest_record: Mapping[str, Any],
    replay_file_record: Mapping[str, Any],
) -> None:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping) or not all(
        isinstance(name, str)
        and Path(name).name == name
        and isinstance(record, Mapping)
        for name, record in outputs.items()
    ):
        raise ValueError("staged visualization output records are invalid")
    expected_names = {
        REPLAY_MANIFEST_FILE,
        REPLAY_FILE,
        RENDER_MANIFEST_FILE,
        *outputs,
    }
    if len(expected_names) != len(outputs) + 3:
        raise ValueError("staged visualization output names collide")
    entries = list(staging.iterdir())
    if (
        {entry.name for entry in entries} != expected_names
        or len(entries) != len(expected_names)
        or any(entry.is_symlink() or not entry.is_file() for entry in entries)
    ):
        raise ValueError("staged visualization inventory differs")
    if manifest.get("replay_manifest") != dict(replay_manifest_record) or manifest.get(
        "replay_file"
    ) != dict(replay_file_record):
        raise ValueError("staged replay provenance records differ")

    manifest_path = staging / RENDER_MANIFEST_FILE
    payload = manifest_path.read_bytes()
    if manifest_path.read_bytes() != payload:
        raise ValueError("staged visualization manifest changed while reading")
    expected_payload = (
        json.dumps(dict(manifest), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    if payload != expected_payload:
        raise ValueError("staged visualization manifest bytes differ")
    loaded = json.loads(payload)
    if not isinstance(loaded, dict) or loaded != dict(manifest):
        raise ValueError("staged visualization manifest content differs")
    unsigned = dict(loaded)
    digest = unsigned.pop("canonical_payload_sha256", None)
    if digest != _canonical_sha256(unsigned):
        raise ValueError("staged visualization manifest self hash differs")
    entries = list(staging.iterdir())
    if (
        {entry.name for entry in entries} != expected_names
        or len(entries) != len(expected_names)
        or any(entry.is_symlink() or not entry.is_file() for entry in entries)
    ):
        raise ValueError("staged visualization inventory changed during validation")
    _reverify_file_record(
        staging / REPLAY_MANIFEST_FILE, staging, replay_manifest_record
    )
    _reverify_file_record(staging / REPLAY_FILE, staging, replay_file_record)
    for name, record in outputs.items():
        _reverify_file_record(staging / name, staging, record)
    if manifest_path.read_bytes() != payload:
        raise ValueError("staged visualization manifest changed during validation")


def _publish_directory_atomically(
    output: Path,
    build: Callable[[Path], dict[str, Any]],
    *,
    before_publish: Callable[[Path, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Build a complete directory beside ``output`` and publish it once."""

    if output.exists() or output.is_symlink():
        raise FileExistsError(f"visualization output already exists: {output}")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise ValueError("visualization output parent is absent or aliased")
    staging = _make_staging_directory(
        output.parent, prefix=f".{output.name}.render-staging-"
    )
    owned = True
    try:
        manifest = build(staging)
        if before_publish is not None:
            before_publish(staging, manifest)
        _rename_directory_no_replace(staging, output)
        owned = False
    except Exception:
        if owned:
            shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def render(arguments: argparse.Namespace) -> dict[str, Any]:
    """Render three all-method animations and the frozen paper-useful panels."""

    _validate_output_disjoint(
        arguments.output_dir,
        [("visualization replay input", arguments.replay_dir)],
        label="visualization render output",
    )
    import matplotlib as mpl

    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation

    _reverify_visualization_source()
    replay_root = arguments.replay_dir.resolve()
    if not replay_root.is_dir() or arguments.replay_dir.is_symlink():
        raise ValueError("render requires an existing unaliased replay directory")
    if arguments.output_dir.is_symlink() or arguments.output_dir.parent.is_symlink():
        raise ValueError("visualization output or its parent is aliased")
    output = arguments.output_dir.resolve()
    replay_manifest_record = _file_record(
        replay_root / REPLAY_MANIFEST_FILE, replay_root
    )
    replay_file_record = _file_record(replay_root / REPLAY_FILE, replay_root)
    bundle = load_replay(
        replay_root,
        expected_visualization_source_sha256=(
            arguments.expected_replay_producer_sha256
        ),
    )
    _reverify_file_record(
        replay_root / REPLAY_MANIFEST_FILE, replay_root, replay_manifest_record
    )
    _reverify_file_record(replay_root / REPLAY_FILE, replay_root, replay_file_record)
    if arguments.ffmpeg is not None:
        ffmpeg = arguments.ffmpeg.resolve()
        if ffmpeg.is_symlink() or not ffmpeg.is_file():
            raise ValueError("ffmpeg path is absent or aliased")
        mpl.rcParams["animation.ffmpeg_path"] = str(ffmpeg)
    if not FFMpegWriter.isAvailable():
        raise RuntimeError("Matplotlib cannot locate ffmpeg")
    _plot_style()
    quads = _visible_quads(bundle)
    qualitative_limits = color_limits(bundle, quads)
    exact_static_limits = static_color_limits(bundle, quads)
    selected_animation_horizons = animation_horizons(arguments.frame_stride)

    def build(staging: Path) -> dict[str, Any]:
        shutil.copyfile(
            replay_root / REPLAY_MANIFEST_FILE, staging / REPLAY_MANIFEST_FILE
        )
        shutil.copyfile(replay_root / REPLAY_FILE, staging / REPLAY_FILE)
        if (
            _file_record(staging / REPLAY_MANIFEST_FILE, staging)
            != replay_manifest_record
            or _file_record(staging / REPLAY_FILE, staging) != replay_file_record
        ):
            raise ValueError("copied replay packet differs from its verified input")
        names: list[str] = []
        diagnostic_omissions: dict[str, list[str]] = {}
        for seed_offset, seed in enumerate(SEEDS):
            figure, update = _animation_figure(
                bundle,
                seed_offset=seed_offset,
                quads=quads,
                limits=qualitative_limits,
            )
            animation = FuncAnimation(
                figure,
                update,
                frames=selected_animation_horizons,
                interval=1000.0 / arguments.fps,
                blit=False,
                repeat=True,
            )
            name = f"density_deployments_seed{seed}.mp4"
            writer = FFMpegWriter(
                fps=arguments.fps,
                codec="libx264",
                bitrate=arguments.bitrate,
                extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
            )
            animation.save(staging / name, writer=writer, dpi=arguments.dpi)
            plt.close(figure)
            names.append(name)
        for horizon in STATIC_HORIZONS:
            density_figure = _density_snapshot(
                bundle, horizon=horizon, quads=quads, limits=exact_static_limits
            )
            density_name = f"density_deployments_h{horizon:03d}.pdf"
            density_figure.savefig(staging / density_name, format="pdf", dpi=300)
            plt.close(density_figure)
            names.append(density_name)
            error_figure = _error_snapshot(
                bundle, horizon=horizon, quads=quads, limits=exact_static_limits
            )
            error_name = f"density_signed_error_h{horizon:03d}.pdf"
            error_figure.savefig(staging / error_name, format="pdf", dpi=300)
            plt.close(error_figure)
            names.append(error_name)
        trace_figure = _error_trace_panel(bundle)
        trace_name = "density_error_traces.pdf"
        trace_figure.savefig(staging / trace_name, format="pdf", dpi=300)
        plt.close(trace_figure)
        names.append(trace_name)
        tradeoff_figure, diagnostic_omissions = _tradeoff_panel(bundle)
        tradeoff_name = "corrective_tradeoffs.pdf"
        tradeoff_figure.savefig(staging / tradeoff_name, format="pdf", dpi=300)
        plt.close(tradeoff_figure)
        names.append(tradeoff_name)

        _reverify_file_record(
            replay_root / REPLAY_MANIFEST_FILE, replay_root, replay_manifest_record
        )
        _reverify_file_record(
            replay_root / REPLAY_FILE, replay_root, replay_file_record
        )
        _reverify_file_record(
            staging / REPLAY_MANIFEST_FILE, staging, replay_manifest_record
        )
        _reverify_file_record(staging / REPLAY_FILE, staging, replay_file_record)
        manifest = {
            "schema": RENDER_SCHEMA,
            "status": "complete",
            "classification": "VISUALIZATION_ONLY",
            "experiment_id": EXPERIMENT_ID,
            "qualitative_contract_sha256": QUALITATIVE_CONTRACT_SHA256,
            "prospective_opened": False,
            "sealed_opened": False,
            "visualization_script": dict(VISUALIZATION_SOURCE_AT_IMPORT),
            "replay_producer_visualization_script": dict(
                bundle.manifest["source"]["visualization_script"]
            ),
            "replay_manifest": replay_manifest_record,
            "replay_file": replay_file_record,
            "outputs": {name: _file_record(staging / name, staging) for name in names},
            "rendering": {
                "paper_figures_have_no_overall_titles": True,
                "spatial_representation": (
                    "flat native-quadrilateral mean of native vertex values"
                ),
                "interpolation": "none",
                "fixed_x_limits": list(X_LIMITS),
                "fixed_y_limits": list(Y_LIMITS),
                "continuous_qualitative_color_limits": {
                    key: list(value) for key, value in qualitative_limits.items()
                },
                "exact_evaluator_static_color_limits": {
                    key: list(value) for key, value in exact_static_limits.items()
                },
                "visible_native_quad_count": int(quads.shape[0]),
                "static_horizons": list(STATIC_HORIZONS),
                "animation_horizons": list(selected_animation_horizons),
                "animation_frame_stride": arguments.frame_stride,
                "animation_fps": arguments.fps,
                "animation_dpi": arguments.dpi,
                "animation_codec": "libx264/yuv420p",
                "diagnostic_nonfinite_pairs": diagnostic_omissions,
                "artifact_data_provenance": {
                    "density_deployments_h*.pdf": (
                        "exact evaluator registered snapshots"
                    ),
                    "density_signed_error_h*.pdf": (
                        "exact evaluator registered snapshots"
                    ),
                    "density_deployments_seed*.mp4": (
                        "independent continuous qualitative CUDA rerun"
                    ),
                    "density_error_traces.pdf": (
                        "independent continuous qualitative CUDA rerun"
                    ),
                    "corrective_tradeoffs.pdf": (
                        "verified evaluator aggregate summaries"
                    ),
                },
                "caption_boundary": (
                    "Static h35/h104/h208 panels are exact evaluator snapshots; "
                    "animations and density traces are an independent qualitative "
                    "CUDA rerun and are not long-horizon numerical reproductions."
                ),
            },
            "claim_boundary": _claim_boundary(),
        }
        manifest["canonical_payload_sha256"] = _canonical_sha256(manifest)
        _reverify_visualization_source()
        _reverify_file_record(
            replay_root / REPLAY_MANIFEST_FILE, replay_root, replay_manifest_record
        )
        _reverify_file_record(
            replay_root / REPLAY_FILE, replay_root, replay_file_record
        )
        _write_json(staging / RENDER_MANIFEST_FILE, manifest)
        return manifest

    def verify_before_publish(staging: Path, manifest: dict[str, Any]) -> None:
        _reverify_visualization_source()
        _reverify_file_record(
            replay_root / REPLAY_MANIFEST_FILE, replay_root, replay_manifest_record
        )
        _reverify_file_record(
            replay_root / REPLAY_FILE, replay_root, replay_file_record
        )
        _verify_staged_render_packet(
            staging,
            manifest,
            replay_manifest_record=replay_manifest_record,
            replay_file_record=replay_file_record,
        )

    return _publish_directory_atomically(
        output, build, before_publish=verify_before_publish
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    replay_parser = subparsers.add_parser("replay", help="run verified inference")
    replay_parser.add_argument(
        "--population-role",
        choices=("development", "prospective", "sealed"),
        default="development",
    )
    replay_parser.add_argument("--contract", type=Path, required=True)
    replay_parser.add_argument("--successor-contract", type=Path, required=True)
    replay_parser.add_argument("--preregistration", type=Path, required=True)
    replay_parser.add_argument("--dataset-dir", type=Path, required=True)
    replay_parser.add_argument("--calibration-dir", type=Path, required=True)
    replay_parser.add_argument("--evaluation-dir", type=Path, required=True)
    replay_parser.add_argument(
        "--training-dir", type=Path, action="append", required=True
    )
    replay_parser.add_argument("--output-dir", type=Path, required=True)
    replay_parser.add_argument("--device", default="cuda")

    render_parser = subparsers.add_parser("render", help="render portable replay")
    render_parser.add_argument("--replay-dir", type=Path, required=True)
    render_parser.add_argument("--output-dir", type=Path, required=True)
    render_parser.add_argument("--ffmpeg", type=Path)
    render_parser.add_argument(
        "--expected-replay-producer-sha256",
        default=VISUALIZATION_SOURCE_AT_IMPORT["sha256"],
        help=(
            "expected SHA-256 of the visualization source that produced the replay; "
            "set this explicitly when rendering a verified historical replay"
        ),
    )
    render_parser.add_argument("--fps", type=int, default=12)
    render_parser.add_argument("--frame-stride", type=int, default=2)
    render_parser.add_argument("--dpi", type=int, default=120)
    render_parser.add_argument("--bitrate", type=int, default=4200)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "replay":
            result = replay(arguments)
        else:
            if (
                arguments.fps <= 0
                or arguments.frame_stride <= 0
                or arguments.dpi <= 0
                or arguments.bitrate <= 0
            ):
                raise ValueError("fps, frame stride, dpi, and bitrate must be positive")
            result = render(arguments)
    except ProtectedPopulationError as error:
        print(
            f"NACA successor visualization protected access refused: {error}",
            file=sys.stderr,
        )
        return 4
    except torch.cuda.OutOfMemoryError as error:
        print(
            f"NACA successor visualization infrastructure failure: {error}",
            file=sys.stderr,
        )
        return 3
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"NACA successor visualization failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
