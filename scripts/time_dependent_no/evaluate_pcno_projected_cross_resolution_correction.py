#!/usr/bin/env python3
"""Run the CPU-only W26-L5 rank-8 projected-correction A1 harness.

Only ``dry-run`` and deterministic ``synthetic`` are available.  This entry
point has no checkpoint, dataset, reference, remote, or recurrent-rollout
loader and does not authorize a scientific execution.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utility.time_dependent_no.pcno_artifacts import git_state, sha256_files
from utility.time_dependent_no.pcno_cross_resolution_correction import (
    MODEL_NAMES,
    DiagnosticSnapshot,
    NativeIncrementBasis,
    ResolutionContract,
    prepare_common_native_inputs,
)
from utility.time_dependent_no.pcno_cross_resolution_teacher_forced import (
    build_fixed_cosine_projector,
    with_payload_sha256,
)
from utility.time_dependent_no.pcno_projected_cross_resolution_correction import (
    PROJECTED_SELECTION_VIEW,
    project_parallel_field,
    projected_coefficient_stability,
    projected_correction_field,
    projected_grouped_crossfit,
    projected_native_prediction,
    projected_synchronized_fusion_step,
)
from utility.time_dependent_no.pcno_resolution_transfer import Resolution

SCHEMA = "pcno_projected_cross_resolution_correction_a1_v1"
WORKING_ID = "W26-L5-P1-R8-A1"
SOURCE_PATHS = (
    "docs/time_dependent_no/W26_L5_RANK8_PROJECTED_CORRECTION_PREREGISTRATION.md",
    "utility/time_dependent_no/pcno_projected_cross_resolution_correction.py",
    "scripts/time_dependent_no/evaluate_pcno_projected_cross_resolution_correction.py",
    "tests/time_dependent_no/test_pcno_projected_cross_resolution_correction.py",
)
FROZEN_A2_SOURCE_SHA256 = {
    "docs/time_dependent_no/W26_L5_CROSS_RESOLUTION_PREREGISTRATION.md": (
        "157f02824b05eb346576fc90845edbd7fcb8b4fe2247b48e53d9168bc13fc2b7"
    ),
    "utility/time_dependent_no/pcno_cross_resolution_correction.py": (
        "1b5038b7b52f6eeb810263e194779a85039f97bef29220fcce49f60d7e04509d"
    ),
    "utility/time_dependent_no/pcno_cross_resolution_teacher_forced.py": (
        "b245aa3b9940a2966cc35a415c217f2076a8d4f42b4c19e95599891fee01450d"
    ),
    "scripts/time_dependent_no/evaluate_pcno_cross_resolution_correction.py": (
        "3cf3b13b3312ece83f2163b67efd30e433ca2fa2e68dd416bbd0d7a762fcab70"
    ),
    "scripts/time_dependent_no/evaluate_pcno_cross_resolution_teacher_forced.py": (
        "c741985291391fd04a0890b29ebfe62f775b99f154c34b51d29d7376c76d7030"
    ),
    "tests/time_dependent_no/test_pcno_cross_resolution_correction.py": (
        "9d2aa04cf8c955e4bfc02507e31a416239198dce126bdcb36595b6e93fbae9af"
    ),
    "tests/time_dependent_no/test_pcno_cross_resolution_teacher_forced.py": (
        "526ce7b783c1a6764e37d215480bd719b2fdde8e3c1a2d94c0da918684525b0e"
    ),
}
PRIOR_A2_ARTIFACT_SHA256 = {
    "calibration": ("00c37737eefb332695aae18d25c5f0641cdd5c4f9526852b9eadce547be02a29"),
    "evaluation": ("f02e76a8ee8b00db74d322922b4fd63d412042b26908825854b8b5cac7127c8f"),
}


def _git_status_short(paths: Sequence[str] | None = None) -> list[str]:
    command = ["git", "status", "--short"]
    if paths is not None:
        command.extend(("--", *paths))
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return ["unknown"]
    return result.stdout.splitlines() if result.returncode == 0 else ["unknown"]


def source_identity() -> dict[str, Any]:
    current_frozen = sha256_files(FROZEN_A2_SOURCE_SHA256, root=ROOT)
    return {
        "git": git_state(ROOT),
        "global_status_short": _git_status_short(),
        "relevant_status_short": _git_status_short(SOURCE_PATHS),
        "file_sha256": sha256_files(SOURCE_PATHS, root=ROOT),
        "frozen_a2_source_sha256": current_frozen,
        "frozen_a2_source_identity_exact": (current_frozen == FROZEN_A2_SOURCE_SHA256),
    }


def canonical_dry_run_contract() -> dict[str, Any]:
    return with_payload_sha256(
        {
            "schema": SCHEMA,
            "working_id": WORKING_ID,
            "status": "contract_only",
            "stage": "A1 dry-run only",
            "executes_checkpoint": False,
            "executes_dataset_arrays": False,
            "executes_recurrence": False,
            "remote_execution_authorized": False,
            "sealed_population_authorized": False,
            "family": "dynamic finite-volume shock-vortex only",
            "prior_evidence": {
                "artifact_sha256": PRIOR_A2_ARTIFACT_SHA256,
                "use": "hypothesis-generating; not an independent test",
                "open_population_reuse": "adaptive_open_validation",
            },
            "policy": {
                "features": [
                    "native increment minus prolonged coarse increment",
                    "restricted fine increment minus native increment",
                ],
                "offline_target": (
                    "native reference increment minus native predicted increment"
                ),
                "projection": PROJECTED_SELECTION_VIEW,
                "projector": (
                    "physical-volume weighted QR of the first eight fixed "
                    "physical-cosine modes on dynamic node type 0"
                ),
                "excluded_nodes": "correction is exactly zero on node types 1--3",
                "candidate_models": list(MODEL_NAMES),
                "coefficient_scope": (
                    "one fixed pair across cases/times in this dynamic-FV contract"
                ),
                "cross_family_coefficients": False,
                "true_error_available_at_inference": False,
            },
            "next_execution_boundary": (
                "separate authorization for one adaptive open-validation "
                "teacher-forced diagnostic; no recurrence or sealed data"
            ),
            "source_identity": source_identity(),
        }
    )


def _native_grid(
    resolution: Resolution,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx, ny = resolution
    x = (np.arange(nx, dtype=np.float64) + 0.5) * (2.0 / nx)
    y = (np.arange(ny, dtype=np.float64) + 0.5) * (1.0 / ny)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    nodes = np.column_stack((xx.reshape(-1), yy.reshape(-1)))
    volumes = np.full(nx * ny, 2.0 / (nx * ny), dtype=np.float64)
    node_type = np.zeros(nx * ny, dtype=np.int64)
    node_type[:nx] = 1
    return nodes, volumes, node_type


def synthetic_smoke_summary(seed: int = 0) -> dict[str, Any]:
    """Run deterministic projected-fit and synchronized-step CPU closures."""

    rng = np.random.default_rng(int(seed))
    contract = ResolutionContract(coarse=(4, 2), native=(8, 4), fine=(16, 8))
    nodes, volumes, node_type = _native_grid(contract.native)
    projector = build_fixed_cosine_projector(
        nodes,
        volumes,
        node_type,
        rank=8,
    )
    component_scale = np.asarray((0.5, 0.75, 1.0, 1.25))
    alpha_true = 0.35
    beta_true = -0.2
    expected_groups = {
        f"g{group}": tuple(f"g{group}_m{member}" for member in range(2))
        for group in range(4)
    }
    expected_calls = (0, 1, 2)
    snapshots = []
    maximum_projection_closure = 0.0

    for group_id, case_ids in expected_groups.items():
        for case_id in case_ids:
            for input_call in expected_calls:
                coarse_feature = rng.normal(size=(nodes.shape[0], 4))
                fine_feature = rng.normal(size=(nodes.shape[0], 4))
                projected_coarse, coarse_closure = project_parallel_field(
                    coarse_feature,
                    projector,
                )
                projected_fine, fine_closure = project_parallel_field(
                    fine_feature,
                    projector,
                )
                noise_parts, _ = projector.split(
                    0.05 * rng.normal(size=(nodes.shape[0], 4))
                )
                target = (
                    alpha_true * projected_coarse
                    + beta_true * projected_fine
                    + noise_parts["orthogonal"]
                    + noise_parts["excluded"]
                )
                zero = np.zeros_like(target)
                snapshots.append(
                    DiagnosticSnapshot(
                        case_id=case_id,
                        group_id=group_id,
                        input_call=input_call,
                        basis=NativeIncrementBasis(
                            native_increment=zero,
                            coarse_on_native=zero,
                            fine_on_native=zero,
                            native_minus_coarse=coarse_feature,
                            fine_minus_native=fine_feature,
                        ),
                        target_correction=target,
                        volumes=volumes,
                        component_scale=component_scale,
                    )
                )
                maximum_projection_closure = max(
                    maximum_projection_closure,
                    coarse_closure.maximum_reconstruction_abs,
                    coarse_closure.maximum_idempotence_abs,
                    coarse_closure.relative_parallel_orthogonal_inner,
                    fine_closure.maximum_reconstruction_abs,
                    fine_closure.maximum_idempotence_abs,
                    fine_closure.relative_parallel_orthogonal_inner,
                )

    crossfit = projected_grouped_crossfit(
        snapshots,
        resolution=contract.native,
        projector=projector,
        expected_groups=expected_groups,
        expected_input_calls=expected_calls,
    )
    coefficients = crossfit["selected_coefficients"]
    coefficient_error = (
        float("inf")
        if coefficients is None
        else max(
            abs(float(coefficients[0]) - alpha_true),
            abs(float(coefficients[1]) - beta_true),
        )
    )
    stability = projected_coefficient_stability(
        crossfit,
        minimum_same_sign_folds=len(expected_groups),
    )

    example = snapshots[0]
    correction = projected_correction_field(
        example.basis,
        projector,
        alpha=alpha_true,
        beta=beta_true,
    )
    target_parts, _ = projector.split(example.target_correction)
    remaining_parts, _ = projector.split(example.target_correction - correction)
    raw_prediction = rng.normal(size=(nodes.shape[0], 4)).astype(np.float32)
    zero_prediction = projected_native_prediction(
        raw_prediction,
        example.basis,
        projector,
        alpha=0.0,
        beta=0.0,
    )

    native_state = rng.normal(size=(nodes.shape[0], 4))
    expected_inputs = prepare_common_native_inputs(native_state, contract=contract)
    call_log: list[tuple[Resolution, np.ndarray]] = []

    def predictor(resolution: Resolution, state: np.ndarray) -> np.ndarray:
        call_log.append((resolution, np.array(state, copy=True)))
        multiplier = {
            contract.coarse: 0.75,
            contract.native: 1.0,
            contract.fine: 1.25,
        }[resolution]
        return state + 0.01 * multiplier

    step = projected_synchronized_fusion_step(
        native_state,
        contract=contract,
        projector=projector,
        predictor=predictor,
        alpha=0.0,
        beta=0.0,
    )
    call_order = [resolution for resolution, _ in call_log]
    common_inputs_exact = all(
        np.array_equal(state, expected_inputs.model_inputs[resolution])
        for resolution, state in call_log
    )
    source = source_identity()
    checks = {
        "frozen_a2_source_identity": source["frozen_a2_source_identity_exact"],
        "two_term_selected": crossfit["selected_model"] == "two_term",
        "coefficient_recovery": coefficient_error <= 1.0e-10,
        "coefficient_stability": stability["status"] == "passed",
        "projection_closure": maximum_projection_closure <= 1.0e-10,
        "parallel_target_removed": (
            np.max(np.abs(remaining_parts["parallel"])) <= 1.0e-10
        ),
        "orthogonal_target_unchanged": np.allclose(
            remaining_parts["orthogonal"],
            target_parts["orthogonal"],
            atol=1.0e-12,
            rtol=0.0,
        ),
        "excluded_target_unchanged": np.array_equal(
            remaining_parts["excluded"],
            target_parts["excluded"],
        ),
        "excluded_correction_zero": np.count_nonzero(
            correction[~projector.interior_mask]
        )
        == 0,
        "zero_prediction_exact": np.array_equal(zero_prediction, raw_prediction),
        "zero_prediction_dtype_exact": zero_prediction.dtype == raw_prediction.dtype,
        "exactly_three_predictions": call_order
        == [contract.coarse, contract.native, contract.fine],
        "common_native_inputs_exact": common_inputs_exact,
        "zero_synchronized_update_exact": np.array_equal(
            step.next_native_state,
            step.predictions[contract.native],
        ),
    }
    return with_payload_sha256(
        {
            "schema": SCHEMA,
            "working_id": WORKING_ID,
            "status": "passed" if all(checks.values()) else "failed",
            "seed": int(seed),
            "checks": {key: bool(value) for key, value in checks.items()},
            "selected_model": crossfit["selected_model"],
            "selected_coefficients": coefficients,
            "true_coefficients": [alpha_true, beta_true],
            "maximum_coefficient_error": coefficient_error,
            "projection_closure": crossfit["projection_closure"],
            "coefficient_stability": stability,
            "prediction_call_count": len(call_log),
            "checkpoint_loaded": False,
            "dataset_loaded": False,
            "reference_loaded": False,
            "source_identity": source,
        }
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "dry-run",
        help="print the local contract without opening scientific inputs",
    )
    synthetic = commands.add_parser(
        "synthetic",
        help="run deterministic CPU projection, fit, and bookkeeping closures",
    )
    synthetic.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "dry-run":
        payload = canonical_dry_run_contract()
        exit_code = 0
    elif args.command == "synthetic":
        payload = synthetic_smoke_summary(args.seed)
        exit_code = 0 if payload["status"] == "passed" else 1
    else:  # pragma: no cover
        raise AssertionError(f"unknown command: {args.command}")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
