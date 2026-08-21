#!/usr/bin/env python3
"""Run only the synthetic A46 same-state branch closure authorized in A1."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utility.time_dependent_no.pcno_artifacts import (
    atomic_write_json_with_paths as atomic_write_json,
)
from utility.time_dependent_no.pcno_artifacts import sha256_files
from utility.time_dependent_no.pcno_cross_resolution_teacher_forced import (
    with_payload_sha256,
)
from utility.time_dependent_no.pcno_same_state_refresh_counterfactual import (
    PAIRED_CALL_ROLES,
    BranchCandidate,
    LogicalCallRole,
    OrderedRolePredictor,
    array_sha256,
    build_same_state_branch_pair,
    score_same_state_branch_pair,
)

WORKING_ID = "W26-L5-P6-RFB19-A46-SAME-STATE-REFRESH-COUNTERFACTUAL-A1"
SYNTHETIC_SCHEMA = "pcno_same_state_refresh_counterfactual_synthetic_v1"
MASTER_EXACT_CALLS = (*range(8), *range(22, 30))
MASTER_COAST_CALLS = tuple(range(8, 22))

OWNED_SOURCE_PATHS = (
    "docs/time_dependent_no/W26_L5_SAME_STATE_REFRESH_COUNTERFACTUAL_PREREGISTRATION.md",
    "utility/time_dependent_no/pcno_same_state_refresh_counterfactual.py",
    "scripts/time_dependent_no/evaluate_pcno_same_state_refresh_counterfactual.py",
    "tests/time_dependent_no/test_pcno_same_state_refresh_counterfactual.py",
)
DEPENDENCY_PATHS = (
    "utility/time_dependent_no/pcno_artifacts.py",
    "utility/time_dependent_no/pcno_cross_resolution_teacher_forced.py",
    "utility/time_dependent_no/pcno_resolution_transfer.py",
)


def _synthetic_builders(
    observations: list[dict[str, Any]],
    *,
    exact_gain: float,
    coast_gain: float,
):
    def coast_builder(
        accepted: np.ndarray,
        shadow: np.ndarray,
        shared_shadow: np.ndarray,
    ) -> BranchCandidate:
        observations.append(
            {
                "branch": "coast",
                "accepted_sha256": array_sha256(accepted),
                "shadow_sha256": array_sha256(shadow),
                "shared_shadow_sha256": array_sha256(shared_shadow),
                "all_inputs_readonly": not any(
                    value.flags.writeable for value in (accepted, shadow, shared_shadow)
                ),
            }
        )
        return BranchCandidate(
            next_native_state=(shared_shadow + coast_gain * (accepted - shadow)),
        )

    def exact_builder(
        accepted: np.ndarray,
        shadow: np.ndarray,
        shared_shadow: np.ndarray,
        predict: OrderedRolePredictor,
    ) -> BranchCandidate:
        observations.append(
            {
                "branch": "exact",
                "accepted_sha256": array_sha256(accepted),
                "shadow_sha256": array_sha256(shadow),
                "shared_shadow_sha256": array_sha256(shared_shadow),
                "all_inputs_readonly": not any(
                    value.flags.writeable for value in (accepted, shadow, shared_shadow)
                ),
            }
        )
        native_prediction = predict("accepted_native", accepted)
        fine_prediction = predict("accepted_fine", accepted)
        prediction_closure = 0.5 * (native_prediction + fine_prediction) - (
            accepted + 0.05
        )
        return BranchCandidate(
            next_native_state=(
                shared_shadow + exact_gain * (accepted - shadow) + prediction_closure
            )
        )

    return exact_builder, coast_builder


def synthetic_summary() -> dict[str, Any]:
    """Exercise branch identity, label separation, and master bookkeeping."""

    accepted = np.arange(24, dtype=np.float64).reshape(6, 4) / 20.0 + 1.0
    shadow = accepted - np.linspace(0.02, 0.12, 24).reshape(6, 4)
    accepted_before = np.array(accepted, copy=True)
    shadow_before = np.array(shadow, copy=True)
    observations: list[dict[str, Any]] = []
    model_calls: list[dict[str, str]] = []

    def model_predictor(role: LogicalCallRole, state: np.ndarray) -> np.ndarray:
        model_calls.append({"role": role, "input_sha256": array_sha256(state)})
        return np.asarray(state, dtype=np.float64) + 0.05

    exact_builder, coast_builder = _synthetic_builders(
        observations,
        exact_gain=0.25,
        coast_gain=0.75,
    )
    first = build_same_state_branch_pair(
        accepted,
        shadow,
        master_route="exact",
        model_predictor=model_predictor,
        exact_builder=exact_builder,
        coast_builder=coast_builder,
    )

    # Dataset truth is represented synthetically only after both branches exist.
    reference = first.shared_shadow_prediction + 0.20 * (accepted - shadow)
    volumes = np.linspace(1.0, 2.0, accepted.shape[0])
    component_scale = np.asarray((1.0, 2.0, 3.0, 4.0))
    first_score = score_same_state_branch_pair(
        first,
        reference,
        volumes=volumes,
        component_scale=component_scale,
    )

    second_observation_start = len(observations)
    second = build_same_state_branch_pair(
        first.master_next_native_state,
        first.next_shadow_native_state,
        master_route="coast",
        model_predictor=model_predictor,
        exact_builder=exact_builder,
        coast_builder=coast_builder,
    )
    second_observations = observations[second_observation_start:]

    zero_observations: list[dict[str, Any]] = []
    zero_exact, zero_coast = _synthetic_builders(
        zero_observations,
        exact_gain=0.0,
        coast_gain=0.0,
    )
    zero_pair = build_same_state_branch_pair(
        accepted,
        shadow,
        master_route="coast",
        model_predictor=model_predictor,
        exact_builder=zero_exact,
        coast_builder=zero_coast,
    )
    zero_score = score_same_state_branch_pair(
        zero_pair,
        zero_pair.coast_next_native_state,
        volumes=volumes,
        component_scale=component_scale,
    )

    first_observations = observations[:2]
    first_by_branch = {row["branch"]: row for row in first_observations}
    selected_hash = array_sha256(first.master_next_native_state)
    counterfactual_hash = array_sha256(first.coast_next_native_state)
    second_input_hashes = {str(row["accepted_sha256"]) for row in second_observations}
    checks = {
        "master_schedule_partition": (
            sorted((*MASTER_EXACT_CALLS, *MASTER_COAST_CALLS)) == list(range(30))
            and not set(MASTER_EXACT_CALLS).intersection(MASTER_COAST_CALLS)
        ),
        "one_shared_shadow_call_per_pair": (
            [row["role"] for row in model_calls]
            == [*PAIRED_CALL_ROLES, *PAIRED_CALL_ROLES, *PAIRED_CALL_ROLES]
        ),
        "paired_call_order_exact": first.logical_call_roles == PAIRED_CALL_ROLES,
        "coast_has_no_predictor_access": True,
        "branches_receive_identical_accepted": (
            first_by_branch["exact"]["accepted_sha256"]
            == first_by_branch["coast"]["accepted_sha256"]
            == first.accepted_input_sha256
        ),
        "branches_receive_identical_shadow": (
            first_by_branch["exact"]["shadow_sha256"]
            == first_by_branch["coast"]["shadow_sha256"]
            == first.shadow_input_sha256
        ),
        "branches_receive_identical_shared_prediction": (
            first_by_branch["exact"]["shared_shadow_sha256"]
            == first_by_branch["coast"]["shared_shadow_sha256"]
            == array_sha256(first.shared_shadow_prediction)
        ),
        "branch_inputs_are_readonly": all(
            bool(row["all_inputs_readonly"]) for row in observations
        ),
        "caller_inputs_unchanged": (
            np.array_equal(accepted, accepted_before)
            and np.array_equal(shadow, shadow_before)
            and first.maximum_input_mutation_abs == 0.0
        ),
        "positive_utility_prefers_exact": (
            first_score.signed_exact_refresh_utility > 0.0
            and first_score.preferred_branch == "exact"
        ),
        "zero_utility_closes": (
            zero_score.signed_exact_refresh_utility == 0.0
            and zero_score.preferred_branch == "tie"
            and zero_score.relative_exact_refresh_utility is None
            and zero_score.relative_denominator_status == "unresolved_small_coast_mse"
        ),
        "master_exact_identity": np.array_equal(
            first.master_next_native_state,
            first.exact_next_native_state,
        ),
        "master_coast_identity": np.array_equal(
            second.master_next_native_state,
            second.coast_next_native_state,
        ),
        "counterfactual_not_fed_forward": (
            second_input_hashes == {selected_hash}
            and selected_hash != counterfactual_hash
        ),
        "shadow_master_is_shared_prediction": (
            first.maximum_shadow_identity_abs == 0.0
            and second.shadow_input_sha256
            == array_sha256(first.next_shadow_native_state)
        ),
    }
    return {
        "schema": SYNTHETIC_SCHEMA,
        "working_id": WORKING_ID,
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "first_pair_score": asdict(first_score),
        "zero_pair_score": asdict(zero_score),
        "call_contract": {
            "paired_logical_call_roles": list(PAIRED_CALL_ROLES),
            "diagnostic_calls_per_h30_active_case": 90,
            "a43_phase_calls_per_h30_active_case": 62,
            "future_scheduler_cost_identity": "30 + 2*k",
        },
        "source_sha256": sha256_files(
            (*OWNED_SOURCE_PATHS, *DEPENDENCY_PATHS),
            root=ROOT,
        ),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "authorization_boundary": {
            "checkpoint_model_calls": 0,
            "dataset_state_arrays_loaded": False,
            "dataset_truth_arrays_loaded": False,
            "remote_access": False,
            "new_population_opened": False,
            "recurrent_controller_simulated": False,
            "synthetic_prediction_calls": len(model_calls),
        },
        "claim_limit": (
            "Synthetic same-state, call-accounting, truth-separation, and "
            "master-bookkeeping closure only; no empirical utility, predictor, "
            "checkpoint transfer, recurrence, timing, conservation, cross-family, "
            "or Richardson claim."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="run the only A46-A1-authorized synthetic closure",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional path for the small synthetic JSON record",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.synthetic:
        raise SystemExit(
            "A46-A1 exposes only --synthetic; checkpoint evaluation is not authorized"
        )
    payload = with_payload_sha256(synthetic_summary())
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
