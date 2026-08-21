#!/usr/bin/env python3
"""Run the CPU-only W26-L5 P0/A1 cross-resolution closure harness.

This entry point has no checkpoint or dataset loader.  The separately
authorized A2 evaluator must consume the frozen contract reported by
``dry-run`` and must not infer authorization from this script.
"""

from __future__ import annotations

import argparse
import hashlib
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

from utility.time_dependent_no.pcno_artifacts import (
    git_state,
    sha256_files,
)
from utility.time_dependent_no.pcno_cross_resolution_correction import (
    MODEL_NAMES,
    DiagnosticSnapshot,
    ResolutionContract,
    common_native_increment_basis,
    grouped_crossfit,
    mapped_increment_errors,
    prepare_common_native_inputs,
    synchronized_fusion_step,
)
from utility.time_dependent_no.pcno_resolution_transfer import (
    Resolution,
)

SCHEMA = "pcno_cross_resolution_correction_a1_v1"
WORKING_ID = "W26-L5-P0-A1"
SOURCE_PATHS = (
    "docs/time_dependent_no/W26_L5_CROSS_RESOLUTION_PREREGISTRATION.md",
    "utility/time_dependent_no/pcno_cross_resolution_correction.py",
    "scripts/time_dependent_no/evaluate_pcno_cross_resolution_correction.py",
    "tests/time_dependent_no/test_pcno_cross_resolution_correction.py",
)
EXPECTED_UPSTREAM_SHA256 = {
    "checkpoint": "95e6c180a3298c4b38662d8c6cb77301c0e7fd0286e9a53571bddc487b8379c9",
    "normalization_file": "4c931c813d318f9a3012814c9803cf85fbb30ce4c68739535047b2aff6f0faf4",
    "normalization_digest": "9df7efb2f1aadf315f399dcabf3b366bf1b2f46794b026cbc5b7d80ba22e1a1a",
    "split": "1be17494eaac3902159763a9e9a6c562d39c31957739899f50c28e37b48a921d",
    "data": "f8d228ae3c6e08fe6697f37df21476fe621abc354d0b0a6eed2a623ed2b9f96c",
    "family_manifest": "2c0d0c516d38dc826c23edda76e1a0ef668ff4692caf72f1bd710150629c7dc8",
    "d063_run_contract": "09d331bc376788f4a038fcb7d82389951d61e5a1eb67bb8e85b3fd39360fd9fc",
    "d063_summary": "e44238902348f6517412844220d71332e4ed05803138a255d50cb2f35f781e0b",
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
    if result.returncode != 0:
        return ["unknown"]
    return result.stdout.splitlines()


def source_identity() -> dict[str, Any]:
    """Bind current committed source plus global and scoped worktree dirt."""

    return {
        "git": git_state(ROOT),
        "global_status_short": _git_status_short(),
        "relevant_status_short": _git_status_short(SOURCE_PATHS),
        "file_sha256": sha256_files(SOURCE_PATHS, root=ROOT),
    }


def _with_payload_sha256(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["payload_sha256_scope"] = (
        "canonical sorted compact JSON excluding only payload_sha256"
    )
    encoded = json.dumps(
        result,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    result["payload_sha256"] = hashlib.sha256(encoded).hexdigest()
    return result


def canonical_dry_run_contract() -> dict[str, Any]:
    return _with_payload_sha256(
        {
            "schema": SCHEMA,
            "working_id": WORKING_ID,
            "status": "contract_only",
            "stage": "P0/A1 dry-run only",
            "executes_checkpoint": False,
            "executes_dataset_arrays": False,
            "a2_authorized": False,
            "family": "dynamic finite-volume shock-vortex only",
            "common_source": {
                "solver": "WENO5-HLLC-SSPRK3",
                "source_resolution": "1000x400",
                "source_dtype": "float64",
                "saved_dt": 0.01,
                "checkpoint_call_dt": 0.02,
                "checkpoint_step_stride": 2,
                "input_call_convention": "zero-based model-call ordinal minus one",
                "input_calls": list(range(30)),
                "reference": "evolved retained common-source states",
            },
            "resolution_contract": {
                "coarse": "125x50",
                "native": "250x100",
                "fine": "500x200",
                "physical_volumes": "uniform Cartesian dx*dy on each grid",
                "coarse_from_native": "float64 conservative block-average restriction",
                "fine_from_native": "float64 piecewise-constant prolongation",
                "mapped_objects": "predicted state increments, never full predictions",
                "model_boundary": "one explicit FP32 rounding on each actual model grid",
                "nesting_floors": "record pre-model and post-FP32 closures separately",
            },
            "model_contract": {
                "boundary_policy": "model_all_nodes raw recurrence; no intervention",
                "normalization": "checkpoint-bound D063 normalization, unchanged",
                "component_scales": "D063 reference scales, fixed before evaluation",
                "gradient_policy": "freeze the current checkpoint and operator",
                "gradient_decision": (
                    "no implementation fix before P1; physical derivative scaling is "
                    "correct, while support geometry is a leading bounded mechanism "
                    "without unique attribution"
                ),
            },
            "calibration": {
                "cases": [
                    f"sv_e{strength:02d}_{offset}"
                    for strength in (1, 2, 3, 4, 5, 7, 8, 9, 10)
                    for offset in ("y00", "y08")
                ],
                "fit_input_calls": list(range(20)),
                "held_out_time_input_calls": list(range(20, 30)),
                "strength_pair_folds": 9,
            },
            "evaluation": {
                "cases": [
                    f"sv_e{strength:02d}_{offset}"
                    for strength in (0, 6, 11)
                    for offset in ("y00", "y08")
                ],
                "loading_blocked_until_calibration_gate": True,
            },
            "offline_target": (
                "native reference increment minus native predicted increment"
            ),
            "deployable_features": [
                "native increment minus prolonged coarse increment",
                "restricted fine increment minus native increment",
            ],
            "true_error_is_offline_only": True,
            "candidate_models": list(MODEL_NAMES),
            "per_case_oracle_deployable": False,
            "source_identity": source_identity(),
            "expected_upstream_sha256_not_opened_by_a1": EXPECTED_UPSTREAM_SHA256,
        }
    )


def _maximum_absolute(value: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(value, dtype=np.float64))))


def _uniform_integral(value: np.ndarray) -> np.ndarray:
    return np.mean(np.asarray(value, dtype=np.float64), axis=0)


def synthetic_smoke_summary(seed: int = 0) -> dict[str, Any]:
    """Run deterministic end-to-end CPU closures without external inputs."""

    rng = np.random.default_rng(int(seed))
    contract = ResolutionContract(coarse=(4, 2), native=(8, 4), fine=(16, 8))
    alpha_true = 0.4
    beta_true = -0.25
    expected_groups = {
        f"g{group}": tuple(f"g{group}_m{member}" for member in range(2))
        for group in range(4)
    }
    expected_calls = (0, 1, 2)
    volumes = np.full(
        contract.native[0] * contract.native[1],
        2.0 / (contract.native[0] * contract.native[1]),
    )
    component_scale = np.asarray((0.5, 0.75, 1.0, 1.25))
    snapshots: list[DiagnosticSnapshot] = []
    maximum_sign_closure = 0.0
    maximum_integral_closure = 0.0
    maximum_pre_model_nesting_floor = 0.0
    maximum_post_fp32_nesting_floor = 0.0

    for group_id, case_ids in expected_groups.items():
        for case_id in case_ids:
            for input_call in expected_calls:
                native_state = rng.normal(
                    size=(contract.native[0] * contract.native[1], 4)
                )
                prepared = prepare_common_native_inputs(
                    native_state,
                    contract=contract,
                )
                increments = {
                    contract.coarse: 0.02
                    * rng.normal(size=(contract.coarse[0] * contract.coarse[1], 4)),
                    contract.native: 0.02
                    * rng.normal(size=(contract.native[0] * contract.native[1], 4)),
                    contract.fine: 0.02
                    * rng.normal(size=(contract.fine[0] * contract.fine[1], 4)),
                }
                predictions = {
                    resolution: prepared.model_inputs[resolution]
                    + increments[resolution]
                    for resolution in (
                        contract.coarse,
                        contract.native,
                        contract.fine,
                    )
                }
                basis = common_native_increment_basis(
                    prepared,
                    predictions,
                    contract=contract,
                )
                target = (
                    alpha_true * basis.native_minus_coarse
                    + beta_true * basis.fine_minus_native
                )
                reference_increment = basis.native_increment + target
                errors = mapped_increment_errors(
                    basis,
                    native_reference_increment=reference_increment,
                    contract=contract,
                )
                maximum_sign_closure = max(
                    maximum_sign_closure,
                    _maximum_absolute(
                        basis.native_minus_coarse
                        - (errors["native"] - errors["coarse_on_native"])
                    ),
                    _maximum_absolute(
                        basis.fine_minus_native
                        - (errors["fine_on_native"] - errors["native"])
                    ),
                    _maximum_absolute(target + errors["native"]),
                )
                maximum_integral_closure = max(
                    maximum_integral_closure,
                    _maximum_absolute(
                        _uniform_integral(increments[contract.coarse])
                        - _uniform_integral(basis.coarse_on_native)
                    ),
                    _maximum_absolute(
                        _uniform_integral(increments[contract.fine])
                        - _uniform_integral(basis.fine_on_native)
                    ),
                )
                maximum_pre_model_nesting_floor = max(
                    maximum_pre_model_nesting_floor,
                    prepared.nesting_floors["pre_model_coarse_from_native_max_abs"],
                    prepared.nesting_floors["pre_model_fine_to_native_max_abs"],
                )
                maximum_post_fp32_nesting_floor = max(
                    maximum_post_fp32_nesting_floor,
                    prepared.nesting_floors["post_fp32_coarse_from_native_max_abs"],
                    prepared.nesting_floors["post_fp32_fine_to_native_max_abs"],
                )
                snapshots.append(
                    DiagnosticSnapshot(
                        case_id=case_id,
                        group_id=group_id,
                        input_call=input_call,
                        basis=basis,
                        target_correction=target,
                        volumes=volumes,
                        component_scale=component_scale,
                    )
                )

    crossfit = grouped_crossfit(
        snapshots,
        resolution=contract.native,
        expected_groups=expected_groups,
        expected_input_calls=expected_calls,
    )
    selected_coefficients = crossfit["selected_coefficients"]
    coefficient_error = (
        float("inf")
        if selected_coefficients is None
        else max(
            abs(float(selected_coefficients[0]) - alpha_true),
            abs(float(selected_coefficients[1]) - beta_true),
        )
    )

    recurrence_native = rng.normal(size=(contract.native[0] * contract.native[1], 4))
    expected_recurrence_inputs = prepare_common_native_inputs(
        recurrence_native,
        contract=contract,
    )
    call_log: list[tuple[Resolution, np.ndarray]] = []

    def predictor(resolution: Resolution, state: np.ndarray) -> np.ndarray:
        call_log.append((resolution, np.array(state, copy=True)))
        multiplier = {
            contract.coarse: 0.75,
            contract.native: 1.0,
            contract.fine: 1.25,
        }[resolution]
        return state + multiplier * 0.01

    step = synchronized_fusion_step(
        recurrence_native,
        contract=contract,
        predictor=predictor,
        alpha=0.0,
        beta=0.0,
    )
    call_order = [resolution for resolution, _ in call_log]
    common_inputs_exact = all(
        np.array_equal(
            state,
            expected_recurrence_inputs.model_inputs[resolution],
        )
        for resolution, state in call_log
    )
    checks = {
        "pre_model_nesting_exact": maximum_pre_model_nesting_floor == 0.0,
        "post_fp32_nesting_floor_reported": np.isfinite(
            maximum_post_fp32_nesting_floor
        ),
        "mapped_increment_sign_closure": maximum_sign_closure <= 1.0e-14,
        "mapped_increment_integral_closure": maximum_integral_closure <= 1.0e-14,
        "band_closure": crossfit["maximum_band_closure"] <= 1.0e-12,
        "two_term_selected": crossfit["selected_model"] == "two_term",
        "coefficient_recovery": coefficient_error <= 1.0e-10,
        "exactly_three_predictions": call_order
        == [contract.coarse, contract.native, contract.fine],
        "common_native_inputs_exact": common_inputs_exact,
        "zero_prediction_exact": np.array_equal(
            step.next_native_state,
            step.predictions[contract.native],
        ),
    }
    status = "passed" if all(checks.values()) else "failed"
    return _with_payload_sha256(
        {
            "schema": SCHEMA,
            "working_id": WORKING_ID,
            "status": status,
            "seed": int(seed),
            "checks": {name: bool(value) for name, value in checks.items()},
            "maximum_pre_model_nesting_floor": maximum_pre_model_nesting_floor,
            "maximum_post_fp32_nesting_floor": maximum_post_fp32_nesting_floor,
            "maximum_sign_closure": maximum_sign_closure,
            "maximum_integral_closure": maximum_integral_closure,
            "maximum_band_closure": crossfit["maximum_band_closure"],
            "selected_model": crossfit["selected_model"],
            "selected_coefficients": selected_coefficients,
            "true_coefficients": [alpha_true, beta_true],
            "maximum_coefficient_error": coefficient_error,
            "prediction_call_count": len(call_log),
            "checkpoint_loaded": False,
            "dataset_loaded": False,
            "source_identity": source_identity(),
        }
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "dry-run",
        help="print the frozen A1/A2 boundary without opening external inputs",
    )
    synthetic = subparsers.add_parser(
        "synthetic",
        help="run deterministic CPU closure and cross-fit smoke diagnostics",
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
    else:  # pragma: no cover - argparse prevents this branch
        raise AssertionError(f"unknown command: {args.command}")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
