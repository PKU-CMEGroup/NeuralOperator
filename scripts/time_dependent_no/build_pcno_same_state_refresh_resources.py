#!/usr/bin/env python3
"""Build and audit the frozen A46 matched checkpoint/truth resources."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no.train_pcno_euler2d_residual import (
    build_data_contract_summary,
    build_model,
    manifest_train_val_test_split,
    set_seed,
)
from scripts.time_dependent_no.train_pcno_euler2d_residual import (
    parse_args as parse_training_args,
)
from scripts.time_dependent_no.train_pcno_euler2d_residual import (
    validate_args as validate_training_args,
)
from utility.time_dependent_no.pcno_artifacts import sha256_file
from utility.time_dependent_no.pcno_euler2d import (
    PCNOEuler2DShardStore,
    digest_mapping,
    fit_normalization,
)
from utility.time_dependent_no.pcno_same_state_refresh_metadata import (
    collect_registered_case_ids,
    load_json_object,
)
from utility.time_dependent_no.pcno_same_state_refresh_resources import (
    DUAL_REFERENCE_SCHEMA,
    EXPECTED_DATA_MANIFEST_DIGEST,
    EXPECTED_FINE_RESOLUTION,
    EXPECTED_NATIVE_RESOLUTION,
    EXPECTED_NORMALIZATION,
    EXPECTED_NORMALIZATION_DIGEST,
    RESOURCE_AUDIT_SCHEMA,
    RESOURCE_BUILD_WORKING_ID,
    atomic_write_json,
    build_population_manifest,
    build_resource_plan,
    canonical_json_sha256,
    case_by_id,
    config_for_case,
    derive_native_reference,
    frozen_selector_active,
    load_resource_plan,
    matched_training_arguments,
    resource_position_descriptor,
    sha256_array,
)
from utility.time_dependent_no.shock_vortex_fv import (
    reference_contract_checks,
    run_shock_vortex_reference,
)

CLOSURE_RELATIVE_TOLERANCE = 1.0e-10
INITIAL_QUADRATURE_RELATIVE_TOLERANCE = 1.0e-10
INTEGRATED_RESTRICTION_TOLERANCE = 1.0e-12


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="write the immutable pre-outcome plan")
    plan.add_argument("--output-path", type=Path, required=True)

    training = commands.add_parser(
        "training-command", help="emit the explicit matched trainer command"
    )
    training.add_argument("--python", default=sys.executable)
    training.add_argument("--data-dir", type=Path, required=True)
    training.add_argument("--training-output-dir", type=Path, required=True)
    training.add_argument("--output-path", type=Path, required=True)

    preflight = commands.add_parser(
        "preflight-training",
        help="validate data, normalization, parser, and model shape before training",
    )
    preflight.add_argument("--plan", type=Path, required=True)
    preflight.add_argument("--data-dir", type=Path, required=True)
    preflight.add_argument("--training-output-dir", type=Path, required=True)
    preflight.add_argument("--output-path", type=Path, required=True)
    preflight.add_argument("--require-pass", action="store_true")

    generate = commands.add_parser(
        "generate-case", help="run one common-source dual-resolution FV case"
    )
    generate.add_argument("--plan", type=Path, required=True)
    generate.add_argument("--case-id", required=True)
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.add_argument("--device", choices=("cpu", "cuda"), default="cuda")

    audit = commands.add_parser(
        "audit", help="bind all passed summaries without opening truth arrays"
    )
    audit.add_argument("--plan", type=Path, required=True)
    audit.add_argument("--artifact-root", type=Path, required=True)
    audit.add_argument(
        "--opened-source-manifest", type=Path, action="append", required=True
    )
    audit.add_argument("--population-output", type=Path, required=True)
    audit.add_argument("--audit-output", type=Path, required=True)
    return parser.parse_args(argv)


def _write_new_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"output path already exists: {path}")
    atomic_write_json(path, value)


def _atomic_save_npz(path: Path, **arrays: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _training_command(args: argparse.Namespace) -> dict[str, Any]:
    trainer = "scripts/time_dependent_no/train_pcno_euler2d_residual.py"
    arguments = matched_training_arguments(args.data_dir, args.training_output_dir)
    argv = [str(args.python), "-u", trainer, *arguments]
    return {
        "working_id": RESOURCE_BUILD_WORKING_ID,
        "python": str(args.python),
        "trainer": trainer,
        "arguments": arguments,
        "argv": argv,
        "posix_shell_command": shlex.join(argv),
        "independent_initialization": True,
        "init_checkpoint": None,
        "resume_checkpoint": None,
        "a46_branch_predictions": 0,
    }


def _preflight_training(args: argparse.Namespace) -> dict[str, Any]:
    plan = load_resource_plan(args.plan, ROOT)
    training_argv = matched_training_arguments(args.data_dir, args.training_output_dir)
    parsed = parse_training_args(training_argv)
    if not torch.cuda.is_available():
        raise RuntimeError("matched training preflight requires the selected CUDA host")
    device = torch.device("cuda")
    validate_training_args(parsed, device)
    store = PCNOEuler2DShardStore(args.data_dir)
    train_keys, val_keys, test_keys = manifest_train_val_test_split(store)
    normalization = fit_normalization(
        store,
        train_keys,
        step_stride=parsed.step_stride,
        time_stride=parsed.stats_time_stride,
        gamma=float(store.manifest.get("gamma", 1.4)),
        mach_scale_floor=parsed.mach_scale_floor,
    )
    normalization_payload = normalization.to_dict()
    data_contract = build_data_contract_summary(
        store,
        train_keys=train_keys,
        val_keys=val_keys,
        test_keys=test_keys,
        normalization=normalization,
    )
    set_seed(parsed.seed)
    model = build_model(parsed, normalization, zero_initialize=False)
    model_config = model.model_config()
    expected_config = plan["training_contract"]["model_config"]
    config_checks = {
        key: model_config.get(key) == value for key, value in expected_config.items()
    }
    checks = {
        "resource_plan_valid": True,
        "cuda_available": True,
        "data_manifest_exact": store.manifest_digest == EXPECTED_DATA_MANIFEST_DIGEST,
        "manifest_split_complete": bool(train_keys and val_keys and test_keys),
        "normalization_values_exact": normalization_payload == EXPECTED_NORMALIZATION,
        "normalization_digest_exact": digest_mapping(normalization_payload)
        == EXPECTED_NORMALIZATION_DIGEST,
        "data_contract_normalization_exact": data_contract["normalization_digest"]
        == EXPECTED_NORMALIZATION_DIGEST,
        "model_config_exact": all(config_checks.values()),
        "physical_node_types": parsed.model_node_type_input == "physical"
        and parsed.node_type_channel_control == "standard",
        "no_boundary_fields": parsed.boundary_field_mode == "none"
        and parsed.boundary_residual_mode == "none",
        "raw_boundary_recurrence": parsed.boundary_mode == "model_all_nodes",
        "stride_exact": parsed.step_stride == 2,
        "independent_initialization": parsed.init_checkpoint is None
        and parsed.resume_checkpoint is None,
        "gradient_implementation_unmodified_by_launcher": True,
    }
    status = "passed" if all(checks.values()) else "failed"
    return {
        "schema": "pcno_same_state_refresh_training_preflight_v1",
        "working_id": RESOURCE_BUILD_WORKING_ID,
        "status": status,
        "checks": checks,
        "model_config_checks": config_checks,
        "model_config": model_config,
        "normalization": normalization_payload,
        "normalization_digest": digest_mapping(normalization_payload),
        "data_manifest_digest": store.manifest_digest,
        "data_contract": data_contract,
        "split_counts": {
            "train": len(train_keys),
            "validation": len(val_keys),
            "test": len(test_keys),
        },
        "training_arguments": training_argv,
        "resource_plan_payload_sha256": plan["payload_sha256"],
        "activity": {
            "checkpoint_tensor_loads": 0,
            "optimizer_steps": 0,
            "model_predictions": 0,
            "a46_branch_predictions": 0,
            "controllers_run": 0,
        },
    }


def _generate_case(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    plan = load_resource_plan(args.plan, ROOT)
    case = case_by_id(plan, args.case_id)
    config = config_for_case(plan, args.case_id)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = perf_counter()
    result = run_shock_vortex_reference(config, device=device, dtype=torch.float64)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = perf_counter() - started
    solver_checks = reference_contract_checks(
        result,
        closure_relative_tolerance=CLOSURE_RELATIVE_TOLERANCE,
        initial_quadrature_tolerance=INITIAL_QUADRATURE_RELATIVE_TOLERANCE,
    )
    derived = derive_native_reference(result)
    native_states = derived["native_states"]
    native_geometry = derived["native_geometry"]
    descriptor = resource_position_descriptor(
        native_states[0],
        nodes=native_geometry.cell_centers,
        volumes=native_geometry.cell_volume,
        y_min=config.y_min,
        y_max=config.y_max,
    )
    observed_active = bool(
        descriptor.normalized_wall_distance is not None
        and frozen_selector_active(descriptor.normalized_wall_distance, 0)
    )
    checks = {
        **solver_checks,
        "native_states_finite": bool(np.isfinite(native_states).all()),
        "fine_states_finite": bool(np.isfinite(result.states).all()),
        "physical_volumes_positive": derived["fine_volume_min"] > 0.0
        and derived["native_volume_min"] > 0.0,
        "nested_state_restriction_exact": derived["nested_restriction_max_abs"] == 0.0,
        "integrated_state_restriction_consistent": derived["integrated_state_max_abs"]
        <= INTEGRATED_RESTRICTION_TOLERANCE,
        "selector_resolved": descriptor.status == "ok",
        "selector_matches_frozen_inventory": observed_active
        is case["expected_selector_active"],
        "single_evolution_dual_resolution": True,
    }
    status = "passed" if all(checks.values()) else "failed"
    configuration_sha256 = canonical_json_sha256(config.to_dict())
    physical_times_sha256 = sha256_array(result.times)
    metadata = {
        "schema": DUAL_REFERENCE_SCHEMA,
        "working_id": RESOURCE_BUILD_WORKING_ID,
        "case_id": args.case_id,
        "parameters": case["parameters"],
        "population": case["population"],
        "strength_group": case["strength_group"],
        "position_id": case["position_id"],
        "resource_plan_file_sha256": sha256_file(args.plan),
        "resource_plan_payload_sha256": plan["payload_sha256"],
        "source_sha256": plan["source_sha256"],
        "configuration_sha256": configuration_sha256,
        "physical_times_sha256": physical_times_sha256,
        "evolution_resolution": [config.nx, config.ny],
        "fine_resolution": list(EXPECTED_FINE_RESOLUTION),
        "native_resolution": list(EXPECTED_NATIVE_RESOLUTION),
        "reference_construction": (
            "one high-fidelity evolution retained at 500x200; 250x100 is the "
            "exact conservative block restriction of those same evolved states"
        ),
        "device": str(device),
        "dtype": "float64",
        "elapsed_seconds": elapsed_seconds,
        "accepted_state_clipping_or_floors": False,
        "future_reference_boundary_values": False,
    }

    args.output_dir.mkdir(parents=True)
    artifact_path = args.output_dir / "reference.npz"
    _atomic_save_npz(
        artifact_path,
        schema=np.asarray(DUAL_REFERENCE_SCHEMA),
        conservative_states_500x200=result.states,
        conservative_states_250x100=native_states,
        physical_times=result.times,
        cell_centers_500x200=result.geometry.cell_centers,
        cell_volume_500x200=result.geometry.cell_volume,
        cell_centers_250x100=native_geometry.cell_centers,
        cell_volume_250x100=native_geometry.cell_volume,
        interval_boundary_exchange=result.interval_boundary_exchange,
        accepted_step_times=result.accepted_step_times,
        accepted_step_dt=result.accepted_step_dt,
        accepted_step_interval=result.accepted_step_interval,
        case_contract_json=np.asarray(json.dumps(case, sort_keys=True)),
        config_json=np.asarray(json.dumps(config.to_dict(), sort_keys=True)),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    summary = {
        **metadata,
        "status": status,
        "checks": checks,
        "reference_artifact": artifact_path.name,
        "reference_artifact_sha256": sha256_file(artifact_path),
        "artifact_bytes": artifact_path.stat().st_size,
        "arrays": {
            "conservative_states_500x200": list(result.states.shape),
            "conservative_states_250x100": list(native_states.shape),
            "physical_times": list(result.times.shape),
        },
        "fine_volume_min": derived["fine_volume_min"],
        "native_volume_min": derived["native_volume_min"],
        "nested_restriction_max_abs": derived["nested_restriction_max_abs"],
        "integrated_state_max_abs": derived["integrated_state_max_abs"],
        "integrated_state_relative_l2": derived["integrated_state_relative_l2"],
        "selector_status": descriptor.status,
        "selector_normalized_wall_distance": descriptor.normalized_wall_distance,
        "selector_active": observed_active,
        "accepted_steps": int(result.accepted_step_times.size),
        "rejected_attempts": int(np.sum(result.interval_rejection_count)),
        "face_reconstruction_fallbacks": int(
            np.sum(result.interval_face_fallback_count)
        ),
        "minimum_density": result.minimum_density,
        "minimum_pressure": result.minimum_pressure,
        "maximum_interval_closure_relative_l2": float(
            np.max(result.interval_closure_relative_l2)
        ),
        "activity": {
            "high_fidelity_evolutions": 1,
            "a46_branch_predictions": 0,
            "checkpoint_calls": 0,
            "controllers_run": 0,
            "truth_outcomes_used_for_selection": 0,
        },
        "claim_limit": (
            "One generated dynamic-FV trajectory with literal common-source "
            "dual-resolution truth; no neural prediction, correction, recurrent "
            "benefit, or conservation-by-construction claim."
        ),
    }
    atomic_write_json(args.output_dir / "summary.json", summary)
    if status != "passed":
        raise RuntimeError(f"resource case failed contract checks: {args.case_id}")
    return summary


def _audit(args: argparse.Namespace) -> dict[str, Any]:
    if args.population_output.exists() or args.audit_output.exists():
        raise FileExistsError("population/audit output already exists")
    plan = load_resource_plan(args.plan, ROOT)
    opened = [load_json_object(path) for path in args.opened_source_manifest]
    opened_case_ids = collect_registered_case_ids(*opened)
    summaries: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    for case in plan["cases"]:
        case_root = args.artifact_root / case["case_id"]
        summary_path = case_root / "summary.json"
        artifact_path = case_root / "reference.npz"
        summary = load_json_object(summary_path)
        artifact_sha256 = sha256_file(artifact_path)
        if summary.get("reference_artifact_sha256") != artifact_sha256:
            raise ValueError(f"reference artifact digest mismatch: {case['case_id']}")
        summary_sha256 = sha256_file(summary_path)
        summaries.append({**summary, "summary_sha256": summary_sha256})
        artifact_rows.append(
            {
                "case_id": case["case_id"],
                "summary_sha256": summary_sha256,
                "reference_artifact_sha256": artifact_sha256,
                "reference_artifact_bytes": artifact_path.stat().st_size,
            }
        )
    population, population_report = build_population_manifest(
        plan,
        plan_file_sha256=sha256_file(args.plan),
        summary_rows=summaries,
        opened_case_ids=opened_case_ids,
    )
    _write_new_json(args.population_output, population)
    audit = {
        "schema": RESOURCE_AUDIT_SCHEMA,
        "working_id": RESOURCE_BUILD_WORKING_ID,
        "status": "passed",
        "resource_plan_sha256": sha256_file(args.plan),
        "resource_plan_payload_sha256": plan["payload_sha256"],
        "population_manifest_sha256": sha256_file(args.population_output),
        "population_payload_sha256": population["payload_sha256"],
        "population_report": population_report,
        "opened_case_count": len(opened_case_ids),
        "artifacts": artifact_rows,
        "activity": population["activity"],
        "claim_limit": (
            "Resource identity and unopened population metadata only; no A46 "
            "branch inference, fitted utility, or recurrent result."
        ),
    }
    _write_new_json(args.audit_output, audit)
    return audit


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "plan":
        payload = build_resource_plan(ROOT)
        _write_new_json(args.output_path, payload)
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return 0
    if args.command == "training-command":
        payload = _training_command(args)
        _write_new_json(args.output_path, payload)
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return 0
    if args.command == "preflight-training":
        payload = _preflight_training(args)
        _write_new_json(args.output_path, payload)
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return 2 if args.require_pass and payload["status"] != "passed" else 0
    if args.command == "generate-case":
        payload = _generate_case(args)
        print(
            json.dumps(
                {
                    "case_id": payload["case_id"],
                    "status": payload["status"],
                    "elapsed_seconds": payload["elapsed_seconds"],
                    "reference_artifact_sha256": payload["reference_artifact_sha256"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    payload = _audit(args)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
