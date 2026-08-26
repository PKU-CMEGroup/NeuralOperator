#!/usr/bin/env python3
"""Run one fresh CPU/FP64 P0-A2 native-solver process."""

from __future__ import annotations

import argparse
import platform
import sys
import traceback
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utility.time_dependent_no.p0_restart_sufficiency import (  # noqa: E402
    A0_ARTIFACT_MANIFEST_SHA256,
    CASE_MANIFEST_SHA256,
    COMMON_HORIZON,
    NATIVE_NX,
    NATIVE_NY,
    NATIVE_SOLVER_MANIFEST_SHA256,
    atomic_write_json,
    canonical_json_sha256,
    load_bound_json,
    load_json_object,
    prepare_solver_input,
    resolve_manifest_member,
    sha256_file,
    validate_artifact_manifest,
    validate_case_manifest,
    validate_execution_contract,
    validate_native_solver_manifest,
    verify_artifact_member,
)
from utility.time_dependent_no.p0_restart_sufficiency_a2 import (  # noqa: E402
    A1_CLOSEOUT_MANIFEST_SHA256,
    A2_SOLVER_ARTIFACT_SCHEMA,
    A2_SOLVER_SUMMARY_SCHEMA,
    PROCESS_ROLES,
    admissibility_summary,
    atomic_save_npy,
    boundary_balance_residual,
    bind_payload_sha256,
    build_artifact_manifest,
    case_contract,
    expected_solver_array_shape,
    integrated_conservative_change,
    prepare_fresh_directory,
    raw_array_sha256,
    require_exact_array_bytes,
    require_solver_balance,
    validate_a1_closeout,
    validate_a2_source_manifest,
)
from utility.time_dependent_no.shock_vortex_coarse_cfd import (  # noqa: E402
    CoarseCFDRollout,
    run_coarse_cfd_rollout,
)
from utility.time_dependent_no.shock_vortex_fv import (  # noqa: E402
    ShockVortexFVConfig,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-root", type=Path, required=True)
    parser.add_argument("--a1-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--process-role", choices=PROCESS_ROLES, required=True)
    parser.add_argument(
        "--owner-authorized-a2",
        action="store_true",
        help="Required acknowledgement of the owner's explicit P0-A2 authorization.",
    )
    return parser.parse_args(argv)


def _config() -> ShockVortexFVConfig:
    return ShockVortexFVConfig(
        nx=NATIVE_NX,
        ny=NATIVE_NY,
        coarse_nx=NATIVE_NX,
        coarse_ny=NATIVE_NY,
        t_final=COMMON_HORIZON,
        output_times=(0.0, COMMON_HORIZON),
    ).validated()


def _validate_rollout(result: CoarseCFDRollout, *, initial: np.ndarray) -> dict:
    if result.states.shape != expected_solver_array_shape():
        raise ValueError("native solver returned the wrong retained-state shape")
    if result.states.dtype != np.float64:
        raise ValueError("native solver did not return FP64 retained states")
    if result.interval_boundary_exchange.shape != (1, 4):
        raise ValueError("native solver returned the wrong boundary-exchange shape")
    if result.interval_boundary_exchange.dtype != np.float64:
        raise ValueError("native solver did not return FP64 boundary exchange")
    if result.rejected_attempts != 0:
        raise ValueError("native solver used an inadmissible-step retry")
    if result.face_reconstruction_fallbacks != 0:
        raise ValueError("native solver used a reconstructed-face fallback")
    physical = admissibility_summary(result.states.reshape(-1, 4))
    if not physical["finite"] or not physical["admissible"]:
        raise ValueError("native solver produced a nonfinite or inadmissible state")
    volumes = np.full(NATIVE_NX * NATIVE_NY, result.config.dx * result.config.dy)
    delta = integrated_conservative_change(result.states[-1], initial, volumes=volumes)
    exchange = np.asarray(result.interval_boundary_exchange[0], dtype=np.float64)
    residual = boundary_balance_residual(delta, exchange)
    require_solver_balance(residual)
    return {
        **physical,
        "integrated_conservative_change": delta.tolist(),
        "recorded_outward_boundary_exchange": exchange.tolist(),
        "boundary_balance_residual": residual,
    }


def _require_matching_calls(first: CoarseCFDRollout, second: CoarseCFDRollout) -> None:
    require_exact_array_bytes(first.states, second.states, label="solver states")
    require_exact_array_bytes(
        first.interval_boundary_exchange,
        second.interval_boundary_exchange,
        label="boundary exchange",
    )
    for name in (
        "accepted_steps",
        "rejected_attempts",
        "face_reconstruction_fallbacks",
    ):
        if getattr(first, name) != getattr(second, name):
            raise ValueError(f"same-process solver count differs: {name}")


def _run(args: argparse.Namespace) -> dict:
    if not args.owner_authorized_a2:
        raise ValueError("P0-A2 requires explicit owner authorization")
    inputs_root = args.inputs_root.resolve(strict=True)
    source_root = args.source_root.resolve(strict=True)
    a1_receipt = validate_a1_closeout(args.a1_root)

    if sha256_file(args.source_manifest) != args.expected_source_manifest_sha256:
        raise ValueError("A2 native source-manifest SHA-256 mismatch")
    source_payload = load_json_object(args.source_manifest)
    source_members = validate_a2_source_manifest(
        source_payload,
        source_root=source_root,
        source_role="native_solver",
    )

    artifact_path = resolve_manifest_member(
        inputs_root, "manifests/artifact_manifest.json"
    )
    artifact = load_bound_json(artifact_path, A0_ARTIFACT_MANIFEST_SHA256)
    artifact_members = validate_artifact_manifest(artifact)
    case_path = verify_artifact_member(
        inputs_root=inputs_root,
        artifact_members=artifact_members,
        relative_path="manifests/case_manifest.json",
    )
    if sha256_file(case_path) != CASE_MANIFEST_SHA256:
        raise ValueError("case manifest does not match the fixed A0 selection")
    cases = validate_case_manifest(load_json_object(case_path))
    execution_path = verify_artifact_member(
        inputs_root=inputs_root,
        artifact_members=artifact_members,
        relative_path="manifests/execution_contract.json",
    )
    validate_execution_contract(load_json_object(execution_path))
    native_manifest_path = verify_artifact_member(
        inputs_root=inputs_root,
        artifact_members=artifact_members,
        relative_path="manifests/native_solver_source_manifest.json",
    )
    native_manifest = load_bound_json(
        native_manifest_path, NATIVE_SOLVER_MANIFEST_SHA256
    )
    validate_native_solver_manifest(native_manifest, source_root=source_root)

    config = _config()
    config_sha256 = canonical_json_sha256(config.to_dict())
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

    rows: list[dict] = []
    started = perf_counter()
    for case in cases:
        relative = "data/full_trajectory_root/" + case["state_relative_path"]
        state_path = verify_artifact_member(
            inputs_root=inputs_root,
            artifact_members=artifact_members,
            relative_path=relative,
        )
        states = np.load(state_path, mmap_mode="r", allow_pickle=False)
        initial = prepare_solver_input(states[int(case["frame"])])
        if raw_array_sha256(initial) != case["input_frame_sha256"]:
            raise ValueError("selected solver input no longer matches the fixed frame")

        with torch.inference_mode():
            first = run_coarse_cfd_rollout(
                initial,
                config,
                device="cpu",
                dtype=torch.float64,
            )
            first_checks = _validate_rollout(first, initial=initial)
            second = run_coarse_cfd_rollout(
                initial,
                config,
                device="cpu",
                dtype=torch.float64,
            )
            _validate_rollout(second, initial=initial)
        _require_matching_calls(first, second)

        case_label = f"{case['trajectory_id']}_f{case['frame']}"
        states_relative = f"cases/{case_label}/solver_states.npy"
        exchange_relative = f"cases/{case_label}/boundary_exchange.npy"
        atomic_save_npy(args.output_dir / states_relative, first.states)
        atomic_save_npy(
            args.output_dir / exchange_relative,
            first.interval_boundary_exchange[0],
        )
        rows.append(
            {
                "trajectory_id": case["trajectory_id"],
                "frame": case["frame"],
                "stored_input_sha256": case["input_frame_sha256"],
                "solver_states_path": states_relative,
                "solver_states_raw_sha256": raw_array_sha256(first.states),
                "boundary_exchange_path": exchange_relative,
                "boundary_exchange_raw_sha256": raw_array_sha256(
                    first.interval_boundary_exchange[0]
                ),
                "same_process_state_bytes_identical": True,
                "same_process_exchange_bytes_identical": True,
                "accepted_steps": first.accepted_steps,
                "rejected_attempts": first.rejected_attempts,
                "face_fallbacks": first.face_reconstruction_fallbacks,
                "first_call_core_seconds": first.core_seconds,
                "repeat_call_core_seconds": second.core_seconds,
                **first_checks,
            }
        )

    summary = {
        "schema": A2_SOLVER_SUMMARY_SCHEMA,
        "stage": "P0-A2",
        "status": "pass",
        "process_role": args.process_role,
        "claim_boundary": (
            "Exactly three fixed open-validation native-grid stored-state restarts; "
            "no checkpoint, model, training, A3, test, or strength-OOD access."
        ),
        "authorization": {"a2": True, "a3": False},
        "activity": {
            "native_solver_calls": 2 * len(rows),
            "checkpoint_files_opened": False,
            "checkpoint_deserialized": False,
            "model_constructed": False,
            "training_executed": False,
            "selected_state_files_opened": len(rows),
        },
        "bindings": {
            "a0_artifact_manifest_sha256": A0_ARTIFACT_MANIFEST_SHA256,
            "a1_closeout_manifest_sha256": A1_CLOSEOUT_MANIFEST_SHA256,
            "a1_summary_payload_sha256": a1_receipt["summary_payload_sha256"],
            "source_manifest_sha256": sha256_file(args.source_manifest),
            "source_mapping_sha256": canonical_json_sha256(source_members),
            "solver_config_sha256": config_sha256,
        },
        "cases_contract": case_contract(),
        "solver": {
            "configuration": config.to_dict(),
            "input_dtype": "little-endian float32 stored bytes",
            "compute_dtype": "torch.float64",
            "output_dtype": "numpy.float64",
            "device": "cpu",
            "same_process_calls_per_case": 2,
        },
        "cases": rows,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "elapsed_seconds": perf_counter() - started,
    }
    summary = bind_payload_sha256(summary)
    atomic_write_json(args.output_dir / "summary.json", summary)
    manifest = build_artifact_manifest(
        args.output_dir, schema=A2_SOLVER_ARTIFACT_SCHEMA
    )
    atomic_write_json(args.output_dir / "artifact_manifest.json", manifest)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    prepare_fresh_directory(args.output_dir)
    try:
        summary = _run(args)
    except Exception as exc:
        failure = {
            "schema": A2_SOLVER_SUMMARY_SCHEMA,
            "stage": "P0-A2",
            "status": "failed",
            "process_role": args.process_role,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "authorization": {"a2": bool(args.owner_authorized_a2), "a3": False},
        }
        atomic_write_json(args.output_dir / "summary.json", failure)
        atomic_write_json(
            args.output_dir / "artifact_manifest.json",
            build_artifact_manifest(args.output_dir, schema=A2_SOLVER_ARTIFACT_SCHEMA),
        )
        print(f"P0-A2 solver process failed: {type(exc).__name__}: {exc}")
        return 1
    print(
        f"P0-A2 {summary['process_role']} pass: "
        f"{len(summary['cases'])} cases, payload={summary['payload_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
