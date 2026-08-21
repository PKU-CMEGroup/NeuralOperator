#!/usr/bin/env python3
"""Run the registered full-PCNO arm of the matched H320 comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no.evaluate_pcfno_h320_rollout import (
    ANIMATION_KEYS,
    EVENT_NAMES,
    PRODUCTION_STEPS,
    THRESHOLDS,
    TRAJECTORY_KEYS,
    TRUTH_STEPS,
    build_calibration_contract,
    build_final_hash_manifest,
    run_case,
    survival_rows,
    verify_final_hash_manifest,
    write_csv,
    write_json,
    write_npz_atomic,
)
from scripts.time_dependent_no.evaluate_pcno_euler2d_residual import (
    PCNOEuler2DShardStore,
    build_model,
    load_checkpoint,
    preprocessing_contract_audit,
    select_device,
    sha256_file,
)
from scripts.time_dependent_no.train_pcno_bump_gradient_ablation import (
    B1_DATA_MANIFEST_SHA256,
    B1_SOURCE_SET_SHA256,
    checkpoint_differential_branch_mode,
    verify_frozen_b1_sources,
)

SCHEMA = "w26_l1_pcno_h320_rollout_v2"
ANIMATION_SCHEMA = "w26_l1_pcno_h320_animation_bundle_v2"
WORKING_ID = "W26-L1-PCNO-H320-A1R1-S20260718"
CHECKPOINT_SHA256 = "bd7b580ef882432215473eeff1ea8a672c614bb3ea72f2659d2b5d25bc259a39"
PREFIX_SUMMARY_SHA256 = (
    "09bdeea0e2fcc7a7e2c90243e81545b7cfa25f0448f7b380fb1e3679c82bca0f"
)
PREFIX_INVENTORY_SHA256 = (
    "2f497b44475e41b1b1d06a2de34f3f69536fa845e01450cb21a7b9247abaa6fb"
)
PCFNO_CALIBRATION_SHA256 = (
    "568c489bf237eca73c040fe2f5149f1caeb5974272414cc188f2630ca40e99b2"
)
PREFIX_REPLAY_DIAGNOSTIC_SHA256 = (
    "26badfc6adfa2e5444d037768e52e3bc19f3a97caac7789a6c8dfedaa03fb0d2"
)

SOURCE_PATHS = (
    "docs/time_dependent_no/W26_L1_PCFNO_PCNO_H320_COMPARISON_PREREGISTRATION.md",
    "scripts/time_dependent_no/evaluate_pcno_h320_rollout.py",
    "scripts/time_dependent_no/visualize_pcfno_pcno_h320_comparison.py",
    "tests/time_dependent_no/test_pcfno_pcno_h320_comparison.py",
    "scripts/time_dependent_no/evaluate_pcfno_h320_rollout.py",
    "scripts/time_dependent_no/evaluate_pcno_long_horizon_stability.py",
    "scripts/time_dependent_no/evaluate_pcno_euler2d_residual.py",
    "scripts/time_dependent_no/train_pcno_bump_gradient_ablation.py",
    "utility/time_dependent_no/pcno_inadmissibility.py",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "production"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--training-data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prefix-rollout-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    return args


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def source_hashes() -> dict[str, str]:
    missing = [path for path in SOURCE_PATHS if not (ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"registered source files are missing: {missing}")
    return {path: sha256_file(ROOT / path) for path in SOURCE_PATHS}


def canonical_prefix_inventory(prefix_root: Path) -> dict[str, Any]:
    trajectory_root = prefix_root / "trajectories"
    if not trajectory_root.is_dir():
        raise FileNotFoundError(trajectory_root)
    expected_names = sorted(f"trajectory_{key}.npz" for key in TRAJECTORY_KEYS)
    observed_names = sorted(path.name for path in trajectory_root.glob("*.npz"))
    if observed_names != expected_names:
        raise ValueError(
            "full-PCNO prefix trajectory inventory changed; "
            f"expected={expected_names}, observed={observed_names}"
        )
    rows: list[dict[str, Any]] = []
    digest_lines: list[str] = []
    for name in expected_names:
        path = trajectory_root / name
        digest = sha256_file(path)
        rows.append({"path": name, "size": path.stat().st_size, "sha256": digest})
        digest_lines.append(f"{digest}  {name}\n")
    digest = hashlib.sha256("".join(digest_lines).encode("utf-8")).hexdigest()
    return {"sha256": digest, "files": rows}


def verify_prefix_root(prefix_root: Path) -> dict[str, Any]:
    summary_path = prefix_root / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary_sha256 = sha256_file(summary_path)
    if summary_sha256 != PREFIX_SUMMARY_SHA256:
        raise ValueError("full-PCNO H79 prefix summary SHA-256 changed")
    inventory = canonical_prefix_inventory(prefix_root)
    if inventory["sha256"] != PREFIX_INVENTORY_SHA256:
        raise ValueError("full-PCNO H79 trajectory inventory SHA-256 changed")
    return {
        "summary_sha256": summary_sha256,
        "trajectory_inventory_sha256": inventory["sha256"],
        "trajectory_file_count": len(inventory["files"]),
        "files": {row["path"]: row for row in inventory["files"]},
    }


def expected_prefix(
    key: str,
    *,
    prefix_root: Path,
    prefix_contract: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    name = f"trajectory_{key}.npz"
    path = prefix_root / "trajectories" / name
    required = {
        "schema",
        "trajectory_key",
        "initial_conservative",
        "pcno_baseline_predictions_conservative",
        "checkpoint_sha256",
        "test_manifest_digest",
        "training_manifest_digest",
        "baseline_valid_length",
        "baseline_failure_cause",
        "baseline_failure_call",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"full-PCNO prefix {name} lacks arrays: {missing}")
        if (
            str(np.asarray(archive["schema"]).item())
            != "pcno_euler2d_official_rollout_v1"
        ):
            raise ValueError(f"full-PCNO prefix {name} has an unsupported schema")
        if str(np.asarray(archive["trajectory_key"]).item()) != key:
            raise ValueError(f"full-PCNO prefix key changed for {name}")
        if str(np.asarray(archive["checkpoint_sha256"]).item()) != CHECKPOINT_SHA256:
            raise ValueError(f"full-PCNO prefix checkpoint changed for {name}")
        for field in ("test_manifest_digest", "training_manifest_digest"):
            if str(np.asarray(archive[field]).item()) != B1_DATA_MANIFEST_SHA256:
                raise ValueError(f"full-PCNO prefix {field} changed for {name}")
        if int(np.asarray(archive["baseline_valid_length"]).item()) != TRUTH_STEPS:
            raise ValueError(f"full-PCNO prefix length changed for {name}")
        if str(np.asarray(archive["baseline_failure_cause"]).item()) != "completed":
            raise ValueError(f"full-PCNO H79 prefix was not completed for {name}")
        if int(np.asarray(archive["baseline_failure_call"]).item()) != -1:
            raise ValueError(f"full-PCNO H79 prefix has a failure call for {name}")
        initial = np.array(archive["initial_conservative"], copy=True)
        predictions = np.array(
            archive["pcno_baseline_predictions_conservative"], copy=True
        )
    if initial.ndim != 2 or initial.shape[-1] != 4:
        raise ValueError(f"full-PCNO initial state shape changed for {name}")
    if predictions.shape != (TRUTH_STEPS, *initial.shape):
        raise ValueError(f"full-PCNO H79 prediction shape changed for {name}")
    if initial.dtype != np.float32 or predictions.dtype != np.float32:
        raise ValueError(f"full-PCNO prefix dtype changed for {name}")
    file_record = prefix_contract["files"].get(name)
    if not isinstance(file_record, Mapping):
        raise TypeError(f"full-PCNO prefix inventory lacks a record for {name}")
    prefix = np.concatenate((initial[None], predictions), axis=0)
    return prefix, {
        "source": "matched_full_pcno_h79_trajectory",
        "path_name": name,
        "source_sha256": file_record["sha256"],
        "source_size": file_record["size"],
        "expected_calls": TRUTH_STEPS,
    }


def full_gradient_audit(model: torch.nn.Module) -> dict[str, Any]:
    backbone = getattr(model, "backbone", model)
    rows = []
    for layer, module in enumerate(backbone.gws):
        weight = module.gw2.weight.detach()
        rows.append(
            {
                "layer": layer,
                "element_count": int(weight.numel()),
                "nonzero_count": int(torch.count_nonzero(weight).cpu()),
                "max_abs": float(torch.max(torch.abs(weight)).cpu()),
            }
        )
    if len(rows) != 4 or any(row["nonzero_count"] == 0 for row in rows):
        raise ValueError("registered full PCNO lacks an active learned gw2 layer")
    return {"all_gw2_layers_nonzero": True, "layers": rows}


def _patch_full_identity(
    case: dict[str, Any], bundle: dict[str, np.ndarray] | None
) -> None:
    case["claim_boundary"] = (
        "reference-free descriptive full-PCNO recurrence; not accuracy, physical "
        "validity, conservation, asymptotic stability, or a seed-general gradient claim"
    )
    if bundle is None:
        return
    bundle["schema"] = np.asarray(ANIMATION_SCHEMA)
    bundle["working_id"] = np.asarray(WORKING_ID)
    bundle["checkpoint_sha256"] = np.asarray(CHECKPOINT_SHA256)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    horizon = TRUTH_STEPS + 2 if args.mode == "smoke" else PRODUCTION_STEPS
    keys = ("187",) if args.mode == "smoke" else TRAJECTORY_KEYS
    retained_keys = ("187",) if args.mode == "smoke" else ANIMATION_KEYS
    device = select_device(args.device)
    if args.mode == "production" and device.type != "cuda":
        raise RuntimeError("registered production H320 requires CUDA")

    frozen_sources = verify_frozen_b1_sources()
    checkpoint_hash = sha256_file(args.checkpoint)
    if checkpoint_hash != CHECKPOINT_SHA256:
        raise ValueError("checkpoint SHA-256 differs from the registered full PCNO")
    checkpoint = load_checkpoint(args.checkpoint)
    if checkpoint_differential_branch_mode(checkpoint) != "full":
        raise ValueError("checkpoint is not the registered full-PCNO mode")
    if int(checkpoint["step_stride"]) != 1:
        raise ValueError("registered full PCNO requires step_stride=1")

    training_store = PCNOEuler2DShardStore(args.training_data_dir)
    test_store = PCNOEuler2DShardStore(args.data_dir)
    if training_store.manifest_digest != B1_DATA_MANIFEST_SHA256:
        raise ValueError("training data manifest differs from registered B1")
    missing = sorted(set(TRAJECTORY_KEYS) - set(test_store.keys))
    if missing:
        raise KeyError(f"test store lacks registered trajectories: {missing}")
    preprocessing = preprocessing_contract_audit(checkpoint, training_store, test_store)
    prefix_contract = verify_prefix_root(args.prefix_rollout_dir)
    model = build_model(checkpoint, device)
    model.eval()
    gradient_audit = full_gradient_audit(model)
    state_scale = np.asarray(
        checkpoint["normalization"]["state_scale"], dtype=np.float64
    )
    calibration = build_calibration_contract(
        test_store,
        TRAJECTORY_KEYS,
        gamma=float(model.gamma),
        state_scale=state_scale,
    )
    sources = source_hashes()

    args.output_dir.mkdir(parents=True)
    run_contract = {
        "schema": SCHEMA,
        "working_id": WORKING_ID,
        "status": "running",
        "mode": args.mode,
        "requested_horizon": horizon,
        "trajectory_keys": list(keys),
        "full_registered_population": list(TRAJECTORY_KEYS),
        "animation_keys": list(retained_keys),
        "precision": "fp32_batch_one",
        "accuracy_evaluated": False,
        "truth_free_calls": [80, PRODUCTION_STEPS],
        "recurrence": (
            "exact SHA-bound retained H0-H79 prefix; resume from deployed H79; "
            "input causal boundary; full PCNO; output causal boundary; deployed "
            "feedback; finite invalid continues; returned nonfinite stops"
        ),
        "retained_prefix_resume": {
            "source_calls": [0, TRUTH_STEPS],
            "first_new_call": TRUTH_STEPS + 1,
            "new_call_count": horizon - TRUTH_STEPS,
            "prefix_replayed": False,
            "diagnostic_sha256": PREFIX_REPLAY_DIAGNOSTIC_SHA256,
        },
        "checkpoint": {
            "sha256": checkpoint_hash,
            "differential_branch_mode": "full",
        },
        "data": {
            "training_manifest_digest": training_store.manifest_digest,
            "evaluation_manifest_digest": test_store.manifest_digest,
        },
        "frozen_b1_source_set_sha256": B1_SOURCE_SET_SHA256,
        "frozen_b1_sources": frozen_sources,
        "extension_source_sha256": sources,
        "preprocessing": preprocessing,
        "prefix_contract": {
            key: value for key, value in prefix_contract.items() if key != "files"
        },
        "gradient_audit": gradient_audit,
        "thresholds": THRESHOLDS.as_dict(),
        "calibration_payload_sha256": calibration["payload_sha256"],
        "expected_pcfno_calibration_file_sha256": PCFNO_CALIBRATION_SHA256,
    }
    write_json(args.output_dir / "run_contract.json", run_contract)
    write_json(args.output_dir / "calibration_contract.json", calibration)

    all_rows: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    for key in keys:
        prefix, prefix_record = expected_prefix(
            key,
            prefix_root=args.prefix_rollout_dir,
            prefix_contract=prefix_contract,
        )
        rows, case, bundle = run_case(
            model,
            test_store,
            key,
            horizon=horizon,
            checkpoint=checkpoint,
            device=device,
            prefix=prefix,
            prefix_record=prefix_record,
            calibration=calibration,
            retain_states=key in retained_keys,
            resume_from_retained_prefix=True,
        )
        _patch_full_identity(case, bundle)
        all_rows.extend(rows)
        cases.append(case)
        if bundle is not None:
            write_npz_atomic(args.output_dir / f"trajectory_{key}.npz", bundle)
        print(
            json.dumps(
                {
                    "trajectory": key,
                    "recorded_calls": case["recorded_call_count"],
                    "terminal_events": case["terminal_events"],
                    "first_failures": {
                        name: case["events"][name]["first_failure_call"]
                        for name in EVENT_NAMES
                    },
                },
                sort_keys=True,
            ),
            flush=True,
        )

    case_rows: list[dict[str, Any]] = []
    for case in cases:
        row: dict[str, Any] = {
            "trajectory": case["trajectory"],
            "recorded_call_count": case["recorded_call_count"],
            "termination_call": case["termination_call"],
            "terminal_admissible": case["terminal_events"]["admissible"],
            "terminal_bounded": case["terminal_events"]["bounded"],
            "terminal_finite": case["terminal_events"]["finite"],
            "terminal_common_amplitude_ratio": case["terminal_common_amplitude_ratio"],
            "terminal_common_scaled_rms_ratio": case[
                "terminal_common_scaled_rms_ratio"
            ],
            "terminal_min_density": case["terminal_min_density"],
            "terminal_min_internal_energy": case["terminal_min_internal_energy"],
            "terminal_min_pressure": case["terminal_min_pressure"],
            "maximum_common_amplitude_ratio": case["maximum_common_amplitude_ratio"],
            "maximum_common_scaled_rms_ratio": case["maximum_common_scaled_rms_ratio"],
            "maximum_normalizer_excursion": case["maximum_normalizer_excursion"],
            "model_recovery_calls": case["model_recovery_calls"],
            "output_boundary_recovery_calls": case["output_boundary_recovery_calls"],
            "prefix_calls_checked": case["prefix"]["checked_calls"],
            "prefix_bitwise_equal": case["prefix"]["all_checked_calls_bitwise_equal"],
            "prefix_source_bound": case["prefix"]["source_bound"],
            "prefix_resume_state_sha256": case["prefix"]["resume_state_sha256"],
        }
        for event_name in EVENT_NAMES:
            event = case["events"][event_name]
            row[f"{event_name}_first_failure_call"] = event["first_failure_call"]
            row[f"{event_name}_accepted_prefix_calls"] = event["accepted_prefix_calls"]
            row[f"{event_name}_recovered_after_failure"] = event[
                "recovered_after_failure"
            ]
        case_rows.append(row)

    write_csv(args.output_dir / "call_metrics.csv", all_rows)
    write_csv(args.output_dir / "case_metrics.csv", case_rows)
    write_csv(args.output_dir / "survival.csv", survival_rows(cases, horizon))
    summary = {
        "schema": SCHEMA,
        "working_id": WORKING_ID,
        "status": "completed",
        "mode": args.mode,
        "requested_horizon": horizon,
        "trajectory_count": len(cases),
        "accuracy_evaluated": False,
        "terminal_counts": {
            name: sum(bool(case["terminal_events"][name]) for case in cases)
            for name in EVENT_NAMES
        },
        "ever_failed_counts": {
            name: sum(
                case["events"][name]["first_failure_call"] is not None for case in cases
            )
            for name in EVENT_NAMES
        },
        "all_prefixes_bitwise_equal": None,
        "all_prefixes_source_bound": all(
            case["prefix"]["source_bound"] for case in cases
        ),
        "cases": {case["trajectory"]: case for case in cases},
        "claim_boundary": (
            "H80-H320 is reference-free descriptive recurrence only; no accuracy, "
            "defect, physical conservation, physical validity, seed-general "
            "gradient-ablation, or asymptotic-stability claim"
        ),
    }
    write_json(args.output_dir / "summary.json", summary)
    run_contract["status"] = "completed"
    write_json(args.output_dir / "run_contract.json", run_contract)

    artifact_names = [
        "run_contract.json",
        "calibration_contract.json",
        "call_metrics.csv",
        "case_metrics.csv",
        "survival.csv",
        "summary.json",
        *[f"trajectory_{key}.npz" for key in retained_keys],
    ]
    manifest = build_final_hash_manifest(args.output_dir, artifact_names)
    write_json(args.output_dir / "final_hash_manifest.json", manifest)
    verification = verify_final_hash_manifest(args.output_dir, manifest)
    summary["final_hash_manifest"] = {
        "sha256": sha256_file(args.output_dir / "final_hash_manifest.json"),
        **verification,
    }
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run(args)
    print(json.dumps(_json_safe(summary), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
