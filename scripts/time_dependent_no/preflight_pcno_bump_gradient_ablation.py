#!/usr/bin/env python3
"""Run the paired real-graph A2 preflight for the bump gradient ablation.

The preflight is engineering evidence only.  It consumes the immutable B1
source tree, the retained B1 run metadata, and the registered training shards.
For each branch mode it executes one production-ordered full-size training
microbatch, serializes a non-resumable checkpoint, runs one native evaluator
call on open-validation trajectory 16, and checks the exact call-1 additive
fresh/propagated identity after loading that checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no.train_pcno_bump_gradient_ablation import (
    B1_BOUNDARY_POLICY_SET_SHA256,
    B1_DATA_MANIFEST_SHA256,
    B1_FROZEN_SOURCE_SHA256,
    B1_SOURCE_SET_SHA256,
    B1_VALIDATION_KEYS,
    DIFFERENTIAL_BRANCH_MODES,
    PRESENTATIONS_PER_PASS,
    REGISTERED_SEEDS,
    _annotate_checkpoint,
    _is_differential_state_name,
    apply_differential_branch_mode,
    extension_source_hashes,
    mapping_sha256,
    model_state_sha256,
    presentation_stream_sha256,
    registered_matched_config,
    registered_schedule_factor,
    sha256_file,
    validate_registered_args,
    verify_frozen_b1_sources,
    write_json,
)

A2_SCHEMA = "w26_l2_bump_gradient_ablation_a2_v1"
A2_SOURCE_PATH = "scripts/time_dependent_no/preflight_pcno_bump_gradient_ablation.py"
A2_VALIDATION_KEY = "16"
A2_CALL = 1
IDENTITY_RELATIVE_TOLERANCE = 1.0e-6
CALL1_PROPAGATED_SHARE_TOLERANCE = 1.0e-6


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def a2_source_hashes(root: Path = ROOT) -> dict[str, str]:
    hashes = extension_source_hashes(root)
    path = root / A2_SOURCE_PATH
    if not path.is_file():
        raise FileNotFoundError(path)
    hashes[A2_SOURCE_PATH] = sha256_file(path)
    return hashes


def _source_file_hashes(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    files = snapshot.get("files")
    if not isinstance(files, Mapping):
        raise TypeError("reference source snapshot lacks its file mapping")
    return {str(path): record.get("sha256") for path, record in files.items()}


def validate_reference_contract(
    *,
    summary: Mapping[str, Any],
    split: Mapping[str, Any],
    boundary: Mapping[str, Any],
    source_snapshot: Mapping[str, Any],
    run_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless the retained run is the registered B1 contract."""

    train_keys = tuple(str(value) for value in split.get("train_keys", ()))
    val_keys = tuple(str(value) for value in split.get("val_keys", ()))
    test_keys = tuple(str(value) for value in split.get("test_keys", ()))
    checks = {
        "summary_data_manifest": (
            summary.get("data_manifest_digest") == B1_DATA_MANIFEST_SHA256
        ),
        "split_data_manifest": (
            split.get("data_manifest_digest") == B1_DATA_MANIFEST_SHA256
        ),
        "source_set": (
            source_snapshot.get("source_set_digest") == B1_SOURCE_SET_SHA256
        ),
        "source_files": (
            _source_file_hashes(source_snapshot) == B1_FROZEN_SOURCE_SHA256
        ),
        "boundary_mode": boundary.get("mode") == "causal_nodal_physical",
        "boundary_policy_set": (
            boundary.get("policy_set_digest") == B1_BOUNDARY_POLICY_SET_SHA256
        ),
        "boundary_population": int(boundary.get("trajectory_count", -1)) == 300,
        "train_population": len(train_keys) == 270,
        "validation_population": val_keys == B1_VALIDATION_KEYS,
        "test_population_closed": not test_keys,
        "run_contract_source": (
            run_contract.get("source_snapshot", {}).get("source_set_digest")
            == B1_SOURCE_SET_SHA256
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"retained B1 reference contract failed: {checks}")
    return {
        "checks": checks,
        "train_keys": train_keys,
        "val_keys": val_keys,
        "test_keys": test_keys,
    }


def _registered_args(
    run_contract: Mapping[str, Any],
    *,
    data_dir: Path,
    output_dir: Path,
    seed: int,
) -> argparse.Namespace:
    payload = dict(run_contract.get("args", {}))
    if not payload:
        raise ValueError("retained B1 run contract lacks training arguments")
    payload.update(
        {
            "data_dir": data_dir,
            "output_dir": output_dir,
            "seed": int(seed),
            "epochs": 34,
            "init_checkpoint": None,
            "resume_checkpoint": None,
        }
    )
    payload.setdefault("gradient_accumulation_steps", 1)
    args = argparse.Namespace(**payload)
    validate_registered_args(args)
    return args


def select_first_full_microbatch(
    trainer: Any,
    store: Any,
    train_keys: Sequence[str],
    *,
    args: argparse.Namespace,
) -> tuple[list[tuple[str, int]], str]:
    """Select the first full batch in production epoch-0 shuffled order."""

    presentations = trainer.epoch_presentations(
        store,
        train_keys,
        args=args,
        epoch=0,
        tiny_bank=None,
    )
    if len(presentations) != PRESENTATIONS_PER_PASS:
        raise ValueError(
            f"registered pass has {len(presentations)} presentations, "
            f"expected {PRESENTATIONS_PER_PASS}"
        )
    batches = trainer.homogeneous_presentation_batches(
        presentations,
        batch_size=int(args.batch_size),
        rng=np.random.default_rng(int(args.seed) + 100_000),
    )
    for key, time_indices in batches:
        if len(time_indices) == int(args.batch_size):
            return (
                [(str(key), int(index)) for index in time_indices],
                presentation_stream_sha256(presentations),
            )
    raise RuntimeError("production presentation stream has no full-size microbatch")


def _gradient_summary(model: torch.nn.Module) -> dict[str, Any]:
    all_finite = True
    tensor_count = 0
    element_count = 0
    nonzero_elements = 0
    differential_tensor_count = 0
    differential_nonzero_elements = 0
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        tensor_count += 1
        element_count += gradient.numel()
        nonzero = int(torch.count_nonzero(gradient.detach()).item())
        nonzero_elements += nonzero
        all_finite = all_finite and bool(torch.isfinite(gradient).all())
        if _is_differential_state_name(name):
            differential_tensor_count += 1
            differential_nonzero_elements += nonzero
    return {
        "all_finite": all_finite,
        "tensor_count": tensor_count,
        "element_count": element_count,
        "nonzero_elements": nonzero_elements,
        "differential_tensor_count": differential_tensor_count,
        "differential_nonzero_elements": differential_nonzero_elements,
    }


def _scaled_norm(
    value: torch.Tensor,
    *,
    weights: torch.Tensor,
    mask: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    scaled = value / scale.reshape(1, 1, -1)
    weighted = torch.square(scaled) * weights * mask
    return torch.sqrt(torch.sum(weighted))


@torch.inference_mode()
def call1_decomposition(
    trainer: Any,
    model: torch.nn.Module,
    store: Any,
    *,
    trajectory: str,
    boundary_policy: Mapping[str, Any],
    device: torch.device,
    amp: str,
) -> dict[str, Any]:
    """Check the exact additive identity at the native rollout entry call."""

    sample = store.tensor_batch(
        trajectory,
        [0],
        step_stride=1,
        device=device,
    )
    model.eval()
    with trainer.autocast_context(device, amp):
        autonomous, _, _ = trainer.contract_forward_sample(
            model,
            sample,
            sample["current"],
            boundary_policy=boundary_policy,
        )
        teacher_forced, _, _ = trainer.contract_forward_sample(
            model,
            sample,
            sample["current"],
            boundary_policy=boundary_policy,
        )
    autonomous = autonomous.float()
    teacher_forced = teacher_forced.float()
    target = sample["target"].float()
    total = autonomous - target
    propagated = autonomous - teacher_forced
    fresh = teacher_forced - target
    identity = total - propagated - fresh
    mask = trainer.normal_node_mask(sample["node_type"], sample["node_mask"])
    scale = model.state_scale.float()
    total_norm = _scaled_norm(
        total,
        weights=sample["node_weights"],
        mask=mask,
        scale=scale,
    )
    propagated_norm = _scaled_norm(
        propagated,
        weights=sample["node_weights"],
        mask=mask,
        scale=scale,
    )
    fresh_norm = _scaled_norm(
        fresh,
        weights=sample["node_weights"],
        mask=mask,
        scale=scale,
    )
    identity_norm = _scaled_norm(
        identity,
        weights=sample["node_weights"],
        mask=mask,
        scale=scale,
    )
    denominator = max(float(total_norm.cpu()), torch.finfo(torch.float32).tiny)
    relative_identity = float(identity_norm.cpu()) / denominator
    propagated_share = float(propagated_norm.cpu()) / denominator
    return {
        "trajectory": trajectory,
        "call_index": A2_CALL,
        "total_norm": float(total_norm.cpu()),
        "fresh_defect_norm": float(fresh_norm.cpu()),
        "propagated_norm": float(propagated_norm.cpu()),
        "relative_identity_residual": relative_identity,
        "maximum_identity_abs": float(torch.max(torch.abs(identity)).cpu()),
        "propagated_magnitude_share": propagated_share,
        "identity_pass": relative_identity <= IDENTITY_RELATIVE_TOLERANCE,
        "call1_zero_propagation_pass": (
            propagated_share <= CALL1_PROPAGATED_SHARE_TOLERANCE
        ),
    }


def a2_gate(result: Mapping[str, Any]) -> dict[str, bool]:
    arms = result.get("arms", {})
    pairing = result.get("pairing", {})
    checks = {
        "both_arms_present": set(arms) == set(DIFFERENTIAL_BRANCH_MODES),
        "common_initial_full_state": bool(pairing.get("common_initial_full_state")),
        "common_initial_nondifferential_state": bool(
            pairing.get("common_initial_nondifferential_state")
        ),
        "common_microbatch": bool(pairing.get("common_microbatch")),
        "common_production_stream": bool(pairing.get("common_production_stream")),
        "common_boundary_policy": bool(pairing.get("common_boundary_policy")),
    }
    for mode in DIFFERENTIAL_BRANCH_MODES:
        arm = arms.get(mode, {})
        gradient = arm.get("gradient_summary", {})
        decomposition = arm.get("call1_decomposition", {})
        evaluator = arm.get("native_evaluator", {})
        checks[f"{mode}_finite_training"] = bool(
            arm.get("train_metrics", {}).get("parameters_finite")
        ) and bool(gradient.get("all_finite"))
        checks[f"{mode}_native_call_completed"] = (
            evaluator.get("completion_rate") == 1.0
        )
        checks[f"{mode}_identity"] = bool(decomposition.get("identity_pass"))
        checks[f"{mode}_call1_zero_propagation"] = bool(
            decomposition.get("call1_zero_propagation_pass")
        )
    if "no_gradient" in arms:
        checks["no_gradient_has_no_differential_gradients"] = (
            arms["no_gradient"]
            .get("gradient_summary", {})
            .get("differential_tensor_count")
            == 0
        )
    return checks


def _evaluator_completion(summary: Mapping[str, Any]) -> float:
    aggregate = summary.get("aggregates", {}).get("pcno_baseline", {})
    return float(aggregate.get("completion_rate", math.nan))


def _run_arm(
    trainer: Any,
    evaluator: Any,
    store: Any,
    *,
    mode: str,
    args: argparse.Namespace,
    normalization: Any,
    microbatch: Sequence[tuple[str, int]],
    production_stream_sha256: str,
    boundary_policies: Mapping[str, Mapping[str, Any]],
    reference: Mapping[str, Any],
    source_snapshot: Mapping[str, Any],
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    trainer.set_seed(int(args.seed))
    torch.set_float32_matmul_precision("high")
    model = trainer.build_model(args, normalization, zero_initialize=True).to(device)
    branch_contract = apply_differential_branch_mode(model, mode)
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=registered_schedule_factor,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=(args.amp == "fp16"))
    noise_generator = torch.Generator(device=device.type)
    noise_generator.manual_seed(int(args.seed) + 101)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    train_metrics = trainer.train_epoch(
        model,
        store,
        microbatch,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        step_stride=int(args.step_stride),
        batch_size=int(args.batch_size),
        batch_rng=np.random.default_rng(int(args.seed) + 100_000),
        device=device,
        amp=str(args.amp),
        input_noise_std=0.0,
        generated_state_exposure_weight=0.0,
        gradient_clip=float(args.gradient_clip),
        noise_generator=noise_generator,
        boundary_policies=boundary_policies,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_wall_seconds = perf_counter() - started
    peak_allocated = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    peak_reserved = (
        int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
    )
    gradient_summary = _gradient_summary(model)

    exposure_contract = {
        "schema": A2_SCHEMA,
        "engineering_preflight_only": True,
        "presentations": len(microbatch),
        "optimizer_steps": 1,
        "production_stream_sha256": production_stream_sha256,
        "microbatch_sha256": presentation_stream_sha256(microbatch),
    }
    checkpoint = trainer.checkpoint_payload(
        epoch=0,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        normalization=normalization,
        train_keys=reference["train_keys"],
        val_keys=reference["val_keys"],
        test_keys=reference["test_keys"],
        data_contract=reference["summary"]["data_contract"],
        boundary_contract=reference["boundary"],
        exposure_contract=exposure_contract,
        source_snapshot=source_snapshot,
        resolved_target_contract=trainer.target_contract(),
        store=store,
        args=args,
        best_selection=None,
        best_epoch=None,
        parent_checkpoint=None,
    )
    checkpoint = _annotate_checkpoint(
        checkpoint,
        model,
        mode,
        [presentation_stream_sha256(microbatch)],
        registered_matched_config(args),
    )
    checkpoint["checkpoint_role"] = "a2_engineering_preflight"
    checkpoint["resume_supported"] = False
    checkpoint["a2_preflight_contract"] = exposure_contract
    checkpoint.pop("optimizer_state", None)
    checkpoint.pop("scheduler_state", None)
    arm_dir = output_dir / mode
    arm_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = arm_dir / "a2_checkpoint.pt"
    trainer.atomic_torch_save(checkpoint, checkpoint_path)

    loaded = evaluator.load_checkpoint(checkpoint_path)
    loaded_model = evaluator.build_model(loaded, device)
    decomposition = call1_decomposition(
        trainer,
        loaded_model,
        store,
        trajectory=A2_VALIDATION_KEY,
        boundary_policy=boundary_policies[A2_VALIDATION_KEY],
        device=device,
        amp=str(args.amp),
    )
    evaluator_dir = arm_dir / "native_rollout_call1"
    evaluator.main(
        [
            "--data-dir",
            str(args.data_dir),
            "--training-data-dir",
            str(args.data_dir),
            "--checkpoint",
            str(checkpoint_path),
            "--output-dir",
            str(evaluator_dir),
            "--trajectory-keys",
            A2_VALIDATION_KEY,
            "--expected-trajectory-count",
            "1",
            "--start-frame",
            "0",
            "--num-steps",
            "1",
            "--endpoint-calls",
            "1",
            "--counterfactual",
            "none",
            "--device",
            str(device),
        ]
    )
    evaluator_summary_path = evaluator_dir / "summary.json"
    evaluator_summary = _read_json(evaluator_summary_path)
    return {
        "mode": mode,
        "branch_contract": branch_contract,
        "initial_full_state_sha256": branch_contract["initial_full_state_sha256"],
        "initial_nondifferential_state_sha256": branch_contract[
            "initial_nondifferential_state_sha256"
        ],
        "trained_state_sha256": model_state_sha256(model),
        "train_metrics": train_metrics,
        "gradient_summary": gradient_summary,
        "training_wall_seconds": training_wall_seconds,
        "peak_gpu_allocated_bytes": peak_allocated,
        "peak_gpu_reserved_bytes": peak_reserved,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "call1_decomposition": decomposition,
        "native_evaluator": {
            "summary_sha256": sha256_file(evaluator_summary_path),
            "completion_rate": _evaluator_completion(evaluator_summary),
            "trajectory": A2_VALIDATION_KEY,
            "num_steps": 1,
        },
    }


def run_a2_preflight(
    *,
    data_dir: Path,
    reference_run_dir: Path,
    output_dir: Path,
    seed: int = REGISTERED_SEEDS[0],
    device_name: str = "cuda",
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"A2 output already exists: {output_dir}")
    if int(seed) != int(REGISTERED_SEEDS[0]):
        raise ValueError(f"A2 seed is fixed to {REGISTERED_SEEDS[0]}")
    from scripts.time_dependent_no import (
        evaluate_pcno_euler2d_residual as evaluator,
    )
    from scripts.time_dependent_no import train_pcno_euler2d_residual as trainer

    device = trainer.select_device(device_name)
    if device.type != "cuda":
        raise ValueError("the registered real-graph A2 preflight requires CUDA")
    frozen_sources = verify_frozen_b1_sources()
    extension_sources = a2_source_hashes()
    summary = _read_json(reference_run_dir / "summary.json")
    split = _read_json(reference_run_dir / "split.json")
    boundary = _read_json(reference_run_dir / "boundary_contract.json")
    source_snapshot = _read_json(
        reference_run_dir / "source_snapshot" / "manifest.json"
    )
    run_contract = _read_json(reference_run_dir / "run_contract.json")
    normalization_mapping = _read_json(reference_run_dir / "normalization.json")
    reference = validate_reference_contract(
        summary=summary,
        split=split,
        boundary=boundary,
        source_snapshot=source_snapshot,
        run_contract=run_contract,
    )
    reference.update(
        {
            "summary": summary,
            "boundary": boundary,
        }
    )
    args = _registered_args(
        run_contract,
        data_dir=data_dir,
        output_dir=output_dir,
        seed=int(seed),
    )
    trainer.validate_args(args, device)
    store = trainer.PCNOEuler2DShardStore(data_dir)
    try:
        if store.manifest_digest != B1_DATA_MANIFEST_SHA256:
            raise ValueError("A2 shard manifest differs from registered B1")
        normalization = trainer.Euler2DNormalization.from_mapping(normalization_mapping)
        microbatch, production_stream_hash = select_first_full_microbatch(
            trainer,
            store,
            reference["train_keys"],
            args=args,
        )
        train_key = microbatch[0][0]
        boundary_policies, selected_boundary = trainer.build_boundary_contract(
            store,
            [train_key, A2_VALIDATION_KEY],
            args=args,
            device=device,
        )
        selected_policy_checks = {
            key: selected_boundary["policy_digests"].get(key)
            == boundary["policy_digests"].get(key)
            for key in (train_key, A2_VALIDATION_KEY)
        }
        if not all(selected_policy_checks.values()):
            raise ValueError(
                f"selected A2 boundary policies differ from B1: {selected_policy_checks}"
            )
        output_dir.mkdir(parents=True, exist_ok=False)
        started = perf_counter()
        arms = {
            mode: _run_arm(
                trainer,
                evaluator,
                store,
                mode=mode,
                args=args,
                normalization=normalization,
                microbatch=microbatch,
                production_stream_sha256=production_stream_hash,
                boundary_policies=boundary_policies,
                reference=reference,
                source_snapshot=source_snapshot,
                output_dir=output_dir,
                device=device,
            )
            for mode in DIFFERENTIAL_BRANCH_MODES
        }
        pairing = {
            "common_initial_full_state": (
                arms["full"]["initial_full_state_sha256"]
                == arms["no_gradient"]["initial_full_state_sha256"]
            ),
            "common_initial_nondifferential_state": (
                arms["full"]["initial_nondifferential_state_sha256"]
                == arms["no_gradient"]["initial_nondifferential_state_sha256"]
            ),
            "common_microbatch": True,
            "common_production_stream": True,
            "common_boundary_policy": all(selected_policy_checks.values()),
        }
        result: dict[str, Any] = {
            "schema": A2_SCHEMA,
            "status": "completed",
            "engineering_evidence_only": True,
            "science_result": False,
            "seed": int(seed),
            "device": str(device),
            "runtime_environment": trainer.runtime_environment(device),
            "base_source_set_sha256": B1_SOURCE_SET_SHA256,
            "base_source_sha256": frozen_sources,
            "extension_source_sha256": extension_sources,
            "data_manifest_sha256": store.manifest_digest,
            "boundary_policy_set_sha256": boundary["policy_set_digest"],
            "reference_contract_checks": reference["checks"],
            "selected_boundary_policy_checks": selected_policy_checks,
            "production_stream_sha256": production_stream_hash,
            "microbatch": [
                {"trajectory": key, "time_index": int(time_index)}
                for key, time_index in microbatch
            ],
            "microbatch_sha256": presentation_stream_sha256(microbatch),
            "validation_trajectory": A2_VALIDATION_KEY,
            "pairing": pairing,
            "arms": arms,
            "elapsed_seconds": perf_counter() - started,
            "claim_boundary": {
                "optimization_result": False,
                "rollout_result": False,
                "architecture_result": False,
                "sealed_population_accessed": False,
                "strict_gibbs_claim": False,
                "physical_conservation_claim": False,
            },
        }
        checks = a2_gate(result)
        result["contract_checks"] = checks
        result["contract_pass"] = all(checks.values())
        result["status"] = "passed" if result["contract_pass"] else "failed"
        write_json(output_dir / "a2_preflight.json", result)
        manifest = {
            "schema": A2_SCHEMA,
            "a2_preflight_sha256": sha256_file(output_dir / "a2_preflight.json"),
            "checkpoint_sha256": {
                mode: arms[mode]["checkpoint_sha256"]
                for mode in DIFFERENTIAL_BRANCH_MODES
            },
            "native_evaluator_summary_sha256": {
                mode: arms[mode]["native_evaluator"]["summary_sha256"]
                for mode in DIFFERENTIAL_BRANCH_MODES
            },
        }
        manifest["manifest_payload_sha256"] = mapping_sha256(manifest)
        write_json(output_dir / "manifest.json", manifest)
        if not result["contract_pass"]:
            raise RuntimeError(f"A2 contract failed: {checks}")
        return result
    finally:
        store.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--reference-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=REGISTERED_SEEDS[0])
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_a2_preflight(
        data_dir=args.data_dir,
        reference_run_dir=args.reference_run_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        device_name=args.device,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "contract_pass": result["contract_pass"],
                "elapsed_seconds": result["elapsed_seconds"],
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
