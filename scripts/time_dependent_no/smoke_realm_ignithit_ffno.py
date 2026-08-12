"""Run the frozen D088 P1c FFNO-M engineering smoke on one CUDA device.

The command verifies the exact open IgnitHIT tree, performs one synthetic
forward/backward probe, accumulates one effective batch of all 26 train cases
into exactly one optimizer step, and rolls the first registered validation case
through all 29 released transitions. It never loads or discovers a test object,
saves no checkpoint, and is not a persistent training entry point.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utility.time_dependent_no.realm_benchmark import (
    IGNITHIT_OPEN_MANIFEST_SHA256,
    MagnitudeEnvelope,
    RealmNormalizer,
    canonical_json_sha256,
    decoded_admissibility,
    decoded_boundedness,
    decoded_spatial_pearson,
    grouped_normalized_prediction_error,
    parse_manifest_payload,
    validate_ignithit_open_manifest,
)
from utility.time_dependent_no.realm_ffno import (
    RealmFFNO2d,
    RealmFFNOConfig,
    grouped_next_state_mse,
    normalize_realm_coordinates,
    parameter_count_within_reported_tolerance,
    trainable_parameter_count,
)
from utility.time_dependent_no.realm_ignithit import (
    BOUNDEDNESS_EXPANSION_FACTOR,
    BOUNDEDNESS_QUANTILE,
    BOX_COX_LAMBDA,
    PRIMARY_BOX_COX_EPSILON,
    SCALE_STABILIZER,
    STD_CORRECTION,
    TRAIN_GROUPS,
    VAL_GROUPS,
    load_ignithit_metadata,
    load_ignithit_trajectory,
    sha256_file,
    trajectory_relative_path,
    validate_local_open_tree,
)

RUN_ID = "d088_realm_ignithit_p1c_personalgpu_20260812a"
SEED = 20_260_812
DEVICE = "cuda:0"
DTYPE = torch.float32
MICROBATCH_SIZE = 1
EFFECTIVE_BATCH_SIZE = 26
ACCUMULATION_STEPS = 26
TRAIN_INPUT_FRAME = 0
TRAIN_TARGET_FRAME = 1
VALIDATION_GROUP = "phi=_t_15_3_t"
VALIDATION_HORIZON = 29
OPTIMIZER_LR = 1.0e-3
OPTIMIZER_WEIGHT_DECAY = 0.0
OPTIMIZER_BETAS = (0.9, 0.999)
OPTIMIZER_EPS = 1.0e-8
MAX_WALL_SECONDS = 900
NORMALIZER_ARRAYS_SHA256 = (
    "368d243b5e0f71b6380ee5f49fb9f5cf2724baa94ddece620d3ceae55285ca20"
)
NORMALIZER_ARRAY_KEYS = frozenset(
    {
        "primary_mean",
        "primary_std",
        "primary_scale",
        "source_sensitivity_mean",
        "source_sensitivity_std",
        "source_sensitivity_scale",
        "raw_train_mean",
        "raw_train_std",
        "train_max_abs",
    }
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--normalizer-arrays", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    return parser


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("manifest root must be a JSON object")
    return payload


def _is_within(child: Path, parent: Path) -> bool:
    child_resolved = child.resolve(strict=False)
    parent_resolved = parent.resolve(strict=False)
    return (
        child_resolved == parent_resolved or parent_resolved in child_resolved.parents
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _finite_or_none(value: float) -> float | None:
    return value if np.isfinite(value) else None


def _require_budget(started_at: float) -> None:
    if time.monotonic() - started_at > MAX_WALL_SECONDS:
        raise RuntimeError("P1c smoke exceeded its 900-second end-to-end cap")


def _configure_determinism() -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def _normalizer_from_arrays(
    path: Path,
) -> tuple[RealmNormalizer, torch.Tensor, dict[str, Any]]:
    if sha256_file(path) != NORMALIZER_ARRAYS_SHA256:
        raise ValueError("normalizer arrays SHA-256 does not match frozen P1b")
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != NORMALIZER_ARRAY_KEYS:
            raise ValueError("normalizer array keys do not match the P1b contract")
        mean = np.asarray(payload["primary_mean"])
        scale = np.asarray(payload["primary_scale"])
        train_max_abs = np.asarray(payload["train_max_abs"])
    for name, value in (
        ("mean", mean),
        ("scale", scale),
        ("train_max_abs", train_max_abs),
    ):
        if value.shape != (12,) or not np.isfinite(value).all():
            raise ValueError(f"normalizer {name} must be a finite 12-channel vector")
    if not (scale > 0.0).all() or not (train_max_abs >= 0.0).all():
        raise ValueError("normalizer scale/envelope domain is invalid")
    normalizer = RealmNormalizer(
        mean=torch.as_tensor(mean, dtype=DTYPE),
        scale=torch.as_tensor(scale, dtype=DTYPE),
        transformed_channels=tuple(range(8)),
        channel_axis=1,
        box_cox_lambda=BOX_COX_LAMBDA,
        box_cox_epsilon=PRIMARY_BOX_COX_EPSILON,
        std_correction=STD_CORRECTION,
        scale_stabilizer=SCALE_STABILIZER,
    )
    summary = {
        "sha256": NORMALIZER_ARRAYS_SHA256,
        "primary_box_cox_epsilon": PRIMARY_BOX_COX_EPSILON,
        "box_cox_lambda": BOX_COX_LAMBDA,
        "channel_axis_for_model_tensors": 1,
    }
    return normalizer, torch.as_tensor(train_max_abs, dtype=DTYPE), summary


def _source_manifest() -> dict[str, Any]:
    paths = (
        REPO_ROOT / "utility" / "time_dependent_no" / "realm_benchmark.py",
        REPO_ROOT / "utility" / "time_dependent_no" / "realm_ignithit.py",
        REPO_ROOT / "utility" / "time_dependent_no" / "realm_ffno.py",
        Path(__file__).resolve(),
    )
    rows = [
        {
            "path": path.relative_to(REPO_ROOT).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    payload: dict[str, Any] = {
        "schema": "d088_ignithit_p1c_executed_source_v1",
        "files": rows,
    }
    payload["canonical_payload_sha256"] = canonical_json_sha256(payload)
    return payload


def _runtime_manifest(device: torch.device) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    payload: dict[str, Any] = {
        "schema": "d088_ignithit_p1c_personal_gpu_runtime_v1",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device_name": properties.name,
        "device_total_memory_bytes": properties.total_memory,
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "dtype": "float32",
        "autocast_enabled": False,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "seed": SEED,
    }
    payload["canonical_payload_sha256"] = canonical_json_sha256(payload)
    return payload


def _phase_memory(device: torch.device) -> dict[str, int]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "free_bytes_after_phase": free_bytes,
        "total_bytes": total_bytes,
    }


def _load_pair(
    data_root: Path, groups: Sequence[str]
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for group in groups:
        relative = trajectory_relative_path("train", group)
        trajectory = load_ignithit_trajectory(
            data_root.joinpath(*PurePosixPath(relative).parts)
        )
        inputs.append(trajectory[TRAIN_INPUT_FRAME])
        targets.append(trajectory[TRAIN_TARGET_FRAME])
    return (
        torch.from_numpy(np.stack(inputs, axis=0)),
        torch.from_numpy(np.stack(targets, axis=0)),
    )


def _all_finite_gradients(model: torch.nn.Module) -> bool:
    return all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def _frozen_config(model_config: RealmFFNOConfig) -> dict[str, Any]:
    return {
        "schema": "d088_ignithit_p1c_personal_gpu_config_v1",
        "run_id": RUN_ID,
        "resource_class": "owner_selected_personal_gpu_workstation",
        "device": DEVICE,
        "seed": SEED,
        "state_representation": "released_physical_then_primary_box_cox_zscore",
        "target_parameterization": "direct_next_normalized_state",
        "exposure": "one_call",
        "model": {
            "family": "independent_ffno_m_reconstruction",
            **model_config.__dict__,
            "reported_parameter_target": 8_936_500,
            "parameter_relative_tolerance": 0.005,
        },
        "coordinates": {
            "static_channels": 2,
            "per_channel_min_subtraction": True,
            "common_denominator": "released_coordinate_channel_zero_range",
        },
        "precision": {
            "model_input_output": "float32",
            "autocast": False,
            "tf32": False,
            "deterministic_algorithms": True,
        },
        "train_smoke": {
            "ordered_groups": list(TRAIN_GROUPS),
            "microbatch_size": MICROBATCH_SIZE,
            "accumulation_steps": ACCUMULATION_STEPS,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "input_frame": TRAIN_INPUT_FRAME,
            "target_frame": TRAIN_TARGET_FRAME,
            "optimizer": "Adam",
            "learning_rate": OPTIMIZER_LR,
            "weight_decay": OPTIMIZER_WEIGHT_DECAY,
            "betas": list(OPTIMIZER_BETAS),
            "epsilon": OPTIMIZER_EPS,
            "amsgrad": False,
            "foreach": False,
            "fused": False,
            "optimizer_steps": 1,
            "scheduler": None,
            "gradient_clipping": None,
        },
        "validation_smoke": {
            "group": VALIDATION_GROUP,
            "start_frame": 0,
            "horizon_calls": VALIDATION_HORIZON,
            "selection_or_tuning_use": False,
        },
        "max_end_to_end_seconds": MAX_WALL_SECONDS,
        "checkpoint_written": False,
        "sealed_test_available": False,
    }


def run_smoke(
    manifest_path: Path,
    data_root: Path,
    normalizer_arrays_path: Path,
    report_dir: Path,
) -> dict[str, Any]:
    started_at = time.monotonic()
    payload = _load_json(manifest_path)
    repository, revision, entries = parse_manifest_payload(payload)
    manifest_summary = validate_ignithit_open_manifest(repository, revision, entries)
    inventory = validate_local_open_tree(data_root, entries)
    metadata = load_ignithit_metadata(data_root / "data" / "data.npz")
    normalizer, train_max_abs, normalizer_summary = _normalizer_from_arrays(
        normalizer_arrays_path
    )
    if (
        tuple(metadata.train_groups) != TRAIN_GROUPS
        or tuple(metadata.val_groups) != VAL_GROUPS
    ):
        raise ValueError("metadata split order differs from the frozen smoke contract")
    if VALIDATION_GROUP != VAL_GROUPS[0]:
        raise RuntimeError(
            "validation smoke group must remain the first registered val key"
        )
    if len(TRAIN_GROUPS) != EFFECTIVE_BATCH_SIZE or (
        MICROBATCH_SIZE * ACCUMULATION_STEPS != EFFECTIVE_BATCH_SIZE
    ):
        raise RuntimeError("effective-batch contract is internally inconsistent")

    _configure_determinism()
    device = torch.device(DEVICE)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("P1c requires exactly one visible CUDA device")
    torch.cuda.set_device(device)

    coordinate_tensor = torch.from_numpy(metadata.coords).unsqueeze(0).to(dtype=DTYPE)
    coordinates = normalize_realm_coordinates(coordinate_tensor).to(device)
    model_config = RealmFFNOConfig()
    config = _frozen_config(model_config)
    config["canonical_payload_sha256"] = canonical_json_sha256(config)
    model = RealmFFNO2d(model_config).to(device=device, dtype=DTYPE)
    parameter_count = trainable_parameter_count(model)
    parameter_gate = parameter_count_within_reported_tolerance(parameter_count)
    if not parameter_gate:
        raise RuntimeError("FFNO-M parameter count is outside the frozen tolerance")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    model.zero_grad(set_to_none=True)
    synthetic_state = torch.randn(
        MICROBATCH_SIZE, 12, 128, 128, device=device, dtype=DTYPE
    )
    synthetic_truth = torch.randn_like(synthetic_state)
    torch.cuda.synchronize(device)
    synthetic_started = time.monotonic()
    synthetic_prediction = model(synthetic_state, coordinates)
    synthetic_loss, _ = grouped_next_state_mse(synthetic_prediction, synthetic_truth)
    synthetic_loss.backward()
    torch.cuda.synchronize(device)
    synthetic_seconds = time.monotonic() - synthetic_started
    synthetic_finite = bool(torch.isfinite(synthetic_prediction).all()) and bool(
        torch.isfinite(synthetic_loss)
    )
    synthetic_gradients_finite = _all_finite_gradients(model)
    synthetic_memory = _phase_memory(device)
    synthetic_loss_value = float(synthetic_loss.detach().item())
    del synthetic_state, synthetic_truth, synthetic_prediction, synthetic_loss
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    if not synthetic_finite or not synthetic_gradients_finite:
        raise RuntimeError("synthetic forward/backward returned a nonfinite value")
    _require_budget(started_at)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=OPTIMIZER_LR,
        betas=OPTIMIZER_BETAS,
        eps=OPTIMIZER_EPS,
        weight_decay=OPTIMIZER_WEIGHT_DECAY,
        amsgrad=False,
        foreach=False,
        fused=False,
    )
    torch.cuda.reset_peak_memory_stats(device)
    group_loss_sums: Counter[str] = Counter()
    train_loss_sum = 0.0
    train_predictions_finite = True
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    train_started = time.monotonic()
    for group in TRAIN_GROUPS:
        _require_budget(started_at)
        input_native, target_native = _load_pair(data_root, (group,))
        input_normalized = normalizer.encode(
            input_native.to(device=device, dtype=DTYPE)
        )
        target_normalized = normalizer.encode(
            target_native.to(device=device, dtype=DTYPE)
        )
        prediction = model(input_normalized, coordinates)
        loss, by_group = grouped_next_state_mse(prediction, target_normalized)
        current_finite = bool(torch.isfinite(prediction).all()) and bool(
            torch.isfinite(loss)
        )
        train_predictions_finite = train_predictions_finite and current_finite
        if not current_finite:
            raise RuntimeError("real train smoke returned a nonfinite value")
        train_loss_sum += float(loss.detach().item())
        for name, value in by_group.items():
            group_loss_sums[name] += float(value.detach().item())
        (loss / EFFECTIVE_BATCH_SIZE).backward()
        del (
            input_native,
            target_native,
            input_normalized,
            target_normalized,
            prediction,
            loss,
        )
    train_gradients_finite = _all_finite_gradients(model)
    if not train_gradients_finite:
        raise RuntimeError("real train smoke returned a nonfinite gradient")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    train_seconds = time.monotonic() - train_started
    train_memory = _phase_memory(device)
    parameters_finite_after_step = all(
        bool(torch.isfinite(parameter).all()) for parameter in model.parameters()
    )
    if not parameters_finite_after_step:
        raise RuntimeError(
            "the registered optimizer step produced nonfinite parameters"
        )
    torch.cuda.empty_cache()
    _require_budget(started_at)

    validation_relative = trajectory_relative_path("val", VALIDATION_GROUP)
    validation_native = load_ignithit_trajectory(
        data_root.joinpath(*PurePosixPath(validation_relative).parts)
    )
    validation_tensor = torch.from_numpy(validation_native).to(
        device=device, dtype=DTYPE
    )
    validation_truth_normalized = normalizer.encode(validation_tensor)
    current = validation_truth_normalized[:1]
    envelope = MagnitudeEnvelope(
        max_abs=train_max_abs.to(device),
        quantile=BOUNDEDNESS_QUANTILE,
        channel_axis=1,
    )
    predictions: list[torch.Tensor] = []
    decoded_predictions: list[torch.Tensor] = []
    call_rows: list[dict[str, Any]] = []
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    validation_started = time.monotonic()
    with torch.no_grad():
        for call in range(1, VALIDATION_HORIZON + 1):
            _require_budget(started_at)
            prediction = model(current, coordinates)
            normalized_finite = bool(torch.isfinite(prediction).all())
            decoded = normalizer.decode(prediction, inverse_domain_policy="nan")
            decoded_finite = bool(torch.isfinite(decoded).all())
            admissibility = decoded_admissibility(decoded, channel_axis=1)
            boundedness = decoded_boundedness(
                decoded,
                envelope,
                expansion_factor=BOUNDEDNESS_EXPANSION_FACTOR,
                channel_axis=1,
            )
            call_rows.append(
                {
                    "call": call,
                    "normalized_finite": normalized_finite,
                    "decoded_finite": decoded_finite,
                    "admissible_released_state": admissibility.admissible,
                    "admissibility_violations": list(admissibility.violations),
                    "bounded_10x_train_max": boundedness.bounded,
                    "max_boundedness_ratio_by_channel": [
                        _finite_or_none(float(value))
                        for value in boundedness.max_ratio_by_channel
                    ],
                }
            )
            predictions.append(prediction.detach())
            decoded_predictions.append(decoded.detach())
            if not normalized_finite or not decoded_finite:
                break
            current = prediction
    torch.cuda.synchronize(device)
    validation_seconds = time.monotonic() - validation_started
    validation_memory = _phase_memory(device)

    prediction_stack = torch.stack(predictions, dim=1)
    decoded_stack = torch.stack(decoded_predictions, dim=1)
    observed_calls = prediction_stack.shape[1]
    truth_normalized = validation_truth_normalized[1 : observed_calls + 1].unsqueeze(0)
    truth_decoded = validation_tensor[1 : observed_calls + 1].unsqueeze(0)
    normalized_metrics = grouped_normalized_prediction_error(
        prediction_stack,
        truth_normalized,
    )
    correlation = decoded_spatial_pearson(decoded_stack, truth_decoded)
    correlation_statuses: Counter[str] = Counter()
    for case in correlation.statuses:
        for call in case:
            correlation_statuses.update(call)

    elapsed_seconds = time.monotonic() - started_at
    gates = {
        "exact_open_manifest": manifest_summary["manifest_sha256"]
        == IGNITHIT_OPEN_MANIFEST_SHA256,
        "exact_file_inventory": len(inventory) == 34,
        "sealed_test_objects_absent": all(
            "/test/" not in f"/{row['path']}/" for row in inventory
        ),
        "parameter_count": parameter_gate,
        "synthetic_forward_backward_finite": synthetic_finite
        and synthetic_gradients_finite,
        "effective_batch_26": len(TRAIN_GROUPS) == EFFECTIVE_BATCH_SIZE
        and ACCUMULATION_STEPS == 26,
        "one_optimizer_step_finite": train_predictions_finite
        and train_gradients_finite
        and parameters_finite_after_step,
        "validation_h29_normalized_finite": observed_calls == VALIDATION_HORIZON
        and all(row["normalized_finite"] for row in call_rows),
        "validation_h29_decoded_finite": observed_calls == VALIDATION_HORIZON
        and all(row["decoded_finite"] for row in call_rows),
        "wall_time_cap": elapsed_seconds <= MAX_WALL_SECONDS,
    }
    metrics = {
        "schema": "d088_ignithit_p1c_smoke_metrics_v1",
        "parameter_count": parameter_count,
        "synthetic": {
            "loss": _finite_or_none(synthetic_loss_value),
            "seconds": synthetic_seconds,
            "memory": synthetic_memory,
        },
        "train_one_step": {
            "optimizer_steps": 1,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "microbatch_size": MICROBATCH_SIZE,
            "accumulation_steps": ACCUMULATION_STEPS,
            "mean_grouped_loss_before_step": train_loss_sum / EFFECTIVE_BATCH_SIZE,
            "mean_loss_by_group_before_step": {
                name: value / EFFECTIVE_BATCH_SIZE
                for name, value in sorted(group_loss_sums.items())
            },
            "seconds": train_seconds,
            "memory": train_memory,
        },
        "validation_rollout": {
            "group": VALIDATION_GROUP,
            "requested_calls": VALIDATION_HORIZON,
            "observed_calls": observed_calls,
            "realm_npe_mean_observed_prefix": _finite_or_none(
                normalized_metrics.realm_npe_mean
            ),
            "realm_npe_sum_source_observed_prefix": _finite_or_none(
                normalized_metrics.realm_npe_sum_source
            ),
            "decoded_correlation_case_first": _finite_or_none(
                correlation.population_case_first_mean
            ),
            "correlation_status_counts": dict(sorted(correlation_statuses.items())),
            "admissible_call_count": sum(
                row["admissible_released_state"] for row in call_rows
            ),
            "bounded_call_count": sum(
                row["bounded_10x_train_max"] for row in call_rows
            ),
            "seconds": validation_seconds,
            "memory": validation_memory,
            "calls": call_rows,
        },
        "elapsed_seconds": elapsed_seconds,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "anti_claims": [
            "one optimizer step is not model training or a baseline result",
            "the validation smoke is not a selection or tuning result",
            "no checkpoint was written",
            "no test trajectory was present or opened",
            "released-state admissibility is not complete composition conservation",
        ],
    }
    source_manifest = _source_manifest()
    runtime_manifest = _runtime_manifest(device)
    input_manifest = {
        "schema": "d088_ignithit_p1c_inputs_v1",
        "open_manifest_sha256": IGNITHIT_OPEN_MANIFEST_SHA256,
        "open_manifest_payload_sha256": canonical_json_sha256(payload),
        "open_entry_count": len(entries),
        "open_total_bytes": sum(entry.size for entry in entries),
        "normalizer": normalizer_summary,
        "test_object_opened": False,
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, Mapping[str, Any]] = {
        "config.json": config,
        "input_manifest.json": input_manifest,
        "source_manifest.json": source_manifest,
        "runtime_manifest.json": runtime_manifest,
        "smoke_metrics.json": metrics,
    }
    for name, content in files.items():
        _write_json(report_dir / name, content)
    hashes = {name: sha256_file(report_dir / name) for name in files}
    final_manifest = {
        "schema": "d088_ignithit_p1c_final_hash_manifest_v1",
        "files": hashes,
        "self_hash_excluded": True,
    }
    _write_json(report_dir / "final_hash_manifest.json", final_manifest)
    summary = {
        "schema": "d088_ignithit_p1c_summary_v1",
        "run_id": RUN_ID,
        "all_gates_pass": metrics["all_gates_pass"],
        "final_hash_manifest_sha256": sha256_file(
            report_dir / "final_hash_manifest.json"
        ),
        "model_instantiated": True,
        "checkpoint_loaded": False,
        "checkpoint_written": False,
        "test_object_opened": False,
    }
    _write_json(report_dir / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        for source in (args.manifest, args.normalizer_arrays):
            if args.report_dir.resolve(strict=False) == source.resolve(strict=False):
                raise ValueError("report directory must not overwrite an input")
        if _is_within(args.report_dir, args.data_root):
            raise ValueError("report directory must be outside the exact data root")
        if args.report_dir.exists() and any(args.report_dir.iterdir()):
            raise ValueError("report directory must be absent or empty")
        summary = run_smoke(
            args.manifest,
            args.data_root,
            args.normalizer_arrays,
            args.report_dir,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        build_parser().error(str(exc))
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0 if summary["all_gates_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
