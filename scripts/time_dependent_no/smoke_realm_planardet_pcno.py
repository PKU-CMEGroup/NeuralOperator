"""Profile one full-grid two-call PlanarDet residual-PCNO optimizer step."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utility.time_dependent_no.realm_benchmark import (
    canonical_json_sha256,
    predict_two_call_final,
)
from utility.time_dependent_no.realm_pcno import (
    RealmPCNOConfig,
    RealmRegularGridPCNO,
    build_realm_regular_grid_geometry,
    estimate_realm_pcno_static_bytes,
)
from utility.time_dependent_no.realm_planardet import (
    CANONICAL_SPATIAL_SHAPE_YX,
    DOMAIN_LENGTHS_XY,
    sha256_file,
)
from utility.time_dependent_no.realm_planardet_artifacts import (
    EXECUTABLE_ENTRYPOINTS,
    build_source_manifest,
)
from utility.time_dependent_no.realm_planardet_runtime import grouped_planardet_mse

SCHEMA = "w26_l4_planardet_pd0_a2_full_grid_smoke_v1"
ENTRYPOINT = "scripts/time_dependent_no/smoke_realm_planardet_pcno.py"
SEED = 20_260_815
MODE_COUNTS_XY = (8, 8)
FC_DIM = 128
CHANNELS = 13
SAFETY_HEADROOM_FRACTION = 0.20
EXPECTED_PARAMETERS = {128: 19_157_393, 96: 10_780_401}
MAX_WALL_SECONDS = 1_800.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--width", type=int, choices=tuple(EXPECTED_PARAMETERS), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--prior-width128-result",
        type=Path,
        help="required for width 96; must prove width 128 OOM or insufficient headroom",
    )
    return parser


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError("smoke output path must be absent")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def validate_width_ladder(
    width: int,
    prior_path: Path | None,
    *,
    source_manifest: Mapping[str, Any],
) -> str | None:
    if width == 128:
        if prior_path is not None:
            raise ValueError("width 128 must be the first registered smoke envelope")
        return None
    if width != 96 or prior_path is None:
        raise ValueError("width 96 requires the prior width-128 smoke result")
    if prior_path.is_symlink() or not prior_path.is_file():
        raise ValueError("prior width-128 result must be a regular file")
    try:
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read prior width-128 result: {exc}") from exc
    if not isinstance(prior, Mapping):
        raise TypeError("prior width-128 result root must be an object")
    unsigned = {
        key: value for key, value in prior.items() if key != "canonical_payload_sha256"
    }
    configuration = prior.get("configuration")
    prior_status = prior.get("status")
    prior_width = (
        configuration.get("layers", [None])[0]
        if isinstance(configuration, Mapping)
        else prior.get("width")
    )
    if (
        prior.get("canonical_payload_sha256") != canonical_json_sha256(unsigned)
        or prior.get("schema") != SCHEMA
        or prior_status not in {"oom", "insufficient_headroom"}
        or prior_width != 128
        or prior.get("passes_memory_gate") is not False
        or not (
            (prior_status == "oom" and prior.get("passes_full_grid_step") is False)
            or (
                prior_status == "insufficient_headroom"
                and prior.get("passes_full_grid_step") is True
            )
        )
        or prior.get("test_object_opened") is not False
        or prior.get("dataset_array_opened") is not False
        or prior.get("source_manifest") != source_manifest
    ):
        raise ValueError("prior result does not authorize the width-96 fallback")
    return sha256_file(prior_path)


def _configure_determinism() -> None:
    workspace_config = os.environ.setdefault(
        "CUBLAS_WORKSPACE_CONFIG",
        ":4096:8",
    )
    if workspace_config != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must equal :4096:8")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _synthetic_coordinates() -> np.ndarray:
    height, width = CANONICAL_SPATIAL_SHAPE_YX
    x = np.linspace(0.1298875, 0.1091125, width, dtype=np.float64)
    y = np.linspace(0.0097875, 0.0002125, height, dtype=np.float64)
    x_grid, y_grid = np.meshgrid(x, y, indexing="xy")
    return np.stack((y_grid, x_grid), axis=0).astype(np.float32)


def _phase_memory(device: torch.device) -> dict[str, int]:
    free, total = torch.cuda.mem_get_info(device)
    return {
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "max_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "max_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "device_free_bytes": free,
        "device_total_bytes": total,
    }


def _runtime_manifest(device: torch.device) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    payload: dict[str, Any] = {
        "schema": "w26_l4_planardet_pd0_a2_cuda_runtime_v2",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count(),
        "device_name": properties.name,
        "device_total_memory": properties.total_memory,
        "bf16_supported": torch.cuda.is_bf16_supported(),
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


def _check_budget(started_at: float) -> None:
    if time.monotonic() - started_at > MAX_WALL_SECONDS:
        raise TimeoutError("full-grid smoke exceeded its frozen wall-time budget")


def run_smoke(width: int, device_name: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("smoke requires exactly one visible CUDA device")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the intended bfloat16 autocast path is unsupported")
    _configure_determinism()
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started_at = time.monotonic()
    free_before, total_memory = torch.cuda.mem_get_info(device)
    external_used_before = total_memory - free_before

    coordinates_numpy = _synthetic_coordinates()
    geometry_started = time.monotonic()
    geometry = build_realm_regular_grid_geometry(
        coordinates_numpy,
        domain_lengths_xy=DOMAIN_LENGTHS_XY,
        released_coordinate_order=("y", "x"),
    )
    geometry_seconds = time.monotonic() - geometry_started
    _check_budget(started_at)

    config = RealmPCNOConfig(
        channels=CHANNELS,
        mode_counts_xy=MODE_COUNTS_XY,
        layers=(width,) * 5,
        fc_dim=FC_DIM,
        zero_initialize_head=True,
    )
    model_started = time.monotonic()
    model = RealmRegularGridPCNO(config=config, geometry=geometry).to(
        device=device, dtype=torch.float32
    )
    model_seconds = time.monotonic() - model_started
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if parameter_count != EXPECTED_PARAMETERS[width]:
        raise RuntimeError("PCNO parameter count differs from the frozen envelope")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1.0e-3,
        weight_decay=0.0,
        betas=(0.9, 0.999),
        eps=1.0e-8,
    )
    memory_after_model = _phase_memory(device)
    _check_budget(started_at)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED)
    height, grid_width = CANONICAL_SPATIAL_SHAPE_YX
    input_prepare_started = time.monotonic()
    state_cpu = torch.randn(
        (1, CHANNELS, height, grid_width),
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    )
    target_cpu = state_cpu + 0.01 * torch.randn(
        state_cpu.shape,
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    )
    input_prepare_seconds = time.monotonic() - input_prepare_started
    transfer_started = time.monotonic()
    state = state_cpu.to(device)
    target = target_cpu.to(device)
    coordinates = torch.from_numpy(coordinates_numpy).unsqueeze(0).to(device)
    torch.cuda.synchronize(device)
    input_h2d_seconds = time.monotonic() - transfer_started
    memory_after_inputs = _phase_memory(device)

    optimizer.zero_grad(set_to_none=True)
    step_started = time.monotonic()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        prediction = predict_two_call_final(
            model,
            state,
            coordinates,
            parameterization="residual",
        )
        loss, by_group = grouped_planardet_mse(prediction, target)
    if not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(loss)):
        raise RuntimeError("full-grid smoke produced a nonfinite prediction or loss")
    loss.backward()
    if not all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
        if parameter.requires_grad
    ):
        raise RuntimeError("full-grid smoke produced a missing or nonfinite gradient")
    optimizer.step()
    torch.cuda.synchronize(device)
    step_seconds = time.monotonic() - step_started
    memory_after_step = _phase_memory(device)
    _check_budget(started_at)

    peak_reserved = torch.cuda.max_memory_reserved(device)
    effective_peak = external_used_before + peak_reserved
    headroom_bytes = total_memory - effective_peak
    headroom_fraction = headroom_bytes / total_memory
    passes_memory_gate = headroom_fraction >= SAFETY_HEADROOM_FRACTION
    return {
        "schema": SCHEMA,
        "status": "pass" if passes_memory_gate else "insufficient_headroom",
        "passes_full_grid_step": True,
        "passes_memory_gate": passes_memory_gate,
        "seed": SEED,
        "device": device_name,
        "precision": {
            "parameters_and_geometry": "float32",
            "autocast": "bfloat16",
            "optimizer_state": "float32",
        },
        "execution": "detached_first_call_then_second_call_loss_backward_adamw_step",
        "parameterization": "residual_in_transformed_normalized_state",
        "configuration": {
            "channels": CHANNELS,
            "shape_yx": list(CANONICAL_SPATIAL_SHAPE_YX),
            "mode_counts_xy": list(MODE_COUNTS_XY),
            "layers": [width] * 5,
            "fc_dim": FC_DIM,
            "zero_initialize_head": True,
            "microbatch_size": 1,
            "parameter_count": parameter_count,
        },
        "static_memory_estimate": estimate_realm_pcno_static_bytes(
            height, grid_width, MODE_COUNTS_XY
        ),
        "memory": {
            "external_used_before_bytes": external_used_before,
            "after_model": memory_after_model,
            "after_inputs": memory_after_inputs,
            "after_step": memory_after_step,
            "process_peak_reserved_bytes": peak_reserved,
            "effective_peak_bytes": effective_peak,
            "headroom_bytes": headroom_bytes,
            "headroom_fraction": headroom_fraction,
            "required_headroom_fraction": SAFETY_HEADROOM_FRACTION,
        },
        "timing_seconds": {
            "geometry_cpu": geometry_seconds,
            "model_and_fourier_cache": model_seconds,
            "synthetic_cpu_input_preparation": input_prepare_seconds,
            "state_target_coordinates_h2d": input_h2d_seconds,
            "two_call_backward_optimizer_step": step_seconds,
            "profiled_microbatch_end_to_end": input_h2d_seconds + step_seconds,
            "total": time.monotonic() - started_at,
        },
        "loss": float(loss.detach().item()),
        "loss_by_group": {
            name: float(value.detach().item()) for name, value in by_group.items()
        },
        "prediction_finite": True,
        "gradients_finite": True,
        "test_object_opened": False,
        "dataset_array_opened": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_manifest = build_source_manifest(
        REPO_ROOT,
        entrypoints=EXECUTABLE_ENTRYPOINTS,
    )
    try:
        prior_width128_sha256 = validate_width_ladder(
            args.width,
            args.prior_width128_result,
            source_manifest=source_manifest,
        )
    except (OSError, TypeError, ValueError) as exc:
        build_parser().error(str(exc))
    try:
        result = run_smoke(args.width, args.device)
    except (RuntimeError, TimeoutError, torch.OutOfMemoryError) as exc:
        result = {
            "schema": SCHEMA,
            "status": "oom" if isinstance(exc, torch.OutOfMemoryError) else "failed",
            "passes_full_grid_step": False,
            "passes_memory_gate": False,
            "width": args.width,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "test_object_opened": False,
            "dataset_array_opened": False,
        }
    result["runtime_manifest"] = (
        _runtime_manifest(torch.device(args.device))
        if torch.cuda.is_available()
        else None
    )
    result["source_manifest"] = source_manifest
    result["source_manifest_sha256"] = source_manifest["canonical_payload_sha256"]
    result["prior_width128_result_sha256"] = prior_width128_sha256
    result["canonical_payload_sha256"] = canonical_json_sha256(result)
    _write_json(args.output, result)
    result["output_sha256"] = sha256_file(args.output)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["passes_memory_gate"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
