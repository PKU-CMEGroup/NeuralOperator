"""Shared, policy-neutral runtime primitives for schema-4 Euler2D PCNOs."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from utility.time_dependent_no.pcno_euler2d import (
    NODE_TYPE_FEATURE_ONE_HOT,
    Euler2DNormalization,
    PCNOEuler2DResidual,
)

CHECKPOINT_SCHEMA_VERSION = 4
MODEL_NODE_TYPE_INPUTS = ("physical", "all_normal")
AMP_MODES = ("none", "bf16", "fp16")


def select_device(name: str) -> torch.device:
    """Resolve one of the legal runtime devices without silently accepting typos."""

    if name not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"unsupported device selection: {name}")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def synchronize(device: torch.device) -> None:
    """Synchronize CUDA work while remaining a no-op on CPU."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def autocast_context(device: torch.device, amp: str):
    """Return the explicit evaluation/training precision context."""

    if amp not in AMP_MODES:
        raise ValueError(f"unsupported AMP mode: {amp}")
    if amp == "none":
        return nullcontext()
    if device.type != "cuda":
        raise ValueError("mixed-precision PCNO execution requires CUDA")
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def load_checkpoint_payload(path: Path) -> dict[str, Any]:
    """Deserialize a checkpoint mapping without applying schema or policy checks."""

    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError("PCNO checkpoint payload must be a mapping")
    return dict(payload)


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Deserialize a schema-4 checkpoint without applying evaluator policy."""

    payload = load_checkpoint_payload(path)
    if int(payload.get("checkpoint_schema_version", -1)) != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"unsupported checkpoint schema in {path}")
    return payload


def checkpoint_model_node_type_input(checkpoint: Mapping[str, Any]) -> str:
    """Resolve the model-facing type contract, defaulting legacy runs to physical."""

    training_args = checkpoint.get("training_args", {})
    from_args = (
        training_args.get("model_node_type_input")
        if isinstance(training_args, Mapping)
        else None
    )
    declared = checkpoint.get("model_node_type_input")
    if declared is not None and from_args is not None and declared != from_args:
        raise ValueError("checkpoint node-type input declarations disagree")
    mode = str(declared if declared is not None else from_args or "physical")
    if mode not in MODEL_NODE_TYPE_INPUTS:
        raise ValueError(f"unsupported checkpoint node-type input: {mode}")
    return mode


def build_checkpoint_model(
    checkpoint: Mapping[str, Any],
    device: torch.device,
    *,
    model_node_type_input: str,
) -> tuple[PCNOEuler2DResidual, Euler2DNormalization]:
    """Construct a strict model under the caller-selected node-type contract."""

    if model_node_type_input not in MODEL_NODE_TYPE_INPUTS:
        raise ValueError(f"unsupported model node-type input: {model_node_type_input}")
    normalization = Euler2DNormalization.from_mapping(checkpoint["normalization"])
    config = checkpoint["model_config"]
    model = PCNOEuler2DResidual(
        normalization=normalization,
        k_max=int(config["k_max"]),
        domain_lengths=tuple(config["domain_lengths"]),
        layers=tuple(config["layers"]),
        fc_dim=int(config["fc_dim"]),
        nmeasures=int(config["nmeasures"]),
        zero_initialize=False,
        node_type_feature_mode=str(
            config.get("node_type_feature_mode", NODE_TYPE_FEATURE_ONE_HOT)
        ),
        boundary_field_mode=str(config.get("boundary_field_mode", "none")),
        boundary_field_names=tuple(config.get("boundary_field_names", ())),
        boundary_residual_mode=str(config.get("boundary_residual_mode", "none")),
        boundary_residual_names=tuple(config.get("boundary_residual_names", ())),
        boundary_residual_width=int(config.get("boundary_residual_width", 64)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.model_node_type_input = model_node_type_input
    model.eval()
    return model, normalization


def forward_sample(
    model: PCNOEuler2DResidual,
    sample: Mapping[str, torch.Tensor],
    current: torch.Tensor,
) -> torch.Tensor:
    """Apply a PCNO to one collated sample under its declared node-type mode."""

    node_type = sample["node_type"]
    if getattr(model, "model_node_type_input", "physical") == "all_normal":
        node_type = torch.zeros_like(node_type)
    return model(
        current,
        node_mask=sample["node_mask"],
        nodes=sample["nodes"],
        node_weights=sample["node_weights"],
        node_rhos=sample["node_rhos"],
        directed_edges=sample["directed_edges"],
        edge_gradient_weights=sample["edge_gradient_weights"],
        node_type=node_type,
        mach=sample["mach"],
        boundary_features=sample.get("boundary_features"),
    )


def expand_homogeneous_sample(
    sample: Mapping[str, torch.Tensor], batch_size: int
) -> dict[str, torch.Tensor]:
    """Expand one immutable geometry sample across a homogeneous state batch."""

    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer")
    expanded: dict[str, torch.Tensor] = {}
    for name, value in sample.items():
        if not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[0] != 1:
            raise ValueError(f"sample tensor {name!r} must have leading dimension one")
        expanded[name] = value.expand(batch_size, *value.shape[1:])
    return expanded


@torch.inference_mode()
def timed_model_call(
    model: PCNOEuler2DResidual,
    sample: Mapping[str, torch.Tensor],
    current: torch.Tensor,
    *,
    device: torch.device,
    amp: str,
    repeats: int,
    return_batch: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Time calls and return either the first item or the full batch as float64."""

    if repeats < 1:
        raise ValueError("repeat-forward must be positive")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    predictions: list[np.ndarray] = []
    seconds: list[float] = []
    for _ in range(repeats):
        synchronize(device)
        started = perf_counter()
        with autocast_context(device, amp):
            prediction = forward_sample(model, sample, current)
        synchronize(device)
        seconds.append(perf_counter() - started)
        selected = prediction if return_batch else prediction[:1]
        predictions.append(selected.detach().float().cpu().numpy().astype(np.float64))
    reference = predictions[0]
    repeat_max_abs = max(
        float(np.max(np.abs(value - reference))) for value in predictions
    )
    output = reference if return_batch else reference[0]
    return output, {
        "forward_seconds": seconds,
        "mean_forward_seconds": float(np.mean(seconds)),
        "minimum_forward_seconds": float(np.min(seconds)),
        "repeat_max_abs": repeat_max_abs,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
    }
