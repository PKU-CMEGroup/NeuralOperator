"""State-dict-stable controls for the PCNO differential branch."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from types import MethodType
from typing import Any

import torch

DIFFERENTIAL_BRANCH_MODES = ("full", "no_gradient")
DIFFERENTIAL_BRANCH_CONTRACT_SCHEMA = "w26_l2_bump_gradient_ablation_v1"


def _backbone(model: torch.nn.Module) -> torch.nn.Module:
    backbone = getattr(model, "backbone", model)
    gws = getattr(backbone, "gws", None)
    if not isinstance(gws, torch.nn.ModuleList) or not gws:
        raise TypeError("model does not expose a nonempty PCNO differential ModuleList")
    return backbone


def _is_differential_state_name(name: str) -> bool:
    return name.startswith("gws.") or ".gws." in name


def model_state_sha256(
    model: torch.nn.Module,
    *,
    include: Callable[[str], bool] | None = None,
) -> str:
    """Hash tensor names, dtypes, shapes, and exact bytes deterministically."""

    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if include is not None and not include(name):
            continue
        tensor = value.detach().cpu().contiguous()
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(raw)
    return digest.hexdigest()


def _functional_zero_gradient_forward(
    module: torch.nn.Module,
    x: torch.Tensor,
    *_: Any,
    **__: Any,
) -> torch.Tensor:
    gw2 = getattr(module, "gw2", None)
    if gw2 is None or not hasattr(gw2, "out_channels"):
        raise TypeError("differential module does not expose gw2.out_channels")
    return x.new_zeros((x.shape[0], int(gw2.out_channels), x.shape[-1]))


def apply_differential_branch_mode(
    model: torch.nn.Module,
    mode: str,
    *,
    record_initialization: bool = True,
) -> dict[str, Any]:
    """Apply a state-dict-stable full or functional-no-gradient PCNO mode."""

    if mode not in DIFFERENTIAL_BRANCH_MODES:
        raise ValueError(f"mode must lie in {DIFFERENTIAL_BRANCH_MODES}")
    backbone = _backbone(model)
    prior = getattr(backbone, "_w26_l2_differential_branch_mode", None)
    if prior is not None and prior != mode:
        raise ValueError(f"model already configured as {prior!r}, not {mode!r}")

    state_keys_before = tuple(model.state_dict())
    initial_full_hash = model_state_sha256(model) if record_initialization else None
    initial_shared_hash = (
        model_state_sha256(
            model, include=lambda name: not _is_differential_state_name(name)
        )
        if record_initialization
        else None
    )
    stored_differential = 0
    trainable_differential = 0
    layer_counts: list[int] = []
    for module in backbone.gws:
        count = sum(parameter.numel() for parameter in module.parameters())
        layer_counts.append(count)
        stored_differential += count
        if mode == "no_gradient":
            for parameter in module.parameters():
                parameter.requires_grad_(False)
            # Preserve the historical evaluator and decomposer contract: a zero
            # gw2 makes the native GradientLayer forward identically zero.
            with torch.no_grad():
                module.gw2.weight.zero_()
            module.forward = MethodType(_functional_zero_gradient_forward, module)
        trainable_differential += sum(
            parameter.numel()
            for parameter in module.parameters()
            if parameter.requires_grad
        )

    if tuple(model.state_dict()) != state_keys_before:
        raise RuntimeError("functional ablation changed checkpoint state-dict keys")
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if mode == "no_gradient" and trainable_differential != 0:
        raise RuntimeError(
            "functional no-gradient mode retains trainable branch parameters"
        )
    if mode == "full" and trainable_differential != stored_differential:
        raise RuntimeError("full mode unexpectedly freezes differential parameters")

    contract = {
        "schema": DIFFERENTIAL_BRANCH_CONTRACT_SCHEMA,
        "mode": mode,
        "implementation": (
            "native_pcno_differential_branch"
            if mode == "full"
            else "state_dict_stable_exact_zero_forward_frozen_parameters"
        ),
        "differential_layer_count": len(backbone.gws),
        "differential_parameters_by_layer": layer_counts,
        "stored_differential_parameters": stored_differential,
        "trainable_differential_parameters": trainable_differential,
        "stored_total_parameters": total_parameters,
        "trainable_total_parameters": trainable_parameters,
        "state_dict_keys_unchanged": True,
        "exact_zero_forward": mode == "no_gradient",
        "ordinary_evaluator_exact_zero_via_zero_gw2": mode == "no_gradient",
        "initial_full_state_sha256": initial_full_hash,
        "initial_nondifferential_state_sha256": initial_shared_hash,
    }
    backbone._w26_l2_differential_branch_mode = mode
    model.differential_branch_contract = contract
    initialization_control = dict(getattr(model, "initialization_control", {}))
    initialization_control["differential_branch"] = contract
    model.initialization_control = initialization_control
    return contract
