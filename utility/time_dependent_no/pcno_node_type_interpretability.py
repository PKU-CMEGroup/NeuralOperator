"""Frozen node-type interventions and nonperturbing PCNO activation probes."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import torch
from torch.nn import functional as F

NORMAL_NODE_TYPE = 0
NUM_NODE_TYPES = 4
TYPE_INPUT_START = 7
TYPE_INPUT_STOP = 11

FAMILY_NODE_TYPE_NAMES: Mapping[str, Mapping[int, str]] = {
    "bump": {
        0: "normal",
        1: "wall",
        2: "outflow",
        3: "inflow",
    },
    "dynamic_fv": {
        0: "interior",
        1: "y_symmetry_contact",
        2: "x_extrapolation_contact",
        3: "both_contacts",
    },
}


def _validate_family(family: str) -> Mapping[int, str]:
    try:
        return FAMILY_NODE_TYPE_NAMES[str(family)]
    except KeyError as exc:
        supported = ", ".join(sorted(FAMILY_NODE_TYPE_NAMES))
        raise ValueError(
            f"unsupported node-type family {family!r}; expected one of {supported}"
        ) from exc


def _validate_node_type(node_type: torch.Tensor) -> None:
    if node_type.ndim not in {2, 3}:
        raise ValueError("node_type must have shape [B,N] or [B,N,1]")
    if node_type.ndim == 3 and node_type.shape[-1] != 1:
        raise ValueError("three-dimensional node_type must end in a singleton axis")
    if bool(((node_type < 0) | (node_type >= NUM_NODE_TYPES)).any()):
        raise ValueError("node_type contains an unsupported code")


def replace_node_types_with_normal(
    node_type: torch.Tensor,
    *,
    family: str,
    source_type: int | None = None,
) -> torch.Tensor:
    """Return an all-normal or single-family-type-to-normal intervention."""

    names = _validate_family(family)
    _validate_node_type(node_type)
    if source_type is not None:
        source_type = int(source_type)
        if source_type == NORMAL_NODE_TYPE or source_type not in names:
            raise ValueError("source_type must be one of the three nonnormal codes")

    intervened = node_type.clone()
    if source_type is None:
        intervened.fill_(NORMAL_NODE_TYPE)
    else:
        intervened[node_type == source_type] = NORMAL_NODE_TYPE
    return intervened


def node_type_intervention_name(
    *,
    family: str,
    source_type: int | None,
) -> str:
    """Return an unambiguous family-local intervention label."""

    names = _validate_family(family)
    if source_type is None:
        return "all_normal"
    source_type = int(source_type)
    if source_type == NORMAL_NODE_TYPE or source_type not in names:
        raise ValueError("source_type must be one of the three nonnormal codes")
    return f"{names[source_type]}_to_{names[NORMAL_NODE_TYPE]}"


def type_lift_delta(
    backbone: torch.nn.Module,
    correct_node_type: torch.Tensor,
    intervened_node_type: torch.Tensor,
) -> torch.Tensor:
    """Compute the exact analytical correct-minus-intervened lift signal.

    The maintained Euler wrapper concatenates coordinates, quadrature density,
    normalized state, four one-hot type channels, and Mach.  The type columns
    are therefore ``7:11``.  If a type-k node is replaced by type zero, this
    returns ``W_type (e_k - e_0)`` at that node.
    """

    _validate_node_type(correct_node_type)
    _validate_node_type(intervened_node_type)
    correct = (
        correct_node_type.squeeze(-1)
        if correct_node_type.ndim == 3
        else correct_node_type
    )
    intervened = (
        intervened_node_type.squeeze(-1)
        if intervened_node_type.ndim == 3
        else intervened_node_type
    )
    if correct.shape != intervened.shape:
        raise ValueError("correct and intervened node types must share shape")

    fc0 = getattr(backbone, "fc0", None)
    if not isinstance(fc0, torch.nn.Linear):
        raise TypeError("PCNO backbone must expose a linear fc0 lifting layer")
    if fc0.in_features < TYPE_INPUT_STOP:
        raise ValueError(
            "PCNO lift does not contain the frozen Euler type-input columns 7:11"
        )

    device = fc0.weight.device
    correct_one_hot = F.one_hot(
        correct.to(device=device, dtype=torch.int64),
        num_classes=NUM_NODE_TYPES,
    ).to(dtype=fc0.weight.dtype)
    intervened_one_hot = F.one_hot(
        intervened.to(device=device, dtype=torch.int64),
        num_classes=NUM_NODE_TYPES,
    ).to(dtype=fc0.weight.dtype)
    type_difference = correct_one_hot - intervened_one_hot
    type_weights = fc0.weight[:, TYPE_INPUT_START:TYPE_INPUT_STOP]
    return torch.matmul(type_difference, type_weights.transpose(0, 1))


def tensor_equivalence_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, Any]:
    """Return exact-shape/dtype and numerical equivalence diagnostics."""

    same_shape = reference.shape == candidate.shape
    same_dtype = reference.dtype == candidate.dtype
    if not same_shape:
        return {
            "same_shape": False,
            "same_dtype": same_dtype,
            "exact_equal": False,
            "max_abs": None,
            "relative_l2": None,
        }
    difference = (candidate - reference).detach().to(dtype=torch.float64)
    reference_norm = torch.linalg.vector_norm(
        reference.detach().to(dtype=torch.float64)
    )
    difference_norm = torch.linalg.vector_norm(difference)
    return {
        "same_shape": True,
        "same_dtype": same_dtype,
        "exact_equal": bool(torch.equal(reference, candidate)),
        "max_abs": float(difference.abs().max().cpu()),
        "relative_l2": float(
            (difference_norm / reference_norm.clamp_min(1.0e-30)).cpu()
        ),
    }


class PCNOActivationRecorder:
    """Record one PCNO forward pass using read-only module hooks.

    Hook callbacks retain detached tensor views.  They do not clone, move, or
    mutate tensors while the model is executing.  ``snapshot`` performs any
    requested cloning only after the hooked forward has completed.
    """

    def __init__(self, backbone: torch.nn.Module) -> None:
        self.backbone = backbone
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._records: defaultdict[str, list[torch.Tensor]] = defaultdict(list)

    @property
    def active(self) -> bool:
        return bool(self._handles)

    @staticmethod
    def _as_bnc(value: torch.Tensor, *, channel_first: bool) -> torch.Tensor:
        if not isinstance(value, torch.Tensor) or value.ndim != 3:
            raise TypeError("instrumented PCNO tensors must be rank-three tensors")
        return value.permute(0, 2, 1).detach() if channel_first else value.detach()

    def _record(self, name: str, value: torch.Tensor, *, channel_first: bool) -> None:
        self._records[name].append(self._as_bnc(value, channel_first=channel_first))

    def _forward_pre_hook(self, name: str, *, channel_first: bool):
        def hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
            if not inputs:
                raise RuntimeError(f"instrumented module {name} received no input")
            self._record(name, inputs[0], channel_first=channel_first)

        return hook

    def _forward_hook(self, name: str, *, channel_first: bool):
        def hook(
            _module: torch.nn.Module,
            _inputs: tuple[Any, ...],
            output: torch.Tensor,
        ) -> None:
            self._record(name, output, channel_first=channel_first)

        return hook

    def __enter__(self):
        if self.active:
            raise RuntimeError("activation recorder is already active")
        self._records.clear()
        backbone = self.backbone
        required = ("fc0", "sp_convs", "ws", "gws", "fc1", "fc2")
        missing = [name for name in required if not hasattr(backbone, name)]
        if missing:
            raise TypeError(f"PCNO backbone is missing modules: {', '.join(missing)}")
        block_count = len(backbone.ws)
        if not (len(backbone.sp_convs) == block_count == len(backbone.gws)):
            raise ValueError("PCNO branch module lists must have equal length")

        self._handles.append(
            backbone.register_forward_pre_hook(
                self._forward_pre_hook("model_input", channel_first=False)
            )
        )
        self._handles.append(
            backbone.fc0.register_forward_hook(
                self._forward_hook("post_lift", channel_first=False)
            )
        )
        for index in range(block_count):
            self._handles.append(
                backbone.ws[index].register_forward_pre_hook(
                    self._forward_pre_hook(f"block.{index}.input", channel_first=True)
                )
            )
            self._handles.append(
                backbone.sp_convs[index].register_forward_hook(
                    self._forward_hook(f"block.{index}.integral", channel_first=True)
                )
            )
            self._handles.append(
                backbone.ws[index].register_forward_hook(
                    self._forward_hook(f"block.{index}.pointwise", channel_first=True)
                )
            )
            self._handles.append(
                backbone.gws[index].register_forward_hook(
                    self._forward_hook(
                        f"block.{index}.differential", channel_first=True
                    )
                )
            )
        self._handles.append(
            backbone.fc1.register_forward_pre_hook(
                self._forward_pre_hook("decoder.input", channel_first=False)
            )
        )
        self._handles.append(
            backbone.fc1.register_forward_hook(
                self._forward_hook("decoder.linear", channel_first=False)
            )
        )
        self._handles.append(
            backbone.fc2.register_forward_pre_hook(
                self._forward_pre_hook("decoder.hidden", channel_first=False)
            )
        )
        self._handles.append(
            backbone.fc2.register_forward_hook(
                self._forward_hook("normalized_residual", channel_first=False)
            )
        )
        return self

    def __exit__(self, *_exc_info: object) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()

    def snapshot(
        self,
        *,
        device: str | torch.device | None = "cpu",
    ) -> dict[str, torch.Tensor]:
        """Clone the completed one-call trace and add block-output aliases."""

        if self.active:
            raise RuntimeError("exit the recorder context before taking a snapshot")
        block_count = len(self.backbone.ws)
        expected = {
            "model_input",
            "post_lift",
            "decoder.input",
            "decoder.linear",
            "decoder.hidden",
            "normalized_residual",
        }
        for index in range(block_count):
            expected.update(
                {
                    f"block.{index}.input",
                    f"block.{index}.integral",
                    f"block.{index}.pointwise",
                    f"block.{index}.differential",
                }
            )
        unexpected = set(self._records) - expected
        if unexpected:
            raise RuntimeError(
                f"unexpected activation records: {', '.join(sorted(unexpected))}"
            )
        missing = expected - set(self._records)
        if missing:
            raise RuntimeError(
                f"missing activation records: {', '.join(sorted(missing))}"
            )
        repeated = [name for name, values in self._records.items() if len(values) != 1]
        if repeated:
            raise RuntimeError(
                "activation recorder requires exactly one forward call; repeated: "
                + ", ".join(sorted(repeated))
            )

        target_device = None if device is None else torch.device(device)
        result = {}
        for name in sorted(expected):
            value = self._records[name][0]
            if target_device is not None:
                value = value.to(device=target_device)
            result[name] = value.clone()
        for index in range(block_count):
            source = (
                f"block.{index + 1}.input"
                if index + 1 < block_count
                else "decoder.input"
            )
            result[f"block.{index}.output"] = result[source]
        return result


def activation_differences(
    correct: Mapping[str, torch.Tensor],
    intervened: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return correct-minus-intervened differences for a paired trace."""

    if set(correct) != set(intervened):
        raise ValueError("paired activation traces must contain identical fields")
    differences = {}
    for name in correct:
        if correct[name].shape != intervened[name].shape:
            raise ValueError(f"paired activation shape mismatch at {name}")
        differences[name] = correct[name] - intervened[name]
    return differences
