"""Offline clean-path error forecasts and separate signed response probes.

The forecast differentiates the complete deployed transition at clean states
only. It never reads a learned rollout or feeds a nonlinear probe into its
recurrence. All returned tensors are detached; this is not a training loss.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch


def _field(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch Tensor")
    if value.ndim < 1 or value.numel() == 0:
        raise ValueError(f"{name} must have nonempty state dimensions")
    if value.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"{name} must be real float32 or float64")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")


def _evaluate(step, index, state):
    value = step(index, state)
    _field(value, "step output")
    if (
        value.shape != state.shape
        or value.dtype != state.dtype
        or value.device != state.device
    ):
        raise ValueError("step output must match the state shape, dtype and device")
    return value


def _jvp(step, index, state, direction):
    if torch.is_inference_mode_enabled():
        raise ValueError("tangent evaluation requires autograd, not inference_mode")
    # Use reverse-over-reverse: PCNO's CUDA custom backward operators do not
    # implement the torch.func forward-mode interface.
    with torch.enable_grad():
        value, tangent = torch.autograd.functional.jvp(
            lambda x: _evaluate(step, index, x),
            state.detach().clone(),
            direction.detach().clone(),
            create_graph=False,
            strict=False,
        )
    if not torch.isfinite(tangent).all():
        raise ArithmeticError("nonfinite tangent response")
    return value.detach(), tangent.detach()


def forced_tangent_forecast(
    step: Callable[[int, torch.Tensor], torch.Tensor],
    reference: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Propagate ``e[n+1] = b[n] + D step(n, u[n]) e[n]`` from zero error.

    ``reference`` has shape ``[T+1, ...]`` with at least one state dimension.
    ``step`` must be a deterministic complete transition on that device/dtype.
    It is called only at the clean ``reference[n]``, including when the
    predicted error is large. The caller owns model mode and validity limits.

    Returns ``clean_forcing[T,...]`` and ``predicted_error[T+1,...]`` in
    physical state units. ``identity_response_error[T+1,...]`` is the explicit
    J=I baseline: cumulative clean forcing, without response amplification.
    No parameter gradients are accumulated and no time-spanning graph is kept.
    An outer ``torch.no_grad()`` is supported; ``inference_mode`` is not.
    """
    _field(reference, "reference")
    if reference.ndim < 2 or reference.shape[0] < 2:
        raise ValueError("reference must have shape [T+1, ...] with T at least one")
    reference = reference.detach()
    errors = [torch.zeros_like(reference[0])]
    identity_errors = [torch.zeros_like(reference[0])]
    forcing = []
    for index in range(reference.shape[0] - 1):
        value, response = _jvp(step, index, reference[index], errors[-1])
        defect = value - reference[index + 1]
        error = defect + response
        identity_error = defect + identity_errors[-1]
        if not all(torch.isfinite(x).all() for x in (defect, error, identity_error)):
            raise ArithmeticError("nonfinite forced tangent recurrence")
        forcing.append(defect)
        errors.append(error)
        identity_errors.append(identity_error)
    return {
        "clean_forcing": torch.stack(forcing).detach(),
        "predicted_error": torch.stack(errors).detach(),
        "identity_response_error": torch.stack(identity_errors).detach(),
    }


def finite_amplitude_response(
    step: Callable[[int, torch.Tensor], torch.Tensor],
    step_index: int,
    state: torch.Tensor,
    direction: torch.Tensor,
    multipliers: Sequence[float],
) -> dict[str, torch.Tensor]:
    """Probe signed displacements separately from construction of a forecast.

    For each positive multiplier ``a``, query ``step(n, state +/- a*direction)``.
    ``odd_response`` is half the plus/minus difference; ``even_response`` is
    their average minus the clean prediction. ``linear_response = a*Jd``.
    The two remainders subtract that signed linear prediction from each
    nonlinear response. The direction is not normalized or projected.

    Base value and Jd have the state shape; other fields have leading probe
    dimension, except ``multipliers[K]``. Realized signed displacements are
    retained to distinguish finite-precision rounding from nonlinear error;
    the linear responses use the intended displacements. Completely unresolved
    signed inputs are rejected. These measurements never update a forecast.
    """
    if (
        isinstance(step_index, bool)
        or not isinstance(step_index, int)
        or step_index < 0
    ):
        raise ValueError("step_index must be a nonnegative integer")
    _field(state, "state")
    _field(direction, "direction")
    if (
        direction.shape != state.shape
        or direction.dtype != state.dtype
        or direction.device != state.device
    ):
        raise ValueError("direction must match the state shape, dtype and device")
    if not torch.any(direction != 0):
        raise ValueError("a zero direction does not define a response probe")
    amplitudes = torch.as_tensor(
        multipliers, dtype=state.dtype, device=state.device
    ).detach()
    if (
        amplitudes.ndim != 1
        or amplitudes.numel() == 0
        or not torch.isfinite(amplitudes).all()
        or not torch.all(amplitudes > 0)
    ):
        raise ValueError("multipliers must be a nonempty positive finite sequence")
    state, direction = state.detach(), direction.detach()
    signed_inputs = []
    for amplitude in amplitudes:
        plus, minus = state + amplitude * direction, state - amplitude * direction
        if not torch.isfinite(plus).all() or not torch.isfinite(minus).all():
            raise ValueError("nonfinite signed probe input")
        if torch.equal(plus, state) or torch.equal(minus, state):
            raise ValueError("signed displacement is unresolved in the state dtype")
        signed_inputs.append((plus, minus))
    base, tangent = _jvp(step, step_index, state, direction)
    rows = {
        name: []
        for name in (
            "linear_response",
            "odd_response",
            "even_response",
            "plus_remainder",
            "minus_remainder",
            "plus_displacement",
            "minus_displacement",
        )
    }
    with torch.no_grad():
        for amplitude, (plus, minus) in zip(amplitudes, signed_inputs, strict=True):
            plus_value = _evaluate(step, step_index, plus)
            minus_value = _evaluate(step, step_index, minus)
            plus_response, minus_response = plus_value - base, minus_value - base
            linear = amplitude * tangent
            values = {
                "linear_response": linear,
                "odd_response": (plus_response - minus_response) / 2,
                "even_response": (plus_response + minus_response) / 2,
                "plus_remainder": plus_response - linear,
                "minus_remainder": minus_response + linear,
                "plus_displacement": plus - state,
                "minus_displacement": minus - state,
            }
            if not all(torch.isfinite(value).all() for value in values.values()):
                raise ArithmeticError("nonfinite finite-amplitude response")
            for name, value in values.items():
                rows[name].append(value)
    return {
        "base_value": base,
        "jvp": tangent,
        "multipliers": amplitudes.clone(),
        **{name: torch.stack(values).detach() for name, values in rows.items()},
    }


def decompose_forecast_error(
    step: Callable[[int, torch.Tensor], torch.Tensor],
    step_index: int,
    reference: torch.Tensor,
    forecast: dict[str, torch.Tensor],
    observed_errors: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Posthoc decomposition of a saved forecast discrepancy, never a forecast.

    For saved forecast error z, observed error e and d=z-e, measure
    d_next = Jd - R_ng(e) + offset + closure_residual. Both JVPs are based at
    clean u. R_ng uses no-grad evaluations at u and reconstructed u+e; offset
    is saved clean forcing minus the replayed no-grad clean defect.

    Returned vectors and scalar L2 closure diagnostics are detached FP64 on
    the input device; model calls retain the input dtype. Closure includes
    staged/replayed update differences, JVP-linearity roundoff and possible
    u+e reconstruction/replay drift. The caller checks saved actual-state
    replay and chooses the closure tolerance; no error bound is asserted.
    """
    if (
        isinstance(step_index, bool)
        or not isinstance(step_index, int)
        or reference.ndim < 2
        or not 0 <= step_index < reference.shape[0] - 1
    ):
        raise ValueError("step_index must select a complete reference transition")
    predicted = forecast["predicted_error"]
    forcing = forecast["clean_forcing"]
    if (
        predicted.shape != reference.shape
        or observed_errors.shape != reference.shape
        or forcing.shape != reference[:-1].shape
    ):
        raise ValueError("saved forecast and observations must match the reference")
    u, next_u = reference[step_index : step_index + 2].detach()
    z, next_z = predicted[step_index : step_index + 2].detach()
    e, next_e = observed_errors[step_index : step_index + 2].detach()
    saved_forcing = forcing[step_index].detach()
    for value in (u, next_u, z, next_z, e, next_e, saved_forcing):
        _field(value, "decomposition field")
        if value.dtype != u.dtype or value.device != u.device:
            raise ValueError("decomposition fields must share dtype and device")
    with torch.no_grad():
        _, propagated = _jvp(step, step_index, u, z - e)
        _, response_e = _jvp(step, step_index, u, e)
        clean = _evaluate(step, step_index, u.clone()).double()
        displaced = _evaluate(step, step_index, u + e).double()
        discrepancy = next_z.double() - next_e.double()
        remainder = displaced - clean - response_e.double()
        offset = saved_forcing.double() - (clean - next_u.double())
        propagated = propagated.double()
        reconstructed = propagated - remainder + offset
        residual = discrepancy - reconstructed
        norm = torch.linalg.vector_norm
        scale = torch.maximum(
            norm(discrepancy), norm(propagated) + norm(remainder) + norm(offset)
        )
        relative = torch.where(
            scale > 0, norm(residual) / scale, torch.zeros_like(scale)
        )
        return {
            "discrepancy_now": z.double() - e.double(),
            "discrepancy_next": discrepancy,
            "propagated_discrepancy": propagated,
            "actual_remainder": remainder,
            "clean_kernel_offset": offset,
            "reconstructed_discrepancy": reconstructed,
            "closure_residual": residual,
            "closure_scale_l2": scale,
            "closure_relative_l2": relative,
        }
