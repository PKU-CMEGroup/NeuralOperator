"""Offline intervention responses along observed baseline trajectories."""
from __future__ import annotations

import torch

from utility.time_dependent_no.forced_tangent import _evaluate, _field, _jvp


def initial_displacement_response(step, baseline, initial, *, start_index=0):
    """Propagate an initial state change, with separate nonrecurrent probes.

    z[n+1]=D step(v[n])z[n]. The nonlinear signed remainders never change z or
    the queried baseline states. q[n+1]=D step(v[n])q[n]+r_plus[n] estimates
    accumulated omitted curvature; it is neither a bound nor a candidate path.
    """
    _field(baseline, "baseline")
    _field(initial, "initial")
    if baseline.ndim < 2 or len(baseline) < 2:
        raise ValueError("baseline requires at least two states")
    if (initial.shape != baseline.shape[1:] or initial.dtype != baseline.dtype
            or initial.device != baseline.device):
        raise ValueError("initial displacement must match baseline states")
    if type(start_index) is not int or start_index < 0:
        raise ValueError("start_index must be a nonnegative integer")
    baseline = baseline.detach()
    z, q = [initial.detach()], [torch.zeros_like(initial)]
    plus, minus, replay = [], [], []
    for k, state in enumerate(baseline[:-1]):
        n = start_index+k
        value, next_z = _jvp(step, n, state, z[-1])
        _, propagated_q = _jvp(step, n, state, q[-1])
        with torch.no_grad():
            if torch.any(z[-1] != 0) and (torch.equal(state+z[-1], state)
                                         or torch.equal(state-z[-1], state)):
                raise ValueError("probe displacement is unresolved")
            plus.append(_evaluate(step, n, state+z[-1])-value-next_z)
            minus.append(_evaluate(step, n, state-z[-1])-value+next_z)
            q.append(propagated_q+plus[-1])
            _field(q[-1], "curvature estimate")
            z.append(next_z)
            replay.append(value-baseline[k+1])
    return {name: torch.stack(values).detach() for name, values in (
        ("displacement", z), ("curvature_estimate", q),
        ("plus_remainder", plus), ("minus_remainder", minus),
        ("baseline_replay_residual", replay))}


def blend_sensitivity(base_step, other_step, baseline):
    """Differentiate ``F_a=(1-a)F_0+aF_1`` at a=0, with a fixed initial state.

    The returned z satisfies z[n+1]=DF_0(v[n])z[n]+F_1(v[n])-F_0(v[n]).
    Both maps are queried only at the supplied baseline states. Their replay
    residual is reported separately and never injected as parameter forcing.
    The caller must check that these states are a trajectory of the baseline.
    No candidate trajectory is generated and no time-spanning graph is kept.
    """
    _field(baseline, "baseline")
    if baseline.ndim < 2 or len(baseline) < 2:
        raise ValueError("baseline must have shape [T+1, ...] with T positive")
    baseline = baseline.detach()
    sensitivity = [torch.zeros_like(baseline[0])]
    local_only = [torch.zeros_like(baseline[0])]
    forcing, replay = [], []
    for n, state in enumerate(baseline[:-1]):
        value, propagated = _jvp(base_step, n, state, sensitivity[-1])
        with torch.no_grad():
            change = _evaluate(other_step, n, state) - value
            next_sensitivity = propagated + change
            if not torch.isfinite(next_sensitivity).all():
                raise ArithmeticError("nonfinite intervention sensitivity")
            sensitivity.append(next_sensitivity)
            local_only.append(local_only[-1] + change)
            forcing.append(change)
            replay.append(value - baseline[n + 1])
    return dict(sensitivity=torch.stack(sensitivity).detach(),
                map_difference=torch.stack(forcing).detach(),
                identity_response_sensitivity=torch.stack(local_only).detach(),
                baseline_replay_residual=torch.stack(replay).detach())


def local_blend_remainder(base_step, other_step, index, state, direction,
                          next_direction, coefficients):
    """Check the first variation on sparse signed, nonrecurrent probes.

    For a signed a, compare F_a(v+a*z) with F_0(v)+a*z_next. Negative
    coefficients are derivative probes, not a proposed deployed correction.
    Results are vectors in physical units; no uniform remainder bound is claimed.
    """
    _field(state, "state")
    for name, value in (("direction", direction), ("next_direction", next_direction)):
        _field(value, name)
        if value.shape != state.shape or value.dtype != state.dtype or value.device != state.device:
            raise ValueError("directions must match the state")
    coefficients = torch.as_tensor(coefficients, device=state.device, dtype=state.dtype)
    if (coefficients.ndim != 1 or len(coefficients) == 0 or
            not torch.isfinite(coefficients).all() or not (coefficients > 0).all()):
        raise ValueError("coefficients must be finite and positive")
    with torch.no_grad():
        base = _evaluate(base_step, index, state)
        residuals, displacements = [], []
        for a in coefficients:
            pair, realized = [], []
            for sign in (1, -1):
                displaced = state + sign*a*direction
                if torch.equal(displaced, state):
                    raise ValueError("probe displacement is unresolved")
                first = _evaluate(base_step, index, displaced)
                other = _evaluate(other_step, index, displaced)
                value = first + sign*a*(other-first)
                pair.append(value-base-sign*a*next_direction)
                realized.append(displaced-state)
            residuals.append(torch.stack(pair))
            displacements.append(torch.stack(realized))
    return dict(coefficients=coefficients.detach(), remainder=torch.stack(residuals),
                realized_displacement=torch.stack(displacements))
