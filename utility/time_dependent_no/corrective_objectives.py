"""Controlled sequence objectives and offline corrective-response measurements."""
from __future__ import annotations

import numpy as np
import torch

SEQUENCE_MODES = ("clean", "fixed", "detached", "attached")


def mixed_target(recovery, dynamics, dynamics_weight):
    if recovery.shape != dynamics.shape or not 0 <= dynamics_weight <= 1:
        raise ValueError("equal target shapes and a weight in [0, 1] are required")
    return ((1-dynamics_weight)*recovery + dynamics_weight*dynamics).detach()


def sample_windows(path_count, state_count, depth, *, per_path=32, seed=2026100401):
    last_start = state_count-depth-1
    if path_count < 1 or depth < 1 or per_path < 2 or last_start < per_path-1:
        raise ValueError("not enough complete sequence windows")
    rng = np.random.default_rng(seed)
    strata = np.array_split(np.arange(1, last_start+1), per_path-1)
    return [dict(path_index=p, start=t) for p in range(path_count)
            for t in [0, *[int(rng.choice(s)) for s in strata]]]


def sequence_batch(train, rows, indices, depth):
    indices = np.asarray(indices)
    if (indices.ndim != 1 or not len(indices) or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0) or np.any(indices >= len(rows)) or depth < 1):
        raise ValueError("invalid sequence indices or depth")
    selected = [rows[i] for i in indices]
    paths = np.array([r["path_index"] for r in selected])
    starts = np.array([r["start"] for r in selected])
    if (np.any(paths < 0) or np.any(paths >= len(train)) or np.any(starts < 0)
            or np.any(starts+depth >= train.shape[1])):
        raise ValueError("sequence crosses a trajectory boundary")
    return torch.from_numpy(np.array(train[paths[:, None], starts[:, None]+np.arange(depth+1)], copy=True))


def _sequence_check(sequence):
    if sequence.ndim < 3 or sequence.shape[1] < 2 or not torch.isfinite(sequence).all():
        raise ValueError("a finite batch of complete sequences is required")


def sequence_update(model, optimizer, sequence, mode, *, fixed_inputs=None):
    """One update; detached inputs preserve current-model forward trajectories."""
    _sequence_check(sequence)
    if mode not in SEQUENCE_MODES:
        raise ValueError("unknown sequence mode")
    depth = sequence.shape[1]-1
    if mode == "fixed" and (fixed_inputs is None or fixed_inputs.shape != sequence[:, :-1].shape
                            or not torch.isfinite(fixed_inputs).all()):
        raise ValueError("fixed inputs must align with the complete sequence")
    optimizer.zero_grad(set_to_none=True)
    state, losses, attached_terms = sequence[:, 0], [], []
    for j in range(depth):
        if mode == "clean":
            state = sequence[:, j]
        elif mode == "fixed":
            state = fixed_inputs[:, j].detach()
        prediction = model(state)["next_state"]
        loss = ((prediction-sequence[:, j+1])/model.train_scale).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite sequence loss")
        losses.append(float(loss.detach()))
        if mode == "attached":
            attached_terms.append(loss/depth)
            state = prediction
        else:
            (loss/depth).backward()
            state = prediction.detach()
    if mode == "attached":
        torch.stack(attached_terms).sum().backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    norm = sum(g.detach().double().square().sum() for g in gradients).sqrt()
    if not torch.isfinite(norm):
        raise RuntimeError("nonfinite sequence gradient")
    optimizer.step()
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise RuntimeError("nonfinite updated parameter")
    return dict(step_mse_scaled=losses, loss=sum(losses)/depth, gradient_l2=float(norm),
                state_predictions=int(sequence.shape[0]*depth))


def gradient_diagnostics(model, sequence):
    """Gradients at a common parameter value; no optimizer or .grad mutation."""
    _sequence_check(sequence)
    depth = sequence.shape[1]-1
    state = sequence[:, 0].detach().requires_grad_(True)
    states, losses = [], []
    for j in range(depth):
        state = model(state)["next_state"]
        states.append(state)
        losses.append(((state-sequence[:, j+1])/model.train_scale).square().mean()/depth)
    direct, future, cosines = [], [], []
    for j, state in enumerate(states):
        immediate, = torch.autograd.grad(losses[j], state, retain_graph=True)
        later = (torch.autograd.grad(sum(losses[j+1:]), state, retain_graph=True)[0]
                 if j+1 < depth else torch.zeros_like(state))
        a, b = immediate.detach().double().flatten(1), later.detach().double().flatten(1)
        an, bn = a.square().mean(1).sqrt(), b.square().mean(1).sqrt()
        cosine = torch.full_like(an, float("nan"))
        valid = (an > 0) & (bn > 0)
        cosine[valid] = (a*b).mean(1)[valid]/(an[valid]*bn[valid])
        direct.append(an.cpu().numpy()); future.append(bn.cpu().numpy())
        cosines.append(cosine.cpu().numpy())
    params = [p for p in model.parameters() if p.requires_grad]
    full = torch.autograd.grad(sum(losses), params, allow_unused=True)
    # Re-evaluate identical saved inputs, with only the input differentiation cut.
    detached_loss = sum(((model(x.detach())["next_state"]-sequence[:, j+1]) /
                         model.train_scale).square().mean()/depth
                        for j, x in enumerate([sequence[:, 0], *states[:-1]]))
    detached = torch.autograd.grad(detached_loss, params, allow_unused=True)
    a = torch.cat([(torch.zeros_like(p) if g is None else g).detach().double().flatten()
                   for p, g in zip(params, full)])
    b = torch.cat([(torch.zeros_like(p) if g is None else g).detach().double().flatten()
                   for p, g in zip(params, detached)])
    an, bn = a.norm(), b.norm()
    return dict(state_direct_rms=np.stack(direct, axis=1),
                state_future_rms=np.stack(future, axis=1), state_cosine=np.stack(cosines, axis=1),
                parameter_attached_l2=float(an), parameter_detached_l2=float(bn),
                parameter_cosine=float(torch.dot(a, b)/(an*bn)) if an > 0 and bn > 0 else float("nan"))


def pca_basis(bank, rank):
    """Thin PCA through a sample Gram matrix, discarding unresolved directions."""
    bank = np.asarray(bank, dtype=np.float64)
    if bank.ndim != 2 or len(bank) < 2 or rank < 1 or not np.isfinite(bank).all():
        raise ValueError("finite sample matrix and positive rank required")
    mean = bank.mean(axis=0)
    centered = bank-mean
    values, vectors = np.linalg.eigh(centered@centered.T)
    order = np.argsort(values)[::-1]
    threshold = max(float(values.max()), 0.)*max(bank.shape)*np.finfo(np.float64).eps
    selected = [i for i in order[:rank] if values[i] > threshold]
    basis = ((vectors[:, selected].T@centered)/np.sqrt(values[selected])[:, None]
             if selected else np.empty((0, bank.shape[1])))
    return mean, basis


def empirical_geometry(bank, queries, *, scale, ranks=(8, 16)):
    bank, queries = np.asarray(bank, dtype=float), np.asarray(queries, dtype=float)
    if (queries.ndim != 2 or bank.ndim != 2 or queries.shape[1] != bank.shape[1]
            or not np.isfinite(queries).all() or not np.isfinite(scale) or scale <= 0):
        raise ValueError("aligned finite query fields and positive scale required")
    mean, basis = pca_basis(bank, max(ranks))
    nearest = np.full(len(queries), np.inf)
    # Difference form avoids cancellation close to an archived training state.
    for state in bank:
        nearest = np.minimum(nearest, np.mean((queries-state)**2, axis=1)/scale**2)
    centered = queries-mean
    values = dict(nearest_mse=nearest)
    for rank in ranks:
        b = basis[:rank]
        residual = centered-(centered@b.T)@b
        values[f"pca_mse_r{rank}"] = np.mean(residual**2, axis=1)/scale**2
        values[f"effective_rank_r{rank}"] = np.full(len(queries), len(b), dtype=int)
    return values


def response_statistics(learned, true, displacement, scale):
    learned, true, displacement = [np.asarray(x, dtype=float) for x in (learned, true, displacement)]
    if learned.shape != true.shape or learned.shape != displacement.shape:
        raise ValueError("response and displacement shapes must agree")
    rms = lambda x: float(np.sqrt(np.mean(x*x)))
    radius, ln, tn = rms(displacement), rms(learned), rms(true)
    if not np.isfinite([radius, ln, tn, scale]).all() or radius <= 0 or scale <= 0:
        raise ValueError("finite responses, nonzero displacement and positive scale required")
    return dict(input_rms_scaled=radius/scale, learned_gain=ln/radius, true_gain=tn/radius,
                response_error_gain=rms(learned-true)/radius,
                cosine=float(np.mean(learned*true)/(ln*tn)) if ln > 0 and tn > 0 else float("nan"))
