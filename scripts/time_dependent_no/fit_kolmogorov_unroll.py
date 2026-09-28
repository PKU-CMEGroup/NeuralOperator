"""Matched K=4 PCNO unrolling with detached or full temporal gradients.

Run ``python -m scripts.time_dependent_no.fit_kolmogorov_unroll --input TRAIN
--output NEW_DIRECTORY --arm detached|full --phase resource|fit --device cuda``.
Both arms restart the same parent with fresh Adam, sample the same four clean
sequence starts per update, and average all four successor losses. There is no
solver, extra history at deployment, or development-based checkpoint selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.time_dependent_no import fit_kolmogorov_recovery as base

ARMS = ("detached", "full")
RECIPE = "fixed32_unroll4_v1"
DEPTH, BATCH_SIZE, FIT_UPDATES, RESOURCE_UPDATES = 4, 4, 4096, 16
LEARNING_RATE, SAMPLER_SEED, MAX_FIT_SECONDS = 1e-4, 17, 4 * 3600
SOURCE_PATHS = base.SOURCE_PATHS | {
    "scripts/time_dependent_no/fit_kolmogorov_unroll.py",
    "tests/time_dependent_no/test_fit_kolmogorov_unroll.py",
}


def sequence_batch(train, indices):
    """Every full K-step window is eligible; never cross trajectory boundaries."""
    starts = train.shape[1] - DEPTH
    indices = np.asarray(indices)
    if (starts < 1 or indices.ndim != 1 or not len(indices)
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0) or np.any(indices >= len(train) * starts)):
        raise ValueError("invalid sequence indices")
    paths, times = np.divmod(indices, starts)
    return torch.from_numpy(np.array(
        train[paths[:, None], times[:, None] + np.arange(DEPTH + 1)], copy=True))


def update(model, optimizer, sequence, arm, device):
    if arm not in ARMS or sequence.shape[1] != DEPTH + 1:
        raise ValueError("unknown arm or sequence depth")
    if not torch.isfinite(sequence).all():
        raise ValueError("nonfinite training sequence")
    sequence = sequence.to(device)
    optimizer.zero_grad(set_to_none=True)
    state, losses, terms = sequence[:, 0], [], []
    for step in range(DEPTH):
        state = model(state)["next_state"]
        loss = ((state - sequence[:, step + 1]) / model.train_scale).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite unrolled loss")
        losses.append(float(loss.detach()))
        if arm == "detached":
            (loss / DEPTH).backward()
            state = state.detach()
        else:
            terms.append(loss / DEPTH)
    if arm == "full":
        torch.stack(terms).sum().backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    gradient_norm = torch.stack([g.detach().double().square().sum()
                                 for g in gradients]).sum().sqrt()
    if not torch.isfinite(gradient_norm):
        raise RuntimeError("nonfinite unrolled gradient")
    optimizer.step()
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise RuntimeError("nonfinite updated parameter")
    return dict(step_mse_scaled=losses, loss=sum(losses) / DEPTH,
                gradient_l2=float(gradient_norm))


def recipe_identity(manifest, parent_fit, manifest_hash, arm):
    if arm not in ARMS:
        raise ValueError("unknown unroll arm")
    parent = parent_fit["checkpoint_identity"]
    return dict(recipe=RECIPE, arm=arm,
                parent_checkpoint_sha256=manifest["artifacts"]["checkpoint.pt"],
                input_manifest_sha256=manifest_hash, model_config=parent_fit["model_config"],
                train_scale_float64=parent["train_scale_float64"],
                train_scale_model_float32=parent["train_scale_model_float32"],
                updates=FIT_UPDATES, depth=DEPTH, batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE, sampler_seed=SAMPLER_SEED,
                loss_weights=[1. / DEPTH] * DEPTH,
                temporal_gradients=arm == "full",
                targets="exact archived clean sequence; no extra labels",
                deployment="unchanged one-state restricted PCNO transition")


def _run(input_path, output, arm, phase, device, result):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(17)
    manifest, captured, parent_fit, train, manifest_hash = base.load_inputs(input_path)
    sources = {name: base.sha256(base.ROOT / name) for name in sorted(SOURCE_PATHS)}
    output.mkdir(parents=True, exist_ok=False)
    model = base.load_model(captured, parent_fit, device)
    replay = base.terminal_replay(model, captured["teacher.npz"], device, len(base.TRAIN_SEEDS))
    if float(model.train_scale) != parent_fit["checkpoint_identity"]["train_scale_model_float32"]:
        raise ValueError("parent normalizer mismatch")
    model.train().requires_grad_(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    starts_per_path = train.shape[1] - DEPTH
    sampler = base.EpochSampler(len(train) * starts_per_path, BATCH_SIZE, SAMPLER_SEED)
    identity = recipe_identity(manifest, parent_fit, manifest_hash, arm)
    result.update(schema_version=1, status="running", identity=identity,
                  input_manifest=manifest, sources=sources, parent_replay=replay,
                  optimizer="fresh Adam; parent optimizer/RNG discarded",
                  numerical=dict(dtype="float32", amp=False, tf32=False, cpu_threads=2),
                  device=str(device), torch_version=str(torch.__version__), artifacts={},
                  starts_per_path=starts_per_path,
                  state_predictions_per_update=BATCH_SIZE * DEPTH,
                  max_fit_seconds=MAX_FIT_SECONDS)
    base.write_json(output / "result.json", result)
    count = RESOURCE_UPDATES if phase == "resource" else FIT_UPDATES
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    base._sync(device)
    started, durations = time.perf_counter(), []
    with (output / "updates.jsonl").open("x") as log:
        for step in range(1, count + 1):
            tick = time.perf_counter()
            indices = sampler.next_indices()
            row = update(model, optimizer, sequence_batch(train, indices), arm, device)
            base._sync(device)
            durations.append(time.perf_counter() - tick)
            row.update(update=step, sequence_indices=indices.tolist(), seconds=durations[-1])
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
            result["updates_completed"] = step
            if step % 256 == 0:
                base.write_json(output / "result.json", result)
            if phase == "fit" and time.perf_counter() - started > MAX_FIT_SECONDS:
                raise RuntimeError("unroll fit exceeded the declared wall budget")
            if step == RESOURCE_UPDATES:
                estimate = np.mean(durations[min(4, len(durations) - 1):]) * FIT_UPDATES
                if phase == "fit" and estimate > MAX_FIT_SECONDS:
                    raise RuntimeError("warmed unroll estimate exceeds the declared wall budget")
    warm = durations[min(4, len(durations) - 1):]
    result.update(status="completed", training_seconds=time.perf_counter() - started,
                  warmed_seconds_per_update=float(np.mean(warm)),
                  estimated_fit_seconds=float(np.mean(warm)) * FIT_UPDATES,
                  within_fit_budget=float(np.mean(warm)) * FIT_UPDATES <= MAX_FIT_SECONDS,
                  peak_cuda_allocated_bytes=(torch.cuda.max_memory_allocated(device)
                                             if torch.device(device).type == "cuda" else None),
                  peak_cuda_reserved_bytes=(torch.cuda.max_memory_reserved(device)
                                            if torch.device(device).type == "cuda" else None),
                  final_training_row=row)
    if phase == "fit":
        checkpoint = output / "terminal.pt"
        torch.save(dict(identity=identity, update=count, schedule_position=count,
                        model=model.state_dict()), checkpoint)
        result["artifacts"]["terminal.pt"] = base.sha256(checkpoint)
    result["artifacts"]["updates.jsonl"] = base.sha256(output / "updates.jsonl")
    if sources != {name: base.sha256(base.ROOT / name) for name in sorted(SOURCE_PATHS)}:
        raise RuntimeError("unroll source changed during execution")
    if base.sha256(input_path / "manifest.json") != manifest_hash:
        raise RuntimeError("input manifest changed during execution")
    if any(base.sha256(input_path / name) != digest for name, digest in manifest["artifacts"].items()):
        raise RuntimeError("training input changed during execution")
    base.write_json(output / "result.json", result)
    return result


def run(input_path, output_path, arm, phase, device):
    if arm not in ARMS or phase not in ("resource", "fit"):
        raise ValueError("unknown arm or phase")
    input_path, output = Path(input_path).resolve(), Path(output_path).resolve()
    if output.is_relative_to(input_path):
        raise ValueError("output must be outside the immutable input directory")
    if output.exists():
        raise FileExistsError(output)
    result = dict(status="initializing", phase=phase, arm=arm, updates_completed=0)
    try:
        return _run(input_path, output, arm, phase, device, result)
    except Exception as error:
        if output.exists():
            result.update(status="failed", error=f"{type(error).__name__}: {error}")
            base.write_json(output / "result.json", result)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--phase", required=True, choices=("resource", "fit"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args.input, args.output, args.arm, args.phase, args.device)


if __name__ == "__main__":
    main()
