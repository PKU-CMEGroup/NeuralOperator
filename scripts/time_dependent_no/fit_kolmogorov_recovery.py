"""Matched clean continuation and Gaussian recovery on one fixed training set.

Run ``python -m scripts.time_dependent_no.fit_kolmogorov_recovery --input PACKET
--output NEW_DIRECTORY --arm clean_continuation|recovery --phase resource|fit
--device cuda``. Both phases restart the original parent with fresh Adam.
There is no solver, development population, or checkpoint selection here.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from scripts.time_dependent_no.fit_kolmogorov_clean import EpochSampler
from scripts.time_dependent_no.evaluate_kolmogorov_tangent_forecast import (
    MODEL_SOURCES, ROOT, load_model, sha256, terminal_replay, write_json,
)
from utility.time_dependent_no.pcno_kolmogorov import canonicalize_vorticity

TRAIN_SEEDS = list(range(2026090611, 2026090619)) + list(range(2026090801, 2026090825))
TRAIN_SHAPE = (32, 513, 256, 256)
INPUT_FILES = {"train.npy", "checkpoint.pt", "fit_result.json", "teacher.npz"}
FIT_UPDATES, RESOURCE_UPDATES, BATCH_SIZE = 4096, 16, 8
SIGMA, LEARNING_RATE, MAX_FIT_SECONDS = 0.01, 1e-4, 3 * 3600
SOURCE_PATHS = MODEL_SOURCES | {
    "scripts/time_dependent_no/fit_kolmogorov_recovery.py",
    "tests/time_dependent_no/test_fit_kolmogorov_recovery.py",
    "scripts/time_dependent_no/fit_kolmogorov_clean.py",
    "scripts/time_dependent_no/evaluate_kolmogorov_tangent_forecast.py",
    "utility/time_dependent_no/forced_tangent.py",
}


def load_inputs(path):
    """Authenticate the train-only packet before mapping any training arrays."""
    path = Path(path).resolve()
    raw = (path / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    expected = dict(schema_version=1, role="fixed_data_train", train_seeds=TRAIN_SEEDS,
                    shape=list(TRAIN_SHAPE), dtype="float32")
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("input must be the fixed 32-path training population")
    if set(manifest["artifacts"]) != INPUT_FILES or set(manifest["sources"]) != MODEL_SOURCES:
        raise ValueError("unexpected artifact or model source set")
    for name, digest in manifest["sources"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"source hash mismatch: {name}")
    captured = {}
    for name in sorted(INPUT_FILES):
        member = (path / name).resolve()
        if not member.is_relative_to(path):
            raise ValueError("artifact escapes input directory")
        if name == "train.npy":
            digest = sha256(member)
        else:
            captured[name] = member.read_bytes()
            digest = hashlib.sha256(captured[name]).hexdigest()
        if digest != manifest["artifacts"][name]:
            raise ValueError(f"artifact hash mismatch: {name}")
    fit = json.loads(captured["fit_result.json"])
    terminal = fit["terminal_checkpoint"]
    if (fit["updates_completed"] != 49152 or terminal["update"] != 49152
            or terminal["file"] != "terminal_049152.pt"
            or terminal["sha256"] != manifest["artifacts"]["checkpoint.pt"]
            or fit["model_config"] != fit["checkpoint_identity"]["model_config"]
            or fit["model_config"]["resolution"] != TRAIN_SHAPE[-1]):
        raise ValueError("parent fit/checkpoint binding mismatch")
    train = np.load(path / "train.npy", mmap_mode="r", allow_pickle=False)
    if train.shape != TRAIN_SHAPE or train.dtype != np.float32:
        raise ValueError("training array shape/dtype mismatch")
    with np.load(io.BytesIO(captured["teacher.npz"]), allow_pickle=False) as teacher:
        indices = teacher["sentinel_global_indices"]
        if (set(teacher.files) != {"sentinel_global_indices", "sentinel_input", "sentinel_target", "sentinel_raw", "sentinel_next"}
                or indices.shape != (4,) or not np.issubdtype(indices.dtype, np.integer)
                or np.any(indices < 0) or np.any(indices >= train.shape[0] * (train.shape[1] - 1))):
            raise ValueError("teacher must contain four training-only sentinels")
        paths, steps = np.divmod(indices, train.shape[1] - 1)
        if not np.array_equal(teacher["sentinel_input"], train[paths, steps]):
            raise ValueError("teacher inputs do not match archived training states")
        if not np.array_equal(teacher["sentinel_target"], train[paths, steps + 1]):
            raise ValueError("teacher targets do not match archived training successors")
    return manifest, captured, fit, train, hashlib.sha256(raw).hexdigest()


def gaussian_noise(shape, scale, generator):
    """Gaussian on the canonical subspace, with E[RMS**2]=(SIGMA*scale)**2."""
    n = shape[-1]
    rank = (2 * (n // 3) + 1) ** 2 - 1
    white = torch.randn(shape, generator=generator, dtype=torch.float32, device="cpu")
    return canonicalize_vorticity(white) * (SIGMA * scale * n / math.sqrt(rank))


def training_batches(train, first_indices, second_indices, arm, scale, generator):
    """Only the second input changes; targets are exact archived successors."""
    if arm not in ("clean_continuation", "recovery"):
        raise ValueError("unknown training arm")
    batches = []
    noise_rms = 0.0
    for branch, indices in enumerate((first_indices, second_indices)):
        paths, steps = np.divmod(indices, train.shape[1] - 1)
        inputs = torch.from_numpy(np.array(train[paths, steps], copy=True))
        targets = torch.from_numpy(np.array(train[paths, steps + 1], copy=True))
        if branch == 1 and arm == "recovery":
            noise = gaussian_noise(inputs.shape, scale, generator)
            inputs = inputs + noise
            noise_rms = float(noise.double().square().mean().sqrt()) / scale
        batches.append((inputs, targets))
    return batches, noise_rms


def update(model, optimizer, batches, device):
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for inputs, targets in batches:
        if not (torch.isfinite(inputs).all() and torch.isfinite(targets).all()):
            raise ValueError("nonfinite training pair")
        prediction = model(inputs.to(device))["next_state"]
        loss = ((prediction - targets.to(device)) / model.train_scale).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite training loss")
        (0.5 * loss).backward()
        losses.append(float(loss.detach()))
        del prediction, loss
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    gradient_norm = torch.stack([g.detach().double().square().sum() for g in gradients]).sum().sqrt()
    if not torch.isfinite(gradient_norm):
        raise RuntimeError("nonfinite gradient")
    optimizer.step()
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise RuntimeError("nonfinite updated parameter")
    return dict(first_mse_scaled=losses[0], second_mse_scaled=losses[1],
                loss=0.5 * sum(losses), gradient_l2=float(gradient_norm))


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _run(input_path, output_path, arm, phase, device, result):
    if arm not in ("clean_continuation", "recovery") or phase not in ("resource", "fit"):
        raise ValueError("unknown arm or phase")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(17)
    manifest, captured, fit, train, manifest_hash = load_inputs(input_path)
    sources = {name: sha256(ROOT / name) for name in sorted(SOURCE_PATHS)}
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=False)
    model = load_model(captured, fit, device)
    replay = terminal_replay(model, captured["teacher.npz"], device, len(TRAIN_SEEDS))
    model.train().requires_grad_(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    pair_count = train.shape[0] * (train.shape[1] - 1)
    samplers = [EpochSampler(pair_count, BATCH_SIZE, seed) for seed in (17, 1701)]
    noise_generator = torch.Generator(device="cpu").manual_seed(1702)
    scale = float(model.train_scale)
    identity = dict(parent_checkpoint_sha256=manifest["artifacts"]["checkpoint.pt"],
                    input_manifest_sha256=manifest_hash, arm=arm,
                    recipe="fixed32_gaussian_recovery_v1", model_config=fit["model_config"],
                    train_scale_float64=fit["checkpoint_identity"]["train_scale_float64"],
                    train_scale_model_float32=scale, updates=FIT_UPDATES,
                    learning_rate=LEARNING_RATE, batch_size_per_branch=BATCH_SIZE,
                    branch_weights=[0.5, 0.5], sampler_seeds=[17, 1701], noise_seed=1702,
                    noise_expected_rms_scaled=SIGMA if arm == "recovery" else 0.0,
                    targets="exact archived clean successors; no extra labels")
    result.update(schema_version=1, status="running", phase=phase, identity=identity,
                  input_manifest=manifest, sources=sources, parent_replay=replay,
                  updates_completed=0, optimizer="fresh Adam; parent optimizer/RNG discarded",
                  numerical=dict(dtype="float32", amp=False, tf32=False, cpu_threads=2),
                  device=str(device), torch_version=str(torch.__version__), artifacts={})
    write_json(output / "result.json", result)
    count = RESOURCE_UPDATES if phase == "resource" else FIT_UPDATES
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    started, durations = time.perf_counter(), []
    with (output / "updates.jsonl").open("x") as log:
        for step in range(1, count + 1):
            tick = time.perf_counter()
            first, second = [sampler.next_indices() for sampler in samplers]
            batches, noise_rms = training_batches(train, first, second, arm, scale, noise_generator)
            row = update(model, optimizer, batches, device)
            _sync(device)
            durations.append(time.perf_counter() - tick)
            row.update(update=step, first_indices=first.tolist(), second_indices=second.tolist(),
                       noise_rms_scaled=noise_rms, seconds=durations[-1])
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
            result["updates_completed"] = step
            if step % 256 == 0:
                write_json(output / "result.json", result)
            if phase == "fit" and time.perf_counter() - started > MAX_FIT_SECONDS:
                raise RuntimeError("fixed fit exceeded the three-hour budget")
            if step == RESOURCE_UPDATES:
                estimate = float(np.mean(durations[min(4, len(durations) - 1):])) * FIT_UPDATES
                if phase == "fit" and estimate > MAX_FIT_SECONDS:
                    raise RuntimeError("warmed fit estimate exceeds three-hour budget")
    warm = durations[min(4, len(durations) - 1):]
    result.update(status="completed", updates_completed=count,
                  training_seconds=time.perf_counter() - started,
                  warmed_seconds_per_update=float(np.mean(warm)),
                  estimated_fit_seconds=float(np.mean(warm)) * FIT_UPDATES,
                  within_three_hour_fit_budget=float(np.mean(warm)) * FIT_UPDATES <= MAX_FIT_SECONDS,
                  peak_cuda_allocated_bytes=(torch.cuda.max_memory_allocated(device)
                                             if torch.device(device).type == "cuda" else None),
                  peak_cuda_reserved_bytes=(torch.cuda.max_memory_reserved(device)
                                            if torch.device(device).type == "cuda" else None),
                  final_training_row=row)
    if phase == "fit":
        checkpoint = output / "terminal.pt"
        torch.save(dict(identity=identity, update=count, schedule_position=count,
                        model=model.state_dict()), checkpoint)
        result["artifacts"]["terminal.pt"] = sha256(checkpoint)
    result["artifacts"]["updates.jsonl"] = sha256(output / "updates.jsonl")
    if sources != {name: sha256(ROOT / name) for name in sorted(SOURCE_PATHS)}:
        raise RuntimeError("training source changed during execution")
    write_json(output / "result.json", result)
    return result


def run(input_path, output_path, arm, phase, device):
    input_path, output_path = Path(input_path).resolve(), Path(output_path).resolve()
    if output_path.is_relative_to(input_path):
        raise ValueError("output must be outside the immutable input directory")
    if output_path.exists():
        raise FileExistsError(output_path)
    result = dict(status="initializing", phase=phase, arm=arm, updates_completed=0)
    try:
        return _run(input_path, output_path, arm, phase, device, result)
    except Exception as error:
        if output_path.exists():
            result.update(status="failed", error=f"{type(error).__name__}: {error}")
            write_json(output_path / "result.json", result)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=("clean_continuation", "recovery"))
    parser.add_argument("--phase", required=True, choices=("resource", "fit"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args.input, args.output, args.arm, args.phase, args.device)


if __name__ == "__main__":
    main()
