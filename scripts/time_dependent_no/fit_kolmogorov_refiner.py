"""Bounded, train-only acquisition pilot for conditional PCNO PDE-Refiner.

Run ``python -m scripts.time_dependent_no.fit_kolmogorov_refiner --input TRAIN
--output NEW_DIRECTORY --phase resource|fit --device cuda``. The fit stops at
8,192 updates, retaining raw/EMA weights and continuation state. Training-only
probes diagnose acquisition; no development state or autonomous rollout is read.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.time_dependent_no import fit_kolmogorov_recovery as base
from utility.time_dependent_no.pcno_kolmogorov_refiner import (
    PeriodicVorticityRefiner, noise_tape,
)

RECIPE = "fixed32_conditional_refiner_acquisition_v1"
FIT_UPDATES, RESOURCE_UPDATES, BATCH_SIZE = 8192, 16, 8
PROBE_INTERVAL = 2048
LEARNING_RATE, EMA_DECAY, MAX_FIT_SECONDS = 1e-4, .995, 5*3600
PROBE_STEPS, SAMPLER_SEEDS = (0, 7, 31, 255), (101, 211, 307)
SOURCE_PATHS = base.SOURCE_PATHS | {
    "utility/time_dependent_no/pcno_kolmogorov_refiner.py",
    "scripts/time_dependent_no/fit_kolmogorov_refiner.py",
    "tests/time_dependent_no/test_pcno_kolmogorov_refiner.py",
    "tests/time_dependent_no/test_fit_kolmogorov_refiner.py",
}


def presentation(model, inputs, targets, generator):
    levels = torch.randint(0, 4, (len(inputs),), generator=generator).to(inputs.device)
    noise = torch.randn(inputs.shape, generator=generator, dtype=inputs.dtype).to(inputs.device)
    clean = (targets-inputs)/model.train_scale
    candidate, velocity = model.schedule.presentation(clean, noise, levels)
    return candidate, velocity, levels


@torch.no_grad()
def update_ema(ema, model):
    for averaged, actual in zip(ema.parameters(), model.parameters(), strict=True):
        averaged.mul_(EMA_DECAY).add_(actual, alpha=1-EMA_DECAY)


def update(model, ema, optimizer, batches, generator, device):
    optimizer.zero_grad(set_to_none=True)
    stage_sse, stage_count = np.zeros(4), np.zeros(4, dtype=np.int64)
    for inputs, targets in batches:
        inputs, targets = inputs.to(device), targets.to(device)
        candidate, target, levels = presentation(model, inputs, targets, generator)
        per_example = (model(inputs, candidate, levels)-target).square().mean((-2, -1))
        loss = .5*per_example.mean()
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite Refiner loss")
        loss.backward()
        for level in range(4):
            selected = levels == level
            stage_sse[level] += float(per_example[selected].detach().sum())
            stage_count[level] += int(selected.sum())
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    norm = torch.stack([g.detach().double().square().sum() for g in gradients]).sum().sqrt()
    if not torch.isfinite(norm):
        raise RuntimeError("nonfinite Refiner gradient")
    optimizer.step()
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise RuntimeError("nonfinite Refiner parameter")
    update_ema(ema, model)
    return dict(loss=float(stage_sse.sum()/stage_count.sum()), level_sse=stage_sse.tolist(),
                level_count=stage_count.tolist(), gradient_l2=float(norm))


@torch.no_grad()
def acquisition_probe(model, train, device, *, resource=False):
    """Fixed clean training inputs, fresh held-fixed noise, no self-composition.

    Report v loss relative to a zero-v predictor, clean reconstruction error at
    every noise level, and individual four-call clean errors for three tapes.
    These are acquisition measurements, not a population generalization test.
    """
    started = time.perf_counter()
    paths = np.repeat(np.arange(len(train)), len(PROBE_STEPS))
    steps = np.tile(PROBE_STEPS, len(train))
    if resource:
        paths, steps = paths[:BATCH_SIZE], steps[:BATCH_SIZE]
    generator = torch.Generator().manual_seed(90119)
    sampler_generators = [torch.Generator().manual_seed(s) for s in SAMPLER_SEEDS]
    losses, zero_losses, reconstructed = np.zeros(4), np.zeros(4), np.zeros(4)
    deployed, denominators, persistence, rows = np.zeros(3), 0., 0., []
    model.eval()
    for start in range(0, len(paths), BATCH_SIZE):
        p, t = paths[start:start+BATCH_SIZE], steps[start:start+BATCH_SIZE]
        inputs = torch.from_numpy(np.array(train[p, t], copy=True)).to(device)
        targets = torch.from_numpy(np.array(train[p, t+1], copy=True)).to(device)
        clean = (targets-inputs)/model.train_scale
        denominators += float(targets.double().square().sum())
        persistence += float((targets-inputs).double().square().sum())
        for level in range(4):
            levels = torch.full((len(inputs),), level, dtype=torch.int64, device=device)
            noise = torch.randn(inputs.shape, generator=generator).to(device)
            sample, velocity_target = model.schedule.presentation(clean, noise, levels)
            pred = model(inputs, sample, levels)
            losses[level] += float((pred-velocity_target).double().square().sum())
            zero_losses[level] += float(velocity_target.double().square().sum())
            decoded = model.schedule.reconstruct(sample, pred, levels)
            reconstructed[level] += float((decoded-clean).double().square().sum())
        for realization, rng in enumerate(sampler_generators):
            prediction = model.transition(inputs, noise_tape(inputs, rng))["next_state"]
            errors = (prediction-targets).double().square().sum((-2, -1)).cpu().numpy()
            deployed[realization] += float(errors.sum())
            for j, (path, step) in enumerate(zip(p, t, strict=True)):
                rows.append(dict(path_index=int(path), input_step=int(step),
                                 sampler_seed=SAMPLER_SEEDS[realization], sse=float(errors[j]),
                                 target_sse=float(targets[j].double().square().sum())))
    count = len(paths)*train.shape[-1]**2
    return dict(role="train_acquisition_only", pairs=len(paths), input_steps=list(PROBE_STEPS),
                level_velocity_mse=(losses/count).tolist(),
                level_zero_velocity_mse=(zero_losses/count).tolist(),
                level_mse_over_zero=(losses/np.maximum(zero_losses, 1e-300)).tolist(),
                level_reconstructed_increment_rms_scaled=np.sqrt(reconstructed/count).tolist(),
                deployed_relative_l2=np.sqrt(deployed/denominators).tolist(),
                persistence_relative_l2=float(np.sqrt(persistence/denominators)),
                sampler_seeds=list(SAMPLER_SEEDS), per_pair=rows,
                recurrent_calls=0, seconds=time.perf_counter()-started)


def checkpoint(path, model, ema, optimizer, samplers, generator, identity, update):
    temporary = path.with_suffix(".tmp")
    torch.save(dict(identity=identity, update=update, model=model.state_dict(),
                    ema=ema.state_dict(), optimizer=optimizer.state_dict(),
                    samplers=[s.state_dict() for s in samplers],
                    noise_rng=generator.get_state(), torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []), temporary)
    temporary.replace(path)
    return base.sha256(path)


def _run(input_path, output, phase, device, result):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(17)
    manifest, captured, parent_fit, train, train_hash = base.load_inputs(input_path)
    sources = {name: base.sha256(base.ROOT / name) for name in sorted(SOURCE_PATHS)}
    parent = base.load_model(captured, parent_fit, device)
    replay = base.terminal_replay(parent, captured["teacher.npz"], device, len(base.TRAIN_SEEDS))
    model = PeriodicVorticityRefiner(**parent_fit["model_config"],
        train_scale=parent_fit["checkpoint_identity"]["train_scale_float64"]).to(device)
    model.initialize_from_parent(parent)
    with np.load(io.BytesIO(captured["teacher.npz"]), allow_pickle=False) as sentinels, torch.no_grad():
        inputs = torch.from_numpy(sentinels["sentinel_input"]).to(device)
        levels = torch.full((len(inputs),), 3, dtype=torch.int64, device=device)
        generated = inputs-model.train_scale*model(inputs, torch.zeros_like(inputs), levels)
        expected = parent(inputs)["raw_next"]
        transplant_error = float((generated-expected).double().square().mean().sqrt()/model.train_scale)
        if transplant_error > 1e-6:
            raise RuntimeError("parent-to-level3 transplant failed")
    del parent, captured, inputs, expected, generated
    model.train().requires_grad_(True)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 betas=(.9, .999), eps=1e-8, weight_decay=0)
    samplers = [base.EpochSampler(len(train)*(train.shape[1]-1), BATCH_SIZE, s) for s in (17, 1701)]
    generator = torch.Generator().manual_seed(1719)
    identity = dict(recipe=RECIPE, parent_checkpoint_sha256=manifest["artifacts"]["checkpoint.pt"],
        input_manifest_sha256=train_hash, model_config=parent_fit["model_config"],
        train_scale=float(model.train_scale), updates=FIT_UPDATES, batch_size=2*BATCH_SIZE,
        microbatch_size=BATCH_SIZE, pair_sampler_seeds=[17, 1701], noise_seed=1719,
        learning_rate=LEARNING_RATE, ema_decay=EMA_DECAY, betas=list(model.schedule.betas),
        alphas_cumprod=list(model.schedule.alphas_cumprod), input_channels=7,
        initialization="parent body; zero added lifting weights; negated final head",
        target="v prediction of normalized clean successor increment; clean current condition",
        noise="iid white Gaussian; uniform per-example level 0..3",
        deployment="EMA; reverse levels 3,2,1,0; final physical-state restriction only",
        selection="fixed acquisition terminal; no rollout-based selection")
    output.mkdir(parents=True, exist_ok=False)
    result.update(status="running", identity=identity, sources=sources, input_manifest=manifest,
                  parent_replay=replay, transplant_rms_scaled=transplant_error,
                  parameter_count=sum(p.numel() for p in model.parameters()),
                  additional_solver_labels=0, additional_clean_trajectories=0,
                  device=str(device), torch_version=str(torch.__version__),
                  precision="FP32; no AMP/TF32", max_fit_seconds=MAX_FIT_SECONDS,
                  probes=[], artifacts={})
    base.write_json(output / "result.json", result)
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started, durations = time.perf_counter(), []
    count = RESOURCE_UPDATES if phase == "resource" else FIT_UPDATES
    with (output / "updates.jsonl").open("x") as log:
        for step in range(1, count+1):
            tick = time.perf_counter()
            indices = [s.next_indices() for s in samplers]
            batches = []
            for selected in indices:
                p, t = np.divmod(selected, train.shape[1]-1)
                batches.append(tuple(torch.from_numpy(np.array(train[p, t+k], copy=True)) for k in (0, 1)))
            row = update(model, ema, optimizer, batches, generator, device)
            base._sync(device)
            durations.append(time.perf_counter()-tick)
            row.update(update=step, pair_indices=[i.tolist() for i in indices], seconds=durations[-1])
            log.write(json.dumps(row, allow_nan=False)+"\n")
            log.flush()
            result["updates_completed"] = step
            if step % 256 == 0:
                result["elapsed_seconds"] = time.perf_counter()-started
                base.write_json(output / "result.json", result)
            if step == RESOURCE_UPDATES:
                estimate = float(np.mean(durations[4:]))*FIT_UPDATES
                if phase == "fit" and estimate > MAX_FIT_SECONDS:
                    raise RuntimeError("resource estimate exceeds five-hour fit budget")
            if phase == "fit" and step % PROBE_INTERVAL == 0:
                result["artifacts"]["latest.pt"] = checkpoint(output/"latest.pt", model, ema,
                    optimizer, samplers, generator, identity, step)
                probe = acquisition_probe(ema, train, device)
                probe["update"] = step
                result["probes"].append(probe)
                base.write_json(output / "result.json", result)
            if phase == "fit" and time.perf_counter()-started > MAX_FIT_SECONDS:
                raise RuntimeError("five-hour acquisition budget exceeded")
    if phase == "resource":
        probe = acquisition_probe(ema, train, device, resource=True)
        probe["update"] = count
        result["probes"].append(probe)
    else:
        (output / "latest.pt").rename(output / "terminal.pt")
        result["artifacts"]["terminal.pt"] = result["artifacts"].pop("latest.pt")
    estimate = float(np.mean(durations[4:]))*FIT_UPDATES
    result.update(status="completed", elapsed_seconds=time.perf_counter()-started,
        warmed_seconds_per_update=float(np.mean(durations[4:])), estimated_fit_seconds=estimate,
        within_fit_budget=estimate <= MAX_FIT_SECONDS,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if torch.device(device).type == "cuda" else None,
        stopped_before_development_and_autonomous_evaluation=True,
        scientific_status="acquisition pilot; adequacy and method efficacy not established")
    result["artifacts"]["updates.jsonl"] = base.sha256(output / "updates.jsonl")
    if sources != {name: base.sha256(base.ROOT/name) for name in sorted(SOURCE_PATHS)}:
        raise RuntimeError("source changed during acquisition")
    if (base.sha256(input_path/"manifest.json") != train_hash or
            any(base.sha256(input_path/k) != v for k, v in manifest["artifacts"].items())):
        raise RuntimeError("training input changed during acquisition")
    base.write_json(output / "result.json", result)
    return result


def run(input_path, output_path, phase, device):
    input_path, output = Path(input_path).resolve(), Path(output_path).resolve()
    if phase not in ("resource", "fit"):
        raise ValueError("unknown phase")
    if output.is_relative_to(input_path) or input_path.is_relative_to(output):
        raise ValueError("output must be disjoint from immutable input")
    if output.exists():
        raise FileExistsError(output)
    result = dict(schema_version=1, status="initializing", phase=phase, updates_completed=0)
    try:
        return _run(input_path, output, phase, device, result)
    except Exception as error:
        if output.exists():
            result.update(status="failed", error=f"{type(error).__name__}: {error}")
            base.write_json(output / "result.json", result)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("resource", "fit"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args.input, args.output, args.phase, args.device)


if __name__ == "__main__":
    main()
