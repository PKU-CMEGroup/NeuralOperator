"""Fit the fixed-data, one-state PCNO adaptation of conditional diffusion.

Run python -m scripts.time_dependent_no.fit_kolmogorov_acdm --input TRAIN
--output NEW_DIRECTORY --phase resource|fit --device cuda.
The fixed 8,192-update terminal and training probes are retained for subsequent
mechanism/rollout evaluation. Acquisition error is not a method-selection gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn.functional import smooth_l1_loss

from scripts.time_dependent_no import fit_kolmogorov_recovery as base
from utility.time_dependent_no.pcno_kolmogorov_acdm import PeriodicVorticityACDM, noise_tape

RECIPE = "fixed32_conditional_acdm_v1"
FIT_UPDATES, RESOURCE_UPDATES, BATCH_SIZE = 8192, 16, 8
PROBE_INTERVAL, PROBE_STEPS = 2048, (0, 7, 31, 255)
PROBE_LEVELS, SAMPLER_SEEDS = (0, 4, 9, 14, 19), (101, 211, 307)
LEARNING_RATE, MAX_FIT_SECONDS = 1e-4, 6*3600
SOURCE_PATHS = base.SOURCE_PATHS | {
    "utility/time_dependent_no/pcno_kolmogorov_acdm.py",
    "scripts/time_dependent_no/fit_kolmogorov_acdm.py",
    "tests/time_dependent_no/test_pcno_kolmogorov_acdm.py",
    "tests/time_dependent_no/test_fit_kolmogorov_acdm.py",
}


def presentation(model, inputs, targets, generator):
    clean = torch.stack((inputs, targets), 1)/model.train_scale
    levels = torch.randint(0, 20, (len(inputs),), generator=generator).to(inputs.device)
    noise = torch.randn(clean.shape, generator=generator, dtype=clean.dtype).to(inputs.device)
    return model.schedule.add_noise(clean, noise, levels), noise, levels


def update(model, optimizer, batches, generator, device):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    level_loss, level_count = np.zeros(20), np.zeros(20, dtype=np.int64)
    channel_sse, entries = np.zeros(2), 0
    for inputs, targets in batches:
        inputs, targets = inputs.to(device), targets.to(device)
        noised, noise, levels = presentation(model, inputs, targets, generator)
        predicted = model(noised, levels)
        per_example = smooth_l1_loss(predicted, noise, reduction="none").mean((1, 2, 3))
        loss = per_example.mean()/len(batches)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite ACDM loss")
        loss.backward()
        for level in range(20):
            chosen = levels == level
            level_loss[level] += float(per_example[chosen].detach().sum())
            level_count[level] += int(chosen.sum())
        channel_sse += (predicted-noise).detach().double().square().sum((0, 2, 3)).cpu().numpy()
        entries += len(inputs)*inputs.shape[-1]**2
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    norm = torch.stack([g.detach().double().square().sum() for g in gradients]).sum().sqrt()
    if not torch.isfinite(norm):
        raise RuntimeError("nonfinite ACDM gradient")
    optimizer.step()
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise RuntimeError("nonfinite ACDM parameter")
    return dict(loss=float(level_loss.sum()/level_count.sum()), level_loss_sum=level_loss.tolist(),
                level_count=level_count.tolist(), joint_noise_mse=(channel_sse/entries).tolist(),
                gradient_l2=float(norm))


@torch.no_grad()
def acquisition_probe(model, train, device, *, resource=False):
    """Clean training queries and fixed random tapes; no autonomous states."""
    started = time.perf_counter()
    paths = np.repeat(np.arange(len(train)), len(PROBE_STEPS))
    steps = np.tile(PROBE_STEPS, len(train))
    if resource:
        paths, steps = paths[:BATCH_SIZE], steps[:BATCH_SIZE]
    generator = torch.Generator().manual_seed(90120)
    samplers = [torch.Generator().manual_seed(s) for s in SAMPLER_SEEDS]
    losses, reconstructed = np.zeros((len(PROBE_LEVELS), 2)), np.zeros((len(PROBE_LEVELS), 2))
    deployed, denominators, persistence, rows = np.zeros(3), 0., 0., []
    model.eval()
    for start in range(0, len(paths), BATCH_SIZE):
        p, t = paths[start:start+BATCH_SIZE], steps[start:start+BATCH_SIZE]
        inputs = torch.from_numpy(np.array(train[p, t], copy=True)).to(device)
        targets = torch.from_numpy(np.array(train[p, t+1], copy=True)).to(device)
        clean = torch.stack((inputs, targets), 1)/model.train_scale
        denominators += float(targets.double().square().sum())
        persistence += float((targets-inputs).double().square().sum())
        for i, level in enumerate(PROBE_LEVELS):
            levels = torch.full((len(inputs),), level, dtype=torch.int64, device=device)
            noise = torch.randn(clean.shape, generator=generator, dtype=inputs.dtype).to(device)
            noised = model.schedule.add_noise(clean, noise, levels)
            predicted = model(noised, levels)
            losses[i] += (predicted-noise).double().square().sum((0, 2, 3)).cpu().numpy()
            decoded = model.schedule.reconstruct(noised, predicted, levels)
            reconstructed[i] += (decoded-clean).double().square().sum((0, 2, 3)).cpu().numpy()
        for realization, rng in enumerate(samplers):
            predicted = model.transition(inputs, noise_tape(inputs, rng))["next_state"]
            errors = (predicted-targets).double().square().sum((-2, -1)).cpu().numpy()
            deployed[realization] += float(errors.sum())
            for j, (path, step) in enumerate(zip(p, t, strict=True)):
                rows.append(dict(path_index=int(path), input_step=int(step),
                    sampler_seed=SAMPLER_SEEDS[realization], sse=float(errors[j]),
                    target_sse=float(targets[j].double().square().sum())))
    count = len(paths)*train.shape[-1]**2
    return dict(role="train_acquisition_only", pairs=len(paths), input_steps=list(PROBE_STEPS),
        probe_levels=list(PROBE_LEVELS), joint_noise_mse=(losses/count).tolist(),
        joint_reconstruction_rms_scaled=np.sqrt(reconstructed/count).tolist(),
        deployed_relative_l2=np.sqrt(deployed/denominators).tolist(),
        persistence_relative_l2=float(np.sqrt(persistence/denominators)),
        sampler_seeds=list(SAMPLER_SEEDS), per_pair=rows, recurrent_calls=0,
        calls_per_transition=20, seconds=time.perf_counter()-started)


def checkpoint(path, model, optimizer, samplers, generator, identity, update):
    temporary = path.with_suffix(".tmp")
    torch.save(dict(identity=identity, update=update, model=model.state_dict(),
        optimizer=optimizer.state_dict(), samplers=[s.state_dict() for s in samplers],
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
    sources = {n: base.sha256(base.ROOT/n) for n in sorted(SOURCE_PATHS)}
    parent = base.load_model(captured, parent_fit, device)
    replay = base.terminal_replay(parent, captured["teacher.npz"], device, len(base.TRAIN_SEEDS))
    model = PeriodicVorticityACDM(**parent_fit["model_config"],
        train_scale=parent_fit["checkpoint_identity"]["train_scale_float64"]).to(device)
    model.initialize_from_parent(parent)
    del parent, captured
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 betas=(.9, .999), eps=1e-8, weight_decay=0)
    samplers = [base.EpochSampler(len(train)*(train.shape[1]-1), BATCH_SIZE, s) for s in (17, 1701)]
    generator = torch.Generator().manual_seed(1720)
    identity = dict(recipe=RECIPE, parent_checkpoint_sha256=manifest["artifacts"]["checkpoint.pt"],
        input_manifest_sha256=train_hash, model_config=parent_fit["model_config"],
        train_scale=float(model.train_scale), updates=FIT_UPDATES, batch_size=2*BATCH_SIZE,
        microbatch_size=BATCH_SIZE, pair_sampler_seeds=[17, 1701], noise_seed=1720,
        learning_rate=LEARNING_RATE, ema=False, diffusion_steps=20,
        betas=list(model.schedule.betas), alphas_cumprod=list(model.schedule.alphas_cumprod),
        input_channels=23, output_channels=2, initialization="parent body; zero added lifting columns and joint head",
        target="independent Gaussian epsilon on normalized joint current and full successor",
        loss="mean Huber delta 1 on both fields", conditioning="one state; noised at matching diffusion level",
        deployment="20 DDPM calls; fixed condition noise per physical step; final physical restriction",
        upstream_commit="123e71b8d8f0f8cdb53b6bf29201b050c6446299",
        selection="fixed terminal; acquisition scores do not veto finite rollout evaluation")
    output.mkdir(parents=True, exist_ok=False)
    result.update(status="running", identity=identity, sources=sources, input_manifest=manifest,
        parent_replay=replay, parameter_count=sum(p.numel() for p in model.parameters()),
        additional_solver_labels=0, additional_clean_trajectories=0, device=str(device),
        torch_version=str(torch.__version__), precision="FP32; no AMP/TF32", probes=[], artifacts={})
    base.write_json(output/"result.json", result)
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started, durations = time.perf_counter(), []
    count = RESOURCE_UPDATES if phase == "resource" else FIT_UPDATES
    with (output/"updates.jsonl").open("x") as log:
        for step in range(1, count+1):
            tick = time.perf_counter()
            indices = [sampler.next_indices() for sampler in samplers]
            batches = []
            for chosen in indices:
                p, t = np.divmod(chosen, train.shape[1]-1)
                batches.append(tuple(torch.from_numpy(np.array(train[p, t+k], copy=True)) for k in (0, 1)))
            row = update(model, optimizer, batches, generator, device)
            base._sync(device)
            durations.append(time.perf_counter()-tick)
            row.update(update=step, pair_indices=[i.tolist() for i in indices], seconds=durations[-1])
            log.write(json.dumps(row, allow_nan=False)+"\n")
            log.flush()
            result["updates_completed"] = step
            if step % 256 == 0:
                result["elapsed_seconds"] = time.perf_counter()-started
                base.write_json(output/"result.json", result)
            if step == RESOURCE_UPDATES and phase == "fit":
                if float(np.mean(durations[4:]))*FIT_UPDATES > MAX_FIT_SECONDS:
                    raise RuntimeError("resource estimate exceeds six-hour fit budget")
            if phase == "fit" and step % PROBE_INTERVAL == 0:
                result["artifacts"]["latest.pt"] = checkpoint(output/"latest.pt", model,
                    optimizer, samplers, generator, identity, step)
                probe = acquisition_probe(model, train, device)
                probe["update"] = step
                result["probes"].append(probe)
                base.write_json(output/"result.json", result)
            if phase == "fit" and time.perf_counter()-started > MAX_FIT_SECONDS:
                raise RuntimeError("six-hour fit budget exceeded")
    if phase == "resource":
        probe = acquisition_probe(model, train, device, resource=True)
        probe["update"] = count
        result["probes"].append(probe)
    else:
        (output/"latest.pt").rename(output/"terminal.pt")
        result["artifacts"]["terminal.pt"] = result["artifacts"].pop("latest.pt")
    estimate = float(np.mean(durations[4:]))*FIT_UPDATES
    result.update(status="completed", elapsed_seconds=time.perf_counter()-started,
        warmed_seconds_per_update=float(np.mean(durations[4:])), estimated_fit_seconds=estimate,
        within_fit_budget=estimate <= MAX_FIT_SECONDS,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if torch.device(device).type == "cuda" else None,
        scientific_status="acquisition stage; mechanism and rollout measurements follow separately")
    result["artifacts"]["updates.jsonl"] = base.sha256(output/"updates.jsonl")
    if sources != {n: base.sha256(base.ROOT/n) for n in sorted(SOURCE_PATHS)}:
        raise RuntimeError("source changed during fitting")
    if (base.sha256(input_path/"manifest.json") != train_hash or
            any(base.sha256(input_path/n) != h for n, h in manifest["artifacts"].items())):
        raise RuntimeError("training input changed during fitting")
    result["immutable_inputs_unchanged"] = True
    base.write_json(output/"result.json", result)
    return result


def run(input_path, output_path, phase, device):
    input_path, output = Path(input_path).resolve(), Path(output_path).resolve()
    if phase not in ("resource", "fit"):
        raise ValueError("unknown phase")
    if output.is_relative_to(input_path) or input_path.is_relative_to(output):
        raise ValueError("output must be disjoint from immutable input")
    if output.exists():
        raise FileExistsError(output)
    result = dict(schema_version=1, status="initializing", phase=phase, updates_completed=0,
                  development_access=False, protected_access=False, autonomous_steps=0)
    try:
        return _run(input_path, output, phase, device, result)
    except Exception as error:
        if output.exists():
            result.update(status="failed", error=f"{type(error).__name__}: {error}")
            base.write_json(output/"result.json", result)
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
