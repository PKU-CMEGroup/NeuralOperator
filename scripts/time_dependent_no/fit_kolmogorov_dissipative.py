"""MNO-style shell loss on fixed clean data; no deployed postprocessing.

Run: python -m scripts.time_dependent_no.fit_kolmogorov_dissipative
--input TRAIN_PACKET --output NEW_DIRECTORY --phase resource|fit --device cuda.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.time_dependent_no import fit_kolmogorov_recovery as base
from utility.time_dependent_no.kolmogorov_dissipative import (
    CONTRACTION, SHELL_INNER, SHELL_OUTER, SHELL_WEIGHT, shell_error, shell_samples,
)

RECIPE = 'fixed32_mno_shell_v1'
FIT_UPDATES = 4096
SOURCE_PATHS = base.SOURCE_PATHS | {
    'utility/time_dependent_no/kolmogorov_dissipative.py',
    'scripts/time_dependent_no/fit_kolmogorov_dissipative.py',
    'tests/time_dependent_no/test_kolmogorov_dissipative.py',
}


def recipe_identity(manifest, parent, manifest_hash):
    return dict(
        recipe=RECIPE, arm='shell_prior', updates=FIT_UPDATES,
        parent_checkpoint_sha256=manifest['artifacts']['checkpoint.pt'],
        input_manifest_sha256=manifest_hash, model_config=parent['model_config'],
        train_scale_float64=parent['checkpoint_identity']['train_scale_float64'],
        train_scale_model_float32=parent['checkpoint_identity']['train_scale_model_float32'],
        learning_rate=base.LEARNING_RATE, batch_size_per_branch=base.BATCH_SIZE,
        clean_branch_weights=[0.5, 0.5], sampler_seeds=[17, 1701], shell_seed=1703,
        shell_rms_interval=[SHELL_INNER, SHELL_OUTER], contraction=CONTRACTION,
        shell_weight=SHELL_WEIGHT, shell_loss='mean per-sample error MSE / input mean square',
        shell_distribution='isotropic mean-zero bandlimited direction; uniform radius',
        targets='stored clean successors plus synthetic half-input shell targets; no solver labels',
    )


def update(model, optimizer, batches, shell, device):
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for inputs, targets in batches:
        prediction = model(inputs.to(device))['next_state']
        loss = ((prediction - targets.to(device)) / model.train_scale).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite clean loss')
        (0.5 * loss).backward()
        losses.append(float(loss.detach()))
    shell = shell.to(device)
    prior = shell_error(model(shell)['next_state'], shell)
    if not torch.isfinite(prior):
        raise RuntimeError('nonfinite shell loss')
    (SHELL_WEIGHT * prior).backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    norm = torch.stack([g.detach().double().square().sum() for g in gradients]).sum().sqrt()
    if not torch.isfinite(norm):
        raise RuntimeError('nonfinite gradient')
    optimizer.step()
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise RuntimeError('nonfinite parameter')
    return dict(first_mse_scaled=losses[0], second_mse_scaled=losses[1],
                shell_relative_mse=float(prior.detach()), gradient_l2=float(norm),
                loss=0.5 * sum(losses) + SHELL_WEIGHT * float(prior.detach()))


def shell_probe(model, resolution, device):
    generator = torch.Generator().manual_seed(90037)
    rows = []
    with torch.no_grad():
        for _ in range(8):
            x = shell_samples((8, resolution, resolution), generator).to(device)
            y = model(x)['next_state']
            if not torch.isfinite(y).all():
                raise RuntimeError('nonfinite shell acquisition probe')
            input_ms = x.double().square().mean((-2, -1))
            relative = (y.double() - CONTRACTION * x).square().mean((-2, -1)) / input_ms
            gains = (y.double().square().mean((-2, -1)) / input_ms).sqrt()
            for radius, error, gain in zip(input_ms.sqrt(), relative, gains):
                rows.append(dict(input_rms=float(radius), relative_mse=float(error), rms_ratio=float(gain)))
    return dict(seed=90037, samples=rows,
                mean_relative_mse=float(np.mean([r['relative_mse'] for r in rows])),
                mean_rms_ratio=float(np.mean([r['rms_ratio'] for r in rows])))


def run(input_path, output_path, phase, device='cuda'):
    source, output = Path(input_path).resolve(), Path(output_path).resolve()
    if phase not in ('resource', 'fit') or output.is_relative_to(source):
        raise ValueError('invalid phase or output inside input')
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(17)
    manifest, captured, parent, train, manifest_hash = base.load_inputs(source)
    sources = {name: base.sha256(base.ROOT / name) for name in sorted(SOURCE_PATHS)}
    model = base.load_model(captured, parent, device)
    replay = base.terminal_replay(model, captured['teacher.npz'], device, len(base.TRAIN_SEEDS))
    identity = recipe_identity(manifest, parent, manifest_hash)
    output.mkdir(parents=True)
    result = dict(status='running', phase=phase, identity=identity, input_manifest=manifest,
                  sources=sources, parent_replay=replay, artifacts={}, updates_completed=0,
                  device=str(device), torch_version=str(torch.__version__),
                  optimizer='fresh Adam; lr=1e-4, betas=(.9,.999), eps=1e-8, no weight decay',
                  numerical=dict(dtype='float32', amp=False, tf32=False, cpu_threads=2))
    base.write_json(output / 'result.json', result)
    try:
        result['shell_before'] = shell_probe(model, train.shape[-1], device)
        model.train().requires_grad_(True)
        optimizer = torch.optim.Adam(model.parameters(), lr=base.LEARNING_RATE,
                                     betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
        samplers = [base.EpochSampler(train.shape[0] * (train.shape[1] - 1), base.BATCH_SIZE, seed)
                    for seed in (17, 1701)]
        generator = torch.Generator().manual_seed(1703)
        count = base.RESOURCE_UPDATES if phase == 'resource' else FIT_UPDATES
        if torch.device(device).type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        base._sync(device)
        started, durations = time.perf_counter(), []
        with (output / 'updates.jsonl').open('x') as log:
            for step in range(1, count + 1):
                tick = time.perf_counter()
                first, second = [sampler.next_indices() for sampler in samplers]
                batches, _ = base.training_batches(train, first, second, 'clean_continuation',
                                                    float(model.train_scale), generator)
                shell = shell_samples(batches[0][0].shape, generator)
                row = update(model, optimizer, batches, shell, device)
                base._sync(device)
                durations.append(time.perf_counter() - tick)
                row.update(update=step, seconds=durations[-1], first_indices=first.tolist(),
                           second_indices=second.tolist())
                log.write(json.dumps(row, allow_nan=False) + '\n')
                log.flush()
                result['updates_completed'] = step
                if step % 256 == 0:
                    base.write_json(output / 'result.json', result)
                if time.perf_counter() - started > 4 * 3600:
                    raise RuntimeError('fixed fit exceeded four-hour allowance')
        training_seconds = time.perf_counter() - started
        model.eval().requires_grad_(False)
        result.update(status='completed', training_seconds=training_seconds,
                      warmed_seconds_per_update=float(np.mean(durations[4:] or durations)),
                      shell_after=shell_probe(model, train.shape[-1], device),
                      final_training_row=row,
                      peak_cuda_allocated_bytes=(torch.cuda.max_memory_allocated(device)
                                                 if torch.device(device).type == 'cuda' else None))
        if phase == 'fit':
            torch.save(dict(identity=identity, update=count, schedule_position=count,
                            model=model.state_dict()), output / 'terminal.pt')
            result['artifacts']['terminal.pt'] = base.sha256(output / 'terminal.pt')
        result['artifacts']['updates.jsonl'] = base.sha256(output / 'updates.jsonl')
        if sources != {name: base.sha256(base.ROOT / name) for name in sorted(SOURCE_PATHS)}:
            raise RuntimeError('training sources changed during execution')
    except Exception as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
        base.write_json(output / 'result.json', result)
        raise
    base.write_json(output / 'result.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('resource', 'fit'), required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    run(args.input, args.output, args.phase, args.device)
