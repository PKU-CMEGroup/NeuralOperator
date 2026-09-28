"""Matched clean finetuning and evaluation with capped PCNO channel matrices.

Run --input TRAIN --output NEW --phase resource|fit, or add --evaluation-input
OPEN_DEV --fit-output FIT and use --phase assay|rollout. Default device: cuda.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.time_dependent_no import fit_kolmogorov_recovery as base
from scripts.time_dependent_no import evaluate_kolmogorov_recovery as common
from utility.time_dependent_no.pcno_kolmogorov import PeriodicVorticityPCNO
from utility.time_dependent_no.pcno_kolmogorov_spectral_control import restrict_channel_maps

FIT_UPDATES = 4096
RECIPE = 'fixed32_pcno_channel_norm_cap_v1'
SOURCE_PATHS = base.SOURCE_PATHS | common.SOURCE_PATHS | {
    'utility/time_dependent_no/pcno_kolmogorov_spectral_control.py',
    'scripts/time_dependent_no/fit_kolmogorov_spectral_control.py',
    'tests/time_dependent_no/test_pcno_spectral_control.py',
}


def recipe_identity(manifest, parent, train_hash):
    return dict(recipe=RECIPE, arm='channel_norm_cap', updates=FIT_UPDATES,
                parent_checkpoint_sha256=manifest['artifacts']['checkpoint.pt'],
                input_manifest_sha256=train_hash, model_config=parent['model_config'],
                train_scale_float64=parent['checkpoint_identity']['train_scale_float64'],
                train_scale_model_float32=parent['checkpoint_identity']['train_scale_model_float32'],
                learning_rate=base.LEARNING_RATE, batch_size_per_branch=base.BATCH_SIZE,
                sampler_seeds=[17, 1701], branch_weights=[.5, .5], spectral_norm_cap=1.,
                targets='exact archived clean successors; no extra labels',
                limitation='channel matrix caps; no bound on the complete residual transition')


def fit(input_path, output_path, phase, device='cuda'):
    source, output = Path(input_path).resolve(), Path(output_path).resolve()
    if phase not in ('resource', 'fit') or output.is_relative_to(source):
        raise ValueError('invalid phase or output inside input')
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(17)
    manifest, captured, parent, train, train_hash = base.load_inputs(source)
    sources = {name: base.sha256(base.ROOT / name) for name in sorted(SOURCE_PATHS)}
    model = base.load_model(captured, parent, device)
    replay = base.terminal_replay(model, captured['teacher.npz'], device, len(base.TRAIN_SEEDS))
    model = restrict_channel_maps(model).train().requires_grad_(True)
    identity = recipe_identity(manifest, parent, train_hash)
    output.mkdir(parents=True)
    result = dict(status='running', phase=phase, identity=identity, input_manifest=manifest,
                  sources=sources, parent_replay=replay, artifacts={}, updates_completed=0,
                  device=str(device), torch_version=str(torch.__version__),
                  optimizer='fresh Adam; identical to clean continuation',
                  numerical=dict(dtype='float32/complex64 matrices', amp=False, tf32=False, cpu_threads=2))
    base.write_json(output / 'result.json', result)
    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=base.LEARNING_RATE,
                                     betas=(.9, .999), eps=1e-8, weight_decay=0)
        samplers = [base.EpochSampler(train.shape[0] * (train.shape[1] - 1), base.BATCH_SIZE, seed)
                    for seed in (17, 1701)]
        generator = torch.Generator().manual_seed(1702)
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
                row = base.update(model, optimizer, batches, device)
                base._sync(device)
                durations.append(time.perf_counter() - tick)
                row.update(update=step, seconds=durations[-1], first_indices=first.tolist(), second_indices=second.tolist())
                log.write(json.dumps(row, allow_nan=False) + '\n')
                log.flush()
                result['updates_completed'] = step
                if step % 256 == 0:
                    base.write_json(output / 'result.json', result)
                if time.perf_counter() - started > 8 * 3600:
                    raise RuntimeError('fixed fit exceeded eight-hour allowance')
        result.update(status='completed', training_seconds=time.perf_counter() - started,
                      warmed_seconds_per_update=float(np.mean(durations[4:] or durations)),
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


def load_fitted(captured, parent, manifest, train_hash, fit_output, device):
    path = Path(fit_output)
    raw = (path / 'result.json').read_bytes()
    result = json.loads(raw)
    identity = recipe_identity(manifest, parent, train_hash)
    if (result.get('status') != 'completed' or result.get('phase') != 'fit'
            or result.get('updates_completed') != FIT_UPDATES or result.get('identity') != identity
            or result.get('input_manifest') != manifest):
        raise ValueError('spectral fit identity/terminal mismatch')
    if result['sources'] != {name: base.sha256(base.ROOT / name) for name in sorted(SOURCE_PATHS)}:
        raise ValueError('spectral fit source mismatch')
    checkpoint_bytes = (path / 'terminal.pt').read_bytes()
    digest = hashlib.sha256(checkpoint_bytes).hexdigest()
    if digest != result['artifacts']['terminal.pt']:
        raise ValueError('spectral checkpoint hash mismatch')
    checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location='cpu', weights_only=True)
    if (checkpoint['identity'] != identity or checkpoint['update'] != FIT_UPDATES
            or checkpoint['schedule_position'] != FIT_UPDATES):
        raise ValueError('spectral checkpoint terminal mismatch')
    model = restrict_channel_maps(PeriodicVorticityPCNO(
        **identity['model_config'], train_scale=identity['train_scale_float64']))
    model.load_state_dict(checkpoint['model'], strict=True)
    if float(model.train_scale) != identity['train_scale_model_float32']:
        raise ValueError('spectral normalizer mismatch')
    model.to(device).eval().requires_grad_(False)
    hashes = {'result.json': hashlib.sha256(raw).hexdigest(), 'terminal.pt': digest}
    return model, dict(identity=identity, **hashes), hashes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('resource', 'fit', 'assay', 'rollout'), required=True)
    parser.add_argument('--evaluation-input', type=Path)
    parser.add_argument('--fit-output', type=Path)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.phase in ('resource', 'fit'):
        if args.evaluation_input is not None or args.fit_output is not None:
            parser.error('training accepts the training packet only')
        fit(args.input, args.output, args.phase, args.device)
    else:
        if args.evaluation_input is None or args.fit_output is None:
            parser.error('evaluation requires the open development packet and fitted checkpoint')
        common.run(args.input, args.evaluation_input, args.output, args.phase, args.device,
                   args.fit_output, model_loader=load_fitted, source_paths=SOURCE_PATHS)
