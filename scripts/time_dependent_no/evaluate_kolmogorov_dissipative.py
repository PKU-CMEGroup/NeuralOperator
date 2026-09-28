"""Evaluate the shell-trained map or fixed online-REC plus an RMS-ball cap.

Run with --training-input TRAIN --evaluation-input OPEN_DEV --fit-output FIT
--output NEW_DIRECTORY --arm shell|envelope --phase assay|rollout --device cuda.
The envelope arm loads the existing Gaussian-recovery fit without refitting.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path

import torch

from scripts.time_dependent_no import evaluate_kolmogorov_recovery as common
from scripts.time_dependent_no import fit_kolmogorov_dissipative as training
from utility.time_dependent_no.kolmogorov_dissipative import EnvelopeProjection, RMS_BOUND
from utility.time_dependent_no.pcno_kolmogorov import PeriodicVorticityPCNO

SOURCE_PATHS = common.SOURCE_PATHS | training.SOURCE_PATHS | {
    'scripts/time_dependent_no/evaluate_kolmogorov_dissipative.py',
}


def load_shell(captured, parent, manifest, train_hash, fit_output, device):
    path = Path(fit_output)
    raw = (path / 'result.json').read_bytes()
    result = json.loads(raw)
    identity = training.recipe_identity(manifest, parent, train_hash)
    if (result.get('status') != 'completed' or result.get('phase') != 'fit'
            or result.get('updates_completed') != training.FIT_UPDATES
            or result.get('identity') != identity or result.get('input_manifest') != manifest):
        raise ValueError('shell fit identity/terminal/scope mismatch')
    if set(result['sources']) != training.SOURCE_PATHS:
        raise ValueError('shell fit source closure mismatch')
    for name, digest in result['sources'].items():
        if common.sha256(common.ROOT / name) != digest:
            raise ValueError('shell fit source hash mismatch')
    checkpoint_bytes = (path / 'terminal.pt').read_bytes()
    digest = hashlib.sha256(checkpoint_bytes).hexdigest()
    if digest != result['artifacts']['terminal.pt']:
        raise ValueError('shell checkpoint hash mismatch')
    checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location='cpu', weights_only=True)
    if (checkpoint['identity'] != identity or checkpoint['update'] != training.FIT_UPDATES
            or checkpoint['schedule_position'] != training.FIT_UPDATES):
        raise ValueError('shell checkpoint terminal identity mismatch')
    model = PeriodicVorticityPCNO(**identity['model_config'], train_scale=identity['train_scale_float64'])
    model.load_state_dict(checkpoint['model'], strict=True)
    if float(model.train_scale) != identity['train_scale_model_float32']:
        raise ValueError('shell checkpoint scale mismatch')
    model.to(device).eval().requires_grad_(False)
    hashes = {'terminal.pt': digest, 'result.json': hashlib.sha256(raw).hexdigest()}
    return model, dict(identity=identity, **hashes), hashes


def run(training_input, evaluation_input, fit_output, output, arm, phase, device='cuda'):
    if arm not in ('shell', 'envelope') or phase not in ('assay', 'rollout'):
        raise ValueError('unknown representative or phase')
    loaded = []

    def loader(captured, parent, manifest, train_hash, fit_path, target_device):
        if arm == 'shell':
            return load_shell(captured, parent, manifest, train_hash, fit_path, target_device)
        model, identity, hashes = common.evaluation_model(
            captured, parent, manifest, train_hash, fit_path, target_device)
        if identity.get('identity', {}).get('arm') != 'recovery':
            raise ValueError('envelope comparison requires the fixed online recovery fit')
        model = EnvelopeProjection(model, record_calls=phase == 'rollout').eval()
        loaded.append(model)
        identity = dict(identity, deployed_correction=dict(
            kind='radial projection after canonicalization', rms_bound=RMS_BOUND,
            input_information='analytic PDE envelope; no online solver, reference, or fitted geometry'))
        return model, identity, hashes

    result = common.run(training_input, evaluation_input, output, phase, device, fit_output,
                        model_loader=loader, source_paths=SOURCE_PATHS)
    if arm == 'envelope' and phase == 'rollout':
        calls, offset = loaded[0].calls, 0
        for case in result['cases']:
            rows = calls[offset:offset + case['completed_steps']]
            offset += case['completed_steps']
            if len(rows) != case['completed_steps'] or any(len(r['factors']) != 1 for r in rows):
                raise RuntimeError('envelope call/rollout accounting mismatch')
            active = [i + 1 for i, row in enumerate(rows) if row['factors'][0] < 1]
            case['envelope'] = dict(first_activation=active[0] if active else None,
                                    active_steps=active, calls=rows)
        if offset != len(calls):
            raise RuntimeError('unaccounted envelope calls')
        common.write_json(Path(output) / 'result.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('training-input', 'evaluation-input', 'fit-output', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--arm', choices=('shell', 'envelope'), required=True)
    parser.add_argument('--phase', choices=('assay', 'rollout'), required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    run(args.training_input, args.evaluation_input, args.fit_output, args.output,
        args.arm, args.phase, args.device)
