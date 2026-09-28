"""Evaluate the fixed ACDM terminal on the already-open Kolmogorov population.

Run python -m scripts.time_dependent_no.evaluate_kolmogorov_acdm
--training-input TRAIN --evaluation-input OPEN --fit-output FIT --output NEW
--phase assay|rollout --device cuda. No fitting or new solver labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.time_dependent_no import evaluate_kolmogorov_recovery as base
from scripts.time_dependent_no import fit_kolmogorov_acdm as fit
from utility.time_dependent_no.pcno_kolmogorov_acdm import PeriodicVorticityACDM, noise_tape

TEACHER_STEPS = (*range(8), 31, 255)
SAMPLER_SEEDS = fit.SAMPLER_SEEDS
SOURCE_PATHS = base.SOURCE_PATHS | fit.SOURCE_PATHS | {
    "scripts/time_dependent_no/evaluate_kolmogorov_acdm.py",
    "tests/time_dependent_no/test_evaluate_kolmogorov_acdm.py",
}


@torch.no_grad()
def transition(model, values, tape):
    try:
        result = model.transition(values, tape)
    except ValueError as error:
        # The frozen sampler detects an internal nonfinite diffusion candidate
        # through its joint-field guard. Inputs are validated before recurrence.
        if str(error) in ("joint field must be finite and match model grid/device/dtype",
                          "candidate must be a finite aligned field"):
            raise FloatingPointError("nonfinite internal diffusion candidate") from error
        raise
    if not all(bool(torch.isfinite(result[key]).all()) for key in ("raw_next", "next_state")):
        raise FloatingPointError("nonfinite generated successor")
    return result


class SampledTransition:
    """One random transition per physical step, for the existing rollout scorer."""

    def __init__(self, model, seed, noise_factory=noise_tape):
        self.model = model
        self.noise_factory = noise_factory
        self.generator = torch.Generator().manual_seed(seed)
        self.calls = 0

    def __call__(self, values):
        self.calls += 1
        return transition(self.model, values, self.noise_factory(values, self.generator))


def shared_predictions(model, values, seed, device, noise_factory=noise_tape):
    inputs = torch.from_numpy(np.array(values, dtype=np.float32, copy=True)).to(device)
    rng = torch.Generator().manual_seed(seed)
    one = noise_factory(inputs[:1], rng)
    tape = tuple(value.expand_as(inputs) for value in one)
    output = transition(model, inputs, tape)
    return tuple(output[key].cpu().numpy().copy() for key in ("raw_next", "next_state"))


def sample_moments(samples, target):
    samples, target = np.asarray(samples, dtype=np.float64), np.asarray(target, dtype=np.float64)
    mean = samples.mean(0)
    error = float(np.mean(np.sum((samples-target)**2, axis=(-2, -1))))
    bias = base.squared(mean-target)
    spread = float(np.mean(np.sum((samples-mean)**2, axis=(-2, -1))))
    return dict(mean_sample_sse=error, mean_prediction_sse=bias, sampling_sse=spread,
                closure_sse=error-bias-spread)


def spectrum(field):
    """Isotropic shells of vorticity enstrophy; sum equals half mean square."""
    field = np.asarray(field, dtype=np.float64)
    k = np.fft.fftfreq(len(field))*len(field)
    shell = np.rint(np.sqrt(k[:, None]**2+k[None, :]**2)).astype(int)
    power = .5*np.abs(np.fft.fft2(field)/field.size)**2
    return np.bincount(shell.ravel(), weights=power.ravel())


def phase_error_sse(prediction, target):
    """Grid translations in x and quarter-period y shifts preserving cos(4y)."""
    prediction, target = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    n = len(target)
    correlation = np.fft.ifft2(np.fft.fft2(prediction)*np.conj(np.fft.fft2(target))).real
    shifts = np.arange(0, n, n//4)
    x, index = np.unravel_index(np.argmax(correlation[:, shifts]), (n, len(shifts)))
    return base.squared(prediction-np.roll(target, (int(x), int(shifts[index])), axis=(0, 1)))


def teacher_queries(model, populations, device, noise_factory=noise_tape):
    arrays, rows = {}, []
    for role, truth in populations.items():
        paths, queries, n = len(truth), len(TEACHER_STEPS), truth.shape[-1]
        p = np.repeat(np.arange(paths), queries)
        t = np.tile(TEACHER_STEPS, paths)
        samples = np.empty((3, len(p), n, n), dtype=np.float32)
        raw_sse = np.empty((3, len(p)), dtype=np.float64)
        for realization, seed in enumerate(SAMPLER_SEEDS):
            model_step = SampledTransition(model, seed+(10000 if role == "development" else 0), noise_factory)
            for start in range(0, len(p), fit.BATCH_SIZE):
                ids = slice(start, start+fit.BATCH_SIZE)
                raw, nxt = base.predict(model_step, truth[p[ids], t[ids]], device)
                target = np.asarray(truth[p[ids], t[ids]+1], dtype=np.float64)
                samples[realization, ids] = nxt
                raw_sse[realization, ids] = np.sum((raw.astype(np.float64)-target)**2, axis=(-2, -1))
        arrays[role+"_samples"] = samples.reshape(3, paths, queries, n, n)
        arrays[role+"_targets"] = np.array(truth[p, t+1], copy=True).reshape(paths, queries, n, n)
        for j, (path, step) in enumerate(zip(p, t, strict=True)):
            target = np.asarray(truth[path, step+1], dtype=np.float64)
            rows.append(dict(role=role, path_index=int(path), input_step=int(step), query_index=j % queries,
                target_sse=base.squared(target), moments=sample_moments(samples[:, j], target),
                sampler_sse=np.sum((samples[:, j].astype(np.float64)-target)**2, axis=(-2, -1)).tolist(),
                sampler_raw_sse=raw_sse[:, j].tolist()))
    return arrays, rows


def response_queries(model, populations, donors, scale, device, noise_factory=noise_tape):
    gaussian_rng = torch.Generator().manual_seed(90017)
    inputs, outputs, rows = [], [], []
    for role, truth in populations.items():
        seeds = fit.base.TRAIN_SEEDS[:8] if role == "train" else base.DEVELOPMENT_SEEDS
        for path, seed in enumerate(seeds):
            for anchor, step in enumerate(base.INPUT_STEPS):
                center, target = np.array(truth[path, step:step+2], copy=True)
                gaussian = fit.base.gaussian_noise((1, *center.shape), scale, gaussian_rng)[0].numpy()
                for direction, delta in (("gaussian", gaussian), ("native_parent", donors[role+"_errors"][path, anchor])):
                    plus, minus = center+delta, center-delta
                    if direction == "native_parent" and role+"_states" in donors:
                        plus = donors[role+"_states"][path, anchor].copy()
                        if not np.array_equal(plus-center, delta):
                            raise ValueError("native donor input mismatch")
                    index = len(inputs)
                    inputs.append(np.stack((center, plus, minus, target)))
                    generated = []
                    for sampler_seed in SAMPLER_SEEDS:
                        tape_seed = 1000*seed+10*step+sampler_seed
                        raw, nxt = shared_predictions(model, np.stack((center, plus, minus)), tape_seed, device, noise_factory)
                        row = dict(role=role, seed=seed, input_step=step, direction=direction,
                            sampler_seed=sampler_seed, tape_seed=tape_seed, array_index=index,
                            **base.pair_metrics(nxt[0], nxt[1], nxt[2], target, plus, minus, center, scale))
                        row["raw"] = base.pair_metrics(raw[0], raw[1], raw[2], target, plus, minus, center, scale)
                        rows.append(row)
                        generated.append(nxt)
                    outputs.append(np.stack(generated))
    return dict(inputs=np.stack(inputs), predictions=np.stack(outputs)), rows


def load_model(directory, manifest, parent_fit, train_hash, device):
    directory = Path(directory)
    result = json.loads((directory/"result.json").read_text())
    identity = result["identity"]
    if (result["status"] != "completed" or result["updates_completed"] != fit.FIT_UPDATES
            or identity["recipe"] != fit.RECIPE or identity["input_manifest_sha256"] != train_hash
            or result["input_manifest"] != manifest or identity["model_config"] != parent_fit["model_config"]
            or identity["parent_checkpoint_sha256"] != manifest["artifacts"]["checkpoint.pt"]
            or set(result["sources"]) != fit.SOURCE_PATHS):
        raise ValueError("completed fixed acquisition identity mismatch")
    if any(base.sha256(base.ROOT/name) != digest for name, digest in result["sources"].items()):
        raise ValueError("acquisition source changed")
    if any(base.sha256(directory/name) != digest for name, digest in result["artifacts"].items()):
        raise ValueError("acquisition artifact changed")
    saved = torch.load(directory/"terminal.pt", map_location="cpu", weights_only=True)
    if saved["identity"] != identity or saved["update"] != fit.FIT_UPDATES:
        raise ValueError("terminal identity mismatch")
    model = PeriodicVorticityACDM(**identity["model_config"], train_scale=identity["train_scale"])
    model.load_state_dict(saved["model"], strict=True)
    if float(model.train_scale) != identity["train_scale"]:
        raise ValueError("normalizer mismatch")
    return model.to(device).eval().requires_grad_(False), result


def run(training_input, evaluation_input, fit_output, output, phase, device="cuda", *,
        model_loader=load_model, noise_factory=noise_tape, acquisition_probe=fit.acquisition_probe,
        source_paths=SOURCE_PATHS, calls_per_transition=20):
    """Shared stochastic-map assay; Refiner supplies its loader and four-noise tape."""
    training_input, evaluation_input, fit_output, output = map(lambda p: Path(p).resolve(),
        (training_input, evaluation_input, fit_output, output))
    if phase not in ("assay", "rollout") or any(output.is_relative_to(p) or p.is_relative_to(output)
            for p in (training_input, evaluation_input, fit_output)):
        raise ValueError("invalid phase or overlapping output")
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest, captured, parent_fit, train, train_hash = fit.base.load_inputs(training_input)
    evaluation, development, donors, eval_hash = base.load_evaluation(evaluation_input, manifest)
    del captured
    model, fitted = model_loader(fit_output, manifest, parent_fit, train_hash, device)
    sources = {name: base.sha256(base.ROOT/name) for name in sorted(source_paths)}
    bindings = {str(root/name): digest for root, record in ((training_input, manifest), (evaluation_input, evaluation),
                (fit_output, fitted)) for name, digest in record["artifacts"].items()}
    bindings.update({str(training_input/"manifest.json"): train_hash, str(evaluation_input/"manifest.json"): eval_hash,
        str(fit_output/"result.json"): base.sha256(fit_output/"result.json")})
    output.mkdir(parents=True, exist_ok=False)
    result = dict(status="running", phase=phase, protected_access=False, additional_solver_labels=0,
        training_manifest_sha256=train_hash, evaluation_manifest_sha256=eval_hash,
        checkpoint_sha256=fitted["artifacts"]["terminal.pt"], sources=sources, input_bindings=bindings,
        sampler_seeds=list(SAMPLER_SEEDS), calls_per_transition=calls_per_transition,
        deployment=fitted["identity"]["deployment"], train_scale=float(model.train_scale),
        precision="FP32; no AMP/TF32", artifacts={}, cases=[], torch_version=str(torch.__version__))
    started = time.perf_counter()
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    base.write_json(output/"result.json", result)
    try:
        populations = dict(train=train, development=development)
        if phase == "assay":
            replay = acquisition_probe(model, train, device, resource=True)
            expected = [r for r in fitted["probes"][-1]["per_pair"] if r["path_index"] < 2]
            if replay["per_pair"] != expected:
                raise ValueError("terminal generation replay mismatch")
            result["exact_generation_replay_pairs"] = 8
            arrays, rows = teacher_queries(model, populations, device, noise_factory)
            result["artifacts"]["teacher.npz"] = base.save_arrays(output/"teacher.npz", arrays)
            result.update(teacher_rows=rows, teacher_steps=list(TEACHER_STEPS), recurrent_calls=0)
            base.write_json(output/"result.json", result)
            del arrays
            arrays, rows = response_queries(model, populations, donors, float(model.train_scale), device, noise_factory)
            result["artifacts"]["responses.npz"] = base.save_arrays(output/"responses.npz", arrays)
            result.update(responses=rows, gaussian_seed=90017, response_input_order=["center", "positive", "negative", "target"],
                          response_output_order=["clean", "positive", "negative"])
        else:
            for role, truth in populations.items():
                seeds = fit.base.TRAIN_SEEDS[:8] if role == "train" else base.DEVELOPMENT_SEEDS
                for path, seed in enumerate(seeds):
                    for sampler_seed in SAMPLER_SEEDS:
                        tape_seed = 1000*seed+sampler_seed
                        deployed = SampledTransition(model, tape_seed, noise_factory)
                        tick = time.perf_counter()
                        case, snapshots = base.rollout_case(deployed, truth[path], float(model.train_scale), device)
                        case.update(role=role, seed=seed, sampler_seed=sampler_seed, tape_seed=tape_seed,
                                    transition_attempts=deployed.calls, seconds=time.perf_counter()-tick)
                        snapshots["spectrum"] = np.stack([spectrum(v) for v in snapshots["state"]])
                        snapshots["reference_spectrum"] = np.stack([spectrum(v) for v in snapshots["reference"]])
                        snapshots["phase_sse"] = np.array([phase_error_sse(a, b) for a, b in
                            zip(snapshots["state"], snapshots["reference"], strict=True)])
                        name = f"rollout_{role}_{seed}_sample_{sampler_seed}.npz"
                        result["artifacts"][name] = base.save_arrays(output/name, snapshots)
                        result["cases"].append(case)
                        result["pooled_horizons_by_sample"] = {str(s): base.pooled_horizons(
                            [c for c in result["cases"] if c["sampler_seed"] == s]) for s in SAMPLER_SEEDS}
                        base.write_json(output/"result.json", result)
            result["amplitude_guard_rms_over_train_scale"] = base.AMPLITUDE_LIMIT
        if sources != {name: base.sha256(base.ROOT/name) for name in sorted(source_paths)}:
            raise RuntimeError("evaluation source changed")
        if any(base.sha256(path) != digest for path, digest in bindings.items()):
            raise RuntimeError("evaluation input changed")
        result.update(status="completed", immutable_inputs_unchanged=True)
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        result.update(elapsed_seconds=time.perf_counter()-started,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if torch.device(device).type == "cuda" else None)
        base.write_json(output/"result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("training-input", "evaluation-input", "fit-output", "output"):
        parser.add_argument("--"+name, required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("assay", "rollout"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args.training_input, args.evaluation_input, args.fit_output, args.output, args.phase, args.device)


if __name__ == "__main__":
    main()
