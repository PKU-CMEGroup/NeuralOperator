"""Frozen, onset-matched REC/DYN inputs and numerically qualified solver labels.

Run ``python -m scripts.time_dependent_no.generate_kolmogorov_target_bank
--training-input TRAIN --output NEW_DIRECTORY --role train|probe --device cuda``.
The probe role additionally requires ``--evaluation-input OPEN_DEVELOPMENT``.
Training and unused probes have disjoint anchor times and Gaussian draws. Only
the unchanged parent supplies native directions; adapted models are never used.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.time_dependent_no import fit_kolmogorov_recovery as base
from utility.time_dependent_no.kolmogorov_reference import (
    KolmogorovReferenceConfig, KolmogorovReferenceStepper, resize_dealiased_vorticity,
)

RECIPE = "fixed32_onset_target_bank_v1"
TRAIN_STEPS, PROBE_STEPS = (4, 8, 12, 16), (6, 10, 14)
AMPLITUDES, DIRECTIONS, SIGNS = (.01, .1), ("gaussian", "native_parent"), (1, -1)
GAUSSIAN_SEEDS = {"train": 2026091801, "probe": 2026091802}
REFERENCE = KolmogorovReferenceConfig(resolution=256, viscosity=.01, macro_dt=.05)
MAX_SECONDS = 3 * 3600
ARRAY_DTYPES = {"centers": "float32", "clean_targets": "float32",
                "solver_clean": "float64", "inputs": "float32", "dynamics_targets": "float64"}
LIMITS = dict(projection_over_displacement=1e-4, rounding_over_displacement=1e-4,
              clean_restart_relative_l2=1e-3, clean_restart_over_displacement=1e-3,
              coarse_temporal_response_over_displacement=1e-3,
              fine_temporal_response_over_displacement=1e-3,
              spatial_response_over_displacement=1e-2,
              discarded_fine_response_over_displacement=1e-2,
              spatial_state_relative_l2=1e-3, discarded_fine_state_relative_l2=1e-3)
SOURCE_PATHS = base.SOURCE_PATHS | {
    "scripts/time_dependent_no/generate_kolmogorov_target_bank.py",
    "tests/time_dependent_no/test_generate_kolmogorov_target_bank.py",
    "utility/time_dependent_no/kolmogorov_reference.py",
}


def rms(values):
    return float(np.sqrt(np.mean(np.asarray(values, dtype=np.float64)**2)))


def input_rows(center, native, scale, generator, stepper):
    """Exact shared FP32 inputs; unit-RMS filtered directions, not a Gaussian law."""
    gaussian = stepper.canonicalize(generator.standard_normal(center.shape))
    for kind, direction in zip(DIRECTIONS, (gaussian, native)):
        direction = stepper.canonicalize(np.asarray(direction, dtype=np.float64))
        norm = rms(direction)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("undefined native/Gaussian direction")
        for amplitude in AMPLITUDES:
            for sign in SIGNS:
                intended = center.astype(np.float64) + sign * amplitude * scale * direction / norm
                x = intended.astype(np.float32)
                eta = rms(x.astype(np.float64) - center)
                yield dict(direction=kind, amplitude=amplitude, sign=sign,
                           input_rms_scaled=eta / scale,
                           rounding_over_displacement=rms(x - intended) / eta,
                           projection_over_displacement=rms(stepper.canonicalize(x) - x) / eta), x


def refinement_metrics(states, inputs):
    """Compare solver responses, preserving fine-grid content discarded at N."""
    n = inputs.shape[-1]
    a, b, c, d = (states[key] for key in "ABCD")
    down = lambda fields: np.stack([resize_dealiased_vorticity(x, n) for x in fields])
    c_low = down(c)
    c_discard = c - np.stack([resize_dealiased_vorticity(x, 2 * n) for x in c_low])
    rows = []
    for j in range(1, len(inputs)):
        eta = rms(inputs[j] - inputs[0])
        response = lambda values: values[j] - values[0]
        rows.append(dict(
            coarse_temporal_response_over_displacement=rms(response(b) - response(a)) / eta,
            fine_temporal_response_over_displacement=rms(response(d) - response(c)) / eta,
            spatial_response_over_displacement=rms(response(c_low) - response(b)) / eta,
            discarded_fine_response_over_displacement=rms(response(c_discard)) / eta,
            spatial_state_relative_l2=rms(c_low[j] - b[j]) / rms(c_low[j]),
            discarded_fine_state_relative_l2=rms(c_discard[j]) / rms(c[j]),
        ))
    return rows


def check_metrics(metrics):
    failed = {k: v for k, v in metrics.items()
              if k in LIMITS and (not np.isfinite(v) or v > LIMITS[k])}
    if failed:
        raise ValueError(f"solver/input qualification failed: {failed}")


def bank_layout(role):
    if role == "train":
        paths = [("train", i, s) for i, s in enumerate(base.TRAIN_SEEDS)]
        steps = TRAIN_STEPS
    elif role == "probe":
        # Import lazily: the evaluator also dispatches this experiment's fitter.
        from scripts.time_dependent_no import evaluate_kolmogorov_recovery as evaluation
        paths = [("train", i, s) for i, s in enumerate(base.TRAIN_SEEDS[:8])]
        paths += [("development", i, s) for i, s in enumerate(evaluation.DEVELOPMENT_SEEDS)]
        steps = PROBE_STEPS
    else:
        raise ValueError("unknown bank role")
    return [dict(role=r, path_index=i, seed=s, input_step=t)
            for r, i, s in paths for t in steps]


def source_hashes(role):
    paths = SOURCE_PATHS
    if role == "probe":
        from scripts.time_dependent_no import evaluate_kolmogorov_recovery as evaluation
        paths = paths | evaluation.SOURCE_PATHS
    return {p: base.sha256(base.ROOT / p) for p in sorted(paths)}


def load_bank(path, role, training_hash, parent_hash):
    """Authenticate role and every file before decoding arrays, including probes."""
    path = Path(path).resolve()
    raw = (path / "result.json").read_bytes()
    result = json.loads(raw)
    expected = dict(status="completed", recipe=RECIPE, role=role,
                    training_manifest_sha256=training_hash,
                    parent_checkpoint_sha256=parent_hash,
                    anchors=bank_layout(role), amplitudes=list(AMPLITUDES),
                    directions=list(DIRECTIONS), signs=list(SIGNS),
                    gaussian_seed=GAUSSIAN_SEEDS[role], reference_config=asdict(REFERENCE),
                    limits=LIMITS, qualified=True)
    if any(result.get(k) != v for k, v in expected.items()):
        raise ValueError("bank role/population/recipe/qualification mismatch")
    if result["sources"] != source_hashes(role):
        raise ValueError("bank source hash mismatch")
    required = {k + ".npy" for k in ARRAY_DTYPES}
    if not required <= set(result["artifacts"]):
        raise ValueError("bank missing target/input arrays")
    for name, digest in result["artifacts"].items():
        member = (path / name).resolve()
        if not member.is_relative_to(path) or base.sha256(member) != digest:
            raise ValueError("bank artifact hash mismatch")
    anchors = len(result["anchors"])
    rows = anchors * len(AMPLITUDES) * len(DIRECTIONS) * len(SIGNS)
    expected_rows = [dict(anchor=i, direction=d, amplitude=a, sign=s)
                     for i in range(anchors) for d in DIRECTIONS for a in AMPLITUDES for s in SIGNS]
    if len(result["rows"]) != rows or any(
            any(row.get(k) != v for k, v in expected.items())
            for row, expected in zip(result["rows"], expected_rows)):
        raise ValueError("bank row order/target mapping mismatch")
    arrays = {}
    for name, dtype in ARRAY_DTYPES.items():
        array = np.load(path / (name + ".npy"), mmap_mode="r", allow_pickle=False)
        count = rows if name in ("inputs", "dynamics_targets") else anchors
        if array.shape != (count, REFERENCE.resolution, REFERENCE.resolution) or array.dtype != dtype:
            raise ValueError("bank array shape/dtype mismatch")
        arrays[name] = array
    return result, arrays, base.sha256(path / "result.json")


def run(training_input, output_path, role, device, evaluation_input=None):
    if role not in GAUSSIAN_SEEDS or (role == "probe") != (evaluation_input is not None):
        raise ValueError("probe requires open development; train must not receive it")
    training_input, output = Path(training_input).resolve(), Path(output_path).resolve()
    inputs = [training_input] + ([Path(evaluation_input).resolve()] if evaluation_input else [])
    if any(output.is_relative_to(p) for p in inputs):
        raise ValueError("output must be outside immutable inputs")
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest, captured, parent_fit, train, train_hash = base.load_inputs(training_input)
    populations, eval_manifest, eval_hash = {"train": train}, None, None
    if role == "probe":
        from scripts.time_dependent_no import evaluate_kolmogorov_recovery as evaluation
        eval_manifest, dev, _, eval_hash = evaluation.load_evaluation(evaluation_input, manifest)
        populations["development"] = dev
    model = base.load_model(captured, parent_fit, device)
    replay = base.terminal_replay(model, captured["teacher.npz"], device, len(base.TRAIN_SEEDS))
    scale = float(model.train_scale)
    anchors = bank_layout(role)
    sources = source_hashes(role)
    output.mkdir(parents=True, exist_ok=False)
    result = dict(recipe=RECIPE, role=role, status="running", qualified=False,
                  training_manifest_sha256=train_hash, evaluation_manifest_sha256=eval_hash,
                  parent_checkpoint_sha256=manifest["artifacts"]["checkpoint.pt"],
                  reference_config=asdict(REFERENCE), train_scale=scale, parent_replay=replay,
                  anchors=anchors, amplitudes=list(AMPLITUDES), directions=list(DIRECTIONS),
                  signs=list(SIGNS), gaussian_seed=GAUSSIAN_SEEDS[role], limits=LIMITS,
                  sources=sources, rows=[], anchor_metrics=[], refinements=[], solver_calls=[],
                  artifacts={}, completed_anchors=0, max_seconds=MAX_SECONDS,
                  target_semantics="REC: archived FP32 successor; DYN: FP64 Phi(P(exact FP32 x)); labels cast to FP32 in fit",
                  information="fixed clean data; extra offline solver labels; probe never used for optimization")
    started = time.perf_counter()
    base.write_json(output / "result.json", result)
    try:
        n, per_anchor = REFERENCE.resolution, len(AMPLITUDES) * len(DIRECTIONS) * len(SIGNS)
        arrays = {k: np.lib.format.open_memmap(output / (k + ".npy"), mode="w+", dtype=dtype,
                   shape=(len(anchors) * (per_anchor if k in ("inputs", "dynamics_targets") else 1), n, n))
                  for k, dtype in ARRAY_DTYPES.items()}
        configs = dict(A=REFERENCE, B=replace(REFERENCE, dt_max=REFERENCE.dt_max / 2, cfl=REFERENCE.cfl / 2),
                       C=replace(REFERENCE, resolution=2*n, dt_max=REFERENCE.dt_max / 2, cfl=REFERENCE.cfl / 2),
                       D=replace(REFERENCE, resolution=2*n, dt_max=REFERENCE.dt_max / 4, cfl=REFERENCE.cfl / 4))
        steppers = {k: KolmogorovReferenceStepper(c) for k, c in configs.items()}
        rng, native_path, previous_path = np.random.default_rng(GAUSSIAN_SEEDS[role]), {}, None
        for i, anchor in enumerate(anchors):
            key = (anchor["role"], anchor["path_index"])
            reference = populations[key[0]][key[1]]
            if key != previous_path:
                state = np.array(reference[0], copy=True)
                native_path = {}
                with torch.no_grad():
                    for t in range(1, max(TRAIN_STEPS if role == "train" else PROBE_STEPS) + 1):
                        state = model(torch.from_numpy(state[None]).to(device))["next_state"][0].cpu().numpy().copy()
                        if not np.isfinite(state).all():
                            raise ValueError("nonfinite parent direction donor")
                        native_path[t] = state
                previous_path = key
            t = anchor["input_step"]
            center, target = np.array(reference[t:t+2], copy=True)
            pairs = list(input_rows(center, native_path[t].astype(np.float64) - center, scale, rng, steppers["A"]))
            physical = np.stack([center.astype(np.float64)] + [x.astype(np.float64) for _, x in pairs])
            canonical = np.stack([steppers["A"].canonicalize(x) for x in physical])
            sentinel = role == "train" and anchor["path_index"] < 2 and t in (TRAIN_STEPS[0], TRAIN_STEPS[-1])
            states = {}
            for name in ("ABCD" if sentinel else "A"):
                values = canonical if name in "AB" else np.stack([resize_dealiased_vorticity(x, 2*n) for x in canonical])
                outputs = []
                for j, x in enumerate(values):
                    if time.perf_counter() - started > MAX_SECONDS:
                        raise TimeoutError("target bank exceeded declared wall budget")
                    tick = time.perf_counter()
                    solved = steppers[name].advance_canonical(x)
                    outputs.append(solved.state)
                    result["solver_calls"].append(dict(anchor=i, member=j, variant=name,
                        seconds=time.perf_counter() - tick, **asdict(solved.diagnostics)))
                states[name] = np.stack(outputs)
            restart = rms(states["A"][0] - target)
            metrics = dict(anchor=i, clean_restart_relative_l2=restart / rms(target),
                           clean_restart_over_displacement=restart / min(rms(x - center) for x in physical[1:]))
            result["anchor_metrics"].append(metrics)
            check_metrics(metrics)
            arrays["centers"][i], arrays["clean_targets"][i] = center, target
            arrays["solver_clean"][i] = states["A"][0]
            for j, (row, x) in enumerate(pairs):
                idx = i * per_anchor + j
                y = states["A"][j+1]
                row.update(anchor=i, solver_response_rms_scaled=rms(y - states["A"][0]) / scale,
                           target_difference_rms_scaled=rms(y - target) / scale,
                           label_rounding_over_displacement=rms(y.astype(np.float32) - y) / rms(x.astype(np.float64) - center))
                check_metrics(row)
                if row["label_rounding_over_displacement"] > LIMITS["rounding_over_displacement"]:
                    raise ValueError("FP32 dynamics label rounding is unresolved")
                arrays["inputs"][idx], arrays["dynamics_targets"][idx] = x, y
                result["rows"].append(row)
            if sentinel:
                filename = f"refinement_{i:03d}.npz"
                np.savez(output / filename, physical_inputs=physical, canonical_inputs=canonical, **states)
                result["artifacts"][filename] = base.sha256(output / filename)
                refined = refinement_metrics(states, physical)
                result["refinements"].append(dict(anchor=i, rows=refined))
                for row in refined:
                    check_metrics(row)
            for array in arrays.values():
                array.flush()
            result.update(completed_anchors=i+1, elapsed_seconds=time.perf_counter() - started)
            base.write_json(output / "result.json", result)
        for name in ARRAY_DTYPES:
            result["artifacts"][name + ".npy"] = base.sha256(output / (name + ".npy"))
        for path, bound in ((training_input, manifest), (Path(evaluation_input) if evaluation_input else None, eval_manifest)):
            if path is not None and any(base.sha256(path / k) != v for k, v in bound["artifacts"].items()):
                raise RuntimeError("bank input changed during generation")
        if base.sha256(training_input / "manifest.json") != train_hash or (
                evaluation_input and base.sha256(Path(evaluation_input) / "manifest.json") != eval_hash):
            raise RuntimeError("bank input manifest changed during generation")
        if sources != source_hashes(role):
            raise RuntimeError("bank source changed during generation")
        result.update(status="completed", qualified=True, elapsed_seconds=time.perf_counter() - started,
                      qualification_scope="all input/restart/rounding checks; B/C/D refinement on first two train paths at first/last train anchor only")
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}",
                      elapsed_seconds=time.perf_counter() - started)
        base.write_json(output / "result.json", result)
        raise
    base.write_json(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-input", required=True, type=Path)
    parser.add_argument("--evaluation-input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--role", required=True, choices=("train", "probe"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args.training_input, args.output, args.role, args.device, args.evaluation_input)


if __name__ == "__main__":
    main()
