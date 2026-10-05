"""Time-resolved clean and reached-state errors for the approved fixed-data pair.

Run ``python -m scripts.time_dependent_no.evaluate_kolmogorov_local_defects
--run-root EXISTING_FIXED_DATA_RUN --output NEW_DIRECTORY --device cuda``.
The default pilot uses the first original training and validation trajectories.
Solver labels diagnose frozen autonomous rollouts; they never enter prediction.
An independent clean probe is not available in the existing run.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.time_dependent_no import fit_kolmogorov_recovery as training
from scripts.time_dependent_no.evaluate_kolmogorov_recovery import (
    DEVELOPMENT_SEEDS, evaluation_model, load_evaluation, predict,
)
from scripts.time_dependent_no.evaluate_kolmogorov_tangent_forecast import (
    ROOT, sha256, write_json,
)
from utility.time_dependent_no.kolmogorov_reference import (
    KolmogorovReferenceConfig, KolmogorovReferenceStepper,
)

MODEL_NAMES = ("pretrained", "clean_continuation")
CONFIG = KolmogorovReferenceConfig(resolution=256, viscosity=0.01, macro_dt=0.05)
CHECK_STEPS = (0, 7, 15, 31)


def mse(value, scale):
    """Spatial mean square, normalized by the same frozen training scale."""
    value = np.asarray(value, dtype=np.float64)
    if not np.isfinite(value).all() or not np.isfinite(scale) or scale <= 0:
        raise ValueError("finite values and a positive fixed scale are required")
    return float(np.mean(value**2) / scale**2)


def error_terms(prediction, flow_target, clean_next, scale):
    """Exact additive decomposition, including the signed cross term."""
    prediction, flow_target, clean_next = [
        np.asarray(x, dtype=np.float64) for x in (prediction, flow_target, clean_next)
    ]
    defect, propagated = prediction - flow_target, flow_target - clean_next
    return dict(local_mse=mse(defect, scale), propagated_mse=mse(propagated, scale),
                cross_term=float(2 * np.mean(defect * propagated) / scale**2),
                next_error_mse=mse(prediction - clean_next, scale))


def rollout(model, truth, device):
    """Autonomous composition: the solver and future clean states are absent."""
    states = [np.array(truth[0], dtype=np.float32, copy=True)]
    for _ in range(len(truth) - 1):
        states.append(predict(model, states[-1][None], device)[1][0])
        if np.max(np.abs(states[-1])) > 1e6:
            raise FloatingPointError("rollout exceeded the recorded amplitude guard")
    return np.stack(states)


def score_paths(models, truth, reached, labels, scale, device):
    """Every map sees every input source and its identical numerical target."""
    horizon = truth.shape[0] - 1
    clean = np.empty((len(models), horizon))
    local = np.empty((len(models), len(reached), horizon))
    accumulated = np.empty((len(reached), horizon + 1))
    propagated = np.empty((len(reached), horizon))
    cross = np.empty_like(propagated)
    for m, model in enumerate(models):
        for n in range(horizon):
            inputs = np.stack([truth[n], *reached[:, n]])
            predictions = predict(model, inputs, device)[1].astype(np.float64)
            clean[m, n] = mse(predictions[0] - truth[n + 1], scale)
            for source in range(len(reached)):
                local[m, source, n] = mse(predictions[source + 1] - labels[source, n], scale)
    # Use the stored autonomous next state for the exact decomposition. Batched
    # common-input evaluation can differ slightly from single-state inference.
    for source in range(len(reached)):
        for n in range(horizon + 1):
            accumulated[source, n] = mse(
                reached[source, n].astype(np.float64) - truth[n].astype(np.float64), scale)
        for n in range(horizon):
            terms = error_terms(reached[source, n + 1], labels[source, n], truth[n + 1], scale)
            propagated[source, n] = terms["propagated_mse"]
            cross[source, n] = terms["cross_term"]
            # Own-rollout defect should use the actual deployed prediction.
            local[source, source, n] = terms["local_mse"]
    return clean, local, accumulated, propagated, cross


def solver_labels(reached, truth, config, scale):
    """Restart on reached states, recording projection and sampled refinement."""
    solver = KolmogorovReferenceStepper(config)
    refined = KolmogorovReferenceStepper(replace(
        config, dt_max=config.dt_max / 2, cfl=config.cfl / 2))
    horizon = truth.shape[0] - 1
    labels = np.empty((2, horizon, *truth.shape[-2:]), dtype=np.float64)
    rows = []
    checks = set(n for n in CHECK_STEPS if n < horizon) | {horizon - 1}
    for source in range(2):
        for n in range(horizon):
            if source == 1 and n == 0:
                labels[source, n] = labels[0, n]
                continue
            state = reached[source, n].astype(np.float64)
            canonical = solver.canonicalize(state)
            step = solver.advance_canonical(canonical)
            labels[source, n] = step.state
            row = dict(source=MODEL_NAMES[source], input_step=n,
                       projection_rms_scaled=np.sqrt(mse(canonical - state, scale)),
                       **asdict(step.diagnostics))
            if n in checks:
                fine_time = refined.advance_canonical(canonical).state
                row.update(temporal_rms_scaled=np.sqrt(mse(step.state - fine_time, scale)))
            rows.append(row)
            print(f"solver source={MODEL_NAMES[source]} input_step={n}", flush=True)
    clean_checks = []
    for n in sorted(checks):
        step = solver.advance_projected(truth[n])
        fine = refined.advance_projected(truth[n])
        clean_checks.append(dict(input_step=n,
                                 archived_next_rms_scaled=np.sqrt(mse(step.state - truth[n + 1], scale)),
                                 temporal_rms_scaled=np.sqrt(mse(step.state - fine.state, scale)),
                                 **asdict(step.diagnostics)))
    return labels, rows, clean_checks


def physical_diagnostics(states, config):
    solver = KolmogorovReferenceStepper(config)
    names = ("mean_vorticity", "kinetic_energy", "enstrophy", "palinstrophy")
    values = []
    for state in states:
        row = asdict(solver.diagnostics_canonical(solver.canonicalize(state)))
        # The raw mean also exposes float32 departures hidden by projection.
        row["mean_vorticity"] = float(np.mean(state, dtype=np.float64))
        values.append([row[k] for k in names])
    return np.asarray(values), names


def run(args, result):
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = Path(args.run_root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    result.update(schema_version=1, status="running", pilot=args.population == "pilot",
                  probe_status="missing: no independently initialized compatible clean checkpoint",
                  horizon=args.horizon, macro_dt=CONFIG.macro_dt,
                  case_selection="first original training and validation cases, before new diagnostics"
                  if args.population == "pilot" else "original eight training and four validation cases",
                  solver_config=asdict(CONFIG),
                  numerical=dict(torch_version=str(torch.__version__), numpy_version=np.__version__,
                                 device=args.device, tf32=False, model_dtype="float32", solver_dtype="float64"),
                  normalization="spatial mean-square error divided by fixed training scale squared",
                  local_time="input n predicts output n+1; plot at (n+1)*macro_dt",
                  qualification="sampled temporal refinement of the fixed-grid map; no uniform numerical bound",
                  reference_semantics="FP64 Fourier-Galerkin flow from the canonical projection of the unchanged FP32 model input")
    write_json(output / "result.json", result)
    manifest, captured, fit, train, train_hash = training.load_inputs(root / "input/train")
    _, validation, _, validation_hash = load_evaluation(root / "input/development", manifest)
    models, identities = [], []
    for fit_output in (None, root / "fit_clean_continuation"):
        model, identity, _ = evaluation_model(captured, fit, manifest, train_hash, fit_output, args.device)
        models.append(model)
        identities.append(identity)
    scale = float(models[0].train_scale)
    result.update(train_scale=scale, model_identities=identities,
                  training_manifest_sha256=train_hash, validation_manifest_sha256=validation_hash,
                  sources={name: sha256(ROOT / name) for name in (
                      "scripts/time_dependent_no/evaluate_kolmogorov_local_defects.py",
                      "utility/time_dependent_no/kolmogorov_reference.py")})
    cases = [("train", i, training.TRAIN_SEEDS[i], train) for i in range(1 if result["pilot"] else 8)]
    cases += [("validation", i, DEVELOPMENT_SEEDS[i], validation)
              for i in range(1 if result["pilot"] else 4)]
    clean, local, accumulated, propagated, cross, physics = [], [], [], [], [], []
    result["cases"] = []
    for population, index, seed, dataset in cases:
        truth = np.array(dataset[index, :args.horizon + 1], dtype=np.float32, copy=True)
        reached = np.stack([rollout(model, truth, args.device) for model in models])
        labels, qualification, clean_checks = solver_labels(reached, truth, CONFIG, scale)
        scores = score_paths(models, truth, reached, labels, scale, args.device)
        for destination, value in zip((clean, local, accumulated, propagated, cross), scores):
            destination.append(value)
        physical, physical_names = zip(*[physical_diagnostics(path, CONFIG) for path in (truth, *reached)])
        physics.append(np.stack(physical))
        case_id = f"{population}_{seed}"
        np.savez_compressed(output / f"{case_id}.npz", ground_truth=truth,
                            reached=reached, solver_next=labels)
        result["cases"].append(dict(case_id=case_id, population=population, seed=seed,
                                    qualification=qualification, clean_restarts=clean_checks))
        result["seconds"] = time.perf_counter() - started
        write_json(output / "result.json", result)
    arrays = dict(model_names=np.array(MODEL_NAMES),
                  case_ids=np.array([c["case_id"] for c in result["cases"]]),
                  populations=np.array([c["population"] for c in result["cases"]]),
                  input_steps=np.arange(args.horizon), clean_mse=np.stack(clean, axis=1),
                  local_mse=np.stack(local, axis=1), rollout_mse=np.stack(accumulated, axis=1),
                  propagated_mse=np.stack(propagated, axis=1), cross_term=np.stack(cross, axis=1),
                  physical_values=np.stack(physics, axis=1), physical_names=np.array(physical_names[0]))
    np.savez_compressed(output / "metrics.npz", **arrays)
    result.update(status="completed", seconds=time.perf_counter() - started,
                  metrics_sha256=sha256(output / "metrics.npz"))
    write_json(output / "result.json", result)
    print(json.dumps(dict(status=result["status"], cases=len(cases), seconds=result["seconds"])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--population", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--horizon", type=int, choices=range(1, 33), default=32)
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error("output must be a new directory")
    result = {}
    try:
        run(args, result)
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        if Path(args.output).is_dir():
            write_json(Path(args.output) / "result.json", result)
        raise


if __name__ == "__main__":
    main()
