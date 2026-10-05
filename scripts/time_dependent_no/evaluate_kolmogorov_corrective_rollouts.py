"""Autonomous corrective rollouts and an independently trained clean probe.

Run with --run-root OLD_FIXED_DATA_RUN --comparison-root SIX_RUN_OUTPUT
--probe-fit CLEAN_PROBE_FIT --output NEW_DIRECTORY --device cuda [--watch].
The same first training/validation cases are followed for 32 steps. Watch waits
for fixed terminal checkpoints; it never evaluates intermediate weights.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.time_dependent_no import evaluate_kolmogorov_local_defects as diagnostic
from scripts.time_dependent_no import fit_kolmogorov_clean_probe as probe_training
from scripts.time_dependent_no import run_kolmogorov_displacement_comparison as comparison
from utility.time_dependent_no.pcno_kolmogorov import PeriodicVorticityPCNO

training = diagnostic.training
METHODS = ("pretrained", "clean_continuation") + tuple(
    c + "_" + t for c in comparison.CONDITIONS for t in comparison.TARGETS)
HORIZON = 32
CONFIG = diagnostic.CONFIG


def load_terminal(path, original, manifest, train_hash, device, probe=False, prepared_hash=None):
    result = json.loads((path / "result.json").read_text())
    identity = result["identity"]
    count = 49152 if probe else comparison.FIT_UPDATES
    if (result.get("status") != "completed" or result.get("updates_completed") != count
            or identity.get("input_manifest_sha256") != train_hash
            or identity.get("model_config") != original["model_config"]):
        raise ValueError("terminal model population/configuration/update mismatch")
    if probe:
        if (identity.get("run_id") != "CM_NEXT_KF_CLEAN_PROBE_20261001A"
                or identity.get("original_weights_loaded") is not False
                or identity.get("recipe", {}).get("initialization_seed") != 29
                or identity.get("train_array_sha256") != manifest["artifacts"]["train.npy"]):
            raise ValueError("not the independently initialized clean probe")
    elif (identity.get("protocol") != comparison.protocol()
          or identity.get("prepared_manifest_sha256") != prepared_hash
          or path.name != identity.get("condition", "") + "_" + identity.get("target", "")):
        raise ValueError("corrective target/input protocol mismatch")
    expected_sources = probe_training.sources() if probe else comparison.sources()
    if identity.get("sources") != expected_sources:
        raise ValueError("trained source identity mismatch")
    for name, digest in identity["sources"].items():
        if training.sha256(training.ROOT / name) != digest:
            raise ValueError(f"trained source mismatch: {name}")
    terminal = path / "terminal.pt"
    digest = training.sha256(terminal)
    if digest != result["artifacts"].get("terminal.pt"):
        raise ValueError("terminal checkpoint hash mismatch")
    saved = torch.load(terminal, map_location="cpu", weights_only=False)
    if saved["identity"] != identity or saved["update"] != count:
        raise ValueError("terminal checkpoint identity mismatch")
    model = PeriodicVorticityPCNO(**original["model_config"],
                                train_scale=original["checkpoint_identity"]["train_scale_float64"])
    model.load_state_dict(saved["model"], strict=True)
    if float(model.train_scale) != original["checkpoint_identity"]["train_scale_model_float32"]:
        raise ValueError("changed normalization scale")
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError("nonfinite terminal model")
    model.to(device).eval().requires_grad_(False)
    return model, dict(checkpoint_sha256=digest, identity=identity,
                       result_sha256=training.sha256(path / "result.json"))


def predict(model, state, device):
    return diagnostic.predict(model, np.asarray(state, dtype=np.float32)[None], device)[1][0]


def evaluate_case(model, probe, truth, scale, device, solver=None, fine=None):
    """Solver/probe observe stored autonomous states and never feed the rollout."""
    solver = solver or diagnostic.KolmogorovReferenceStepper(CONFIG)
    fine = fine or diagnostic.KolmogorovReferenceStepper(replace(
        CONFIG, dt_max=CONFIG.dt_max / 2, cfl=CONFIG.cfl / 2))
    reached = diagnostic.rollout(model, truth, device)
    h = len(truth) - 1
    values = {key: np.empty(h) for key in (
        "clean_mse", "own_defect_mse", "probe_defect_mse", "propagated_mse", "cross_term")}
    values["rollout_mse"] = np.array([diagnostic.mse(
        x.astype(np.float64) - u.astype(np.float64), scale) for x, u in zip(reached, truth)])
    labels = np.empty((h, *truth.shape[-2:]), dtype=np.float64)
    probe_next = np.empty((h, *truth.shape[-2:]), dtype=np.float32)
    checks = []
    for n in range(h):
        state = reached[n].astype(np.float64)
        canonical = solver.canonicalize(state)
        labels[n] = solver.advance_canonical(canonical).state
        probe_next[n] = predict(probe, reached[n], device)
        clean_prediction = predict(model, truth[n], device)
        values["clean_mse"][n] = diagnostic.mse(clean_prediction.astype(np.float64) - truth[n+1], scale)
        values["probe_defect_mse"][n] = diagnostic.mse(probe_next[n].astype(np.float64) - labels[n], scale)
        terms = diagnostic.error_terms(reached[n+1], labels[n], truth[n+1], scale)
        values["own_defect_mse"][n] = terms["local_mse"]
        for key in ("propagated_mse", "cross_term"):
            values[key][n] = terms[key]
        check = dict(input_step=n, projection_rms_scaled=diagnostic.mse(canonical - state, scale)**.5)
        if n in diagnostic.CHECK_STEPS or n == h - 1:
            check["temporal_rms_scaled"] = diagnostic.mse(fine.advance_canonical(canonical).state - labels[n], scale)**.5
        comparison.check_metrics(check)
        checks.append(check)
    residual = values["rollout_mse"][1:] - values["own_defect_mse"] - values["propagated_mse"] - values["cross_term"]
    if not np.allclose(residual, 0, rtol=0, atol=1e-10):
        raise ValueError("signed error decomposition does not close")
    physical, names = diagnostic.physical_diagnostics(reached, CONFIG)
    values["physical_values"] = physical
    fields = dict(reached=reached, solver_next=labels, probe_next=probe_next)
    return values, fields, dict(numerical_checks=checks, error_identity_max_residual=float(np.max(np.abs(residual)))), names


def await_terminal(comparison_root, name, watch):
    started = time.monotonic()
    while True:
        path = comparison_root / name / "result.json"
        try:
            result = json.loads(path.read_text()) if path.is_file() else {}
        except json.JSONDecodeError:
            result = {}
        if result.get("status") == "completed":
            return
        if result.get("status") == "failed":
            raise RuntimeError(f"training failed: {name}")
        queue = json.loads((comparison_root / "queue.json").read_text())
        if queue.get("status") == "failed":
            raise RuntimeError("training queue failed")
        if not watch:
            raise RuntimeError(f"fixed terminal not ready: {name}; use --watch to wait")
        if time.monotonic() - started > 8 * 3600:
            raise TimeoutError(f"terminal wait timed out: {name}")
        time.sleep(30)


def run(args):
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root, comparison_root, probe_fit, output = map(Path, (args.run_root, args.comparison_root, args.probe_fit, args.output))
    if any(output.resolve().is_relative_to(p.resolve()) for p in (root / "input", probe_fit, comparison_root)):
        raise ValueError("evaluation output must be separate from training artifacts")
    output.mkdir(parents=True, exist_ok=False)
    state = dict(status="initializing", methods=list(METHODS), completed_methods=[],
                 horizon=HORIZON, macro_dt=CONFIG.macro_dt, solver_config=asdict(CONFIG),
                 normalization="RMS divided by fixed clean-training scale; no percentage factor",
                 local_time="input step n; prediction and solver advance to n+1",
                 scope="same first training and first validation cases as Experiment 1",
                 probe_interpretation="independent clean predictor defect, not an exact manifold distance",
                 device=args.device, torch_version=str(torch.__version__), numpy_version=np.__version__,
                 sources={name: training.sha256(training.ROOT / name) for name in (
                     "scripts/time_dependent_no/evaluate_kolmogorov_corrective_rollouts.py",
                     "scripts/time_dependent_no/evaluate_kolmogorov_local_defects.py",
                     "utility/time_dependent_no/kolmogorov_reference.py")})
    comparison.write_json(output / "result.json", state)
    try:
        manifest, captured, original, train, train_hash = training.load_inputs(root / "input/train")
        _, validation, _, val_hash = diagnostic.load_evaluation(root / "input/development", manifest)
        prepared = json.loads((comparison_root / "displacements/result.json").read_text())
        if (prepared.get("status") != "completed" or not prepared.get("qualified")
                or prepared.get("protocol") != comparison.protocol()
                or prepared.get("input_manifest_sha256") != train_hash):
            raise ValueError("displaced training input receipt is not qualified")
        prepared_hash = training.sha256(comparison_root / "displacements/result.json")
        probe, probe_identity = load_terminal(probe_fit, original, manifest, train_hash, args.device, probe=True)
        scale = float(probe.train_scale)
        truths = np.stack([np.array(data[0, :HORIZON+1], copy=True) for data in (train, validation)])
        case_ids = [f"train_{training.TRAIN_SEEDS[0]}", f"validation_{diagnostic.DEVELOPMENT_SEEDS[0]}"]
        state.update(train_scale=scale, case_ids=case_ids, populations=["train", "validation"],
                     training_manifest_sha256=train_hash, validation_manifest_sha256=val_hash,
                     prepared_manifest_sha256=prepared_hash, probe_identity=probe_identity)
        solver = diagnostic.KolmogorovReferenceStepper(CONFIG)
        fine = diagnostic.KolmogorovReferenceStepper(replace(CONFIG, dt_max=CONFIG.dt_max/2, cfl=CONFIG.cfl/2))
        probe_clean, clean_checks, physical_truth = [], [], []
        for case_id, truth in zip(case_ids, truths):
            probe_clean.append([diagnostic.mse(predict(probe, truth[n], args.device).astype(np.float64) - truth[n+1], scale)
                                for n in range(HORIZON)])
            physical_truth.append(diagnostic.physical_diagnostics(truth, CONFIG)[0])
            for n in diagnostic.CHECK_STEPS:
                solved = solver.advance_projected(truth[n]).state
                check = dict(case_id=case_id, input_step=n,
                             clean_restart_rms_scaled=diagnostic.mse(solved-truth[n+1], scale)**.5,
                             temporal_rms_scaled=diagnostic.mse(solved-fine.advance_projected(truth[n]).state, scale)**.5)
                comparison.check_metrics(check)
                clean_checks.append(check)
        np.savez_compressed(output / "ground_truth.npz", ground_truth=truths,
                            probe_clean_mse=np.array(probe_clean), physical_truth=np.array(physical_truth))
        state.update(ground_truth_sha256=training.sha256(output / "ground_truth.npz"), clean_checks=clean_checks)
        comparison.write_json(output / "result.json", state)
        for name in METHODS:
            state.update(status="waiting_for_terminal" if name not in METHODS[:2] else "evaluating", current_method=name)
            comparison.write_json(output / "result.json", state)
            if name in METHODS[:2]:
                fitted = None if name == "pretrained" else root / "fit_clean_continuation"
                model, identity, _ = diagnostic.evaluation_model(captured, original, manifest, train_hash, fitted, args.device)
            else:
                await_terminal(comparison_root, name, args.watch)
                model, identity = load_terminal(comparison_root / name, original, manifest, train_hash,
                                                args.device, prepared_hash=prepared_hash)
            state.update(status="evaluating", current_method=name)
            comparison.write_json(output / "result.json", state)
            destination = output / name
            destination.mkdir()
            scores, qualifications, hashes = [], [], {}
            tick = time.perf_counter()
            for case_id, truth in zip(case_ids, truths):
                score, fields, qualification, physical_names = evaluate_case(model, probe, truth, scale, args.device, solver, fine)
                scores.append(score)
                qualifications.append(dict(case_id=case_id, **qualification))
                filename = case_id + ".npz"
                np.savez_compressed(destination / filename, **fields)
                hashes[filename] = training.sha256(destination / filename)
            np.savez_compressed(destination / "metrics.npz", **{key: np.stack([s[key] for s in scores]) for key in scores[0]})
            hashes["metrics.npz"] = training.sha256(destination / "metrics.npz")
            comparison.write_json(destination / "result.json", dict(status="completed", name=name, identity=identity,
                                  artifacts=hashes, qualifications=qualifications, physical_names=list(physical_names),
                                  seconds=time.perf_counter()-tick))
            state["completed_methods"].append(name)
            comparison.write_json(output / "result.json", state)
            print(json.dumps(dict(method=name, status="completed", seconds=time.perf_counter()-tick)), flush=True)
            del model
            gc.collect()
            if torch.device(args.device).type == "cuda":
                torch.cuda.empty_cache()
            from scripts.time_dependent_no.plot_kolmogorov_corrective_rollouts import plot
            plot(output, output / "figures", comparison_root)
        state.update(status="completed")
        state.pop("current_method", None)
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        comparison.write_json(output / "result.json", state)
    return state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("run-root", "comparison-root", "probe-fit", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--watch", action="store_true")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
