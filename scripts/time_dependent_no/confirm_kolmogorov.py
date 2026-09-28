"""Evaluation-only confirmation with frozen models and pre-outcome forecasts.

Run ``python -m scripts.time_dependent_no.confirm_kolmogorov --scope scope.json
--output RUN --phase reference|baseline|clean|forecast|freeze|outcomes|reduce``.
The scope binds 64 new ID starts and nine completed fits; no optimizer is used.
Historical evaluation wrappers and their population validators are unchanged.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time
from types import ModuleType, SimpleNamespace

import numpy as np
import torch

from scripts.time_dependent_no import evaluate_kolmogorov_recovery as evaluation
from scripts.time_dependent_no import evaluate_kolmogorov_adaptation_forecast as forecast
from scripts.time_dependent_no.screen_kolmogorov_trajectory_readiness import _initial, _relative
from utility.time_dependent_no.kolmogorov_reference import (
    KolmogorovReferenceConfig, KolmogorovReferenceStepper, resize_dealiased_vorticity,
)

sha256, write_json, save_arrays = evaluation.sha256, evaluation.write_json, evaluation.save_arrays
ROOT = Path(__file__).resolve().parents[2]
CANDIDATES = tuple(f"seed{s}_{arm}" for s in (17, 18, 19)
                   for arm in ("clean_only", "preserve_response"))
MAPS = ("clean_continuation", "recovery", "dynamics", *CANDIDATES)
SOURCE_PATHS = forecast.SOURCE_PATHS | evaluation.SOURCE_PATHS | {
    "scripts/time_dependent_no/confirm_kolmogorov.py",
    "tests/time_dependent_no/test_confirm_kolmogorov.py",
    "tests/time_dependent_no/test_forced_tangent.py",
    "scripts/time_dependent_no/screen_kolmogorov_trajectory_readiness.py",
    "utility/time_dependent_no/kolmogorov_reference.py",
}


def read_json(path):
    return json.loads(Path(path).read_bytes())


def load_population(path, scope_digest, seeds, shape):
    record = read_json(path / "result.json")
    expected = dict(status="completed", role="confirmation", scope_sha256=scope_digest,
                    seeds=seeds, shape=list(shape), dtype="float64")
    if any(record.get(k) != v for k, v in expected.items()):
        raise ValueError("confirmation population scope/shape mismatch")
    if len(seeds) != len(set(seeds)) or set(record["artifacts"]) != {
            f"reference_{seed}.npy" for seed in seeds}:
        raise ValueError("confirmation population identities mismatch")
    for name, digest in record["artifacts"].items():
        if sha256(path / name) != digest:
            raise ValueError("confirmation population hash mismatch")
    return record


def reference_case(path, manifest, seed):
    if seed not in manifest["seeds"]:
        raise ValueError("seed outside confirmation scope")
    values = np.load(path / f"reference_{seed}.npy", mmap_mode="r", allow_pickle=False)
    if (values.shape != tuple(manifest["shape"]) or values.dtype != np.float64
            or not np.isfinite(values).all()):
        raise ValueError("invalid confirmation reference array")
    # Match the existing training/development evaluation precision, not a new law.
    return np.array(values, dtype=np.float32)


def generate_case(output, seed, config, steps):
    started = time.time()
    stepper = KolmogorovReferenceStepper(config)
    current = _initial(stepper, seed)
    values = np.empty((steps+1, config.resolution, config.resolution), dtype=np.float64)
    values[0] = current
    rows = [dict(step=0, structure=asdict(stepper.diagnostics_canonical(current)))]
    for n in range(1, steps+1):
        tick = time.perf_counter()
        result = stepper.advance_canonical(current)
        current = result.state
        values[n] = current
        rows.append(dict(step=n, call=asdict(result.diagnostics), seconds=time.perf_counter()-tick,
                         structure=asdict(stepper.diagnostics_canonical(current))))
    name = f"reference_{seed}.npy"
    with (output / name).open("xb") as stream:
        np.save(stream, values, allow_pickle=False)
    return dict(seed=seed, role="confirmation", file=name, sha256=sha256(output/name),
                steps=steps, diagnostics=rows, seconds=time.time()-started)


def paired_effect(clean, preserve, seeds):
    if len(clean) != len(preserve) or not clean or len(seeds) != len(set(seeds)):
        raise ValueError("invalid paired population")
    complete, ratios, pooled, wins = [], [], [], []
    for left, right in zip(clean, preserve, strict=True):
        indexed = [{row["seed"]: row for row in group} for group in (left, right)]
        if any(len(group) != len(seeds) or set(index) != set(seeds)
               for group, index in zip((left, right), indexed, strict=True)):
            raise ValueError("paired trajectory identities mismatch")
        pairs = [[next(h for h in index[s]["horizons"] if h["horizon"] == 32)
                  for index in indexed] for s in seeds]
        complete.append(sum(a["complete"] and b["complete"] for a, b in pairs))
        if complete[-1] != len(seeds) or any(a["relative_l2"] <= 0 or b["relative_l2"] <= 0
                                             for a, b in pairs):
            ratios.append(None); pooled.append(None); wins.append(None)
            continue
        ratios.append([b["relative_l2"] / a["relative_l2"] for a, b in pairs])
        errors = [np.sqrt(sum(p[k]["sse"] for p in pairs) /
                          sum(p[k]["target_sse"] for p in pairs)) for k in (0, 1)]
        pooled.append(float(errors[1]/errors[0]))
        wins.append(sum(value < 1 for value in ratios[-1]))
    result = dict(independent_trajectories=len(seeds), paired_fits=len(clean), complete_pairs=complete,
                  per_fit_pooled_ratios=pooled, per_fit_win_counts=wins,
                  geometric_mean_ratio=None, bootstrap_log_interval=None)
    if any(row is None for row in ratios):
        result["interpretation"] = "Unconditional log contrast unresolved: incomplete or zero-error pairs."
        return result
    per_path = np.log(np.asarray(ratios)).mean(axis=0)
    indices = np.random.default_rng(2026092317).integers(0, len(seeds), size=(10000, len(seeds)))
    result.update(mean_log_ratio=float(per_path.mean()), geometric_mean_ratio=float(np.exp(per_path.mean())),
                  per_trajectory_log_ratios=per_path.tolist(),
                  bootstrap_log_interval=np.quantile(per_path[indices].mean(axis=1), [.025, .975]).tolist(),
                  interpretation="Trajectory bootstrap conditional on the three fixed fits and their common parent.")
    return result


def saved_forecast_metrics(arrays, scale):
    a = {k: np.asarray(v, dtype=np.float64) for k, v in arrays.items()}
    squared = lambda x: float(np.square(x).sum())
    errors = [a["baseline"] + a[k] - a["reference"]
              for k in ("displacement", "identity_displacement")]
    linear = np.linalg.norm(a["linear_response"].reshape(len(a["linear_response"]), -1), axis=1)
    residual = np.maximum(*[np.linalg.norm(a[k].reshape(len(linear), -1), axis=1)
                            for k in ("plus_remainder", "minus_remainder")])
    ratios = np.divide(residual, linear, out=np.zeros_like(linear), where=linear > 0)
    ratios[(linear == 0) & (residual > 0)] = np.inf
    offset = np.sqrt(np.square(a["ad_no_grad_offset"]).reshape(len(linear), -1).mean(axis=1))
    maximum = float(ratios.max())
    return dict(predicted_sse=squared(errors[0][1:]), identity_sse=squared(errors[1][1:]),
                base_sse=squared((a["baseline"]-a["reference"])[1:]),
                curvature_sse=squared(a["curvature_estimate"][1:]),
                clean_sse=squared(a["clean_forcing"]), target_sse=squared(a["reference"][1:]),
                max_remainder_over_linear=maximum if np.isfinite(maximum) else None,
                max_ad_offset_scaled=float(offset.max())/scale)


def require_freeze(root, scope_digest):
    freeze = read_json(root / "prediction_freeze.json")
    if (freeze.get("status") != "completed" or freeze.get("scope_sha256") != scope_digest
            or freeze.get("candidate_outcomes_started") is not False or freeze["frozen_unix"] > time.time()):
        raise ValueError("forecast freeze scope/chronology mismatch")
    for name, digest in freeze["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path) != digest:
            raise ValueError("forecast freeze artifact hash mismatch")
    return freeze


def numerical_checks(root, seeds, config):
    """Fixed clean-state restarts; not full-trajectory continuum convergence."""
    output = []
    base = KolmogorovReferenceStepper(config)
    half = KolmogorovReferenceStepper(replace(config, dt_max=config.dt_max/2, cfl=config.cfl/2))
    fine = KolmogorovReferenceStepper(replace(config, resolution=2*config.resolution,
                                              dt_max=config.dt_max/2, cfl=config.cfl/2))
    for seed in seeds[:2]:
        values = np.load(root / f"reference_{seed}.npy", mmap_mode="r", allow_pickle=False)
        for n in (0, 16, 32):
            x = np.array(values[n], copy=True)
            coarse, middle = base.advance_canonical(x).state, half.advance_canonical(x).state
            high = fine.advance_canonical(resize_dealiased_vorticity(x, fine.config.resolution)).state
            restricted = resize_dealiased_vorticity(high, config.resolution)
            row = dict(seed=seed, input_step=n, base_replay_exact=bool(np.array_equal(coarse, values[n+1])),
                       temporal_relative_l2=_relative(middle, coarse),
                       spatial_relative_l2=_relative(coarse, restricted),
                       fine_discarded_relative_l2=_relative(
                           resize_dealiased_vorticity(restricted, fine.config.resolution), high))
            row["passed"] = (row["base_replay_exact"] and row["temporal_relative_l2"] <= 1e-5
                             and row["spatial_relative_l2"] <= 1e-3
                             and row["fine_discarded_relative_l2"] <= 1e-3)
            name = f"reference_check_{seed}_{n}.npz"
            row.update(file=name, sha256=save_arrays(root/name,
                dict(input=x, base=coarse, half=middle, fine=high)))
            output.append(row)
    return output


def load_seed17(parent, manifest, train_hash, path, historical, device):
    """Use the original fit validator; require byte-identical inference sources."""
    fitter = "scripts/time_dependent_no/fit_kolmogorov_response_preserving.py"
    fit_test = "tests/time_dependent_no/test_fit_kolmogorov_response_preserving.py"
    sources = read_json(path/"result.json")["sources"]
    for name, digest in sources.items():
        if sha256(historical/name) != digest:
            raise ValueError("historical fit source hash mismatch")
        if name not in (fitter, fit_test) and sha256(ROOT/name) != digest:
            raise ValueError("historical/current inference source mismatch")
    # The first pair predates --seed. Load its unchanged validator without
    # changing shared module globals or writing into the historical directory.
    legacy = ModuleType("confirmation_seed17_loader")
    legacy.__file__ = str(historical/fitter)
    exec(compile((historical/fitter).read_bytes(), legacy.__file__, "exec"), legacy.__dict__)
    legacy.base = SimpleNamespace(ROOT=historical, sha256=sha256)
    return legacy.load_fitted(parent, manifest, train_hash, path, device)


def model_for(scope, key, device, training_inputs):
    manifest, captured, parent, _, train_hash = training_inputs
    path = Path(scope["maps"][key]["path"])
    if key.startswith("seed17_"):
        model, identity, _ = load_seed17(parent, manifest, train_hash, path,
                                        Path(scope["seed17_source_root"]), device)
    else:
        model, identity, _ = evaluation.evaluation_model(captured, parent, manifest, train_hash, path, device)
    if evaluation.checkpoint_hash(identity) != scope["maps"][key]["checkpoint_sha256"]:
        raise ValueError("selected model/checkpoint mismatch")
    return model, identity


def rollout_model(scope, key, directory, population, device, training_inputs):
    directory.mkdir()
    model, identity = model_for(scope, key, device, training_inputs)
    record = dict(status="running", model=identity, cases=[], artifacts={}, started_unix=time.time())
    for seed in scope["seeds"]:
        reference = reference_case(directory.parent.parent/"reference", population, seed)
        case, arrays = evaluation.rollout_case(model, reference, float(model.train_scale), device)
        case.update(seed=seed, role="confirmation")
        name = f"rollout_{seed}.npz"
        record["artifacts"][name] = save_arrays(directory/name, arrays)
        record["cases"].append(case)
        write_json(directory/"result.json", record)
    record.update(status="completed", finished_unix=time.time())
    write_json(directory/"result.json", record)
    return record


def freeze_predictions(scope, digest, root):
    """Independent NumPy reduction of saved arrays before adapted outcomes."""
    if (root/"outcomes").exists():
        raise ValueError("cannot freeze after candidate outcomes start")
    record = dict(status="running", scope_sha256=digest, files={}, predictions={},
                  candidate_outcomes_started=False)
    for key in CANDIDATES:
        directory = root/"forecast"/key
        stored = read_json(directory/"result.json")
        record["files"][f"forecast/{key}/result.json"] = sha256(directory/"result.json")
        if [c["seed"] for c in stored["cases"]] != scope["seeds"] or stored["status"] != "completed":
            raise ValueError("incomplete or reordered forecast population")
        cases, failures = [], []
        for case in stored["cases"]:
            if case["status"] != "completed":
                failures.append(case)
                continue
            name = f"forecast_{case['seed']}.npz"
            path = directory/name
            if sha256(path) != stored["artifacts"][name]:
                raise ValueError("forecast array hash mismatch")
            record["files"][f"forecast/{key}/{name}"] = stored["artifacts"][name]
            with np.load(path, allow_pickle=False) as arrays:
                measured = saved_forecast_metrics(arrays, stored["train_scale"])
                replay = np.zeros_like(arrays["identity_displacement"])
                for n, forcing in enumerate(arrays["map_forcing"]):
                    replay[n+1] = replay[n] + forcing
                np.testing.assert_array_equal(replay, arrays["identity_displacement"])
            for metric in ("predicted_sse", "base_sse", "curvature_sse", "target_sse",
                           "max_remainder_over_linear", "max_ad_offset_scaled"):
                if measured[metric] is None or case[metric] is None:
                    if measured[metric] != case[metric]:
                        raise ValueError("nonfinite forecast accounting mismatch")
                else:
                    np.testing.assert_allclose(measured[metric], case[metric], rtol=1e-10, atol=1e-15)
            cases.append(dict(seed=case["seed"], **measured))
        if failures:
            predicted = dict(locally_qualified=False, predicted_relative_l2=None,
                             range_relative_l2=None, failed_cases=failures,
                             recommend_over_dynamics=False)
        else:
            predicted = forecast.summarize(cases)
            denominator = sum(c["target_sse"] for c in cases)
            predicted.update(identity_relative_l2=float(np.sqrt(sum(c["identity_sse"] for c in cases)/denominator)),
                             calibrated_clean_relative_l2=scope["clean_calibration_gain"] *
                                float(np.sqrt(sum(c["clean_sse"] for c in cases)/denominator)))
            predicted["recommend_over_dynamics"] = (predicted["locally_qualified"] and
                predicted["range_relative_l2"][1] < .95*predicted["baseline_relative_l2"])
        clean = read_json(root/"clean"/key/"result.json")
        base_clean = read_json(root/"clean/dynamics/result.json")
        predicted["clean_only_recommendation"] = clean["bands"][0]["relative_l2"] < base_clean["bands"][0]["relative_l2"]
        record["predictions"][key] = predicted
        record["files"][f"clean/{key}/result.json"] = sha256(root/"clean"/key/"result.json")
    for path in (root/"clean/dynamics/result.json", root/"baseline/dynamics/result.json", root/"reference/result.json"):
        record["files"][path.relative_to(root).as_posix()] = sha256(path)
    record.update(status="completed", frozen_unix=time.time())
    with (root/"prediction_freeze.json").open("x") as stream:
        stream.write(json.dumps(record, indent=2, allow_nan=False)+"\n")
    return record


def run(scope_path, root, phase, device="cuda"):
    scope, digest = read_json(scope_path), sha256(scope_path)
    seeds = scope["seeds"]
    if (scope["role"] != "confirmation" or len(seeds) != 64 or len(set(seeds)) != 64
            or set(seeds) & set(scope["excluded_seed_ids"]) or scope["steps"] != 128
            or set(scope["maps"]) != set(MAPS) or scope["historical_protected_access"] is not False):
        raise ValueError("requires the approved independent 64-path, nine-map confirmation")
    root = root.resolve()
    sources = scope["sources"]
    if set(sources) != SOURCE_PATHS or any(sha256(ROOT/n) != h for n, h in sources.items()):
        raise ValueError("confirmation source binding mismatch")
    for path, expected in scope["input_bindings"].items():
        if sha256(Path(path)) != expected:
            raise ValueError("immutable input hash mismatch")
    for entry in scope["maps"].values():
        path = Path(entry["path"]).resolve()
        if root.is_relative_to(path) or path.is_relative_to(root):
            raise ValueError("confirmation output overlaps a fitted input")
    root.mkdir(parents=True, exist_ok=True)
    status_path = root/(phase+".json")
    if status_path.exists():
        raise FileExistsError(status_path)
    record = dict(phase=phase, status="running", scope_sha256=digest, started_unix=time.time(),
                  optimization_steps=0, historical_protected_access=False)
    write_json(status_path, record)
    try:
        config = KolmogorovReferenceConfig(**scope["reference_config"])
        if phase == "reference":
            directory = root/"reference"
            directory.mkdir()
            population = dict(status="running", role="confirmation", scope_sha256=digest, seeds=seeds,
                              shape=[129, config.resolution, config.resolution], dtype="float64", cases=[], artifacts={})
            for seed in seeds:
                case = generate_case(directory, seed, config, 128)
                population["cases"].append(case)
                population["artifacts"][case["file"]] = case["sha256"]
                write_json(directory/"result.json", population)
                print(json.dumps(dict(phase=phase, completed_paths=len(population["cases"]), seed=seed)), flush=True)
            population["numerical_checks"] = numerical_checks(directory, seeds, config)
            if not all(row["passed"] for row in population["numerical_checks"]):
                raise ArithmeticError("sampled reference check failed; no population replacement")
            population["status"] = "completed"
            write_json(directory/"result.json", population)
        elif phase == "freeze":
            freeze_predictions(scope, digest, root)
        elif phase == "reduce":
            require_freeze(root, digest)
            groups = {k: read_json(root/"outcomes"/k/"result.json")["cases"] for k in CANDIDATES}
            result = paired_effect([groups[f"seed{s}_clean_only"] for s in (17,18,19)],
                                   [groups[f"seed{s}_preserve_response"] for s in (17,18,19)], seeds)
            write_json(root/"paired_h32.json", result)
        else:
            population = load_population(root/"reference", digest, seeds, (129, config.resolution, config.resolution))
            if phase == "outcomes":
                require_freeze(root, digest)
            torch.set_num_threads(2)
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
            inputs = evaluation.training.load_inputs(Path(scope["training_input"]))
            (root/phase).mkdir()
            if phase in ("baseline", "outcomes"):
                keys = ("dynamics",) if phase == "baseline" else tuple(k for k in MAPS if k != "dynamics")
                for key in keys:
                    rollout_model(scope, key, root/phase/key, population, device, inputs)
                    print(json.dumps(dict(phase=phase, map=key, status="completed")), flush=True)
            elif phase == "clean":
                for key in MAPS:
                    directory = root/phase/key
                    directory.mkdir()
                    model, identity = model_for(scope, key, device, inputs)
                    parts = []
                    for seed in seeds:
                        reference = reference_case(root/"reference", population, seed)
                        arrays, _ = evaluation.teacher_assay(model, {"confirmation": reference[None,:33]}, device)
                        parts.append(arrays)
                    arrays = {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}
                    bands = [dict(input_start=a, input_stop=b, paths=64,
                                  relative_l2=evaluation.relative(float(arrays["confirmation_sse"][:,a:b].sum()),
                                                                  float(arrays["confirmation_target_sse"][:,a:b].sum())))
                             for a,b in ((0,8),(8,16),(16,32))]
                    clean = dict(status="completed", model=identity, seeds=seeds, bands=bands,
                                 artifacts={"clean.npz": save_arrays(directory/"clean.npz", arrays)})
                    write_json(directory/"result.json", clean)
                    print(json.dumps(dict(phase=phase, map=key, status="completed")), flush=True)
                    del model
            elif phase == "forecast":
                base_directory = root/"baseline/dynamics"
                base = read_json(base_directory/"result.json")
                if base["status"] != "completed" or evaluation.checkpoint_hash(base["model"]) != scope["maps"]["dynamics"]["checkpoint_sha256"]:
                    raise ValueError("baseline completion/identity mismatch")
                for key in CANDIDATES:
                    directory = root/phase/key
                    directory.mkdir()
                    model, identity = model_for(scope, key, device, inputs)
                    scale = float(model.train_scale)
                    step = lambda n,x: model(x[None])["next_state"][0]
                    check = forecast.validate_jvp(step, torch.from_numpy(np.array(inputs[3][0,0], copy=True)).to(device), scale)
                    if not check["passed"]:
                        raise ArithmeticError("JVP numerical check failed")
                    result = dict(status="running", model=identity, train_scale=scale, jvp_check=check,
                                  candidate_autonomous_steps=0, cases=[], artifacts={})
                    for seed, base_case in zip(seeds, base["cases"], strict=True):
                        if base_case["seed"] != seed:
                            raise ValueError("baseline seed ordering mismatch")
                        reference = reference_case(root/"reference", population, seed)[:9]
                        if not base_case["horizons"][0]["complete"]:
                            result["cases"].append(dict(seed=seed, status="baseline_incomplete"))
                            continue
                        name = f"rollout_{seed}.npz"
                        baseline = forecast.load_baseline(base_directory/name, base["artifacts"][name], reference)
                        try:
                            b, u = [torch.from_numpy(x).to(device) for x in (baseline, reference)]
                            arrays = forecast.forecast_on_baseline(step, b, u)
                            metrics = forecast.case_metrics(b, u, arrays, scale)
                            name = f"forecast_{seed}.npz"
                            result["artifacts"][name] = save_arrays(directory/name, dict(baseline=b, reference=u, **arrays))
                            result["cases"].append(dict(seed=seed, status="completed", **metrics))
                        except (FloatingPointError, ArithmeticError) as error:
                            result["cases"].append(dict(seed=seed, status="forecast_failed", error=str(error)))
                        write_json(directory/"result.json", result)
                    result["status"] = "completed"
                    write_json(directory/"result.json", result)
                    print(json.dumps(dict(phase=phase, map=key, status="completed")), flush=True)
                    del model
            else:
                raise ValueError("unknown confirmation phase")
        if sha256(scope_path) != digest or any(sha256(ROOT/n) != h for n,h in sources.items()):
            raise RuntimeError("scope or source changed during phase")
        record.update(status="completed", finished_unix=time.time())
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record["elapsed_seconds"] = time.time()-record["started_unix"]
        write_json(status_path, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("reference", "baseline", "clean", "forecast", "freeze", "outcomes", "reduce"), required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args.scope, args.output, args.phase, args.device)


if __name__ == "__main__":
    main()
