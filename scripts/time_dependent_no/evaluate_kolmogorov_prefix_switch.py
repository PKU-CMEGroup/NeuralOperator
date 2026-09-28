"""Fixed REC-prefix/DYN-suffix forecast and separately bound outcome evaluation.

Run with python -m scripts.time_dependent_no.evaluate_kolmogorov_prefix_switch.
Only the first eight fixed training paths are permitted. No fitting or solver.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.time_dependent_no import evaluate_kolmogorov_recovery as evaluation
from scripts.time_dependent_no.evaluate_kolmogorov_tangent_forecast import validate_jvp
from utility.time_dependent_no.intervention_sensitivity import initial_displacement_response

SWITCH_STEP, ENDPOINT = 8, 16
SOURCE_PATHS = evaluation.SOURCE_PATHS | {
    "scripts/time_dependent_no/evaluate_kolmogorov_prefix_switch.py",
    "scripts/time_dependent_no/evaluate_kolmogorov_tangent_forecast.py",
    "utility/time_dependent_no/intervention_sensitivity.py",
    "utility/time_dependent_no/forced_tangent.py",
    "tests/time_dependent_no/test_intervention_sensitivity.py",
    "tests/time_dependent_no/test_forced_tangent.py",
}


def forecast_decision(predicted_sse, base_sse, curvature_sse, target_sse, max_remainder):
    values = np.array([predicted_sse, base_sse, curvature_sse, target_sse, max_remainder])
    if not np.isfinite(values).all() or (values < 0).any() or target_sse <= 0:
        raise ValueError("forecast reductions must be finite with positive reference energy")
    predicted, baseline, curvature = np.sqrt(values[:3]/target_sse)
    width = max(.1*predicted, 2*curvature)
    return dict(predicted_relative_l2=float(predicted), baseline_relative_l2=float(baseline),
        curvature_relative_l2=float(curvature), range_relative_l2=[max(0., float(predicted-width)),
        float(predicted+width)], max_remainder_over_linear=float(max_remainder),
        selected=bool(max_remainder <= .1 and predicted+width <= .95*baseline),
        interpretation="empirical forecast tolerance, not a confidence interval or uniform bound")


def switch_rollout(first, later, reference, scale, device):
    calls, extra = 0, None

    def model(state):
        nonlocal calls, extra
        selected = first if calls < SWITCH_STEP else later
        calls += 1
        result = selected(state)
        if calls == ENDPOINT:
            extra = {"step": ENDPOINT, "state": result["next_state"][0].detach().cpu().numpy().copy(),
                     "raw": result["raw_next"][0].detach().cpu().numpy().copy(),
                     "reference": np.array(reference[ENDPOINT], copy=True)}
        return result

    result, arrays = evaluation.rollout_case(model, reference, scale, device)
    if extra is not None and ENDPOINT not in arrays["step"]:
        order = np.argsort(np.append(arrays["step"], ENDPOINT))
        arrays = {k: np.concatenate((v, np.asarray([extra[k]])))[order] for k, v in arrays.items()}
    return result, arrays


def run(args):
    output = args.output.resolve()
    for source in (args.training_input, args.base_fit, args.prefix_fit,
                   args.base_rollout, args.prefix_rollout):
        source = source.resolve()
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("output must be separate from immutable inputs")
    output.mkdir(parents=True, exist_ok=False)
    record = dict(status="incomplete", phase=args.phase, role="train_design",
        switch_step=SWITCH_STEP, primary_endpoint=ENDPOINT,
        seeds=evaluation.training.TRAIN_SEEDS[:8], protected_access=False,
        development_access=False, solver_calls=0, optimization_steps=0,
        candidate_autonomous_steps=0, artifacts={}, cases=[], probes=[])
    started, bindings = time.monotonic(), {}
    try:
        if args.phase == "outcome":
            if args.forecast is None or evaluation.sha256(args.forecast) != args.forecast_sha256:
                raise ValueError("outcome requires the frozen numerical forecast hash")
            forecast = json.loads(args.forecast.read_bytes())
            if (forecast.get("status") != "completed" or forecast.get("phase") != "forecast"
                    or forecast.get("switch_step") != SWITCH_STEP
                    or forecast.get("primary_endpoint") != ENDPOINT
                    or forecast.get("seeds") != record["seeds"]
                    or not forecast["forecast"]["selected"]):
                raise ValueError("forecast did not select this fixed candidate")
            bindings[args.forecast] = args.forecast_sha256
            record["forecast_sha256"] = args.forecast_sha256
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.device(args.device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        manifest, captured, parent, train, train_hash = evaluation.training.load_inputs(args.training_input)
        record["training_manifest_sha256"] = train_hash
        record["sources"] = {n: evaluation.sha256(evaluation.ROOT/n) for n in sorted(SOURCE_PATHS)}
        models, identities, priors = {}, {}, {}
        for label, fit, prior_dir, digest in (
                ("base", args.base_fit, args.base_rollout, args.base_sha256),
                ("prefix", args.prefix_fit, args.prefix_rollout, args.prefix_sha256)):
            model, identity, hashes = evaluation.evaluation_model(captured, parent, manifest, train_hash, fit, device)
            models[label], identities[label] = model, identity
            bindings.update({fit/n: h for n, h in hashes.items()})
            prior_path = prior_dir/"result.json"
            if evaluation.sha256(prior_path) != digest:
                raise ValueError("prior rollout hash mismatch")
            prior = json.loads(prior_path.read_bytes())
            if (prior["status"] != "completed" or prior["phase"] != "rollout"
                    or prior["training_manifest_sha256"] != train_hash
                    or evaluation.checkpoint_hash(prior["model"]) != evaluation.checkpoint_hash(identity)):
                raise ValueError("prior rollout model/population mismatch")
            priors[label], bindings[prior_path] = prior, digest
        if (identities["base"]["identity"]["recipe"] != evaluation.targets.RECIPE
                or identities["base"]["identity"]["arm"] != "dynamics"
                or identities["prefix"]["identity"]["recipe"] != "fixed32_gaussian_recovery_v1"
                or identities["prefix"]["identity"]["arm"] != "recovery"):
            raise ValueError("requires bank-DYN and online recovery endpoints")
        record["models"] = identities
        if args.phase == "outcome" and (record["models"] != forecast["models"]
                or train_hash != forecast["training_manifest_sha256"]
                or record["sources"] != forecast["sources"]):
            raise ValueError("outcome differs from frozen forecast inputs or implementation")
        scale = float(models["base"].train_scale)
        if scale != float(models["prefix"].train_scale):
            raise ValueError("model scales differ")
        record["train_scale"] = scale
        prefixes = []
        for index, seed in enumerate(record["seeds"]):
            name = f"rollout_train_{seed}.npz"
            path = args.prefix_rollout/name
            digest = priors["prefix"]["artifacts"][name]
            if evaluation.sha256(path) != digest:
                raise ValueError("prefix snapshots hash mismatch")
            bindings[path] = digest
            with np.load(path, allow_pickle=False) as saved:
                chosen = saved["step"] <= SWITCH_STEP
                if not np.array_equal(saved["step"][chosen], np.arange(SWITCH_STEP+1)):
                    raise ValueError("complete recovery prefix is required")
                if not np.array_equal(saved["reference"][chosen], train[index, :SWITCH_STEP+1]):
                    raise ValueError("prefix reference differs")
                prefixes.append(saved["state"][chosen].copy())
        if args.phase == "forecast":
            dense, replay, bound = evaluation.replay_reached_states(models["base"], dict(train=train),
                scale, device, priors["base"], args.base_rollout)
            record["baseline_replay"] = replay
            bindings.update(bound)
            step = lambda n, x: models["base"](x[None])["next_state"][0]
            state = torch.from_numpy(dense["train"][0, SWITCH_STEP]).to(device)
            record["jvp_check"] = validate_jvp(step, state, scale)
            if not record["jvp_check"]["passed"]:
                raise ArithmeticError("JVP numerical qualification failed")
            totals, max_ratio = np.zeros(4), 0.
            for index, seed in enumerate(record["seeds"]):
                baseline = torch.from_numpy(dense["train"][index, SWITCH_STEP:ENDPOINT+1]).to(device)
                reference = torch.from_numpy(np.array(train[index, SWITCH_STEP:ENDPOINT+1], copy=True)).to(device)
                prefix = torch.from_numpy(prefixes[index][-1]).to(device)
                response = initial_displacement_response(step, baseline, prefix-baseline[0], start_index=SWITCH_STEP)
                z, q = response["displacement"], response["curvature_estimate"]
                squared = lambda x: float(x.detach().double().square().sum())
                replay_rms = float(response["baseline_replay_residual"].double().square().mean((-2, -1)).sqrt().max())/scale
                reconstruction_rms = float((baseline[0]+z[0]-prefix).double().square().mean().sqrt())/scale
                if max(replay_rms, reconstruction_rms) > 1e-6:
                    raise ArithmeticError("baseline replay or prefix reconstruction exceeds tolerance")
                for k in range(ENDPOINT-SWITCH_STEP):
                    denominator = np.sqrt(squared(z[k+1]))
                    for sign, key in ((1, "plus_remainder"), (-1, "minus_remainder")):
                        residual = np.sqrt(squared(response[key][k]))
                        ratio = residual/denominator if denominator else (0. if residual == 0 else float("inf"))
                        if not np.isfinite(ratio):
                            raise ArithmeticError("nonzero remainder with zero linear response")
                        max_ratio = max(max_ratio, ratio)
                        record["probes"].append(dict(seed=seed, step=SWITCH_STEP+k, sign=sign,
                            remainder_over_linear=float(ratio)))
                values = [squared(baseline[-1].double()+z[-1].double()-reference[-1].double()),
                          squared(baseline[-1].double()-reference[-1].double()), squared(q[-1]), squared(reference[-1])]
                totals += values
                record["cases"].append(dict(seed=seed, predicted_sse=values[0], base_sse=values[1],
                    curvature_sse=values[2], target_sse=values[3], replay_rms_scaled=replay_rms,
                    reconstruction_rms_scaled=reconstruction_rms))
                name = f"train_{seed}.npz"
                record["artifacts"][name] = evaluation.save_arrays(output/name, dict(
                    baseline=baseline, reference=reference, prefix=prefix, **response))
            record["forecast"] = forecast_decision(*totals, max_ratio)
        else:
            for index, seed in enumerate(record["seeds"]):
                case, arrays = switch_rollout(models["prefix"], models["base"], train[index], scale, device)
                chosen = arrays["step"] <= SWITCH_STEP
                if not np.array_equal(arrays["state"][chosen], prefixes[index]):
                    raise ArithmeticError("hybrid prefix differs from frozen recovery")
                record["cases"].append(dict(role="train", seed=seed, **case))
                record["candidate_autonomous_steps"] += len(case["rows"])
                name = f"rollout_train_{seed}.npz"
                record["artifacts"][name] = evaluation.save_arrays(output/name, arrays)
            complete = all(c["completed_steps"] >= ENDPOINT and
                           (c["failed_at_step"] is None or c["failed_at_step"] > ENDPOINT) for c in record["cases"])
            observed = (float(np.sqrt(sum(c["rows"][ENDPOINT-1]["sse"] for c in record["cases"])/
                        sum(c["rows"][ENDPOINT-1]["target_sse"] for c in record["cases"]))) if complete else None)
            low, high = forecast["forecast"]["range_relative_l2"]
            record["primary_outcome"] = dict(complete=complete, relative_l2=observed,
                forecast_range=[low, high], forecast_passed=bool(complete and low <= observed <= high))
            record["pooled_horizons"] = evaluation.pooled_horizons(record["cases"])
        if any(evaluation.sha256(p) != h for p, h in bindings.items()):
            raise RuntimeError("immutable inputs changed")
        if any(evaluation.sha256(evaluation.ROOT/n) != h for n, h in record["sources"].items()):
            raise RuntimeError("source changed")
        record.update(status="completed", immutable_inputs_unchanged=True)
    except Exception as error:
        record["error"] = dict(type=type(error).__name__, message=str(error))
    finally:
        record["seconds"] = time.monotonic()-started
        record["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated() if torch.cuda.is_available() and args.device.startswith("cuda") else 0
        evaluation.write_json(output/"result.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("forecast", "outcome"), required=True)
    for name in ("training-input", "base-fit", "prefix-fit", "base-rollout", "prefix-rollout", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("base-sha256", "prefix-sha256"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--forecast", type=Path)
    parser.add_argument("--forecast-sha256")
    parser.add_argument("--device", default="cpu")
    result = run(parser.parse_args())
    print(json.dumps({k: result[k] for k in ("status", "phase", "forecast", "primary_outcome", "error") if k in result}))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
