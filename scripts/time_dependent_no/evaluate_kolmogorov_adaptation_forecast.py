"""H8 prediction around a saved bank-DYN path, without candidate self-composition.

Run with ``python -m scripts.time_dependent_no.evaluate_kolmogorov_adaptation_forecast``.
Inputs are the fixed training/open-development manifests, base fit and rollout,
and one frozen candidate fit. Signed nonlinear probes only diagnose validity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.time_dependent_no import evaluate_kolmogorov_recovery as evaluation
from scripts.time_dependent_no import fit_kolmogorov_response_preserving as fitting
from scripts.time_dependent_no.evaluate_kolmogorov_tangent_forecast import validate_jvp
from utility.time_dependent_no.forced_tangent import _jvp, finite_amplitude_response, forced_tangent_forecast

HORIZON = 8
SOURCE_PATHS = fitting.SOURCE_PATHS | {
    "scripts/time_dependent_no/evaluate_kolmogorov_adaptation_forecast.py",
    "tests/time_dependent_no/test_evaluate_kolmogorov_adaptation_forecast.py",
    "utility/time_dependent_no/forced_tangent.py",
}


def forecast_on_baseline(step, baseline, reference):
    """z_next = G(v)-v_next + DG(v)z; probes never update z.

    The utility's 'clean_forcing' is a map difference on the supplied DYN path
    here, not the clean-reference defect. Record those two quantities separately.
    q_next = DG(v)q + r_plus(v,z) is an error proxy, not a rigorous bound.
    """
    if baseline.shape != reference.shape or not torch.equal(baseline[0], reference[0]):
        raise ValueError("baseline/reference must share shape and initial state")
    forecast = forced_tangent_forecast(step, baseline)
    z, forcing = forecast["predicted_error"], forecast["clean_forcing"]
    q = [torch.zeros_like(baseline[0])]
    arrays = {k: [] for k in ("linear_response", "plus_remainder", "minus_remainder",
                              "clean_forcing", "ad_no_grad_offset")}
    for n in range(len(baseline)-1):
        _, propagated = _jvp(step, n, baseline[n], q[-1])
        if torch.any(z[n] != 0):
            probe = finite_amplitude_response(step, n, baseline[n], z[n], [1.])
            linear = probe["linear_response"][0]
            plus, minus = probe["plus_remainder"][0], probe["minus_remainder"][0]
        else:
            linear = plus = minus = torch.zeros_like(z[n])
        q.append((propagated + plus).detach())
        with torch.no_grad():
            clean = step(n, reference[n]) - reference[n+1]
            offset = step(n, baseline[n]) - (forcing[n] + baseline[n+1])
        for key, value in (("linear_response", linear), ("plus_remainder", plus),
                           ("minus_remainder", minus), ("clean_forcing", clean),
                           ("ad_no_grad_offset", offset)):
            arrays[key].append(value.detach())
    return dict(map_forcing=forcing, displacement=z,
                identity_displacement=forecast["identity_response_error"],
                curvature_estimate=torch.stack(q),
                **{key: torch.stack(values) for key, values in arrays.items()})


def case_metrics(baseline, reference, forecast, scale):
    squared = lambda x: float(x.detach().double().square().sum())
    predicted = baseline.double() + forecast["displacement"].double() - reference.double()
    linear = forecast["linear_response"].double().flatten(1).norm(dim=1)
    residual = torch.stack([forecast[key].double().flatten(1).norm(dim=1)
                            for key in ("plus_remainder", "minus_remainder")]).amax(dim=0)
    ratios = torch.where(linear > 0, residual / linear,
                         torch.where(residual == 0, 0., torch.inf))
    maximum = float(ratios.max())
    offset = forecast["ad_no_grad_offset"].double().flatten(1).square().mean(1).sqrt()
    return dict(predicted_sse=squared(predicted[1:]),
                base_sse=squared(baseline[1:].double()-reference[1:].double()),
                curvature_sse=squared(forecast["curvature_estimate"][1:]),
                target_sse=squared(reference[1:]),
                max_remainder_over_linear=maximum if np.isfinite(maximum) else None,
                max_ad_offset_scaled=float(offset.max())/scale)


def summarize(cases):
    totals = np.array([sum(row[k] for row in cases) for k in
                       ("predicted_sse", "base_sse", "curvature_sse", "target_sse")])
    if not np.isfinite(totals).all() or (totals < 0).any() or totals[3] <= 0:
        raise ValueError("finite reductions and positive reference energy required")
    predicted, base, curvature = np.sqrt(totals[:3]/totals[3])
    width = max(.1*predicted, 2*curvature)
    ratios = [row["max_remainder_over_linear"] for row in cases]
    maximum = max(ratios) if all(r is not None for r in ratios) else None
    qualified = (maximum is not None and maximum <= .1 and width <= .25*predicted
                 and max(row["max_ad_offset_scaled"] for row in cases) <= 1e-6)
    return dict(predicted_relative_l2=float(predicted), baseline_relative_l2=float(base),
                curvature_relative_l2=float(curvature),
                range_relative_l2=[max(0., float(predicted-width)), float(predicted+width)],
                max_remainder_over_linear=maximum, locally_qualified=bool(qualified),
                interpretation="empirical tolerance; neither confidence interval nor uniform bound")


def load_baseline(path, digest, reference):
    if evaluation.sha256(path) != digest:
        raise ValueError("baseline snapshot hash mismatch")
    with np.load(path, allow_pickle=False) as saved:
        selected = saved["step"] <= HORIZON
        if not np.array_equal(saved["step"][selected], np.arange(HORIZON+1)):
            raise ValueError("baseline requires all H8 steps")
        if not np.array_equal(saved["reference"][selected], reference):
            raise ValueError("baseline reference mismatch")
        baseline = saved["state"][selected].copy()
    if (baseline.dtype != np.float32 or baseline.shape != reference.shape
            or not np.isfinite(baseline).all() or not np.array_equal(baseline[0], reference[0])):
        raise ValueError("invalid baseline states or initial condition")
    return baseline


def run(args):
    output = args.output.resolve()
    for source in (args.training_input, args.evaluation_input, args.base_fit,
                   args.base_rollout, args.fit_output):
        source = source.resolve()
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("output must be separate from immutable inputs")
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    record = dict(status="incomplete", phase="adaptation_forecast", primary_horizon=HORIZON,
                  primary_metric="pooled relative L2 across steps 1 through 8",
                  started_unix=started, candidate_autonomous_steps=0, solver_calls=0,
                  protected_access=False, optimization_steps=0, cases=[], artifacts={})
    bindings = {}
    try:
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.device(args.device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        manifest, captured, parent, train, train_hash = evaluation.training.load_inputs(args.training_input)
        eval_manifest, development, _, eval_hash = evaluation.load_evaluation(args.evaluation_input, manifest)
        record.update(training_manifest_sha256=train_hash, evaluation_manifest_sha256=eval_hash,
                      sources={n: evaluation.sha256(evaluation.ROOT/n) for n in sorted(SOURCE_PATHS)})
        for root, data, digest in ((args.training_input, manifest, train_hash),
                                  (args.evaluation_input, eval_manifest, eval_hash)):
            bindings[root/"manifest.json"] = digest
            bindings.update({root/n: h for n, h in data["artifacts"].items()})
        base_result = json.loads((args.base_fit/"result.json").read_bytes())
        base_hash = evaluation.sha256(args.base_fit/"terminal.pt")
        if (base_result["status"] != "completed" or base_result["identity"]["arm"] != "dynamics"
                or base_result["identity"]["recipe"] != evaluation.targets.RECIPE
                or base_result["artifacts"]["terminal.pt"] != base_hash):
            raise ValueError("requires completed bank-DYN base fit")
        bindings.update({args.base_fit/n: evaluation.sha256(args.base_fit/n)
                         for n in ("result.json", "terminal.pt")})
        prior, bound = evaluation.bound_evaluation(args.base_rollout, "rollout", train_hash, eval_hash)
        bindings.update(bound)
        if evaluation.checkpoint_hash(prior["model"]) != base_hash:
            raise ValueError("base rollout checkpoint mismatch")
        model, identity, fit_hashes = evaluation.evaluation_model(
            captured, parent, manifest, train_hash, args.fit_output, device)
        bindings.update({args.fit_output/n: h for n, h in fit_hashes.items()})
        record.update(model=identity, baseline_model=prior["model"], train_scale=float(model.train_scale))
        scale = record["train_scale"]
        step = lambda n, x: model(x[None])["next_state"][0]
        record["jvp_check"] = validate_jvp(step, torch.from_numpy(np.array(train[0, 0], copy=True)).to(device), scale)
        if not record["jvp_check"]["passed"]:
            raise ArithmeticError("JVP finite-difference check failed")
        for role, values, seeds in (("train", train, evaluation.training.TRAIN_SEEDS[:8]),
                                   ("development", development, evaluation.DEVELOPMENT_SEEDS)):
            for index, seed in enumerate(seeds):
                name = f"rollout_{role}_{seed}.npz"
                path, digest = args.base_rollout/name, prior["artifacts"][name]
                reference = np.array(values[index, :HORIZON+1], copy=True)
                baseline = load_baseline(path, digest, reference)
                bindings[path] = digest
                baseline, reference = [torch.from_numpy(v).to(device) for v in (baseline, reference)]
                forecast = forecast_on_baseline(step, baseline, reference)
                record["cases"].append(dict(role=role, seed=seed, **case_metrics(baseline, reference, forecast, scale)))
                name = f"forecast_{role}_{seed}.npz"
                record["artifacts"][name] = evaluation.save_arrays(output/name,
                    dict(baseline=baseline, reference=reference, **forecast))
                evaluation.write_json(output/"result.json", record)
            record[role] = summarize([c for c in record["cases"] if c["role"] == role])
        if any(evaluation.sha256(path) != h for path, h in bindings.items()):
            raise RuntimeError("immutable input changed")
        if any(evaluation.sha256(evaluation.ROOT/n) != h for n, h in record["sources"].items()):
            raise RuntimeError("source changed")
        record.update(status="completed", finished_unix=time.time(),
                      prior_artifacts={str(p): h for p, h in bindings.items()})
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record["elapsed_seconds"] = time.time()-started
        if torch.cuda.is_available():
            record["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        evaluation.write_json(output/"result.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("training-input", "evaluation-input", "base-fit", "base-rollout", "fit-output", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
