"""Train-only first variation of bank-DYN toward online recovery; no blend rollout.

Run with python -m scripts.time_dependent_no.evaluate_kolmogorov_intervention.
The fixed calibration uses eight already-observed training paths, H16 as its
primary endpoint, H8/H32 as context, and sparse signed local probes. It does not
select or evaluate an autonomous candidate, nor certify a uniform error bound.
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
from utility.time_dependent_no.intervention_sensitivity import (
    blend_sensitivity, local_blend_remainder,
)

ANCHORS = (4, 8, 12, 16)
COEFFICIENTS = (0.01, 0.03, 0.1)
HORIZON = 32
PRIMARY_HORIZON = 16
REPLAY_LIMIT = 1e-6
SOURCE_PATHS = evaluation.SOURCE_PATHS | {
    "scripts/time_dependent_no/evaluate_kolmogorov_intervention.py",
    "utility/time_dependent_no/intervention_sensitivity.py",
    "tests/time_dependent_no/test_intervention_sensitivity.py",
    "tests/time_dependent_no/test_forced_tangent.py",
}


def loss_coefficients(error, sensitivity, reference):
    """Coefficients of ||error + alpha*sensitivity||^2, reduced in FP64."""
    e, z, u = [value.detach().double() for value in (error, sensitivity, reference)]
    return dict(a=float(e.square().sum()), b=float((e*z).sum()),
                c=float(z.square().sum()), target_sse=float(u.square().sum()))


def rms(value):
    return float(value.detach().double().square().mean().sqrt())


def run(training_input, base_fit, other_fit, prior_rollout, prior_sha256, output_path,
        device="cpu"):
    output = Path(output_path).resolve()
    for path in (training_input, base_fit, other_fit, prior_rollout):
        parent = Path(path).resolve()
        if output == parent or output.is_relative_to(parent) or parent.is_relative_to(output):
            raise ValueError("output must be separate from immutable inputs")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    device = torch.device(device)
    record = dict(status="incomplete", phase="inputs", role="train_calibration",
        recipe="dyn_online_recovery_first_variation_v1", primary_horizon=PRIMARY_HORIZON,
        horizon=HORIZON, seeds=evaluation.training.TRAIN_SEEDS[:8],
        local_probe_steps=list(ANCHORS), local_probe_coefficients=list(COEFFICIENTS),
        candidate_autonomous_steps=0, solver_calls=0, optimization_steps=0,
        protected_access=False, development_access=False, model_forward_calls={},
        interpretation="local parameter sensitivity on observed DYN paths; not a uniform bound",
        artifacts={}, curves=[], probes=[], clean_queries=[])
    bindings = {}
    try:
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        manifest, captured, parent_fit, train, train_hash = evaluation.training.load_inputs(training_input)
        record["training_manifest_sha256"] = train_hash
        record["sources"] = {name: evaluation.sha256(evaluation.ROOT/name) for name in sorted(SOURCE_PATHS)}
        models, identities = {}, {}
        for label, path in (("base", base_fit), ("other", other_fit)):
            model, identity, hashes = evaluation.evaluation_model(
                captured, parent_fit, manifest, train_hash, path, device)
            models[label], identities[label] = model, identity
            bindings.update({Path(path)/name: digest for name, digest in hashes.items()})
            record["model_forward_calls"][label] = 0

            def count_forward(_model, _arguments, name=label):
                record["model_forward_calls"][name] += 1

            model.register_forward_pre_hook(count_forward)
        if (identities["base"]["identity"]["recipe"] != evaluation.targets.RECIPE
                or identities["base"]["identity"]["arm"] != "dynamics"
                or identities["other"]["identity"]["recipe"] != "fixed32_gaussian_recovery_v1"
                or identities["other"]["identity"]["arm"] != "recovery"):
            raise ValueError("requires bank-DYN base and online recovery other map")
        record["models"] = identities
        prior_path = Path(prior_rollout)/"result.json"
        if evaluation.sha256(prior_path) != prior_sha256:
            raise ValueError("prior rollout receipt mismatch")
        prior = json.loads(prior_path.read_bytes())
        if (prior["status"] != "completed" or prior["phase"] != "rollout"
                or prior["training_manifest_sha256"] != train_hash
                or evaluation.checkpoint_hash(prior["model"]) != evaluation.checkpoint_hash(identities["base"])):
            raise ValueError("prior rollout model/input mismatch")
        bindings[prior_path] = prior_sha256
        record["prior_rollout_sha256"] = prior_sha256
        scale = float(models["base"].train_scale)
        if scale != float(models["other"].train_scale):
            raise ValueError("endpoint normalizers differ")
        record["train_scale_model_float32"] = scale
        record["phase"] = "baseline_replay"
        dense, replay_rows, replay_bindings = evaluation.replay_reached_states(
            models["base"], dict(train=train), scale, device, prior, prior_rollout)
        bindings.update(replay_bindings)
        record["baseline_replay"] = replay_rows
        base_step = lambda n, state: models["base"](state[None])["next_state"][0]
        other_step = lambda n, state: models["other"](state[None])["next_state"][0]
        record["phase"] = "jvp_check"
        first = torch.from_numpy(dense["train"][0, 0]).to(device)
        record["jvp_check"] = validate_jvp(base_step, first, scale)
        if not record["jvp_check"]["passed"]:
            raise ArithmeticError("PCNO JVP finite-difference check failed")
        record["phase"] = "sensitivity"
        for index, seed in enumerate(record["seeds"]):
            baseline = torch.from_numpy(dense["train"][index]).to(device)
            reference = torch.from_numpy(np.array(train[index, :HORIZON+1], copy=True)).to(device)
            response = blend_sensitivity(base_step, other_step, baseline)
            replay_rms = max(rms(value)/scale for value in response["baseline_replay_residual"])
            if replay_rms > REPLAY_LIMIT:
                raise ArithmeticError("JVP forward and archived baseline recurrence disagree")
            z = response["sensitivity"]
            for n in range(1, HORIZON+1):
                error = baseline[n].double()-reference[n].double()
                record["curves"].append(dict(seed=seed, step=n,
                    **loss_coefficients(error, z[n], reference[n]),
                    identity_response=loss_coefficients(
                        error, response["identity_response_sensitivity"][n], reference[n]),
                    replay_rms_scaled=replay_rms))
            for n in ANCHORS:
                probe = local_blend_remainder(base_step, other_step, n, baseline[n],
                                              z[n], z[n+1], COEFFICIENTS)
                for k, coefficient in enumerate(COEFFICIENTS):
                    linear_rms = coefficient*rms(z[n+1])
                    for sign_index, sign in enumerate((1, -1)):
                        remainder = rms(probe["remainder"][k, sign_index])
                        record["probes"].append(dict(seed=seed, step=n,
                            coefficient=coefficient, sign=sign,
                            displacement_rms_scaled=rms(probe["realized_displacement"][k, sign_index])/scale,
                            linear_rms_scaled=linear_rms/scale,
                            remainder_rms_scaled=remainder/scale,
                            remainder_over_linear=remainder/linear_rms if linear_rms else None))
            with torch.no_grad():
                for n in range(8):
                    first, second = base_step(n, reference[n]), other_step(n, reference[n])
                    record["clean_queries"].append(dict(seed=seed, input_step=n,
                        **loss_coefficients(first.double()-reference[n+1].double(),
                                            second.double()-first.double(), reference[n+1])))
            name = f"train_{seed}.npz"
            record["artifacts"][name] = evaluation.save_arrays(output/name, dict(
                baseline=baseline, reference=reference, **response))
            evaluation.write_json(output/"progress.json", dict(completed_paths=index+1,
                total_paths=len(record["seeds"]), elapsed_seconds=time.monotonic()-started))
        if any(evaluation.sha256(path) != digest for path, digest in bindings.items()):
            raise RuntimeError("immutable endpoint or prior artifact changed")
        if any(evaluation.sha256(evaluation.ROOT/name) != digest for name, digest in record["sources"].items()):
            raise RuntimeError("diagnostic source changed during evaluation")
        record.update(status="completed", phase="feasibility_complete", immutable_inputs_unchanged=True)
    except Exception as error:
        record["error"] = dict(type=type(error).__name__, message=str(error))
    finally:
        record["seconds"] = time.monotonic()-started
        record["runtime"] = dict(torch=torch.__version__, numpy=np.__version__, device=str(device))
        record["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device) if device.type == "cuda" and torch.cuda.is_available() else 0
        evaluation.write_json(output/"result.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("training-input", "base-fit", "other-fit", "prior-rollout", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--prior-sha256", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = run(args.training_input, args.base_fit, args.other_fit, args.prior_rollout,
                 args.prior_sha256, args.output, args.device)
    print(json.dumps(dict(status=result["status"], phase=result["phase"], error=result.get("error"))))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
