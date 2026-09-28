"""Reduce bank-DYN clean error with or without preserving its finite responses.

Run as a module with --input TRAIN --bank TRAIN_BANK --teacher-fit BANK_DYN
--teacher-query TRAIN_TARGET_ASSAY --output NEW_DIRECTORY
--arm clean_only|preserve_response --phase resource|fit --device cuda [--seed 17].
Only training inputs and a frozen training-query packet are read. Deployment
remains one ordinary restricted PCNO transition; there is no online teacher.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.time_dependent_no import fit_kolmogorov_recovery as base
from scripts.time_dependent_no import fit_kolmogorov_targets as targets
from scripts.time_dependent_no import evaluate_kolmogorov_recovery as evaluation

RECIPE = "fixed32_response_preserving_v1"
ARMS = ("clean_only", "preserve_response")
FIT_UPDATES, RESOURCE_UPDATES, BATCH_SIZE = 4096, 16, 8
LEARNING_RATE = 1e-4
MAX_FIT_SECONDS = {"clean_only": 3*3600, "preserve_response": 5*3600}
SOURCE_PATHS = targets.SOURCE_PATHS | evaluation.SOURCE_PATHS | {
    "scripts/time_dependent_no/fit_kolmogorov_response_preserving.py",
    "tests/time_dependent_no/test_fit_kolmogorov_response_preserving.py",
}
RECEIPTS = ("bank_result.json", "teacher_fit_result.json", "teacher_query_result.json")


def recipe_identity(manifest, parent, train_hash, receipts, arm, seed=17):
    if arm not in ARMS:
        raise ValueError("unknown response-preserving arm")
    if type(seed) is not int or seed < 0:
        raise ValueError("sampler seed must be a nonnegative integer")
    bank, fitted, query = [json.loads(receipts[k]) for k in RECEIPTS]
    bank_hash, fit_hash, query_hash = [hashlib.sha256(receipts[k]).hexdigest() for k in RECEIPTS]
    initial = fitted["identity"]
    expected = targets.recipe_identity(manifest, parent, train_hash, bank_hash, "dynamics")
    if (bank.get("role") != "train" or not bank.get("qualified")
            or bank.get("status") != "completed" or bank.get("recipe") != targets.bank.RECIPE
            or bank.get("training_manifest_sha256") != train_hash
            or bank.get("parent_checkpoint_sha256") != manifest["artifacts"]["checkpoint.pt"]
            or fitted.get("status") != "completed" or fitted.get("phase") != "fit"
            or fitted.get("updates_completed") != targets.FIT_UPDATES or initial != expected
            or query.get("status") != "completed" or query.get("phase") != "target_assay"
            or query.get("bank_role") != "train" or query.get("recurrent_model_calls") != 0
            or query.get("training_manifest_sha256") != train_hash
            or query.get("bank_manifest_sha256") != bank_hash
            or query["model"].get("identity") != initial
            or query["model"].get("terminal.pt") != fitted["artifacts"]["terminal.pt"]
            or query["model"].get("result.json") != fit_hash):
        raise ValueError("teacher/bank must be the completed, matched training-only DYN query")
    return dict(recipe=RECIPE, arm=arm, updates=FIT_UPDATES,
                parent_checkpoint_sha256=manifest["artifacts"]["checkpoint.pt"],
                initial_checkpoint_sha256=fitted["artifacts"]["terminal.pt"],
                input_manifest_sha256=train_hash, bank_manifest_sha256=bank_hash,
                teacher_fit_sha256=fit_hash, teacher_query_sha256=query_hash,
                model_config=parent["model_config"],
                train_scale_float64=initial["train_scale_float64"],
                train_scale_model_float32=initial["train_scale_model_float32"],
                learning_rate=LEARNING_RATE, batch_size_per_branch=BATCH_SIZE,
                clean_branch_weights=[.5, .5], sampler_seeds=[seed, 100*seed+1, 100*seed+3],
                response_weight=1. if arm == "preserve_response" else 0.,
                response_target="frozen DYN(x)-DYN(u), both in the common restricted state space",
                response_gradients="through both G(x) and G(u); frozen target",
                deployment="unchanged one-state restricted PCNO transition")


def update(model, optimizer, clean_batches, response_batch, device, weight):
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for x, y in clean_batches:
        prediction = model(x.to(device))["next_state"]
        loss = ((prediction-y.to(device))/model.train_scale).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite clean loss")
        (.5*loss).backward()
        losses.append(float(loss.detach()))
        del prediction, loss
    response_value = 0.
    if weight:
        u, x, fixed_response = [v.to(device) for v in response_batch]
        # Neither side is detached: an additive output bias must cancel here.
        response = model(x)["next_state"] - model(u)["next_state"]
        penalty = ((response-fixed_response)/model.train_scale).square().mean()
        if not torch.isfinite(penalty):
            raise RuntimeError("nonfinite response loss")
        (weight*penalty).backward()
        response_value = float(penalty.detach())
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    norm = torch.stack([g.detach().double().square().sum() for g in gradients]).sum().sqrt()
    if not torch.isfinite(norm):
        raise RuntimeError("nonfinite response-preserving gradient")
    optimizer.step()
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise RuntimeError("nonfinite updated parameter")
    return dict(first_mse_scaled=losses[0], second_mse_scaled=losses[1],
                response_mse_scaled=response_value,
                loss=.5*sum(losses)+weight*response_value, gradient_l2=float(norm))


def load_fitted(parent, manifest, train_hash, path, device):
    """Load this exact terminal recipe for the existing assay/rollout evaluator."""
    path = Path(path)
    raw = (path / "result.json").read_bytes()
    result = json.loads(raw)
    receipts = {k: (path / k).read_bytes() for k in RECEIPTS}
    hashes = {k: hashlib.sha256(v).hexdigest() for k, v in receipts.items()}
    if any(result["artifacts"].get(k) != v for k, v in hashes.items()):
        raise ValueError("response fit receipt hash mismatch")
    identity = recipe_identity(manifest, parent, train_hash, receipts, result["identity"].get("arm"),
                               result["identity"]["sampler_seeds"][0])
    if (result.get("status") != "completed" or result.get("phase") != "fit"
            or result.get("updates_completed") != FIT_UPDATES
            or result.get("input_manifest") != manifest or result["identity"] != identity):
        raise ValueError("response fit identity/schedule mismatch")
    sources = {k: base.sha256(base.ROOT / k) for k in sorted(SOURCE_PATHS)}
    if result["sources"] != sources:
        raise ValueError("response fit source closure mismatch")
    raw_checkpoint = (path / "terminal.pt").read_bytes()
    hashes.update({"result.json": hashlib.sha256(raw).hexdigest(),
                   "terminal.pt": hashlib.sha256(raw_checkpoint).hexdigest()})
    if result["artifacts"]["terminal.pt"] != hashes["terminal.pt"]:
        raise ValueError("response fit checkpoint hash mismatch")
    checkpoint = torch.load(io.BytesIO(raw_checkpoint), map_location="cpu", weights_only=True)
    if (checkpoint["identity"] != identity or checkpoint["update"] != FIT_UPDATES
            or checkpoint["schedule_position"] != FIT_UPDATES):
        raise ValueError("response checkpoint identity/schedule mismatch")
    model = evaluation.PeriodicVorticityPCNO(**identity["model_config"], train_scale=identity["train_scale_float64"])
    model.load_state_dict(checkpoint["model"], strict=True)
    if float(model.train_scale) != identity["train_scale_model_float32"]:
        raise ValueError("response checkpoint normalizer mismatch")
    model.to(device).eval().requires_grad_(False)
    return model, dict(identity=identity, **hashes), hashes


def run(input_path, bank_path, teacher_fit, teacher_query, output_path, arm, phase, device, seed=17):
    if arm not in ARMS or phase not in ("resource", "fit"):
        raise ValueError("unknown arm or phase")
    input_path, bank_path, teacher_fit, teacher_query, output = [Path(p).resolve() for p in
        (input_path, bank_path, teacher_fit, teacher_query, output_path)]
    if any(output.is_relative_to(p) for p in (input_path, bank_path, teacher_fit, teacher_query)):
        raise ValueError("output must be outside immutable inputs")
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(seed)
    manifest, captured, parent, train, train_hash = base.load_inputs(input_path)
    bank, arrays, bank_hash = targets.bank.load_bank(bank_path, "train", train_hash,
                                                   manifest["artifacts"]["checkpoint.pt"])
    receipts = dict(zip(RECEIPTS, [(bank_path / "result.json").read_bytes(),
                                  (teacher_fit / "result.json").read_bytes(),
                                  (teacher_query / "result.json").read_bytes()]))
    identity = recipe_identity(manifest, parent, train_hash, receipts, arm, seed)
    model, _, teacher_hashes = evaluation.evaluation_model(captured, parent, manifest, train_hash, teacher_fit, device)
    query = json.loads(receipts["teacher_query_result.json"])
    prediction_path = teacher_query / "target_predictions.npz"
    if base.sha256(prediction_path) != query["artifacts"]["target_predictions.npz"]:
        raise ValueError("teacher training predictions hash mismatch")
    with np.load(prediction_path) as saved:
        center_pred, displaced_pred = saved["center_prediction"], saved["displaced_prediction"]
    anchor_indices = np.array([v["anchor"] for v in bank["rows"]])
    if (center_pred.shape != arrays["centers"].shape or displaced_pred.shape != arrays["inputs"].shape
            or center_pred.dtype != np.float32 or displaced_pred.dtype != np.float32
            or not np.isfinite(center_pred).all() or not np.isfinite(displaced_pred).all()
            or len(query["target_responses"]) != len(anchor_indices)
            or [v["anchor"] for v in query["target_responses"]] != anchor_indices.tolist()):
        raise ValueError("teacher prediction shape/order mismatch")
    for i, anchor in enumerate(bank["anchors"]):
        p, t = anchor["path_index"], anchor["input_step"]
        if (not np.array_equal(arrays["centers"][i], train[p, t]) or
                not np.array_equal(arrays["clean_targets"][i], train[p, t+1])):
            raise ValueError("response bank must use exact archived training pairs")
    fixed_response = (displaced_pred.astype(np.float64)-center_pred[anchor_indices]).astype(np.float32)
    replay_errors = []
    for values, expected in ((arrays["centers"], center_pred), (arrays["inputs"], displaced_pred)):
        replay = evaluation.predict(model, values[:BATCH_SIZE], device)[1]
        error = float(np.linalg.norm(replay.astype(np.float64)-expected[:BATCH_SIZE]) /
                      np.linalg.norm(expected[:BATCH_SIZE].astype(np.float64)))
        replay_errors.append(error)
        if not np.isfinite(error) or error > 1e-5:
            raise ValueError("frozen teacher replay mismatch")
    del center_pred, displaced_pred
    sources = {k: base.sha256(base.ROOT / k) for k in sorted(SOURCE_PATHS)}
    bindings = {input_path / "manifest.json": train_hash, bank_path / "result.json": bank_hash,
                prediction_path: query["artifacts"]["target_predictions.npz"],
                teacher_query / "result.json": identity["teacher_query_sha256"]}
    for directory, artifacts in ((input_path, manifest["artifacts"]),
                                 (bank_path, bank["artifacts"]), (teacher_fit, teacher_hashes)):
        bindings.update({directory / k: v for k, v in artifacts.items()})
    output.mkdir(parents=True)
    for name, content in receipts.items():
        (output / name).write_bytes(content)
    result = dict(status="running", phase=phase, identity=identity, input_manifest=manifest,
                  sources=sources, artifacts={k: base.sha256(output / k) for k in RECEIPTS},
                  updates_completed=0, teacher_replay=dict(relative_l2=max(replay_errors), per_query=replay_errors),
                  optimizer="fresh Adam; teacher optimizer discarded", device=str(device),
                  numerical=dict(dtype="float32", amp=False, tf32=False, cpu_threads=2),
                  torch_version=str(torch.__version__), max_fit_seconds=MAX_FIT_SECONDS[arm])
    base.write_json(output / "result.json", result)
    model.train().requires_grad_(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(.9, .999), eps=1e-8, weight_decay=0)
    clean_count = len(train)*(train.shape[1]-1)
    samplers = [base.EpochSampler(n, BATCH_SIZE, stream_seed) for n, stream_seed in
                zip((clean_count, clean_count, len(anchor_indices)), identity["sampler_seeds"])]
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    base._sync(device)
    started, durations = time.perf_counter(), []
    count = RESOURCE_UPDATES if phase == "resource" else FIT_UPDATES
    tensor = lambda a: torch.from_numpy(np.array(a, dtype=np.float32, copy=True))
    try:
        with (output / "updates.jsonl").open("x") as log:
            for step in range(1, count+1):
                tick = time.perf_counter()
                first, second, selected = [s.next_indices() for s in samplers]
                clean, _ = base.training_batches(train, first, second, "clean_continuation", float(model.train_scale), None)
                response = ((tensor(arrays["centers"][anchor_indices[selected]]), tensor(arrays["inputs"][selected]),
                             tensor(fixed_response[selected])) if identity["response_weight"] else None)
                row = update(model, optimizer, clean, response, device, identity["response_weight"])
                base._sync(device)
                durations.append(time.perf_counter()-tick)
                row.update(update=step, first_indices=first.tolist(), second_indices=second.tolist(),
                           response_indices=selected.tolist(), seconds=durations[-1])
                log.write(json.dumps(row, allow_nan=False)+"\n")
                log.flush()
                result["updates_completed"] = step
                if step % 256 == 0:
                    base.write_json(output / "result.json", result)
                if phase == "fit" and time.perf_counter()-started > MAX_FIT_SECONDS[arm]:
                    raise RuntimeError("response fit exceeded declared wall budget")
                if step == RESOURCE_UPDATES:
                    estimate = float(np.mean(durations[min(4, len(durations)-1):]))*FIT_UPDATES
                    if phase == "fit" and estimate > MAX_FIT_SECONDS[arm]:
                        raise RuntimeError("warmed response fit estimate exceeds wall budget")
        seconds = float(np.mean(durations[min(4, len(durations)-1):]))
        result.update(status="completed", training_seconds=time.perf_counter()-started,
                      warmed_seconds_per_update=seconds, estimated_fit_seconds=seconds*FIT_UPDATES,
                      within_fit_budget=seconds*FIT_UPDATES <= MAX_FIT_SECONDS[arm],
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if torch.device(device).type == "cuda" else None,
                      peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device) if torch.device(device).type == "cuda" else None,
                      final_training_row=row)
        if sources != {k: base.sha256(base.ROOT / k) for k in sorted(SOURCE_PATHS)}:
            raise RuntimeError("response fit sources changed")
        if any(base.sha256(p) != digest for p, digest in bindings.items()):
            raise RuntimeError("response fit input changed")
        if phase == "fit":
            torch.save(dict(identity=identity, update=count, schedule_position=count, model=model.state_dict()), output / "terminal.pt")
            result["artifacts"]["terminal.pt"] = base.sha256(output / "terminal.pt")
        result["artifacts"]["updates.jsonl"] = base.sha256(output / "updates.jsonl")
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        base.write_json(output / "result.json", result)
        raise
    base.write_json(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "bank", "teacher-fit", "teacher-query", "output"):
        parser.add_argument("--"+name, required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--phase", required=True, choices=("resource", "fit"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=17,
                        help="paired sampler seed; all other fitting settings remain fixed")
    args = parser.parse_args()
    run(args.input, args.bank, args.teacher_fit, args.teacher_query, args.output, args.arm, args.phase,
        args.device, args.seed)


if __name__ == "__main__":
    main()
