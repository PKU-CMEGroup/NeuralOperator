"""Train a reusable independent clean probe on the authenticated fixed data.

Run ``python -m scripts.time_dependent_no.fit_kolmogorov_clean_probe --input
TRAIN_PACKET --output NEW_DIRECTORY --phase resource|fit --device cuda``.
Resource measures sixteen updates without retaining weights. Fit starts afresh
and retains the fixed 49,152-update terminal, regardless of diagnostic outcomes.
Use ``--resume`` with the same fit directory to recover its atomic latest state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from scripts.time_dependent_no import fit_kolmogorov_clean as clean
from scripts.time_dependent_no import fit_kolmogorov_recovery as training

RUN_ID = "CM_NEXT_KF_CLEAN_PROBE_20261001A"
FIT_UPDATES, RESOURCE_UPDATES, BATCH_SIZE, SEED = 49152, 16, 8, 29
CHECKPOINT_INTERVAL, PROGRESS_INTERVAL, MAX_FIT_SECONDS = 1024, 256, 14 * 3600
SOURCE_PATHS = training.SOURCE_PATHS | {
    "scripts/time_dependent_no/fit_kolmogorov_clean_probe.py",
    "tests/time_dependent_no/test_fit_kolmogorov_clean_probe.py",
}


def sources():
    return {name: training.sha256(training.ROOT / name) for name in sorted(SOURCE_PATHS)}


def state_fingerprint(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(f"{name}:{array.dtype}:{array.shape}\n".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def make_model(config, scale, device):
    """Fresh initialization only; the authenticated parent weights are not loaded."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.device(device).type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    model = clean.PeriodicVorticityPCNO(**config, train_scale=scale).to(device)
    torch.nn.init.zeros_(model.pcno.fc2.weight)
    torch.nn.init.zeros_(model.pcno.fc2.bias)
    return model


def training_batch(train, indices):
    paths, steps = np.divmod(indices, train.shape[1] - 1)
    return tuple(torch.from_numpy(np.array(train[paths, times], copy=True))
                 for times in (steps, steps + 1))


def update(model, optimizer, batch, device, position):
    """The original single clean objective and unchanged one-indexed LR schedule."""
    learning_rate = clean.clean_lr(position)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    inputs, targets = (value.to(device) for value in batch)
    if not (torch.isfinite(inputs).all() and torch.isfinite(targets).all()):
        raise ValueError("nonfinite clean training pair")
    optimizer.zero_grad(set_to_none=True)
    prediction = model(inputs)["next_state"]
    loss = ((prediction - targets) / model.train_scale).square().mean()
    if not torch.isfinite(loss):
        raise RuntimeError("nonfinite clean training loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"),
                                          error_if_nonfinite=True)
    optimizer.step()
    return dict(learning_rate=learning_rate, loss_normalized_mse=float(loss.detach()),
                unclipped_gradient_norm=float(norm))


def recipe():
    return dict(updates=FIT_UPDATES, batch_size=BATCH_SIZE, initialization_seed=SEED,
                sampler_seed=SEED, zero_final_head=True,
                optimizer=dict(name="Adam", betas=[0.9, 0.999], eps=1e-8, weight_decay=0.0),
                learning_rate="fit_kolmogorov_clean.clean_lr(update), unchanged full history",
                sampling="complete without-replacement permutation of clean transitions per epoch",
                loss="mean(((deployed_next_state-clean_next_state)/fixed_train_scale)^2)",
                precision="float32; no AMP/TF32; no gradient clipping",
                selection="fixed terminal update; no validation or rollout-based selection",
                normalizer="unchanged original training scale from authenticated packet")


def _trim_uncheckpointed_log(path, update):
    """Resume replays only updates newer than the durable checkpoint."""
    temporary = path.with_name(path.name + ".tmp")
    with path.open(encoding="utf-8") as source, temporary.open("w", encoding="utf-8") as target:
        for line in source:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break  # An interrupted append may leave an incomplete final line.
            if row["update"] > update:
                break
            target.write(line)
    temporary.replace(path)


def _run(input_path, output, phase, device, result, resume):
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest, captured, original, train, manifest_hash = training.load_inputs(input_path)
    del captured  # Authenticated parent checkpoint is never deserialized into this model.
    source_hashes = sources()
    scale = original["checkpoint_identity"]["train_scale_float64"]
    model = make_model(original["model_config"], scale, device)
    if float(model.train_scale) != original["checkpoint_identity"]["train_scale_model_float32"]:
        raise ValueError("probe normalizer differs from the original model")
    identity = dict(run_id=RUN_ID, model_config=original["model_config"],
                    train_scale_float64=scale, train_scale_model_float32=float(model.train_scale),
                    input_manifest_sha256=manifest_hash,
                    train_array_sha256=manifest["artifacts"]["train.npy"],
                    original_checkpoint_sha256=manifest["artifacts"]["checkpoint.pt"],
                    original_weights_loaded=False, initialization_state_sha256=state_fingerprint(model),
                    recipe=recipe(), sources=source_hashes)
    optimizer = torch.optim.Adam(model.parameters(), lr=clean.clean_lr(1),
                                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    sampler = clean.EpochSampler(train.shape[0] * (train.shape[1] - 1), BATCH_SIZE, SEED)
    position = 0
    if resume:
        if result["identity"] != identity or result["phase"] != "fit":
            raise ValueError("resume identity or phase differs from the original fit")
        position = clean.load_clean_checkpoint(output / "latest.pt", model, optimizer,
                                              sampler, identity=identity)
        if not 0 <= position < FIT_UPDATES:
            raise ValueError("resume checkpoint is not an unfinished fit")
        _trim_uncheckpointed_log(output / "updates.jsonl", position)
    else:
        output.mkdir(parents=True, exist_ok=False)
    result.update(schema_version=1, status="running", phase=phase, identity=identity,
                  input_manifest=manifest, updates_completed=position,
                  resumed_from_update=position if resume else None,
                  original_weights_loaded=False, development_access=False,
                  rollout_access=False, solver_calls=0, device=str(device),
                  max_session_training_seconds=MAX_FIT_SECONDS,
                  torch_version=str(torch.__version__), artifacts={})
    result.pop("error", None)
    training.write_json(output / "result.json", result)
    model.train()

    def checkpoint():
        digest = clean.save_clean_checkpoint(output / "latest.pt", model, optimizer,
                                             sampler, update=position, identity=identity)
        result["latest_checkpoint"] = dict(file="latest.pt", update=position, sha256=digest)
        training.write_json(output / "result.json", result)

    if phase == "fit" and not resume:
        checkpoint()
    count = RESOURCE_UPDATES if phase == "resource" else FIT_UPDATES
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    training._sync(device)
    started, durations = time.perf_counter(), []
    with (output / "updates.jsonl").open("a" if resume else "x", encoding="utf-8") as log:
        for position in range(position + 1, count + 1):
            tick = time.perf_counter()
            selected = sampler.next_indices()
            row = update(model, optimizer, training_batch(train, selected), device, position)
            training._sync(device)
            durations.append(time.perf_counter() - tick)
            row.update(update=position, epoch=sampler.epoch, sampler_cursor=sampler.cursor,
                       train_transition_indices=selected.tolist(), seconds=durations[-1])
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
            result.update(updates_completed=position, final_training_row=row,
                          session_training_seconds=time.perf_counter() - started)
            if position % CHECKPOINT_INTERVAL == 0 and phase == "fit":
                checkpoint()
            if position % PROGRESS_INTERVAL == 0:
                training.write_json(output / "result.json", result)
                print(json.dumps(dict(update=position, loss=row["loss_normalized_mse"],
                                      seconds=result["session_training_seconds"])), flush=True)
            if phase == "fit" and time.perf_counter() - started > MAX_FIT_SECONDS:
                checkpoint()
                raise RuntimeError("clean probe fit exceeded its session time budget")
            if phase == "fit" and len(durations) == RESOURCE_UPDATES:
                estimate = float(np.mean(durations[min(4, len(durations) - 1):])) * (count - position)
                if estimate > MAX_FIT_SECONDS - (time.perf_counter() - started):
                    checkpoint()
                    raise RuntimeError("warmed remaining-fit estimate exceeds the session time budget")
    warm = durations[min(4, len(durations) - 1):]
    result.update(warmed_seconds_per_update=float(np.mean(warm)),
                  estimated_fit_seconds=float(np.mean(warm)) * FIT_UPDATES,
                  within_fit_budget=float(np.mean(warm)) * FIT_UPDATES <= MAX_FIT_SECONDS,
                  peak_cuda_allocated_bytes=(torch.cuda.max_memory_allocated(device)
                                             if torch.device(device).type == "cuda" else None),
                  peak_cuda_reserved_bytes=(torch.cuda.max_memory_reserved(device)
                                            if torch.device(device).type == "cuda" else None))
    if sources() != source_hashes:
        raise RuntimeError("training source changed during execution")
    if phase == "fit":
        result["artifacts"]["terminal.pt"] = clean.save_clean_checkpoint(
            output / "terminal.pt", model, optimizer, sampler, update=count, identity=identity)
    result["artifacts"]["updates.jsonl"] = training.sha256(output / "updates.jsonl")
    result["status"] = "completed"
    training.write_json(output / "result.json", result)
    training.write_json(output / "artifact_manifest.json", dict(
        run_id=RUN_ID, phase=phase, status="completed", sources=source_hashes,
        artifacts={**result["artifacts"],
                   "result.json": training.sha256(output / "result.json")}))
    return result


def run(input_path, output_path, phase, device, *, resume=False):
    input_path, output = Path(input_path).resolve(), Path(output_path).resolve()
    if phase not in ("resource", "fit") or (resume and phase != "fit"):
        raise ValueError("resume is available only for the fit phase")
    if output.is_relative_to(input_path):
        raise ValueError("output must be outside the immutable input directory")
    if output.exists() and not resume:
        raise FileExistsError(output)
    result = (json.loads((output / "result.json").read_text(encoding="utf-8"))
              if resume else dict(status="initializing", phase=phase, updates_completed=0))
    if resume and result["status"] == "completed":
        raise ValueError("completed fit must not be resumed")
    try:
        return _run(input_path, output, phase, device, result, resume)
    except Exception as error:
        if output.exists():
            result.update(status="failed", error=f"{type(error).__name__}: {error}")
            training.write_json(output / "result.json", result)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("resource", "fit"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.input, args.output, args.phase, args.device, resume=args.resume)


if __name__ == "__main__":
    main()
