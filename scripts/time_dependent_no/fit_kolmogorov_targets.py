"""Matched REC/DYN adaptation with one shared frozen displaced-input bank.

Run ``python -m scripts.time_dependent_no.fit_kolmogorov_targets --input TRAIN
--bank TRAIN_BANK --output NEW_DIRECTORY --arm recovery|dynamics
--phase resource|fit --device cuda``. Neither phase opens a probe population.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.time_dependent_no import fit_kolmogorov_recovery as base
from scripts.time_dependent_no import generate_kolmogorov_target_bank as bank

ARMS, RECIPE = ("recovery", "dynamics"), "fixed32_paired_targets_v1"
FIT_UPDATES, RESOURCE_UPDATES, BATCH_SIZE = 4096, 16, 8
LEARNING_RATE, MAX_FIT_SECONDS = 1e-4, 3 * 3600
SOURCE_PATHS = bank.SOURCE_PATHS | {
    "scripts/time_dependent_no/fit_kolmogorov_targets.py",
    "tests/time_dependent_no/test_fit_kolmogorov_targets.py",
}


def training_batches(train, arrays, anchor_indices, clean_indices, bank_indices, arm):
    if arm not in ARMS:
        raise ValueError("unknown target arm")
    p, t = np.divmod(clean_indices, train.shape[1] - 1)
    tensor = lambda a: torch.from_numpy(np.array(a, dtype=np.float32, copy=True))
    clean = (tensor(train[p, t]), tensor(train[p, t+1]))
    target = (arrays["clean_targets"][anchor_indices[bank_indices]] if arm == "recovery"
              else arrays["dynamics_targets"][bank_indices])
    return clean, (tensor(arrays["inputs"][bank_indices]), tensor(target))


def recipe_identity(manifest, parent_fit, manifest_hash, bank_hash, arm):
    if arm not in ARMS or not isinstance(bank_hash, str) or len(bank_hash) != 64:
        raise ValueError("unknown target arm or bank identity")
    parent = parent_fit["checkpoint_identity"]
    return dict(recipe=RECIPE, arm=arm,
                parent_checkpoint_sha256=manifest["artifacts"]["checkpoint.pt"],
                input_manifest_sha256=manifest_hash, bank_manifest_sha256=bank_hash,
                model_config=parent_fit["model_config"],
                train_scale_float64=parent["train_scale_float64"],
                train_scale_model_float32=parent["train_scale_model_float32"],
                updates=FIT_UPDATES, batch_size_per_branch=BATCH_SIZE,
                learning_rate=LEARNING_RATE, sampler_seeds=[17, 1701], branch_weights=[.5, .5],
                targets=("archived clean successor at bank center" if arm == "recovery"
                         else "trusted Phi(P(exact FP32 bank input)), cast to FP32"),
                deployment="unchanged one-state restricted PCNO transition")


def _run(input_path, bank_path, output, arm, phase, device, result):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(17)
    manifest, captured, parent_fit, train, manifest_hash = base.load_inputs(input_path)
    bank_result, arrays, bank_hash = bank.load_bank(
        bank_path, "train", manifest_hash, manifest["artifacts"]["checkpoint.pt"])
    for i, anchor in enumerate(bank_result["anchors"]):
        p, t = anchor["path_index"], anchor["input_step"]
        if (not np.array_equal(arrays["centers"][i], train[p, t]) or
                not np.array_equal(arrays["clean_targets"][i], train[p, t+1])):
            raise ValueError("bank center/successor is not the archived training pair")
    sources = {name: base.sha256(base.ROOT / name) for name in sorted(SOURCE_PATHS)}
    model = base.load_model(captured, parent_fit, device)
    replay = base.terminal_replay(model, captured["teacher.npz"], device, len(base.TRAIN_SEEDS))
    if float(model.train_scale) != bank_result["train_scale"]:
        raise ValueError("bank/parent normalizer mismatch")
    output.mkdir(parents=True, exist_ok=False)
    model.train().requires_grad_(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 betas=(.9, .999), eps=1e-8, weight_decay=0)
    clean_sampler = base.EpochSampler(len(train) * (train.shape[1]-1), BATCH_SIZE, 17)
    bank_sampler = base.EpochSampler(len(arrays["inputs"]), BATCH_SIZE, 1701)
    anchor_indices = np.array([r["anchor"] for r in bank_result["rows"]])
    identity = recipe_identity(manifest, parent_fit, manifest_hash, bank_hash, arm)
    # The exact bank receipt accompanies the checkpoint without copying its arrays.
    (output / "bank_result.json").write_bytes((bank_path / "result.json").read_bytes())
    result.update(schema_version=1, status="running", identity=identity,
                  input_manifest=manifest, sources=sources, parent_replay=replay,
                  artifacts={"bank_result.json": bank_hash}, bank_rows=len(anchor_indices),
                  optimizer="fresh Adam; parent optimizer/RNG discarded",
                  numerical=dict(dtype="float32", amp=False, tf32=False, cpu_threads=2),
                  device=str(device), torch_version=str(torch.__version__),
                  max_fit_seconds=MAX_FIT_SECONDS)
    base.write_json(output / "result.json", result)
    count = RESOURCE_UPDATES if phase == "resource" else FIT_UPDATES
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    base._sync(device)
    started, durations = time.perf_counter(), []
    with (output / "updates.jsonl").open("x") as log:
        for step in range(1, count+1):
            tick = time.perf_counter()
            first, second = clean_sampler.next_indices(), bank_sampler.next_indices()
            batches = training_batches(train, arrays, anchor_indices, first, second, arm)
            row = base.update(model, optimizer, batches, device)
            base._sync(device)
            durations.append(time.perf_counter() - tick)
            row.update(update=step, first_indices=first.tolist(), second_indices=second.tolist(), seconds=durations[-1])
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
            result["updates_completed"] = step
            if step % 256 == 0:
                base.write_json(output / "result.json", result)
            if phase == "fit" and time.perf_counter() - started > MAX_FIT_SECONDS:
                raise RuntimeError("target fit exceeded the declared wall budget")
            if step == RESOURCE_UPDATES:
                estimate = np.mean(durations[min(4, len(durations)-1):]) * FIT_UPDATES
                if phase == "fit" and estimate > MAX_FIT_SECONDS:
                    raise RuntimeError("warmed target fit estimate exceeds the wall budget")
    mean_seconds = float(np.mean(durations[min(4, len(durations)-1):]))
    result.update(status="completed", training_seconds=time.perf_counter() - started,
                  warmed_seconds_per_update=mean_seconds, estimated_fit_seconds=mean_seconds * FIT_UPDATES,
                  within_fit_budget=mean_seconds * FIT_UPDATES <= MAX_FIT_SECONDS,
                  peak_cuda_allocated_bytes=(torch.cuda.max_memory_allocated(device)
                                             if torch.device(device).type == "cuda" else None),
                  peak_cuda_reserved_bytes=(torch.cuda.max_memory_reserved(device)
                                            if torch.device(device).type == "cuda" else None),
                  final_training_row=row)
    if phase == "fit":
        torch.save(dict(identity=identity, update=count, schedule_position=count,
                        model=model.state_dict()), output / "terminal.pt")
        result["artifacts"]["terminal.pt"] = base.sha256(output / "terminal.pt")
    result["artifacts"]["updates.jsonl"] = base.sha256(output / "updates.jsonl")
    if sources != {name: base.sha256(base.ROOT / name) for name in sorted(SOURCE_PATHS)}:
        raise RuntimeError("target fit sources changed during execution")
    for path, receipt, digest in ((input_path, manifest, manifest_hash), (bank_path, bank_result, bank_hash)):
        name = "manifest.json" if path == input_path else "result.json"
        if base.sha256(path / name) != digest or any(base.sha256(path / k) != v for k, v in receipt["artifacts"].items()):
            raise RuntimeError("target fit input changed during execution")
    base.write_json(output / "result.json", result)
    return result


def run(input_path, bank_path, output_path, arm, phase, device):
    if arm not in ARMS or phase not in ("resource", "fit"):
        raise ValueError("unknown arm or phase")
    input_path, bank_path, output = [Path(p).resolve() for p in (input_path, bank_path, output_path)]
    if any(output.is_relative_to(p) for p in (input_path, bank_path)):
        raise ValueError("output must be outside immutable inputs")
    if output.exists():
        raise FileExistsError(output)
    result = dict(status="initializing", phase=phase, arm=arm, updates_completed=0)
    try:
        return _run(input_path, bank_path, output, arm, phase, device, result)
    except Exception as error:
        if output.exists():
            result.update(status="failed", error=f"{type(error).__name__}: {error}")
            base.write_json(output / "result.json", result)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--phase", required=True, choices=("resource", "fit"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args.input, args.bank, args.output, args.arm, args.phase, args.device)


if __name__ == "__main__":
    main()
