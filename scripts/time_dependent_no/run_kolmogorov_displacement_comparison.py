"""Compare REC/DYN targets for Gaussian, one-step and mixed native errors.

Run ``python -m scripts.time_dependent_no.run_kolmogorov_displacement_comparison
--input TRAIN_PACKET --output NEW_DIRECTORY --after-probe PROBE_FIT --device cuda``.
The queue waits for the independent probe, prepares shared training inputs and
solver labels, then trains six fixed-budget continuations sequentially. It never
opens validation data. The old rescaled-direction experiments remain unchanged.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gc
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from scripts.time_dependent_no import fit_kolmogorov_recovery as base
from utility.time_dependent_no.kolmogorov_reference import (
    KolmogorovReferenceConfig, KolmogorovReferenceStepper,
)

RECIPE = "fixed32_native_displacement_comparison_v1"
CONDITIONS = ("native_one", "native_mixed", "gaussian")
TARGETS = ("recovery", "dynamics")
LENGTHS = (1, 2, 4, 8)
ROWS_PER_PATH, INPUT_SEED, NOISE_SEED = 32, 2026100102, 2026100103
FIT_UPDATES, BATCH_SIZE, CHECKPOINT_INTERVAL = 4096, 8, 1024
LEARNING_RATE, MAX_FIT_SECONDS, MAX_PREPARE_SECONDS = 1e-4, 3 * 3600, 4 * 3600
REFERENCE = KolmogorovReferenceConfig(resolution=256, viscosity=.01, macro_dt=.05)
LIMITS = dict(projection_rms_scaled=1e-5, clean_restart_rms_scaled=1e-3,
              temporal_rms_scaled=1e-3, label_rounding_rms_scaled=1e-5)
SOURCE_PATHS = base.SOURCE_PATHS | {
    "utility/time_dependent_no/kolmogorov_reference.py",
    "scripts/time_dependent_no/run_kolmogorov_displacement_comparison.py",
    "tests/time_dependent_no/test_run_kolmogorov_displacement_comparison.py",
}


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    base.write_json(temporary, value)
    temporary.replace(path)


def sources():
    return {name: base.sha256(base.ROOT / name) for name in sorted(SOURCE_PATHS)}


def protocol():
    return dict(recipe=RECIPE, conditions=list(CONDITIONS), targets=list(TARGETS),
                lengths=list(LENGTHS), rows_per_path=ROWS_PER_PATH,
                input_seed=INPUT_SEED, noise_seed=NOISE_SEED,
                reference_config=asdict(REFERENCE), qualification_limits=LIMITS,
                training=dict(updates=FIT_UPDATES, batch_size_per_branch=BATCH_SIZE,
                              learning_rate=LEARNING_RATE, branch_weights=[.5, .5],
                              sampler_seeds=[17, 1701], initialization_seed=17,
                              optimizer=dict(name="Adam", betas=[.9, .999], eps=1e-8, weight_decay=0.),
                              precision="float32; no AMP/TF32; no gradient clipping",
                              selection="fixed terminal; no validation selection"),
                native="F^k(u[t-k]); clean center u[t]; no error rescaling",
                gaussian="mean-zero band-limited Gaussian; expected RMS equals pooled native_mixed RMS",
                sampling="same endpoints in all conditions; one random time per equal-width stratum",
                information="training population only; extra offline labels for DYN",
                wait_for="completed independent probe and verified terminal checkpoint")


def rms(x):
    values = np.asarray(x, dtype=np.float64)
    if not np.isfinite(values).all():
        raise FloatingPointError("nonfinite field")
    return float(np.sqrt(np.mean(values**2)))


def sample_rows(path_count, transition_count):
    """Equal endpoint distribution and equal counts of each mixed rollout length."""
    if ROWS_PER_PATH % len(LENGTHS) or transition_count - max(LENGTHS) < ROWS_PER_PATH:
        raise ValueError("insufficient time strata or unbalanced rollout lengths")
    rng = np.random.default_rng(INPUT_SEED)
    strata = np.array_split(np.arange(max(LENGTHS), transition_count), ROWS_PER_PATH)
    rows = []
    for path in range(path_count):
        lengths = np.tile(LENGTHS, ROWS_PER_PATH // len(LENGTHS))
        rng.shuffle(lengths)
        for times, length in zip(strata, lengths):
            rows.append(dict(path_index=path, input_step=int(rng.choice(times)),
                             mixed_length=int(length)))
    return rows


def native_inputs(model, train, rows, mixed, device):
    """Return actual frozen-model predictions, never normalized directions."""
    predictions = []
    for row in rows:
        k = row["mixed_length"] if mixed else 1
        state = torch.from_numpy(np.array(
            train[row["path_index"], row["input_step"] - k][None], copy=True)).to(device)
        with torch.no_grad():
            for _ in range(k):
                state = model(state)["next_state"]
                if not torch.isfinite(state).all() or state.abs().max() > 1e6:
                    raise FloatingPointError("frozen generator failed; no clipping or dropped rows")
        predictions.append(state[0].cpu().numpy().copy())
    return np.stack(predictions)


def gaussian_input(center, expected_rms, generator, solver):
    """A Gaussian draw retains its random norm; only its ensemble scale is set."""
    n = center.shape[-1]
    rank = (2 * (n // 3) + 1)**2 - 1
    noise = solver.canonicalize(generator.standard_normal(center.shape))
    return (center.astype(np.float64) + noise * expected_rms * n / math.sqrt(rank)).astype(np.float32)


def check_metrics(row):
    for key, limit in LIMITS.items():
        if key in row and (not np.isfinite(row[key]) or row[key] > limit):
            raise ValueError(f"numerical qualification failed: {key}={row[key]} > {limit}")


def prepare_inputs(train, model, manifest, manifest_hash, output):
    output.mkdir(parents=True, exist_ok=False)
    rows = sample_rows(len(train), train.shape[1] - 1)
    count, n, scale = len(rows), train.shape[-1], float(model.train_scale)
    result = dict(status="running", protocol=protocol(), sources=sources(),
                  input_manifest_sha256=manifest_hash,
                  pretrained_checkpoint_sha256=manifest["artifacts"]["checkpoint.pt"],
                  train_scale=scale, rows=rows, artifacts={}, completed_labels=0,
                  development_access=False, refinements=[], clean_checks=[], label_checks=[])
    started = time.perf_counter()
    write_json(output / "result.json", result)
    arrays = {}
    try:
        for name in ("centers", "clean_targets"):
            arrays[name] = np.lib.format.open_memmap(
                output / (name + ".npy"), mode="w+", dtype="float32", shape=(count, n, n))
        for condition in CONDITIONS:
            for suffix, dtype in (("inputs", "float32"), ("dynamics", "float64")):
                name = condition + "_" + suffix
                arrays[name] = np.lib.format.open_memmap(
                    output / (name + ".npy"), mode="w+", dtype=dtype, shape=(count, n, n))
        model.eval().requires_grad_(False)
        for begin in range(0, count, BATCH_SIZE):
            selected = rows[begin:begin + BATCH_SIZE]
            indices = slice(begin, begin + len(selected))
            for offset, row in enumerate(selected, begin):
                p, t = row["path_index"], row["input_step"]
                arrays["centers"][offset] = train[p, t]
                arrays["clean_targets"][offset] = train[p, t + 1]
            arrays["native_one_inputs"][indices] = native_inputs(model, train, selected, False, next(model.parameters()).device)
            arrays["native_mixed_inputs"][indices] = native_inputs(model, train, selected, True, next(model.parameters()).device)
            if time.perf_counter() - started > MAX_PREPARE_SECONDS:
                raise TimeoutError("input preparation exceeded its wall budget")
        magnitudes = {}
        for condition in CONDITIONS[:2]:
            magnitudes[condition] = [rms(x.astype(np.float64) - center) / scale
                                     for x, center in zip(arrays[condition + "_inputs"], arrays["centers"])]
        expected_rms = scale * float(np.sqrt(np.mean(np.square(magnitudes["native_mixed"]))))
        if not np.isfinite(expected_rms) or expected_rms <= 0:
            raise ValueError("invalid Gaussian scale calibration")
        result["gaussian_expected_rms_scaled"] = expected_rms / scale
        solver = KolmogorovReferenceStepper(REFERENCE)
        fine = KolmogorovReferenceStepper(replace(REFERENCE, dt_max=REFERENCE.dt_max / 2,
                                                  cfl=REFERENCE.cfl / 2))
        rng = np.random.default_rng(NOISE_SEED)
        for i, center in enumerate(arrays["centers"]):
            arrays["gaussian_inputs"][i] = gaussian_input(center, expected_rms, rng, solver)
        magnitudes["gaussian"] = [rms(x.astype(np.float64) - center) / scale
                                  for x, center in zip(arrays["gaussian_inputs"], arrays["centers"])]
        result["displacement_rms_scaled"] = magnitudes
        result["displacement_summary"] = {
            name: dict(pooled_rms_scaled=float(np.sqrt(np.mean(np.square(values)))),
                       quantiles_scaled=np.quantile(values, [0, .25, .5, .75, 1]).tolist())
            for name, values in magnitudes.items()}
        sentinels = set(np.linspace(0, count - 1, min(16, count), dtype=int).tolist())
        sentinels.update(next(i for i, row in enumerate(rows) if row["mixed_length"] == k)
                         for k in LENGTHS)
        result["refinement_row_indices"] = sorted(sentinels)
        write_json(output / "result.json", result)
        for i, row in enumerate(rows):
            clean = solver.advance_projected(arrays["centers"][i]).state
            clean_check = dict(row=i, clean_restart_rms_scaled=rms(clean - arrays["clean_targets"][i]) / scale)
            check_metrics(clean_check)
            result["clean_checks"].append(clean_check)
            for condition in CONDITIONS:
                x = arrays[condition + "_inputs"][i].astype(np.float64)
                canonical = solver.canonicalize(x)
                if condition == "native_mixed" and row["mixed_length"] == 1:
                    if not np.array_equal(x, arrays["native_one_inputs"][i]):
                        raise ValueError("one-step inputs disagree between exposure conditions")
                    label = arrays["native_one_dynamics"][i]
                    reused = True
                else:
                    label = solver.advance_canonical(canonical).state
                    reused = False
                arrays[condition + "_dynamics"][i] = label
                check = dict(row=i, condition=condition, reused_label=reused,
                             projection_rms_scaled=rms(canonical - x) / scale,
                             label_rounding_rms_scaled=rms(label.astype(np.float32) - label) / scale)
                if i in sentinels:
                    refined = fine.advance_canonical(canonical).state
                    check["temporal_rms_scaled"] = rms(refined - label) / scale
                    result["refinements"].append(dict(check))
                check_metrics(check)
                result["label_checks"].append(check)
                result["completed_labels"] += 1
                if time.perf_counter() - started > MAX_PREPARE_SECONDS:
                    raise TimeoutError("solver labeling exceeded its wall budget")
            if i % 32 == 0 or i == count - 1:
                result["elapsed_seconds"] = time.perf_counter() - started
                write_json(output / "result.json", result)
                print(json.dumps(dict(stage="labels", rows=i + 1, total=count)), flush=True)
        for name, array in arrays.items():
            array.flush()
            result["artifacts"][name + ".npy"] = base.sha256(output / (name + ".npy"))
        if sources() != result["sources"]:
            raise RuntimeError("source changed during input preparation")
        result.update(status="completed", qualified=True, elapsed_seconds=time.perf_counter() - started,
                      qualification_scope="all clean restarts, projections and rounding; sampled temporal refinement")
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "result.json", result)
    return result


def load_prepared(output, train, manifest, manifest_hash):
    result = json.loads((output / "result.json").read_text())
    expected = dict(status="completed", qualified=True, protocol=protocol(), sources=sources(),
                    input_manifest_sha256=manifest_hash,
                    pretrained_checkpoint_sha256=manifest["artifacts"]["checkpoint.pt"],
                    rows=sample_rows(len(train), train.shape[1] - 1))
    if any(result.get(k) != v for k, v in expected.items()):
        raise ValueError("prepared input identity/qualification mismatch")
    names = ["centers", "clean_targets"] + [c + "_" + s for c in CONDITIONS for s in ("inputs", "dynamics")]
    if set(result["artifacts"]) != {name + ".npy" for name in names}:
        raise ValueError("prepared input artifact set mismatch")
    arrays = {}
    for name in names:
        filename = output / (name + ".npy")
        if base.sha256(filename) != result["artifacts"][filename.name]:
            raise ValueError("prepared input artifact hash mismatch")
        values = np.load(filename, mmap_mode="r", allow_pickle=False)
        dtype = "float64" if name.endswith("_dynamics") else "float32"
        if values.shape != (len(result["rows"]), *train.shape[-2:]) or values.dtype != dtype:
            raise ValueError("prepared input array shape/dtype mismatch")
        arrays[name] = values
    for i, row in enumerate(result["rows"]):
        p, t = row["path_index"], row["input_step"]
        if not np.array_equal(arrays["centers"][i], train[p, t]) or not np.array_equal(arrays["clean_targets"][i], train[p, t+1]):
            raise ValueError("clean center/target differs from archived training data")
    return result, arrays, base.sha256(output / "result.json")


def training_batches(train, arrays, first, second, condition, target):
    if condition not in CONDITIONS or target not in TARGETS:
        raise ValueError("unknown condition or target")
    p, t = np.divmod(first, train.shape[1] - 1)
    tensor = lambda x: torch.from_numpy(np.array(x, dtype=np.float32, copy=True))
    y = arrays["clean_targets"] if target == "recovery" else arrays[condition + "_dynamics"]
    return ((tensor(train[p, t]), tensor(train[p, t+1])),
            (tensor(arrays[condition + "_inputs"][second]), tensor(y[second])))


def save_checkpoint(path, model, optimizer, samplers, position, identity):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(dict(identity=identity, update=position, model=model.state_dict(),
                    optimizer=optimizer.state_dict(), samplers=[s.state_dict() for s in samplers],
                    torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []), temporary)
    temporary.replace(path)
    return base.sha256(path)


def train_condition(train, captured, original, input_hash, prepared_hash, arrays, condition, target, output, device):
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(17)
    model = base.load_model(captured, original, device)
    replay = base.terminal_replay(model, captured["teacher.npz"], device, len(base.TRAIN_SEEDS))
    model.train().requires_grad_(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(.9, .999), eps=1e-8, weight_decay=0.)
    samplers = [base.EpochSampler(len(train) * (train.shape[1] - 1), BATCH_SIZE, 17),
                base.EpochSampler(len(arrays["centers"]), BATCH_SIZE, 1701)]
    identity = dict(protocol=protocol(), condition=condition, target=target, sources=sources(),
                    input_manifest_sha256=input_hash, prepared_manifest_sha256=prepared_hash,
                    model_config=original["model_config"], train_scale=float(model.train_scale))
    result = dict(status="running", identity=identity, updates_completed=0, artifacts={},
                  pretrained_replay=replay, development_access=False,
                  optimizer="fresh Adam; pretrained optimizer discarded")
    write_json(output / "result.json", result)
    started, durations = time.perf_counter(), []
    try:
        save_checkpoint(output / "latest.pt", model, optimizer, samplers, 0, identity)
        if torch.device(device).type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        with (output / "updates.jsonl").open("x") as log:
            for position in range(1, FIT_UPDATES + 1):
                tick = time.perf_counter()
                first, second = [s.next_indices() for s in samplers]
                batches = training_batches(train, arrays, first, second, condition, target)
                row = base.update(model, optimizer, batches, device)
                base._sync(device)
                durations.append(time.perf_counter() - tick)
                row.update(update=position, clean_indices=first.tolist(), displaced_indices=second.tolist(), seconds=durations[-1])
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
                result["updates_completed"] = position
                if position % CHECKPOINT_INTERVAL == 0:
                    digest = save_checkpoint(output / "latest.pt", model, optimizer, samplers, position, identity)
                    result["latest_checkpoint"] = dict(update=position, sha256=digest)
                if position % 256 == 0:
                    write_json(output / "result.json", result)
                    print(json.dumps(dict(condition=condition, target=target, update=position, loss=row["loss"])), flush=True)
                if position == 16 and np.mean(durations[4:]) * FIT_UPDATES > MAX_FIT_SECONDS:
                    raise TimeoutError("estimated training time exceeds fixed wall budget")
                if time.perf_counter() - started > MAX_FIT_SECONDS:
                    raise TimeoutError("training exceeded fixed wall budget")
        if sources() != identity["sources"]:
            raise RuntimeError("source changed during training")
        digest = save_checkpoint(output / "terminal.pt", model, optimizer, samplers, FIT_UPDATES, identity)
        reloaded = torch.load(output / "terminal.pt", map_location="cpu", weights_only=False)
        if reloaded["identity"] != identity or reloaded["update"] != FIT_UPDATES or any(
                not torch.equal(value.cpu(), reloaded["model"][key]) for key, value in model.state_dict().items()):
            raise RuntimeError("terminal checkpoint reload differs from trained state")
        result.update(status="completed", terminal_reload_verified=True,
                      artifacts={"terminal.pt": digest, "updates.jsonl": base.sha256(output / "updates.jsonl")},
                      warmed_seconds_per_update=float(np.mean(durations[min(4, len(durations)-1):])),
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if torch.device(device).type == "cuda" else None,
                      training_seconds=time.perf_counter() - started, final_training_row=row)
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "result.json", result)
    return result


def probe_state(path):
    """Do not release the queue on a missing, failed or partial probe terminal."""
    result_file = path / "result.json"
    if not result_file.exists():
        return False
    try:
        result = json.loads(result_file.read_text())
    except json.JSONDecodeError:
        return False  # The existing probe writes this small progress file in place.
    if result.get("status") == "failed":
        raise RuntimeError("independent probe failed; queue stopped")
    if result.get("status") != "completed":
        return False
    identity = result["identity"]
    if (result.get("phase") != "fit" or result.get("updates_completed") != 49152
            or identity.get("run_id") != "CM_NEXT_KF_CLEAN_PROBE_20261001A"
            or identity.get("original_weights_loaded") is not False):
        raise ValueError("wrong probe dependency")
    terminal = path / "terminal.pt"
    if not terminal.exists() or base.sha256(terminal) != result["artifacts"].get("terminal.pt"):
        raise ValueError("probe terminal identity mismatch")
    saved = torch.load(terminal, map_location="cpu", weights_only=False)
    if saved["identity"] != identity or saved["update"] != 49152:
        raise ValueError("probe terminal reload mismatch")
    return True


def run(input_path, output, after_probe, device):
    input_path, output = Path(input_path).resolve(), Path(output).resolve()
    if output.is_relative_to(input_path) or (after_probe and output.is_relative_to(Path(after_probe).resolve())):
        raise ValueError("output must be outside immutable inputs and probe")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    state = dict(status="waiting_for_probe" if after_probe else "preparing", protocol=protocol(),
                 sources=sources(), training_runs=[], created_unix=time.time())
    write_json(output / "queue.json", state)
    try:
        wait_start = time.monotonic()
        while after_probe and not probe_state(Path(after_probe)):
            if time.monotonic() - wait_start > 8 * 3600:
                raise TimeoutError("probe did not complete within queue wait allowance")
            time.sleep(30)
        state["status"] = "preparing"
        state["started_unix"] = time.time()
        write_json(output / "queue.json", state)
        manifest, captured, original, train, input_hash = base.load_inputs(input_path)
        model = base.load_model(captured, original, device)
        base.terminal_replay(model, captured["teacher.npz"], device, len(base.TRAIN_SEEDS))
        prepare_inputs(train, model, manifest, input_hash, output / "displacements")
        del model
        gc.collect()
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
        _, arrays, prepared_hash = load_prepared(output / "displacements", train, manifest, input_hash)
        state["prepared_manifest_sha256"] = prepared_hash
        for condition in CONDITIONS:
            for target in TARGETS:
                name = condition + "_" + target
                state.update(status="training", current_run=name)
                write_json(output / "queue.json", state)
                result = train_condition(train, captured, original, input_hash, prepared_hash, arrays,
                                         condition, target, output / name, device)
                state["training_runs"].append(dict(name=name, status=result["status"], terminal_sha256=result["artifacts"]["terminal.pt"]))
                write_json(output / "queue.json", state)
                gc.collect()
                if torch.device(device).type == "cuda":
                    torch.cuda.empty_cache()
        if (base.sha256(input_path / "manifest.json") != input_hash or sources() != state["sources"]
                or any(base.sha256(input_path / name) != digest for name, digest in manifest["artifacts"].items())):
            raise RuntimeError("source/input identity changed during queue")
        state.update(status="completed", finished_unix=time.time())
        state.pop("current_run", None)
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "queue.json", state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--after-probe", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args.input, args.output, args.after_probe, args.device)


if __name__ == "__main__":
    main()
