"""October 4 midpoint and controlled four-step study; see EXPERIMENT_PLAN.md.

Run as a module with --run-root, --comparison-root, --probe-fit,
--followup-root, --output (a new directory), and --device cuda.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import numpy as np
import torch

from scripts.time_dependent_no import run_kolmogorov_corrective_followup as followup
from utility.time_dependent_no import corrective_objectives as objectives

base, previous, evaluation = followup.base, followup.previous, followup.evaluation
write_json, release = followup.write_json, followup.release
RUN_ID = "KF_MULTISTEP_20261004A"
DEPTH, BATCH, UPDATES, RESOURCE_UPDATES, WINDOWS_PER_PATH = 4, 4, 4096, 16, 32
MAX_TRAIN_SECONDS = 6*3600
METHODS = ("mixed_midpoint", "clean4", "fixed4", "detached4", "attached4")
BASELINES = ("pretrained", "clean_continuation", "clean_centers_u4096",
             "native_mixed_recovery", "native_mixed_dynamics")
SOURCE_NAMES = (*followup.SOURCE_NAMES,
    "scripts/time_dependent_no/evaluate_kolmogorov_corrective_rollouts.py",
    "scripts/time_dependent_no/evaluate_kolmogorov_local_defects.py",
    "scripts/time_dependent_no/evaluate_kolmogorov_recovery.py",
    "scripts/time_dependent_no/run_kolmogorov_multistep_study.py",
    "scripts/time_dependent_no/diagnose_kolmogorov_multistep.py",
    "scripts/time_dependent_no/plot_kolmogorov_multistep_study.py",
    "utility/time_dependent_no/corrective_objectives.py",
    "tests/time_dependent_no/test_kolmogorov_multistep_study.py")


def tensor(value, device="cpu"):
    return torch.from_numpy(np.array(value, dtype=np.float32, copy=True)).to(device)


class Study:
    def __init__(self, args):
        self.out, self.device = Path(args.output), args.device
        self.context = followup.Study(SimpleNamespace(
            run_root=args.run_root, comparison_root=args.comparison_root,
            probe_fit=args.probe_fit, pilot=Path(args.followup_root)/"evaluation",
            output=args.followup_root, device=args.device))
        self.context.registry = json.loads((Path(args.followup_root)/"models.json").read_text())
        for name in ("train", "train_hash", "scale", "arrays", "prepared_hash", "original"):
            setattr(self, name, getattr(self.context, name))
        self.sources = {**previous.sources(), **self.context.sources,
                        **{n: base.sha256(base.ROOT/n) for n in SOURCE_NAMES}}
        self.registry = {}

    def verify_training_inputs(self):
        packet = self.context.original_root/"input/train"
        prepared = self.context.comparison/"displacements"
        manifests = ((packet/"manifest.json", self.train_hash),
                     (prepared/"result.json", self.prepared_hash),
                     (self.out/"inputs/manifest.json", self.window_hash))
        for path, expected in manifests:
            if base.sha256(path) != expected:
                raise ValueError("training input manifest changed")
        for root, record in ((packet, self.context.manifest), (prepared, self.context.prepared)):
            for name, expected in record["artifacts"].items():
                if base.sha256(root/name) != expected:
                    raise ValueError("training input artifact changed")
        record = json.loads((self.out/"inputs/manifest.json").read_text())
        if base.sha256(self.out/"inputs/fixed_inputs.npy") != record["fixed_inputs_sha256"]:
            raise ValueError("frozen training inputs changed")

    def prepare_windows(self):
        out = self.out/"inputs"
        out.mkdir(parents=True, exist_ok=False)
        self.rows = objectives.sample_windows(len(self.train), self.train.shape[1], DEPTH,
                                              per_path=WINDOWS_PER_PATH)
        model, identity = self.context.old_model("pretrained")
        model.eval()
        shape = (len(self.rows), DEPTH, *self.train.shape[2:])
        fields = np.lib.format.open_memmap(out/"fixed_inputs.npy", mode="w+", dtype="float32", shape=shape)
        with torch.no_grad():
            for start in range(0, len(self.rows), BATCH):
                indices = np.arange(start, min(start+BATCH, len(self.rows)))
                state = objectives.sequence_batch(self.train, self.rows, indices, DEPTH)[:, 0].to(self.device)
                for j in range(DEPTH):
                    fields[indices, j] = state.cpu().numpy()
                    if j+1 < DEPTH:
                        state = model(state)["next_state"]
        fields.flush()
        del fields, model
        release()
        write_json(out/"manifest.json", dict(run_id=RUN_ID, depth=DEPTH, rows=self.rows,
            sampling_seed=2026100401, training_manifest_sha256=self.train_hash,
            generator=identity, fixed_inputs_sha256=base.sha256(out/"fixed_inputs.npy"), sources=self.sources))
        self.window_hash = base.sha256(out/"manifest.json")
        self.fixed_inputs = np.load(out/"fixed_inputs.npy", mmap_mode="r")

    def measure_gradients(self, model, out, position):
        # The same onset windows at every checkpoint; all come from training data.
        indices = [i for i, r in enumerate(self.rows) if r["start"] == 0][:BATCH]
        sequence = objectives.sequence_batch(self.train, self.rows, indices, DEPTH).to(self.device)
        values = objectives.gradient_diagnostics(model, sequence)
        path = out/f"gradients_{position:06d}.npz"
        np.savez_compressed(path, sequence_indices=indices, **values)
        return dict(file=path.name, sha256=base.sha256(path))

    def train_model(self, name, *, resource=False):
        if name not in METHODS:
            raise ValueError("unknown registered method")
        out = self.out/("resource" if resource else "training")/name
        out.mkdir(parents=True, exist_ok=False)
        model, initial = self.context.old_model("pretrained")
        model.train().requires_grad_(True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        midpoint = name == "mixed_midpoint"
        mode = None if midpoint else name[:-1]
        samplers = ([base.EpochSampler(len(self.train)*(self.train.shape[1]-1), 8, 17),
                     base.EpochSampler(len(self.arrays["centers"]), 8, 1701)] if midpoint else
                    [base.EpochSampler(len(self.rows), BATCH, 1701)])
        identity = dict(run_id=RUN_ID, name=name, initial=initial, sources=self.sources,
            training_manifest_sha256=self.train_hash, window_manifest_sha256=self.window_hash,
            prepared_manifest_sha256=self.prepared_hash, dynamics_weight=.5 if midpoint else None,
            recipe=dict(lr=1e-4, depth=None if midpoint else DEPTH,
                batch_size=8 if midpoint else BATCH, loss="half clean, half mixed target" if midpoint else "mean over four next states",
                sampler_seeds=[17, 1701] if midpoint else [1701], resource_only=resource))
        result = dict(status="running", identity=identity, updates_completed=0, checkpoints={}, gradients={})
        write_json(out/"result.json", result)
        steps = RESOURCE_UPDATES if resource else UPDATES
        tick, durations = time.monotonic(), []
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        try:
            result["gradients"]["0"] = self.measure_gradients(model, out, 0)
            with (out/"updates.jsonl").open("x") as log:
                for position in range(1, steps+1):
                    base._sync(self.device)
                    start = time.monotonic()
                    if midpoint:
                        clean, displaced = [s.next_indices() for s in samplers]
                        p, t = np.divmod(clean, self.train.shape[1]-1)
                        rec = tensor(self.arrays["clean_targets"][displaced])
                        dyn = tensor(self.arrays["native_mixed_dynamics"][displaced])
                        batches = ((tensor(self.train[p, t]), tensor(self.train[p, t+1])),
                                   (tensor(self.arrays["native_mixed_inputs"][displaced]),
                                    objectives.mixed_target(rec, dyn, .5)))
                        row = base.update(model, optimizer, batches, self.device)
                        constant = .125*float(((dyn.double()-rec.double())/self.scale).square().mean())
                        row.update(clean_indices=clean.tolist(), displaced_indices=displaced.tolist(),
                                   weighted_target_loss=row["loss"]+constant)
                    else:
                        indices = samplers[0].next_indices()
                        sequence = objectives.sequence_batch(self.train, self.rows, indices, DEPTH).to(self.device)
                        fixed = tensor(self.fixed_inputs[indices], self.device) if mode == "fixed" else None
                        row = objectives.sequence_update(model, optimizer, sequence, mode, fixed_inputs=fixed)
                        row["sequence_indices"] = indices.tolist()
                    base._sync(self.device)
                    durations.append(time.monotonic()-start)
                    row.update(update=position, seconds=durations[-1],
                               forward_calls=2 if midpoint else DEPTH,
                               state_predictions=16 if midpoint else BATCH*DEPTH)
                    log.write(json.dumps(row, allow_nan=False)+"\n")
                    result["updates_completed"] = position
                    if position % 128 == 0:
                        log.flush()
                        write_json(out/"result.json", result)
                        print(json.dumps(dict(training=name, update=position, loss=row["loss"])), flush=True)
                    if not resource and position in {1024, 2048, UPDATES}:
                        path = out/f"checkpoint_{position:06d}.pt"
                        digest = previous.save_checkpoint(path, model, optimizer, samplers, position, identity)
                        saved = torch.load(path, map_location="cpu", weights_only=False)
                        if saved["identity"] != identity or saved["update"] != position or any(
                                not torch.equal(v.cpu(), saved["model"][k]) for k, v in model.state_dict().items()):
                            raise ValueError("checkpoint reload differs from trained weights")
                        del saved
                        result["checkpoints"][str(position)] = dict(file=path.name, sha256=digest)
                        result["gradients"][str(position)] = self.measure_gradients(model, out, position)
                        write_json(out/"result.json", result)
                    if time.monotonic()-tick > MAX_TRAIN_SECONDS:
                        raise TimeoutError("training exceeded six-hour allowance")
            if resource:
                # Check diagnostics again with Adam state and parameter gradients
                # resident, as they are at a real training checkpoint.
                result["gradients"][str(steps)] = self.measure_gradients(model, out, steps)
            # Use warmed updates; separately include the cost of four diagnostic passes.
            warmed = durations[min(4, len(durations)-1):]
            diagnostic_seconds = max(0., time.monotonic()-tick-sum(durations))
            estimate = (float(np.median(warmed))*UPDATES+
                        4*diagnostic_seconds/len(result["gradients"]))
            result.update(status="completed", seconds=time.monotonic()-tick,
                estimated_training_seconds=estimate, within_budget=estimate <= MAX_TRAIN_SECONDS,
                peak_memory_bytes=torch.cuda.max_memory_allocated() if self.device == "cuda" else 0,
                updates_sha256=base.sha256(out/"updates.jsonl"))
            if not resource:
                terminal = result["checkpoints"][str(UPDATES)]
                self.registry[name] = dict(checkpoint=str((out/terminal["file"]).relative_to(self.out)),
                    sha256=terminal["sha256"], identity=identity, update=UPDATES)
                write_json(self.out/"models.json", self.registry)
        except Exception as error:
            result.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            write_json(out/"result.json", result)
            del model, optimizer
            release()
        return result

    def model(self, name):
        if name not in self.registry:
            return self.context.model(name)
        item = self.registry[name]
        path = self.out/item["checkpoint"]
        if base.sha256(path) != item["sha256"]:
            raise ValueError("checkpoint hash mismatch")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["identity"] != item["identity"] or saved["update"] != UPDATES:
            raise ValueError("checkpoint identity mismatch")
        model = evaluation.PeriodicVorticityPCNO(**self.original["model_config"], train_scale=self.scale)
        model.load_state_dict(saved["model"], strict=True)
        return model.to(self.device).eval().requires_grad_(False), item

    def evaluation_data(self):
        source = self.context.out/"evaluation"
        meta = json.loads((source/"result.json").read_text())
        expected_ids = ([f"train_{s}" for s in base.TRAIN_SEEDS[:8]]+
                        [f"validation_{s}" for s in followup.diagnostic.DEVELOPMENT_SEEDS])
        if (meta["status"] != "completed" or meta["training_manifest_sha256"] != self.train_hash
                or meta["horizon"] != 32 or meta["case_ids"] != expected_ids
                or meta["populations"] != ["train"]*8+["validation"]*4
                or meta["train_scale"] != self.scale):
            raise ValueError("evaluation population or normalization changed")
        if base.sha256(source/"ground_truth.npz") != meta["ground_truth_sha256"]:
            raise ValueError("ground-truth hash mismatch")
        out = self.out/"evaluation"
        out.mkdir()
        shutil.copy2(source/"ground_truth.npz", out/"ground_truth.npz")
        with np.load(out/"ground_truth.npz") as data:
            self.truths = data["ground_truth"]
        self.case_ids = meta["case_ids"]
        self.evaluation_state = dict(meta, run_id=RUN_ID, status="running", completed_methods=[],
            sources=self.sources, reused_evaluation_sha256=base.sha256(source/"result.json"))
        probe, probe_identity = self.context.old_model("seed29_pretrained")
        del probe
        release()
        for name in BASELINES:
            record = json.loads((source/name/"result.json").read_text())
            if record["probe_name"] != "seed29_pretrained" or record["probe_identity"] != probe_identity:
                raise ValueError("reused evaluation probe identity mismatch")
            model, identity = self.model(name)
            if record["status"] != "completed" or record["identity"] != identity:
                raise ValueError("baseline model identity mismatch")
            del model
            release()
            target = out/name
            target.mkdir()
            for file, digest in record["artifacts"].items():
                if base.sha256(source/name/file) != digest:
                    raise ValueError("baseline artifact hash mismatch")
                shutil.copy2(source/name/file, target/file)
            write_json(target/"result.json", record)
            self.evaluation_state["completed_methods"].append(name)
        write_json(out/"result.json", self.evaluation_state)

    def evaluate(self, name):
        out = self.out/"evaluation"/name
        out.mkdir()
        model, identity = self.model(name)
        probe, probe_identity = self.context.old_model("seed29_pretrained")
        scores, checks, artifacts = [], [], {}
        for case, truth in zip(self.case_ids, self.truths):
            score, fields, check, physical_names = evaluation.evaluate_case(
                model, probe, truth, self.scale, self.device)
            np.savez_compressed(out/(case+".npz"), **fields)
            artifacts[case+".npz"] = base.sha256(out/(case+".npz"))
            scores.append(score)
            checks.append(dict(check, case_id=case))
            print(json.dumps(dict(evaluating=name, case=case)), flush=True)
        np.savez_compressed(out/"metrics.npz", **{k: np.stack([s[k] for s in scores]) for k in scores[0]})
        artifacts["metrics.npz"] = base.sha256(out/"metrics.npz")
        write_json(out/"result.json", dict(status="completed", name=name, identity=identity,
            probe_name="seed29_pretrained", probe_identity=probe_identity, artifacts=artifacts,
            qualifications=checks, physical_names=list(physical_names)))
        self.evaluation_state["completed_methods"].append(name)
        write_json(out.parent/"result.json", self.evaluation_state)
        del model, probe
        release()

    def execute(self):
        self.prepare_windows()
        for name in ("attached4", "detached4", "fixed4", "clean4", "mixed_midpoint"):
            self.phase("resource", name)
            result = self.train_model(name, resource=True)
            if not result["within_budget"]:
                raise RuntimeError(f"{name} exceeds the training time allowance")
        self.phase("baseline_evaluation")
        self.evaluation_data()
        for name in METHODS:
            self.phase("input_verification", name)
            self.verify_training_inputs()
            self.phase("training", name)
            self.train_model(name)
            self.phase("evaluation", name)
            self.evaluate(name)
            self.plot()
        self.evaluation_state["status"] = "completed"
        write_json(self.out/"evaluation/result.json", self.evaluation_state)
        self.phase("diagnostics")
        from scripts.time_dependent_no.diagnose_kolmogorov_multistep import diagnose
        diagnose(self)
        self.plot()
        self.verify_training_inputs()

    def phase(self, phase, name=None):
        path = self.out/"queue.json"
        state = json.loads(path.read_text())
        state.update(phase=phase, current_method=name, updated_unix=time.time())
        write_json(path, state)

    def plot(self):
        from scripts.time_dependent_no.plot_kolmogorov_multistep_study import plot
        plot(self.out, self.out/"figures")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-root", "comparison-root", "probe-fit", "followup-root", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    out = args.output.resolve()
    for root in (args.run_root, args.comparison_root, args.probe_fit, args.followup_root):
        if out.is_relative_to(root.resolve()):
            raise ValueError("output must be outside immutable source results")
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    state = dict(status="initializing", run_id=RUN_ID, started_unix=time.time(),
                 planned_training_updates=len(METHODS)*UPDATES, validation_selection=False)
    write_json(out/"queue.json", state)
    try:
        study = Study(args)
        state.update(status="running", sources=study.sources)
        write_json(out/"queue.json", state)
        study.execute()
        if any(base.sha256(base.ROOT/n) != h for n, h in study.sources.items()):
            raise RuntimeError("source changed during study")
        state.update(status="completed", finished_unix=time.time())
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}", finished_unix=time.time())
        raise
    finally:
        current = json.loads((out/"queue.json").read_text())
        write_json(out/"queue.json", {**current, **state})


if __name__ == "__main__":
    main()
