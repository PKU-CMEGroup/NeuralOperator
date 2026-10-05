"""Complete the October 2 corrective-study controls without changing old runs.

Run with --run-root ORIGINAL_RUN --comparison-root SIX_RUN_OUTPUT
--probe-fit PROBE_FIT --pilot SAVED_PILOT --output NEW_DIRECTORY --device cuda.
The fixed queue owns target diagnostics, duration/coverage controls and the
original twelve-case evaluation. Its protocol is in EXPERIMENT_PLAN.md.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from scripts.time_dependent_no import evaluate_kolmogorov_corrective_rollouts as evaluation
from scripts.time_dependent_no import run_kolmogorov_displacement_comparison as previous

base = previous.base
diagnostic = evaluation.diagnostic
RUN_ID = "KF_CORRECTIVE_FOLLOWUP_20261002A"
HORIZON = 32
SOURCE_NAMES = (
    "scripts/time_dependent_no/run_kolmogorov_corrective_followup.py",
    "scripts/time_dependent_no/plot_kolmogorov_corrective_followup.py",
    "tests/time_dependent_no/test_kolmogorov_corrective_followup.py",
)


def write_json(path, value):
    previous.write_json(Path(path), value)


def row_mse(value, scale):
    value = np.asarray(value, dtype=np.float64)
    if not np.isfinite(value).all() or not np.isfinite(scale) or scale <= 0:
        raise ValueError("finite fields and positive training scale required")
    return np.mean(value**2, axis=(-2, -1)) / scale**2


def paired_scores(prediction, dynamics, clean_next, scale):
    prediction, dynamics, clean_next = [np.asarray(x, dtype=np.float64)
                                        for x in (prediction, dynamics, clean_next)]
    if prediction.shape != dynamics.shape or prediction.shape != clean_next.shape:
        raise ValueError("predictions and both targets must have identical shapes")
    d, q = prediction - dynamics, dynamics - clean_next
    scores = dict(rec_mse=row_mse(prediction-clean_next, scale),
                  dyn_mse=row_mse(d, scale), separation_mse=row_mse(q, scale),
                  cross_term=2*np.mean(d*q, axis=(-2, -1))/scale**2)
    if not np.allclose(scores["rec_mse"], scores["dyn_mse"] +
                       scores["separation_mse"] + scores["cross_term"], atol=1e-11, rtol=1e-10):
        raise ValueError("paired target identity does not close")
    return scores


def predict(model, states, device):
    states = np.asarray(states)
    outputs = []
    for start in range(0, len(states), 8):
        inputs = np.array(states[start:start+8], dtype=np.float32, copy=True)
        outputs.append(diagnostic.predict(model, inputs, device)[1])
    return np.concatenate(outputs)


def early_rows(path_count, transition_count=512):
    if transition_count < 48:
        raise ValueError("insufficient late-time strata")
    rng = np.random.default_rng(2026100201)
    rows = []
    for path in range(path_count):
        times = list(range(1, 8)) + sorted(rng.choice(np.arange(8, 32), 9, replace=False).tolist())
        times += [int(rng.choice(s)) for s in np.array_split(np.arange(32, transition_count), 16)]
        rows.extend(dict(path_index=path, input_step=t, mixed_length=1) for t in times)
    return rows


def profile_gaussian(center, gaussian, native_rms, original_expected_rms):
    if original_expected_rms <= 0 or native_rms < 0:
        raise ValueError("invalid amplitude profile")
    return (np.asarray(center, dtype=np.float64) +
            (np.asarray(gaussian, dtype=np.float64)-center)*native_rms/original_expected_rms).astype(np.float32)


def release(model=None):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class Study:
    def __init__(self, args):
        self.args, self.out = args, Path(args.output)
        self.original_root, self.comparison = Path(args.run_root), Path(args.comparison_root)
        self.probe_fit, self.pilot = Path(args.probe_fit), Path(args.pilot)
        self.device = args.device
        self.manifest, self.captured, self.original, self.train, self.train_hash = base.load_inputs(
            self.original_root / "input/train")
        self.prepared, self.arrays, self.prepared_hash = previous.load_prepared(
            self.comparison / "displacements", self.train, self.manifest, self.train_hash)
        self.scale = float(self.prepared["train_scale"])
        self.sources = {n: base.sha256(base.ROOT/n) for n in SOURCE_NAMES}
        self.registry = {}

    def old_model(self, name):
        if name in ("pretrained", "clean_continuation"):
            fitted = None if name == "pretrained" else self.original_root / "fit_clean_continuation"
            model, identity, _ = diagnostic.evaluation_model(
                self.captured, self.original, self.manifest, self.train_hash, fitted, self.device)
        else:
            probe = name == "seed29_pretrained"
            path = self.probe_fit if probe else self.comparison/name
            model, identity = evaluation.load_terminal(
                path, self.original, self.manifest, self.train_hash, self.device,
                probe=probe, prepared_hash=self.prepared_hash)
        return model, identity

    def model(self, name):
        if name in evaluation.METHODS or name == "seed29_pretrained":
            return self.old_model(name)
        item = self.registry[name]
        path = self.out/item["checkpoint"]
        if base.sha256(path) != item["sha256"]:
            raise ValueError("follow-up checkpoint hash mismatch")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["identity"] != item["identity"] or saved["update"] != item["update"]:
            raise ValueError("follow-up checkpoint identity mismatch")
        model = evaluation.PeriodicVorticityPCNO(
            **self.original["model_config"], train_scale=self.scale)
        model.load_state_dict(saved["model"], strict=True)
        model.to(self.device).eval().requires_grad_(False)
        return model, item

    def target_diagnostics(self, source_path, output_name, methods):
        out = self.out/output_name
        out.mkdir()
        source_path = Path(source_path)
        source = json.loads((source_path/"result.json").read_text())
        if source["status"] != "completed" or source["training_manifest_sha256"] != self.train_hash:
            raise ValueError("wrong completed pilot")
        if base.sha256(source_path/"ground_truth.npz") != source["ground_truth_sha256"]:
            raise ValueError("pilot truth hash mismatch")
        with np.load(source_path/"ground_truth.npz") as f:
            truth = f["ground_truth"].astype(np.float64)
        scores, requests, identities = {}, [], {}
        for name in methods:
            model, identity = self.model(name)
            identities[name] = identity
            clean = predict(model, truth[:, :-1].reshape(-1, 256, 256), self.device).reshape(truth[:, 1:].shape)
            scores[name+"__clean_mse"] = row_mse(clean.astype(float)-truth[:, 1:], self.scale)
            generators = ["seed29_pretrained" if name.startswith("seed29_") else "pretrained"]
            if "_recovery" in name or "_dynamics" in name:
                partner = name.replace("_recovery", "_dynamics") if "_recovery" in name else name.replace("_dynamics", "_recovery")
                generators += [n for n in (name, partner) if n in methods]
            for generator in dict.fromkeys(generators):
                record = json.loads((source_path/generator/"result.json").read_text())
                cases = []
                for i, case in enumerate(source["case_ids"]):
                    file = source_path/generator/(case+".npz")
                    if base.sha256(file) != record["artifacts"][file.name]:
                        raise ValueError("saved field hash mismatch")
                    with np.load(file) as f:
                        reached, labels = f["reached"], f["solver_next"]
                    y = reached[1:] if name == generator else predict(model, reached[:-1], self.device)
                    cases.append(paired_scores(y, labels, truth[i, 1:], self.scale))
                for key in cases[0]:
                    scores[name+"__"+generator+"__"+key] = np.stack([c[key] for c in cases])
                requests.append(dict(predictor=name, input_generator=generator))
            del model
            release()
        np.savez_compressed(out/"metrics.npz", **scores)
        write_json(out/"result.json", dict(status="completed", run_id=RUN_ID,
            case_ids=source["case_ids"], populations=source["populations"], requests=requests,
            train_scale=self.scale, macro_dt=.05, horizon=HORIZON, identities=identities,
            source_result_sha256=base.sha256(source_path/"result.json"), sources=self.sources,
            metrics_sha256=base.sha256(out/"metrics.npz")))
        print(output_name+" completed", flush=True)

    def original_data(self):
        return dict(rows=[dict(r, mixed_length=1) for r in self.prepared["rows"]], centers=self.arrays["centers"],
                    clean_targets=self.arrays["clean_targets"],
                    inputs=self.arrays["native_one_inputs"], dynamics=self.arrays["native_one_dynamics"],
                    identity=dict(prepared_manifest_sha256=self.prepared_hash, condition="native_one"))

    def prepare(self, condition):
        out = self.out/"inputs"/condition
        out.mkdir(parents=True)
        rows = early_rows(len(self.train), self.train.shape[1]-1) if condition == "early_one" else self.prepared["rows"]
        n = self.train.shape[-1]
        arrays = {key: np.lib.format.open_memmap(out/(key+".npy"), mode="w+", dtype=dtype,
                    shape=(len(rows), n, n)) for key, dtype in (
                        ("centers", "float32"), ("clean_targets", "float32"),
                        ("inputs", "float32"), ("dynamics", "float64"))}
        for i, row in enumerate(rows):
            p, t = row["path_index"], row["input_step"]
            arrays["centers"][i], arrays["clean_targets"][i] = self.train[p, t], self.train[p, t+1]
        if condition == "gaussian_profile":
            magnitudes = self.prepared["displacement_rms_scaled"]["native_mixed"]
            expected = self.prepared["gaussian_expected_rms_scaled"]
            for i in range(len(rows)):
                arrays["inputs"][i] = profile_gaussian(arrays["centers"][i], self.arrays["gaussian_inputs"][i], magnitudes[i], expected)
            generator_identity = dict(original_gaussian_manifest_sha256=self.prepared_hash)
        else:
            generator = "seed29_pretrained" if condition == "seed29_one" else "pretrained"
            model, generator_identity = self.old_model(generator)
            for start in range(0, len(rows), 8):
                selected = rows[start:start+8]
                arrays["inputs"][start:start+8] = previous.native_inputs(model, self.train, selected, False, self.device)
            del model
            release()
        solver = diagnostic.KolmogorovReferenceStepper(diagnostic.CONFIG)
        fine = diagnostic.KolmogorovReferenceStepper(replace(diagnostic.CONFIG,
                    dt_max=diagnostic.CONFIG.dt_max/2, cfl=diagnostic.CONFIG.cfl/2))
        norms = np.array([previous.rms(x.astype(float)-u)/self.scale for x, u in zip(arrays["inputs"], arrays["centers"])])
        checks = []
        refine = set(np.linspace(0, len(rows)-1, 16, dtype=int)) | {int(norms.argmax())}
        tick = time.monotonic()
        for i in range(len(rows)):
            x = arrays["inputs"][i].astype(float)
            canonical = solver.canonicalize(x)
            label = solver.advance_canonical(canonical).state
            arrays["dynamics"][i] = label
            clean = solver.advance_projected(arrays["centers"][i]).state
            check = dict(row=i, projection_rms_scaled=previous.rms(canonical-x)/self.scale,
                clean_restart_rms_scaled=previous.rms(clean-arrays["clean_targets"][i])/self.scale,
                label_rounding_rms_scaled=previous.rms(label.astype(np.float32).astype(float)-label)/self.scale)
            if i in refine:
                check["temporal_rms_scaled"] = previous.rms(fine.advance_canonical(canonical).state-label)/self.scale
            previous.check_metrics(check)
            checks.append(check)
            if (i+1) % 128 == 0:
                write_json(out/"progress.json", dict(condition=condition, labels=i+1, total=len(rows)))
            if time.monotonic()-tick > 4*3600:
                raise TimeoutError("input preparation exceeded four hours")
        artifacts = {}
        for key, array in arrays.items():
            array.flush()
            artifacts[key+".npy"] = base.sha256(out/(key+".npy"))
        identity = dict(status="completed", condition=condition, rows=rows, artifacts=artifacts,
            training_manifest_sha256=self.train_hash, generator_identity=generator_identity,
            displacement_rms_scaled=norms.tolist(), checks=checks, sources=self.sources)
        write_json(out/"result.json", identity)
        return dict(rows=rows, **arrays, identity=dict(prepared_manifest_sha256=base.sha256(out/"result.json"), condition=condition))

    def training_scores(self, model, data, output):
        model.eval()
        clean = predict(model, data["centers"], self.device)
        displaced = predict(model, data["inputs"], self.device)
        scores = paired_scores(displaced, data["dynamics"], data["clean_targets"], self.scale)
        scores["clean_mse"] = row_mse(clean.astype(float)-data["clean_targets"], self.scale)
        scores["input_step"] = np.array([r["input_step"] for r in data["rows"]])
        scores["length"] = np.array([r["mixed_length"] for r in data["rows"]])
        np.savez_compressed(output, **scores)
        model.train()
        return {k: float(np.sqrt(np.mean(v))) for k, v in scores.items() if k.endswith("mse")}

    def train_run(self, name, data, target, *, initial="pretrained", resume=None, steps=4096):
        out = self.out/"training"/name
        out.mkdir(parents=True)
        torch.manual_seed(17)
        model, initial_identity = self.old_model(resume or initial)
        model.train().requires_grad_(True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(.9, .999), eps=1e-8, weight_decay=0.)
        samplers = [base.EpochSampler(len(self.train)*(self.train.shape[1]-1), 8, 17),
                    base.EpochSampler(len(data["rows"]), 8, 1701)]
        start = 0
        if resume:
            checkpoint = torch.load(self.comparison/resume/"terminal.pt", map_location="cpu", weights_only=False)
            start = restore_training_state(checkpoint, model, optimizer, samplers)
            if start != 4096:
                raise ValueError("duration test must resume original update 4096")
        identity = dict(run_id=RUN_ID, name=name, target=target, initial=initial,
            resumed_method=resume, initial_identity=initial_identity, data=data["identity"],
            training_manifest_sha256=self.train_hash, model_config=self.original["model_config"],
            train_scale=self.scale, terminal_update=steps, sources=self.sources,
            recipe=dict(batch_size_per_branch=8, weights=[.5, .5], lr=1e-4,
                        sampler_seeds=[17, 1701], selection="fixed checkpoints; no validation selection"))
        result = dict(status="running", identity=identity, updates_completed=start, checkpoints={}, measurements={})
        write_json(out/"result.json", result)
        started = time.monotonic()
        result["measurements"][str(start)] = self.training_scores(model, data, out/f"training_{start:06d}.npz")
        tensor = lambda x: torch.from_numpy(np.array(x, dtype=np.float32, copy=True))
        checkpoints = {steps, 4096, 6144, 8192} if resume else {steps} | set(range(1024, steps+1, 1024))
        with (out/"updates.jsonl").open("x") as log:
            for position in range(start+1, steps+1):
                first, second = [s.next_indices() for s in samplers]
                p, t = np.divmod(first, self.train.shape[1]-1)
                x = data["centers"] if target == "clean" else data["inputs"]
                y = data["dynamics"] if target == "dynamics" else data["clean_targets"]
                batches = ((tensor(self.train[p, t]), tensor(self.train[p, t+1])),
                           (tensor(x[second]), tensor(y[second])))
                row = base.update(model, optimizer, batches, self.device)
                row.update(update=position, clean_indices=first.tolist(), displaced_indices=second.tolist())
                log.write(json.dumps(row, allow_nan=False)+"\n")
                result["updates_completed"] = position
                if position % 256 == 0:
                    log.flush()
                    write_json(out/"result.json", result)
                    print(json.dumps(dict(training=name, update=position, loss=row["loss"])), flush=True)
                if position in checkpoints:
                    file = out/f"checkpoint_{position:06d}.pt"
                    digest = previous.save_checkpoint(file, model, optimizer, samplers, position, identity)
                    saved = torch.load(file, map_location="cpu", weights_only=False)
                    if saved["identity"] != identity or saved["update"] != position or any(
                            not torch.equal(v.cpu(), saved["model"][k]) for k,v in model.state_dict().items()):
                        raise ValueError("new checkpoint reload differs from trained model")
                    del saved
                    measurement = self.training_scores(model, data, out/f"training_{position:06d}.npz")
                    result["checkpoints"][str(position)] = dict(file=file.name, sha256=digest)
                    result["measurements"][str(position)] = measurement
                    if position in (4096, 8192):
                        key = name+f"_u{position}"
                        self.registry[key] = dict(checkpoint=str(file.relative_to(self.out)), sha256=digest,
                            identity=identity, update=position, probe="pretrained" if initial == "seed29_pretrained" else "seed29_pretrained")
                    write_json(out/"result.json", result)
                if time.monotonic()-started > (steps-start)*3.0+900:
                    raise TimeoutError("training exceeded its measured-speed wall allowance")
        result.update(status="completed", seconds=time.monotonic()-started,
                      updates_sha256=base.sha256(out/"updates.jsonl"))
        write_json(out/"result.json", result)
        write_json(self.out/"models.json", self.registry)
        del model, optimizer
        release()

    def evaluation_data(self):
        _, validation, _, val_hash = diagnostic.load_evaluation(self.original_root/"input/development", self.manifest)
        truths = np.stack([np.array(x[:HORIZON+1], copy=True) for x in self.train[:8]] +
                          [np.array(x[:HORIZON+1], copy=True) for x in validation])
        ids = [f"train_{x}" for x in base.TRAIN_SEEDS[:8]] + [f"validation_{x}" for x in diagnostic.DEVELOPMENT_SEEDS]
        if len(ids) != 12 or len(truths) != 12:
            raise ValueError("evaluation must use the original eight/four population")
        self.truths, self.case_ids = truths, ids
        self.roles = ["train"]*8 + ["validation"]*4
        out = self.out/"evaluation"
        out.mkdir()
        self.evaluation_state = dict(status="running", run_id=RUN_ID, horizon=HORIZON, macro_dt=.05,
            case_ids=ids, populations=self.roles, train_scale=self.scale, completed_methods=[],
            training_manifest_sha256=self.train_hash, validation_manifest_sha256=val_hash,
            scope="original eight training and four validation trajectories", sources=self.sources)
        physical = np.stack([diagnostic.physical_diagnostics(t, diagnostic.CONFIG)[0] for t in truths])
        probes = {}
        for name in ("pretrained", "seed29_pretrained"):
            model, _ = self.old_model(name)
            clean = predict(model, truths[:, :-1].reshape(-1, 256, 256), self.device).reshape(truths[:, 1:].shape)
            probes[name+"_clean_mse"] = row_mse(clean.astype(float)-truths[:, 1:], self.scale)
            del model
            release()
        solver = diagnostic.KolmogorovReferenceStepper(diagnostic.CONFIG)
        checks = []
        for case, truth in zip(ids, truths):
            for n in diagnostic.CHECK_STEPS:
                row = dict(case_id=case, input_step=n,
                    clean_restart_rms_scaled=previous.rms(solver.advance_projected(truth[n]).state-truth[n+1])/self.scale)
                previous.check_metrics(row)
                checks.append(row)
        np.savez_compressed(out/"ground_truth.npz", ground_truth=truths, physical_truth=physical, **probes)
        self.evaluation_state.update(ground_truth_sha256=base.sha256(out/"ground_truth.npz"), clean_checks=checks)
        write_json(out/"result.json", self.evaluation_state)

    def evaluate(self, name):
        out = self.out/"evaluation"/name
        out.mkdir()
        self.evaluation_state.update(status="running", current_method=name)
        write_json(out.parent/"result.json", self.evaluation_state)
        model, identity = self.model(name)
        probe_name = "pretrained" if name.startswith("seed29_") else "seed29_pretrained"
        probe, probe_identity = self.old_model(probe_name)
        scores, checks, artifacts = [], [], {}
        old_result = None
        if name in evaluation.METHODS:
            old_result = json.loads((self.pilot/name/"result.json").read_text())
            old_metadata = json.loads((self.pilot/"result.json").read_text())
            if identity != old_result["identity"]:
                raise ValueError("pilot checkpoint changed")
        tick = time.monotonic()
        for case, truth in zip(self.case_ids, self.truths):
            file = case+".npz"
            if old_result and case in old_metadata["case_ids"]:
                index = old_metadata["case_ids"].index(case)
                if base.sha256(self.pilot/name/file) != old_result["artifacts"][file]:
                    raise ValueError("pilot reuse hash mismatch")
                shutil.copy2(self.pilot/name/file, out/file)
                with np.load(self.pilot/name/"metrics.npz") as f:
                    score = {key: f[key][index] for key in f.files}
                qualification = old_result["qualifications"][index]
                physical_names = old_result["physical_names"]
            else:
                score, fields, qualification, physical_names = evaluation.evaluate_case(
                    model, probe, truth, self.scale, self.device)
                np.savez_compressed(out/file, **fields)
            scores.append(score)
            checks.append(dict(qualification, case_id=case))
            artifacts[file] = base.sha256(out/file)
            print(json.dumps(dict(evaluating=name, case=case)), flush=True)
        np.savez_compressed(out/"metrics.npz", **{k: np.stack([s[k] for s in scores]) for k in scores[0]})
        artifacts["metrics.npz"] = base.sha256(out/"metrics.npz")
        write_json(out/"result.json", dict(status="completed", name=name, identity=identity,
            probe_name=probe_name, probe_identity=probe_identity, artifacts=artifacts,
            qualifications=checks, physical_names=list(physical_names), seconds=time.monotonic()-tick))
        self.evaluation_state["completed_methods"].append(name)
        self.evaluation_state.pop("current_method", None)
        write_json(out.parent/"result.json", self.evaluation_state)
        del model, probe
        release()

    def sampling_summary(self):
        source = self.pilot.parent/"training_investigation"
        receipt = json.loads((source/"result.json").read_text())
        if receipt["input_manifest_sha256"] != self.prepared_hash or receipt["status"] != "completed":
            raise ValueError("training-investigation receipt mismatch")
        if base.sha256(source/"metrics.npz") != receipt["metrics_sha256"]:
            raise ValueError("training investigation metrics changed")
        with np.load(source/"metrics.npz") as m:
            arrays = {c: np.asarray(v) for c, v in self.prepared["displacement_rms_scaled"].items()}
            arrays.update(input_step=np.array([r["input_step"] for r in self.prepared["rows"]]),
                length=np.array([r["mixed_length"] for r in self.prepared["rows"]]),
                rec_mse=m["native_mixed_recovery__native_mixed__rec_mse"],
                dyn_mse=m["native_mixed_dynamics__native_mixed__dyn_mse"])
        np.savez_compressed(self.out/"sampling.npz", **arrays)
        write_json(self.out/"sampling.json", dict(status="completed", prepared_manifest_sha256=self.prepared_hash,
            source_metrics_sha256=receipt["metrics_sha256"], metrics_sha256=base.sha256(self.out/"sampling.npz")))

    def execute(self):
        self.sampling_summary()
        self.target_diagnostics(self.pilot, "pilot_targets", evaluation.METHODS)
        self.plot()
        self.evaluation_data()
        for name in (*evaluation.METHODS, "seed29_pretrained"):
            self.evaluate(name)
        # This complete snapshot remains separate from later training results.
        self.evaluation_state["status"] = "completed"
        write_json(self.out/"evaluation/result.json", self.evaluation_state)
        self.target_diagnostics(self.out/"evaluation", "original_targets", evaluation.METHODS)
        self.plot()
        for name in evaluation.METHODS[2:]:
            condition, target = name.rsplit("_", 1)
            data = self.original_data()
            data.update(inputs=self.arrays[condition+"_inputs"], dynamics=self.arrays[condition+"_dynamics"])
            data["identity"] = dict(prepared_manifest_sha256=self.prepared_hash, condition=condition)
            if condition == "native_mixed":
                data["rows"] = self.prepared["rows"]
            self.train_run(name, data, target, resume=name, steps=8192)
            self.evaluate(name+"_u8192")
            self.plot()
        self.train_run("clean_centers", self.original_data(), "clean", steps=8192)
        for step in (4096, 8192):
            self.evaluate(f"clean_centers_u{step}")
        for condition, initial, targets in (
                ("early_one", "pretrained", ("clean", "recovery", "dynamics")),
                ("gaussian_profile", "pretrained", ("recovery", "dynamics")),
                ("seed29_one", "seed29_pretrained", ("clean", "recovery", "dynamics"))):
            data = self.prepare(condition)
            for target in targets:
                name = condition+"_"+target
                self.train_run(name, data, target, initial=initial)
                self.evaluate(name+"_u4096")
                self.plot()
            del data
            release()
        self.evaluation_state["status"] = "completed"
        write_json(self.out/"evaluation/result.json", self.evaluation_state)
        self.target_diagnostics(self.out/"evaluation", "all_targets", self.evaluation_state["completed_methods"])
        self.plot()

    def plot(self):
        from scripts.time_dependent_no.plot_kolmogorov_corrective_followup import plot
        plot(self.out, self.out/"figures")


def restore_training_state(saved, model, optimizer, samplers):
    model.load_state_dict(saved["model"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    if len(saved["samplers"]) != len(samplers):
        raise ValueError("sampler count changed")
    for sampler, state in zip(samplers, saved["samplers"]):
        sampler.load_state_dict(state)
    torch.set_rng_state(saved["torch_rng"])
    if torch.cuda.is_available() and saved["cuda_rng"]:
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
    return int(saved["update"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-root", "comparison-root", "probe-fit", "pilot", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    out = args.output.resolve()
    for root in (args.run_root/"input", args.comparison_root, args.probe_fit, args.pilot):
        if out.is_relative_to(root.resolve()):
            raise ValueError("output must be outside immutable source results")
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    state = dict(status="initializing", run_id=RUN_ID, started_unix=time.time(),
                 planned_training_updates=65536, validation_selection=False)
    write_json(out/"queue.json", state)
    try:
        study = Study(args)
        state.update(status="running", sources=study.sources)
        write_json(out/"queue.json", state)
        study.execute()
        if any(base.sha256(base.ROOT/n) != h for n, h in study.sources.items()):
            raise RuntimeError("source changed during follow-up")
        state.update(status="completed", finished_unix=time.time())
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}", finished_unix=time.time())
        raise
    finally:
        write_json(out/"queue.json", state)


if __name__ == "__main__":
    main()
