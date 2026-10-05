"""Offline target, empirical-geometry and response measurements for the study.

Called by run_kolmogorov_multistep_study after the five terminal evaluations.
The diagnostic fields never enter training or checkpoint selection.
"""
from __future__ import annotations

from dataclasses import replace
import json

import numpy as np

from scripts.time_dependent_no import run_kolmogorov_corrective_followup as followup
from utility.time_dependent_no import corrective_objectives as objectives


def training_bank(train, training_index, step):
    indices = np.arange(len(train))
    if training_index is not None:
        if not 0 <= training_index < len(train):
            raise ValueError("invalid excluded training trajectory")
        indices = indices[indices != training_index]
    return np.asarray(train[indices, step], dtype=float).reshape(len(indices), -1)


def response_queries(center, error, basis, scale):
    """Finite actual-size components and centered probes at two RMS amplitudes."""
    center, error = np.asarray(center, dtype=float), np.asarray(error, dtype=float)
    flat = error.ravel()
    parallel = ((flat@basis.T)@basis).reshape(error.shape)
    inputs, requests, deltas, unresolved = [center.astype(np.float32)], [], [], []
    for direction, displacement in (("actual_error", error), ("pca_parallel", parallel),
                                    ("pca_orthogonal", error-parallel)):
        rms = float(np.sqrt(np.mean(displacement**2)))
        if rms <= 1e-10*scale:
            unresolved.append(direction)
            continue
        for kind, amplitude in (("finite", rms/scale), ("central", .01), ("central", .005)):
            delta = displacement if kind == "finite" else displacement*(amplitude*scale/rms)
            positive = len(inputs)
            inputs.append((center+delta).astype(np.float32))
            negative = 0
            if kind == "central":
                negative = len(inputs)
                inputs.append((center-delta).astype(np.float32))
            requests.append(dict(direction=direction, kind=kind, amplitude=amplitude,
                                 positive=positive, negative=negative))
            deltas.append(delta)
    return np.stack(inputs), requests, np.asarray(deltas).reshape((-1, *center.shape)), unresolved


def response_vectors(outputs, requests):
    return np.stack([(outputs[r["positive"]].astype(float)-outputs[r["negative"]])*
                     (.5 if r["kind"] == "central" else 1.) for r in requests])


def score_responses(prediction, truth, fine, requests, deltas, scale):
    if not requests:
        values = {key: np.empty(0) for key in ("input_rms_scaled", "learned_gain", "true_gain",
            "response_error_gain", "cosine", "solver_refinement_relative",
            "learned_amplitude_relative", "true_amplitude_relative")}
        values["local_response_resolved"] = np.empty(0, dtype=bool)
        return values
    learned, actual, refined = [response_vectors(x, requests) for x in (prediction, truth, fine)]
    rows = []
    for y, z, f, delta in zip(learned, actual, refined, deltas):
        row = objectives.response_statistics(y, z, delta, scale)
        norm = float(np.linalg.norm(f))
        row["solver_refinement_relative"] = float(np.linalg.norm(f-z)/norm) if norm > 0 else np.nan
        rows.append(row)
    values = {key: np.array([r[key] for r in rows]) for key in rows[0]}
    # Consecutive central requests share the center and direction. Relative
    # agreement is descriptive; it does not certify a uniform derivative bound.
    for label, fields in (("learned", learned), ("true", actual)):
        agreement = np.full(len(requests), np.nan)
        for j, r in enumerate(requests[:-1]):
            if r["kind"] != "central" or r["amplitude"] != .01:
                continue
            other = requests[j+1]
            if other["kind"] != "central" or other["amplitude"] != .005:
                raise ValueError("central amplitude pairs must stay together")
            first, second = fields[j]/.01, fields[j+1]/.005
            norm = float(np.linalg.norm(second))
            agreement[j:j+2] = float(np.linalg.norm(first-second)/norm) if norm > 0 else np.nan
        values[label+"_amplitude_relative"] = agreement
    values["local_response_resolved"] = ((values["solver_refinement_relative"] <= .01) &
        (values["true_amplitude_relative"] <= .05) & (values["learned_amplitude_relative"] <= .05))
    return values


def read_fields(root, name, case):
    record = json.loads((root/name/"result.json").read_text())
    path = root/name/(case+".npz")
    if followup.base.sha256(path) != record["artifacts"][path.name]:
        raise ValueError("diagnostic field hash mismatch")
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def common_targets(study, out, names):
    root = study.out/"evaluation"
    fields = [read_fields(root, "pretrained", c) for c in study.case_ids]
    shape = study.truths[:, 1:].shape
    reached = np.stack([f["reached"][:-1] for f in fields])
    dynamics = np.stack([f["solver_next"] for f in fields])
    values = {}
    for name in names:
        model, _ = study.model(name)
        prediction = followup.predict(model, reached.reshape(-1, *shape[-2:]), study.device).reshape(shape)
        values.update({name+"__"+k: v for k, v in followup.paired_scores(
            prediction, dynamics, study.truths[:, 1:], study.scale).items()})
        del model
        followup.release()
    np.savez_compressed(out/"common_targets.npz", **values)


def geometry(study, out, names):
    root = study.out/"evaluation"
    scores = {name: [] for name in ("ground_truth", *names)}
    for i, case in enumerate(study.case_ids):
        paths = [study.truths[i], *[read_fields(root, n, case)["reached"] for n in names]]
        per_step = []
        for step in range(paths[0].shape[0]):
            bank = training_bank(study.train, i if i < 8 else None, step)
            queries = np.stack([p[step].ravel() for p in paths])
            per_step.append(objectives.empirical_geometry(bank, queries, scale=study.scale))
        for j, name in enumerate(scores):
            scores[name].append({k: np.array([s[k][j] for s in per_step]) for k in per_step[0]})
        print(json.dumps(dict(geometry=case)), flush=True)
    values = {name+"__"+k: np.stack([s[k] for s in cases])
              for name, cases in scores.items() for k in cases[0]}
    np.savez_compressed(out/"geometry.npz", **values)


def responses(study, out, names):
    requests, blocks, displacements, unresolved = [], [], [], []
    offset = 0
    for i, case in enumerate(study.case_ids):
        fields = read_fields(study.out/"evaluation", "pretrained", case)
        for step in (1, 4, 8, 16, 31):
            center = study.truths[i, step].astype(float)
            error = fields["reached"][step].astype(float)-center
            bank = training_bank(study.train, i if i < 8 else None, step)
            _, basis = objectives.pca_basis(bank, 16)
            inputs, rows, deltas, missing = response_queries(center, error, basis, study.scale)
            for row in rows:
                row.update(positive=row["positive"]+offset, negative=row["negative"]+offset,
                           case_id=case, input_step=step, population="train" if i < 8 else "validation",
                           effective_rank=len(basis))
            requests.extend(rows)
            blocks.append(inputs)
            displacements.append(deltas)
            unresolved.extend(dict(case_id=case, input_step=step, direction=d) for d in missing)
            offset += len(inputs)
    inputs, deltas = np.concatenate(blocks), np.concatenate(displacements)
    del blocks, displacements
    np.save(out/"response_inputs.npy", inputs)
    np.save(out/"response_displacements.npy", deltas)
    solver = followup.diagnostic.KolmogorovReferenceStepper(followup.diagnostic.CONFIG)
    fine = followup.diagnostic.KolmogorovReferenceStepper(replace(followup.diagnostic.CONFIG,
        dt_max=followup.diagnostic.CONFIG.dt_max/2, cfl=followup.diagnostic.CONFIG.cfl/2))
    truth = np.lib.format.open_memmap(out/"response_solver.npy", mode="w+", dtype="float64", shape=inputs.shape)
    refined = np.lib.format.open_memmap(out/"response_solver_fine.npy", mode="w+", dtype="float64", shape=inputs.shape)
    checks = []
    for i, state in enumerate(inputs):
        canonical = solver.canonicalize(state.astype(float))
        truth[i] = solver.advance_canonical(canonical).state
        refined[i] = fine.advance_canonical(canonical).state
        check = dict(input_row=i, projection_rms_scaled=followup.previous.rms(canonical-state)/study.scale,
            temporal_rms_scaled=followup.previous.rms(truth[i]-refined[i])/study.scale)
        followup.previous.check_metrics(check)
        checks.append(check)
        if (i+1) % 64 == 0:
            print(json.dumps(dict(response_solver_inputs=i+1, total=len(inputs))), flush=True)
    truth.flush(); refined.flush()
    values = {}
    for name in names:
        model, _ = study.model(name)
        prediction = followup.predict(model, inputs, study.device)
        values.update({name+"__"+k: v for k, v in score_responses(
            prediction, truth, refined, requests, deltas, study.scale).items()})
        del model, prediction
        followup.release()
    np.savez_compressed(out/"responses.npz", **values)
    followup.write_json(out/"response_requests.json", dict(requests=requests,
        unresolved=unresolved, solver_checks=checks,
        local_response_qualification=dict(solver_refinement_relative=.01, amplitude_agreement_relative=.05),
        interpretation="sampled responses along empirical PCA components; no exact normal or tangent identification"))


def diagnose(study):
    out = study.out/"diagnostics"
    out.mkdir()
    names = study.evaluation_state["completed_methods"]
    record = dict(status="running", methods=names, case_ids=study.case_ids,
        populations=study.evaluation_state["populations"], training_manifest_sha256=study.train_hash,
        source_evaluation_sha256=followup.base.sha256(study.out/"evaluation/result.json"),
        geometric_bank="same-time training states; evaluated training trajectory excluded",
        geometric_ranks=[8, 16], train_scale=study.scale, artifacts={})
    followup.write_json(out/"result.json", record)
    try:
        for name, operation in (("common_targets", common_targets), ("geometry", geometry), ("responses", responses)):
            record["current_operation"] = name
            followup.write_json(out/"result.json", record)
            operation(study, out, names)
            for path in out.iterdir():
                if path.name != "result.json" and path.name not in record["artifacts"]:
                    record["artifacts"][path.name] = followup.base.sha256(path)
            followup.write_json(out/"result.json", record)
        record.update(status="completed", current_operation=None)
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        followup.write_json(out/"result.json", record)
