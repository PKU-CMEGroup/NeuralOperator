"""Separate common-input assay and autonomous evaluation of the recovery pilot.

Run ``python -m scripts.time_dependent_no.evaluate_kolmogorov_recovery
--training-input TRAIN --evaluation-input OPEN_DEVELOPMENT --phase assay|rollout|replay|reached_response
--output NEW_DIRECTORY [--fit-output COMPLETED_FIT] --device cuda``.
Omitting --fit-output evaluates the untouched parent. Assay never composes a
trajectory; its targets are exact clean successors, not displaced solver labels.
Replay requires --prior-rollout and verifies its saved states exactly.
Reached-response requires --donor-replays containing replay_parent and
replay_recovery. It uses those frozen states, never a new autonomous trajectory.
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

from scripts.time_dependent_no import fit_kolmogorov_recovery as training
from scripts.time_dependent_no import fit_kolmogorov_unroll as unroll
from scripts.time_dependent_no import fit_kolmogorov_targets as targets
from scripts.time_dependent_no import evaluate_kolmogorov_clean_rollout as historical
from scripts.time_dependent_no.evaluate_kolmogorov_tangent_forecast import (
    ROOT, load_model, save_arrays, sha256, terminal_replay, write_json,
)
from utility.time_dependent_no.pcno_kolmogorov import PeriodicVorticityPCNO

DEVELOPMENT_SEEDS = list(range(2026090621, 2026090625))
INPUT_STEPS = (4, 5, 6, 31)
TIME_BANDS = ((0, 8), (8, 16), (16, 32), (32, 64), (64, 256), (256, 512))
HORIZONS = (8, 32, 128)
SNAPSHOTS = set(range(9)) | {31, 32, 63, 64, 127, 128}
AMPLITUDE_LIMIT = 1e6
REPLAY_HORIZON = 32
REACHED_STEPS = (4, 8, 12, 16, 31)
REACHED_AMPLITUDES = (0.01, 0.03, 0.1)
REACHED_GAUSSIAN_SEED = 90027
SOURCE_PATHS = unroll.SOURCE_PATHS | targets.SOURCE_PATHS | set(historical.SOURCE_PATHS) | {
    "scripts/time_dependent_no/evaluate_kolmogorov_recovery.py",
    "tests/time_dependent_no/test_evaluate_kolmogorov_recovery.py",
}


def load_evaluation(path, training_manifest):
    """Open only the named development population and frozen parent donors."""
    path = Path(path).resolve()
    raw = (path / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    expected = dict(schema_version=1, role="open_development",
                    development_seeds=DEVELOPMENT_SEEDS,
                    train_seeds=training.TRAIN_SEEDS[:8], input_steps=list(INPUT_STEPS),
                    parent_checkpoint_sha256=training_manifest["artifacts"]["checkpoint.pt"])
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError("evaluation population/donor scope mismatch")
    if set(manifest["artifacts"]) != {"development.npy", "donors.npz"}:
        raise ValueError("evaluation requires exactly development.npy and donors.npz")
    for name, digest in manifest["artifacts"].items():
        member = (path / name).resolve()
        if not member.is_relative_to(path) or sha256(member) != digest:
            raise ValueError(f"evaluation artifact hash mismatch: {name}")
    development = np.load(path / "development.npy", mmap_mode="r", allow_pickle=False)
    if (development.shape != (len(DEVELOPMENT_SEEDS), *training.TRAIN_SHAPE[1:])
            or development.dtype != np.float32):
        raise ValueError("development array shape/dtype mismatch")
    with np.load(path / "donors.npz", allow_pickle=False) as saved:
        required = {"train_seeds", "development_seeds", "input_steps",
                    "train_errors", "development_errors"}
        allowed = required | {"train_states", "development_states"}
        if not required <= set(saved.files) <= allowed:
            raise ValueError("unexpected donor array set")
        donors = {name: saved[name].copy() for name in saved.files}
    for key, value in (("train_seeds", training.TRAIN_SEEDS[:8]),
                       ("development_seeds", DEVELOPMENT_SEEDS),
                       ("input_steps", INPUT_STEPS)):
        if not np.array_equal(donors[key], value):
            raise ValueError("donor population/time index mismatch")
    for role, count in (("train", len(training.TRAIN_SEEDS[:8])),
                        ("development", len(DEVELOPMENT_SEEDS))):
        for suffix in ("errors", "states"):
            key = f"{role}_{suffix}"
            if suffix == "states" and key not in donors:
                continue
            array = donors[key]
            if (array.shape != (count, len(INPUT_STEPS), *training.TRAIN_SHAPE[-2:])
                    or array.dtype != np.float32 or not np.isfinite(array).all()):
                raise ValueError("donor shape/dtype/finite mismatch")
    return manifest, development, donors, hashlib.sha256(raw).hexdigest()


def evaluation_model(captured, parent_fit, train_manifest, train_hash, fit_output, device):
    if fit_output is None:
        model = load_model(captured, parent_fit, device)
        replay = terminal_replay(model, captured["teacher.npz"], device,
                                 len(training.TRAIN_SEEDS))
        return model, dict(arm="untouched_parent", checkpoint_sha256=
                           train_manifest["artifacts"]["checkpoint.pt"], replay=replay), {}
    path = Path(fit_output).resolve()
    result_bytes = (path / "result.json").read_bytes()
    result = json.loads(result_bytes)
    identity = result["identity"]
    if identity.get("recipe") == "fixed32_response_preserving_v1":
        from scripts.time_dependent_no import fit_kolmogorov_response_preserving as response_fit
        return response_fit.load_fitted(parent_fit, train_manifest, train_hash, path, device)
    parent_identity = parent_fit["checkpoint_identity"]
    is_unroll = identity.get("recipe") == unroll.RECIPE
    is_targets = identity.get("recipe") == targets.RECIPE
    fitter = targets if is_targets else unroll if is_unroll else training
    fit_updates, fit_sources = fitter.FIT_UPDATES, fitter.SOURCE_PATHS
    expected_recipe = (targets.RECIPE if is_targets else unroll.RECIPE if is_unroll
                       else "fixed32_gaussian_recovery_v1")
    arms = targets.ARMS if is_targets else unroll.ARMS if is_unroll else ("clean_continuation", "recovery")
    extra_hashes = {}
    recipe = dict(learning_rate=training.LEARNING_RATE,
                  batch_size_per_branch=training.BATCH_SIZE, branch_weights=[0.5, 0.5],
                  sampler_seeds=[17, 1701], noise_seed=1702,
                  noise_expected_rms_scaled=training.SIGMA if identity.get("arm") == "recovery" else 0.)
    if is_unroll:
        recipe = unroll.recipe_identity(train_manifest, parent_fit, train_hash, identity.get("arm"))
    if is_targets:
        bank_hash = sha256(path / "bank_result.json")
        if bank_hash != identity.get("bank_manifest_sha256") or bank_hash != result["artifacts"].get("bank_result.json"):
            raise ValueError("fit bank receipt hash mismatch")
        bank = json.loads((path / "bank_result.json").read_bytes())
        if (bank.get("status") != "completed" or bank.get("role") != "train"
                or bank.get("qualified") is not True or bank.get("recipe") != targets.bank.RECIPE
                or bank.get("training_manifest_sha256") != train_hash
                or bank.get("parent_checkpoint_sha256") != train_manifest["artifacts"]["checkpoint.pt"]):
            raise ValueError("fit bank receipt scope/qualification mismatch")
        recipe = targets.recipe_identity(train_manifest, parent_fit, train_hash, bank_hash, identity.get("arm"))
        extra_hashes["bank_result.json"] = bank_hash
    if (result.get("status") != "completed" or result.get("phase") != "fit"
            or result.get("updates_completed") != fit_updates
            or identity.get("updates") != fit_updates
            or identity.get("recipe") != expected_recipe
            or identity.get("arm") not in arms
            or identity.get("input_manifest_sha256") != train_hash
            or identity.get("parent_checkpoint_sha256") != train_manifest["artifacts"]["checkpoint.pt"]
            or identity.get("model_config") != parent_fit["model_config"]
            or result.get("input_manifest") != train_manifest
            or any(identity.get(k) != v for k, v in recipe.items())
            or any(identity.get(k) != parent_identity[k] for k in
                   ("train_scale_float64", "train_scale_model_float32"))):
        raise ValueError("fit identity/schedule/parent mismatch")
    if set(result["sources"]) != fit_sources:
        raise ValueError("fit source closure mismatch")
    for name, digest in result["sources"].items():
        member = (ROOT / name).resolve()
        if not member.is_relative_to(ROOT) or sha256(member) != digest:
            raise ValueError("fit source hash mismatch")
    checkpoint_bytes = (path / "terminal.pt").read_bytes()
    if hashlib.sha256(checkpoint_bytes).hexdigest() != result["artifacts"]["terminal.pt"]:
        raise ValueError("fit checkpoint hash mismatch")
    checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    if (checkpoint["identity"] != identity or checkpoint["update"] != fit_updates
            or checkpoint["schedule_position"] != fit_updates):
        raise ValueError("fit checkpoint identity/schedule mismatch")
    model = PeriodicVorticityPCNO(**identity["model_config"],
                                train_scale=identity["train_scale_float64"])
    model.load_state_dict(checkpoint["model"], strict=True)
    if float(model.train_scale) != identity["train_scale_model_float32"]:
        raise ValueError("fit checkpoint normalizer mismatch")
    model.to(device).eval().requires_grad_(False)
    hashes = {**extra_hashes, "result.json": hashlib.sha256(result_bytes).hexdigest(),
              "terminal.pt": hashlib.sha256(checkpoint_bytes).hexdigest()}
    return model, dict(identity=identity, **hashes), hashes


def predict(model, values, device):
    values = np.array(values, dtype=np.float32, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite evaluation input")
    with torch.no_grad():
        output = model(torch.from_numpy(values).to(device))
    raw, nxt = [output[key].detach().cpu().numpy().copy()
                for key in ("raw_next", "next_state")]
    if any(a.shape != values.shape or a.dtype != np.float32 for a in (raw, nxt)):
        raise ValueError("deployed model shape/dtype changed")
    if not all(np.isfinite(a).all() for a in (raw, nxt)):
        raise FloatingPointError("nonfinite prediction")
    return raw, nxt


def squared(value):
    return float(np.square(np.asarray(value, dtype=np.float64)).sum())


def target_assay(model, bank_result, arrays, scale, device):
    """Score identical inputs against both targets, without composing a path."""
    centers, predicted = [], []
    for start in range(0, len(arrays["centers"]), training.BATCH_SIZE):
        centers.append(predict(model, arrays["centers"][start:start+training.BATCH_SIZE], device)[1])
    for start in range(0, len(arrays["inputs"]), training.BATCH_SIZE):
        predicted.append(predict(model, arrays["inputs"][start:start+training.BATCH_SIZE], device)[1])
    centers, predicted = np.concatenate(centers), np.concatenate(predicted)
    rows = []
    for i, metadata in enumerate(bank_result["rows"]):
        a = metadata["anchor"]
        center, y = centers[a].astype(np.float64), predicted[i].astype(np.float64)
        clean, dynamics = arrays["clean_targets"][a].astype(np.float64), arrays["dynamics_targets"][i]
        response, trusted = y - center, dynamics - arrays["solver_clean"][a]
        eta = targets.bank.rms(arrays["inputs"][i].astype(np.float64) - arrays["centers"][a])
        vectors = dict(clean_forcing=center-clean, recovery_error=y-clean,
                       dynamics_defect=y-dynamics, response=response, trusted_response=trusted,
                       response_defect=response-trusted, target_difference=dynamics-clean)
        row = dict(bank_result["anchors"][a], **metadata)
        row.update({k + "_rms_scaled": targets.bank.rms(v) / scale for k, v in vectors.items()})
        row.update(response_gain=targets.bank.rms(response) / eta,
                   trusted_response_gain=targets.bank.rms(trusted) / eta,
                   response_defect_over_displacement=targets.bank.rms(response-trusted) / eta,
                   forcing_response_alignment_scaled=float(np.mean((center-clean)*response)) / scale**2)
        rows.append(row)
    return dict(center_prediction=centers, displaced_prediction=predicted), rows


def relative(sse, denominator):
    return float(np.sqrt(sse / denominator)) if denominator > 0 else None


def pair_metrics(clean, positive, negative, target, plus_input, minus_input, center, scale):
    """Exact antithetic algebra, including finite-even response and alignment."""
    clean, positive, negative, target, plus_input, minus_input, center = [
        np.asarray(x, dtype=np.float64) for x in
        (clean, positive, negative, target, plus_input, minus_input, center)]
    b = clean - target
    odd = (positive - negative) / 2
    even = (positive + negative) / 2 - clean
    combined = b + even
    loss = (squared(positive - target) + squared(negative - target)) / 2
    factor = center.size * scale**2
    odd_input = (plus_input - minus_input) / 2
    p_input, m_input = squared(plus_input - center), squared(minus_input - center)
    return dict(forcing_rms_scaled=np.sqrt(squared(b) / factor),
                odd_rms_scaled=np.sqrt(squared(odd) / factor),
                even_rms_scaled=np.sqrt(squared(even) / factor),
                bias_plus_even_rms_scaled=np.sqrt(squared(combined) / factor),
                paired_mse_scaled=loss / factor,
                positive_mse_scaled=squared(positive - target) / factor,
                negative_mse_scaled=squared(negative - target) / factor,
                signed_alignment_scaled=2 * float(np.sum(combined * odd)) / factor,
                antithetic_closure_scaled=(loss - squared(combined) - squared(odd)) / factor,
                positive_closure_scaled=(squared(positive - target) - squared(combined)
                                        - squared(odd) - 2 * np.sum(combined * odd)) / factor,
                positive_input_rms_scaled=np.sqrt(p_input / factor),
                negative_input_rms_scaled=np.sqrt(m_input / factor),
                input_centering_rms_scaled=np.sqrt(squared((plus_input + minus_input) / 2 - center) / factor),
                positive_response_gain=relative(squared(positive - clean), p_input),
                negative_response_gain=relative(squared(negative - clean), m_input),
                odd_response_gain=relative(squared(odd), squared(odd_input)))


def teacher_assay(model, populations, device):
    arrays, summary = {}, []
    for role, values in populations.items():
        paths, states = values.shape[:2]
        count = paths * (states - 1)
        rows = {key: np.empty(count, dtype=np.float64) for key in
                ("sse", "raw_sse", "target_sse", "restriction_sse", "mean_error",
                 "energy_error", "enstrophy_error")}
        for start in range(0, count, training.BATCH_SIZE):
            ids = np.arange(start, min(start + training.BATCH_SIZE, count))
            p, n = np.divmod(ids, states - 1)
            target = np.array(values[p, n + 1], dtype=np.float64)
            raw, nxt = predict(model, values[p, n], device)
            for j, index in enumerate(ids):
                a, b = raw[j].astype(np.float64), nxt[j].astype(np.float64)
                rows["sse"][index] = squared(b - target[j])
                rows["raw_sse"][index] = squared(a - target[j])
                rows["target_sse"][index] = squared(target[j])
                rows["restriction_sse"][index] = squared(a - b)
                pred, truth = [historical.field_diagnostics(v) for v in (b, target[j])]
                for key, source in (("mean_error", "mean_vorticity"),
                                    ("energy_error", "kinetic_energy"),
                                    ("enstrophy_error", "enstrophy")):
                    rows[key][index] = pred[source] - truth[source]
        for key, value in rows.items():
            arrays[f"{role}_{key}"] = value.reshape(paths, states - 1)
        for lo, hi in TIME_BANDS:
            if hi > states - 1:
                continue
            selected = {key: val.reshape(paths, states - 1)[:, lo:hi] for key, val in rows.items()}
            denominator = float(selected["target_sse"].sum())
            summary.append(dict(role=role, input_start=lo, input_stop=hi, pairs=paths * (hi - lo),
                                relative_l2=relative(float(selected["sse"].sum()), denominator),
                                raw_relative_l2=relative(float(selected["raw_sse"].sum()), denominator)))
    return arrays, summary


def response_assay(model, populations, donors, scale, device):
    generator = torch.Generator(device="cpu").manual_seed(90017)
    saved, rows = {}, []
    for role, values in populations.items():
        seeds = training.TRAIN_SEEDS[:8] if role == "train" else DEVELOPMENT_SEEDS
        for path_index, seed in enumerate(seeds):
            for anchor_index, n in enumerate(INPUT_STEPS):
                center, target = np.array(values[path_index, n:n + 2], copy=True)
                clean_raw, clean = predict(model, center[None], device)
                gaussian = training.gaussian_noise((1, *center.shape), scale, generator)[0].numpy()
                for kind, direction in (("gaussian", gaussian),
                                        ("native_parent", donors[f"{role}_errors"][path_index, anchor_index])):
                    plus, minus = center + direction, center - direction
                    native_rounding = 0.0
                    if kind == "native_parent" and f"{role}_states" in donors:
                        actual = donors[f"{role}_states"][path_index, anchor_index]
                        if not np.array_equal(actual - center, direction):
                            raise ValueError("native donor state/error disagreement")
                        native_rounding = np.sqrt(squared(plus.astype(np.float64) - actual) / center.size) / scale
                        plus = actual.copy()
                    raw, nxt = predict(model, np.stack([plus, minus]), device)
                    row = dict(role=role, seed=seed, input_step=n, direction=kind,
                               native_reconstruction_rms_scaled=float(native_rounding),
                               **pair_metrics(clean[0], nxt[0], nxt[1], target, plus, minus, center, scale))
                    row["raw"] = pair_metrics(clean_raw[0], raw[0], raw[1], target,
                                              plus, minus, center, scale)
                    row["restriction_rms_scaled"] = [
                        np.sqrt(squared(a.astype(np.float64) - b) / center.size) / scale
                        for a, b in ((clean_raw[0], clean[0]), (raw[0], nxt[0]), (raw[1], nxt[1]))]
                    rows.append(row)
                    for key, value in dict(center=center, target=target, positive_input=plus,
                                           negative_input=minus, clean_raw=clean_raw[0], clean=clean[0],
                                           positive_raw=raw[0], negative_raw=raw[1],
                                           positive=nxt[0], negative=nxt[1]).items():
                        saved.setdefault(key, []).append(value.copy())
    return {key: np.stack(value) for key, value in saved.items()}, rows


def rollout_case(model, reference, scale, device):
    """Exact FP32 next_state recurrence, sparse snapshots, honest stop censoring."""
    current = np.array(reference[0], copy=True)
    snapshots = {"step": [0], "state": [current.copy()], "raw": [current.copy()],
                 "reference": [current.copy()]}
    rows, status, failed = [], "completed", None
    for n in range(1, max(HORIZONS) + 1):
        try:
            raw, nxt = predict(model, current[None], device)
        except FloatingPointError:
            status, failed = "nonfinite_prediction", n
            break
        current = nxt[0]
        target = np.asarray(reference[n], dtype=np.float64)
        pred = current.astype(np.float64)
        structure, truth_structure = [historical.field_diagnostics(v) for v in (pred, target)]
        row = dict(step=n, sse=squared(pred - target), target_sse=squared(target),
                   raw_sse=squared(raw[0].astype(np.float64) - target),
                   restriction_sse=squared(raw[0].astype(np.float64) - pred),
                   rms_scaled=float(np.sqrt(np.mean(pred**2)) / scale),
                   structure=structure, truth_structure=truth_structure,
                   structure_error={k: structure[k] - truth_structure[k] for k in structure})
        rows.append(row)
        stopped = row["rms_scaled"] > AMPLITUDE_LIMIT
        if n in SNAPSHOTS or stopped:
            for key, value in (("step", n), ("state", current), ("raw", raw[0]),
                               ("reference", reference[n])):
                snapshots[key].append(value.copy() if isinstance(value, np.ndarray) else value)
        if stopped:
            status, failed = "amplitude_limit", n
            break
    horizons = []
    for h in HORIZONS:
        complete = len(rows) >= h and (failed is None or failed > h)
        sse = sum(r["sse"] for r in rows[:h]) if complete else None
        target_sse = sum(r["target_sse"] for r in rows[:h]) if complete else None
        horizons.append(dict(horizon=h, complete=complete, sse=sse, target_sse=target_sse,
                             relative_l2=relative(sse, target_sse) if complete else None))
    return dict(status=status, failed_at_step=failed, completed_steps=len(rows),
                rows=rows, horizons=horizons), {k: np.asarray(v) for k, v in snapshots.items()}


def pooled_horizons(cases):
    output = []
    for role in ("train", "development"):
        selected = [case for case in cases if case["role"] == role]
        for h in HORIZONS:
            values = [next(x for x in c["horizons"] if x["horizon"] == h) for c in selected]
            complete = bool(values) and all(v["complete"] for v in values)
            output.append(dict(role=role, horizon=h, paths=len(values),
                               complete_paths=sum(v["complete"] for v in values),
                               relative_l2=relative(sum(v["sse"] for v in values),
                                                    sum(v["target_sse"] for v in values)) if complete else None))
    return output


def checkpoint_hash(identity):
    return identity.get("checkpoint_sha256", identity.get("terminal.pt"))


def bound_evaluation(path, phase, train_hash, evaluation_hash):
    path = Path(path)
    raw = (path / "result.json").read_bytes()
    record = json.loads(raw)
    if (record.get("status") != "completed" or record.get("phase") != phase
            or record.get("training_manifest_sha256") != train_hash
            or record.get("evaluation_manifest_sha256") != evaluation_hash):
        raise ValueError("prior evaluation scope/phase mismatch")
    return record, {path / "result.json": hashlib.sha256(raw).hexdigest()}


def replay_reached_states(model, populations, scale, device, prior, prior_path):
    """Replay the old recurrence, retaining every state and checking old anchors."""
    arrays, rows, bindings = {}, [], {}
    for role, values in populations.items():
        seeds = training.TRAIN_SEEDS[:8] if role == "train" else DEVELOPMENT_SEEDS
        dense = []
        for index, seed in enumerate(seeds):
            current = np.array(values[index, 0], copy=True)
            states = [current.copy()]
            for n in range(1, REPLAY_HORIZON + 1):
                _, nxt = predict(model, current[None], device)
                current = nxt[0]
                if np.sqrt(squared(current) / current.size) / scale > AMPLITUDE_LIMIT:
                    raise FloatingPointError("dense replay reached amplitude guard")
                states.append(current.copy())
            states = np.stack(states)
            name = f"rollout_{role}_{seed}.npz"
            member = Path(prior_path) / name
            if sha256(member) != prior["artifacts"][name]:
                raise ValueError("prior rollout snapshot hash mismatch")
            bindings[member] = prior["artifacts"][name]
            with np.load(member, allow_pickle=False) as saved:
                selected = saved["step"] <= REPLAY_HORIZON
                steps = saved["step"][selected]
                if (steps[0] != 0 or steps[-1] != REPLAY_HORIZON
                        or not np.array_equal(saved["reference"][selected],
                                              values[index, steps])
                        or not np.array_equal(saved["state"][selected], states[steps])):
                    raise ValueError("dense replay differs from prior rollout")
            rows.append(dict(role=role, seed=seed, replayed_steps=REPLAY_HORIZON,
                             verified_steps=steps.tolist(), bitwise_equal=True))
            dense.append(states)
        arrays[role] = np.stack(dense)
    return arrays, rows, bindings


def load_reached_donors(root, train_hash, evaluation_hash, parent_hash, sources):
    """Only completed parent/recovery replays of the same permitted population."""
    donors, bindings, identities = {}, {}, {}
    for kind in ("parent", "recovery"):
        path = Path(root) / f"replay_{kind}"
        record, receipt = bound_evaluation(path, "replay", train_hash, evaluation_hash)
        identity = record["model"]
        valid_model = (checkpoint_hash(identity) == parent_hash if kind == "parent"
                       else identity.get("identity", {}).get("arm") == "recovery"
                       and identity["identity"]["parent_checkpoint_sha256"] == parent_hash)
        if (not valid_model or record.get("sources") != sources
                or record.get("replay_steps") != list(range(REPLAY_HORIZON + 1))
                or record.get("replay_seeds") != dict(train=training.TRAIN_SEEDS[:8],
                                                       development=DEVELOPMENT_SEEDS)
                or set(record["artifacts"]) != {"dense.npz"}):
            raise ValueError("reached donor model/source/population mismatch")
        member = path / "dense.npz"
        if sha256(member) != record["artifacts"]["dense.npz"]:
            raise ValueError("reached donor array hash mismatch")
        with np.load(member, allow_pickle=False) as saved:
            if set(saved.files) != {"train", "development"}:
                raise ValueError("unexpected reached donor arrays")
            donor = {key: saved[key].copy() for key in saved.files}
        for role, count in (("train", len(training.TRAIN_SEEDS[:8])),
                            ("development", len(DEVELOPMENT_SEEDS))):
            if (donor[role].shape != (count, REPLAY_HORIZON + 1, *training.TRAIN_SHAPE[-2:])
                    or donor[role].dtype != np.float32 or not np.isfinite(donor[role]).all()):
                raise ValueError("reached donor shape/dtype/finite mismatch")
        donors[kind] = donor
        identities[kind] = dict(checkpoint_sha256=checkpoint_hash(identity),
                                 result_sha256=receipt[path / "result.json"])
        bindings.update(receipt)
        bindings[member] = record["artifacts"]["dense.npz"]
    return donors, identities, bindings


def reached_probes(center, parent, recovery, gaussian, scale):
    """Common physical inputs; native endpoints are exact, zeros have no direction."""
    center64 = np.asarray(center, dtype=np.float64)
    states = dict(parent=parent, recovery=recovery)
    directions = {key: np.asarray(value, dtype=np.float64) - center64
                  for key, value in states.items()}
    directions["gaussian"] = np.asarray(gaussian, dtype=np.float64)
    norms = {key: np.sqrt(squared(value) / center.size) for key, value in directions.items()}
    amplitudes = [(f"fixed_{a:g}", a * scale) for a in REACHED_AMPLITUDES]
    amplitudes += [(f"native_{key}", norms[key]) for key in states]
    probes, skipped = [], []
    for direction, delta in directions.items():
        if norms[direction] == 0:
            skipped.append(direction)
            continue
        for label, amplitude in amplitudes:
            displacement = delta * (amplitude / norms[direction])
            plus = (center64 + displacement).astype(np.float32)
            minus = (center64 - displacement).astype(np.float32)
            if direction in states and label == f"native_{direction}":
                plus = np.array(states[direction], copy=True)
            digest = hashlib.sha256(np.stack([plus, minus]).tobytes()).hexdigest()
            probes.append((dict(direction=direction, amplitude=label,
                                requested_rms_scaled=float(amplitude / scale),
                                donor_rms_scaled=float(norms[direction] / scale),
                                input_sha256=digest), plus, minus))
    return probes, skipped


def reached_response_assay(model, populations, donors, scale, device):
    generator = torch.Generator(device="cpu").manual_seed(REACHED_GAUSSIAN_SEED)
    rows, skipped, saved, vector_rows = [], [], {}, []
    for role, values in populations.items():
        seeds = training.TRAIN_SEEDS[:8] if role == "train" else DEVELOPMENT_SEEDS
        for index, seed in enumerate(seeds):
            for n in REACHED_STEPS:
                center, target = np.array(values[index, n:n + 2], copy=True)
                gaussian = training.gaussian_noise((1, *center.shape), scale, generator)[0].numpy()
                probes, zero = reached_probes(
                    center, donors["parent"][role][index, n],
                    donors["recovery"][role][index, n], gaussian, scale)
                skipped += [dict(role=role, seed=seed, input_step=n, direction=d) for d in zero]
                clean_raw, clean = predict(model, center[None], device)
                for metadata, plus, minus in probes:
                    raw, nxt = predict(model, np.stack([plus, minus]), device)
                    row = dict(role=role, seed=seed, input_step=n, **metadata,
                               **pair_metrics(clean[0], nxt[0], nxt[1], target,
                                              plus, minus, center, scale))
                    row["raw"] = pair_metrics(clean_raw[0], raw[0], raw[1], target,
                                              plus, minus, center, scale)
                    # First path per role, each time and direction at its native
                    # amplitude (Gaussian at 0.01); all scalar rows are retained.
                    label = ("fixed_0.01" if metadata["direction"] == "gaussian"
                             else "native_" + metadata["direction"])
                    if index == 0 and metadata["amplitude"] == label:
                        vector_rows.append(len(rows))
                        for key, value in dict(
                                center=center, target=target, positive_input=plus,
                                negative_input=minus, clean=clean[0], clean_raw=clean_raw[0],
                                positive=nxt[0], negative=nxt[1],
                                positive_raw=raw[0], negative_raw=raw[1]).items():
                            saved.setdefault(key, []).append(value.copy())
                    rows.append(row)
            print(json.dumps(dict(phase="reached_response", role=role, seed=seed,
                                  rows_completed=len(rows))), flush=True)
    arrays = {key: np.stack(value) for key, value in saved.items()}
    arrays["row_indices"] = np.asarray(vector_rows, dtype=np.int64)
    return arrays, rows, skipped


def run(training_input, evaluation_input, output_path, phase, device="cuda", fit_output=None,
        prior_rollout=None, donor_replays=None, target_bank=None, bank_role=None,
        model_loader=None, source_paths=None):
    if phase not in ("assay", "rollout", "replay", "reached_response", "target_assay"):
        raise ValueError("unknown evaluation phase")
    if (phase == "replay") != (prior_rollout is not None):
        raise ValueError("only replay requires prior_rollout")
    if (phase == "reached_response") != (donor_replays is not None):
        raise ValueError("only reached_response requires donor_replays")
    if ((phase == "target_assay") != (target_bank is not None)
            or (phase == "target_assay" and bank_role not in ("train", "probe"))
            or (phase != "target_assay" and bank_role is not None)):
        raise ValueError("only target_assay requires target_bank and bank_role")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest, captured, parent_fit, train, train_hash = training.load_inputs(training_input)
    evaluation_manifest, development, donors, evaluation_hash = load_evaluation(evaluation_input, manifest)
    source_paths = SOURCE_PATHS if source_paths is None else source_paths
    model_loader = evaluation_model if model_loader is None else model_loader
    sources = {name: sha256(ROOT / name) for name in sorted(source_paths)}
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=False)
    result = dict(schema_version=1, status="running", phase=phase, artifacts={}, sources=sources,
                  training_manifest_sha256=train_hash, evaluation_manifest_sha256=evaluation_hash,
                  input_manifests=dict(training=manifest, evaluation=evaluation_manifest),
                  information="open development; reused starts; no new population confirmation",
                  device=str(device), torch_version=str(torch.__version__))
    extra_bindings = {}
    started = time.perf_counter()
    write_json(output / "result.json", result)
    try:
        model, identity, fit_hashes = model_loader(captured, parent_fit, manifest,
                                                 train_hash, fit_output, device)
        result["model"] = identity
        scale = float(model.train_scale)
        result["train_scale"] = scale
        populations = dict(train=train, development=development)
        if phase == "assay":
            teacher, summary = teacher_assay(model, populations, device)
            result["artifacts"]["teacher.npz"] = save_arrays(output / "teacher.npz", teacher)
            probes, rows = response_assay(model, populations, donors, scale, device)
            result.update(teacher_bands=summary, responses=rows, gaussian_seed=90017,
                          gaussian_expected_rms_scaled=training.SIGMA, recurrent_model_calls=0,
                          response_array_row_order="same order as responses")
            result["artifacts"]["responses.npz"] = save_arrays(output / "responses.npz", probes)
        elif phase == "target_assay":
            bank_result, arrays, bank_hash = targets.bank.load_bank(
                target_bank, bank_role, train_hash, manifest["artifacts"]["checkpoint.pt"])
            if bank_role == "probe" and bank_result["evaluation_manifest_sha256"] != evaluation_hash:
                raise ValueError("target probe development binding mismatch")
            vectors, rows = target_assay(model, bank_result, arrays, scale, device)
            result["artifacts"]["target_predictions.npz"] = save_arrays(output / "target_predictions.npz", vectors)
            result.update(target_responses=rows, bank_role=bank_role, bank_manifest_sha256=bank_hash,
                          recurrent_model_calls=0, response_array_row_order="same order as bank rows")
            extra_bindings[Path(target_bank) / "result.json"] = bank_hash
            extra_bindings.update({Path(target_bank) / k: v for k, v in bank_result["artifacts"].items()})
        elif phase == "replay":
            prior, extra_bindings = bound_evaluation(
                prior_rollout, "rollout", train_hash, evaluation_hash)
            if checkpoint_hash(prior["model"]) != checkpoint_hash(identity):
                raise ValueError("prior rollout checkpoint mismatch")
            arrays, rows, bindings = replay_reached_states(
                model, populations, scale, device, prior, prior_rollout)
            extra_bindings.update(bindings)
            result["artifacts"]["dense.npz"] = save_arrays(output / "dense.npz", arrays)
            result.update(replay_checks=rows, replay_steps=list(range(REPLAY_HORIZON + 1)),
                          replay_seeds=dict(train=training.TRAIN_SEEDS[:8],
                                            development=DEVELOPMENT_SEEDS))
        elif phase == "reached_response":
            reached, identities, extra_bindings = load_reached_donors(
                donor_replays, train_hash, evaluation_hash,
                manifest["artifacts"]["checkpoint.pt"], sources)
            probes, rows, skipped = reached_response_assay(model, populations, reached, scale, device)
            result["artifacts"]["responses.npz"] = save_arrays(output / "responses.npz", probes)
            result.update(responses=rows, skipped_zero_directions=skipped,
                          donor_identities=identities, input_steps=list(REACHED_STEPS),
                          fixed_amplitudes=list(REACHED_AMPLITUDES),
                          gaussian_seed=REACHED_GAUSSIAN_SEED, recurrent_model_calls=0,
                          gaussian_law="unit-RMS Gaussian direction at fixed physical amplitudes",
                          response_array_row_order="row_indices indexes responses; bounded vector subset")
        else:
            cases = []
            for role, values in populations.items():
                seeds = training.TRAIN_SEEDS[:8] if role == "train" else DEVELOPMENT_SEEDS
                for index, seed in enumerate(seeds):
                    case, snapshots = rollout_case(model, values[index], scale, device)
                    case.update(role=role, seed=seed)
                    cases.append(case)
                    name = f"rollout_{role}_{seed}.npz"
                    result["artifacts"][name] = save_arrays(output / name, snapshots)
                    result.update(cases=cases, pooled_horizons=pooled_horizons(cases))
                    write_json(output / "result.json", result)
            result["amplitude_guard_rms_over_train_scale"] = AMPLITUDE_LIMIT
        if sources != {name: sha256(ROOT / name) for name in sorted(source_paths)}:
            raise RuntimeError("evaluation sources changed during execution")
        for base, bindings in ((training_input, manifest["artifacts"]),
                               (evaluation_input, evaluation_manifest["artifacts"]),
                               (fit_output, fit_hashes)):
            if base is not None and any(sha256(Path(base) / k) != v for k, v in bindings.items()):
                raise RuntimeError("evaluation input changed during execution")
        if (sha256(Path(training_input) / "manifest.json") != train_hash
                or sha256(Path(evaluation_input) / "manifest.json") != evaluation_hash):
            raise RuntimeError("input manifest changed during execution")
        if any(sha256(path) != digest for path, digest in extra_bindings.items()):
            raise RuntimeError("bound prior evaluation changed during execution")
        result["prior_artifacts"] = {str(path): digest for path, digest in extra_bindings.items()}
        result.update(status="completed", elapsed_seconds=time.perf_counter() - started)
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}",
                      elapsed_seconds=time.perf_counter() - started)
        write_json(output / "result.json", result)
        raise
    write_json(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-input", required=True, type=Path)
    parser.add_argument("--evaluation-input", required=True, type=Path)
    parser.add_argument("--fit-output", type=Path)
    parser.add_argument("--phase", required=True,
                        choices=("assay", "rollout", "replay", "reached_response", "target_assay"))
    parser.add_argument("--prior-rollout", type=Path)
    parser.add_argument("--donor-replays", type=Path)
    parser.add_argument("--target-bank", type=Path)
    parser.add_argument("--bank-role", choices=("train", "probe"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args.training_input, args.evaluation_input, args.output, args.phase,
        args.device, args.fit_output, args.prior_rollout, args.donor_replays,
        args.target_bank, args.bank_role)


if __name__ == "__main__":
    main()
