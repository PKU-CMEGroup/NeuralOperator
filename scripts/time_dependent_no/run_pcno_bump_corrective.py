#!/usr/bin/env python3
"""Run the fixed solver-free Bump comparison (smoke, then a sequential pilot).

Invocation and scientific contract: B5_BUMP_SOLVER_FREE_COMPARISON.md.
All outputs are isolated; historical runs and the historical test are untouched.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no.train_pcno_bump_scaling import (
    balanced_queue_presentations,
    fixed_evaluation_pairs,
    load_split_manifest,
    presentation_stream_sha256,
)
from utility.time_dependent_no.pcno_artifacts import (
    atomic_torch_save,
    digest_array,
    sha256_file,
    write_json,
)
from utility.time_dependent_no.pcno_bump_corrective import (
    EVALUATION_ARMS,
    TRAIN_ARMS,
    BumpRefiner,
    BumpRefinerScheduler,
    MappedTrainingPCA,
    detached_prefix,
    exposure_probability,
    generator_for,
    geometry_descriptor,
    geometry_remap,
    keyed_seed,
    learning_rate,
    map_step,
    nearest_training_geometry,
    normal_loss,
    prefix_start,
    refiner_loss,
    refiner_step,
    update_ema,
)
from utility.time_dependent_no.pcno_euler2d import (
    PCNOEuler2DResidual,
    PCNOEuler2DShardStore,
    apply_admissible_primitive_noise,
    build_graph_causal_boundary_policy,
    conservative_admissibility,
    fit_normalization,
    weighted_scaled_mse,
    weighted_scaled_relative_l2,
)
from utility.time_dependent_no.pcno_rollout import close_boundary, endpoint_diagnostics
from utility.time_dependent_no.pcno_runtime import autocast_context

EXPERIMENT = "B5_BUMP_SOLVER_FREE_COMPARISON_20260905A"
SPLIT_PATH = ROOT / "docs/time_dependent_no/D094_BUMP_SCALING_SPLIT_MANIFEST.json"
AMENDMENT = ROOT / "docs/time_dependent_no/B5_BUMP_SOLVER_FREE_COMPARISON.md"
DATA_SHA256 = "5d5373fdcc682544bf330fba6d54fe65509dabe936baee7443c71a8c0c8d9fa7"
SELECTION_KEYS = (
    "13",
    "41",
    "43",
    "48",
    "66",
    "91",
    "108",
    "127",
    "181",
    "190",
    "246",
    "259",
    "268",
    "274",
    "296",
    "299",
)
ENDPOINTS = (1, 20, 40, 60, 79)
UPDATES = 16384
SEED = 20260718


class ScopedStore(PCNOEuler2DShardStore):
    """Check roles before opening arrays; bind the bytes actually accessed."""

    def __init__(self, root, train_keys, development_keys, *, allow_development=False):
        super().__init__(
            root, max_cached_trajectories=8, max_cached_geometry_bytes=128 * 1024**2
        )
        if self.manifest_digest != DATA_SHA256:
            raise ValueError("Bump dataset manifest mismatch")
        self.train_keys = set(train_keys)
        self.development_keys = set(development_keys)
        self.allow_development = allow_development
        self.accessed = {}
        self.verified_states = set()

    def array(self, key, name):
        key = str(key)
        if key not in self.train_keys and not (
            self.allow_development and key in self.development_keys
        ):
            raise PermissionError(f"key {key} is outside the active population")
        array = super().array(key, name)
        identity = f"{key}/{name}"
        if identity not in self.accessed:
            path = self._path(key, name)
            self.accessed[identity] = {
                "relative_path": path.relative_to(self.root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        return array

    def states(self, key):
        states = super().states(key)
        if key not in self.verified_states:
            if digest_array(states) != self.entry(key)["state_digest"]:
                raise ValueError(f"stored state digest mismatch for {key}")
            self.verified_states.add(key)
        return states


def source_records():
    paths = {
        Path(__file__).resolve(),
        AMENDMENT,
        SPLIT_PATH,
        ROOT / "tests/time_dependent_no/test_pcno_bump_corrective.py",
    }
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        # PyTorch also exposes virtual modules with relative pseudo-filenames
        # such as "_classes.py". They are not repository source files.
        if isinstance(path, (str, Path)) and Path(path).is_absolute():
            path = Path(path).resolve()
            if path.is_relative_to(ROOT) and path.suffix == ".py":
                paths.add(path)
    return {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in sorted(paths)}


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def model_for(arm, norm, seed, device, *, tiny=False):
    torch.manual_seed(seed)
    cls = BumpRefiner if arm == "PCNO_PDEREFINER_K3_VPRED" else PCNOEuler2DResidual
    kwargs = {"k_max": 2, "layers": (8, 8), "fc_dim": 8} if tiny else {}
    return cls(normalization=norm, **kwargs).to(device)


def model_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def prepare(store, train_keys, device):
    norm = fit_normalization(store, train_keys)
    policies, records = {}, {}
    for key in train_keys:
        policies[key], records[key] = build_graph_causal_boundary_policy(
            store, key, device=device
        )
    return norm, policies, records


def load_completed(output, arm, norm, seed, device, source, *, ema=False):
    directory = output / arm
    receipt = json.loads((directory / "training.json").read_text())
    path = directory / "model.pt"
    if receipt["model_sha256"] != sha256_file(path) or receipt["source"] != source:
        raise ValueError("completed training checkpoint binding differs")
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload["updates"] != UPDATES or payload["normalization"] != norm.to_dict():
        raise ValueError("checkpoint schedule/normalization differs")
    model = model_for(arm, norm, seed, device)
    model.load_state_dict(payload["ema" if ema else "model"], strict=True)
    return model.eval().requires_grad_(False)


def train_arm(
    arm,
    store,
    train_keys,
    norm,
    policies,
    output,
    source,
    seed,
    device,
    amp,
    calibration=None,
    base=None,
    *,
    smoke=False,
):
    directory = output / arm
    directory.mkdir()
    model = model_for(arm, norm, seed, device)
    initial_digest = model_digest(model)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = (
        BumpRefinerScheduler(calibration["refiner_sigma"]) if calibration else None
    )
    epochs = 1 if smoke else UPDATES // 256
    steps_in_epoch = 4 if smoke else 256
    forward_calls = backward_calls = 0
    depth_counts = {str(k): 0 for k in range(4)}
    loss_history = []
    started = time.perf_counter()
    model.train()
    for epoch in range(epochs):
        pairs = balanced_queue_presentations(
            store,
            train_keys,
            step_stride=1,
            count=256,
            seed=seed,
            epoch=epoch,
        )[:steps_in_epoch]
        if smoke:
            largest = max(train_keys, key=lambda k: store.entry(k)["num_nodes"])
            pairs = [(largest, t) for t in (0, 3, 39, 78)]
        rows = []
        for local, (key, t) in enumerate(pairs):
            step = epoch * 256 + local
            sample = store.tensor_sample(key, t, step_stride=1, device=device)
            policy = policies[key]
            rng = np.random.default_rng(keyed_seed(seed, "exposure", step))
            generator = generator_for(device, seed, arm, step)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(step, UPDATES)
            optimizer.zero_grad(set_to_none=True)
            # The smoke exercises maximal prefix depth even before the curriculum starts.
            requested = 3 if smoke else int(rng.integers(1, 4))
            actual = 0
            with autocast_context(device, amp):
                if arm == "PCNO_PDEREFINER_K3_VPRED":
                    loss, actual = refiner_loss(
                        model, sample, policy, scheduler, generator
                    )
                    forward_calls += 1
                elif arm == "PREFIX_ERROR_CORRECTOR_K13":
                    start, actual = prefix_start(t + 1, requested, training_map=False)
                    initial = store.tensor_sample(
                        key, start, step_stride=1, device=device
                    )["current"]
                    proposal = detached_prefix(base, sample, initial, actual, policy)
                    corrected = map_step(model, sample, proposal, policy)
                    identity = map_step(model, sample, sample["target"], policy)
                    loss = 0.5 * normal_loss(corrected, sample["target"], sample, model)
                    loss = loss + 0.5 * normal_loss(
                        identity, sample["target"], sample, model
                    )
                    forward_calls += actual + 2
                elif arm == "CURRICULUM_EMA_PREFIX_K13" and (
                    smoke or rng.random() < exposure_probability(step, UPDATES)
                ):
                    start, actual = prefix_start(t + 1, requested, training_map=True)
                    initial = store.tensor_sample(
                        key, start, step_stride=1, device=device
                    )["current"]
                    current = detached_prefix(ema, sample, initial, actual, policy)
                    prediction = map_step(model, sample, current, policy)
                    loss = normal_loss(prediction, sample["target"], sample, model)
                    forward_calls += actual + 1
                else:
                    prediction = map_step(model, sample, sample["current"], policy)
                    loss = normal_loss(prediction, sample["target"], sample, model)
                    forward_calls += 1
                    if arm == "IID_RECOVERY":
                        noisy = apply_admissible_primitive_noise(
                            sample["current"],
                            calibration["primitive_noise_std"],
                            gamma=norm.gamma,
                            generator=generator,
                        )
                        recovery = map_step(model, sample, noisy, policy)
                        loss = 0.5 * loss + 0.5 * normal_loss(
                            recovery, sample["target"], sample, model
                        )
                        forward_calls += 1
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite training loss: {arm}, step {step}")
            loss.backward()
            backward_calls += 1
            grad = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            update_ema(ema, model)
            depth_counts[str(actual)] += 1
            rows.append((float(loss.detach()), float(grad.detach())))
        synchronize(device)
        row = {
            "epoch": epoch + 1,
            "updates": epoch * 256 + len(pairs),
            "loss": float(np.mean([x[0] for x in rows])),
            "maximum_preclip_gradient": max(x[1] for x in rows),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": time.perf_counter() - started,
            "forward_calls": forward_calls,
            "backward_calls": backward_calls,
            "depth_or_refiner_level_counts": dict(depth_counts),
            "presentation_sha256": presentation_stream_sha256(pairs),
        }
        loss_history.append(row)
        with (directory / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps({"arm": arm, **row}), flush=True)
        if not smoke and ((epoch + 1) % 4 == 0 or epoch + 1 == epochs):
            # Atomic replacement is limited to this run's rolling recovery checkpoint.
            atomic_torch_save(
                {
                    "updates": row["updates"],
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "source": source,
                    "normalization": norm.to_dict(),
                },
                directory / "last.pt",
            )
    checkpoint = {
        "arm": arm,
        "updates": loss_history[-1]["updates"],
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "source": source,
        "normalization": norm.to_dict(),
    }
    if not smoke:
        atomic_torch_save(checkpoint, directory / "model.pt")
    write_json(
        directory / "training.json",
        {
            "status": "smoke_pass" if smoke else "complete",
            "arm": arm,
            "seed": seed,
            "initial_parameter_digest": initial_digest,
            "source": source,
            "model_sha256": sha256_file(directory / "model.pt") if not smoke else None,
            "normalization": norm.to_dict(),
            "updates": loss_history[-1]["updates"],
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "forward_calls": forward_calls,
            "backward_calls": backward_calls,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else 0,
            "train_keys": train_keys,
            "development_states_used": False,
            "historical_test_population_opened": False,
        },
    )
    model.zero_grad(set_to_none=True)
    return model.eval().requires_grad_(False), ema


@torch.no_grad()
def calibrate(model, store, keys, policies, norm, device, amp, seed):
    pairs = fixed_evaluation_pairs(store, keys, step_stride=1, windows_per_trajectory=8)
    errors, residual_errors = [], []
    for key, t in pairs:
        sample = store.tensor_sample(key, t, step_stride=1, device=device)
        with autocast_context(device, amp):
            prediction = map_step(model, sample, sample["current"], policies[key])
        for scale, rows in (
            (model.state_scale, errors),
            (model.residual_scale, residual_errors),
        ):
            rows.append(
                float(
                    weighted_scaled_mse(
                        prediction,
                        sample["target"],
                        sample["node_weights"],
                        sample["node_mask"],
                        scale,
                    )
                )
            )
    target_rms = float(np.sqrt(np.mean(errors)))

    def noise_rms(sigma):
        values = []
        for key, t in pairs:
            sample = store.tensor_sample(key, t, step_stride=1, device=device)
            rng = generator_for(device, seed, "noise_calibration", key, t)
            noisy = apply_admissible_primitive_noise(
                sample["current"], sigma, generator=rng
            )
            clean = close_boundary(sample["current"], policies[key], gamma=norm.gamma)
            noisy = close_boundary(noisy, policies[key], gamma=norm.gamma)
            values.append(
                float(
                    weighted_scaled_mse(
                        noisy,
                        clean,
                        sample["node_weights"],
                        sample["node_mask"],
                        model.state_scale,
                    )
                )
            )
        return float(np.sqrt(np.mean(values)))

    # Match the training error scale, with identical keyed normal draws at every trial.
    sigma = 0.001 * target_rms / max(noise_rms(0.001), 1e-12)
    for _ in range(2):
        sigma *= target_rms / max(noise_rms(sigma), 1e-12)
    observed = noise_rms(sigma)
    if not np.isfinite(observed) or abs(observed / target_rms - 1) > 0.05:
        raise ValueError(
            "train-only recovery scale calibration did not close within 5%"
        )
    refiner_sigma = float(np.sqrt(np.mean(residual_errors)))
    BumpRefinerScheduler(refiner_sigma)
    return {
        "parent_clean_digest": model_digest(model),
        "pairs": pairs,
        "state_normalized_prefix_rms": target_rms,
        "primitive_noise_std": sigma,
        "observed_closed_noise_rms": observed,
        "refiner_sigma": refiner_sigma,
        "development_states_used": False,
        "rule": "match_clean_train_one_prefix_RMS_without_rollout_scale_tuning",
    }


def mapped_projector(store, key, train_keys, descriptors, norm, device, rank):
    nodes = np.array(store.array(key, "nodes"), copy=True)
    types = np.array(store.array(key, "node_type"), copy=True)
    desc = geometry_descriptor(nodes, types, float(store.entry(key)["mach"]))
    donor, distance = nearest_training_geometry(desc, descriptors)
    if donor not in train_keys:
        raise PermissionError("projection donor is not in the training population")
    src_nodes = np.array(store.array(donor, "nodes"), copy=True)
    src_types = np.array(store.array(donor, "node_type"), copy=True)
    indices, weights = geometry_remap(src_nodes, src_types, nodes, types)
    # Copy before a later store access can evict/close the donor's memory map.
    states = store.states(donor)
    mapped = np.stack(
        [np.einsum("nk,nkc->nc", weights, states[t][indices]) for t in range(79)]
    )
    target_weights = np.array(store.array(key, "node_weights"), copy=True)
    projector = MappedTrainingPCA(mapped, norm, target_weights, rank, device)
    return projector, {
        "donor": donor,
        "descriptor_distance": distance,
        "rank": rank,
        "effective_rank": projector.effective_rank,
        "development_values_used_to_fit": False,
    }


@torch.no_grad()
def evaluate_arm(
    arm,
    model,
    base,
    store,
    keys,
    train_keys,
    descriptors,
    norm,
    policies,
    output,
    device,
    amp,
    calibration,
    seed,
):
    rows = []
    schedule = BumpRefinerScheduler(calibration["refiner_sigma"])
    calls = (
        4
        if isinstance(model, BumpRefiner)
        else 2
        if arm == "PREFIX_ERROR_CORRECTOR_K13"
        else 1
    )
    for key in keys:
        sample = store.tensor_sample(key, 0, step_stride=1, device=device)
        policy = policies[key]
        projector, projection_record = None, None
        if arm.startswith("MAPPED_TRAIN_PCA_R"):
            projector, projection_record = mapped_projector(
                store,
                key,
                train_keys,
                descriptors,
                norm,
                device,
                int(arm.rsplit("R", 1)[1]),
            )

        def deploy(
            current,
            label,
            step,
            *,
            key=key,
            sample=sample,
            policy=policy,
            projector=projector,
        ):
            rng = generator_for(device, seed, "evaluation", key, label, step)
            with autocast_context(device, amp):
                if isinstance(model, BumpRefiner):
                    return refiner_step(model, sample, current, policy, schedule, rng)
                if arm == "PREFIX_ERROR_CORRECTOR_K13":
                    current = map_step(base, sample, current, policy)
                prediction = map_step(model, sample, current, policy)
            if projector is not None:
                prediction = close_boundary(
                    projector(prediction), policy, gamma=norm.gamma
                )
            return prediction.float()

        truth = np.array(store.states(key), copy=True)
        one_step_errors = []
        identity_errors = []
        for t in np.linspace(0, 78, 8, dtype=int):
            current = torch.tensor(truth[t : t + 1], device=device)
            target = torch.tensor(truth[t + 1 : t + 2], device=device)
            prediction = deploy(current, "one_step", int(t))
            error = weighted_scaled_relative_l2(
                prediction.double(),
                target.double(),
                sample["node_weights"],
                sample["node_mask"],
                model.state_scale,
            )
            one_step_errors.append(float(error) if torch.isfinite(error) else None)
            if arm == "PREFIX_ERROR_CORRECTOR_K13":
                with autocast_context(device, amp):
                    identity = map_step(model, sample, current, policy)
                identity_errors.append(
                    float(normal_loss(identity, current, sample, model).sqrt())
                )
            elif projector is not None:
                identity = close_boundary(projector(current), policy, gamma=norm.gamma)
                identity_errors.append(
                    float(normal_loss(identity, current, sample, model).sqrt())
                )
        current = torch.tensor(truth[0:1], device=device)
        rollout = [truth[0]] if key == keys[0] else []
        errors, doses, invalid_steps, endpoints = [], [], [], {}
        elapsed_inference = 0.0
        failure = None
        for t in range(1, 80):
            try:
                synchronize(device)
                start = time.perf_counter()
                prediction = deploy(current, "rollout", t)
                synchronize(device)
                elapsed_inference += time.perf_counter() - start
            except (FloatingPointError, ValueError) as error:
                # Only numerical scheduler failure is a scientific failure; malformed
                # contracts/geometry must still stop the run.
                if "nonfinite" not in str(error):
                    raise
                failure = {"call": t, "reason": str(error)}
                break
            if not torch.isfinite(prediction).all():
                failure = {"call": t, "reason": "nonfinite_state"}
                break
            target = torch.tensor(truth[t : t + 1], device=device)
            relative = weighted_scaled_relative_l2(
                prediction.double(),
                target.double(),
                sample["node_weights"],
                sample["node_mask"],
                model.state_scale,
            )
            errors.append(float(relative))
            validity = conservative_admissibility(prediction, gamma=norm.gamma)
            if not bool(validity["admissible"].all()):
                invalid_steps.append(t)
            if arm == "PREFIX_ERROR_CORRECTOR_K13" or projector is not None:
                with autocast_context(device, amp):
                    uncorrected = map_step(
                        base if base is not None else model, sample, current, policy
                    )
                dose = weighted_scaled_mse(
                    prediction,
                    uncorrected,
                    sample["node_weights"],
                    sample["node_mask"],
                    model.state_scale,
                )
                doses.append(float(dose.sqrt()))
            pred = prediction[0].cpu().numpy()
            if key == keys[0]:
                rollout.append(pred)
            if t in ENDPOINTS:
                structure = endpoint_diagnostics(
                    pred,
                    truth[t],
                    positions=np.array(store.array(key, "nodes")),
                    edges=np.array(store.array(key, "edges")),
                    node_type=np.array(store.array(key, "node_type")),
                    proxy_weights=np.array(store.array(key, "node_weights")),
                    component_scale=norm.state_scale,
                    gamma=norm.gamma,
                    shock_quantile=0.9,
                )
                boundary = sample["node_type"].reshape(-1) != 0
                diff = ((prediction - target) / model.state_scale)[:, boundary]
                structure["boundary_normalized_rms"] = float(
                    diff.square().mean().sqrt()
                )
                structure["admissible"] = bool(validity["admissible"].all())
                for field in ("density", "pressure", "internal_energy"):
                    minimum = validity[field].min()
                    structure[f"minimum_{field}"] = (
                        float(minimum) if torch.isfinite(minimum) else None
                    )
                endpoints[str(t)] = structure
            current = prediction
        row = {
            "arm": arm,
            "key": key,
            "seed": seed,
            "clean_one_step_relative_l2": float(np.mean(one_step_errors))
            if all(x is not None for x in one_step_errors)
            else None,
            "h1": errors[0] if errors else None,
            "h79": errors[-1] if len(errors) == 79 else None,
            "auc": float(np.mean(errors)) if len(errors) == 79 else None,
            "errors": errors,
            "completed_calls": len(errors),
            "failure": failure,
            "invalid_calls": invalid_steps,
            "endpoints": endpoints,
            "mean_correction_normalized_rms": float(np.mean(doses)) if doses else None,
            "clean_identity_normalized_rms": float(np.mean(identity_errors))
            if identity_errors
            else None,
            "deployed_model_calls_per_step": calls,
            "timed_deployment_seconds": elapsed_inference,
            "projection": projection_record,
        }
        rows.append(row)
        write_json(output / f"{arm}_partial.json", {"rows": rows})
        if key == keys[0]:
            np.savez_compressed(
                output / f"{arm}_rollout.npz", states=np.stack(rollout), key=key
            )
        print(
            json.dumps(
                {"evaluation": arm, "key": key, "h79": row["h79"], "auc": row["auc"]}
            ),
            flush=True,
        )
    return rows


def comparison_table(rows):
    by_arm = {
        arm: {r["key"]: r for r in rows if r["arm"] == arm} for arm in EVALUATION_ARMS
    }
    clean = by_arm["CLEAN"]
    table = []
    for arm, cases in by_arm.items():
        if not cases:
            continue
        complete = [r for r in cases.values() if r["h79"] is not None]
        record = {"arm": arm, "complete": len(complete), "cases": len(cases)}
        for metric in ("clean_one_step_relative_l2", "h79", "auc"):
            values = [r[metric] for r in cases.values() if r[metric] is not None]
            record[f"mean_{metric}"] = float(np.mean(values)) if values else None
            ratios = [
                r[metric] / clean[k][metric]
                for k, r in cases.items()
                if r[metric] is not None
                and clean[k][metric] is not None
                and clean[k][metric] > 0
            ]
            record[f"median_{metric}_over_clean"] = (
                float(np.median(ratios)) if ratios else None
            )
        record["joint_h79_auc_wins"] = sum(
            r["h79"] is not None
            and clean[k]["h79"] is not None
            and r["h79"] < clean[k]["h79"]
            and r["auc"] < clean[k]["auc"]
            for k, r in cases.items()
        )
        record["cases_with_inadmissibility"] = sum(
            bool(r["invalid_calls"]) for r in cases.values()
        )
        structure_ratios = {}
        for metric in (
            "front_centroid_distance",
            "shock_strength_log_error",
            "shock_thickness_log_error",
            "boundary_normalized_rms",
        ):
            ratios = []
            for key, row in cases.items():
                value = row["endpoints"].get("79", {}).get(metric)
                reference = clean[key]["endpoints"].get("79", {}).get(metric)
                if value is not None and reference is not None:
                    ratios.append(value / max(reference, 1e-8))
            ratio = float(np.median(ratios)) if ratios else None
            structure_ratios[metric] = ratio
            record[f"median_{metric}_over_clean"] = ratio
        clean_one_step = [r["clean_one_step_relative_l2"] for r in clean.values()]
        finite_clean = all(value is not None for value in clean_one_step)
        clean_mean = float(np.mean(clean_one_step)) if finite_clean else None
        clean_regression = (
            record["mean_clean_one_step_relative_l2"] / clean_mean
            if clean_mean and record["mean_clean_one_step_relative_l2"] is not None
            else None
        )
        no_new_invalid = all(
            not r["invalid_calls"] or bool(clean[k]["invalid_calls"])
            for k, r in cases.items()
        )
        record["pilot_efficacy_rule_pass"] = bool(
            len(complete) == len(cases)
            and len(cases) == len(clean)
            and record["median_h79_over_clean"] is not None
            and record["median_h79_over_clean"] <= 0.9
            and record["median_auc_over_clean"] <= 0.9
            and record["joint_h79_auc_wins"] / len(cases) >= 0.6
            and clean_regression is not None
            and clean_regression <= 1.05
            and no_new_invalid
            and all(r is not None and r <= 1.10 for r in structure_ratios.values())
        )
        table.append(record)
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "run"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--amp", choices=("none", "bf16"), default="bf16")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("a new attempt requires a new output directory")
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    split = load_split_manifest(SPLIT_PATH)
    train_keys = list(split["nested_exposure"]["subsets"]["256"])
    audit_keys = list(split["nested_exposure"]["subsets"]["32"])
    development = list(split["split"]["open_validation_keys"])
    outside = [key for key in development if key not in SELECTION_KEYS]
    if len(outside) != 28 or not set(SELECTION_KEYS).issubset(development):
        raise ValueError("development role identities differ")
    source = source_records()
    write_json(
        args.output / "run_manifest.json",
        {
            "experiment": EXPERIMENT,
            "mode": args.mode,
            "source": source,
            "data_manifest_sha256": DATA_SHA256,
            "train_keys": train_keys,
            "selection_keys": list(SELECTION_KEYS),
            "outside_keys": outside,
            "seed": SEED,
            "updates": UPDATES,
            "arms": EVALUATION_ARMS,
            "device": str(device),
            "torch": torch.__version__,
            "python": platform.python_version(),
            "historical_test_population_opened": False,
            "online_solver_calls": 0,
            "checkpoint_selection": "fixed_terminal_no_development_selection",
        },
    )
    with ScopedStore(args.data_root, train_keys, development) as store:
        norm, policies, policy_records = prepare(store, train_keys, device)
        write_json(args.output / "normalization.json", norm.to_dict())
        write_json(args.output / "boundary_contract.json", policy_records)
        smoke = args.mode == "smoke"
        base, clean_ema = train_arm(
            "CLEAN",
            store,
            train_keys,
            norm,
            policies,
            args.output,
            source,
            SEED,
            device,
            args.amp,
            smoke=smoke,
        )
        del clean_ema
        calibration = (
            {"primitive_noise_std": 0.001, "refiner_sigma": 0.05}
            if smoke
            else calibrate(
                base, store, audit_keys, policies, norm, device, args.amp, SEED
            )
        )
        write_json(args.output / "calibration.json", calibration)
        smoke_rows = []
        for arm in TRAIN_ARMS[1:]:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            model, ema = train_arm(
                arm,
                store,
                train_keys,
                norm,
                policies,
                args.output,
                source,
                SEED,
                device,
                args.amp,
                calibration,
                base,
                smoke=smoke,
            )
            if smoke:
                key = max(train_keys, key=lambda k: store.entry(k)["num_nodes"])
                sample = store.tensor_sample(key, 3, step_stride=1, device=device)
                synchronize(device)
                start = time.perf_counter()
                with torch.no_grad(), autocast_context(device, args.amp):
                    if isinstance(model, BumpRefiner):
                        prediction = refiner_step(
                            model,
                            sample,
                            sample["current"],
                            policies[key],
                            BumpRefinerScheduler(calibration["refiner_sigma"]),
                            generator_for(device, SEED, "smoke_inference"),
                        )
                    else:
                        current = sample["current"]
                        if arm == "PREFIX_ERROR_CORRECTOR_K13":
                            current = map_step(base, sample, current, policies[key])
                        prediction = map_step(model, sample, current, policies[key])
                synchronize(device)
                if not torch.isfinite(prediction).all():
                    raise FloatingPointError("nonfinite full-mesh inference smoke")
                smoke_rows.append(
                    {
                        "arm": arm,
                        "key": key,
                        "nodes": store.entry(key)["num_nodes"],
                        "seconds": time.perf_counter() - start,
                        "peak_gpu_bytes": torch.cuda.max_memory_allocated(device)
                        if device.type == "cuda"
                        else 0,
                    }
                )
            del model, ema
        if smoke:
            write_json(
                args.output / "resource_smoke.json",
                {"status": "pass", "rows": smoke_rows},
            )
        rows = []
        if not smoke:
            store.allow_development = True
            for key in outside:
                policies[key], policy_records[key] = build_graph_causal_boundary_policy(
                    store, key, device=device
                )
            write_json(args.output / "boundary_contract.json", policy_records)
            descriptors = {
                key: geometry_descriptor(
                    np.array(store.array(key, "nodes")),
                    np.array(store.array(key, "node_type")),
                    float(store.entry(key)["mach"]),
                )
                for key in train_keys
            }
            evaluation = args.output / "evaluation"
            evaluation.mkdir()
            for arm in EVALUATION_ARMS:
                if arm.startswith("MAPPED_TRAIN_PCA") or arm == "CLEAN":
                    model = base
                else:
                    checkpoint_arm = "CLEAN" if arm == "CLEAN_EMA" else arm
                    model = load_completed(
                        args.output,
                        checkpoint_arm,
                        norm,
                        SEED,
                        device,
                        source,
                        ema=arm
                        in (
                            "CLEAN_EMA",
                            "CURRICULUM_EMA_PREFIX_K13",
                            "PCNO_PDEREFINER_K3_VPRED",
                        ),
                    )
                rows.extend(
                    evaluate_arm(
                        arm,
                        model,
                        base,
                        store,
                        outside,
                        train_keys,
                        descriptors,
                        norm,
                        policies,
                        evaluation,
                        device,
                        args.amp,
                        calibration,
                        SEED,
                    )
                )
                if model is not base:
                    del model
            table = comparison_table(rows)
            write_json(evaluation / "results.json", {"rows": rows, "table": table})
            with (evaluation / "comparison.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(table[0]))
                writer.writeheader()
                writer.writerows(table)
            key = outside[0]
            np.savez_compressed(
                evaluation / "reference.npz",
                states=np.array(store.states(key)),
                nodes=np.array(store.array(key, "nodes")),
                edges=np.array(store.array(key, "edges")),
                key=key,
            )
        if source_records() != source:
            raise ValueError("source changed during the experiment")
        write_json(
            args.output / "data_access.json",
            {
                "files": store.accessed,
                "verified_state_keys": sorted(store.verified_states, key=int),
                "development_states_opened": not smoke,
                "historical_test_population_opened": False,
            },
        )
    write_json(
        args.output / "final_manifest.json",
        {
            "experiment": EXPERIMENT,
            "status": "smoke_pass" if smoke else "complete",
            "source": source,
            "historical_test_population_opened": False,
            "files": {
                p.relative_to(args.output).as_posix(): {
                    "sha256": sha256_file(p),
                    "bytes": p.stat().st_size,
                }
                for p in sorted(args.output.rglob("*"))
                if p.is_file()
            },
        },
    )
    print(
        json.dumps(
            {
                "status": "smoke_pass" if smoke else "complete",
                "output": str(args.output),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
