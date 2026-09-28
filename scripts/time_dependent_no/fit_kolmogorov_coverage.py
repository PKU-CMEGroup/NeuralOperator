"""Fixed-update Clean fit on the audited 32-path Kolmogorov training union.

Run with ``python -m scripts.time_dependent_no.fit_kolmogorov_coverage --help``.
The validate phase decodes data but creates no model. The fit phase uses CUDA,
the original eight-path normalizer, and exactly the old terminal's LR history.
Closed population, model and evaluation implementations remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import shutil
import zipfile
import zlib
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from scripts.time_dependent_no import evaluate_kolmogorov_clean_rollout as baseline
from scripts.time_dependent_no import extend_kolmogorov_training_population as extension
from scripts.time_dependent_no import fit_kolmogorov_clean as clean

RUN_ID = "CM_NEXT_KF_CLEAN32_MATCHED_20260909B"
REPO_ROOT = Path(__file__).resolve().parents[2]
EXTENSION_MANIFEST_SHA256 = (
    "a0f446389160974f29ad0cec07ce9ac752442b5b7d78f118a8a7e83ffd01a0cc"
)
EXTENSION_AUDIT_SHA256 = (
    "8553e0619f463c2d10cd05ada457020af0930766c6ee26aef8c53016ffacbdc0"
)
SOURCE_PATHS = tuple(
    dict.fromkeys(
        (
            "scripts/time_dependent_no/fit_kolmogorov_coverage.py",
            "tests/time_dependent_no/test_fit_kolmogorov_coverage.py",
            *extension.SOURCE_PATHS,
            *baseline.SOURCE_PATHS,
        )
    )
)
TOTAL_UPDATES = 49152
WALL_SECONDS = 5 * 3600
MINIMUM_FREE_BYTES = 2 * 1024**3


def check_deadline(deadline, stage):
    if perf_counter() >= deadline:
        raise TimeoutError(f"coverage fit wall budget exhausted during {stage}")


def validate_extension(
    packet, source, parent_result, parent_evidence, *, unit_fixture=False
):
    """Bind the previously audited packet before any added state is decoded."""
    packet, source = Path(packet), Path(source)
    digest = clean._hash(packet / "artifact_manifest.json")
    if not unit_fixture and digest != EXTENSION_MANIFEST_SHA256:
        raise ValueError("extension is not the pinned qualified training packet")
    manifest = clean._read(packet / "artifact_manifest.json")
    expected_id = extension.RUN_ID + ("__UNIT_FIXTURE" if unit_fixture else "")
    if (
        manifest["run_id"] != expected_id
        or manifest["status"] != "completed"
        or manifest["source_stable"] is not True
        or manifest["parent_manifest_sha256"] != parent_evidence["manifest_sha256"]
        or set(manifest["sources"]) != set(extension.SOURCE_PATHS)
        or {p.name for p in packet.iterdir()}
        != set(manifest["artifacts"]) | {"artifact_manifest.json"}
    ):
        raise ValueError("extension identity, inventory or source closure mismatch")
    for root, values in (
        (packet, manifest["artifacts"]),
        (source, manifest["sources"]),
        (REPO_ROOT, manifest["sources"]),
    ):
        if any(
            clean._hash(clean._bound(root, name)) != sha for name, sha in values.items()
        ):
            raise ValueError("extension artifact or archived/live source hash mismatch")
    result, launch = (
        clean._read(packet / name) for name in ("result.json", "launch.json")
    )
    protocol = extension.PROTOCOL
    if unit_fixture:
        protocol = replace(
            protocol,
            resolution=16,
            steps=4,
            anchor_steps=(0, 1, 4),
            late_anchor=2,
            late_horizon=2,
            block_steps=2,
            wall_seconds=60,
            seeds=extension.NEW_SEEDS[:2],
            fixture_label="unit_test_only",
        )
    expected_protocol = json.loads(json.dumps(asdict(protocol)))
    index = extension.population_index(protocol.seeds)
    if (
        result["run_id"] != expected_id
        or launch["run_id"] != expected_id
        or result["status"] != "completed"
        or result["engineering_gates_pass"] is not True
        or result["source_stable"] is not True
        or result["parent_evidence_stable"] is not True
        or result["sources_before"] != manifest["sources"]
        or result["sources_after"] != manifest["sources"]
        or launch["sources_before"] != manifest["sources"]
        or result["parent_evidence"] != parent_evidence
        or result["parent_manifest_sha256"] != parent_evidence["manifest_sha256"]
        or result["initial_recipe"] != parent_result["initial_recipe"]
        or result["protocol"] != expected_protocol
        or launch["protocol"] != expected_protocol
        or result["population_index"] != index
        or launch["population_index"] != index
        or result["new_development_trajectories"] != 0
        or result["protected_access"] is not False
        or result["model_evaluated"] is not False
        or result["optimization_steps"] != 0
        or result["retained_new_training_transitions"]
        != len(protocol.seeds) * protocol.steps
        or result["continuation_seeds"] != list(extension.CONTINUATION_SEEDS)
    ):
        raise ValueError("extension completion, split or physical contract mismatch")
    replay = result["runtime_replay"]
    if (
        replay["status"] != "completed"
        or len(replay["calls"]) != 6
        or any(
            r["status"] != "completed"
            or r["repeat_equal"] is not True
            or not 0 <= r["relative_l2"] <= extension.REPLAY_LIMIT
            for r in replay["calls"]
        )
    ):
        raise ValueError("extension solver replay was not qualified")
    gates = extension.numerical_gates(
        result["cases"], protocol, extension.CONTINUATION_SEEDS
    )
    if gates != result["numeric_gates"] or not all(
        g["pass"] is True for g in gates.values()
    ):
        raise ValueError("extension numerical qualification incomplete")
    for case in result["cases"]:
        if (
            case != clean._read(packet / f"case_{case['seed']}.json")
            or case["config"] != parent_result["cases"][0]["config"]
            or [r["step"] for r in case["trajectory_diagnostics"]]
            != list(range(protocol.steps + 1))
        ):
            raise ValueError("extension case or state identity mismatch")
    return result, {
        "manifest_sha256": digest,
        "artifacts": {**manifest["artifacts"], "artifact_manifest.json": digest},
        "sources": manifest["sources"],
    }


@dataclass
class CoveragePopulation:
    states: np.ndarray
    train_scale: float
    index: list
    metadata: dict

    def batch(self, indices, role="train"):
        steps = self.states.shape[1] - 1
        training = len(self.index) - 4
        count, offset = {"train": (training, 0), "development": (4, training)}[role]
        indices = np.asarray(indices)
        if (
            indices.ndim != 1
            or not len(indices)
            or indices.dtype.kind not in "iu"
            or np.any(indices < 0)
            or np.any(indices >= count * steps)
        ):
            raise ValueError("batch indices must name transitions within one role")
        trajectory, times = offset + indices // steps, indices % steps
        return (
            torch.from_numpy(self.states[trajectory, times]),
            torch.from_numpy(self.states[trajectory, times + 1]),
        )


def load_population(
    parent, parent_source, added, added_source, *, unit_fixture=False, receipt=None
):
    """Original float32 states/float64 normalizer are reused exactly, not refit."""
    started = perf_counter()
    receipt = {} if receipt is None else receipt
    old_receipt = receipt.setdefault("original", {})
    old = clean.load_population(
        parent, parent_source, unit_fixture=unit_fixture, receipt=old_receipt
    )
    parent_result = clean._read(Path(parent) / "result.json")
    result, evidence = validate_extension(
        added,
        added_source,
        parent_result,
        old_receipt["evidence"],
        unit_fixture=unit_fixture,
    )
    receipt["extension_evidence"] = evidence
    index = result["population_index"]
    n, steps = old.states.shape[-1], old.states.shape[1] - 1
    training = len(index) - 4
    states = np.empty((len(index), steps + 1, n, n), dtype=np.float32)
    states[:8], states[training:] = old.states[:8], old.states[8:]
    scale = old.train_scale
    del old  # Peak store allocation is old+union, not a float64 union.
    modes = np.fft.fftfreq(n) * n
    mask = (np.abs(modes[:, None]) <= n // 3) & (np.abs(modes[None, :]) <= n // 3)
    mask[0, 0] = False
    added_sum, residual_max = 0.0, 0.0
    inputs_digest = hashlib.sha256()
    for trajectory, case in enumerate(result["cases"], 8):
        expected = 0
        for block in case["blocks"]:
            if block["file"] not in evidence["artifacts"]:
                raise ValueError("added state block is not manifest-bound")
            with np.load(Path(added) / block["file"], allow_pickle=False) as data:
                for step, state in zip(data["steps"], data["states"], strict=True):
                    if (
                        expected > steps
                        or step != expected
                        or state.dtype != np.float64
                        or state.shape != (n, n)
                        or not np.isfinite(state).all()
                    ):
                        raise ValueError(
                            "added state is nonfinite, missing, repeated or mistyped"
                        )
                    digest = clean._state_hash(state)
                    if (
                        digest
                        != case["trajectory_diagnostics"][expected]["state_sha256"]
                    ):
                        raise ValueError("added state/diagnostic hash mismatch")
                    projected = np.fft.ifft2(np.fft.fft2(state) * mask).real
                    mean_square = float(np.mean(state**2, dtype=np.float64))
                    residual = float(
                        np.sqrt(np.mean((projected - state) ** 2))
                        / max(np.sqrt(mean_square), np.finfo(float).eps)
                    )
                    if not np.isfinite(residual) or residual > 1e-11:
                        raise ValueError("added state is not canonical")
                    residual_max = max(residual_max, residual)
                    states[trajectory, expected] = state
                    if not np.isfinite(states[trajectory, expected]).all():
                        raise ValueError("added state cannot be represented in float32")
                    if expected < steps:
                        added_sum += mean_square
                        inputs_digest.update(
                            f"{case['seed']}:{expected}:{digest}\n".encode()
                        )
                    expected += 1
        if expected != steps + 1:
            raise ValueError("added trajectory does not contain all native states")
    metadata = {
        "shape": list(states.shape),
        "dtype": states.dtype.str,
        "state_store_bytes": states.nbytes,
        "state_store_sha256": hashlib.sha256(memoryview(states).cast("B")).hexdigest(),
        "train_input_rms_float64": scale,
        "normalizer_population": "original_eight_training_inputs_only",
        "descriptive_union_input_rms_float64": float(
            np.sqrt((scale**2 * 8 * steps + added_sum) / (training * steps))
        ),
        "added_training_input_identity_sha256": inputs_digest.hexdigest(),
        "development_used_for_scaling": False,
        "added_training_used_for_scaling": False,
        "terminal_train_targets_used_for_scaling": False,
        "transition_counts": {"train": training * steps, "development": 4 * steps},
        "maximum_added_source_canonical_residual": residual_max,
        "load_and_validate_seconds": perf_counter() - started,
    }
    receipt.update(metadata)
    return CoveragePopulation(states, scale, index, metadata)


def summarize_teacher(rows, *, train_scale, node_count, steps, bands, index):
    """Same SSE definitions as Clean, with explicit old/added/validation groups."""
    count = len(index)
    if (
        not np.isfinite(train_scale)
        or train_scale <= 0
        or node_count <= 0
        or steps <= 0
        or len(bands) != 4
        or bands[0] != 0
        or bands[-1] != steps
        or any(a >= b for a, b in pairwise(bands))
        or index != extension.population_index([r["seed"] for r in index[8:-4]])
    ):
        raise ValueError("invalid teacher scale, bands or population index")
    trajectory, times = (
        np.asarray(rows["trajectory_index"]),
        np.asarray(rows["input_step"]),
    )
    if (
        trajectory.shape != (count * steps,)
        or times.shape != trajectory.shape
        or trajectory.dtype.kind not in "iu"
        or times.dtype.kind not in "iu"
        or np.any(trajectory < 0)
        or np.any(trajectory >= count)
        or np.any(times < 0)
        or np.any(times >= steps)
        or not np.array_equal(
            np.sort(trajectory * steps + times), np.arange(count * steps)
        )
    ):
        raise ValueError("teacher rows require complete unique population pairs")
    arrays = {
        name: np.asarray(rows[name], dtype=np.float64) for name in clean.PAIR_COLUMNS
    }
    if any(
        a.shape != trajectory.shape or not np.isfinite(a).all() or np.any(a < 0)
        for a in arrays.values()
    ):
        raise ValueError(
            "teacher squared norms must be aligned finite nonnegative arrays"
        )

    def aggregate(selected):
        sums = {
            name: float(np.sum(a[selected], dtype=np.float64))
            for name, a in arrays.items()
        }
        pairs = int(np.count_nonzero(selected))
        if not pairs or not all(np.isfinite(v) for v in sums.values()):
            raise ValueError("empty or nonfinite pooled teacher metric")

        def ratio(a, b):
            return float(np.sqrt(sums[a] / sums[b])) if sums[b] > 0 else None

        return {
            "pair_count": pairs,
            "squared_norm_sums": sums,
            "relative_l2": ratio("learned_sse", "target_sse"),
            "normalized_mse": sums["learned_sse"]
            / (pairs * node_count * train_scale**2),
            "persistence_relative_l2": ratio("persistence_sse", "target_sse"),
            "zero_relative_l2": ratio("target_sse", "target_sse"),
            "learned_over_persistence": ratio("learned_sse", "persistence_sse"),
            "raw_relative_l2": ratio("raw_sse", "target_sse"),
            "projection_relative_l2": ratio("projection_sse", "target_sse"),
        }

    training = count - 4
    summary = {
        "train": aggregate(trajectory < training),
        "original_train": aggregate(trajectory < 8),
        "added_train": aggregate((trajectory >= 8) & (trajectory < training)),
        "development": aggregate(trajectory >= training),
        "trajectories": [],
    }
    for i, row in enumerate(index):
        selected = trajectory == i
        summary["trajectories"].append(
            {
                "seed": row["seed"],
                "role": row["role"],
                **aggregate(selected),
                "bands": [
                    {
                        "start": a,
                        "stop": b,
                        **aggregate(selected & (times >= a) & (times < b)),
                    }
                    for a, b in pairwise(bands)
                ],
            }
        )
    return summary


def teacher_evaluation(model, population, output, update, bands, deadline, receipt):
    started = perf_counter()
    device = next(model.parameters()).device
    count, frames, n, _ = population.states.shape
    steps, training = frames - 1, count - 4
    rows = {
        "trajectory_index": np.repeat(np.arange(count), steps),
        "input_step": np.tile(np.arange(steps), count),
    }
    rows.update(
        {
            name: np.empty(count * steps, dtype=np.float64)
            for name in (*clean.PAIR_COLUMNS, "raw_output_sse", "next_output_sse")
        }
    )
    # Include both edges of the added population as well as the old sentinels.
    sentinels = (
        0,
        8 * steps - 1,
        8 * steps,
        training * steps - 1,
        training * steps,
        count * steps - 1,
    )
    fields = {name: [] for name in ("input", "target", "raw", "next")}
    model.eval()
    with torch.no_grad():
        for role, offset, size in (
            ("train", 0, training * steps),
            ("development", training * steps, 4 * steps),
        ):
            for start in range(0, size, 8):
                check_deadline(deadline, "teacher evaluation")
                selected = np.arange(start, min(start + 8, size))
                inputs, targets = (
                    v.to(device) for v in population.batch(selected, role)
                )
                predicted = model(inputs)
                receipt["teacher_forward_calls"] += 1
                raw, nxt = predicted["raw_next"], predicted["next_state"]
                x, y, r, z = (v.double() for v in (inputs, targets, raw, nxt))
                values = dict(
                    zip(clean.PAIR_COLUMNS, (z - y, r - y, y, x, x - y, r - z))
                )
                values.update(raw_output_sse=r, next_output_sse=z)
                for name, value in values.items():
                    reduced = value.square().sum(dim=(-2, -1)).cpu().numpy()
                    if not np.isfinite(reduced).all():
                        raise RuntimeError("nonfinite coverage teacher evaluation")
                    rows[name][offset + selected] = reduced
                for local, global_index in enumerate(offset + selected):
                    if int(global_index) in sentinels:
                        for name, value in zip(fields, (inputs, targets, raw, nxt)):
                            fields[name].append(value[local].cpu().numpy().copy())
                check_deadline(deadline, "teacher evaluation")
    summary = summarize_teacher(
        rows,
        train_scale=float(model.train_scale),
        node_count=n * n,
        steps=steps,
        bands=bands,
        index=population.index,
    )
    rows["sentinel_global_indices"] = np.asarray(sentinels)
    rows.update({"sentinel_" + name: np.stack(value) for name, value in fields.items()})
    check_deadline(deadline, "teacher serialization")
    path = output / f"teacher_{update:06d}.npz"
    with path.with_suffix(".tmp").open("xb") as stream:
        np.savez_compressed(stream, **rows)
    path.with_suffix(".tmp").replace(path)
    check_deadline(deadline, "teacher serialization")
    return {
        "update": update,
        "summary": summary,
        "file": path.name,
        "sha256": clean._hash(path),
        "engineering_readiness": clean.check_clean_readiness(summary),
        "seconds": perf_counter() - started,
    }


def evidence_stable(packet, source, evidence):
    try:
        return {p.name for p in packet.iterdir()} == set(evidence["artifacts"]) and all(
            clean._hash(clean._bound(root, name)) == sha
            for root, values in (
                (packet, evidence["artifacts"]),
                (source, evidence["sources"]),
            )
            for name, sha in values.items()
        )
    except (OSError, ValueError):
        return False


def run(
    parent,
    parent_source,
    added,
    added_source,
    old_clean,
    clean_source,
    output,
    phase,
    device_name=None,
    *,
    unit_fixture=False,
):
    if phase not in ("validate", "fit") or (
        phase == "fit" and device_name != ("cpu" if unit_fixture else "cuda")
    ):
        raise ValueError("fit uses CUDA; only explicit unit fixtures can train on CPU")
    parent, parent_source, added, added_source, old_clean, clean_source, output = (
        Path(path).resolve()
        for path in (
            parent,
            parent_source,
            added,
            added_source,
            old_clean,
            clean_source,
            output,
        )
    )
    if any(
        output.is_relative_to(packet) or packet.is_relative_to(output)
        for packet in (parent, added, old_clean)
    ):
        raise ValueError("output must be disjoint from every input packet")
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    total, every, bands = (
        (12, 2, (0, 1, 2, 4))
        if unit_fixture
        else (TOTAL_UPDATES, 4096, (0, 64, 256, 512))
    )
    config = (
        {"resolution": 16, "width": 8, "depth": 2, "modes": 2, "fc_dim": 16}
        if unit_fixture
        else clean.MODEL_CONFIG
    )
    sources = {}
    recipe = {
        "total_updates": total,
        "evaluation_interval": every,
        "batch_size": 8,
        "seed": 17,
        "zero_final_head": True,
        "bands": list(bands),
        "learning_rate": "unchanged clean_lr(update); full baseline terminal history",
        "optimizer": {
            "name": "Adam",
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.0,
        },
        "loss": "mean(((restricted_next-target)/original_float32_train_scale)^2)",
        "sampling": "complete seeded without-replacement transition permutation per epoch",
        "precision": "float32; no AMP/TF32; no gradient clipping; abort nonfinite loss/gradient",
        "selection": "fixed matched-update terminal, regardless of validation outcomes; no automatic extension",
        "normalizer": "original eight training input trajectories only; no union refit",
        "checkpoint_retention": "atomic latest plus fixed terminal; no best checkpoint",
        "evaluation_retention": "all-pair scalar metrics and six input/target/raw/restricted replay sentinels; not all prediction fields",
        "wall_seconds": WALL_SECONDS,
        "wall_scope": "source and input validation through completed terminal replay; final provenance checks reported separately",
        "minimum_free_bytes": MINIMUM_FREE_BYTES,
    }
    record = {
        "run_id": RUN_ID + ("__UNIT_FIXTURE" if unit_fixture else ""),
        "phase": phase,
        "status": "running",
        "stage": "source_validation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sources_before": sources,
        "model_config": config,
        "recipe": recipe,
        "data": {},
        "updates_completed": 0,
        "teacher_forward_calls": 0,
        "evaluations": [],
        "terminal_checkpoint": None,
        "model_created": False,
        "qualified_extension_audit_sha256": None
        if unit_fixture
        else EXTENSION_AUDIT_SHA256,
    }
    bindings = []
    device = torch.device(device_name) if phase == "fit" else None
    try:
        for name in SOURCE_PATHS:
            sources[name] = clean._hash(REPO_ROOT / name)
        clean._write(output / "launch.json", record)
        record["stage"] = "input_validation"
        old, old_evidence = baseline.validate_clean(
            old_clean, clean_source, unit_fixture=unit_fixture
        )
        record["baseline_evidence"] = old_evidence
        bindings.append(("baseline", old_clean, clean_source, old_evidence))
        if (
            old["updates_completed"] != total
            or old["model_config"] != config
            or old["recipe"]["seed"] != 17
            or old["recipe"]["batch_size"] != 8
            or old["recipe"]["bands"] != list(bands)
            or old["recipe"]["optimizer"] != recipe["optimizer"]
        ):
            raise ValueError("baseline does not match the fixed model/update contract")
        try:
            population = load_population(
                parent,
                parent_source,
                added,
                added_source,
                unit_fixture=unit_fixture,
                receipt=record["data"],
            )
        finally:
            # A loader can fail after validating evidence but before returning states.
            for role, packet, source, evidence in (
                (
                    "original",
                    parent,
                    parent_source,
                    record["data"].get("original", {}).get("evidence"),
                ),
                (
                    "extension",
                    added,
                    added_source,
                    record["data"].get("extension_evidence"),
                ),
            ):
                if evidence is not None:
                    bindings.append((role, packet, source, evidence))
        for name in (
            "state_store_sha256",
            "training_input_identity_sha256",
            "train_input_rms_float64",
        ):
            if record["data"]["original"][name] != old["data"][name]:
                raise ValueError(
                    "original states or training normalizer differ from baseline"
                )
        if (
            float(np.float32(population.train_scale))
            != old["checkpoint_identity"]["train_scale_model_float32"]
        ):
            raise ValueError("stored model normalizer differs from baseline")
        record["population_index"] = population.index
        record["baseline_terminal_summary"] = old["evaluations"][-1]["summary"]
        if phase == "fit":
            deadline = started + (300 if unit_fixture else WALL_SECONDS)
            check_deadline(deadline, "input validation")
            record["stage"] = "resource_preflight"
            if shutil.disk_usage(output).free < (
                1024**2 if unit_fixture else MINIMUM_FREE_BYTES
            ):
                raise RuntimeError("insufficient free storage; no automatic deletion")
            if device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable")
            torch.set_num_threads(2)
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            random.seed(17)
            np.random.seed(17)
            torch.manual_seed(17)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(17)
                torch.cuda.reset_peak_memory_stats(device)
            model = clean.PeriodicVorticityPCNO(
                **config, train_scale=population.train_scale
            ).to(device)
            torch.nn.init.zeros_(model.pcno.fc2.weight)
            torch.nn.init.zeros_(model.pcno.fc2.bias)
            record["model_created"] = True
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=clean.clean_lr(1, unit_fixture=unit_fixture),
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.0,
            )
            sampler = clean.EpochSampler(
                population.metadata["transition_counts"]["train"]
            )
            identity = {
                "run_id": record["run_id"],
                "sources": sources,
                "input_manifests": [e["manifest_sha256"] for _, _, _, e in bindings],
                "state_store_sha256": population.metadata["state_store_sha256"],
                "train_scale_float64": population.train_scale,
                "train_scale_model_float32": float(model.train_scale),
                "model_config": config,
                "recipe": recipe,
                "population_index": population.index,
            }
            record["checkpoint_identity"] = identity
            record["parameter_count"] = sum(p.numel() for p in model.parameters())

            def evaluate(update):
                record["stage"] = "teacher_evaluation"
                check_deadline(deadline, "teacher evaluation")
                evaluation = teacher_evaluation(
                    model, population, output, update, bands, deadline, record
                )
                record["evaluations"].append(evaluation)
                with (output / "evaluations.jsonl").open("a", encoding="utf-8") as log:
                    log.write(json.dumps(evaluation, allow_nan=False) + "\n")
                record["stage"] = "checkpoint_write"
                check_deadline(deadline, "checkpoint write")
                sha = clean.save_clean_checkpoint(
                    output / "latest_checkpoint.pt",
                    model,
                    optimizer,
                    sampler,
                    update=update,
                    identity=identity,
                )
                record["latest_checkpoint"] = {
                    "file": "latest_checkpoint.pt",
                    "sha256": sha,
                    "update": update,
                }
                check_deadline(deadline, "checkpoint write")
                clean._write(output / "progress.json", record)
                check_deadline(deadline, "progress write")

            evaluate(0)
            with (output / "updates.jsonl").open("x", encoding="utf-8") as log:
                for update in range(1, total + 1):
                    record["stage"] = "training"
                    check_deadline(deadline, "training")
                    selected = sampler.next_indices()
                    lr = clean.clean_lr(update, unit_fixture=unit_fixture)
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    model.train()
                    clean._sync(device)
                    step_started = perf_counter()
                    inputs, targets = (v.to(device) for v in population.batch(selected))
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(inputs)
                    loss = (
                        ((prediction["next_state"] - targets) / model.train_scale)
                        .square()
                        .mean()
                    )
                    loss_value = float(loss.detach())
                    if not np.isfinite(loss_value):
                        raise RuntimeError("nonfinite coverage training loss")
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float("inf"), error_if_nonfinite=True
                    )
                    optimizer.step()
                    clean._sync(device)
                    record["updates_completed"] = update
                    log.write(
                        json.dumps(
                            {
                                "update": update,
                                "epoch": sampler.epoch,
                                "sampler_cursor": sampler.cursor,
                                "train_transition_indices": selected.tolist(),
                                "learning_rate": lr,
                                "loss_normalized_mse": loss_value,
                                "unclipped_gradient_norm": float(norm),
                                "inclusive_seconds": perf_counter() - step_started,
                            },
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    log.flush()
                    del inputs, targets, prediction, loss
                    check_deadline(deadline, "training")
                    if update % every == 0:
                        optimizer.zero_grad(set_to_none=True)
                        evaluate(update)
            record["stage"] = "terminal_replay"
            check_deadline(deadline, "terminal checkpoint")
            terminal = output / f"terminal_{total:06d}.pt"
            sha = clean.save_clean_checkpoint(
                terminal, model, optimizer, sampler, update=total, identity=identity
            )
            check_deadline(deadline, "terminal checkpoint")
            if (
                clean.load_clean_checkpoint(
                    terminal, model, optimizer, sampler, identity=identity
                )
                != total
            ):
                raise RuntimeError("terminal checkpoint update mismatch")
            check_deadline(deadline, "terminal checkpoint load")
            replay = clean._terminal_replay(
                model, output / record["evaluations"][-1]["file"]
            )
            check_deadline(deadline, "terminal replay")
            record["terminal_checkpoint"] = {
                "file": terminal.name,
                "sha256": sha,
                "update": total,
                "replay": replay,
            }
            earlier, last = (
                record["evaluations"][-3]["summary"],
                record["evaluations"][-1]["summary"],
            )
            record["improvement_window_updates"] = [
                record["evaluations"][-3]["update"],
                record["evaluations"][-1]["update"],
            ]
            record["relative_improvement_last_two_intervals"] = {
                group: 1 - last[group]["relative_l2"] / earlier[group]["relative_l2"]
                if earlier[group]["relative_l2"] is not None
                and earlier[group]["relative_l2"] > 0
                and last[group]["relative_l2"] is not None
                else None
                for group in ("train", "original_train", "added_train", "development")
            }
            record["matched_comparison"] = {
                "baseline_updates": total,
                "new_updates": total,
                "baseline_training_passes": total
                * 8
                / old["data"]["transition_counts"]["train"],
                "new_training_passes": total
                * 8
                / population.metadata["transition_counts"]["train"],
                "old_eight_training": {
                    "baseline": old["evaluations"][-1]["summary"]["train"],
                    "new": last["original_train"],
                },
                "unchanged_validation": {
                    "baseline": old["evaluations"][-1]["summary"]["development"],
                    "new": last["development"],
                },
                "expanded_training": last["train"],
                "added_training": last["added_train"],
            }
            check_deadline(deadline, "fit completion")
        record["status"] = "completed"
    except (
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
        FloatingPointError,
        MemoryError,
        zipfile.BadZipFile,
        EOFError,
        zlib.error,
    ) as error:
        record.update(
            status="incomplete_budget"
            if isinstance(error, TimeoutError)
            else "invalid_input"
            if record["stage"] in ("source_validation", "input_validation")
            else "failed",
            error_type=type(error).__name__,
            error=str(error),
        )
    work_seconds = perf_counter() - started
    after, source_errors = {}, {}
    for name in SOURCE_PATHS:
        try:
            after[name] = clean._hash(REPO_ROOT / name)
        except OSError as error:
            after[name] = None
            source_errors[name] = str(error)
    source_stable = len(sources) == len(SOURCE_PATHS) and after == sources
    input_checks = {
        "baseline": None,
        "original": None,
        "extension": None,
    }
    for role, packet, source, evidence in bindings:
        input_checks[role] = evidence_stable(packet, source, evidence)
    stable = (
        False
        if False in input_checks.values()
        else True
        if all(value is True for value in input_checks.values())
        else None
    )
    if not source_stable or stable is False:
        record["status"] = "invalid_provenance"
    record.update(
        sources_after=after,
        source_stable=source_stable,
        source_check_errors=source_errors,
        input_evidence_stable=stable,
        input_evidence_checks=input_checks,
        work_seconds=work_seconds,
        provenance_check_seconds=perf_counter() - started - work_seconds,
        seconds=perf_counter() - started,
        finished_utc=datetime.now(timezone.utc).isoformat(),
        scientific_training=not unit_fixture and record["updates_completed"] > 0,
        rollouts_evaluated=False,
        solver_calls=0,
        development_used_for_updates=False,
        protected_access=False,
        network_access=False,
        process_peak_rss_bytes=clean._peak_rss_bytes(),
        runtime={
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "system": platform.system(),
            "cuda": torch.version.cuda,
        },
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device)
        if device is not None and device.type == "cuda" and torch.cuda.is_available()
        else None,
        peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(device)
        if device is not None and device.type == "cuda" and torch.cuda.is_available()
        else None,
        interpretation="Fixed-update clean coverage comparison only. Completion and old engineering gates are not clean-gap qualification. Inspect absolute train/validation errors, original/added subsets, every trajectory/time band and optimization progress. No rollout, geometry or correction claim.",
    )
    clean._write(output / "result.json", record)
    clean._write(
        output / "artifact_manifest.json",
        {
            "run_id": record["run_id"],
            "phase": phase,
            "status": record["status"],
            "sources": sources,
            "source_stable": record["source_stable"],
            "input_manifests": [e["manifest_sha256"] for _, _, _, e in bindings],
            "artifacts": {
                p.name: clean._hash(p) for p in sorted(output.iterdir()) if p.is_file()
            },
        },
    )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "parent",
        "parent-source",
        "extension",
        "extension-source",
        "clean",
        "clean-source",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--phase", choices=("validate", "fit"), required=True)
    parser.add_argument("--device", choices=("cuda",))
    args = parser.parse_args()
    if args.phase == "fit" and args.device != "cuda":
        parser.error("--phase fit requires --device cuda")
    if args.phase == "validate" and args.device is not None:
        parser.error("--phase validate does not use a device")
    result = run(
        args.parent,
        args.parent_source,
        args.extension,
        args.extension_source,
        args.clean,
        args.clean_source,
        args.output,
        args.phase,
        args.device,
    )
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
