"""Qualified-population resource check and frozen one-step Clean pilot.

Run ``python -m scripts.time_dependent_no.fit_kolmogorov_clean --parent <C>
--parent-source <frozen-C-source-root> --output <fresh> --device cuda
--phase resource|clean``. Resource is the default sixteen-update timing check
without a checkpoint. Clean follows the fixed terminal-only pilot; no rollouts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from utility.time_dependent_no.pcno_kolmogorov import PeriodicVorticityPCNO

RUN_ID = "CM_NEXT_KF_CLEAN_RESOURCE_20260907A"
CLEAN_RUN_ID = "CM_NEXT_KF_CLEAN_20260907A"
PARENT_RUN_ID = "CM_NEXT_KF_POP_20260907C"
PARENT_MANIFEST_SHA256 = (
    "9ae1c66f2d7b8a2cdb105d7d57c71ee7811e64cc7f17203db7bcb51441c57f9c"
)
REPO_ROOT = Path(__file__).resolve().parents[2]
SEED_ROLES = tuple((s, "train") for s in range(2026090611, 2026090619)) + tuple(
    (s, "development") for s in range(2026090621, 2026090625)
)
PARENT_SOURCE_PATHS = (
    "scripts/time_dependent_no/generate_kolmogorov_population.py",
    "tests/time_dependent_no/test_generate_kolmogorov_population.py",
    "scripts/time_dependent_no/screen_kolmogorov_peak_refinement.py",
    "tests/time_dependent_no/test_screen_kolmogorov_peak_refinement.py",
    "scripts/time_dependent_no/screen_kolmogorov_trajectory_readiness.py",
    "tests/time_dependent_no/test_screen_kolmogorov_trajectory_readiness.py",
    "utility/time_dependent_no/kolmogorov_reference.py",
    "utility/__init__.py",
    "utility/time_dependent_no/__init__.py",
    "utility/adam.py",
    "utility/losses.py",
    "utility/normalizer.py",
)
SOURCE_PATHS = (
    "scripts/time_dependent_no/fit_kolmogorov_clean.py",
    "tests/time_dependent_no/test_fit_kolmogorov_clean.py",
    "utility/time_dependent_no/pcno_kolmogorov.py",
    "tests/time_dependent_no/test_pcno_kolmogorov.py",
    "pcno/__init__.py",
    "pcno/pcno.py",
    "pcno/geo_utility.py",
    "utility/__init__.py",
    "utility/time_dependent_no/__init__.py",
    "utility/adam.py",
    "utility/losses.py",
    "utility/normalizer.py",
)
GATE_LIMITS = {
    "clean_spatial_relative_l2": 1e-3,
    "fine_discarded_state_relative_l2": 1e-3,
    "continuation_endpoint_relative_l2": 2e-3,
}
MODEL_CONFIG = {"resolution": 256, "width": 64, "depth": 4, "modes": 12, "fc_dim": 128}
RESOURCE_RECIPE = {
    "batch_size": 8,
    "updates": 16,
    "optimizer": "Adam",
    "lr": 1e-3,
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "weight_decay": 0.0,
    "seed": 17,
    "zero_final_head": True,
    "sampling": "first batches of a seeded without-replacement train-transition permutation",
    "loss": "mean(((restricted_next-target)/stored_float32_train_scale)^2)",
    "precision": "float32; no AMP/TF32",
    "cpu_threads": 2,
    "data_location": "one float32 CPU state store; gather/transfer only each batch",
    "timing": "synchronized; inclusive=gather+H2D+compute; compute excludes transfer; first3training updates excluded, including coldcache/Adam allocation",
    "inference_batches": [8, 1],
    "inference_warmup_calls": 3,
    "inference_timed_calls": 10,
}


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_hash(state):
    values = np.ascontiguousarray(state, dtype="<f8")
    digest = hashlib.sha256(str(values.shape).encode("ascii"))
    digest.update(memoryview(values).cast("B"))
    return digest.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _bound(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("bound file is absent or outside its declared root")
    return path


def _sources():
    return {name: _hash(REPO_ROOT / name) for name in SOURCE_PATHS}


def _peak_rss_bytes():
    try:
        import resource

        units = 1 if platform.system() == "Darwin" else 1024
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * units)
    except ImportError:
        return None


def validate_parent(parent, source_root, *, unit_fixture=False):
    """Hash all C evidence before decoding trajectory arrays; no solver import."""
    parent, source_root = Path(parent), Path(source_root)
    digest = _hash(parent / "artifact_manifest.json")
    if not unit_fixture and digest != PARENT_MANIFEST_SHA256:
        raise ValueError("parent is not the pinned qualified C packet")
    manifest = _read(parent / "artifact_manifest.json")
    expected_id = PARENT_RUN_ID + ("__UNIT_FIXTURE" if unit_fixture else "")
    if manifest["run_id"] != expected_id or manifest["source_stable"] is not True:
        raise ValueError("parent identity or source stability mismatch")
    if set(manifest["sources"]) != set(PARENT_SOURCE_PATHS):
        raise ValueError("parent source closure mismatch")
    for name, value in manifest["sources"].items():
        if _hash(_bound(source_root, name)) != value:
            raise ValueError("parent archived source hash mismatch")
    if {f.name for f in parent.iterdir()} != set(manifest["artifacts"]) | {
        "artifact_manifest.json"
    }:
        raise ValueError("parent inventory mismatch")
    for name, value in manifest["artifacts"].items():
        if Path(name).name != name or _hash(_bound(parent, name)) != value:
            raise ValueError("parent artifact hash mismatch")
    result = _read(parent / "result.json")
    launch = _read(parent / "launch.json")
    n, steps = (16, 4) if unit_fixture else (256, 512)
    if (
        result["run_id"] != expected_id
        or launch["run_id"] != expected_id
        or result["status"] != "completed"
        or result["engineering_gates_pass"] is not True
        or result["source_stable"] is not True
        or result["parent_evidence_stable"] is not True
        or result["sources_before"] != manifest["sources"]
        or result["sources_after"] != manifest["sources"]
        or launch["sources"] != manifest["sources"]
        or launch["protocol"] != result["protocol"]
        or result["protocol"]["resolution"] != n
        or result["protocol"]["steps"] != steps
        or result["protocol"]["macro_dt"] != 0.05
        or result["initial_recipe"]["zero_mean_velocity"] is not True
        or result["seed_roles"] != [{"seed": s, "role": r} for s, r in SEED_ROLES]
        or [(c["seed"], c["role"]) for c in result["cases"]] != list(SEED_ROLES)
        or result["retained_transitions_by_role"]
        != {"train": 8 * steps, "development": 4 * steps}
        or set(result["numeric_gates"]) != set(GATE_LIMITS)
    ):
        raise ValueError("parent completion, roles or physical contract mismatch")
    for name, limit in GATE_LIMITS.items():
        gate = result["numeric_gates"][name]
        if (
            gate["pass"] is not True
            or gate["complete"] is not True
            or gate["limit"] != limit
            or gate["sample_count"] != gate["expected_sample_count"]
            or gate["sample_count"] <= 0
            or not np.isfinite(gate["maximum"])
            or not 0 <= gate["maximum"] <= limit
        ):
            raise ValueError("parent numerical qualification incomplete")
    for case in result["cases"]:
        if (
            case != _read(parent / f"case_{case['seed']}.json")
            or case["status"] != "completed"
            or case["last_retained_step"] != steps
            or [row["step"] for row in case["trajectory_diagnostics"]]
            != list(range(steps + 1))
            or case["maximum_palinstrophy"]["complete_trajectory"] is not True
            or any(
                case["config"][key] != value
                for key, value in {
                    "resolution": n,
                    "viscosity": 0.01,
                    "linear_drag": 0.1,
                    "forcing_amplitude": 1.0,
                    "forcing_wavenumber": 4,
                    "macro_dt": 0.05,
                    "domain_length": 2 * np.pi,
                }.items()
            )
        ):
            raise ValueError("parent case/state contract mismatch")
    return result, {
        "manifest_sha256": digest,
        "artifacts": {**manifest["artifacts"], "artifact_manifest.json": digest},
        "sources": manifest["sources"],
    }


@dataclass
class Population:
    states: np.ndarray
    train_scale: float
    metadata: dict

    def batch(self, indices, role="train"):
        count, offset = {"train": (8, 0), "development": (4, 8)}[role]
        indices = np.asarray(indices)
        steps = self.states.shape[1] - 1
        if (
            indices.ndim != 1
            or not len(indices)
            or indices.dtype.kind not in "iu"
            or np.any(indices < 0)
            or np.any(indices >= count * steps)
        ):
            raise ValueError("batch indices must name transitions within one role")
        trajectories, times = offset + indices // steps, indices % steps
        return (
            torch.from_numpy(self.states[trajectories, times]),
            torch.from_numpy(self.states[trajectories, times + 1]),
        )


def load_population(parent, source_root, *, unit_fixture=False, receipt=None):
    started = perf_counter()
    receipt = {} if receipt is None else receipt
    result, evidence = validate_parent(parent, source_root, unit_fixture=unit_fixture)
    receipt.update(parent_validated=True, evidence=evidence)
    n, steps = result["protocol"]["resolution"], result["protocol"]["steps"]
    states = np.empty((12, steps + 1, n, n), dtype=np.float32)
    modes = np.fft.fftfreq(n) * n
    mask = (np.abs(modes[:, None]) <= n // 3) & (np.abs(modes[None, :]) <= n // 3)
    mask[0, 0] = False
    input_digest = hashlib.sha256()
    square_sum = 0.0
    max_residual = 0.0
    count = 0
    receipt["arrays_decoded"] = True
    for trajectory, case in enumerate(result["cases"]):
        expected = 0
        for block in case["blocks"]:
            if block["file"] not in evidence["artifacts"]:
                raise ValueError("trajectory block is not manifest-bound")
            with np.load(Path(parent) / block["file"], allow_pickle=False) as data:
                for index, state in zip(data["steps"], data["states"], strict=True):
                    if (
                        index != expected
                        or state.dtype != np.float64
                        or state.shape != (n, n)
                        or not np.isfinite(state).all()
                    ):
                        raise ValueError(
                            "nonfinite, repeated, missing or mistyped state"
                        )
                    digest = _state_hash(state)
                    if (
                        digest
                        != case["trajectory_diagnostics"][expected]["state_sha256"]
                    ):
                        raise ValueError("state/diagnostic hash mismatch")
                    projected = np.fft.ifft2(np.fft.fft2(state) * mask).real
                    rms = float(np.sqrt(np.mean(state**2)))
                    residual = float(np.sqrt(np.mean((projected - state) ** 2))) / max(
                        rms, np.finfo(float).eps
                    )
                    if not np.isfinite(residual) or residual > 1e-11:
                        raise ValueError("source state is not canonical")
                    max_residual = max(max_residual, residual)
                    states[trajectory, expected] = state
                    if not np.isfinite(states[trajectory, expected]).all():
                        raise ValueError("source state is not representable in float32")
                    if trajectory < 8 and expected < steps:
                        square_sum += float(np.mean(state**2, dtype=np.float64))
                        input_digest.update(
                            f"{case['seed']}:{expected}:{digest}\n".encode()
                        )
                        count += 1
                    expected += 1
        if expected != steps + 1:
            raise ValueError("trajectory does not contain every native state")
    scale = float(np.sqrt(square_sum / count))
    if count != 8 * steps or not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid training-input RMS")
    metadata = {
        "shape": list(states.shape),
        "dtype": states.dtype.str,
        "state_store_bytes": states.nbytes,
        "state_store_sha256": hashlib.sha256(memoryview(states).cast("B")).hexdigest(),
        "train_input_rms_float64": scale,
        "training_input_state_count": count,
        "training_input_identity_sha256": input_digest.hexdigest(),
        "training_input_steps": [0, steps - 1],
        "development_used_for_scaling": False,
        "terminal_train_targets_used_for_scaling": False,
        "transition_counts": {"train": 8 * steps, "development": 4 * steps},
        "maximum_source_canonical_residual": max_residual,
        "load_and_validate_seconds": perf_counter() - started,
        "process_peak_rss_after_load_bytes": _peak_rss_bytes(),
    }
    receipt.update(metadata)
    return Population(states, scale, metadata)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _fit_resource(population, device, output, record, *, unit_fixture):
    config = record["model_config"]
    batch_size, updates = record["recipe"]["batch_size"], record["recipe"]["updates"]
    torch.manual_seed(17)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(17)
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    model = PeriodicVorticityPCNO(**config, train_scale=population.train_scale).to(
        device
    )
    torch.nn.init.zeros_(model.pcno.fc2.weight)
    torch.nn.init.zeros_(model.pcno.fc2.bias)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    _sync(device)
    record.update(
        model_config=config,
        batch_size=batch_size,
        expected_updates=updates,
        setup_seconds=perf_counter() - started,
        train_scale_model_float32=float(model.train_scale),
        zero_final_head=True,
        parameter_count=sum(p.numel() for p in model.parameters()),
        state_dict_tensor_bytes=sum(
            v.numel() * v.element_size() for v in model.state_dict().values()
        ),
        updates=[],
        inference={},
    )
    order = torch.randperm(
        population.metadata["transition_counts"]["train"],
        generator=torch.Generator().manual_seed(17),
    ).numpy()
    model.train()
    with (output / "updates.jsonl").open("x", encoding="utf-8") as log:
        for update in range(updates):
            selected = order[update * batch_size : (update + 1) * batch_size]
            _sync(device)
            started = perf_counter()
            inputs, targets = population.batch(selected)
            inputs, targets = inputs.to(device), targets.to(device)
            _sync(device)
            transfer_seconds = perf_counter() - started
            compute_started = perf_counter()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs)
            loss = (
                ((prediction["next_state"] - targets) / model.train_scale)
                .square()
                .mean()
            )
            loss_value = float(loss.detach())
            if not np.isfinite(loss_value):
                raise RuntimeError("nonfinite resource loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float("inf"), error_if_nonfinite=True
            )
            optimizer.step()
            _sync(device)
            row = {
                "update": update,
                "train_transition_indices": selected.tolist(),
                "loss_normalized_mse": loss_value,
                "unclipped_gradient_norm": float(norm),
                "gather_h2d_seconds": transfer_seconds,
                "compute_seconds": perf_counter() - compute_started,
                "inclusive_seconds": perf_counter() - started,
            }
            record["updates"].append(row)
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
    model.eval()
    del inputs, targets, prediction, loss
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        for size in (batch_size, 1):
            rows = []
            selected = order[:size]
            for index in range(13):
                _sync(device)
                started = perf_counter()
                inputs, _ = population.batch(selected)
                inputs = inputs.to(device)
                _sync(device)
                compute_started = perf_counter()
                prediction = model(inputs)
                _sync(device)
                row = {
                    "gather_h2d_seconds": compute_started - started,
                    "compute_seconds": perf_counter() - compute_started,
                    "inclusive_seconds": perf_counter() - started,
                }
                if not all(
                    torch.isfinite(value).all() for value in prediction.values()
                ):
                    raise RuntimeError("nonfinite resource inference")
                if index >= 3:
                    rows.append(row)
            record["inference"][str(size)] = {
                "warmup_calls_excluded": 3,
                "train_transition_indices": selected.tolist(),
                "timings": rows,
                "median_compute_seconds": float(
                    np.median([r["compute_seconds"] for r in rows])
                ),
                "median_inclusive_seconds": float(
                    np.median([r["inclusive_seconds"] for r in rows])
                ),
            }
    record.update(
        timing_warmup_updates_excluded=3,
        median_update_compute_seconds=float(
            np.median([r["compute_seconds"] for r in record["updates"][3:]])
        ),
        median_update_inclusive_seconds=float(
            np.median([r["inclusive_seconds"] for r in record["updates"][3:]])
        ),
        optimizer_steps=updates,
    )


def run_resource(parent, source_root, output, device_name, *, unit_fixture=False):
    if device_name != ("cpu" if unit_fixture else "cuda"):
        raise ValueError(
            "only explicit tiny fixtures use CPU; real resource preset uses CUDA"
        )
    output, parent, source_root = Path(output), Path(parent), Path(source_root)
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    sources_before = _sources()
    device = torch.device(device_name)
    record = {
        "run_id": RUN_ID + ("__UNIT_FIXTURE" if unit_fixture else ""),
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "device": device_name,
        "sources_before": sources_before,
        "data": {},
        "stage": "device_preflight",
        "expected_parent_manifest_sha256": None
        if unit_fixture
        else PARENT_MANIFEST_SHA256,
        "model_config": {
            "resolution": 16,
            "width": 8,
            "depth": 2,
            "modes": 2,
            "fc_dim": 16,
        }
        if unit_fixture
        else MODEL_CONFIG,
        "recipe": {
            **RESOURCE_RECIPE,
            "batch_size": 2,
            "updates": 5,
            "inference_batches": [2, 1],
        }
        if unit_fixture
        else RESOURCE_RECIPE,
        "process_peak_rss_before_load_bytes": _peak_rss_bytes(),
    }
    _write(output / "launch.json", record)
    try:
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        record["stage"] = "parent_loading"
        population = load_population(
            parent, source_root, unit_fixture=unit_fixture, receipt=record["data"]
        )
        record["stage"] = "resource_updates"
        _fit_resource(population, device, output, record, unit_fixture=unit_fixture)
        record["status"] = "completed"
    except (
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
        FloatingPointError,
        MemoryError,
    ) as error:
        record.update(
            status="invalid_parent"
            if record["stage"] == "parent_loading"
            and isinstance(error, (ValueError, KeyError, FileNotFoundError))
            else "failed",
            error_type=type(error).__name__,
            error=str(error),
        )
    sources_after = _sources()
    evidence = record["data"].get("evidence")
    parent_stable = (
        None
        if evidence is None
        else all(
            (root / name).is_file() and _hash(root / name) == digest
            for root, hashes in (
                (parent, evidence["artifacts"]),
                (source_root, evidence["sources"]),
            )
            for name, digest in hashes.items()
        )
    )
    if sources_before != sources_after:
        record["status"] = "invalid_source"
    elif parent_stable is False:
        record["status"] = "invalid_parent"
    record.update(
        sources_after=sources_after,
        source_stable=sources_before == sources_after,
        parent_evidence_stable=parent_stable,
        seconds=perf_counter() - started,
        finished_utc=datetime.now(timezone.utc).isoformat(),
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device)
        if device.type == "cuda" and torch.cuda.is_available()
        else None,
        peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(device)
        if device.type == "cuda" and torch.cuda.is_available()
        else None,
        process_peak_rss_bytes=_peak_rss_bytes(),
        runtime={
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "system": platform.system(),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device)
            if device.type == "cuda" and torch.cuda.is_available()
            else None,
        },
        scientific_training=False,
        checkpoint_saved=False,
        solver_calls=0,
        development_model_evaluation=False,
        protected_access=False,
        network_access=False,
        interpretation="Resource-only fixed16-update measurement, not convergence, held-out accuracy, rollout or correction efficacy. Peakmemory includes coldcache and optimizer state. No checkpoint is retained.",
    )
    _write(output / "result.json", record)
    _write(
        output / "artifact_manifest.json",
        {
            "run_id": record["run_id"],
            "status": record["status"],
            "sources": sources_before,
            "source_stable": record["source_stable"],
            "artifacts": {
                path.name: _hash(path)
                for path in sorted(output.iterdir())
                if path.is_file()
            },
        },
    )
    return record


def clean_lr(update, *, unit_fixture=False):
    """One-indexed frozen warmup/cosine schedule, then one constant extension."""
    warmup, base, extension = (2, 8, 4) if unit_fixture else (512, 32768, 16384)
    if (
        isinstance(update, bool)
        or not isinstance(update, int)
        or not 1 <= update <= base + extension
    ):
        raise ValueError("update lies outside the frozen Clean schedule")
    if update <= warmup:
        return 1e-3 * update / warmup
    if update > base:
        return 1e-4
    return 1e-4 + 0.5 * (1e-3 - 1e-4) * (
        1 + math.cos(math.pi * (update - warmup) / (base - warmup))
    )


class EpochSampler:
    """Complete seeded transition permutations; no dropped or repeated tail."""

    def __init__(self, size, batch_size=8, seed=17):
        if (
            not isinstance(size, int)
            or not isinstance(batch_size, int)
            or size < 1
            or batch_size < 1
            or size % batch_size
        ):
            raise ValueError(
                "population size must be a positive multiple of batch size"
            )
        self.size, self.batch_size = size, batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(size, generator=self.generator)
        self.cursor, self.epoch = 0, 0

    def next_indices(self):
        if self.cursor == self.size:
            self.order = torch.randperm(self.size, generator=self.generator)
            self.cursor = 0
            self.epoch += 1
        selected = (
            self.order[self.cursor : self.cursor + self.batch_size].numpy().copy()
        )
        self.cursor += self.batch_size
        return selected

    def state_dict(self):
        return {
            "size": self.size,
            "batch_size": self.batch_size,
            "order": self.order.clone(),
            "cursor": self.cursor,
            "epoch": self.epoch,
            "generator": self.generator.get_state(),
        }

    def load_state_dict(self, state):
        order = state["order"].cpu()
        if (
            state["size"] != self.size
            or state["batch_size"] != self.batch_size
            or order.dtype != torch.int64
            or order.shape != (self.size,)
            or not torch.equal(torch.sort(order).values, torch.arange(self.size))
            or not 0 <= state["cursor"] <= self.size
            or state["cursor"] % self.batch_size
            or state["epoch"] < 0
        ):
            raise ValueError("invalid sampler continuation state")
        self.order, self.cursor, self.epoch = (
            order.clone(),
            state["cursor"],
            state["epoch"],
        )
        self.generator.set_state(state["generator"].cpu())


PAIR_COLUMNS = (
    "learned_sse",
    "raw_sse",
    "target_sse",
    "input_sse",
    "persistence_sse",
    "projection_sse",
)


def summarize_teacher(rows, *, train_scale, node_count, steps, bands):
    """Pool spatial SSE in float64; never average per-state relative errors."""
    if (
        not np.isfinite(train_scale)
        or train_scale <= 0
        or node_count <= 0
        or steps <= 0
    ):
        raise ValueError("invalid teacher metric scale or shape")
    if (
        len(bands) != 4
        or bands[0] != 0
        or bands[-1] != steps
        or any(a >= b for a, b in pairwise(bands))
    ):
        raise ValueError("three ordered bands must partition the input steps")
    trajectory, times = (
        np.asarray(rows["trajectory_index"]),
        np.asarray(rows["input_step"]),
    )
    if (
        trajectory.shape != (12 * steps,)
        or times.shape != trajectory.shape
        or trajectory.dtype.kind not in "iu"
        or times.dtype.kind not in "iu"
        or np.any(trajectory >= 12)
        or np.any(trajectory < 0)
        or np.any(times >= steps)
        or np.any(times < 0)
        or not np.array_equal(
            np.sort(trajectory * steps + times), np.arange(12 * steps)
        )
    ):
        raise ValueError("teacher rows require complete unique train/development pairs")
    arrays = {name: np.asarray(rows[name], dtype=np.float64) for name in PAIR_COLUMNS}
    if any(
        a.shape != trajectory.shape or not np.isfinite(a).all() or np.any(a < 0)
        for a in arrays.values()
    ):
        raise ValueError(
            "teacher squared norms must be aligned finite nonnegative arrays"
        )

    def aggregate(selected):
        totals = {
            name: float(np.sum(value[selected], dtype=np.float64))
            for name, value in arrays.items()
        }
        if not all(np.isfinite(v) for v in totals.values()):
            raise ValueError("nonfinite pooled teacher norm")
        ratio = lambda numerator, denominator: (
            math.sqrt(numerator / denominator) if denominator > 0 else None
        )
        count = int(np.count_nonzero(selected))
        return {
            "pair_count": count,
            "squared_norm_sums": totals,
            "relative_l2": ratio(totals["learned_sse"], totals["target_sse"]),
            "normalized_mse": totals["learned_sse"]
            / (count * node_count * train_scale**2),
            "persistence_relative_l2": ratio(
                totals["persistence_sse"], totals["target_sse"]
            ),
            "zero_relative_l2": ratio(totals["target_sse"], totals["target_sse"]),
            "learned_over_persistence": ratio(
                totals["learned_sse"], totals["persistence_sse"]
            ),
            "raw_relative_l2": ratio(totals["raw_sse"], totals["target_sse"]),
            "projection_relative_l2": ratio(
                totals["projection_sse"], totals["target_sse"]
            ),
        }

    result = {
        "train": aggregate(trajectory < 8),
        "development": aggregate(trajectory >= 8),
        "trajectories": [],
    }
    for index, (seed, role) in enumerate(SEED_ROLES):
        selected = trajectory == index
        result["trajectories"].append(
            {
                "seed": seed,
                "role": role,
                **aggregate(selected),
                "bands": [
                    {
                        "start": start,
                        "stop": stop,
                        **aggregate(selected & (times >= start) & (times < stop)),
                    }
                    for start, stop in pairwise(bands)
                ],
            }
        )
    return result


def check_clean_readiness(summary):
    global_error = summary["development"]["relative_l2"]
    dev = [row for row in summary["trajectories"] if row["role"] == "development"]
    values = [band["learned_over_persistence"] for row in dev for band in row["bands"]]
    coverage = [row["seed"] for row in dev] == [
        s for s, r in SEED_ROLES if r == "development"
    ] and all(len(row["bands"]) == 3 for row in dev)
    finite = lambda value: value is not None and np.isfinite(value) and value >= 0
    global_pass = bool(global_error <= 0.02) if finite(global_error) else None
    band_pass = (
        bool(all(v <= 0.5 for v in values))
        if coverage and all(finite(v) for v in values)
        else None
    )
    return {
        "global_pass": global_pass,
        "band_pass": band_pass,
        "competence_pass": bool(global_pass and band_pass)
        if global_pass is not None and band_pass is not None
        else None,
        "development_relative_l2_limit": 0.02,
        "band_ratio_limit": 0.5,
        "band_ratios": values,
        "expected_band_count": 12,
    }


def clean_decision(previous_error, current_error, competence_pass, *, extended=False):
    resolved = (
        previous_error is not None
        and current_error is not None
        and np.isfinite(previous_error)
        and np.isfinite(current_error)
        and previous_error > 0
        and current_error >= 0
    )
    improving = bool(current_error < 0.95 * previous_error) if resolved else None
    accepted = competence_pass is True and improving is False
    extend = improving is True and not extended
    return {
        "relative_improvement": float(1 - current_error / previous_error)
        if resolved
        else None,
        "still_improving": improving,
        "extend": extend,
        "accepted_terminal": accepted,
        "extended_terminal": extended,
        "verdict": "extend_once"
        if extend
        else "accepted"
        if accepted
        else "needs_diagnosis",
    }


def save_clean_checkpoint(path, model, optimizer, sampler, *, update, identity):
    """Atomic rolling checkpoint; caller separately preserves fixed terminals."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    cuda = next(model.parameters()).device.type == "cuda"
    value = {
        "identity": identity,
        "update": update,
        "schedule_position": update,
        "model": model.state_dict(),
        "model_training": model.training,
        "optimizer": optimizer.state_dict(),
        "sampler": sampler.state_dict(),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if cuda else [],
        },
    }
    torch.save(value, temporary)
    temporary.replace(path)
    return _hash(path)


def load_clean_checkpoint(path, model, optimizer, sampler, *, identity):
    # Only locally produced, manifest-bound checkpoint files enter this helper.
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value["identity"] != identity or value["update"] != value["schedule_position"]:
        raise ValueError(
            "checkpoint source/parent/config or schedule identity mismatch"
        )
    model.load_state_dict(value["model"])
    model.train(value["model_training"])
    optimizer.load_state_dict(value["optimizer"])
    sampler.load_state_dict(value["sampler"])
    random.setstate(value["rng"]["python"])
    np.random.set_state(value["rng"]["numpy"])
    torch.set_rng_state(value["rng"]["torch_cpu"])
    if value["rng"]["torch_cuda"]:
        if not torch.cuda.is_available():
            raise ValueError("CUDA RNG state cannot be restored without CUDA")
        torch.cuda.set_rng_state_all(value["rng"]["torch_cuda"])
    return value["update"]


def _teacher_evaluation(model, population, output, update, bands, *, receipt=None):
    started = perf_counter()
    device = next(model.parameters()).device
    steps, n = population.states.shape[1] - 1, population.states.shape[-1]
    rows = {
        "trajectory_index": np.repeat(np.arange(12), steps),
        "input_step": np.tile(np.arange(steps), 12),
    }
    for name in (*PAIR_COLUMNS, "raw_output_sse", "next_output_sse"):
        rows[name] = np.empty(12 * steps, dtype=np.float64)
    sentinels = (0, 8 * steps - 1, 8 * steps, 12 * steps - 1)
    fields = {name: [] for name in ("input", "target", "raw", "next")}
    model.eval()
    with torch.no_grad():
        for role, offset, count in (
            ("train", 0, 8 * steps),
            ("development", 8 * steps, 4 * steps),
        ):
            for start in range(0, count, 8):
                indices = np.arange(start, min(start + 8, count))
                inputs, targets = population.batch(indices, role)
                inputs, targets = inputs.to(device), targets.to(device)
                predicted = model(inputs)
                if receipt is not None:
                    receipt["teacher_forward_calls"] += 1
                raw, nxt = predicted["raw_next"], predicted["next_state"]
                x, y, r, z = (a.double() for a in (inputs, targets, raw, nxt))
                arrays = {
                    "learned_sse": z - y,
                    "raw_sse": r - y,
                    "target_sse": y,
                    "input_sse": x,
                    "persistence_sse": x - y,
                    "projection_sse": r - z,
                    "raw_output_sse": r,
                    "next_output_sse": z,
                }
                for name, value in arrays.items():
                    reduced = value.square().sum(dim=(-2, -1)).cpu().numpy()
                    if not np.isfinite(reduced).all():
                        raise RuntimeError("nonfinite teacher-forced evaluation")
                    rows[name][offset + indices] = reduced
                for local, index in enumerate(offset + indices):
                    if int(index) in sentinels:
                        for name, value in zip(fields, (inputs, targets, raw, nxt)):
                            fields[name].append(value[local].cpu().numpy().copy())
    summary = summarize_teacher(
        rows,
        train_scale=float(model.train_scale),
        node_count=n * n,
        steps=steps,
        bands=bands,
    )
    rows["sentinel_global_indices"] = np.asarray(sentinels)
    rows.update(
        {"sentinel_" + name: np.stack(values) for name, values in fields.items()}
    )
    path = output / f"teacher_{update:06d}.npz"
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as stream:
        np.savez_compressed(stream, **rows)
    temporary.replace(path)
    return {
        "update": update,
        "summary": summary,
        "file": path.name,
        "sha256": _hash(path),
        "readiness": check_clean_readiness(summary),
        "seconds": perf_counter() - started,
    }


def _terminal_replay(model, evaluation_path):
    device = next(model.parameters()).device
    model.eval()
    with np.load(evaluation_path, allow_pickle=False) as saved, torch.no_grad():
        result = model(torch.from_numpy(saved["sentinel_input"]).to(device))
        errors = {}
        for name, key in (("raw", "raw_next"), ("next", "next_state")):
            values = result[key].cpu().numpy().astype(np.float64)
            expected = saved["sentinel_" + name].astype(np.float64)
            numerator = np.sqrt(np.mean((values - expected) ** 2, axis=(-2, -1)))
            denominator = np.sqrt(np.mean(expected**2, axis=(-2, -1)))
            relative = np.divide(
                numerator,
                denominator,
                out=np.full_like(numerator, np.inf),
                where=denominator > 0,
            )
            relative[(denominator == 0) & (numerator == 0)] = 0
            if not np.isfinite(relative).all() or np.any(relative > 1e-6):
                raise RuntimeError("terminal checkpoint sentinel replay failed")
            errors[name] = relative.tolist()
    return {"relative_rms_limit": 1e-6, "errors": errors, "passed": True}


def run_clean(parent, source_root, output, device_name, *, unit_fixture=False):
    """Frozen one-step pilot; no rollout or best-checkpoint selection exists."""
    if device_name != ("cpu" if unit_fixture else "cuda"):
        raise ValueError("real Clean uses CUDA; only explicit unit fixtures use CPU")
    output, parent, source_root = Path(output), Path(parent), Path(source_root)
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    device = torch.device(device_name)
    sources_before = _sources()
    base, every, warmup, extension = (
        (8, 2, 2, 4) if unit_fixture else (32768, 4096, 512, 16384)
    )
    bands = (0, 1, 2, 4) if unit_fixture else (0, 64, 256, 512)
    config = (
        {"resolution": 16, "width": 8, "depth": 2, "modes": 2, "fc_dim": 16}
        if unit_fixture
        else MODEL_CONFIG
    )
    recipe = {
        "base_updates": base,
        "evaluation_interval": every,
        "warmup_updates": warmup,
        "extension_updates": extension,
        "extension_compare_lag": 2 * every,
        "batch_size": 8,
        "seed": 17,
        "zero_final_head": True,
        "bands": list(bands),
        "optimizer": {
            "name": "Adam",
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.0,
        },
        "learning_rate": "1e-3*u/warmup, then cosine phase(u-warmup)/(base-warmup) to1e-4; extension constant1e-4",
        "sampling": "complete seeded without-replacement transition permutation per epoch",
        "precision": "float32; no AMP/TF32; no gradient clipping; abort nonfinite loss/gradient",
        "loss": "mean(((restricted_next-target)/stored_float32_train_scale)^2)",
        "teacher_metrics": "Per-pair float64 spatial SSE from float32 inputs/targets/raw/restricted predictions; aggregate squared norms, not means of ratios",
        "extension_rule": "current_error < .95*error_two_evaluations_earlier; at most one extension",
        "competence": {
            "pooled_dev_relative_l2": 0.02,
            "each_dev_trajectory_band_learned_over_persistence": 0.5,
        },
        "selection": "accepted terminal only; competence plus resolved not-improving rule; never rollout selection",
        "checkpoint_retention": "atomic latest plus base terminal and optional extended terminal",
        "terminal_replay_relative_rms_limit": 1e-6,
        "sentinels": "first/last training transition, first/last development transition",
    }
    record = {
        "run_id": CLEAN_RUN_ID + ("__UNIT_FIXTURE" if unit_fixture else ""),
        "status": "running",
        "stage": "device_preflight",
        "device": device_name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sources_before": sources_before,
        "expected_parent_manifest_sha256": None
        if unit_fixture
        else PARENT_MANIFEST_SHA256,
        "model_config": config,
        "recipe": recipe,
        "data": {},
        "updates_completed": 0,
        "teacher_forward_calls": 0,
        "evaluations": [],
        "terminal_checkpoints": {},
        "decision": None,
        "accepted_checkpoint": None,
    }
    _write(output / "launch.json", record)
    try:
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        record["stage"] = "parent_loading"
        population = load_population(
            parent, source_root, unit_fixture=unit_fixture, receipt=record["data"]
        )
        record["stage"] = "model_setup"
        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(17)
            torch.cuda.reset_peak_memory_stats(device)
        model = PeriodicVorticityPCNO(**config, train_scale=population.train_scale).to(
            device
        )
        torch.nn.init.zeros_(model.pcno.fc2.weight)
        torch.nn.init.zeros_(model.pcno.fc2.bias)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=clean_lr(1, unit_fixture=unit_fixture),
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
        )
        sampler = EpochSampler(population.metadata["transition_counts"]["train"])
        identity = {
            "run_id": record["run_id"],
            "sources": sources_before,
            "parent_manifest_sha256": record["data"]["evidence"]["manifest_sha256"],
            "state_store_sha256": population.metadata["state_store_sha256"],
            "train_scale_float64": population.train_scale,
            "train_scale_model_float32": float(model.train_scale),
            "model_config": config,
            "recipe": recipe,
        }
        record["checkpoint_identity"] = identity
        record["parameter_count"] = sum(p.numel() for p in model.parameters())
        record["state_dict_tensor_bytes"] = sum(
            v.numel() * v.element_size() for v in model.state_dict().values()
        )

        def evaluate(update):
            record["stage"] = "teacher_evaluation"
            evaluation = _teacher_evaluation(
                model, population, output, update, bands, receipt=record
            )
            record["evaluations"].append(evaluation)
            with (output / "evaluations.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps(evaluation, allow_nan=False) + "\n")
                log.flush()
            record["stage"] = "checkpoint_write"
            digest = save_clean_checkpoint(
                output / "latest_checkpoint.pt",
                model,
                optimizer,
                sampler,
                update=update,
                identity=identity,
            )
            record["latest_checkpoint"] = {
                "file": "latest_checkpoint.pt",
                "sha256": digest,
                "update": update,
            }
            _write(output / "progress.json", record)
            return evaluation

        evaluate(0)
        target = base
        with (output / "updates.jsonl").open("x", encoding="utf-8") as log:
            while record["updates_completed"] < target:
                record["stage"] = "training"
                update = record["updates_completed"] + 1
                selected = sampler.next_indices()
                lr = clean_lr(update, unit_fixture=unit_fixture)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                model.train()
                _sync(device)
                step_started = perf_counter()
                inputs, targets = population.batch(selected)
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(inputs)
                loss = (
                    ((prediction["next_state"] - targets) / model.train_scale)
                    .square()
                    .mean()
                )
                loss_value = float(loss.detach())
                if not np.isfinite(loss_value):
                    raise RuntimeError("nonfinite Clean training loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float("inf"), error_if_nonfinite=True
                )
                optimizer.step()
                _sync(device)
                row = {
                    "update": update,
                    "epoch": sampler.epoch,
                    "sampler_cursor": sampler.cursor,
                    "train_transition_indices": selected.tolist(),
                    "learning_rate": lr,
                    "loss_normalized_mse": loss_value,
                    "unclipped_gradient_norm": float(norm),
                    "inclusive_seconds": perf_counter() - step_started,
                }
                record["updates_completed"] = update
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
                del inputs, targets, prediction, loss
                if update % every == 0:
                    optimizer.zero_grad(set_to_none=True)
                    evaluation = evaluate(update)
                    if update == target:
                        terminal = output / f"terminal_{update:06d}.pt"
                        if terminal.exists():
                            raise FileExistsError(terminal)
                        digest = save_clean_checkpoint(
                            terminal,
                            model,
                            optimizer,
                            sampler,
                            update=update,
                            identity=identity,
                        )
                        if (
                            load_clean_checkpoint(
                                terminal, model, optimizer, sampler, identity=identity
                            )
                            != update
                        ):
                            raise RuntimeError("terminal checkpoint update mismatch")
                        replay = _terminal_replay(model, output / evaluation["file"])
                        record["terminal_checkpoints"][str(update)] = {
                            "file": terminal.name,
                            "sha256": digest,
                            "replay": replay,
                        }
                        earlier = next(
                            e
                            for e in record["evaluations"]
                            if e["update"] == update - 2 * every
                        )
                        decision = clean_decision(
                            earlier["summary"]["development"]["relative_l2"],
                            evaluation["summary"]["development"]["relative_l2"],
                            evaluation["readiness"]["competence_pass"],
                            extended=update > base,
                        )
                        record["decision"] = decision
                        if decision["extend"]:
                            if update != base:
                                raise RuntimeError(
                                    "second automatic extension is forbidden"
                                )
                            target = base + extension
                        elif decision["accepted_terminal"]:
                            record["accepted_checkpoint"] = terminal.name
                        _write(output / "progress.json", record)
        record["status"] = "completed"
    except (
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
        FloatingPointError,
        MemoryError,
    ) as error:
        record.update(
            status="invalid_parent"
            if record["stage"] == "parent_loading"
            and isinstance(error, (ValueError, KeyError, FileNotFoundError))
            else "failed",
            error_type=type(error).__name__,
            error=str(error),
        )
    sources_after = _sources()
    evidence = record["data"].get("evidence")
    parent_stable = (
        None
        if evidence is None
        else all(
            (root / name).is_file() and _hash(root / name) == digest
            for root, hashes in (
                (parent, evidence["artifacts"]),
                (source_root, evidence["sources"]),
            )
            for name, digest in hashes.items()
        )
    )
    if sources_after != sources_before:
        record["status"] = "invalid_source"
    elif parent_stable is False:
        record["status"] = "invalid_parent"
    if record["status"] != "completed":
        record["accepted_checkpoint"] = None
    record.update(
        sources_after=sources_after,
        source_stable=sources_after == sources_before,
        parent_evidence_stable=parent_stable,
        seconds=perf_counter() - started,
        finished_utc=datetime.now(timezone.utc).isoformat(),
        pilot_verdict="accepted_terminal"
        if record["accepted_checkpoint"]
        else "needs_diagnosis",
        scientific_training=not unit_fixture and record["updates_completed"] > 0,
        scientific_training_requested=not unit_fixture,
        teacher_evaluations_completed=len(record["evaluations"]),
        rollouts_evaluated=False,
        development_used_for_updates=False,
        protected_access=False,
        solver_calls=0,
        network_access=False,
        process_peak_rss_bytes=_peak_rss_bytes(),
        runtime={
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "system": platform.system(),
            "cuda": torch.version.cuda,
        },
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device)
        if device.type == "cuda" and torch.cuda.is_available()
        else None,
        peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(device)
        if device.type == "cuda" and torch.cuda.is_available()
        else None,
        interpretation="One-step Clean competence/convergence pilot only. No rollout, displaced-input qualification, correction ranking or prospective claim. Terminal acceptance requires all competence gates and a resolved plateau check.",
    )
    _write(output / "result.json", record)
    _write(
        output / "artifact_manifest.json",
        {
            "run_id": record["run_id"],
            "status": record["status"],
            "sources": sources_before,
            "source_stable": record["source_stable"],
            "parent_manifest_sha256": None
            if evidence is None
            else evidence["manifest_sha256"],
            "artifacts": {
                path.name: _hash(path)
                for path in sorted(output.iterdir())
                if path.is_file()
            },
        },
    )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--parent-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), required=True)
    parser.add_argument("--phase", choices=("resource", "clean"), default="resource")
    args = parser.parse_args()
    runner = run_clean if args.phase == "clean" else run_resource
    result = runner(args.parent, args.parent_source, args.output, args.device)
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
