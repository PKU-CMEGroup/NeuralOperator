"""Matched clean-continuation/recovery/relabeling from one fixed PCNO parent.

Run ``python -m scripts.time_dependent_no.fit_kolmogorov_paired --help``.
Each invocation validates, resource-tests, or fits exactly one arm. Validation
never deserializes a checkpoint; resource updates retain no checkpoint and are
never resumed by fitting. The small unit fixture is API-only, not a CLI option.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import io
import json
import math
import platform
import random
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from scripts.time_dependent_no import fit_kolmogorov_clean as clean
from scripts.time_dependent_no import generate_kolmogorov_paired_bank as bank

core = bank.core
RUN_ID = "CM_NEXT_KF_PAIRED_ADAPT_20260912A"
REPO_ROOT = Path(__file__).resolve().parents[2]
ARM_NAMES = ("clean_continuation", "recovery", "dynamics")
TOTAL_UPDATES = 4096
RESOURCE_UPDATES = 16
WALL_SECONDS = 3 * 3600
MINIMUM_FREE_BYTES = 512 * 1024**2
PINS = {
    "population": bank.PINS["population"],
    "extension": bank.PINS["extension"],
    "bank": (
        bank.RUN_ID,
        "58adafa766985007556676f78440a1e0ada85793ecd61f898a6c868deb7adeed",
        "93f4c2378b7a57a99aa4e6272a2ebf184eadb985df1c716c37d61ff0a6866c1d",
    ),
    "parent": (
        "CM_NEXT_KF_CLEAN32_MATCHED_20260909B",
        "437f826a273afc331ce5a1441ffd57f230b28bb27d94bc198080dca64f45fa76",
        "69eb5b04ce8182086e65b098bf001cc9d7404452677830b7ce1cde81a32df1b7",
    ),
}
CHECKPOINT_SHA256 = "6f8b106eb3758ac4cdb08ece4313969835bdc9c77afdaffab3f055810c9d0dbf"
SOURCE_PATHS = tuple(
    dict.fromkeys(
        (
            "scripts/time_dependent_no/fit_kolmogorov_paired.py",
            "tests/time_dependent_no/test_fit_kolmogorov_paired.py",
            *clean.SOURCE_PATHS,
            *bank.SOURCE_PATHS,
        )
    )
)
RECIPE = {
    "updates": TOTAL_UPDATES,
    "batch_size_per_branch": 8,
    "seed": 17,
    "clean_sampler_seed": 17,
    "bank_sampler_seed": 1701,
    "optimizer": "fresh Adam",
    "lr": 1e-4,
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "weight_decay": 0.0,
    "branch_weights": [0.5, 0.5],
    "loss": "0.5*mean(((next_clean-y_clean)/state_scale)^2) + 0.5*mean(((next_bank-y_bank)/state_scale)^2)",
    "precision": "FP32; no AMP/TF32/clipping",
    "zero_final_head": False,
    "sampling": "two independent without-replacement streams, replayed across arms; signed control IDs retained",
    "updates_per_clean_pair": 2,
    "updates_per_signed_row": 32,
    "selection": "fixed terminal; no validation, early stop or extension",
    "target_dtype": "stored FP64 A labels explicitly cast to FP32",
    "deployment": "unchanged PCNO next_state with shared output restriction; raw inputs unchanged",
}


def _indices(values, size):
    values = np.asarray(values)
    if (
        values.ndim != 1
        or not len(values)
        or values.dtype.kind not in "iu"
        or np.any(values < 0)
        or np.any(values >= size)
    ):
        raise ValueError("batch IDs must lie within the named training stream")
    return values


@dataclass
class PairedTrainingData:
    states: np.ndarray
    bank_inputs: np.ndarray
    bank_targets: np.ndarray
    anchor_indices: np.ndarray
    train_scale: float
    metadata: dict

    @property
    def clean_size(self):
        return len(self.states) * (self.states.shape[1] - 1)

    @property
    def bank_size(self):
        return 2 * len(self.bank_inputs)

    def clean_batch(self, ids):
        ids = _indices(ids, self.clean_size)
        trajectory, step = np.divmod(ids, self.states.shape[1] - 1)
        return (
            torch.from_numpy(self.states[trajectory, step]),
            torch.from_numpy(self.states[trajectory, step + 1]),
        )

    def bank_batch(self, ids, arm):
        if arm not in ARM_NAMES:
            raise ValueError("unknown corrective arm")
        ids = _indices(ids, self.bank_size)
        anchor, sign_slot = ids // 2, ids % 2 + 1
        input_slot = 0 if arm == "clean_continuation" else sign_slot
        target_slot = sign_slot if arm == "dynamics" else 0
        return (
            torch.from_numpy(self.bank_inputs[anchor, input_slot]),
            torch.from_numpy(self.bank_targets[anchor, target_slot]),
        )


def load_metadata(roots, bindings, deadline, sources, *, unit_fixture=False):
    """Read pinned metadata only; never use a mixed-role population loader."""
    manifests, results = {}, {}
    suffix = "__UNIT_FIXTURE" if unit_fixture else ""
    for role, root in roots.items():
        identity, manifest_sha, result_sha = PINS[role]
        if unit_fixture:
            manifest_sha = clean._hash(root / "artifact_manifest.json")
        manifest = json.loads(
            core.bound_bytes(
                root, "artifact_manifest.json", manifest_sha, bindings, role, deadline
            )
        )
        if unit_fixture:
            result_sha = manifest["artifacts"]["result.json"]
        if manifest["artifacts"]["result.json"] != result_sha:
            raise ValueError("parent result pin differs")
        result = json.loads(
            core.bound_bytes(root, "result.json", result_sha, bindings, role, deadline)
        )
        if (
            manifest["run_id"] != identity + suffix
            or result["run_id"] != identity + suffix
            or result["status"] != "completed"
            or manifest.get("status", "completed" if role == "population" else None)
            != "completed"
            or manifest["source_stable"] is not True
            or result["source_stable"] is not True
        ):
            raise ValueError("parent identity or completion differs")
        manifests[role], results[role] = manifest, result
    parent, paired = results["parent"], results["bank"]
    for role in ("population", "extension"):
        digest = bindings[(role, "artifact_manifest.json")][1]
        if (
            paired["input_files"][role + "/artifact_manifest.json"] != digest
            or digest not in parent["checkpoint_identity"]["input_manifests"]
            or results[role]["engineering_gates_pass"] is not True
        ):
            raise ValueError("training population binding or qualification differs")
    if (
        results["extension"]["parent_manifest_sha256"]
        != bindings[("population", "artifact_manifest.json")][1]
    ):
        raise ValueError("extension population binding differs")
    old, added, steps, _ = bank.selection(unit_fixture=unit_fixture)
    training_index = [
        dict(
            seed=s,
            role="train",
            packet="parent_C" if s in old else "this_packet",
            case_file=f"case_{s}.json",
        )
        for s in old + added
    ]
    expected_index = training_index + (
        []
        if unit_fixture
        else [
            dict(
                seed=s,
                role="development",
                packet="parent_C",
                case_file=f"case_{s}.json",
            )
            for s in range(2026090621, 2026090625)
        ]
    )
    if (
        results["extension"]["population_index"] != expected_index
        or parent["population_index"] != expected_index
        or parent["checkpoint_identity"]["population_index"] != expected_index
    ):
        raise ValueError("population roles/order differ")
    expected_count = len(old + added) * len(steps)
    if (
        paired["phase"] != "generate"
        or paired["qualification_passed"] is not True
        or paired["inputs_stable"] is not True
        or paired["solver_calls_completed"]
        != bank.expected_calls(unit_fixture=unit_fixture)
        or paired["solver_call_attempts"] != paired["solver_calls_completed"]
        or paired["seeds"] != list(old + added)
        or paired["input_steps"] != list(steps)
        or paired["signs"] != [1, -1]
        or len(paired["anchors"]) != expected_count
        or paired["expected_displaced_rows"] != 2 * expected_count
        or paired["validation_arrays_read"] is not False
        or paired["protected_access"] is not False
        or paired["model_calls"] != 0
        or paired["checkpoint_loads"] != 0
        or paired["optimization_steps"] != 0
    ):
        raise ValueError("paired bank completion or information boundary differs")
    total = 12 if unit_fixture else 49152
    terminal = parent["terminal_checkpoint"]
    identity = parent["checkpoint_identity"]
    scale = identity["train_scale_model_float32"]
    if (
        parent["phase"] != "fit"
        or parent["updates_completed"] != total
        or parent["input_evidence_stable"] is not True
        or terminal["update"] != total
        or terminal["file"] != f"terminal_{total:06d}.pt"
        or terminal["sha256"] != manifests["parent"]["artifacts"][terminal["file"]]
        or terminal["replay"]["passed"] is not True
        or identity["sources"] != manifests["parent"]["sources"]
        or identity["model_config"] != parent["model_config"]
        or not math.isfinite(scale)
        or scale <= 0
        or float(np.float32(identity["train_scale_float64"])) != scale
        or paired["train_scale"] != scale
    ):
        raise ValueError("fixed parent checkpoint or normalizer differs")
    if not unit_fixture:
        if (
            parent["model_config"] != clean.MODEL_CONFIG
            or terminal["sha256"] != CHECKPOINT_SHA256
        ):
            raise ValueError("fixed model/checkpoint differs")
        for role, paths in (
            ("bank", bank.SOURCE_PATHS),
            ("parent", clean.SOURCE_PATHS),
        ):
            for name in paths:
                if sources[name] != manifests[role]["sources"][name]:
                    raise ValueError(f"closed dependency changed: {name}")
        for obj, name in (
            (clean.EpochSampler, "scripts/time_dependent_no/fit_kolmogorov_clean.py"),
            (
                clean.PeriodicVorticityPCNO,
                "utility/time_dependent_no/pcno_kolmogorov.py",
            ),
            (
                core.bound_bytes,
                "scripts/time_dependent_no/screen_kolmogorov_common_solver.py",
            ),
        ):
            if Path(inspect.getfile(obj)).resolve() != REPO_ROOT / name:
                raise ValueError("imported implementation is outside source closure")
    return manifests, results


def load_training_data(
    roots, manifests, results, bindings, deadline, *, unit_fixture=False
):
    old, added, selected_steps, _ = bank.selection(unit_fixture=unit_fixture)
    seeds = old + added
    n, horizon = (16, 4) if unit_fixture else (256, 512)
    config = asdict(
        core.KolmogorovReferenceConfig(resolution=n, viscosity=0.01, macro_dt=0.05)
    )
    if results["bank"]["config"] != config:
        raise ValueError("bank numerical map differs")
    states = np.empty((len(seeds), horizon + 1, n, n), dtype=np.float32)
    for i, seed in enumerate(seeds):
        role = "population" if seed in old else "extension"
        root, manifest = roots[role], manifests[role]
        name = f"case_{seed}.json"
        case = json.loads(
            core.bound_bytes(
                root, name, manifest["artifacts"][name], bindings, role, deadline
            )
        )
        if (case["seed"], case["role"], case["status"], case["config"]) != (
            seed,
            "train",
            "completed",
            config,
        ):
            raise ValueError("clean training case identity differs")
        expected = 0
        # Bank bindings omit some early blocks. The clean branch needs every
        # state 0..horizon from each complete case, not only bank-bound blocks.
        for block in case["blocks"]:
            first, last = block["first_step"], block["last_step"]
            if first != expected or not first <= last <= horizon:
                raise ValueError("training blocks have a gap, overlap or wrong horizon")
            if manifest["artifacts"][block["file"]] != block["sha256"]:
                raise ValueError("training block binding differs")
            captured = core.bound_bytes(
                root, block["file"], block["sha256"], bindings, role, deadline
            )
            arrays = core.arrays_from_bytes(captured, deadline)
            del captured
            if (
                set(arrays) != {"states", "steps"}
                or arrays["states"].dtype != np.float64
                or arrays["states"].shape != (last - first + 1, n, n)
                or arrays["steps"].dtype != np.int64
                or not np.array_equal(arrays["steps"], np.arange(first, last + 1))
            ):
                raise ValueError("clean training block dtype/shape/order differs")
            states[i, first : last + 1] = arrays["states"]
            if not np.isfinite(states[i, first : last + 1]).all():
                raise ValueError("clean training state cannot be represented in FP32")
            expected = last + 1
            core.check_deadline(deadline, "clean training block conversion")
        if expected != horizon + 1:
            raise ValueError("training trajectory is incomplete")
    paired = results["bank"]
    count = len(seeds) * len(selected_steps)
    raw_inputs = np.empty((count, 3, n, n), dtype=np.float32)
    targets = np.empty_like(raw_inputs)
    anchor_indices = np.empty((count, 2), dtype=np.int64)
    for a, row in enumerate(paired["anchors"]):
        i, j = divmod(a, len(selected_steps))
        seed, step = seeds[i], selected_steps[j]
        name = f"bank_{seed}_{step:03d}.npz"
        if (
            row["seed"],
            row["role"],
            row["input_step"],
            row["output_step"],
            row["status"],
            row["file"],
        ) != (seed, "train", step, step + 1, "completed", name):
            raise ValueError("bank row role/time/order differs")
        if row["sha256"] != manifests["bank"]["artifacts"][name]:
            raise ValueError("bank row binding differs")
        captured = core.bound_bytes(
            roots["bank"], name, row["sha256"], bindings, "bank", deadline
        )
        # Refined B/C/D answers are diagnostic evidence, never training labels.
        with np.load(io.BytesIO(captured), allow_pickle=False) as archive:
            for key in ("raw_inputs", "A_0", "A_1", "A_2"):
                core.check_deadline(deadline, "bank column decoding")
                value = archive[key]
                shape = (3, n, n) if key == "raw_inputs" else (n, n)
                dtype = np.float32 if key == "raw_inputs" else np.float64
                if (
                    value.dtype != dtype
                    or value.shape != shape
                    or not np.isfinite(value).all()
                    or core.array_hash(value) != row["array_hashes"][key]
                ):
                    raise ValueError("bank column dtype/shape/hash differs")
                if key == "raw_inputs":
                    raw_inputs[a] = value
                else:
                    targets[a, int(key[-1])] = value.astype(np.float32)
        del captured
        if not np.array_equal(raw_inputs[a, 0], states[i, step]):
            raise ValueError(
                "bank clean input differs from the complete training store"
            )
        if not np.isfinite(targets[a]).all():
            raise ValueError("bank target cannot be represented in FP32")
        anchor_indices[a] = (i, step)
        core.check_deadline(deadline, "bank column conversion")
    metadata = {
        "seeds": list(seeds),
        "horizon": horizon,
        "clean_pairs": len(seeds) * horizon,
        "signed_bank_rows": count * 2,
        "bank_order": "seed,input_step,sign(+1,-1)",
        "state_store_sha256": core.array_hash(states),
        "bank_input_sha256": core.array_hash(raw_inputs),
        "bank_target_fp32_sha256": core.array_hash(targets),
        "anchor_indices_sha256": core.array_hash(anchor_indices),
        "bytes": states.nbytes
        + raw_inputs.nbytes
        + targets.nbytes
        + anchor_indices.nbytes,
        "normalizer": "inherited parent FP32 state scale; no refit",
        "target_map": paired["target_map"],
        "validation_arrays_read": False,
    }
    core.check_deadline(deadline, "training store hashing")
    return PairedTrainingData(
        states, raw_inputs, targets, anchor_indices, paired["train_scale"], metadata
    )


def fresh_optimizer(model):
    return torch.optim.Adam(
        model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
    )


def load_parent_model(model, captured_bytes, parent_result, receipt):
    """Only called after capture/hash verification; never restore old Adam/RNG."""
    receipt["checkpoint_load_attempts"] = receipt.get("checkpoint_load_attempts", 0) + 1
    checkpoint = torch.load(
        io.BytesIO(captured_bytes), map_location="cpu", weights_only=False
    )
    identity = parent_result["checkpoint_identity"]
    if (
        checkpoint["identity"] != identity
        or checkpoint["update"] != parent_result["updates_completed"]
        or checkpoint["schedule_position"] != checkpoint["update"]
    ):
        raise ValueError("captured checkpoint identity/update differs")
    model.load_state_dict(checkpoint["model"], strict=True)
    if float(model.train_scale) != identity["train_scale_model_float32"]:
        raise ValueError("checkpoint model normalizer differs")
    if any(not torch.isfinite(t).all() for t in model.state_dict().values()):
        raise ValueError("nonfinite parent model state")
    model.train()
    receipt["checkpoint_loads"] = receipt.get("checkpoint_loads", 0) + 1


def paired_update(model, optimizer, clean_batch, bank_batch, device, deadline, receipt):
    core.check_deadline(deadline, "paired update")
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for inputs, targets in (clean_batch, bank_batch):
        if inputs.dtype != torch.float32 or targets.dtype != torch.float32:
            raise ValueError("learner inputs and targets must already be FP32")
        if not torch.isfinite(inputs).all() or not torch.isfinite(targets).all():
            raise ArithmeticError("nonfinite training branch")
        inputs, targets = inputs.to(device), targets.to(device)
        core.check_deadline(deadline, "training forward")
        receipt["model_forward_calls"] = receipt.get("model_forward_calls", 0) + 1
        predicted = model(inputs)["next_state"]
        if predicted.dtype != torch.float32 or predicted.shape != targets.shape:
            raise ValueError("restricted prediction dtype/shape differs")
        loss = torch.mean(((predicted - targets) / model.train_scale) ** 2)
        if not torch.isfinite(loss):
            raise ArithmeticError("nonfinite training loss")
        core.check_deadline(deadline, "training backward")
        (0.5 * loss).backward()
        losses.append(float(loss.detach()))
        del predicted, loss  # No retained first-branch activation graph.
        core.check_deadline(deadline, "training backward completion")
    norms = [
        torch.linalg.vector_norm(p.grad.detach()).double()
        for p in model.parameters()
        if p.grad is not None
    ]
    if not norms:
        raise ArithmeticError("no parameter gradient")
    gradient_l2 = float(torch.linalg.vector_norm(torch.stack(norms)))
    if not math.isfinite(gradient_l2):
        raise ArithmeticError("nonfinite training gradient")
    core.check_deadline(deadline, "optimizer step")
    receipt["optimizer_step_attempts"] = receipt.get("optimizer_step_attempts", 0) + 1
    optimizer.step()
    clean._sync(torch.device(device))
    receipt["updates_completed"] = receipt.get("updates_completed", 0) + 1
    core.check_deadline(deadline, "optimizer completion")
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ArithmeticError("nonfinite parameters after optimizer step")
    return dict(
        clean_mse_scaled=losses[0],
        bank_mse_scaled=losses[1],
        loss=0.5 * sum(losses),
        gradient_l2=gradient_l2,
    )


def save_terminal(
    path, model, optimizer, clean_sampler, bank_sampler, identity, update
):
    value = {
        "schema": "kolmogorov_paired_adaptation_v1",
        "identity": identity,
        "update": update,
        "parent_update": identity["parent_update"],
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "samplers": {
            "clean": clean_sampler.state_dict(),
            "bank": bank_sampler.state_dict(),
        },
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all()
            if next(model.parameters()).is_cuda
            else [],
        },
    }
    with Path(path).open("xb") as stream:
        torch.save(value, stream)
    return clean._hash(path)


def replay_terminal(model, data, output, terminal, identity, deadline, receipt):
    """Train-only serialization replay; no teacher archive or diagnostic ranking."""
    inputs, _ = data.clean_batch(np.array([0, data.clean_size - 1], dtype=np.int64))
    inputs = inputs.to(next(model.parameters()).device)
    predictions = []
    model.eval()
    for repeat in range(2):
        core.check_deadline(deadline, "terminal replay")
        if repeat:
            captured = (output / terminal["file"]).read_bytes()
            if hashlib.sha256(captured).hexdigest() != terminal["sha256"]:
                raise ValueError("terminal checkpoint changed before replay")
            receipt["checkpoint_load_attempts"] += 1
            value = torch.load(
                io.BytesIO(captured), map_location="cpu", weights_only=False
            )
            if value["identity"] != identity or value["update"] != terminal["update"]:
                raise ValueError("terminal checkpoint identity differs")
            current = model.state_dict()
            if set(value["model"]) != set(current) or any(
                value["model"][key].dtype != tensor.dtype
                or not torch.equal(value["model"][key], tensor.detach().cpu())
                for key, tensor in current.items()
            ):
                raise ValueError("terminal serialized model state differs")
            model.load_state_dict(value["model"], strict=True)
            receipt["checkpoint_loads"] += 1
            del captured, value
            core.check_deadline(deadline, "terminal deserialization")
        with torch.no_grad():
            receipt["model_forward_calls"] += 1
            result = model(inputs)
            predictions.append(
                {
                    k: result[k].detach().cpu().numpy().copy()
                    for k in ("raw_next", "next_state")
                }
            )
        core.check_deadline(deadline, "terminal replay forward completion")
    # Reuse Clean's per-sentinel 1e-6 gate: GPU scatter sums need not be
    # bitwise repeatable. Checkpoint and serialized model identities stay exact.
    errors = {}
    for key in predictions[0]:
        expected = predictions[0][key].astype(np.float64)
        actual = predictions[1][key].astype(np.float64)
        numerator = np.sqrt(np.mean((actual - expected) ** 2, axis=(-2, -1)))
        denominator = np.sqrt(np.mean(expected**2, axis=(-2, -1)))
        relative = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.inf),
            where=denominator > 0,
        )
        relative[(denominator == 0) & (numerator == 0)] = 0
        if not np.isfinite(relative).all() or np.any(relative > 1e-6):
            raise ValueError("terminal same-device replay differs")
        errors[key] = relative.tolist()
    np.savez_compressed(
        output / "terminal_replay.npz",
        inputs=inputs.cpu().numpy(),
        raw_next=predictions[1]["raw_next"],
        next_state=predictions[1]["next_state"],
        before_raw_next=predictions[0]["raw_next"],
        before_next_state=predictions[0]["next_state"],
    )
    core.check_deadline(deadline, "terminal replay serialization")
    return {
        "passed": True,
        "relative_rms_limit": 1e-6,
        "errors": errors,
        "clean_pair_ids": [0, data.clean_size - 1],
        "file": "terminal_replay.npz",
        "sha256": clean._hash(output / "terminal_replay.npz"),
    }


def run(
    population,
    extension,
    paired_bank,
    parent,
    output,
    *,
    phase,
    arm,
    device_name="cuda",
    unit_fixture=False,
):
    if not __debug__:
        raise RuntimeError("optimized Python is not supported")
    if phase not in ("validate", "resource", "fit") or arm not in ARM_NAMES:
        raise ValueError("unknown phase or arm")
    roots = {
        role: Path(value).resolve()
        for role, value in zip(PINS, (population, extension, paired_bank, parent))
    }
    output = Path(output).resolve()
    if (
        output == REPO_ROOT
        or REPO_ROOT.is_relative_to(output)
        or any(
            output.is_relative_to(root) or root.is_relative_to(output)
            for root in roots.values()
        )
        or len(set(roots.values())) != len(roots)
    ):
        raise ValueError("output overlaps source/input or input roles alias")
    output.mkdir(parents=True, exist_ok=False)
    started, sources, bindings = perf_counter(), {}, {}
    deadline = started + WALL_SECONDS
    updates = (
        (4 if unit_fixture else TOTAL_UPDATES)
        if phase == "fit"
        else (2 if unit_fixture else RESOURCE_UPDATES)
    )
    record = {
        "run_id": RUN_ID
        + "_"
        + phase.upper()
        + "_"
        + arm.upper()
        + ("__UNIT_FIXTURE" if unit_fixture else ""),
        "phase": phase,
        "arm": arm,
        "status": "running",
        "stage": "metadata",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "recipe": RECIPE,
        "expected_updates": updates if phase != "validate" else 0,
        "model_construction_attempts": 0,
        "model_created": False,
        "checkpoint_load_attempts": 0,
        "checkpoint_loads": 0,
        "model_forward_calls": 0,
        "optimizer_step_attempts": 0,
        "updates_completed": 0,
        "metadata_validated": False,
        "inputs_validated": False,
        "validation_arrays_read": False,
        "protected_access": False,
        "solver_calls": 0,
        "new_rollouts": False,
        "terminal_checkpoint": None,
        "terminal_candidate": None,
        "wall_seconds": WALL_SECONDS,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "system": platform.system(),
        },
        "interpretation": "Single-seed matched adaptation conditional on one parent and Gaussian law; no selection or rollout ranking here.",
    }
    history, clean_tape, bank_tape = [], [], []
    try:
        for name in SOURCE_PATHS:
            core.check_deadline(deadline, "source hashing")
            sources[name] = clean._hash(REPO_ROOT / name)
        manifests, results = load_metadata(
            roots, bindings, deadline, sources, unit_fixture=unit_fixture
        )
        record["metadata_validated"] = True
        record["stage"] = "training_data"
        data = load_training_data(
            roots, manifests, results, bindings, deadline, unit_fixture=unit_fixture
        )
        record["data"] = data.metadata
        previous = results["parent"]
        checkpoint = previous["terminal_checkpoint"]
        captured = core.bound_bytes(
            roots["parent"],
            checkpoint["file"],
            checkpoint["sha256"],
            bindings,
            "parent",
            deadline,
        )
        record["inputs_validated"] = True
        record["parent_checkpoint_sha256"] = checkpoint["sha256"]
        record["train_scale"] = data.train_scale
        record["model_config"] = previous["model_config"]
        if phase != "validate":
            device = torch.device(device_name)
            if not unit_fixture and (
                device.type != "cuda" or not torch.cuda.is_available()
            ):
                raise ValueError("scientific resource/fit phase requires CUDA")
            if shutil.disk_usage(output).free < (
                1024**2 if unit_fixture else MINIMUM_FREE_BYTES
            ):
                raise RuntimeError("insufficient output storage; no automatic cleanup")
            random.seed(17)
            np.random.seed(17)
            torch.manual_seed(17)
            torch.set_num_threads(2)
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            record["stage"] = "parent_initialization"
            core.check_deadline(deadline, "model construction")
            record["model_construction_attempts"] += 1
            model = clean.PeriodicVorticityPCNO(
                **previous["model_config"], train_scale=data.train_scale
            )
            record["model_created"] = True
            core.check_deadline(deadline, "model construction completion")
            load_parent_model(model, captured, previous, record)
            del captured
            core.check_deadline(deadline, "parent checkpoint deserialization")
            model.to(device).train()
            core.check_deadline(deadline, "model device transfer")
            optimizer = fresh_optimizer(model)
            clean_sampler = clean.EpochSampler(data.clean_size, 8, 17)
            bank_sampler = clean.EpochSampler(data.bank_size, 8, 1701)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            identity = {
                "run_id": record["run_id"],
                "sources": sources,
                "arm": arm,
                "parent_checkpoint_sha256": checkpoint["sha256"],
                "parent_update": previous["updates_completed"],
                "input_manifests": {
                    role: bindings[(role, "artifact_manifest.json")][1]
                    for role in roots
                },
                "model_config": previous["model_config"],
                "train_scale": data.train_scale,
                "data": data.metadata,
                "recipe": RECIPE,
            }
            record["checkpoint_identity"] = identity
            record["stage"] = "updates"
            for update in range(1, updates + 1):
                core.check_deadline(deadline, "sample gathering")
                clean._sync(device)
                tick = perf_counter()
                clean_ids, bank_ids = (
                    clean_sampler.next_indices(),
                    bank_sampler.next_indices(),
                )
                clean_tape.append(clean_ids)
                bank_tape.append(bank_ids)
                row = paired_update(
                    model,
                    optimizer,
                    data.clean_batch(clean_ids),
                    data.bank_batch(bank_ids, arm),
                    device,
                    deadline,
                    record,
                )
                row.update(update=update, seconds=perf_counter() - tick)
                history.append(row)
                if update % 128 == 0 or update == updates:
                    core._json(
                        output / "progress.json",
                        {"run_id": record["run_id"], "update": update, **row},
                    )
                    print(
                        json.dumps({"arm": arm, "update": update, "loss": row["loss"]}),
                        flush=True,
                    )
            if (
                record["updates_completed"] != updates
                or record["model_forward_calls"] != 2 * updates
            ):
                raise RuntimeError("adaptation update/forward accounting differs")
            if phase == "fit":
                record["stage"] = "terminal"
                core.check_deadline(deadline, "terminal serialization start")
                name = f"terminal_{updates:06d}.pt"
                terminal = {
                    "file": name,
                    "update": updates,
                    "sha256": save_terminal(
                        output / name,
                        model,
                        optimizer,
                        clean_sampler,
                        bank_sampler,
                        identity,
                        updates,
                    ),
                }
                record["terminal_candidate"] = terminal
                core.check_deadline(deadline, "terminal serialization")
                terminal["replay"] = replay_terminal(
                    model, data, output, terminal, identity, deadline, record
                )
            if device.type == "cuda":
                record["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(
                    device
                )
                record["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved(
                    device
                )
            warm = history[3:] or history
            record["warm_update_seconds"] = float(np.mean([r["seconds"] for r in warm]))
            record["projected_4096_update_seconds"] = (
                record["warm_update_seconds"] * TOTAL_UPDATES
            )
        core.check_deadline(deadline, "work completion")
        record["status"] = "completed"
    except Exception as error:
        record.update(
            status="incomplete_budget"
            if isinstance(error, core.BudgetExceeded)
            else "failed",
            error_type=type(error).__name__,
            error=str(error),
        )
    # These small receipts are also retained on failure; incomplete fits never
    # become valid terminals merely because partial model bytes exist.
    try:
        core._json(output / "history.json", history)
        if clean_tape:
            np.savez_compressed(
                output / "sampling.npz",
                clean=np.stack(clean_tape),
                bank=np.stack(bank_tape),
            )
        if record["status"] == "completed":
            core.check_deadline(deadline, "history and sampler serialization")
    except Exception as error:
        record["training_record_error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        if record["status"] == "completed":
            record.update(
                status="incomplete_budget"
                if isinstance(error, core.BudgetExceeded)
                else "failed"
            )
    record["work_seconds"] = perf_counter() - started
    final_started = perf_counter()
    after, checks = {}, {}
    for name in SOURCE_PATHS:
        try:
            after[name] = clean._hash(REPO_ROOT / name)
        except OSError:
            after[name] = None
    for (role, name), (path, expected) in bindings.items():
        try:
            checks[f"{role}/{name}"] = clean._hash(path) == expected
        except OSError:
            checks[f"{role}/{name}"] = False
    record.update(
        sources_before=sources,
        sources_after=after,
        source_stable=len(sources) == len(SOURCE_PATHS) and sources == after,
        input_files={f"{r}/{n}": sha for (r, n), (_, sha) in bindings.items()},
        input_checks=checks,
        inputs_stable=record["inputs_validated"]
        and bool(checks)
        and all(checks.values()),
        provenance_seconds=perf_counter() - final_started,
        seconds=perf_counter() - started,
        finished_utc=datetime.now(timezone.utc).isoformat(),
    )
    if record["status"] == "completed" and not (
        record["source_stable"] and record["inputs_stable"]
    ):
        record["status"] = "invalid_provenance"
    if record["status"] == "completed" and phase == "fit":
        record["terminal_checkpoint"] = record["terminal_candidate"]
    core._json(output / "result.json", record)
    core._json(
        output / "artifact_manifest.json",
        {
            "run_id": record["run_id"],
            "phase": phase,
            "arm": arm,
            "status": record["status"],
            "sources": sources,
            "source_stable": record["source_stable"],
            "artifacts": {
                p.name: clean._hash(p) for p in sorted(output.iterdir()) if p.is_file()
            },
        },
    )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("population", "extension", "bank", "parent", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("validate", "resource", "fit"), required=True
    )
    parser.add_argument("--arm", choices=ARM_NAMES, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = run(
        args.population,
        args.extension,
        args.bank,
        args.parent,
        args.output,
        phase=args.phase,
        arm=args.arm,
        device_name=args.device,
    )
    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "status": result["status"],
                "updates": result["updates_completed"],
            }
        )
    )
    raise SystemExit(0 if result["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
