#!/usr/bin/env python3
"""Train and rollout-select a conservative-residual PCNO on 2D Euler shards."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utility.time_dependent_no.pcno_euler2d import (  # noqa: E402
    Euler2DNormalization,
    PCNOEuler2DResidual,
    PCNOEuler2DShardStore,
    apply_admissible_primitive_noise,
    balanced_presentations,
    conservative_admissibility,
    digest_mapping,
    fit_normalization,
    fixed_tiny_presentations,
    homogeneous_presentation_batches,
    parameter_count,
    stratified_train_val_split,
    weighted_scaled_mse,
    weighted_scaled_relative_l2,
)

CHECKPOINT_SCHEMA_VERSION = 4
RESIDUAL_TARGET_KIND = "conservative_variable_residual"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--split-seed", type=int, default=20260718)
    parser.add_argument(
        "--split-mode",
        choices=("stratified", "manifest"),
        default="stratified",
        help=(
            "Use the legacy deterministic train/validation split or require the "
            "exact train/validation/test partition declared by the shard manifest."
        ),
    )
    parser.add_argument("--val-count", type=int, default=30)
    parser.add_argument("--step-stride", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--presentations-per-epoch", type=int, default=4096)
    parser.add_argument("--val-presentations", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--tiny-pairs",
        type=int,
        default=0,
        help="Repeat one immutable pair bank of this size; zero uses balanced sampling.",
    )
    parser.add_argument("--tiny-fit-rel-l2", type=float, default=0.01)
    parser.add_argument("--tiny-fit-loss-ratio", type=float, default=0.01)
    parser.add_argument(
        "--stop-on-tiny-fit",
        action="store_true",
        help="Stop after the immutable tiny bank first passes both gates.",
    )
    parser.add_argument("--stats-time-stride", type=int, default=1)
    parser.add_argument(
        "--mach-scale-floor",
        type=float,
        default=0.1,
        help="Lower bound for the training-set Mach standard deviation.",
    )
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--domain-lengths", type=float, nargs=2, default=(6.0, 2.0))
    parser.add_argument(
        "--layers", type=int, nargs="+", default=(128, 128, 128, 128, 128)
    )
    parser.add_argument("--fc-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--scheduler",
        choices=("constant", "onecycle"),
        default="constant",
        help="Use the strong residual-FNO recipe by default; OneCycle is explicit.",
    )
    parser.add_argument("--lr-div-factor", type=float, default=10.0)
    parser.add_argument("--lr-final-div-factor", type=float, default=1000.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--input-noise-std", type=float, default=0.0)
    parser.add_argument(
        "--generated-state-exposure-weight",
        type=float,
        default=0.0,
        help=(
            "Mix this fraction of detached two-call generated-state loss with "
            "the clean one-step anchor; zero disables exposure."
        ),
    )
    parser.add_argument("--rollout-every", type=int, default=5)
    parser.add_argument("--rollout-val-count", type=int, default=5)
    parser.add_argument("--rollout-steps", type=int, default=20)
    parser.add_argument("--rollout-start-frame", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", choices=("none", "bf16", "fp16"), default="bf16")
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Load model/split/scales and start a fresh matched continuation.",
    )
    checkpoint_group.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Resume model, optimizer, scheduler, epoch, and best-selection state.",
    )
    return parser.parse_args(argv)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate_args(args: argparse.Namespace, device: torch.device) -> None:
    positive_ints = {
        "epochs": args.epochs,
        "presentations_per_epoch": args.presentations_per_epoch,
        "val_presentations": args.val_presentations,
        "batch_size": args.batch_size,
        "step_stride": args.step_stride,
        "stats_time_stride": args.stats_time_stride,
        "rollout_every": args.rollout_every,
        "rollout_val_count": args.rollout_val_count,
        "rollout_steps": args.rollout_steps,
        "checkpoint_every": args.checkpoint_every,
    }
    for name, value in positive_ints.items():
        if value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.tiny_pairs < 0:
        raise ValueError("--tiny-pairs must be nonnegative")
    if args.stop_on_tiny_fit and args.tiny_pairs == 0:
        raise ValueError("--stop-on-tiny-fit requires --tiny-pairs")
    if args.rollout_start_frame < 0:
        raise ValueError("--rollout-start-frame must be nonnegative")
    if args.tiny_fit_rel_l2 <= 0.0 or args.tiny_fit_loss_ratio <= 0.0:
        raise ValueError("tiny-fit thresholds must be positive")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("invalid optimizer configuration")
    if args.gradient_clip < 0.0:
        raise ValueError("--gradient-clip must be nonnegative")
    if args.lr_div_factor <= 0.0 or args.lr_final_div_factor <= 0.0:
        raise ValueError("OneCycle division factors must be positive")
    if args.scheduler == "onecycle" and args.batch_size != 1:
        raise ValueError(
            "OneCycle currently requires --batch-size 1 for exact step accounting"
        )
    if args.input_noise_std < 0.0:
        raise ValueError("--input-noise-std must be nonnegative")
    if (
        not math.isfinite(args.generated_state_exposure_weight)
        or not 0.0 <= args.generated_state_exposure_weight < 1.0
    ):
        raise ValueError("--generated-state-exposure-weight must lie in [0,1)")
    if args.generated_state_exposure_weight > 0.0 and args.input_noise_std > 0.0:
        raise ValueError(
            "generated-state exposure and input noise are separate branches"
        )
    if args.generated_state_exposure_weight > 0.0 and args.tiny_pairs > 0:
        raise ValueError("generated-state exposure is not a tiny-fit mode")
    if not math.isfinite(args.mach_scale_floor) or args.mach_scale_floor <= 0.0:
        raise ValueError("--mach-scale-floor must be positive and finite")
    if args.amp != "none" and device.type != "cuda":
        raise ValueError("mixed precision requires CUDA; use --amp none on CPU")
    if (
        args.amp == "bf16"
        and device.type == "cuda"
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("the selected CUDA device does not support bfloat16")


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    result = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value)
        elif isinstance(value, tuple):
            result[key] = list(value)
        else:
            result[key] = value
    return result


def git_state() -> dict[str, Any]:
    def command(*arguments: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *arguments],
                cwd=ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    status = command("status", "--porcelain")
    return {
        "commit": command("rev-parse", "HEAD"),
        "branch": command("branch", "--show-current"),
        "dirty": bool(status and status != "unknown"),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def build_data_contract_summary(
    store: PCNOEuler2DShardStore,
    *,
    train_keys: Sequence[str],
    val_keys: Sequence[str],
    test_keys: Sequence[str],
    normalization: Euler2DNormalization,
) -> dict[str, Any]:
    """Bind the training data needed by evaluation and the later Line-4 handoff."""

    entries = {key: store.entry(key) for key in store.keys}
    split_contract = {
        "train": list(train_keys),
        "validation": list(val_keys),
        "test": list(test_keys),
        "split_groups": {key: entries[key].get("split_group_id") for key in store.keys},
    }
    geometry_by_trajectory = {
        key: entries[key].get("geometry_digest") for key in store.keys
    }
    source_artifacts = {
        key: entries[key].get("source_reference_sha256") for key in store.keys
    }
    node_counts = sorted({int(entry["num_nodes"]) for entry in entries.values()})
    edge_counts = sorted(
        {int(entry["num_directed_edges"]) for entry in entries.values()}
    )
    resolution_contract: dict[str, Any] = {
        "node_counts": node_counts,
        "directed_edge_counts": edge_counts,
        "coordinate_convention": store.manifest.get("coordinate_convention"),
    }
    time_contract: dict[str, Any] | None = None
    if store.manifest.get("dataset") == "shock_vortex_fv_family":
        first_key = store.keys[0]
        nodes = np.asarray(store.array(first_key, "nodes"), dtype=np.float64)
        nx = int(np.unique(nodes[:, 0]).size)
        ny = int(np.unique(nodes[:, 1]).size)
        if nx * ny != nodes.shape[0]:
            raise ValueError("shock-vortex PCNO nodes do not form the declared grid")
        resolution_contract["stored_grid"] = [nx, ny]
        physical_times = np.asarray(
            store.array(first_key, "physical_times"), dtype=np.float64
        )
        if (
            physical_times.ndim != 1
            or physical_times.size < 2
            or not np.all(np.isfinite(physical_times))
            or not np.all(np.diff(physical_times) > 0.0)
        ):
            raise ValueError("physical time coordinates are invalid")
        deltas = np.diff(physical_times)
        time_contract = {
            "physical_times_digest": digest_array(physical_times),
            "start_time": float(physical_times[0]),
            "final_time": float(physical_times[-1]),
            "saved_calls": int(physical_times.size - 1),
            "delta_t": float(deltas[0]),
            "uniform_delta_t": bool(
                np.allclose(deltas, deltas[0], rtol=0.0, atol=1.0e-14)
            ),
        }
        if not time_contract["uniform_delta_t"]:
            raise ValueError("shock-vortex saved times are not uniformly spaced")

    normalization_payload = normalization.to_dict()
    return {
        "schema": "pcno_euler2d_data_contract_v1",
        "dataset": store.manifest.get("dataset"),
        "data_manifest_digest": store.manifest_digest,
        "source_family_id": store.manifest.get("source_family_id"),
        "source_family_manifest_digest": store.manifest.get(
            "source_family_manifest_digest"
        ),
        "source_artifact_set_digest": digest_mapping(source_artifacts),
        "grouped_split_digest": digest_mapping(split_contract),
        "geometry_contract_digest": digest_mapping(geometry_by_trajectory),
        "unique_geometry_digests": sorted(set(geometry_by_trajectory.values())),
        "resolution_contract": resolution_contract,
        "resolution_contract_digest": digest_mapping(resolution_contract),
        "time_contract": time_contract,
        "time_contract_digest": (
            None if time_contract is None else digest_mapping(time_contract)
        ),
        "state_convention": store.manifest.get("state_convention"),
        "weight_provenance": normalization.weight_provenance,
        "normalization_digest": digest_mapping(normalization_payload),
        "mesh_to_graph_map": store.manifest.get("mesh_to_graph_map"),
        "line4_handoff": {
            "status": "prerequisites_only_baseline_not_frozen",
            "line4_training_truth_authorized": None,
            "line4_front_candidate_available": None,
        },
    }


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if (
        int(checkpoint.get("checkpoint_schema_version", -1))
        != CHECKPOINT_SCHEMA_VERSION
    ):
        raise ValueError(f"unsupported checkpoint schema in {path}")
    return checkpoint


def manifest_train_val_test_split(
    store: PCNOEuler2DShardStore,
) -> tuple[list[str], list[str], list[str]]:
    """Return an exact, complete manifest partition or fail closed."""

    split_names = ("train", "validation", "test")
    raw_splits = store.manifest.get("splits")
    if not isinstance(raw_splits, Mapping) or set(raw_splits) != set(split_names):
        raise ValueError(
            "manifest split mode requires exactly train, validation, and test splits"
        )
    splits: dict[str, list[str]] = {}
    for split_name in split_names:
        raw_keys = raw_splits[split_name]
        if not isinstance(raw_keys, list) or not all(
            isinstance(key, str) for key in raw_keys
        ):
            raise ValueError(f"manifest split {split_name!r} must be a list of keys")
        keys = list(raw_keys)
        if not keys:
            raise ValueError(f"manifest split {split_name!r} must not be empty")
        if len(set(keys)) != len(keys):
            raise ValueError(f"manifest split {split_name!r} contains duplicate keys")
        splits[split_name] = keys

    partition = [key for split_name in split_names for key in splits[split_name]]
    if len(set(partition)) != len(partition):
        raise ValueError("manifest train/validation/test splits overlap")
    store_keys = set(store.keys)
    partition_keys = set(partition)
    if partition_keys != store_keys:
        missing = sorted(store_keys - partition_keys)
        unknown = sorted(partition_keys - store_keys)
        raise ValueError(
            "manifest splits do not exactly partition the shard store: "
            f"missing={missing}, unknown={unknown}"
        )

    actual_counts = {name: len(splits[name]) for name in split_names}
    for field_name in ("declared_split_counts", "prepared_split_counts"):
        raw_counts = store.manifest.get(field_name)
        if not isinstance(raw_counts, Mapping):
            raise ValueError(f"manifest split mode requires {field_name}")
        counts = {name: int(raw_counts.get(name, -1)) for name in split_names}
        if counts != actual_counts:
            raise ValueError(
                f"{field_name} does not match the complete manifest partition: "
                f"{counts} != {actual_counts}"
            )

    for split_name in split_names:
        for key in splits[split_name]:
            if store.entry(key).get("split") != split_name:
                raise ValueError(
                    f"trajectory {key!r} entry disagrees with split {split_name!r}"
                )
    return splits["train"], splits["validation"], splits["test"]


def assert_checkpoint_contract(
    checkpoint: Mapping[str, Any],
    *,
    store: PCNOEuler2DShardStore,
    args: argparse.Namespace,
) -> None:
    if checkpoint["data_manifest_digest"] != store.manifest_digest:
        raise ValueError("checkpoint and current shard manifest digests differ")
    if int(checkpoint["step_stride"]) != args.step_stride:
        raise ValueError("checkpoint and requested step stride differ")
    saved_training_args = checkpoint.get("training_args", {})
    saved_split_mode = str(saved_training_args.get("split_mode", "stratified"))
    if saved_split_mode != args.split_mode:
        raise ValueError("checkpoint and requested split modes differ")
    if args.split_mode == "manifest":
        train_keys, val_keys, test_keys = manifest_train_val_test_split(store)
        checkpoint_partition = (
            [str(key) for key in checkpoint["train_keys"]],
            [str(key) for key in checkpoint["val_keys"]],
            [str(key) for key in checkpoint.get("test_keys", [])],
        )
        if checkpoint_partition != (train_keys, val_keys, test_keys):
            raise ValueError("checkpoint keys differ from the manifest partition")
    saved_mach_floor = float(checkpoint["normalization"]["mach_scale_floor"])
    if not math.isclose(
        saved_mach_floor, args.mach_scale_floor, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("checkpoint and requested Mach scale floor differ")
    actual = checkpoint["model_config"]
    saved_target_kind = str(
        saved_training_args.get(
            "target_kind",
            actual.get("target_kind", RESIDUAL_TARGET_KIND),
        )
    )
    if saved_target_kind != RESIDUAL_TARGET_KIND:
        raise ValueError("checkpoint target is not the conservative residual")
    expected = {
        "k_max": int(args.k_max),
        "domain_lengths": [float(value) for value in args.domain_lengths],
        "layers": [int(value) for value in args.layers],
        "fc_dim": int(args.fc_dim),
    }
    for key, value in expected.items():
        if actual[key] != value:
            raise ValueError(
                f"checkpoint model config mismatch for {key}: {actual[key]} != {value}"
            )


def build_model(
    args: argparse.Namespace,
    normalization: Euler2DNormalization,
    *,
    zero_initialize: bool,
) -> PCNOEuler2DResidual:
    """Construct the conservative-residual baseline."""

    return PCNOEuler2DResidual(
        normalization=normalization,
        k_max=args.k_max,
        domain_lengths=args.domain_lengths,
        layers=args.layers,
        fc_dim=args.fc_dim,
        zero_initialize=zero_initialize,
    )


def target_contract() -> dict[str, Any]:
    """Describe the conservative-residual supervision contract."""

    return {
        "target_kind": RESIDUAL_TARGET_KIND,
        "supervision": "decoded_next_state_loss",
        "coordinate_convention": "conservative_[rho,rho_u,rho_v,E]",
        "emits_physical_face_exchange": False,
        "reference_impulse_supervision": False,
        "conserved_totals_are_outcome_metrics_not_guaranteed_structure": True,
    }


def forward_sample(
    model: PCNOEuler2DResidual,
    sample: Mapping[str, torch.Tensor],
    current: torch.Tensor,
) -> torch.Tensor:
    return model(
        current,
        node_mask=sample["node_mask"],
        nodes=sample["nodes"],
        node_weights=sample["node_weights"],
        node_rhos=sample["node_rhos"],
        directed_edges=sample["directed_edges"],
        edge_gradient_weights=sample["edge_gradient_weights"],
        node_type=sample["node_type"],
        mach=sample["mach"],
    )


def autocast_context(device: torch.device, amp: str):
    if amp == "none":
        return nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def pair_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sample: Mapping[str, torch.Tensor],
    model: PCNOEuler2DResidual,
) -> tuple[torch.Tensor, torch.Tensor]:
    loss = weighted_scaled_mse(
        prediction,
        target,
        sample["node_weights"],
        sample["node_mask"],
        model.state_scale,
    )
    relative_l2 = weighted_scaled_relative_l2(
        prediction,
        target,
        sample["node_weights"],
        sample["node_mask"],
        model.state_scale,
    )
    return loss, relative_l2


def evaluate_pairs(
    model: PCNOEuler2DResidual,
    store: PCNOEuler2DShardStore,
    pairs: Sequence[tuple[str, int]],
    *,
    step_stride: int,
    batch_size: int,
    device: torch.device,
    amp: str,
) -> dict[str, float]:
    model.eval()
    losses = []
    relative_errors = []
    admissible = 0
    batches = homogeneous_presentation_batches(pairs, batch_size=batch_size)
    for key, time_indices in batches:
        sample = store.tensor_batch(
            key,
            time_indices,
            step_stride=step_stride,
            device=device,
        )
        with autocast_context(device, amp):
            prediction = forward_sample(model, sample, sample["current"])
        for batch_index in range(prediction.shape[0]):
            sample_slice = {
                name: value[batch_index : batch_index + 1]
                for name, value in sample.items()
            }
            loss, relative_l2 = pair_metrics(
                prediction[batch_index : batch_index + 1],
                sample["target"][batch_index : batch_index + 1],
                sample_slice,
                model,
            )
            losses.append(float(loss.detach().cpu()))
            relative_errors.append(float(relative_l2.detach().cpu()))
        diagnostics = conservative_admissibility(prediction.float(), gamma=model.gamma)
        per_sample_admissible = (
            diagnostics["admissible"].reshape(prediction.shape[0], -1).all(dim=1)
        )
        admissible += int(per_sample_admissible.sum().cpu())
    return {
        "loss": float(np.mean(losses)),
        "relative_l2": float(np.mean(relative_errors)),
        "admissible_fraction": admissible / len(pairs),
        "presentations": len(pairs),
    }


def finite_minimum(value: torch.Tensor) -> float | None:
    finite = value[torch.isfinite(value)]
    if finite.numel() == 0:
        return None
    return float(finite.min().cpu())


def failure_cause(
    prediction: torch.Tensor,
    *,
    gamma: float,
) -> tuple[str | None, dict[str, float | None]]:
    diagnostics = conservative_admissibility(prediction.float(), gamma=gamma)
    finite = bool(diagnostics["finite_components"].all())
    density = diagnostics["density"]
    internal_energy = diagnostics["internal_energy"]
    pressure = diagnostics["pressure"]
    summary = {
        "min_density": finite_minimum(density),
        "min_internal_energy": finite_minimum(internal_energy),
        "min_pressure": finite_minimum(pressure),
    }
    if (
        not finite
        or not bool(torch.isfinite(internal_energy).all())
        or not bool(torch.isfinite(pressure).all())
    ):
        return "nonfinite_state", summary
    if not bool((density > 0.0).all()):
        return "nonpositive_density", summary
    if not bool((internal_energy > 0.0).all()):
        return "nonpositive_internal_energy", summary
    if not bool((pressure > 0.0).all()):
        return "nonpositive_pressure", summary
    return None, summary


@torch.no_grad()
def rollout_trajectory(
    model: PCNOEuler2DResidual,
    store: PCNOEuler2DShardStore,
    key: str,
    *,
    step_stride: int,
    start_frame: int,
    num_steps: int,
    device: torch.device,
    amp: str,
) -> dict[str, Any]:
    states = store.states(key)
    available = (states.shape[0] - 1 - start_frame) // step_stride
    requested = min(int(num_steps), int(available))
    if requested < 1:
        raise ValueError(
            f"trajectory {key} has no rollout targets at the requested start/stride"
        )
    sample = store.tensor_sample(
        key, start_frame, step_stride=step_stride, device=device
    )
    current = sample["current"]
    errors = []
    minimums = {
        "min_density": math.inf,
        "min_internal_energy": math.inf,
        "min_pressure": math.inf,
    }
    failure = None
    model.eval()
    for call_index in range(requested):
        with autocast_context(device, amp):
            proposal = forward_sample(model, sample, current)
        failure, current_minimums = failure_cause(proposal, gamma=model.gamma)
        for name, value in current_minimums.items():
            if value is not None:
                minimums[name] = min(minimums[name], value)
        if failure is not None:
            break
        target_index = start_frame + (call_index + 1) * step_stride
        target = torch.as_tensor(
            np.array(states[target_index], copy=True),
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        relative_l2 = weighted_scaled_relative_l2(
            proposal.float(),
            target,
            sample["node_weights"],
            sample["node_mask"],
            model.state_scale,
        )
        errors.append(float(relative_l2.cpu()))
        current = proposal
    valid_length = len(errors)
    minimum_summary = {
        name: None if not math.isfinite(value) else value
        for name, value in minimums.items()
    }
    return {
        "trajectory": key,
        "requested_steps": requested,
        "valid_length": valid_length,
        "completed": valid_length == requested,
        "failure_cause": "completed" if valid_length == requested else failure,
        "survival_fraction": valid_length / requested,
        "final_relative_l2": errors[-1] if errors else None,
        "mean_prefix_relative_l2": float(np.mean(errors)) if errors else None,
        **minimum_summary,
    }


@torch.no_grad()
def evaluate_rollouts(
    model: PCNOEuler2DResidual,
    store: PCNOEuler2DShardStore,
    keys: Sequence[str],
    *,
    step_stride: int,
    start_frame: int,
    num_steps: int,
    device: torch.device,
    amp: str,
) -> dict[str, Any]:
    rows = [
        rollout_trajectory(
            model,
            store,
            key,
            step_stride=step_stride,
            start_frame=start_frame,
            num_steps=num_steps,
            device=device,
            amp=amp,
        )
        for key in keys
    ]
    completed = [row for row in rows if row["completed"]]
    scored = (
        completed
        if completed
        else [row for row in rows if row["final_relative_l2"] is not None]
    )
    mean_error = (
        float(np.mean([row["final_relative_l2"] for row in scored])) if scored else None
    )
    return {
        "trajectories": rows,
        "num_trajectories": len(rows),
        "completed": len(completed),
        "completion_rate": len(completed) / len(rows),
        "mean_survival_fraction": float(
            np.mean([row["survival_fraction"] for row in rows])
        ),
        "mean_selection_relative_l2": mean_error,
        "selection_population": "completed" if completed else "valid_prefix",
    }


def selection_tuple(
    rollout: Mapping[str, Any], one_step: Mapping[str, float]
) -> tuple[float, ...]:
    raw_error = rollout["mean_selection_relative_l2"]
    error = float(raw_error) if raw_error is not None else float("nan")
    if not math.isfinite(error):
        error = float(np.finfo(np.float64).max)
    return (
        float(rollout["completion_rate"]),
        -error,
        -float(one_step["relative_l2"]),
    )


def tiny_fit_snapshot(
    initial: Mapping[str, float],
    current: Mapping[str, float],
    *,
    relative_l2_threshold: float,
    loss_ratio_threshold: float,
) -> dict[str, Any]:
    initial_loss = float(initial["loss"])
    current_loss = float(current["loss"])
    loss_ratio = (
        current_loss / initial_loss
        if initial_loss > 0.0
        else (0.0 if current_loss == 0.0 else float(np.finfo(np.float64).max))
    )
    relative_l2 = float(current["relative_l2"])
    admissible_fraction = float(current["admissible_fraction"])
    return {
        "metrics": dict(current),
        "loss_ratio": loss_ratio,
        "relative_l2_threshold": float(relative_l2_threshold),
        "loss_ratio_threshold": float(loss_ratio_threshold),
        "passed": (
            relative_l2 <= relative_l2_threshold
            and loss_ratio <= loss_ratio_threshold
            and admissible_fraction == 1.0
        ),
    }


def tiny_fit_selection(snapshot: Mapping[str, Any]) -> tuple[float, ...]:
    metrics = snapshot["metrics"]
    relative_ratio = float(metrics["relative_l2"]) / float(
        snapshot["relative_l2_threshold"]
    )
    loss_ratio = float(snapshot["loss_ratio"]) / float(snapshot["loss_ratio_threshold"])
    maximum_ratio = max(relative_ratio, loss_ratio)
    return (
        float(bool(snapshot["passed"])),
        float(metrics["admissible_fraction"]),
        -maximum_ratio,
        -relative_ratio,
        -loss_ratio,
    )


def clone_state_dict_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def checkpoint_payload(
    *,
    epoch: int,
    model: PCNOEuler2DResidual,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    normalization: Euler2DNormalization,
    train_keys: Sequence[str],
    val_keys: Sequence[str],
    test_keys: Sequence[str],
    data_contract: Mapping[str, Any],
    resolved_target_contract: Mapping[str, Any],
    store: PCNOEuler2DShardStore,
    args: argparse.Namespace,
    best_selection: Sequence[float] | None,
    best_epoch: int | None,
    parent_checkpoint: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "model_config": model.model_config(),
        "normalization": normalization.to_dict(),
        "train_keys": list(train_keys),
        "val_keys": list(val_keys),
        "test_keys": list(test_keys),
        "split_mode": args.split_mode,
        "data_contract": dict(data_contract),
        "target_contract": dict(resolved_target_contract),
        "target_kind": RESIDUAL_TARGET_KIND,
        "normalization_digest": data_contract["normalization_digest"],
        "data_manifest_digest": store.manifest_digest,
        "step_stride": int(args.step_stride),
        "training_args": jsonable_args(args),
        "config_digest": digest_mapping(jsonable_args(args)),
        "best_selection": None if best_selection is None else list(best_selection),
        "best_epoch": best_epoch,
        "parent_checkpoint": parent_checkpoint,
        "git": git_state(),
        "boundary_mode": "model_all_nodes",
        "raw_recurrence": True,
        "inference_interventions": {
            "future_reference_boundary_values": False,
            "clipping": False,
            "primitive_floors": False,
            "limiter": False,
            "decode_reencode_projection": False,
        },
        "weight_provenance": normalization.weight_provenance,
        "generated_state_exposure": {
            "weight": float(args.generated_state_exposure_weight),
            "depth": 1 if args.generated_state_exposure_weight > 0.0 else 0,
            "first_call_gradient": "detached",
            "clean_one_step_anchor": (
                1.0 - float(args.generated_state_exposure_weight)
            ),
            "raw_generated_state": True,
        },
        "checkpoint_role": "training",
        "resume_supported": True,
    }


def train_epoch(
    model: PCNOEuler2DResidual,
    store: PCNOEuler2DShardStore,
    pairs: Sequence[tuple[str, int]],
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    step_stride: int,
    batch_size: int,
    batch_rng: np.random.Generator,
    device: torch.device,
    amp: str,
    input_noise_std: float,
    generated_state_exposure_weight: float,
    gradient_clip: float,
    noise_generator: torch.Generator,
) -> dict[str, float | None]:
    model.train()
    loss_sum = 0.0
    relative_error_sum = 0.0
    clean_loss_sum = 0.0
    clean_relative_error_sum = 0.0
    generated_loss_sum = 0.0
    generated_relative_error_sum = 0.0
    generated_input_error_sum = 0.0
    batches = homogeneous_presentation_batches(
        pairs,
        batch_size=batch_size,
        rng=batch_rng,
    )
    for key, time_indices in batches:
        sample = store.tensor_batch(
            key,
            time_indices,
            step_stride=step_stride,
            device=device,
        )
        if input_noise_std > 0.0:
            current = apply_admissible_primitive_noise(
                sample["current"],
                input_noise_std,
                gamma=model.gamma,
                generator=noise_generator,
            )
        else:
            current = sample["current"]
        generated_current = None
        generated_input_relative_l2 = None
        if generated_state_exposure_weight > 0.0:
            previous_sample = store.tensor_batch(
                key,
                [index - step_stride for index in time_indices],
                step_stride=step_stride,
                device=device,
            )
            with torch.no_grad(), autocast_context(device, amp):
                generated_current = forward_sample(
                    model,
                    previous_sample,
                    previous_sample["current"],
                ).detach()
                _, generated_input_relative_l2 = pair_metrics(
                    generated_current.float(),
                    sample["current"],
                    sample,
                    model,
                )
            generated_admissibility = conservative_admissibility(
                generated_current.float(),
                gamma=model.gamma,
            )
            if not bool(generated_admissibility["admissible"].all()):
                raise FloatingPointError(
                    "raw generated-state exposure produced an inadmissible "
                    f"input for trajectory {key} frames {time_indices}"
                )
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp):
            prediction = forward_sample(model, sample, current)
            clean_loss, clean_relative_l2 = pair_metrics(
                prediction, sample["target"], sample, model
            )
            if generated_current is None:
                generated_loss = None
                generated_relative_l2 = None
                loss = clean_loss
                relative_l2 = clean_relative_l2
            else:
                generated_prediction = forward_sample(
                    model,
                    sample,
                    generated_current,
                )
                generated_loss, generated_relative_l2 = pair_metrics(
                    generated_prediction,
                    sample["target"],
                    sample,
                    model,
                )
                loss = (
                    1.0 - generated_state_exposure_weight
                ) * clean_loss + generated_state_exposure_weight * generated_loss
                relative_l2 = (
                    (1.0 - generated_state_exposure_weight) * clean_relative_l2
                    + generated_state_exposure_weight * generated_relative_l2
                )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(
                f"nonfinite training loss for trajectory {key} frames {time_indices}"
            )
        scaler.scale(loss).backward()
        if gradient_clip > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        current_batch_size = len(time_indices)
        loss_sum += float(loss.detach().cpu()) * current_batch_size
        relative_error_sum += float(relative_l2.detach().cpu()) * current_batch_size
        clean_loss_sum += float(clean_loss.detach().cpu()) * current_batch_size
        clean_relative_error_sum += (
            float(clean_relative_l2.detach().cpu()) * current_batch_size
        )
        if generated_loss is not None:
            generated_loss_sum += (
                float(generated_loss.detach().cpu()) * current_batch_size
            )
            generated_relative_error_sum += (
                float(generated_relative_l2.detach().cpu()) * current_batch_size
            )
            generated_input_error_sum += (
                float(generated_input_relative_l2.detach().cpu()) * current_batch_size
            )
    return {
        "loss": loss_sum / len(pairs),
        "relative_l2": relative_error_sum / len(pairs),
        "clean_loss": clean_loss_sum / len(pairs),
        "clean_relative_l2": clean_relative_error_sum / len(pairs),
        "generated_loss": (
            generated_loss_sum / len(pairs)
            if generated_state_exposure_weight > 0.0
            else None
        ),
        "generated_relative_l2": (
            generated_relative_error_sum / len(pairs)
            if generated_state_exposure_weight > 0.0
            else None
        ),
        "generated_input_relative_l2": (
            generated_input_error_sum / len(pairs)
            if generated_state_exposure_weight > 0.0
            else None
        ),
        "generated_state_exposure_weight": generated_state_exposure_weight,
        "presentations": len(pairs),
        "optimizer_steps": len(batches),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    device = select_device(args.device)
    validate_args(args, device)
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    store = PCNOEuler2DShardStore(args.data_dir)
    checkpoint_path = args.resume_checkpoint or args.init_checkpoint
    checkpoint = load_checkpoint(checkpoint_path) if checkpoint_path else None

    if checkpoint is not None:
        assert_checkpoint_contract(checkpoint, store=store, args=args)
        train_keys = [str(key) for key in checkpoint["train_keys"]]
        val_keys = [str(key) for key in checkpoint["val_keys"]]
        test_keys = [str(key) for key in checkpoint.get("test_keys", [])]
        normalization = Euler2DNormalization.from_mapping(checkpoint["normalization"])
    else:
        if args.split_mode == "manifest":
            train_keys, val_keys, test_keys = manifest_train_val_test_split(store)
        else:
            train_keys, val_keys = stratified_train_val_split(
                store,
                val_count=args.val_count,
                seed=args.split_seed,
            )
            test_keys = []
        normalization = fit_normalization(
            store,
            train_keys,
            step_stride=args.step_stride,
            time_stride=args.stats_time_stride,
            gamma=float(store.manifest.get("gamma", 1.4)),
            mach_scale_floor=args.mach_scale_floor,
        )
    data_contract = build_data_contract_summary(
        store,
        train_keys=train_keys,
        val_keys=val_keys,
        test_keys=test_keys,
        normalization=normalization,
    )
    if checkpoint is not None and checkpoint.get("data_contract") not in (
        None,
        data_contract,
    ):
        raise ValueError("checkpoint and current resolved data contracts differ")
    if args.rollout_val_count > len(val_keys):
        raise ValueError("--rollout-val-count exceeds the validation split")
    rollout_selection_seed = args.seed + 1991
    if args.rollout_val_count == len(val_keys):
        rollout_keys = list(val_keys)
    else:
        _, rollout_keys = stratified_train_val_split(
            store,
            val_count=args.rollout_val_count,
            seed=rollout_selection_seed,
            keys=val_keys,
        )
    model = build_model(
        args,
        normalization,
        zero_initialize=checkpoint is None,
    ).to(device)
    tiny_bank = None
    if args.tiny_pairs:
        tiny_bank = fixed_tiny_presentations(
            store,
            train_keys,
            step_stride=args.step_stride,
            count=args.tiny_pairs,
            seed=args.seed + 17,
        )
    resolved_target_contract = target_contract()
    if checkpoint is not None:
        saved_target_contract = checkpoint.get("target_contract")
        if saved_target_contract not in (None, resolved_target_contract):
            raise ValueError("checkpoint and current target contracts differ")
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state"], strict=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    if args.scheduler == "constant":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda _: 1.0
        )
    else:
        total_steps = args.epochs * args.presentations_per_epoch
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.learning_rate,
            total_steps=total_steps,
            div_factor=args.lr_div_factor,
            final_div_factor=args.lr_final_div_factor,
        )
    start_epoch = 0
    best_selection: tuple[float, ...] | None = None
    best_epoch: int | None = None
    if args.resume_checkpoint is not None:
        if not bool(checkpoint.get("resume_supported", True)):
            raise ValueError("the selected checkpoint supports initialization only")
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        saved_selection = checkpoint.get("best_selection")
        best_selection = (
            None
            if saved_selection is None
            else tuple(float(x) for x in saved_selection)
        )
        best_epoch = checkpoint.get("best_epoch")
        if start_epoch >= args.epochs:
            raise ValueError("resume checkpoint has already reached --epochs")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "metrics.jsonl"
    if log_path.exists() and args.resume_checkpoint is None:
        raise FileExistsError(f"refusing to append to an existing run log: {log_path}")
    write_json(args.output_dir / "normalization.json", normalization.to_dict())
    split_payload = {
        "split_mode": args.split_mode,
        "split_seed": args.split_seed,
        "train_keys": train_keys,
        "val_keys": val_keys,
        "test_keys": test_keys,
        "rollout_selection_seed": rollout_selection_seed,
        "rollout_keys": rollout_keys,
        "tiny_bank": (
            None
            if tiny_bank is None
            else [
                {"trajectory": key, "time_index": int(time_index)}
                for key, time_index in tiny_bank
            ]
        ),
        "data_manifest_digest": store.manifest_digest,
    }
    write_json(args.output_dir / "split.json", split_payload)

    validation_pairs = balanced_presentations(
        store,
        val_keys,
        step_stride=args.step_stride,
        count=args.val_presentations,
        rng=np.random.default_rng(args.seed + 991),
    )
    if tiny_bank is not None:
        initial_train_metrics = evaluate_pairs(
            model,
            store,
            tiny_bank,
            step_stride=args.step_stride,
            batch_size=args.batch_size,
            device=device,
            amp=args.amp,
        )
    else:
        initial_train_metrics = None

    parent_checkpoint = None
    if args.init_checkpoint is not None:
        parent_checkpoint = {
            "path": str(args.init_checkpoint),
            "sha256": sha256_file(args.init_checkpoint),
            "epoch": int(checkpoint["epoch"]),
        }
    noise_generator = torch.Generator(device=device.type)
    noise_generator.manual_seed(args.seed + 101)
    scaler = torch.amp.GradScaler(device.type, enabled=(args.amp == "fp16"))
    run_start = perf_counter()
    last_train_metrics: dict[str, float | None] | None = None
    last_validation_metrics: dict[str, float] | None = None
    last_rollout_metrics: dict[str, Any] | None = None
    best_tiny_snapshot: dict[str, Any] | None = None
    best_tiny_selection: tuple[float, ...] | None = None
    best_tiny_epoch: int | None = None
    best_tiny_state: dict[str, torch.Tensor] | None = None

    for epoch in range(start_epoch, args.epochs):
        epoch_start = perf_counter()
        if tiny_bank is None:
            pairs = balanced_presentations(
                store,
                train_keys,
                step_stride=args.step_stride,
                count=args.presentations_per_epoch,
                rng=np.random.default_rng(args.seed + epoch),
                minimum_time_index=(
                    args.step_stride
                    if args.generated_state_exposure_weight > 0.0
                    else 0
                ),
            )
        else:
            repetitions = math.ceil(args.presentations_per_epoch / len(tiny_bank))
            pairs = (tiny_bank * repetitions)[: args.presentations_per_epoch]
        last_train_metrics = train_epoch(
            model,
            store,
            pairs,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            step_stride=args.step_stride,
            batch_size=args.batch_size,
            batch_rng=np.random.default_rng(args.seed + 100_000 + epoch),
            device=device,
            amp=args.amp,
            input_noise_std=args.input_noise_std,
            generated_state_exposure_weight=(args.generated_state_exposure_weight),
            gradient_clip=args.gradient_clip,
            noise_generator=noise_generator,
        )
        last_validation_metrics = evaluate_pairs(
            model,
            store,
            validation_pairs,
            step_stride=args.step_stride,
            batch_size=args.batch_size,
            device=device,
            amp=args.amp,
        )
        current_tiny_snapshot = None
        if tiny_bank is not None:
            current_tiny_metrics = evaluate_pairs(
                model,
                store,
                tiny_bank,
                step_stride=args.step_stride,
                batch_size=args.batch_size,
                device=device,
                amp=args.amp,
            )
            current_tiny_snapshot = tiny_fit_snapshot(
                initial_train_metrics,
                current_tiny_metrics,
                relative_l2_threshold=args.tiny_fit_rel_l2,
                loss_ratio_threshold=args.tiny_fit_loss_ratio,
            )
            current_tiny_selection = tiny_fit_selection(current_tiny_snapshot)
            if (
                best_tiny_selection is None
                or current_tiny_selection > best_tiny_selection
            ):
                best_tiny_snapshot = current_tiny_snapshot
                best_tiny_selection = current_tiny_selection
                best_tiny_epoch = epoch
                if bool(current_tiny_snapshot["passed"]):
                    best_tiny_state = clone_state_dict_cpu(model)
        stop_for_tiny_fit = bool(
            args.stop_on_tiny_fit
            and current_tiny_snapshot is not None
            and current_tiny_snapshot["passed"]
        )
        should_rollout = (
            (epoch + 1) % args.rollout_every == 0
            or epoch == start_epoch
            or epoch + 1 == args.epochs
        )
        if should_rollout:
            last_rollout_metrics = evaluate_rollouts(
                model,
                store,
                rollout_keys,
                step_stride=args.step_stride,
                start_frame=args.rollout_start_frame,
                num_steps=args.rollout_steps,
                device=device,
                amp=args.amp,
            )
            candidate = selection_tuple(last_rollout_metrics, last_validation_metrics)
            if best_selection is None or candidate > best_selection:
                best_selection = candidate
                best_epoch = epoch
                payload = checkpoint_payload(
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    normalization=normalization,
                    train_keys=train_keys,
                    val_keys=val_keys,
                    test_keys=test_keys,
                    data_contract=data_contract,
                    resolved_target_contract=resolved_target_contract,
                    store=store,
                    args=args,
                    best_selection=best_selection,
                    best_epoch=best_epoch,
                    parent_checkpoint=parent_checkpoint,
                )
                atomic_torch_save(payload, args.output_dir / "best.pt")
        record = {
            "epoch": epoch,
            "elapsed_seconds": perf_counter() - epoch_start,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train": last_train_metrics,
            "validation": last_validation_metrics,
            "rollout": last_rollout_metrics if should_rollout else None,
            "tiny_fit": current_tiny_snapshot,
            "best_epoch": best_epoch,
            "best_selection": best_selection,
            "gpu_max_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        if (
            (epoch + 1) % args.checkpoint_every == 0
            or epoch + 1 == args.epochs
            or stop_for_tiny_fit
        ):
            payload = checkpoint_payload(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                normalization=normalization,
                train_keys=train_keys,
                val_keys=val_keys,
                test_keys=test_keys,
                data_contract=data_contract,
                resolved_target_contract=resolved_target_contract,
                store=store,
                args=args,
                best_selection=best_selection,
                best_epoch=best_epoch,
                parent_checkpoint=parent_checkpoint,
            )
            atomic_torch_save(payload, args.output_dir / "last.pt")
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_rel_l2": last_train_metrics["relative_l2"],
                    "val_rel_l2": last_validation_metrics["relative_l2"],
                    "rollout_completion": (
                        None
                        if not should_rollout
                        else last_rollout_metrics["completion_rate"]
                    ),
                    "best_epoch": best_epoch,
                    "tiny_fit_passed": (
                        None
                        if current_tiny_snapshot is None
                        else current_tiny_snapshot["passed"]
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if stop_for_tiny_fit:
            break

    tiny_fit = None
    if tiny_bank is not None:
        final_tiny = evaluate_pairs(
            model,
            store,
            tiny_bank,
            step_stride=args.step_stride,
            batch_size=args.batch_size,
            device=device,
            amp=args.amp,
        )
        final_tiny_snapshot = tiny_fit_snapshot(
            initial_train_metrics,
            final_tiny,
            relative_l2_threshold=args.tiny_fit_rel_l2,
            loss_ratio_threshold=args.tiny_fit_loss_ratio,
        )
        final_tiny_selection = tiny_fit_selection(final_tiny_snapshot)
        if best_tiny_selection is None or final_tiny_selection > best_tiny_selection:
            best_tiny_snapshot = final_tiny_snapshot
            best_tiny_selection = final_tiny_selection
            best_tiny_epoch = epoch
            if bool(final_tiny_snapshot["passed"]):
                best_tiny_state = clone_state_dict_cpu(model)
        if best_tiny_snapshot is None or best_tiny_epoch is None:
            raise RuntimeError("tiny-fit evaluation did not produce a selection")
        tiny_fit = {
            "initial": initial_train_metrics,
            "final": final_tiny,
            "final_loss_ratio": final_tiny_snapshot["loss_ratio"],
            "final_passed": final_tiny_snapshot["passed"],
            "best": best_tiny_snapshot["metrics"],
            "best_loss_ratio": best_tiny_snapshot["loss_ratio"],
            "best_epoch": best_tiny_epoch,
            "relative_l2_threshold": args.tiny_fit_rel_l2,
            "loss_ratio_threshold": args.tiny_fit_loss_ratio,
            "passed": best_tiny_snapshot["passed"],
        }
        if best_tiny_state is not None:
            tiny_payload = checkpoint_payload(
                epoch=best_tiny_epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                normalization=normalization,
                train_keys=train_keys,
                val_keys=val_keys,
                test_keys=test_keys,
                data_contract=data_contract,
                resolved_target_contract=resolved_target_contract,
                store=store,
                args=args,
                best_selection=best_selection,
                best_epoch=best_epoch,
                parent_checkpoint=parent_checkpoint,
            )
            tiny_payload["model_state"] = best_tiny_state
            tiny_payload.pop("optimizer_state")
            tiny_payload.pop("scheduler_state")
            tiny_payload["checkpoint_role"] = "tiny_fit_selection"
            tiny_payload["resume_supported"] = False
            tiny_payload["tiny_fit"] = tiny_fit
            atomic_torch_save(tiny_payload, args.output_dir / "tiny_best.pt")

    summary = {
        "mode": f"pcno_euler2d_{RESIDUAL_TARGET_KIND}",
        "target_kind": RESIDUAL_TARGET_KIND,
        "target_contract": resolved_target_contract,
        "device": str(device),
        "amp": args.amp,
        "parameter_count": parameter_count(model),
        "model_config": model.model_config(),
        "normalization": normalization.to_dict(),
        "normalization_digest": data_contract["normalization_digest"],
        "data_contract": data_contract,
        "split_mode": args.split_mode,
        "train_trajectories": len(train_keys),
        "validation_trajectories": len(val_keys),
        "test_trajectories": len(test_keys),
        "step_stride": args.step_stride,
        "batch_size": args.batch_size,
        "boundary_mode": "model_all_nodes",
        "raw_recurrence": True,
        "inference_interventions": {
            "future_reference_boundary_values": False,
            "clipping": False,
            "primitive_floors": False,
            "limiter": False,
            "decode_reencode_projection": False,
        },
        "input_noise_std": args.input_noise_std,
        "generated_state_exposure": {
            "weight": args.generated_state_exposure_weight,
            "depth": 1 if args.generated_state_exposure_weight > 0.0 else 0,
            "first_call_gradient": "detached",
            "clean_one_step_anchor": (1.0 - args.generated_state_exposure_weight),
            "raw_generated_state": True,
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_clip": args.gradient_clip,
            "scheduler": args.scheduler,
        },
        "best_epoch": best_epoch,
        "best_selection": best_selection,
        "last_train": last_train_metrics,
        "last_validation": last_validation_metrics,
        "last_rollout": last_rollout_metrics,
        "tiny_fit": tiny_fit,
        "elapsed_seconds": perf_counter() - run_start,
        "data_manifest_digest": store.manifest_digest,
        "config_digest": digest_mapping(jsonable_args(args)),
        "code_sha256": {
            "trainer": sha256_file(Path(__file__)),
            "pcno_euler2d": sha256_file(
                ROOT / "utility/time_dependent_no/pcno_euler2d.py"
            ),
            "pcno_core": sha256_file(ROOT / "pcno/pcno.py"),
        },
        "parent_checkpoint": parent_checkpoint,
        "git": git_state(),
        "artifacts": {
            "best_checkpoint": str(args.output_dir / "best.pt"),
            "last_checkpoint": str(args.output_dir / "last.pt"),
            "tiny_best_checkpoint": (
                str(args.output_dir / "tiny_best.pt")
                if best_tiny_state is not None
                else None
            ),
            "metrics": str(log_path),
        },
        "artifact_sha256": {
            "best_checkpoint": sha256_file(args.output_dir / "best.pt"),
            "last_checkpoint": sha256_file(args.output_dir / "last.pt"),
            "tiny_best_checkpoint": (
                sha256_file(args.output_dir / "tiny_best.pt")
                if best_tiny_state is not None
                else None
            ),
        },
    }
    write_json(args.output_dir / "summary.json", summary)
    print(
        json.dumps(
            {"summary": str(args.output_dir / "summary.json"), "tiny_fit": tiny_fit},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
