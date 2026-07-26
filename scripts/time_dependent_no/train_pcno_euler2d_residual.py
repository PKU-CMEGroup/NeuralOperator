#!/usr/bin/env python3
"""Train and rollout-select a state-decoded PCNO on 2D Euler shards.

The default remains the Line-3 centered conservative-residual baseline. The
fixed-geometry alternatives are the state-loss-only shared physical-face
decoder and the separately gated D048 divergence-active face target. Only the
latter may load accepted reference face impulses, and then only to recover the
physical boundary exchange used by the passed D047 direct projector. Every
path uses raw recurrence.
"""

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
from utility.time_dependent_no.fv_impulse_diagnostics import (  # noqa: E402
    DirectMinimumNormProjector,
    build_fv_impulse_operators,
    factorize_direct_minimum_winv_norm_projector,
)
from utility.time_dependent_no.pcno_face_impulse import (  # noqa: E402
    DIVERGENCE_ACTIVE_TARGET_KIND,
    FACE_TARGET_KINDS,
    FixedFVFaceGeometry,
    PCNOEuler2DSharedFaceImpulse,
    face_contract_summary,
    load_fixed_fv_face_geometry,
    load_fv_reference_face_trajectory,
    winv_face_loss_and_relative_l2,
)

CHECKPOINT_SCHEMA_VERSION = 4
RESIDUAL_TARGET_KIND = "conservative_variable_residual"
D047_SCHEMA = "fv_divergence_active_target_preflight_v2"
D048_SCHEMA = "pcno_euler2d_canonical_face_tiny_fit_v1"
D048_EXPERIMENT_ID = "D048"
D048_PARAMETER_COUNT = 19_210_028
D048_TINY_BANK = (
    ("sv_e08_y07", 8),
    ("sv_e03_y03", 7),
    ("sv_e03_y01", 44),
    ("sv_e05_y07", 41),
)
D048_GATE_THRESHOLDS = {
    "loss_ratio_max": 1.0e-2,
    "face_relative_l2_max": 1.0e-1,
    "decoded_state_relative_l2_max": 1.0e-3,
    "canonical_reference_closure_relative_l2_max": 1.0e-8,
    "canonical_shard_increment_closure_relative_l2_max": 1.0e-4,
    "reference_closure_relative_l2_max": 1.0e-10,
    "compatibility_relative_l2_max": 1.0e-10,
    "compatibility_projection_relative_l2_max": 1.0e-10,
    "reduced_solve_residual_relative_l2_max": 1.0e-10,
    "canonical_full_winv_norm_ratio_max": 1.000001,
    "wall_forbidden_exchange_absolute_max": 1.0e-14,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--target-kind",
        choices=(RESIDUAL_TARGET_KIND, *sorted(FACE_TARGET_KINDS)),
        default=RESIDUAL_TARGET_KIND,
    )
    parser.add_argument(
        "--family-root",
        type=Path,
        default=None,
        help=(
            "Validated shock-vortex family root; required only for the "
            "shared-face target's physical geometry."
        ),
    )
    parser.add_argument(
        "--canonical-preflight-summary",
        type=Path,
        default=None,
        help=(
            "Passed D047 summary.json; required only for the D048 "
            "divergence-active face target."
        ),
    )
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
    parser.add_argument("--tiny-fit-face-rel-l2", type=float, default=0.1)
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
    parser.add_argument("--face-latent-dim", type=int, default=32)
    parser.add_argument("--face-hidden-dim", type=int, default=128)
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
    if args.face_latent_dim < 4 or args.face_hidden_dim < 1:
        raise ValueError("face decoder widths must be positive and latent dim >= 4")
    if args.target_kind in FACE_TARGET_KINDS:
        if args.family_root is None:
            raise ValueError("--family-root is required for the shared-face target")
        if args.split_mode != "manifest":
            raise ValueError("shared-face training requires --split-mode manifest")
        if args.step_stride != 1:
            raise ValueError("shared-face training currently requires --step-stride 1")
    elif args.family_root is not None:
        raise ValueError("--family-root is only used by the shared-face target")
    if args.target_kind == DIVERGENCE_ACTIVE_TARGET_KIND:
        if args.canonical_preflight_summary is None:
            raise ValueError(
                "--canonical-preflight-summary is required for the D048 target"
            )
    elif args.canonical_preflight_summary is not None:
        raise ValueError(
            "--canonical-preflight-summary is only used by the D048 target"
        )
    if args.tiny_pairs < 0:
        raise ValueError("--tiny-pairs must be nonnegative")
    if args.stop_on_tiny_fit and args.tiny_pairs == 0:
        raise ValueError("--stop-on-tiny-fit requires --tiny-pairs")
    if args.rollout_start_frame < 0:
        raise ValueError("--rollout-start-frame must be nonnegative")
    if (
        args.tiny_fit_rel_l2 <= 0.0
        or args.tiny_fit_face_rel_l2 <= 0.0
        or args.tiny_fit_loss_ratio <= 0.0
    ):
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
    if args.target_kind == DIVERGENCE_ACTIVE_TARGET_KIND:
        expected_exact = {
            "seed": 20260718,
            "split_seed": 20260718,
            "epochs": 50,
            "presentations_per_epoch": 64,
            "val_presentations": 4,
            "batch_size": 1,
            "tiny_pairs": 4,
            "k_max": 8,
            "fc_dim": 128,
            "face_latent_dim": 32,
            "face_hidden_dim": 128,
            "scheduler": "constant",
            "amp": "bf16",
        }
        for name, expected in expected_exact.items():
            actual = getattr(args, name)
            if actual != expected:
                raise ValueError(
                    f"D048 freezes --{name.replace('_', '-')} at {expected!r}; "
                    f"received {actual!r}"
                )
        expected_floats = {
            "tiny_fit_rel_l2": D048_GATE_THRESHOLDS["decoded_state_relative_l2_max"],
            "tiny_fit_face_rel_l2": D048_GATE_THRESHOLDS["face_relative_l2_max"],
            "tiny_fit_loss_ratio": D048_GATE_THRESHOLDS["loss_ratio_max"],
            "learning_rate": 1.0e-3,
            "weight_decay": 1.0e-5,
            "gradient_clip": 1.0,
            "input_noise_std": 0.0,
            "generated_state_exposure_weight": 0.0,
        }
        for name, expected in expected_floats.items():
            actual = float(getattr(args, name))
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-15):
                raise ValueError(
                    f"D048 freezes --{name.replace('_', '-')} at {expected}; "
                    f"received {actual}"
                )
        if tuple(args.domain_lengths) != (2.0, 1.0):
            raise ValueError("D048 freezes --domain-lengths at 2 1")
        if tuple(args.layers) != (128, 128, 128, 128, 128):
            raise ValueError("D048 freezes --layers at five width-128 layers")
        if not args.stop_on_tiny_fit:
            raise ValueError("D048 requires --stop-on-tiny-fit")
        if args.init_checkpoint is not None or args.resume_checkpoint is not None:
            raise ValueError("D048 forbids initialization and resume checkpoints")
        if device.type != "cuda" or args.device != "cuda":
            raise ValueError("D048 requires explicit --device cuda")


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
    if saved_target_kind != args.target_kind:
        raise ValueError("checkpoint and requested target kinds differ")
    expected = {
        "k_max": int(args.k_max),
        "domain_lengths": [float(value) for value in args.domain_lengths],
        "layers": [int(value) for value in args.layers],
        "fc_dim": int(args.fc_dim),
    }
    if args.target_kind in FACE_TARGET_KINDS:
        expected.update(
            {
                "latent_dim": int(args.face_latent_dim),
                "face_hidden_dim": int(args.face_hidden_dim),
            }
        )
    for key, value in expected.items():
        if actual[key] != value:
            raise ValueError(
                f"checkpoint model config mismatch for {key}: {actual[key]} != {value}"
            )


def build_model(
    args: argparse.Namespace,
    normalization: Euler2DNormalization,
    *,
    face_geometry: FixedFVFaceGeometry | None,
    zero_initialize: bool,
) -> PCNOEuler2DResidual:
    """Construct one declared state-decoded target without changing its shape."""

    common = {
        "normalization": normalization,
        "k_max": args.k_max,
        "domain_lengths": args.domain_lengths,
        "layers": args.layers,
        "fc_dim": args.fc_dim,
        "zero_initialize": zero_initialize,
    }
    if args.target_kind == RESIDUAL_TARGET_KIND:
        if face_geometry is not None:
            raise ValueError("residual model received an unused face geometry")
        return PCNOEuler2DResidual(**common)
    if face_geometry is None:
        raise ValueError("shared-face model requires validated face geometry")
    return PCNOEuler2DSharedFaceImpulse(
        **common,
        cell_volume=face_geometry.cell_volume,
        face_center=face_geometry.face_center,
        face_measure=face_geometry.face_measure,
        face_normal=face_geometry.face_normal,
        face_owner=face_geometry.face_owner,
        face_neighbor=face_geometry.face_neighbor,
        face_boundary_tag=face_geometry.face_boundary_tag,
        boundary_tag_names=face_geometry.boundary_tag_names,
        geometry_digest=face_geometry.physical_geometry_digest,
        graph_geometry_digest=face_geometry.graph_geometry_digest,
        fixed_delta_t=face_geometry.fixed_delta_t,
        target_kind=args.target_kind,
        supervision=(
            "direct_canonical_face_winv_loss"
            if args.target_kind == DIVERGENCE_ACTIVE_TARGET_KIND
            else "decoded_next_state_loss_only"
        ),
        reference_impulse_supervision=(
            args.target_kind == DIVERGENCE_ACTIVE_TARGET_KIND
        ),
        latent_dim=args.face_latent_dim,
        face_hidden_dim=args.face_hidden_dim,
    )


def target_contract(
    model: PCNOEuler2DResidual,
    face_geometry: FixedFVFaceGeometry | None,
) -> dict[str, Any]:
    """Describe what the optimized state loss does and does not identify."""

    if isinstance(model, PCNOEuler2DSharedFaceImpulse):
        if face_geometry is None:
            raise ValueError("shared-face model is missing geometry provenance")
        direct = model.target_kind == DIVERGENCE_ACTIVE_TARGET_KIND
        return {
            **face_geometry.target_contract(
                target_kind=model.target_kind,
                supervision=model.supervision,
                reference_impulse_supervision=model.reference_impulse_supervision,
            ),
            "model_face_contract": dict(face_contract_summary(model)),
            "cycle_component_identifiability": (
                "excluded_by_minimum_W_f^-1_divergence_active_target; "
                "no_full_reference_cycle_loss"
                if direct
                else "not_identified_by_state_loss; no full-face reference loss"
            ),
            "decoded_state_role": (
                "outcome_metric_only" if direct else "optimized_state_loss"
            ),
        }
    if face_geometry is not None:
        raise ValueError("residual target received an unused face geometry")
    return {
        "target_kind": RESIDUAL_TARGET_KIND,
        "supervision": "decoded_next_state_loss",
        "coordinate_convention": "conservative_[rho,rho_u,rho_v,E]",
        "emits_physical_face_exchange": False,
        "reference_impulse_supervision": False,
        "conserved_totals_are_outcome_metrics_not_guaranteed_structure": True,
    }


def load_d047_preflight_summary(
    path: Path,
    store: PCNOEuler2DShardStore,
) -> dict[str, Any]:
    """Load the sole passed D047 authorization and fail closed on drift."""

    if not path.is_file():
        raise FileNotFoundError(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("D047 summary must be a JSON object")
    required_equal = {
        "schema": D047_SCHEMA,
        "experiment_id": "D047",
        "status": "passed",
        "shard_manifest_digest": store.manifest_digest,
        "family_id": store.manifest.get("source_family_id"),
        "family_manifest_digest": store.manifest.get("source_family_manifest_digest"),
    }
    for name, expected in required_equal.items():
        if raw.get(name) != expected:
            raise ValueError(
                f"D047 summary mismatch for {name}: {raw.get(name)!r} != {expected!r}"
            )
    config = raw.get("config")
    if not isinstance(config, dict):
        raise ValueError("D047 summary lacks its immutable configuration")
    if config.get("schema") != D047_SCHEMA or config.get("experiment_id") != "D047":
        raise ValueError("D047 summary configuration identity differs")
    if config.get("canonical_solver") != "direct":
        raise ValueError("D048 requires the direct D047 canonical solver")
    if digest_mapping(config) != raw.get("config_digest"):
        raise ValueError("D047 configuration digest does not reproduce")
    source_key = str(config.get("source_geometry_key", ""))
    if source_key not in store.keys or store.entry(source_key).get("split") != "train":
        raise ValueError("D047 source geometry is not an available training case")
    checks = raw.get("checks")
    if (
        not isinstance(checks, dict)
        or not checks
        or not all(value is True for value in checks.values())
    ):
        raise ValueError("D047 did not pass every registered check")
    promotion = raw.get("promotion")
    if (
        not isinstance(promotion, dict)
        or promotion.get("divergence_active_four_pair_tiny_fit_authorized") is not True
    ):
        raise ValueError("D047 does not authorize the four-pair tiny fit")
    if any(
        promotion.get(name) is not False
        for name in (
            "serious_training_authorized",
            "full_reference_cycle_supervision_authorized",
            "test_split_evaluation_authorized",
        )
    ):
        raise ValueError("D047 promotion boundary differs from the registered contract")
    accessed = raw.get("accessed_case_splits")
    if not isinstance(accessed, dict) or accessed.get("test") != []:
        raise ValueError("D047 summary does not prove test-split nonaccess")
    if int(raw.get("row_count", -1)) != 24:
        raise ValueError("D047 summary must contain the exact 24-row cohort")
    code = raw.get("code_sha256")
    current_projector_sha = sha256_file(
        ROOT / "utility/time_dependent_no/fv_impulse_diagnostics.py"
    )
    if not isinstance(code, dict) or code.get("fv_impulse_diagnostics") != (
        current_projector_sha
    ):
        raise ValueError("the D047 projector code has drifted since preflight")
    direct = raw.get("direct_projector")
    if not isinstance(direct, dict):
        raise ValueError("D047 summary lacks direct-projector provenance")
    expected_direct = {
        "solver": "scipy_sparse_superlu",
        "factorization_dtype": "float64",
        "compatibility_projection": "componentwise_arithmetic_mean",
        "regularization": "none",
        "iterative_refinement": "none",
    }
    for name, expected in expected_direct.items():
        if direct.get(name) != expected:
            raise ValueError(f"D047 direct-projector contract differs for {name}")
    thresholds = config.get("gate_thresholds")
    required_thresholds = {
        name: D048_GATE_THRESHOLDS[name]
        for name in (
            "canonical_reference_closure_relative_l2_max",
            "canonical_shard_increment_closure_relative_l2_max",
            "reference_closure_relative_l2_max",
            "compatibility_relative_l2_max",
            "canonical_full_winv_norm_ratio_max",
            "wall_forbidden_exchange_absolute_max",
        )
    }
    required_thresholds.update(
        {
            "direct_compatibility_projection_relative_l2_max": (
                D048_GATE_THRESHOLDS["compatibility_projection_relative_l2_max"]
            ),
            "direct_reduced_solve_residual_relative_l2_max": (
                D048_GATE_THRESHOLDS["reduced_solve_residual_relative_l2_max"]
            ),
        }
    )
    if not isinstance(thresholds, dict):
        raise ValueError("D047 summary lacks gate thresholds")
    for name, expected in required_thresholds.items():
        if not math.isclose(
            float(thresholds.get(name, float("nan"))),
            expected,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError(f"D047 gate threshold differs for {name}")
    return raw


def validate_d047_geometry(
    preflight: Mapping[str, Any],
    geometry: FixedFVFaceGeometry,
) -> None:
    """Bind the passed D047 summary to the geometry loaded for D048."""

    expected = {
        "family_id": geometry.source_family_id,
        "family_manifest_digest": geometry.source_family_manifest_digest,
        "physical_geometry_digest": geometry.physical_geometry_digest,
        "graph_geometry_digest": geometry.graph_geometry_digest,
    }
    for name, value in expected.items():
        if preflight.get(name) != value:
            raise ValueError(f"D047 and D048 geometry differ for {name}")


def _relative_l2_numpy(actual: np.ndarray, expected: np.ndarray) -> float:
    difference = float(np.linalg.norm(np.asarray(actual) - np.asarray(expected)))
    denominator = float(np.linalg.norm(np.asarray(expected)))
    if denominator > 0.0:
        return difference / denominator
    return 0.0 if difference == 0.0 else float(np.finfo(np.float64).max)


def _winv_norm_numpy(face_field: np.ndarray, face_weight: np.ndarray) -> float:
    values = np.asarray(face_field, dtype=np.float64)
    weight = np.asarray(face_weight, dtype=np.float64)
    return float(np.sqrt(np.sum(values * values / weight[:, None])))


def build_d048_canonical_targets(
    store: PCNOEuler2DShardStore,
    *,
    family_root: Path,
    geometry: FixedFVFaceGeometry,
    pairs: Sequence[tuple[str, int]],
    preflight: Mapping[str, Any],
) -> tuple[dict[tuple[str, int], torch.Tensor], np.ndarray, dict[str, Any]]:
    """Build the four float32 labels once from float64 states and D047."""

    operators = build_fv_impulse_operators(
        cell_centers=geometry.cell_center,
        cell_volume=geometry.cell_volume,
        face_centers=geometry.face_center,
        face_measure=geometry.face_measure,
        face_owner=geometry.face_owner,
        face_neighbor=geometry.face_neighbor,
        face_boundary_tag=geometry.face_boundary_tag,
    )
    if operators.topology.to_dict() != preflight.get("topology"):
        raise ValueError("D048 finite-volume topology differs from D047")
    projector: DirectMinimumNormProjector = (
        factorize_direct_minimum_winv_norm_projector(operators)
    )
    if projector.summary() != preflight.get("direct_projector"):
        raise ValueError("D048 factorization does not reproduce D047 provenance")

    unique_pairs = [(str(key), int(time_index)) for key, time_index in pairs]
    if len(unique_pairs) != len(set(unique_pairs)):
        raise ValueError("D048 canonical target bank contains duplicate pairs")
    labels: dict[tuple[str, int], torch.Tensor] = {}
    rows: list[dict[str, Any]] = []
    boundary = operators.boundary_face_indices
    wall_tags = {
        geometry.boundary_tag_names.index("y_min"),
        geometry.boundary_tag_names.index("y_max"),
    }
    wall = np.isin(geometry.face_boundary_tag, tuple(wall_tags))
    for key, time_index in unique_pairs:
        if key not in store.keys or store.entry(key).get("split") != "train":
            raise ValueError(f"D048 may load only registered training cases: {key}")
        reference = load_fv_reference_face_trajectory(
            family_root,
            store,
            geometry,
            key=key,
        )
        if reference.split != "train":
            raise RuntimeError(f"D048 reference split is not train for {key}")
        if not 0 <= time_index < reference.physical_times.size - 1:
            raise IndexError(f"D048 time index is outside reference {key}")
        reference_delta = (
            reference.conservative_states[time_index + 1]
            - reference.conservative_states[time_index]
        )
        shard_states = np.asarray(store.states(key), dtype=np.float64)
        shard_delta = shard_states[time_index + 1] - shard_states[time_index]
        target_integral = -geometry.cell_volume[:, None] * reference_delta
        shard_target_integral = -geometry.cell_volume[:, None] * shard_delta
        reference_impulse = reference.cumulative_face_impulses[time_index]
        solution = projector.solve(
            target_integral,
            reference_impulse[boundary],
        )
        canonical = np.asarray(solution.face_impulse, dtype=np.float64)
        decoded_reference = operators.incidence @ reference_impulse
        decoded_canonical = solution.decoded_cell_integral
        canonical_norm = _winv_norm_numpy(canonical, operators.face_weight)
        reference_norm = _winv_norm_numpy(reference_impulse, operators.face_weight)
        metrics = {
            "reference_closure_relative_l2": _relative_l2_numpy(
                decoded_reference, target_integral
            ),
            "canonical_reference_closure_relative_l2": _relative_l2_numpy(
                decoded_canonical, target_integral
            ),
            "canonical_shard_increment_closure_relative_l2": _relative_l2_numpy(
                decoded_canonical, shard_target_integral
            ),
            "compatibility_relative_l2": solution.compatibility_relative_l2,
            "compatibility_projection_relative_l2": (
                solution.compatibility_projection_relative_l2
            ),
            "reduced_solve_residual_relative_l2": (
                solution.reduced_solve_residual_relative_l2
            ),
            "canonical_full_winv_norm_ratio": (
                canonical_norm / reference_norm if reference_norm > 0.0 else 0.0
            ),
            "wall_forbidden_exchange_absolute": float(
                np.max(np.abs(canonical[wall][:, (0, 1, 3)]), initial=0.0)
            ),
            "boundary_reproduction_absolute_max": float(
                np.max(
                    np.abs(canonical[boundary] - reference_impulse[boundary]),
                    initial=0.0,
                )
            ),
        }
        threshold_names = {
            "reference_closure_relative_l2": "reference_closure_relative_l2_max",
            "canonical_reference_closure_relative_l2": (
                "canonical_reference_closure_relative_l2_max"
            ),
            "canonical_shard_increment_closure_relative_l2": (
                "canonical_shard_increment_closure_relative_l2_max"
            ),
            "compatibility_relative_l2": "compatibility_relative_l2_max",
            "compatibility_projection_relative_l2": (
                "compatibility_projection_relative_l2_max"
            ),
            "reduced_solve_residual_relative_l2": (
                "reduced_solve_residual_relative_l2_max"
            ),
            "canonical_full_winv_norm_ratio": ("canonical_full_winv_norm_ratio_max"),
            "wall_forbidden_exchange_absolute": (
                "wall_forbidden_exchange_absolute_max"
            ),
        }
        scalar_values = np.asarray(list(metrics.values()), dtype=np.float64)
        if not np.all(np.isfinite(canonical)) or not np.all(np.isfinite(scalar_values)):
            raise FloatingPointError(f"D048 canonical target is nonfinite for {key}")
        for metric_name, threshold_name in threshold_names.items():
            if float(metrics[metric_name]) > D048_GATE_THRESHOLDS[threshold_name]:
                raise RuntimeError(
                    f"D048 label {key}@{time_index} fails {metric_name}: "
                    f"{metrics[metric_name]}"
                )
        if metrics["boundary_reproduction_absolute_max"] != 0.0:
            raise RuntimeError(f"D048 boundary impulse was altered for {key}")

        float32_label = np.asarray(canonical, dtype=np.float32)
        float32_decoded = operators.incidence @ float32_label.astype(np.float64)
        float32_closure = _relative_l2_numpy(float32_decoded, target_integral)
        if (
            float32_closure
            > D048_GATE_THRESHOLDS["canonical_shard_increment_closure_relative_l2_max"]
        ):
            raise RuntimeError(f"D048 float32 label closure fails for {key}")
        pair = (key, time_index)
        labels[pair] = torch.from_numpy(float32_label.copy())
        rows.append(
            {
                "trajectory": key,
                "time_index": time_index,
                "target_time_index": time_index + 1,
                "physical_time": float(reference.physical_times[time_index]),
                "target_physical_time": float(reference.physical_times[time_index + 1]),
                "delta_t": float(
                    reference.physical_times[time_index + 1]
                    - reference.physical_times[time_index]
                ),
                "source_reference_sha256": reference.source_reference_sha256,
                "label_sha256": digest_array(float32_label),
                "float32_reference_closure_relative_l2": float32_closure,
                **metrics,
            }
        )
    label_digests = {
        f"{key}@{time_index}": digest_array(labels[(key, time_index)].numpy())
        for key, time_index in unique_pairs
    }
    metadata = {
        "schema": D048_SCHEMA,
        "experiment_id": D048_EXPERIMENT_ID,
        "parent_experiment_id": "D047",
        "parent_config_digest": preflight["config_digest"],
        "direct_projector": projector.summary(),
        "face_weight": "W_f=face_measure*dual_width; W_f^-1 relative loss",
        "face_weight_sha256": digest_array(operators.face_weight),
        "label_set_digest": digest_mapping(label_digests),
        "pairs": rows,
        "accessed_case_splits": {
            "train": sorted({key for key, _ in pairs}),
            "test": [],
        },
        "full_reference_cycle_supervision": False,
        "gate_thresholds": dict(D048_GATE_THRESHOLDS),
    }
    return labels, np.asarray(operators.face_weight), metadata


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


def forward_face_sample(
    model: PCNOEuler2DSharedFaceImpulse,
    sample: Mapping[str, torch.Tensor],
    current: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return model.forward_with_face_impulse(
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


def canonical_face_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    face_weight: np.ndarray | torch.Tensor,
    face_is_interior: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return D048's native loss and independent face-error outcomes."""

    prediction = prediction.float()
    target = target.float()
    interior_loss, interior_relative_l2 = winv_face_loss_and_relative_l2(
        prediction,
        target,
        face_weight=face_weight,
        face_mask=face_is_interior,
    )
    boundary_loss, boundary_relative_l2 = winv_face_loss_and_relative_l2(
        prediction,
        target,
        face_weight=face_weight,
        face_mask=~face_is_interior,
    )
    _, full_relative_l2 = winv_face_loss_and_relative_l2(
        prediction,
        target,
        face_weight=face_weight,
    )
    return {
        "loss": 0.5 * (interior_loss + boundary_loss),
        "face_relative_l2": full_relative_l2,
        "interior_face_relative_l2": interior_relative_l2,
        "boundary_face_relative_l2": boundary_relative_l2,
        "interior_face_loss": interior_loss,
        "boundary_face_loss": boundary_loss,
    }


def _canonical_target_batch(
    targets: Mapping[tuple[str, int], torch.Tensor],
    key: str,
    time_indices: Sequence[int],
    *,
    device: torch.device,
) -> torch.Tensor:
    try:
        batch = torch.stack([targets[(key, int(index))] for index in time_indices])
    except KeyError as error:
        raise KeyError(f"missing canonical D048 label for {error.args[0]}") from error
    return batch.to(device=device, dtype=torch.float32)


def _wall_forbidden_absolute_max(
    face_impulse: torch.Tensor,
    model: PCNOEuler2DSharedFaceImpulse,
) -> torch.Tensor:
    wall_tags = {
        model.boundary_tag_names.index("y_min"),
        model.boundary_tag_names.index("y_max"),
    }
    wall = torch.zeros_like(model.face_boundary_tag, dtype=torch.bool)
    for tag in wall_tags:
        wall |= model.face_boundary_tag == tag
    return torch.max(torch.abs(face_impulse[:, wall][:, :, [0, 1, 3]]))


@torch.no_grad()
def evaluate_canonical_pairs(
    model: PCNOEuler2DSharedFaceImpulse,
    store: PCNOEuler2DShardStore,
    pairs: Sequence[tuple[str, int]],
    *,
    canonical_targets: Mapping[tuple[str, int], torch.Tensor],
    face_weight: np.ndarray,
    step_stride: int,
    batch_size: int,
    device: torch.device,
    amp: str,
) -> dict[str, float]:
    """Evaluate native D048 face fit and decoded-state outcomes together."""

    model.eval()
    sums = {
        "loss": 0.0,
        "face_relative_l2": 0.0,
        "interior_face_relative_l2": 0.0,
        "boundary_face_relative_l2": 0.0,
        "interior_face_loss": 0.0,
        "boundary_face_loss": 0.0,
        "decoded_state_loss": 0.0,
        "relative_l2": 0.0,
    }
    admissible = 0
    wall_forbidden_max = 0.0
    persistence_max = 0.0
    batches = homogeneous_presentation_batches(pairs, batch_size=batch_size)
    for key, time_indices in batches:
        sample = store.tensor_batch(
            key,
            time_indices,
            step_stride=step_stride,
            device=device,
        )
        face_target = _canonical_target_batch(
            canonical_targets,
            key,
            time_indices,
            device=device,
        )
        with autocast_context(device, amp):
            prediction, face_prediction = forward_face_sample(
                model, sample, sample["current"]
            )
        face_result = canonical_face_metrics(
            face_prediction,
            face_target,
            face_weight=face_weight,
            face_is_interior=model.face_is_interior,
        )
        decoded_loss, decoded_relative = pair_metrics(
            prediction.float(), sample["target"], sample, model
        )
        current_batch_size = len(time_indices)
        for name in (
            "loss",
            "face_relative_l2",
            "interior_face_relative_l2",
            "boundary_face_relative_l2",
            "interior_face_loss",
            "boundary_face_loss",
        ):
            sums[name] += float(face_result[name].cpu()) * current_batch_size
        sums["decoded_state_loss"] += float(decoded_loss.cpu()) * current_batch_size
        sums["relative_l2"] += float(decoded_relative.cpu()) * current_batch_size
        diagnostics = conservative_admissibility(prediction.float(), gamma=model.gamma)
        per_sample_admissible = (
            diagnostics["admissible"].reshape(prediction.shape[0], -1).all(dim=1)
        )
        admissible += int(per_sample_admissible.sum().cpu())
        wall_forbidden_max = max(
            wall_forbidden_max,
            float(_wall_forbidden_absolute_max(face_prediction, model).cpu()),
        )
        persistence_max = max(
            persistence_max,
            float(torch.max(torch.abs(prediction.float() - sample["current"])).cpu()),
        )
    result = {name: value / len(pairs) for name, value in sums.items()}
    result.update(
        {
            "admissible_fraction": admissible / len(pairs),
            "wall_forbidden_exchange_absolute_max": wall_forbidden_max,
            "max_persistence_absolute": persistence_max,
            "presentations": len(pairs),
        }
    )
    if not all(math.isfinite(float(value)) for value in result.values()):
        raise FloatingPointError("D048 evaluation produced a nonfinite metric")
    return result


@torch.no_grad()
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
    face_relative_l2_threshold: float | None = None,
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
    face_passed = True
    if face_relative_l2_threshold is not None:
        face_passed = (
            all(
                float(current[name]) <= face_relative_l2_threshold
                for name in (
                    "face_relative_l2",
                    "interior_face_relative_l2",
                    "boundary_face_relative_l2",
                )
            )
            and float(current["wall_forbidden_exchange_absolute_max"]) == 0.0
        )
    return {
        "metrics": dict(current),
        "loss_ratio": loss_ratio,
        "relative_l2_threshold": float(relative_l2_threshold),
        "face_relative_l2_threshold": (
            None
            if face_relative_l2_threshold is None
            else float(face_relative_l2_threshold)
        ),
        "loss_ratio_threshold": float(loss_ratio_threshold),
        "passed": (
            relative_l2 <= relative_l2_threshold
            and loss_ratio <= loss_ratio_threshold
            and admissible_fraction == 1.0
            and face_passed
        ),
    }


def tiny_fit_selection(snapshot: Mapping[str, Any]) -> tuple[float, ...]:
    metrics = snapshot["metrics"]
    relative_ratio = float(metrics["relative_l2"]) / float(
        snapshot["relative_l2_threshold"]
    )
    loss_ratio = float(snapshot["loss_ratio"]) / float(snapshot["loss_ratio_threshold"])
    face_threshold = snapshot.get("face_relative_l2_threshold")
    face_ratios = (
        []
        if face_threshold is None
        else [
            float(metrics[name]) / float(face_threshold)
            for name in (
                "face_relative_l2",
                "interior_face_relative_l2",
                "boundary_face_relative_l2",
            )
        ]
    )
    maximum_ratio = max(relative_ratio, loss_ratio, *face_ratios)
    return (
        float(bool(snapshot["passed"])),
        float(metrics["admissible_fraction"]),
        -maximum_ratio,
        -relative_ratio,
        *(-ratio for ratio in face_ratios),
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
        "target_kind": args.target_kind,
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


def train_canonical_face_epoch(
    model: PCNOEuler2DSharedFaceImpulse,
    store: PCNOEuler2DShardStore,
    pairs: Sequence[tuple[str, int]],
    *,
    canonical_targets: Mapping[tuple[str, int], torch.Tensor],
    face_weight: np.ndarray,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    step_stride: int,
    batch_size: int,
    batch_rng: np.random.Generator,
    device: torch.device,
    amp: str,
    gradient_clip: float,
) -> dict[str, float | None]:
    """Optimize only D048's equal interior/boundary canonical-face loss."""

    model.train()
    metric_names = (
        "loss",
        "face_relative_l2",
        "interior_face_relative_l2",
        "boundary_face_relative_l2",
        "interior_face_loss",
        "boundary_face_loss",
        "decoded_state_loss",
        "relative_l2",
    )
    sums = {name: 0.0 for name in metric_names}
    wall_forbidden_max = 0.0
    all_gradients_finite = True
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
        face_target = _canonical_target_batch(
            canonical_targets,
            key,
            time_indices,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp):
            prediction, face_prediction = forward_face_sample(
                model, sample, sample["current"]
            )
        face_result = canonical_face_metrics(
            face_prediction,
            face_target,
            face_weight=face_weight,
            face_is_interior=model.face_is_interior,
        )
        loss = face_result["loss"]
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(
                f"nonfinite D048 face loss for trajectory {key} frames {time_indices}"
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        if not gradients:
            raise RuntimeError("D048 face loss produced no model gradients")
        gradients_finite = all(bool(torch.isfinite(value).all()) for value in gradients)
        all_gradients_finite = all_gradients_finite and gradients_finite
        if not gradients_finite:
            raise FloatingPointError(
                f"nonfinite D048 gradients for trajectory {key} frames {time_indices}"
            )
        if gradient_clip > 0.0:
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), gradient_clip
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError("D048 gradient norm is nonfinite")
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        with torch.no_grad():
            decoded_loss, decoded_relative = pair_metrics(
                prediction.float(), sample["target"], sample, model
            )
        current_batch_size = len(time_indices)
        for name in (
            "loss",
            "face_relative_l2",
            "interior_face_relative_l2",
            "boundary_face_relative_l2",
            "interior_face_loss",
            "boundary_face_loss",
        ):
            sums[name] += float(face_result[name].detach().cpu()) * current_batch_size
        sums["decoded_state_loss"] += (
            float(decoded_loss.detach().cpu()) * current_batch_size
        )
        sums["relative_l2"] += (
            float(decoded_relative.detach().cpu()) * current_batch_size
        )
        wall_forbidden_max = max(
            wall_forbidden_max,
            float(_wall_forbidden_absolute_max(face_prediction, model).detach().cpu()),
        )
    result: dict[str, float | None] = {
        name: value / len(pairs) for name, value in sums.items()
    }
    result.update(
        {
            "wall_forbidden_exchange_absolute_max": wall_forbidden_max,
            "gradients_finite": float(all_gradients_finite),
            "clean_loss": result["loss"],
            "clean_relative_l2": result["relative_l2"],
            "generated_loss": None,
            "generated_relative_l2": None,
            "generated_input_relative_l2": None,
            "generated_state_exposure_weight": 0.0,
            "presentations": len(pairs),
            "optimizer_steps": len(batches),
        }
    )
    return result


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
    d047_preflight = None
    d047_preflight_sha256 = None
    if args.target_kind == DIVERGENCE_ACTIVE_TARGET_KIND:
        d047_preflight = load_d047_preflight_summary(
            args.canonical_preflight_summary,
            store,
        )
        d047_preflight_sha256 = sha256_file(args.canonical_preflight_summary)
    face_geometry = None
    if args.target_kind in FACE_TARGET_KINDS:
        source_key = (
            str(d047_preflight["config"]["source_geometry_key"])
            if d047_preflight is not None
            else train_keys[0]
        )
        face_geometry = load_fixed_fv_face_geometry(
            args.family_root,
            store,
            source_key=source_key,
        )
        if d047_preflight is not None:
            validate_d047_geometry(d047_preflight, face_geometry)

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
        face_geometry=face_geometry,
        zero_initialize=checkpoint is None,
    ).to(device)
    if (
        args.target_kind == DIVERGENCE_ACTIVE_TARGET_KIND
        and parameter_count(model) != D048_PARAMETER_COUNT
    ):
        raise RuntimeError(
            "D048 parameterization drifted: "
            f"{parameter_count(model)} != {D048_PARAMETER_COUNT}"
        )
    tiny_bank = None
    if args.tiny_pairs:
        tiny_bank = fixed_tiny_presentations(
            store,
            train_keys,
            step_stride=args.step_stride,
            count=args.tiny_pairs,
            seed=args.seed + 17,
        )
    canonical_targets = None
    canonical_face_weight = None
    canonical_target_metadata = None
    if args.target_kind == DIVERGENCE_ACTIVE_TARGET_KIND:
        if tuple(tiny_bank or ()) != D048_TINY_BANK:
            raise RuntimeError(
                f"D048 immutable pair bank drifted: {tiny_bank!r} != "
                f"{list(D048_TINY_BANK)!r}"
            )
        if face_geometry is None or d047_preflight is None:
            raise RuntimeError("D048 geometry or D047 authorization is unresolved")
        canonical_targets, canonical_face_weight, canonical_target_metadata = (
            build_d048_canonical_targets(
                store,
                family_root=args.family_root,
                geometry=face_geometry,
                pairs=tiny_bank,
                preflight=d047_preflight,
            )
        )
        canonical_target_metadata.update(
            {
                "parent_summary_sha256": d047_preflight_sha256,
                "code_sha256": {
                    "trainer": sha256_file(Path(__file__)),
                    "pcno_face_impulse": sha256_file(
                        ROOT / "utility/time_dependent_no/pcno_face_impulse.py"
                    ),
                    "fv_impulse_diagnostics": sha256_file(
                        ROOT / "utility/time_dependent_no/fv_impulse_diagnostics.py"
                    ),
                },
            }
        )
    resolved_target_contract = target_contract(model, face_geometry)
    if canonical_target_metadata is not None:
        resolved_target_contract["canonical_target"] = canonical_target_metadata
        resolved_target_contract["objective"] = (
            "0.5*mean_component_relative_W_f^-1_interior_squared_error + "
            "0.5*mean_component_relative_W_f^-1_boundary_squared_error"
        )
        resolved_target_contract["decoded_state_in_training_objective"] = False
        resolved_target_contract["test_split_accessed"] = False
    if checkpoint is not None:
        saved_target_contract = checkpoint.get("target_contract")
        if saved_target_contract is None:
            if args.target_kind in FACE_TARGET_KINDS:
                raise ValueError("face-target checkpoint lacks its target contract")
        elif saved_target_contract != resolved_target_contract:
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
        if args.target_kind == DIVERGENCE_ACTIVE_TARGET_KIND:
            initial_train_metrics = evaluate_canonical_pairs(
                model,
                store,
                tiny_bank,
                canonical_targets=canonical_targets,
                face_weight=canonical_face_weight,
                step_stride=args.step_stride,
                batch_size=args.batch_size,
                device=device,
                amp=args.amp,
            )
            for name in (
                "loss",
                "face_relative_l2",
                "interior_face_relative_l2",
                "boundary_face_relative_l2",
            ):
                if not math.isclose(
                    float(initial_train_metrics[name]),
                    1.0,
                    rel_tol=0.0,
                    abs_tol=2.0e-6,
                ):
                    raise RuntimeError(f"D048 zero-output unit check failed for {name}")
            if initial_train_metrics["max_persistence_absolute"] != 0.0:
                raise RuntimeError("D048 zero-output model is not exact persistence")
            if initial_train_metrics["wall_forbidden_exchange_absolute_max"] != 0.0:
                raise RuntimeError("D048 wall mask is not an exact structural zero")
        else:
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
        if args.target_kind == DIVERGENCE_ACTIVE_TARGET_KIND:
            if not isinstance(model, PCNOEuler2DSharedFaceImpulse):
                raise TypeError("D048 requires the shared-face model")
            last_train_metrics = train_canonical_face_epoch(
                model,
                store,
                pairs,
                canonical_targets=canonical_targets,
                face_weight=canonical_face_weight,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                step_stride=args.step_stride,
                batch_size=args.batch_size,
                batch_rng=np.random.default_rng(args.seed + 100_000 + epoch),
                device=device,
                amp=args.amp,
                gradient_clip=args.gradient_clip,
            )
        else:
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
            if args.target_kind == DIVERGENCE_ACTIVE_TARGET_KIND:
                current_tiny_metrics = evaluate_canonical_pairs(
                    model,
                    store,
                    tiny_bank,
                    canonical_targets=canonical_targets,
                    face_weight=canonical_face_weight,
                    step_stride=args.step_stride,
                    batch_size=args.batch_size,
                    device=device,
                    amp=args.amp,
                )
            else:
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
                face_relative_l2_threshold=(
                    args.tiny_fit_face_rel_l2
                    if args.target_kind == DIVERGENCE_ACTIVE_TARGET_KIND
                    else None
                ),
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
        if args.target_kind == DIVERGENCE_ACTIVE_TARGET_KIND:
            final_tiny = evaluate_canonical_pairs(
                model,
                store,
                tiny_bank,
                canonical_targets=canonical_targets,
                face_weight=canonical_face_weight,
                step_stride=args.step_stride,
                batch_size=args.batch_size,
                device=device,
                amp=args.amp,
            )
        else:
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
            face_relative_l2_threshold=(
                args.tiny_fit_face_rel_l2
                if args.target_kind == DIVERGENCE_ACTIVE_TARGET_KIND
                else None
            ),
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
            "face_relative_l2_threshold": (
                args.tiny_fit_face_rel_l2
                if args.target_kind == DIVERGENCE_ACTIVE_TARGET_KIND
                else None
            ),
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
        "mode": f"pcno_euler2d_{args.target_kind}",
        "target_kind": args.target_kind,
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
