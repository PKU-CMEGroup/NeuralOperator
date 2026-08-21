#!/usr/bin/env python3
"""Run the D094 native-mesh bump trajectory-scaling pilot.

The maintained Euler trainer owns the model, residual objective, normalization,
boundary closure, optimizer, validation, rollout, and artifact formats.  This
adapter owns only the D094 split, balanced no-replacement presentation stream,
functional PCFNO intervention, comparable seen/held-out pair banks, and
sentinel retention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utility.time_dependent_no.pcno_differential_branch import (
    DIFFERENTIAL_BRANCH_MODES,
    apply_differential_branch_mode,
)
from utility.time_dependent_no.pcno_rollout import FINITE_ONLY_ROLLOUT_POLICY

SCHEMA = "d094_bump_scaling_training_v1"
D094_METRIC_RECEIPT_SCHEMA = "d094_bump_scaling_metric_receipt_v1"
SPLIT_SCHEMA = "d094_bump_trajectory_scaling_split_v1"
REGISTERED_COUNTS = (8, 16, 32, 64, 128, 256)
D094_SELECTION_MODE = "full_horizon_error_first"
D094_ROLLOUT_STEPS = 79
D094_ROLLOUT_SELECTION_COUNT = 16
D094_EPOCHS = 80
D094_OPTIMIZER_STEPS_PER_EPOCH = 256
D094_SCHEDULE_ARMS_BY_DECAY_STEPS = {
    5_120: "prefix_tail",
    20_480: "stretched",
}
D094_SENTINEL_STEPS = (
    256,
    1_280,
    2_560,
    3_840,
    5_120,
    7_680,
    10_240,
    15_360,
    20_480,
)
FULL_SENTINEL_PAYLOAD = "full"
MODEL_ONLY_SENTINEL_PAYLOAD = "model_only"
SENTINEL_PAYLOAD_MODES = (FULL_SENTINEL_PAYLOAD, MODEL_ONLY_SENTINEL_PAYLOAD)
D094_METRIC_SEMANTICS = {
    "online_train_one_step": (
        "metrics.jsonl[].train; one-step metrics accumulated while parameters "
        "change, so they diagnose optimization but are not a fixed-bank estimate"
    ),
    "fixed_seen_train_one_step": (
        "metrics.jsonl[].train.comparable_seen; post-epoch one-step metrics on the "
        "frozen seen-trajectory pair bank"
    ),
    "fixed_open_validation_one_step": (
        "metrics.jsonl[].validation; post-epoch one-step metrics on the frozen "
        "44-trajectory open-validation pair bank"
    ),
    "autonomous_rollout": (
        "metrics.jsonl[].rollout; recurrent free rollout on 16 frozen selection "
        "trajectories, summarized by all-call mean and H79 endpoint error"
    ),
}
REGISTERED_SOURCE_MANIFEST_SHA256 = (
    "5d5373fdcc682544bf330fba6d54fe65509dabe936baee7443c71a8c0c8d9fa7"
)
REGISTERED_PARTITION_DIGEST = (
    "ac8cac650c11b03fe883063d6cf8a93b98554030da3c183bc1af9378e2237adb"
)
REGISTERED_CANONICAL_PAYLOAD_SHA256 = (
    "6e22a0bb754df158b7ea8838adda2ac36dd80ed554e300d84ba1c4478314709b"
)
D094_REGISTERED_SOURCE_FILES = (
    "docs/time_dependent_no/D094_BUMP_SCALING_PREREGISTRATION.md",
    "scripts/time_dependent_no/build_bump_scaling_manifest.py",
    "scripts/time_dependent_no/train_pcno_bump_scaling.py",
    "utility/time_dependent_no/pcno_differential_branch.py",
)


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def d094_checkpoint_metric_snapshot(row: Mapping[str, Any]) -> dict[str, Any]:
    """Extract fixed-scope metrics from one rollout-evaluation history row."""

    train = row.get("train")
    validation = row.get("validation")
    rollout = row.get("rollout")
    if not all(isinstance(value, Mapping) for value in (train, validation, rollout)):
        raise TypeError(
            "a D094 checkpoint metric row requires train, validation, and rollout"
        )
    comparable_seen = train.get("comparable_seen")
    endpoints = rollout.get("mean_endpoint_relative_l2")
    if not isinstance(comparable_seen, Mapping) or not isinstance(endpoints, Mapping):
        raise TypeError("a D094 checkpoint row lacks fixed seen or endpoint metrics")
    values = {
        "epoch": int(row["epoch"]),
        "optimizer_step": int(train["completed_optimizer_steps"]),
        "online_train_one_step_relative_l2": float(train["relative_l2"]),
        "fixed_seen_train_one_step_relative_l2": float(comparable_seen["relative_l2"]),
        "fixed_validation_one_step_relative_l2": float(validation["relative_l2"]),
        "rollout_all_call_mean_relative_l2": float(
            rollout["mean_full_horizon_relative_l2"]
        ),
        "rollout_h79_relative_l2": float(endpoints["79"]),
        "rollout_completion_rate": float(rollout["completion_rate"]),
        "rollout_hard_failure_count": int(rollout.get("hard_failure_count", 0)),
        "physical_admissibility_rate": float(rollout["physical_admissibility_rate"]),
    }
    if not all(
        math.isfinite(value)
        for name, value in values.items()
        if name not in {"epoch", "optimizer_step", "rollout_hard_failure_count"}
    ):
        raise ValueError("D094 checkpoint metric receipt contains a nonfinite scalar")
    return values


def build_d094_metric_receipt(
    summary: Mapping[str, Any],
    metric_rows: Sequence[Mapping[str, Any]],
    scaling_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind selected and terminal metrics without conflating their scopes."""

    if not metric_rows:
        raise ValueError("D094 metric history is empty")
    best_epoch = int(summary["best_epoch"])
    selected_rows = [row for row in metric_rows if int(row["epoch"]) == best_epoch]
    if len(selected_rows) != 1:
        raise ValueError("D094 best epoch does not identify exactly one history row")
    if scaling_contract.get("historical_test_population_accessed") is not False:
        raise ValueError("D094 metric receipt cannot follow historical test access")
    source_snapshot = summary.get("source_snapshot")
    if not isinstance(source_snapshot, Mapping):
        raise TypeError("D094 summary lacks its source snapshot")
    return {
        "schema": D094_METRIC_RECEIPT_SCHEMA,
        "best_epoch": best_epoch,
        "best_selection": [float(value) for value in summary["best_selection"]],
        "selected_checkpoint": d094_checkpoint_metric_snapshot(selected_rows[0]),
        "terminal_checkpoint": d094_checkpoint_metric_snapshot(metric_rows[-1]),
        "metric_scope_contract": {
            "selected_checkpoint": "metrics.jsonl row at summary.best_epoch",
            "terminal_checkpoint": "final metrics.jsonl row",
        },
        "source_set_digest": str(source_snapshot["source_set_digest"]),
        "split_partition_digest": str(scaling_contract["split_partition_digest"]),
        "historical_test_population_accessed": False,
    }


def d094_source_snapshot_files(split_manifest_path: Path) -> tuple[str, ...]:
    """Return the complete repository-relative D094 extension source set."""

    root = ROOT.resolve()
    try:
        split_relative = split_manifest_path.resolve().relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("the D094 split manifest must be stored in the repository") from error
    registered = (*D094_REGISTERED_SOURCE_FILES, split_relative)
    missing = [relative for relative in registered if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"D094 extension source files are missing: {missing}")
    return tuple(sorted(registered))


def load_split_manifest(path: Path, *, require_registered: bool = True) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != SPLIT_SCHEMA:
        raise ValueError("unsupported bump scaling split manifest")

    claimed_canonical = payload.get("canonical_payload_sha256")
    canonical_payload = dict(payload)
    canonical_payload.pop("canonical_payload_sha256", None)
    observed_canonical = canonical_json_sha256(canonical_payload)
    if claimed_canonical != observed_canonical:
        raise ValueError("split manifest canonical payload digest does not close")

    split = payload.get("split")
    nested = payload.get("nested_exposure")
    if not isinstance(split, Mapping) or not isinstance(nested, Mapping):
        raise TypeError("split manifest lacks split or nested exposure metadata")
    observed_partition = canonical_json_sha256(
        {"split": split, "nested_exposure": nested}
    )
    if payload.get("partition_digest") != observed_partition:
        raise ValueError("split manifest partition digest does not close")
    if payload.get("state_arrays_opened") is not False:
        raise ValueError("the registered split must be field blind")
    if payload.get("historical_test_population_opened") is not False:
        raise ValueError("the historical test population must remain unopened")

    train_pool = [str(key) for key in split.get("train_pool_keys", [])]
    validation = [str(key) for key in split.get("open_validation_keys", [])]
    if len(train_pool) != len(set(train_pool)) or len(validation) != len(set(validation)):
        raise ValueError("split keys must be unique")
    if set(train_pool) & set(validation):
        raise ValueError("training and open-validation keys overlap")
    if int(split.get("train_pool_count", -1)) != len(train_pool):
        raise ValueError("training-pool count does not match its keys")
    if int(split.get("open_validation_count", -1)) != len(validation):
        raise ValueError("validation count does not match its keys")

    counts = tuple(int(value) for value in nested.get("counts", []))
    subsets = nested.get("subsets")
    if counts != REGISTERED_COUNTS or not isinstance(subsets, Mapping):
        raise ValueError("split manifest does not contain the registered exposure ladder")
    prior: set[str] = set()
    for count in counts:
        keys = [str(key) for key in subsets.get(str(count), [])]
        current = set(keys)
        if len(keys) != count or len(current) != count:
            raise ValueError(f"n={count} subset has the wrong cardinality")
        if not prior.issubset(current) or not current.issubset(set(train_pool)):
            raise ValueError("training subsets are not nested inside the train pool")
        prior = current
    if prior != set(train_pool):
        raise ValueError("n=256 subset does not close the frozen train pool")

    if require_registered:
        expected = {
            "source_manifest_sha256": REGISTERED_SOURCE_MANIFEST_SHA256,
            "partition_digest": REGISTERED_PARTITION_DIGEST,
            "canonical_payload_sha256": REGISTERED_CANONICAL_PAYLOAD_SHA256,
        }
        for name, value in expected.items():
            if payload.get(name) != value:
                raise ValueError(f"registered split field {name} changed")
    return payload


def _seed_from_parts(*values: Any) -> int:
    encoded = "\x1f".join(str(value) for value in values).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little")


def _window_permutation(
    *, seed: int, key: str, cycle: int, legal_window_count: int
) -> np.ndarray:
    rng = np.random.default_rng(_seed_from_parts(SCHEMA, seed, key, cycle))
    return rng.permutation(legal_window_count)


def balanced_queue_presentations(
    store: Any,
    keys: Sequence[str],
    *,
    step_stride: int,
    count: int,
    seed: int,
    epoch: int,
) -> list[tuple[str, int]]:
    """Return balanced pairs with per-trajectory windows exhausted before reuse."""

    trajectory_keys = [str(key) for key in keys]
    if not trajectory_keys or len(trajectory_keys) != len(set(trajectory_keys)):
        raise ValueError("trajectory keys must be nonempty and unique")
    if step_stride < 1 or count < 1 or epoch < 0:
        raise ValueError("stride/count must be positive and epoch nonnegative")
    if count % len(trajectory_keys) != 0:
        raise ValueError("presentations per epoch must divide evenly across trajectories")
    quota = count // len(trajectory_keys)

    per_key: dict[str, list[int]] = {}
    for key in trajectory_keys:
        legal_count = int(store.entry(key)["num_steps"]) - int(step_stride)
        if legal_count < 1:
            raise ValueError(f"trajectory {key} has no legal one-step window")
        start = epoch * quota
        windows: list[int] = []
        permutation_cache: dict[int, np.ndarray] = {}
        for local_index in range(quota):
            ordinal = start + local_index
            cycle, offset = divmod(ordinal, legal_count)
            permutation = permutation_cache.get(cycle)
            if permutation is None:
                permutation = _window_permutation(
                    seed=seed,
                    key=key,
                    cycle=cycle,
                    legal_window_count=legal_count,
                )
                permutation_cache[cycle] = permutation
            windows.append(int(permutation[offset]))
        per_key[key] = windows

    pairs: list[tuple[str, int]] = []
    for local_index in range(quota):
        order_rng = np.random.default_rng(
            _seed_from_parts(SCHEMA, "trajectory-order", seed, epoch, local_index)
        )
        for key in order_rng.permutation(trajectory_keys).tolist():
            pairs.append((str(key), per_key[str(key)][local_index]))
    assert_balanced_epoch_stream(pairs, trajectory_keys, expected_count=count)
    return pairs


def assert_balanced_epoch_stream(
    pairs: Sequence[tuple[str, int]],
    keys: Sequence[str],
    *,
    expected_count: int,
) -> None:
    trajectory_keys = [str(key) for key in keys]
    if len(pairs) != expected_count:
        raise ValueError("presentation stream has the wrong length")
    if expected_count % len(trajectory_keys) != 0:
        raise ValueError("expected count is not trajectory balanced")
    expected_per_key = expected_count // len(trajectory_keys)
    counts = Counter(str(key) for key, _ in pairs)
    if counts != Counter({key: expected_per_key for key in trajectory_keys}):
        raise ValueError("presentation stream is not exactly trajectory balanced")
    if any(int(time_index) < 0 for _, time_index in pairs):
        raise ValueError("presentation stream contains a negative time index")


def fixed_evaluation_pairs(
    store: Any,
    keys: Sequence[str],
    *,
    step_stride: int,
    windows_per_trajectory: int,
) -> list[tuple[str, int]]:
    """Choose field-blind, evenly spaced truth-input windows for every trajectory."""

    if windows_per_trajectory < 1:
        raise ValueError("windows per trajectory must be positive")
    pairs: list[tuple[str, int]] = []
    for raw_key in keys:
        key = str(raw_key)
        legal_count = int(store.entry(key)["num_steps"]) - int(step_stride)
        if windows_per_trajectory > legal_count:
            raise ValueError("requested more evaluation windows than are legal")
        indices = np.linspace(
            0, legal_count - 1, num=windows_per_trajectory, dtype=np.int64
        )
        if len({int(value) for value in indices}) != windows_per_trajectory:
            raise ValueError("evaluation anchors are not unique")
        pairs.extend((key, int(value)) for value in indices)
    return pairs


def presentation_stream_sha256(pairs: Sequence[tuple[str, int]]) -> str:
    digest = hashlib.sha256()
    for key, time_index in pairs:
        encoded = str(key).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little", signed=False))
        digest.update(encoded)
        digest.update(int(time_index).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def parse_wrapper_args(
    argv: Sequence[str] | None = None,
) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument(
        "--trajectory-count", type=int, required=True, choices=REGISTERED_COUNTS
    )
    parser.add_argument(
        "--differential-branch-mode",
        required=True,
        choices=DIFFERENTIAL_BRANCH_MODES,
    )
    parser.add_argument(
        "--optimizer-steps-per-epoch",
        type=int,
        default=D094_OPTIMIZER_STEPS_PER_EPOCH,
    )
    parser.add_argument("--evaluation-windows-per-trajectory", type=int, default=4)
    parser.add_argument("--comparable-seen-every-epochs", type=int, default=5)
    parser.add_argument("--sentinel-every-epochs", type=int, default=0)
    parser.add_argument("--sentinel-steps", type=int, nargs="*", default=())
    parser.add_argument(
        "--sentinel-payload",
        choices=SENTINEL_PAYLOAD_MODES,
        default=MODEL_ONLY_SENTINEL_PAYLOAD,
    )
    parser.add_argument("--engineering-smoke", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--help", action="store_true")
    wrapper, remaining = parser.parse_known_args(argv)
    if wrapper.help:
        parser.print_help()
        print("\nAll remaining options are passed to train_pcno_euler2d_residual.py.")
        raise SystemExit(0)
    return wrapper, remaining


def configure_parent_args(
    args: argparse.Namespace,
    wrapper: argparse.Namespace,
    split_manifest: Mapping[str, Any],
) -> argparse.Namespace:
    validation_count = len(split_manifest["split"]["open_validation_keys"])
    args.split_mode = "stratified"
    args.split_seed = int(split_manifest["split"]["seed"])
    args.val_count = validation_count
    args.presentation_mode = "balanced"
    args.presentations_per_epoch = int(wrapper.optimizer_steps_per_epoch)
    args.batch_size = 1
    args.gradient_accumulation_steps = 1
    args.tiny_pairs = 0
    args.val_presentations = (
        validation_count * int(wrapper.evaluation_windows_per_trajectory)
    )
    args.checkpoint_every = 1
    args.rollout_val_count = D094_ROLLOUT_SELECTION_COUNT
    args.rollout_steps = D094_ROLLOUT_STEPS
    args.rollout_checkpoints = sorted(
        {int(value) for value in args.rollout_checkpoints} | {D094_ROLLOUT_STEPS}
    )
    args.selection_mode = D094_SELECTION_MODE
    args.rollout_failure_policy = FINITE_ONLY_ROLLOUT_POLICY
    args.differential_branch_mode = str(wrapper.differential_branch_mode)
    args.trajectory_count = int(wrapper.trajectory_count)
    args.scaling_partition_digest = str(split_manifest["partition_digest"])
    args.source_snapshot_extra_files = d094_source_snapshot_files(
        wrapper.split_manifest
    )

    if args.presentations_per_epoch % args.trajectory_count != 0:
        raise ValueError("optimizer steps per epoch must be divisible by trajectory count")
    if wrapper.comparable_seen_every_epochs < 1:
        raise ValueError("the comparable-seen diagnostic period must be positive")
    if wrapper.sentinel_every_epochs < 0:
        raise ValueError("the sentinel period must be nonnegative")
    sentinel_steps = tuple(sorted(int(step) for step in wrapper.sentinel_steps))
    if len(sentinel_steps) != len(set(sentinel_steps)) or any(
        step < 1 for step in sentinel_steps
    ):
        raise ValueError("sentinel steps must be unique positive integers")
    if sentinel_steps and wrapper.sentinel_every_epochs:
        raise ValueError("sentinel steps and a periodic sentinel schedule are exclusive")
    total_optimizer_steps = args.epochs * args.presentations_per_epoch
    if any(
        step > total_optimizer_steps or step % args.presentations_per_epoch != 0
        for step in sentinel_steps
    ):
        raise ValueError(
            "sentinel steps must be epoch boundaries inside the requested run"
        )
    wrapper.sentinel_steps = sentinel_steps
    if wrapper.engineering_smoke:
        args.d094_schedule_arm = "engineering_smoke"
    else:
        expected_total = D094_EPOCHS * D094_OPTIMIZER_STEPS_PER_EPOCH
        if args.epochs != D094_EPOCHS:
            raise ValueError(f"the registered D094 schedule uses {D094_EPOCHS} epochs")
        if args.presentations_per_epoch != D094_OPTIMIZER_STEPS_PER_EPOCH:
            raise ValueError(
                "the registered D094 schedule uses 256 optimizer steps per epoch"
            )
        if total_optimizer_steps != expected_total:
            raise ValueError("the registered D094 schedule uses 20,480 optimizer steps")
        if args.scheduler != "warmup_cosine":
            raise ValueError("the registered D094 schedule uses warmup_cosine")
        try:
            args.d094_schedule_arm = D094_SCHEDULE_ARMS_BY_DECAY_STEPS[
                int(args.warmup_cosine_decay_steps)
            ]
        except KeyError as error:
            raise ValueError(
                "D094 requires an explicit 5,120-step prefix-tail or 20,480-step "
                "stretched cosine arm"
            ) from error
        if wrapper.sentinel_every_epochs != 0:
            raise ValueError("the registered D094 sentinel schedule is step-explicit")
        if sentinel_steps != D094_SENTINEL_STEPS:
            raise ValueError("the registered D094 sentinel-step inventory is exact")
        if wrapper.sentinel_payload != MODEL_ONLY_SENTINEL_PAYLOAD:
            raise ValueError("registered D094 sentinels are model-only")
    if args.init_checkpoint is not None or args.resume_checkpoint is not None:
        raise ValueError("the D094 engineering adapter does not yet admit init/resume")
    if args.step_stride != 1:
        raise ValueError("the registered bump scaling contract uses one-step stride")
    if args.multistep_loss_steps != 1 or args.multistep_loss_weight != 0.0:
        raise ValueError("the primary bump scaling arm is a one-step residual objective")
    if args.generated_state_exposure_weight != 0.0 or args.input_noise_std != 0.0:
        raise ValueError("exposure and input-noise branches are outside D094 B1")
    return args


def _annotate_checkpoint(
    payload: dict[str, Any],
    *,
    model: torch.nn.Module,
    scaling_contract: Mapping[str, Any],
    presentation_hashes: Sequence[str],
) -> dict[str, Any]:
    branch_contract = dict(model.differential_branch_contract)
    branch_contract["scaling_schema"] = SCHEMA
    payload["differential_branch_contract"] = branch_contract
    payload["model_config"] = dict(payload["model_config"])
    payload["model_config"]["differential_branch_mode"] = branch_contract["mode"]
    checkpoint_contract = dict(scaling_contract)
    checkpoint_contract["presentation_stream_sha256_by_epoch"] = list(
        presentation_hashes
    )
    payload["bump_scaling_contract"] = checkpoint_contract
    payload["training_args"] = dict(payload.get("training_args", {}))
    payload["training_args"]["differential_branch_mode"] = branch_contract["mode"]
    payload["config_digest"] = canonical_json_sha256(payload["training_args"])
    payload["resume_supported"] = False
    payload["resume_blocker"] = (
        "D094 exact-resume exposure accounting is not implemented"
    )
    return payload


def model_only_sentinel_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Strip resume-only state while retaining an evaluation-loadable model."""

    if "model_state" not in payload or "epoch" not in payload:
        raise ValueError("a model-only sentinel requires model state and epoch")
    sentinel = dict(payload)
    sentinel.pop("optimizer_state", None)
    sentinel.pop("scheduler_state", None)
    sentinel["checkpoint_role"] = "model_only_sentinel"
    sentinel["resume_supported"] = False
    sentinel["sentinel_contract"] = {
        "schema": "d094_model_only_sentinel_v1",
        "evaluation_initialization_supported": True,
        "exact_training_resume_supported": False,
    }
    return sentinel


def retain_sentinel_at_epoch(
    *, epoch: int, epochs: int, presentations_per_epoch: int, wrapper: argparse.Namespace
) -> bool:
    """Resolve sparse exact-step retention or the legacy periodic fallback."""

    step = (epoch + 1) * presentations_per_epoch
    if wrapper.sentinel_steps:
        return step in wrapper.sentinel_steps
    return (
        epoch == 0
        or epoch + 1 == epochs
        or (
            wrapper.sentinel_every_epochs > 0
            and (epoch + 1) % wrapper.sentinel_every_epochs == 0
        )
    )


@contextmanager
def installed_scaling_adapter(
    trainer: ModuleType,
    *,
    args: argparse.Namespace,
    wrapper: argparse.Namespace,
    train_keys: Sequence[str],
    validation_keys: Sequence[str],
    scaling_contract: Mapping[str, Any],
    presentation_hashes: list[str],
) -> Iterable[None]:
    originals: list[tuple[Any, str, Any]] = []

    def replace(owner: Any, name: str, value: Any) -> None:
        originals.append((owner, name, getattr(owner, name)))
        setattr(owner, name, value)

    original_build_model = trainer.build_model
    original_train_epoch = trainer.train_epoch
    original_checkpoint_payload = trainer.checkpoint_payload
    original_split = trainer.stratified_train_val_split
    original_balanced = trainer.balanced_presentations
    original_atomic_save = trainer.atomic_torch_save

    def parse_args(_: Sequence[str] | None = None) -> argparse.Namespace:
        return args

    def split(
        store: Any,
        *,
        val_count: int,
        seed: int,
        keys: Sequence[str] | None = None,
    ) -> tuple[list[str], list[str]]:
        if keys is None:
            if val_count != len(validation_keys) or seed != args.split_seed:
                raise ValueError("parent split request differs from D094 manifest")
            return list(train_keys), list(validation_keys)
        return original_split(store, val_count=val_count, seed=seed, keys=keys)

    def epoch_presentations(
        store: Any,
        keys: Sequence[str],
        *,
        args: argparse.Namespace,
        epoch: int,
        tiny_bank: Sequence[tuple[str, int]] | None,
    ) -> list[tuple[str, int]]:
        if tiny_bank is not None or list(map(str, keys)) != list(train_keys):
            raise ValueError("D094 received an unexpected training population")
        return balanced_queue_presentations(
            store,
            keys,
            step_stride=args.step_stride,
            count=args.presentations_per_epoch,
            seed=args.seed,
            epoch=epoch,
        )

    def balanced_presentations(
        store: Any,
        keys: Sequence[str],
        *,
        step_stride: int,
        count: int,
        rng: np.random.Generator,
        minimum_time_index: int = 0,
    ) -> list[tuple[str, int]]:
        if (
            list(map(str, keys)) == list(validation_keys)
            and count == args.val_presentations
            and minimum_time_index == 0
        ):
            return fixed_evaluation_pairs(
                store,
                validation_keys,
                step_stride=step_stride,
                windows_per_trajectory=wrapper.evaluation_windows_per_trajectory,
            )
        return original_balanced(
            store,
            keys,
            step_stride=step_stride,
            count=count,
            rng=rng,
            minimum_time_index=minimum_time_index,
        )

    def build_model(*values: Any, **keywords: Any) -> torch.nn.Module:
        model = original_build_model(*values, **keywords)
        apply_differential_branch_mode(model, wrapper.differential_branch_mode)
        return model

    train_epoch_index = 0

    def train_epoch(*values: Any, **keywords: Any) -> dict[str, Any]:
        nonlocal train_epoch_index
        pairs = values[2] if len(values) >= 3 else keywords["pairs"]
        assert_balanced_epoch_stream(
            pairs, train_keys, expected_count=args.presentations_per_epoch
        )
        presentation_hashes.append(presentation_stream_sha256(pairs))
        metrics = original_train_epoch(*values, **keywords)
        epoch = train_epoch_index
        train_epoch_index += 1
        should_compare = (
            epoch == 0
            or epoch + 1 == args.epochs
            or (epoch + 1) % wrapper.comparable_seen_every_epochs == 0
        )
        metrics["comparable_seen"] = None
        if should_compare:
            model = values[0] if values else keywords["model"]
            store = values[1] if len(values) >= 2 else keywords["store"]
            seen_pairs = fixed_evaluation_pairs(
                store,
                train_keys,
                step_stride=args.step_stride,
                windows_per_trajectory=wrapper.evaluation_windows_per_trajectory,
            )
            metrics["comparable_seen"] = trainer.evaluate_pairs(
                model,
                store,
                seen_pairs,
                step_stride=args.step_stride,
                batch_size=1,
                device=keywords["device"],
                amp=keywords["amp"],
                primary_objective=keywords["primary_objective"],
                boundary_policies=keywords["boundary_policies"],
            )
            metrics["comparable_seen"]["pair_bank_sha256"] = (
                presentation_stream_sha256(seen_pairs)
            )
        metrics["completed_optimizer_steps"] = (
            (epoch + 1) * args.presentations_per_epoch
        )
        metrics["window_equivalent_exposure_per_trajectory"] = (
            metrics["completed_optimizer_steps"] / (79 * len(train_keys))
        )
        return metrics

    def checkpoint_payload(*values: Any, **keywords: Any) -> dict[str, Any]:
        payload = original_checkpoint_payload(*values, **keywords)
        model = keywords.get("model")
        if model is None:
            raise TypeError("checkpoint payload did not receive model by keyword")
        return _annotate_checkpoint(
            payload,
            model=model,
            scaling_contract=scaling_contract,
            presentation_hashes=presentation_hashes,
        )

    def atomic_torch_save(payload: Any, path: Path) -> None:
        original_atomic_save(payload, path)
        if path.name != "last.pt" or not isinstance(payload, Mapping):
            return
        epoch = int(payload["epoch"])
        should_retain = retain_sentinel_at_epoch(
            epoch=epoch,
            epochs=args.epochs,
            presentations_per_epoch=args.presentations_per_epoch,
            wrapper=wrapper,
        )
        if should_retain:
            step = (epoch + 1) * args.presentations_per_epoch
            sentinel = path.parent / "sentinels" / f"step_{step:09d}.pt"
            sentinel.parent.mkdir(parents=True, exist_ok=True)
            sentinel_payload = (
                payload
                if wrapper.sentinel_payload == FULL_SENTINEL_PAYLOAD
                else model_only_sentinel_payload(payload)
            )
            original_atomic_save(sentinel_payload, sentinel)

    replace(trainer, "parse_args", parse_args)
    replace(trainer, "stratified_train_val_split", split)
    replace(trainer, "epoch_presentations", epoch_presentations)
    replace(trainer, "balanced_presentations", balanced_presentations)
    replace(trainer, "build_model", build_model)
    replace(trainer, "train_epoch", train_epoch)
    replace(trainer, "checkpoint_payload", checkpoint_payload)
    replace(trainer, "atomic_torch_save", atomic_torch_save)
    try:
        yield
    finally:
        for owner, name, value in reversed(originals):
            setattr(owner, name, value)


def _source_hashes(split_manifest_path: Path) -> dict[str, str]:
    return {
        relative: sha256_file(ROOT / relative)
        for relative in d094_source_snapshot_files(split_manifest_path)
    }


def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    wrapper, trainer_argv = parse_wrapper_args(argv)
    split_manifest = load_split_manifest(wrapper.split_manifest)
    from scripts.time_dependent_no import train_pcno_euler2d_residual as trainer

    args = configure_parent_args(
        trainer.parse_args(trainer_argv), wrapper, split_manifest
    )
    subset = split_manifest["nested_exposure"]["subsets"][str(wrapper.trajectory_count)]
    train_keys = [str(key) for key in subset]
    validation_keys = [
        str(key) for key in split_manifest["split"]["open_validation_keys"]
    ]

    store = trainer.PCNOEuler2DShardStore(args.data_dir)
    try:
        if str(store.manifest_digest) != split_manifest["source_manifest_sha256"]:
            raise ValueError("training shards do not match the frozen split source")
        unknown = sorted(
            (set(train_keys) | set(validation_keys)) - set(map(str, store.keys))
        )
        if unknown:
            raise ValueError(f"split references unknown trajectory keys: {unknown}")
        fixed_validation = fixed_evaluation_pairs(
            store,
            validation_keys,
            step_stride=args.step_stride,
            windows_per_trajectory=wrapper.evaluation_windows_per_trajectory,
        )
        fixed_seen = fixed_evaluation_pairs(
            store,
            train_keys,
            step_stride=args.step_stride,
            windows_per_trajectory=wrapper.evaluation_windows_per_trajectory,
        )
        first_epoch_stream = balanced_queue_presentations(
            store,
            train_keys,
            step_stride=args.step_stride,
            count=args.presentations_per_epoch,
            seed=args.seed,
            epoch=0,
        )
    finally:
        store.close()

    trainer.validate_args(args, trainer.select_device(args.device))

    source_hashes = _source_hashes(wrapper.split_manifest)
    scaling_contract = {
        "schema": SCHEMA,
        "status": "ready_before_parent_trainer",
        "science_result_eligible": False,
        "engineering_smoke": bool(wrapper.engineering_smoke),
        "trajectory_count": len(train_keys),
        "train_keys": train_keys,
        "open_validation_count": len(validation_keys),
        "historical_test_population_accessed": False,
        "split_schema": split_manifest["schema"],
        "split_partition_digest": split_manifest["partition_digest"],
        "split_canonical_payload_sha256": split_manifest[
            "canonical_payload_sha256"
        ],
        "source_manifest_sha256": split_manifest["source_manifest_sha256"],
        "differential_branch_mode": wrapper.differential_branch_mode,
        "optimizer_steps_per_epoch": args.presentations_per_epoch,
        "requested_epochs": args.epochs,
        "requested_optimizer_steps": args.epochs * args.presentations_per_epoch,
        "schedule_arm": args.d094_schedule_arm,
        "scheduler": args.scheduler,
        "warmup_cosine_decay_steps": args.warmup_cosine_decay_steps,
        "microbatch_size": 1,
        "gradient_accumulation_steps": 1,
        "within_update_duplicate_pairs": 0,
        "sampler": (
            "balanced trajectory cycle; independent per-trajectory shuffled legal-"
            "window queues exhausted before reshuffle"
        ),
        "legal_windows_per_trajectory": 79,
        "window_equivalent_exposure_formula": "optimizer_steps / (79 * n)",
        "fixed_evaluation_windows_per_trajectory": (
            wrapper.evaluation_windows_per_trajectory
        ),
        "fixed_validation_pair_bank_sha256": presentation_stream_sha256(
            fixed_validation
        ),
        "fixed_seen_pair_bank_sha256": presentation_stream_sha256(fixed_seen),
        "first_epoch_presentation_stream_sha256": presentation_stream_sha256(
            first_epoch_stream
        ),
        "comparable_seen_every_epochs": wrapper.comparable_seen_every_epochs,
        "sentinel_every_epochs": wrapper.sentinel_every_epochs,
        "sentinel_steps": list(wrapper.sentinel_steps),
        "sentinel_payload": wrapper.sentinel_payload,
        "checkpoint_selection_mode": args.selection_mode,
        "rollout_failure_policy": args.rollout_failure_policy,
        "rollout_selection_trajectory_count": args.rollout_val_count,
        "rollout_steps": args.rollout_steps,
        "metric_semantics": dict(D094_METRIC_SEMANTICS),
        "source_sha256": source_hashes,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "bump_scaling_preflight.json", scaling_contract)
    if wrapper.preflight_only:
        preflight_contract = dict(scaling_contract)
        preflight_contract.update(
            {
                "status": "preflight_complete",
                "science_result_eligible": False,
                "science_ineligibility_reason": "preflight only; no model was built",
            }
        )
        write_json(
            args.output_dir / "bump_scaling_contract.json", preflight_contract
        )
        return preflight_contract

    presentation_hashes: list[str] = []
    with installed_scaling_adapter(
        trainer,
        args=args,
        wrapper=wrapper,
        train_keys=train_keys,
        validation_keys=validation_keys,
        scaling_contract=scaling_contract,
        presentation_hashes=presentation_hashes,
    ):
        trainer.main([])

    summary_path = args.output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    completed_epochs = int(summary["completed_epochs"])
    actual_steps = int(summary["actual_optimizer_steps"])
    expected_steps = completed_epochs * args.presentations_per_epoch
    if actual_steps != expected_steps or len(presentation_hashes) != completed_epochs:
        raise RuntimeError("completed scaling exposure accounting does not close")
    metric_rows = [
        json.loads(line)
        for line in (args.output_dir / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    metric_receipt = build_d094_metric_receipt(summary, metric_rows, scaling_contract)
    write_json(args.output_dir / "d094_metric_receipt.json", metric_receipt)
    final_contract = dict(scaling_contract)
    final_contract.update(
        {
            "status": (
                "completed_engineering_smoke"
                if wrapper.engineering_smoke
                else "completed_unreplicated_pilot"
            ),
            "completed_epochs": completed_epochs,
            "actual_optimizer_steps": actual_steps,
            "actual_window_equivalent_exposure_per_trajectory": (
                actual_steps / (79 * len(train_keys))
            ),
            "presentation_stream_sha256_by_epoch": presentation_hashes,
            "metric_receipt_schema": D094_METRIC_RECEIPT_SCHEMA,
            "metric_receipt_file": "d094_metric_receipt.json",
            "science_result_eligible": False,
            "science_ineligibility_reason": (
                "engineering smoke only"
                if wrapper.engineering_smoke
                else "single unreplicated pilot; exact-resume gate remains open"
            ),
        }
    )
    write_json(args.output_dir / "bump_scaling_contract.json", final_contract)
    return final_contract


def main(argv: Sequence[str] | None = None) -> int:
    contract = run(argv)
    print(json.dumps(contract, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
