#!/usr/bin/env python3
"""Run the matched W26-L2 bump full/no-gradient training adapter.

This entry point is intentionally an adapter around the frozen L3R-B1 trainer.
It leaves the seven B1 source files byte-identical, verifies their registered
hashes, and changes only the differential-branch execution.  The underlying
trainer still owns data loading, causal boundary closure, losses, validation,
checkpoint selection, and artifact writing.

The registered run uses 34 completed full-coverage passes but follows the
original 40-pass/216,000-step B1 learning-rate schedule.  This separates the
scientific exposure match from B1's throughput-dependent 23-hour stop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utility.time_dependent_no.pcno_differential_branch import (
    DIFFERENTIAL_BRANCH_MODES,
    _is_differential_state_name,
    apply_differential_branch_mode,
    model_state_sha256,
)

SCHEMA = "w26_l2_bump_gradient_ablation_v1"
REGISTERED_SEEDS = (20260718, 20260812, 20260813)

B1_DATA_MANIFEST_SHA256 = (
    "5d5373fdcc682544bf330fba6d54fe65509dabe936baee7443c71a8c0c8d9fa7"
)
B1_BOUNDARY_POLICY_SET_SHA256 = (
    "c7d9f92dadeaef46bcb0c62c0f26458f08b7bfc6d6ec4e124aa840e9a246c13c"
)
B1_SOURCE_SET_SHA256 = (
    "61ea3d2c650a0d5d67b2020c4a97a08dc5ffdfb55498da0a3b26129437c5b730"
)
B1_FROZEN_SOURCE_SHA256 = {
    "docs/time_dependent_no/MECHANISTIC_DIAGNOSTIC_TRACKER.md": (
        "c3014df67a4e7d5ed8f1d48b1c76f664b5a7604557d7ce0b783ca77a19cc7ee4"
    ),
    "docs/time_dependent_no/RESEARCH_DIRECTION_DECISION.md": (
        "dafa88e4bca3b174ff9711272259e6a449dab9d744dac24d0254e6a61d7c2de3"
    ),
    "pcno/pcno.py": (
        "c5bb98fe736b370f277de935c88f6fb22efc23417a0dbab1bb9a08e979a6298b"
    ),
    "scripts/time_dependent_no/evaluate_pcno_euler2d_residual.py": (
        "cf4da6cdbe508a34ebef5d2fa486350108ade098807e738ea6f0d7b0b72f8259"
    ),
    "scripts/time_dependent_no/train_pcno_euler2d_residual.py": (
        "7f7733b421f74411a420d753b51a5c56a798f578462806a37054bee16bc82318"
    ),
    "utility/time_dependent_no/cpg_mesh_contract.py": (
        "02c1a651d30aa42d0be0a20c743c8c65e75a42076ec596edecee97fa4885641b"
    ),
    "utility/time_dependent_no/pcno_euler2d.py": (
        "be15a8df2242f6c632a89b13ebfa2c018e4ef8a1a29954f6869d89ada096a780"
    ),
}

EXTENSION_PATHS = (
    "docs/time_dependent_no/W26_L2_BUMP_GRADIENT_ABLATION_PREREGISTRATION.md",
    "scripts/time_dependent_no/train_pcno_bump_gradient_ablation.py",
    "scripts/time_dependent_no/analyze_pcno_bump_gradient_ablation.py",
    "scripts/time_dependent_no/decompose_pcno_b1_frozen_rollout_error.py",
    "tests/time_dependent_no/test_pcno_bump_gradient_ablation.py",
)

REGISTERED_COMPLETED_PASSES = 34
REGISTERED_SCHEDULER_PASSES = 40
PRESENTATIONS_PER_PASS = 21_330
OPTIMIZER_STEPS_PER_PASS = 5_400
REGISTERED_PRESENTATIONS = REGISTERED_COMPLETED_PASSES * PRESENTATIONS_PER_PASS
REGISTERED_OPTIMIZER_STEPS = REGISTERED_COMPLETED_PASSES * OPTIMIZER_STEPS_PER_PASS
SCHEDULER_OPTIMIZER_STEPS = REGISTERED_SCHEDULER_PASSES * OPTIMIZER_STEPS_PER_PASS
SCHEDULER_WARMUP_STEPS = round(0.02 * SCHEDULER_OPTIMIZER_STEPS)

B1_VALIDATION_KEYS = (
    "7",
    "16",
    "18",
    "23",
    "47",
    "54",
    "58",
    "60",
    "72",
    "82",
    "101",
    "103",
    "112",
    "120",
    "126",
    "128",
    "141",
    "145",
    "150",
    "172",
    "187",
    "188",
    "190",
    "211",
    "227",
    "233",
    "235",
    "251",
    "287",
    "296",
)
B1_PARITY_KEYS = ("16", "60", "128", "141", "235")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mapping_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def verify_frozen_b1_sources(root: Path = ROOT) -> dict[str, str]:
    """Require the exact seven-file L3R-B1 source surface."""

    observed: dict[str, str] = {}
    mismatches: dict[str, dict[str, str | None]] = {}
    for relative, expected in B1_FROZEN_SOURCE_SHA256.items():
        path = root / relative
        actual = sha256_file(path) if path.is_file() else None
        if actual is not None:
            observed[relative] = actual
        if actual != expected:
            mismatches[relative] = {"expected": expected, "actual": actual}
    if mismatches:
        raise ValueError(f"frozen B1 source mismatch: {mismatches}")
    return observed


def extension_source_hashes(root: Path = ROOT) -> dict[str, str]:
    missing = [
        relative for relative in EXTENSION_PATHS if not (root / relative).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"A1 extension files are missing: {missing}")
    return {relative: sha256_file(root / relative) for relative in EXTENSION_PATHS}


def load_checkpoint_for_ablation(
    path: Path, *, device: str | torch.device = "cpu"
) -> dict[str, Any]:
    """Load an ablation checkpoint without constructing or executing its model."""

    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint payload is not a mapping: {path}")
    checkpoint_differential_branch_mode(payload)
    return payload


def presentation_stream_sha256(pairs: Sequence[tuple[str, int]]) -> str:
    digest = hashlib.sha256()
    for key, time_index in pairs:
        encoded = str(key).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little", signed=False))
        digest.update(encoded)
        digest.update(int(time_index).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def checkpoint_differential_branch_mode(checkpoint: Mapping[str, Any]) -> str:
    contract = checkpoint.get("differential_branch_contract")
    if not isinstance(contract, Mapping):
        raise TypeError("checkpoint lacks its differential-branch contract")
    mode = str(contract.get("mode"))
    configured = str(checkpoint.get("model_config", {}).get("differential_branch_mode"))
    if mode not in DIFFERENTIAL_BRANCH_MODES or configured != mode:
        raise ValueError("checkpoint differential-branch declarations disagree")
    return mode


def registered_schedule_factor(step_index: int) -> float:
    """Return the exact original B1 40-pass warmup/cosine factor."""

    step = min(max(int(step_index), 0), SCHEDULER_OPTIMIZER_STEPS - 1)
    if step < SCHEDULER_WARMUP_STEPS:
        progress = step / (SCHEDULER_WARMUP_STEPS - 1)
        return 0.1 + progress * 0.9
    decay_steps = SCHEDULER_OPTIMIZER_STEPS - SCHEDULER_WARMUP_STEPS
    progress = (step - SCHEDULER_WARMUP_STEPS) / (decay_steps - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return 0.02 + 0.98 * cosine


def registered_matched_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "seed": int(args.seed),
        "split_seed": int(args.split_seed),
        "split_mode": str(args.split_mode),
        "val_count": int(args.val_count),
        "step_stride": int(args.step_stride),
        "completed_passes": REGISTERED_COMPLETED_PASSES,
        "scheduler_definition_passes": REGISTERED_SCHEDULER_PASSES,
        "presentations": REGISTERED_PRESENTATIONS,
        "optimizer_steps": REGISTERED_OPTIMIZER_STEPS,
        "scheduler_optimizer_steps": SCHEDULER_OPTIMIZER_STEPS,
        "presentation_mode": str(args.presentation_mode),
        "batch_size": int(args.batch_size),
        "k_max": int(args.k_max),
        "domain_lengths": [float(value) for value in args.domain_lengths],
        "layers": [int(value) for value in args.layers],
        "fc_dim": int(args.fc_dim),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "scheduler": str(args.scheduler),
        "warmup_steps": SCHEDULER_WARMUP_STEPS,
        "minimum_learning_rate": float(args.min_learning_rate),
        "gradient_clip": float(args.gradient_clip),
        "boundary_mode": str(args.boundary_mode),
        "boundary_max_source_hops": int(args.boundary_max_source_hops),
        "rollout_steps": int(args.rollout_steps),
        "rollout_checkpoints": [int(value) for value in args.rollout_checkpoints],
        "parity_rollout_keys": [str(value) for value in args.parity_rollout_keys],
        "parity_max_rollout_relative_l2": float(args.parity_max_rollout_relative_l2),
        "parity_max_one_step_relative_l2": float(args.parity_max_one_step_relative_l2),
        "amp": str(args.amp),
    }


def _assert_value(name: str, actual: Any, expected: Any) -> None:
    if isinstance(expected, float):
        passed = math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1.0e-12)
    elif isinstance(expected, tuple):
        passed = tuple(actual) == expected
    else:
        passed = actual == expected
    if not passed:
        raise ValueError(
            f"registered B1 argument {name}={actual!r}, expected {expected!r}"
        )


def validate_registered_args(args: argparse.Namespace) -> None:
    expected = {
        "split_seed": 20260718,
        "split_mode": "stratified",
        "val_count": 30,
        "step_stride": 1,
        "epochs": REGISTERED_COMPLETED_PASSES,
        "presentation_mode": "full_coverage",
        "val_presentations": 128,
        "batch_size": 4,
        "stats_time_stride": 1,
        "mach_scale_floor": 0.1,
        "k_max": 8,
        "domain_lengths": (6.0, 2.0),
        "layers": (128, 128, 128, 128, 128),
        "fc_dim": 128,
        "learning_rate": 1.0e-3,
        "weight_decay": 1.0e-5,
        "scheduler": "warmup_cosine",
        "warmup_fraction": 0.02,
        "warmup_start_factor": 0.1,
        "min_learning_rate": 2.0e-5,
        "gradient_clip": 1.0,
        "input_noise_std": 0.0,
        "generated_state_exposure_weight": 0.0,
        "rollout_every": 1,
        "rollout_val_count": 30,
        "rollout_start_frame": 0,
        "rollout_steps": 79,
        "rollout_checkpoints": (20, 40, 60, 79),
        "parity_rollout_keys": B1_PARITY_KEYS,
        "parity_rollout_horizon": 20,
        "parity_max_rollout_relative_l2": 0.024284941703081132,
        "parity_max_one_step_relative_l2": 0.005402445773142972,
        "checkpoint_every": 1,
        "boundary_mode": "causal_nodal_physical",
        "boundary_max_source_hops": 3,
        "boundary_rho_inf": 1.4,
        "boundary_p_inf": 1.0,
        "max_wall_hours": 23.0,
        "device": "cuda",
        "amp": "bf16",
    }
    optional_defaults = {
        "gradient_accumulation_steps": 1,
        "model_node_type_input": "physical",
        "node_type_channel_control": "standard",
        "boundary_field_mode": "none",
        "boundary_residual_mode": "none",
        "primary_objective": "normal_closed",
        "boundary_auxiliary": "none",
        "boundary_auxiliary_weight": 0.0,
        "multistep_loss_steps": 1,
        "multistep_loss_weight": 0.0,
    }
    if int(args.seed) not in REGISTERED_SEEDS:
        raise ValueError(f"production seed must lie in {REGISTERED_SEEDS}")
    for name, value in expected.items():
        _assert_value(name, getattr(args, name), value)
    for name, value in optional_defaults.items():
        if hasattr(args, name):
            _assert_value(name, getattr(args, name), value)
    if getattr(args, "init_checkpoint", None) is not None:
        raise ValueError(
            "the matched ablation starts from common random initialization"
        )
    if getattr(args, "resume_checkpoint", None) is not None:
        raise ValueError(
            "A1 does not register resume provenance; restart an interrupted arm"
        )


def _annotate_checkpoint(
    payload: dict[str, Any],
    model: torch.nn.Module,
    mode: str,
    presentation_hashes: Sequence[str],
    matched_config: Mapping[str, Any],
) -> dict[str, Any]:
    contract = dict(model.differential_branch_contract)
    contract.update(
        {
            "base_source_set_sha256": B1_SOURCE_SET_SHA256,
            "data_manifest_sha256": B1_DATA_MANIFEST_SHA256,
            "boundary_policy_set_sha256": B1_BOUNDARY_POLICY_SET_SHA256,
            "matched_config_sha256": mapping_sha256(matched_config),
            "presentation_stream_sha256_by_pass": list(presentation_hashes),
            "completed_pass_target": REGISTERED_COMPLETED_PASSES,
            "scheduler_definition_passes": REGISTERED_SCHEDULER_PASSES,
            "scheduler_optimizer_steps": SCHEDULER_OPTIMIZER_STEPS,
            "optimizer_excludes_frozen_parameters": mode == "no_gradient",
        }
    )
    payload["differential_branch_contract"] = contract
    payload["model_config"] = dict(payload["model_config"])
    payload["model_config"]["differential_branch_mode"] = mode
    payload["training_args"] = dict(payload.get("training_args", {}))
    payload["training_args"]["differential_branch_mode"] = mode
    payload["training_args"]["scheduler_definition_passes"] = (
        REGISTERED_SCHEDULER_PASSES
    )
    payload["config_digest"] = mapping_sha256(payload["training_args"])
    return payload


@contextmanager
def installed_training_adapter(
    trainer: ModuleType,
    *,
    args: argparse.Namespace,
    mode: str,
    presentation_hashes: list[str],
    matched_config: Mapping[str, Any],
) -> Iterable[None]:
    """Install the isolated hooks, then restore every touched symbol."""

    originals: list[tuple[Any, str, Any]] = []

    def replace(owner: Any, name: str, value: Any) -> None:
        originals.append((owner, name, getattr(owner, name)))
        setattr(owner, name, value)

    original_parse_args = trainer.parse_args
    original_build_model = trainer.build_model
    original_train_epoch = trainer.train_epoch
    original_checkpoint_payload = trainer.checkpoint_payload
    original_adamw = trainer.torch.optim.AdamW
    original_lambda_lr = trainer.torch.optim.lr_scheduler.LambdaLR

    def parse_args(_: Sequence[str] | None = None) -> argparse.Namespace:
        return args

    def build_model(*values: Any, **keywords: Any) -> torch.nn.Module:
        model = original_build_model(*values, **keywords)
        contract = apply_differential_branch_mode(model, mode)
        if contract["differential_layer_count"] != 4:
            raise ValueError(
                "registered B1 model must contain four differential layers"
            )
        if contract["stored_differential_parameters"] != 131_076:
            raise ValueError("registered B1 differential parameter count changed")
        if contract["stored_total_parameters"] != 19_155_720:
            raise ValueError("registered B1 total parameter count changed")
        return model

    def train_epoch(*values: Any, **keywords: Any) -> dict[str, Any]:
        pairs = values[2] if len(values) >= 3 else keywords["pairs"]
        digest = presentation_stream_sha256(pairs)
        if len(pairs) != PRESENTATIONS_PER_PASS:
            raise ValueError("full-coverage presentation count changed")
        presentation_hashes.append(digest)
        return original_train_epoch(*values, **keywords)

    def checkpoint_payload(*values: Any, **keywords: Any) -> dict[str, Any]:
        payload = original_checkpoint_payload(*values, **keywords)
        model = keywords.get("model")
        if model is None:
            raise TypeError("checkpoint payload did not receive model by keyword")
        return _annotate_checkpoint(
            payload,
            model,
            mode,
            presentation_hashes,
            matched_config,
        )

    def adamw(parameters: Iterable[torch.nn.Parameter], *values: Any, **keywords: Any):
        parameter_list = list(parameters)
        if parameter_list and isinstance(parameter_list[0], Mapping):
            groups = []
            for original_group in parameter_list:
                group = dict(original_group)
                group["params"] = [
                    parameter
                    for parameter in group["params"]
                    if parameter.requires_grad
                ]
                groups.append(group)
            resolved_parameters: Any = groups
        else:
            resolved_parameters = [
                parameter for parameter in parameter_list if parameter.requires_grad
            ]
        return original_adamw(resolved_parameters, *values, **keywords)

    def lambda_lr(
        optimizer: torch.optim.Optimizer,
        lr_lambda: Any,
        *values: Any,
        **keywords: Any,
    ):
        del lr_lambda
        return original_lambda_lr(
            optimizer, registered_schedule_factor, *values, **keywords
        )

    replace(trainer, "parse_args", parse_args)
    replace(trainer, "build_model", build_model)
    replace(trainer, "train_epoch", train_epoch)
    replace(trainer, "checkpoint_payload", checkpoint_payload)
    replace(trainer.torch.optim, "AdamW", adamw)
    replace(trainer.torch.optim.lr_scheduler, "LambdaLR", lambda_lr)
    try:
        yield
    finally:
        for owner, name, value in reversed(originals):
            setattr(owner, name, value)
        trainer.parse_args = original_parse_args


def _boundary_digest(contract: Mapping[str, Any]) -> str | None:
    for name in ("policy_set_digest", "boundary_policy_set_digest"):
        value = contract.get(name)
        if value is not None:
            return str(value)
    return None


def _load_json_if_present(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def verify_training_data_manifest(trainer: ModuleType, data_dir: Path) -> str:
    """Read and close the exact B1 shard store without context-manager assumptions."""

    store = trainer.PCNOEuler2DShardStore(data_dir)
    try:
        digest = str(store.manifest_digest)
        if digest != B1_DATA_MANIFEST_SHA256:
            raise ValueError("training shard manifest does not match L3R-B1")
        return digest
    finally:
        store.close()


def _finalize_contract(
    output_dir: Path,
    *,
    mode: str,
    args: argparse.Namespace,
    frozen_sources: Mapping[str, str],
    extension_sources: Mapping[str, str],
    presentation_hashes: Sequence[str],
    matched_config: Mapping[str, Any],
) -> dict[str, Any]:
    summary = _load_json_if_present(output_dir / "summary.json")
    boundary = _load_json_if_present(output_dir / "boundary_contract.json")
    completed_passes = len(presentation_hashes)
    actual_presentations = completed_passes * PRESENTATIONS_PER_PASS
    actual_optimizer_steps = completed_passes * OPTIMIZER_STEPS_PER_PASS
    contract = {
        "schema": SCHEMA,
        "status": "completed"
        if completed_passes == REGISTERED_COMPLETED_PASSES
        else "incomplete",
        "science_result_eligible": completed_passes == REGISTERED_COMPLETED_PASSES,
        "mode": mode,
        "seed": int(args.seed),
        "base_source_set_sha256": B1_SOURCE_SET_SHA256,
        "base_source_sha256": dict(frozen_sources),
        "extension_source_sha256": dict(extension_sources),
        "data_manifest_sha256": B1_DATA_MANIFEST_SHA256,
        "boundary_policy_set_sha256": (
            None if boundary is None else _boundary_digest(boundary)
        ),
        "matched_config": dict(matched_config),
        "matched_config_sha256": mapping_sha256(matched_config),
        "presentation_stream_sha256_by_pass": list(presentation_hashes),
        "completed_passes": completed_passes,
        "actual_presentations": actual_presentations,
        "actual_optimizer_steps": actual_optimizer_steps,
        "registered_presentations": REGISTERED_PRESENTATIONS,
        "registered_optimizer_steps": REGISTERED_OPTIMIZER_STEPS,
        "scheduler": {
            "kind": "warmup_cosine",
            "definition_passes": REGISTERED_SCHEDULER_PASSES,
            "total_optimizer_steps": SCHEDULER_OPTIMIZER_STEPS,
            "warmup_steps": SCHEDULER_WARMUP_STEPS,
            "start_factor": 0.1,
            "minimum_factor": 0.02,
            "executed_prefix_steps": actual_optimizer_steps,
        },
        "scheduler_metadata_authority": {
            "authoritative_record": "ablation_contract.scheduler",
            "base_trainer_record_is_nominal": True,
            "reason": (
                "the frozen trainer derives its metadata from the 34-pass execution "
                "argument; the installed LambdaLR adapter executes the registered "
                "40-pass/216000-step schedule prefix recorded above"
            ),
        },
        "base_summary_present": summary is not None,
        "holdout_or_sealed_population_accessed": False,
    }
    if contract["boundary_policy_set_sha256"] != B1_BOUNDARY_POLICY_SET_SHA256:
        contract["science_result_eligible"] = False
        contract["status"] = "failed_boundary_contract"
    write_json(output_dir / "ablation_contract.json", contract)
    return contract


def parse_wrapper_args(
    argv: Sequence[str] | None = None,
) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=__doc__,
        add_help=False,
    )
    parser.add_argument(
        "--differential-branch-mode",
        required=True,
        choices=DIFFERENTIAL_BRANCH_MODES,
    )
    parser.add_argument("--help", action="store_true")
    wrapper, remaining = parser.parse_known_args(argv)
    if wrapper.help:
        parser.print_help()
        print("\nAll remaining options are passed to train_pcno_euler2d_residual.py.")
        raise SystemExit(0)
    return wrapper, remaining


def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    wrapper, trainer_argv = parse_wrapper_args(argv)
    from scripts.time_dependent_no import train_pcno_euler2d_residual as trainer

    args = trainer.parse_args(trainer_argv)
    args.differential_branch_mode = wrapper.differential_branch_mode
    args.scheduler_definition_passes = REGISTERED_SCHEDULER_PASSES
    args.completed_passes = REGISTERED_COMPLETED_PASSES
    validate_registered_args(args)
    frozen_sources = verify_frozen_b1_sources()
    extension_sources = extension_source_hashes()
    verify_training_data_manifest(trainer, args.data_dir)

    matched_config = registered_matched_config(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preflight = {
        "schema": SCHEMA,
        "status": "ready_before_trainer_main",
        "mode": wrapper.differential_branch_mode,
        "seed": int(args.seed),
        "base_source_set_sha256": B1_SOURCE_SET_SHA256,
        "base_source_sha256": frozen_sources,
        "extension_source_sha256": extension_sources,
        "data_manifest_sha256": B1_DATA_MANIFEST_SHA256,
        "matched_config": matched_config,
        "matched_config_sha256": mapping_sha256(matched_config),
        "historical_b1_is_causal_control": False,
        "reason": (
            "historical B1 retained neither initial-state nor presentation-stream "
            "hashes; a fresh paired full run is required"
        ),
    }
    write_json(output_dir / "ablation_preflight.json", preflight)

    presentation_hashes: list[str] = []
    with installed_training_adapter(
        trainer,
        args=args,
        mode=wrapper.differential_branch_mode,
        presentation_hashes=presentation_hashes,
        matched_config=matched_config,
    ):
        trainer.main([])
    contract = _finalize_contract(
        output_dir,
        mode=wrapper.differential_branch_mode,
        args=args,
        frozen_sources=frozen_sources,
        extension_sources=extension_sources,
        presentation_hashes=presentation_hashes,
        matched_config=matched_config,
    )
    if not contract["science_result_eligible"]:
        raise RuntimeError(
            f"ablation run did not close its contract: {contract['status']}"
        )
    return contract


def main(argv: Sequence[str] | None = None) -> int:
    contract = run(argv)
    print(json.dumps(contract, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
