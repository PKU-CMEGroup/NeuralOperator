#!/usr/bin/env python3
"""Close the immutable D094 B1-C3-R1 seed-0 replay matrix.

The original launcher completed all twelve training cells but rejected its own
matrix receipt because it required whole-model initialization hashes to remain
equal across trajectory counts.  Those hashes include data-dependent
normalization buffers.  This audit preserves the failed attempt, reconstructs
each recorded initial state exactly, verifies parameter-only initialization
separately, and writes a corrected hash-bound receipt in a fresh directory.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utility.time_dependent_no.pcno_artifacts import (
    atomic_write_json,
    sha256_file,
    write_source_snapshot,
)
from utility.time_dependent_no.pcno_differential_branch import (
    _is_differential_state_name,
    model_state_sha256,
)
from utility.time_dependent_no.pcno_euler2d import (
    Euler2DNormalization,
    PCNOEuler2DResidual,
)

SCHEMA = "d094_b1_c3_r1_seed0_replay_matrix_receipt_v2"
SUMMARY_SCHEMA = "d094_b1_c3_r1_closeout_audit_v1"
ARTIFACT_SCHEMA = "d094_b1_c3_r1_closeout_artifacts_v1"
ATTEMPT = "d094_b1_c3_r1_seed0_replay_20260825a"
SOURCE_COMMIT = "0da7ad5a66be7bf9ff940ffca777c355bc5341db"
SOURCE_ARCHIVE_SHA256 = (
    "7198ff6c865bb312ca53cf30d54e0c986759ff6151adf3e000d07b52e565e04b"
)
DATA_MANIFEST_SHA256 = (
    "5d5373fdcc682544bf330fba6d54fe65509dabe936baee7443c71a8c0c8d9fa7"
)
SPLIT_MANIFEST_SHA256 = (
    "feb404e295c104c2ae9e66d25bbb809737bdd5d7febb0094a9d4aa3146191ccc"
)
SEED = 20_260_718
TRAJECTORY_COUNTS = (8, 16, 32, 64, 128, 256)
ARCHITECTURE_MODES = (("pcno", "full"), ("pcfno", "no_gradient"))
OPTIMIZER_STEPS = 20_480
STEPS_PER_EPOCH = 256
PRESENTATIONS_PER_TRAJECTORY = 64
REGISTERED_STAGE = "b1_c3_r1_seed0_replay_20480"
SCHEDULE_ARM = "b1_c3_r1_seed0_replay_stretched"

CELL_FILES = (
    "best.pt",
    "last.pt",
    "summary.json",
    "metrics.jsonl",
    "bump_scaling_contract.json",
    "bump_scaling_preflight.json",
    "d094_metric_receipt.json",
    "boundary_contract.json",
    "exposure_contract.json",
    "normalization.json",
    "run_contract.json",
    "split.json",
)

EXTRA_SOURCE_FILES = (
    "docs/time_dependent_no/D094_BUMP_SCALING_PREREGISTRATION.md",
    "docs/time_dependent_no/D094_BUMP_SCALING_SPLIT_MANIFEST.json",
    "scripts/time_dependent_no/closeout_pcno_bump_b1_c3_r1.py",
    "scripts/time_dependent_no/train_pcno_bump_scaling.py",
    "utility/time_dependent_no/pcno_differential_branch.py",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path.name}")
    return value


def _require_regular_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required file is missing or not regular: {path.name}")


def _assert_finite(value: Any) -> None:
    if isinstance(value, Mapping):
        for child in value.values():
            _assert_finite(child)
    elif isinstance(value, list):
        for child in value:
            _assert_finite(child)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("training metrics contain a nonfinite value")


def _cell_name(trajectory_count: int, architecture: str) -> str:
    return (
        f"b1_c3_r1_s{SEED}_n{trajectory_count}_{architecture}_"
        "stretched_20480_20260825a"
    )


def _build_initial_model(
    summary: Mapping[str, Any], normalization: Mapping[str, Any]
) -> PCNOEuler2DResidual:
    config = summary.get("model_config")
    if not isinstance(config, Mapping) or config.get("model") != "PCNOEuler2DResidual":
        raise ValueError("cell model configuration changed")
    if config.get("in_dim") != 12 or config.get("out_dim") != 4:
        raise ValueError("cell model input/output contract changed")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    return PCNOEuler2DResidual(
        normalization=Euler2DNormalization.from_mapping(normalization),
        k_max=int(config["k_max"]),
        domain_lengths=config["domain_lengths"],
        layers=config["layers"],
        fc_dim=int(config["fc_dim"]),
        nmeasures=int(config["nmeasures"]),
        node_type_feature_mode=str(config["node_type_feature_mode"]),
        boundary_field_mode=str(config["boundary_field_mode"]),
        boundary_field_names=config["boundary_field_names"],
        boundary_residual_mode=str(config["boundary_residual_mode"]),
        boundary_residual_names=config["boundary_residual_names"],
        boundary_residual_width=int(config["boundary_residual_width"]),
        zero_initialize=True,
    )


def _reconstruct_initialization(
    summary: Mapping[str, Any], normalization: Mapping[str, Any]
) -> dict[str, str]:
    model = _build_initial_model(summary, normalization)
    parameter_names = set(dict(model.named_parameters()))
    return {
        "full_state_sha256": model_state_sha256(model),
        "parameter_state_sha256": model_state_sha256(
            model, include=lambda name: name in parameter_names
        ),
        "nondifferential_parameter_state_sha256": model_state_sha256(
            model,
            include=lambda name: (
                name in parameter_names and not _is_differential_state_name(name)
            ),
        ),
    }


def _validate_cell(
    cell_root: Path,
    *,
    trajectory_count: int,
    architecture: str,
    mode: str,
) -> dict[str, Any]:
    summary = _load_json(cell_root / "summary.json")
    contract = _load_json(cell_root / "bump_scaling_contract.json")
    run_contract = _load_json(cell_root / "run_contract.json")
    normalization = _load_json(cell_root / "normalization.json")
    expected_contract = {
        "status": "completed_seed0_exact_exposure_replay_cell",
        "science_result_eligible": False,
        "historical_test_population_accessed": False,
        "trajectory_count": trajectory_count,
        "registered_stage": REGISTERED_STAGE,
        "initialization_seed": SEED,
        "differential_branch_mode": mode,
        "actual_optimizer_steps": OPTIMIZER_STEPS,
        "completed_epochs": OPTIMIZER_STEPS // STEPS_PER_EPOCH,
        "schedule_arm": SCHEDULE_ARM,
        "sentinel_steps": [PRESENTATIONS_PER_TRAJECTORY * trajectory_count],
        "sentinel_payload": "model_only",
        "selected_terminal_checkpoint_payload": "model_only_evaluation",
        "automatic_continuation_authorized": False,
    }
    for key, expected in expected_contract.items():
        if contract.get(key) != expected:
            raise ValueError(f"{cell_root.name} contract mismatch: {key}")
    if (
        summary.get("actual_optimizer_steps") != OPTIMIZER_STEPS
        or summary.get("completed_epochs") != OPTIMIZER_STEPS // STEPS_PER_EPOCH
        or summary.get("test_trajectories") not in (0, [])
    ):
        raise ValueError(f"{cell_root.name} summary contract changed")
    run_args = run_contract.get("args")
    if not isinstance(run_args, Mapping):
        raise TypeError(f"{cell_root.name} run arguments are malformed")
    if run_args.get("init_checkpoint") is not None or run_args.get(
        "resume_checkpoint"
    ) is not None:
        raise ValueError(f"{cell_root.name} was not a cold start")

    branch = summary.get("initialization_control", {}).get("differential_branch")
    if not isinstance(branch, Mapping):
        raise TypeError(f"{cell_root.name} lacks its branch initialization contract")
    if (
        branch.get("mode") != mode
        or branch.get("state_dict_keys_unchanged") is not True
        or not isinstance(branch.get("initial_full_state_sha256"), str)
        or not isinstance(branch.get("initial_nondifferential_state_sha256"), str)
    ):
        raise ValueError(f"{cell_root.name} branch initialization contract changed")

    metric_path = cell_root / "metrics.jsonl"
    _require_regular_file(metric_path)
    metric_rows = [
        json.loads(line)
        for line in metric_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not metric_rows:
        raise ValueError(f"{cell_root.name} has no metric rows")
    _assert_finite(metric_rows)

    sentinel_step = PRESENTATIONS_PER_TRAJECTORY * trajectory_count
    relative_files = (*CELL_FILES, f"sentinels/step_{sentinel_step:09d}.pt")
    files: dict[str, dict[str, Any]] = {}
    for relative in relative_files:
        path = cell_root / relative
        _require_regular_file(path)
        files[relative] = {
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }

    reconstruction = _reconstruct_initialization(summary, normalization)
    if reconstruction["full_state_sha256"] != branch["initial_full_state_sha256"]:
        raise ValueError(f"{cell_root.name} initial state cannot be reconstructed")
    return {
        "architecture": architecture,
        "cell": cell_root.name,
        "config_digest": summary["config_digest"],
        "initial_full_state_sha256": branch["initial_full_state_sha256"],
        "initial_nondifferential_state_sha256": branch[
            "initial_nondifferential_state_sha256"
        ],
        "reconstructed_parameter_state_sha256": reconstruction[
            "parameter_state_sha256"
        ],
        "reconstructed_nondifferential_parameter_state_sha256": reconstruction[
            "nondifferential_parameter_state_sha256"
        ],
        "normalization_digest": summary["normalization_digest"],
        "optimizer_steps": contract["actual_optimizer_steps"],
        "presentation_stream_sha256_by_epoch": contract[
            "presentation_stream_sha256_by_epoch"
        ],
        "sentinel_step": sentinel_step,
        "trajectory_count": trajectory_count,
        "files": files,
    }


def build_matrix_receipt(attempt_root: Path) -> dict[str, Any]:
    if attempt_root.is_symlink() or not attempt_root.is_dir():
        raise ValueError("attempt root must be a regular directory")
    source_archive = attempt_root / "source_commit.tar"
    _require_regular_file(source_archive)
    if sha256_file(source_archive) != SOURCE_ARCHIVE_SHA256:
        raise ValueError("original source archive hash changed")

    result_root = attempt_root / "results"
    launch_root = attempt_root / "launch"
    if not result_root.is_dir() or not launch_root.is_dir():
        raise ValueError("attempt lacks its result or launch directory")
    if (launch_root / "matrix_receipt.json").exists():
        raise ValueError("the original attempt unexpectedly has a matrix receipt")
    if (launch_root / "launch.completed").exists():
        raise ValueError("the original attempt unexpectedly has a completion marker")

    cells: list[dict[str, Any]] = []
    full_hashes: set[str] = set()
    parameter_hashes: set[str] = set()
    nondifferential_parameter_hashes: set[str] = set()
    for cell_index, trajectory_count in enumerate(TRAJECTORY_COUNTS, start=1):
        paired: dict[str, dict[str, Any]] = {}
        for architecture, mode in ARCHITECTURE_MODES:
            cell_root = result_root / _cell_name(trajectory_count, architecture)
            cell = _validate_cell(
                cell_root,
                trajectory_count=trajectory_count,
                architecture=architecture,
                mode=mode,
            )
            cells.append(cell)
            paired[architecture] = cell

        for key in (
            "initial_full_state_sha256",
            "initial_nondifferential_state_sha256",
            "normalization_digest",
            "presentation_stream_sha256_by_epoch",
        ):
            if paired["pcno"][key] != paired["pcfno"][key]:
                raise ValueError(
                    f"paired PCNO/PCFNO contract mismatch for n={trajectory_count}: {key}"
                )
        marker_indices = (2 * cell_index - 1, 2 * cell_index)
        for marker_index in marker_indices:
            _require_regular_file(launch_root / f"cell_{marker_index}.completed")

        full_hashes.add(paired["pcno"]["initial_full_state_sha256"])
        parameter_hashes.add(
            paired["pcno"]["reconstructed_parameter_state_sha256"]
        )
        nondifferential_parameter_hashes.add(
            paired["pcno"][
                "reconstructed_nondifferential_parameter_state_sha256"
            ]
        )

    if len(cells) != len(TRAJECTORY_COUNTS) * len(ARCHITECTURE_MODES):
        raise RuntimeError("matrix cell inventory does not close")
    if len(parameter_hashes) != 1 or len(nondifferential_parameter_hashes) != 1:
        raise ValueError("learned-parameter initialization changed across counts")
    if len(full_hashes) == 1:
        raise ValueError("whole-state hashes unexpectedly ignore changed normalizations")

    return {
        "schema": SCHEMA,
        "status": "complete_after_closeout_correction",
        "attempt": ATTEMPT,
        "source_commit": SOURCE_COMMIT,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "data_manifest_sha256": DATA_MANIFEST_SHA256,
        "split_manifest_sha256": SPLIT_MANIFEST_SHA256,
        "historical_test_population_accessed": False,
        "checkpoint_reselection_performed": False,
        "seed": SEED,
        "trajectory_counts": list(TRAJECTORY_COUNTS),
        "architectures": [name for name, _ in ARCHITECTURE_MODES],
        "optimizer_steps_per_cell": OPTIMIZER_STEPS,
        "presentations_per_training_trajectory_at_primary_sentinel": (
            PRESENTATIONS_PER_TRAJECTORY
        ),
        "primary_sentinel_step_by_count": {
            str(count): PRESENTATIONS_PER_TRAJECTORY * count
            for count in TRAJECTORY_COUNTS
        },
        "closeout_correction": {
            "original_launcher_terminal_error": (
                "initialization changed across trajectory counts"
            ),
            "original_attempt_outputs_modified": False,
            "preregistered_pairing_scope": "fixed_seed_and_trajectory_count",
            "whole_state_hash_includes_data_dependent_normalization_buffers": True,
            "whole_state_hash_cardinality_across_counts": len(full_hashes),
            "recorded_initial_state_exactly_reconstructed_for_every_count": True,
            "parameter_only_hash_cardinality_across_counts": len(parameter_hashes),
            "nondifferential_parameter_only_hash_cardinality_across_counts": len(
                nondifferential_parameter_hashes
            ),
            "shared_parameter_state_sha256": next(iter(parameter_hashes)),
            "shared_nondifferential_parameter_state_sha256": next(
                iter(nondifferential_parameter_hashes)
            ),
        },
        "cells": cells,
    }


def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    receipt = build_matrix_receipt(args.attempt_root)
    atomic_write_json(args.output_dir / "matrix_receipt.json", receipt)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "complete",
        "attempt": ATTEMPT,
        "matrix_receipt_schema": SCHEMA,
        "matrix_receipt_sha256": sha256_file(args.output_dir / "matrix_receipt.json"),
        "cell_count": len(receipt["cells"]),
        "historical_test_population_accessed": False,
        "checkpoint_reselection_performed": False,
        "original_attempt_outputs_modified": False,
        "closeout_correction": receipt["closeout_correction"],
    }
    atomic_write_json(args.output_dir / "summary.json", summary)
    source_snapshot = write_source_snapshot(
        args.output_dir, extra_source_files=EXTRA_SOURCE_FILES
    )
    artifact_files = (
        "matrix_receipt.json",
        "summary.json",
        "source_snapshot/manifest.json",
    )
    artifact_manifest = {
        "schema": ARTIFACT_SCHEMA,
        "historical_test_population_accessed": False,
        "source_set_digest": source_snapshot["source_set_digest"],
        "files": {
            name: {
                "bytes": int((args.output_dir / name).stat().st_size),
                "sha256": sha256_file(args.output_dir / name),
            }
            for name in artifact_files
        },
    }
    atomic_write_json(args.output_dir / "artifact_manifest.json", artifact_manifest)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    summary = run(argv)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "cell_count": summary["cell_count"],
                "historical_test_population_accessed": False,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
