#!/usr/bin/env python3
"""Run the maintained error decomposer with the exact frozen L3R-B1 runtime.

The maintained decomposer's metrics and rollout loop are left unchanged.  This
adapter binds only checkpoint loading/model construction, the causal contract
forward, and device selection to the byte-exact B1 evaluator/trainer.  It also
replaces the maintained source list with the files that actually executed and
adds a fail-closed compatibility record to ``summary.json``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no.train_pcno_bump_gradient_ablation import (
    B1_FROZEN_SOURCE_SHA256,
    B1_SOURCE_SET_SHA256,
    extension_source_hashes,
    mapping_sha256,
    sha256_file,
    verify_frozen_b1_sources,
    write_json,
)

SCHEMA = "w26_l2_b1_frozen_decomposition_adapter_v1"
SUMMARY_KEY = "w26_l2_frozen_b1_compatibility"
DECOMPOSER_PATH = "scripts/time_dependent_no/decompose_pcno_euler2d_rollout_error.py"
ADAPTER_PATH = "scripts/time_dependent_no/decompose_pcno_b1_frozen_rollout_error.py"
EXTRA_SOURCE_PATHS = (
    ADAPTER_PATH,
    DECOMPOSER_PATH,
    "scripts/time_dependent_no/train_pcno_bump_gradient_ablation.py",
    "utility/time_dependent_no/pcno_artifacts.py",
    "utility/time_dependent_no/euler2d_metrics.py",
    "utility/time_dependent_no/pcno_ripple_diagnostics.py",
)


def compatibility_modules(
    evaluator: ModuleType, trainer: ModuleType
) -> dict[str, ModuleType]:
    """Build the two import shims required by the maintained decomposer."""

    rollout = ModuleType("utility.time_dependent_no.pcno_rollout")
    rollout.CAUSAL_BOUNDARY_MODE = evaluator.CAUSAL_BOUNDARY_MODE
    rollout.build_bump_checkpoint_model = evaluator.build_model
    rollout.load_bump_checkpoint = evaluator.load_checkpoint
    rollout.contract_forward_sample = trainer.contract_forward_sample

    runtime = ModuleType("utility.time_dependent_no.pcno_runtime")
    runtime.select_device = evaluator.select_device
    return {rollout.__name__: rollout, runtime.__name__: runtime}


@contextmanager
def installed_compatibility_modules(
    modules: Mapping[str, ModuleType],
):
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def decomposition_source_hashes(root: Path = ROOT) -> dict[str, str]:
    """Hash the exact frozen base plus every non-base decomposition source."""

    paths = tuple(B1_FROZEN_SOURCE_SHA256) + EXTRA_SOURCE_PATHS
    missing = [relative for relative in paths if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"decomposition source files are missing: {missing}")
    return {relative: sha256_file(root / relative) for relative in paths}


def compatibility_contract(
    *,
    frozen_sources: Mapping[str, str],
    extension_sources: Mapping[str, str],
    decomposition_sources: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "contract_complete": (
            dict(frozen_sources) == B1_FROZEN_SOURCE_SHA256
            and ADAPTER_PATH in decomposition_sources
            and DECOMPOSER_PATH in decomposition_sources
        ),
        "base_source_set_sha256": B1_SOURCE_SET_SHA256,
        "base_source_sha256": dict(frozen_sources),
        "extension_source_sha256": dict(extension_sources),
        "decomposition_source_sha256": dict(decomposition_sources),
        "decomposition_source_set_sha256": mapping_sha256(decomposition_sources),
        "bindings": {
            "checkpoint_loader": (
                "scripts.time_dependent_no.evaluate_pcno_euler2d_residual."
                "load_checkpoint"
            ),
            "model_builder": (
                "scripts.time_dependent_no.evaluate_pcno_euler2d_residual.build_model"
            ),
            "device_selector": (
                "scripts.time_dependent_no.evaluate_pcno_euler2d_residual.select_device"
            ),
            "contract_forward": (
                "scripts.time_dependent_no.train_pcno_euler2d_residual."
                "contract_forward_sample"
            ),
        },
        "maintained_decomposition_math_unchanged": True,
    }


def load_maintained_decomposer() -> ModuleType:
    """Load a fresh decomposer module while the exact-runtime shims are active."""

    from scripts.time_dependent_no import evaluate_pcno_euler2d_residual as evaluator
    from scripts.time_dependent_no import train_pcno_euler2d_residual as trainer

    path = ROOT / DECOMPOSER_PATH
    spec = importlib.util.spec_from_file_location(
        "_w26_l2_b1_frozen_rollout_decomposer", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load maintained decomposer: {path}")
    module = importlib.util.module_from_spec(spec)
    shims = compatibility_modules(evaluator, trainer)
    with installed_compatibility_modules(shims):
        spec.loader.exec_module(module)
    module._source_hashes = decomposition_source_hashes
    return module


def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    frozen_sources = verify_frozen_b1_sources()
    extension_sources = extension_source_hashes()
    decomposition_sources = decomposition_source_hashes()
    contract = compatibility_contract(
        frozen_sources=frozen_sources,
        extension_sources=extension_sources,
        decomposition_sources=decomposition_sources,
    )
    if not contract["contract_complete"]:
        raise RuntimeError("frozen-B1 decomposition compatibility contract failed")

    decomposer = load_maintained_decomposer()
    args: argparse.Namespace = decomposer.parse_args(argv)
    decomposer.main(argv)
    summary_path = Path(args.output_dir) / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("contract_complete") is not True:
        raise RuntimeError("maintained decomposition contract is incomplete")
    if summary.get("source_files") != decomposition_sources:
        raise RuntimeError("maintained decomposition source record was not replaced")
    summary[SUMMARY_KEY] = contract
    write_json(summary_path, summary)
    return contract


def main(argv: Sequence[str] | None = None) -> int:
    contract = run(argv)
    print(
        json.dumps(
            {
                "status": "complete",
                "adapter_contract_complete": contract["contract_complete"],
                "decomposition_source_set_sha256": contract[
                    "decomposition_source_set_sha256"
                ],
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
