#!/usr/bin/env python3
"""Retain a finer restriction of one frozen shock--vortex family evolution."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utility.time_dependent_no.pcno_resolution_transfer import (  # noqa: E402
    MULTIRES_REFERENCE_SCHEMA,
    parse_resolution,
    restrict_nested_state,
)
from utility.time_dependent_no.shock_vortex_family import (  # noqa: E402
    REFERENCE_ARTIFACT_SCHEMA,
    config_for_family_case,
    family_case_provenance,
    load_shock_vortex_family_manifest,
)
from utility.time_dependent_no.shock_vortex_fv import (  # noqa: E402
    reference_contract_checks,
    run_shock_vortex_reference,
)

CLOSURE_RELATIVE_TOLERANCE = 1.0e-10
INITIAL_QUADRATURE_RELATIVE_TOLERANCE = 1.0e-10
RESTRICTION_CROSSCHECK_ABS_TOLERANCE = 1.0e-12


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--family-root", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--retained-resolution", default="500x200")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def _select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_frozen_reference(
    family_root: Path,
    case_id: str,
    *,
    expected_provenance: Mapping[str, Any],
    expected_config: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], str]:
    case_root = family_root / case_id
    summary_path = case_root / "summary.json"
    artifact_path = case_root / "reference.npz"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    artifact_sha256 = _sha256(artifact_path)
    if summary.get("status") != "passed":
        raise ValueError(f"frozen family reference did not pass: {case_id}")
    if summary.get("reference_artifact_sha256") != artifact_sha256:
        raise ValueError(f"frozen family reference digest mismatch: {case_id}")
    names = (
        "schema",
        "family_contract_json",
        "config_json",
        "conservative_states",
        "physical_times",
        "interval_boundary_exchange",
    )
    with np.load(artifact_path, allow_pickle=False) as artifact:
        missing = sorted(set(names) - set(artifact.files))
        if missing:
            raise ValueError(f"frozen reference is missing arrays: {missing}")
        if artifact["schema"].item() != REFERENCE_ARTIFACT_SCHEMA:
            raise ValueError("frozen reference schema mismatch")
        if json.loads(artifact["family_contract_json"].item()) != expected_provenance:
            raise ValueError("frozen reference family provenance mismatch")
        if json.loads(artifact["config_json"].item()) != expected_config:
            raise ValueError("frozen reference solver configuration mismatch")
        arrays = {
            name: np.array(artifact[name], copy=True)
            for name in (
                "conservative_states",
                "physical_times",
                "interval_boundary_exchange",
            )
        }
    return arrays, artifact_sha256


def _atomic_save_npz(path: Path, **arrays: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(
            f"output directory already exists; choose a new path: {args.output_dir}"
        )
    manifest = load_shock_vortex_family_manifest(args.manifest)
    base_config = config_for_family_case(manifest, args.case_id)
    provenance = family_case_provenance(manifest, args.case_id)
    retained_resolution = parse_resolution(args.retained_resolution)
    source_resolution = (int(base_config.nx), int(base_config.ny))
    training_resolution = (int(base_config.coarse_nx), int(base_config.coarse_ny))
    for resolution, name in (
        (retained_resolution, "retained"),
        (training_resolution, "training"),
    ):
        if source_resolution[0] % resolution[0] or source_resolution[1] % resolution[1]:
            raise ValueError(f"{name} grid must exactly divide the evolution grid")
    if (
        retained_resolution[0] % training_resolution[0]
        or retained_resolution[1] % training_resolution[1]
    ):
        raise ValueError("retained grid must exactly restrict to the training grid")

    frozen, frozen_sha256 = _load_frozen_reference(
        args.family_root,
        args.case_id,
        expected_provenance=provenance,
        expected_config=base_config.to_dict(),
    )
    retained_config = replace(
        base_config,
        coarse_nx=retained_resolution[0],
        coarse_ny=retained_resolution[1],
    ).validated()
    device = _select_device(args.device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = perf_counter()
    result = run_shock_vortex_reference(
        retained_config,
        device=device,
        dtype=torch.float64,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = perf_counter() - started

    solver_checks = reference_contract_checks(
        result,
        closure_relative_tolerance=CLOSURE_RELATIVE_TOLERANCE,
        initial_quadrature_tolerance=INITIAL_QUADRATURE_RELATIVE_TOLERANCE,
    )
    restricted_states = restrict_nested_state(
        result.states,
        fine_resolution=retained_resolution,
        coarse_resolution=training_resolution,
    )
    if restricted_states.shape != frozen["conservative_states"].shape:
        raise ValueError("retained-to-training restriction has the wrong state shape")
    if not np.array_equal(result.times, frozen["physical_times"]):
        raise ValueError("retained and frozen references have different save times")
    state_delta = restricted_states - frozen["conservative_states"]
    state_max_abs = float(np.max(np.abs(state_delta)))
    state_relative_l2 = float(
        np.linalg.norm(state_delta.reshape(-1))
        / max(np.linalg.norm(frozen["conservative_states"].reshape(-1)), 1.0e-300)
    )
    exchange_delta = (
        result.interval_boundary_exchange - frozen["interval_boundary_exchange"]
    )
    exchange_max_abs = float(np.max(np.abs(exchange_delta)))
    crosscheck_passed = (
        math.isfinite(state_max_abs)
        and state_max_abs <= RESTRICTION_CROSSCHECK_ABS_TOLERANCE
    )
    status = "passed" if all(solver_checks.values()) and crosscheck_passed else "failed"

    args.output_dir.mkdir(parents=True)
    artifact_path = args.output_dir / "reference.npz"
    metadata = {
        "schema": MULTIRES_REFERENCE_SCHEMA,
        "case_id": args.case_id,
        "family_contract": provenance,
        "source_resolution": list(source_resolution),
        "retained_resolution": list(retained_resolution),
        "training_resolution": list(training_resolution),
        "reference_construction": (
            "one common high-fidelity evolution retained at the declared fine model "
            "grid; all coarser targets are exact conservative block restrictions"
        ),
        "device": str(device),
        "dtype": "float64",
        "elapsed_seconds": elapsed_seconds,
        "frozen_training_reference_sha256": frozen_sha256,
        "entrypoint_sha256": _sha256(Path(__file__).resolve()),
        "solver_utility_sha256": _sha256(
            ROOT / "utility/time_dependent_no/shock_vortex_fv.py"
        ),
        "family_utility_sha256": _sha256(
            ROOT / "utility/time_dependent_no/shock_vortex_family.py"
        ),
        "translation_utility_sha256": _sha256(
            ROOT / "utility/time_dependent_no/pcno_resolution_transfer.py"
        ),
    }
    _atomic_save_npz(
        artifact_path,
        schema=np.asarray(MULTIRES_REFERENCE_SCHEMA),
        conservative_states=result.states,
        physical_times=result.times,
        interval_boundary_exchange=result.interval_boundary_exchange,
        source_resolution=np.asarray(source_resolution, dtype=np.int64),
        retained_resolution=np.asarray(retained_resolution, dtype=np.int64),
        training_resolution=np.asarray(training_resolution, dtype=np.int64),
        family_contract_json=np.asarray(json.dumps(provenance, sort_keys=True)),
        config_json=np.asarray(json.dumps(retained_config.to_dict(), sort_keys=True)),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        frozen_training_reference_sha256=np.asarray(frozen_sha256),
    )
    summary = {
        **metadata,
        "status": status,
        "solver_contract_checks": solver_checks,
        "restriction_crosscheck": {
            "passed": crosscheck_passed,
            "absolute_tolerance": RESTRICTION_CROSSCHECK_ABS_TOLERANCE,
            "state_max_abs": state_max_abs,
            "state_relative_l2": state_relative_l2,
            "interval_boundary_exchange_max_abs": exchange_max_abs,
        },
        "config": retained_config.to_dict(),
        "reference_artifact": artifact_path.name,
        "reference_artifact_sha256": _sha256(artifact_path),
        "artifact_bytes": artifact_path.stat().st_size,
        "arrays": {
            "conservative_states": list(result.states.shape),
            "physical_times": list(result.times.shape),
            "interval_boundary_exchange": list(result.interval_boundary_exchange.shape),
        },
        "claim_boundary": (
            "This artifact supplies restriction-consistent targets for one named "
            "physical case. It is not an independently evolved native-grid solver map."
        ),
    }
    _atomic_write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), flush=True)
    if status != "passed":
        raise RuntimeError("multiresolution reference failed its contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
