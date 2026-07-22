#!/usr/bin/env python3
"""Generate one case from the frozen 2D shock--vortex perturbation family."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utility.time_dependent_no.shock_vortex_family import (  # noqa: E402
    REFERENCE_ARTIFACT_SCHEMA,
    config_for_family_case,
    family_case_provenance,
    load_shock_vortex_family_manifest,
)
from utility.time_dependent_no.shock_vortex_fv import (  # noqa: E402
    BOUNDARY_TAG_NAMES,
    reference_contract_checks,
    run_shock_vortex_reference,
)

BENCHMARK_SOURCE = "https://doi.org/10.1007/s10915-021-01743-1"
CLOSURE_RELATIVE_TOLERANCE = 1.0e-10
INITIAL_QUADRATURE_RELATIVE_TOLERANCE = 1.0e-10


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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


def _write_accepted_steps(path: Path, result: Any) -> None:
    fields = (
        "step_index",
        "accepted_time",
        "dt",
        "save_interval",
        "rejected_attempts_before_acceptance",
        "face_reconstruction_fallbacks_across_attempts",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(result.accepted_step_times.size):
            writer.writerow(
                {
                    "step_index": index,
                    "accepted_time": float(result.accepted_step_times[index]),
                    "dt": float(result.accepted_step_dt[index]),
                    "save_interval": int(result.accepted_step_interval[index]),
                    "rejected_attempts_before_acceptance": int(
                        result.accepted_step_rejections[index]
                    ),
                    "face_reconstruction_fallbacks_across_attempts": int(
                        result.accepted_step_face_fallbacks[index]
                    ),
                }
            )


def _write_intervals(path: Path, result: Any) -> None:
    fields = (
        "interval_index",
        "start_time",
        "end_time",
        "accepted_steps",
        "rejected_attempts",
        "face_reconstruction_fallbacks",
        "closure_max_abs_integrated_state",
        "closure_relative_l2",
        "boundary_mass_impulse",
        "boundary_x_momentum_impulse",
        "boundary_y_momentum_impulse",
        "boundary_energy_impulse",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(result.times.size - 1):
            exchange = result.interval_boundary_exchange[index]
            writer.writerow(
                {
                    "interval_index": index,
                    "start_time": float(result.times[index]),
                    "end_time": float(result.times[index + 1]),
                    "accepted_steps": int(
                        np.count_nonzero(result.accepted_step_interval == index)
                    ),
                    "rejected_attempts": int(result.interval_rejection_count[index]),
                    "face_reconstruction_fallbacks": int(
                        result.interval_face_fallback_count[index]
                    ),
                    "closure_max_abs_integrated_state": float(
                        result.interval_closure_max_abs[index]
                    ),
                    "closure_relative_l2": float(
                        result.interval_closure_relative_l2[index]
                    ),
                    "boundary_mass_impulse": float(exchange[0]),
                    "boundary_x_momentum_impulse": float(exchange[1]),
                    "boundary_y_momentum_impulse": float(exchange[2]),
                    "boundary_energy_impulse": float(exchange[3]),
                }
            )


def _save_reference_npz(
    path: Path,
    result: Any,
    metadata: dict[str, Any],
    provenance: dict[str, Any],
) -> None:
    geometry = result.geometry
    np.savez_compressed(
        path,
        schema=np.asarray(REFERENCE_ARTIFACT_SCHEMA),
        conservative_states=result.states,
        physical_times=result.times,
        physical_delta_t=np.diff(result.times),
        cumulative_accepted_substep_face_impulses=result.face_impulses,
        cell_centers=geometry.cell_centers,
        cell_volume=geometry.cell_volume,
        face_centers=geometry.face_centers,
        face_measure=geometry.face_measure,
        face_normal=geometry.face_normal,
        face_owner=geometry.face_owner,
        face_neighbor=geometry.face_neighbor,
        face_axis=geometry.face_axis,
        face_boundary_tag=geometry.face_boundary_tag,
        boundary_tag_names_json=np.asarray(json.dumps(BOUNDARY_TAG_NAMES)),
        accepted_step_times=result.accepted_step_times,
        accepted_step_dt=result.accepted_step_dt,
        accepted_step_interval=result.accepted_step_interval,
        accepted_step_rejections=result.accepted_step_rejections,
        accepted_step_face_reconstruction_fallbacks=(
            result.accepted_step_face_fallbacks
        ),
        interval_rejection_count=result.interval_rejection_count,
        interval_face_reconstruction_fallback_count=(
            result.interval_face_fallback_count
        ),
        interval_closure_max_abs=result.interval_closure_max_abs,
        interval_closure_relative_l2=result.interval_closure_relative_l2,
        interval_boundary_exchange=result.interval_boundary_exchange,
        trajectory_id=np.asarray(provenance["case_id"]),
        trajectory_index=np.asarray(provenance["trajectory_index"], dtype=np.int64),
        split=np.asarray(provenance["split"]),
        split_group_id=np.asarray(provenance["split_group_id"]),
        parameters_json=np.asarray(
            json.dumps(provenance["parameters"], sort_keys=True)
        ),
        family_contract_json=np.asarray(json.dumps(provenance, sort_keys=True)),
        coordinate_convention=np.asarray(
            "row-major cell averages; x increases right, y increases up"
        ),
        face_orientation_convention=np.asarray(
            "owner outward on boundary; owner-to-neighbor on interior faces"
        ),
        state_convention=np.asarray("[rho,rho*u,rho*v,total_energy]"),
        boundary_mode=np.asarray("linear x extrapolation; y symmetry"),
        solver_method=np.asarray(
            "dimension-by-dimension primitive WENO5-JS + HLLC + SSPRK3"
        ),
        config_json=np.asarray(json.dumps(result.config.to_dict(), sort_keys=True)),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(
            f"output directory already exists; choose a new path: {args.output_dir}"
        )
    manifest = load_shock_vortex_family_manifest(args.manifest)
    config = config_for_family_case(manifest, args.case_id)
    provenance = family_case_provenance(manifest, args.case_id)
    device = _select_device(args.device)

    args.output_dir.mkdir(parents=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = run_shock_vortex_reference(
        config,
        device=device,
        dtype=torch.float64,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    checks = reference_contract_checks(
        result,
        closure_relative_tolerance=CLOSURE_RELATIVE_TOLERANCE,
        initial_quadrature_tolerance=INITIAL_QUADRATURE_RELATIVE_TOLERANCE,
    )

    source_path = Path(__file__).resolve()
    solver_path = (ROOT / "utility/time_dependent_no/shock_vortex_fv.py").resolve()
    family_path = (ROOT / "utility/time_dependent_no/shock_vortex_family.py").resolve()
    metadata = {
        "schema": REFERENCE_ARTIFACT_SCHEMA,
        "benchmark_source": BENCHMARK_SOURCE,
        "family_contract": provenance,
        "canonical_physical_case": provenance["canonical_physical_case"],
        "canonical_ladder_member": False,
        "device": str(device),
        "dtype": "float64",
        "elapsed_seconds": elapsed,
        "config_digest": hashlib.sha256(
            json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest(),
        "entrypoint_sha256": _sha256(source_path),
        "solver_utility_sha256": _sha256(solver_path),
        "family_utility_sha256": _sha256(family_path),
        "manifest_file_sha256": _sha256(args.manifest),
        "future_reference_boundary_values": False,
        "clipping_or_accepted_state_floors": False,
        "reference_truth_source": (
            "high-fidelity numerical finite-volume evolution from the declared "
            "shock-vortex family initial condition"
        ),
        "initial_state_contract": (
            "conservative cell averages; shock-crossing cells split exactly; "
            "full fine-grid tensor Gauss-Legendre quadrature certified at doubled order"
        ),
        "initial_quadrature_scope": (
            "full fine-grid conservative cell averages compared before restriction"
        ),
        "initial_quadrature_max_abs": result.initial_quadrature_max_abs,
        "initial_quadrature_relative_l2": result.initial_quadrature_relative_l2,
        "restricted_initial_quadrature_max_abs": (
            result.restricted_initial_quadrature_max_abs
        ),
        "restricted_initial_quadrature_relative_l2": (
            result.restricted_initial_quadrature_relative_l2
        ),
    }
    artifact_path = args.output_dir / "reference.npz"
    _save_reference_npz(artifact_path, result, metadata, provenance)
    _write_accepted_steps(args.output_dir / "accepted_steps.csv", result)
    _write_intervals(args.output_dir / "intervals.csv", result)

    summary = {
        **metadata,
        "status": "passed" if all(checks.values()) else "failed",
        "contract_checks": checks,
        "closure_relative_tolerance": CLOSURE_RELATIVE_TOLERANCE,
        "initial_quadrature_relative_tolerance": (
            INITIAL_QUADRATURE_RELATIVE_TOLERANCE
        ),
        "reference_artifact": artifact_path.name,
        "reference_artifact_sha256": _sha256(artifact_path),
        "config": config.to_dict(),
        "arrays": {
            "conservative_states": list(result.states.shape),
            "cumulative_accepted_substep_face_impulses": list(
                result.face_impulses.shape
            ),
            "cell_centers": list(result.geometry.cell_centers.shape),
            "face_centers": list(result.geometry.face_centers.shape),
        },
        "accepted_steps": int(result.accepted_step_times.size),
        "rejected_attempts": int(np.sum(result.interval_rejection_count)),
        "face_reconstruction_fallbacks": int(
            np.sum(result.interval_face_fallback_count)
        ),
        "minimum_density": result.minimum_density,
        "minimum_pressure": result.minimum_pressure,
        "minimum_accepted_dt": float(np.min(result.accepted_step_dt)),
        "maximum_accepted_dt": float(np.max(result.accepted_step_dt)),
        "maximum_interval_closure_relative_l2": float(
            np.max(result.interval_closure_relative_l2)
        ),
        "claim_boundary": {
            "verified_if_passed": (
                "one family trajectory with physical Cartesian geometry, exact "
                "trajectory/split provenance, and accepted-substep face impulses"
            ),
            "still_required": (
                "the declared cohort audit before using the family as neural training data"
            ),
            "unsupported": (
                "neural baseline quality, correction headroom, learned stabilization, "
                "or completion of any other family case"
            ),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    if not all(checks.values()):
        raise RuntimeError("reference artifact failed one or more contract checks")


if __name__ == "__main__":
    main()
