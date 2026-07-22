#!/usr/bin/env python3
"""Run a frozen canonical divergence-active target preflight.

This is a zero-training diagnostic on the validated dynamic finite-volume
family. It never accesses the strength-OOD test split. The accepted reference
boundary impulse is retained, while the interior reference field is projected
onto the minimum-W_f^-1-norm representative with the same divergence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utility.time_dependent_no.fv_impulse_diagnostics import (  # noqa: E402
    CLAIM_BOUNDARY,
    DirectMinimumNormProjector,
    build_fv_impulse_operators,
    decompose_interior_impulse_error,
    factorize_direct_minimum_winv_norm_projector,
    minimum_winv_norm_face_impulse,
)
from utility.time_dependent_no.pcno_euler2d import (  # noqa: E402
    PCNOEuler2DShardStore,
)
from utility.time_dependent_no.pcno_face_impulse import (  # noqa: E402
    load_fixed_fv_face_geometry,
)
from utility.time_dependent_no.shock_vortex_family import (  # noqa: E402
    REFERENCE_ARTIFACT_SCHEMA,
    family_case_provenance,
    load_shock_vortex_family_manifest,
)

SCHEMAS = {
    "lsmr": "fv_divergence_active_target_preflight_v1",
    "direct": "fv_divergence_active_target_preflight_v2",
}
EXPERIMENT_IDS = {"lsmr": "D046", "direct": "D047"}
SOURCE_GEOMETRY_KEY = "sv_e07_y04"
TRAIN_KEYS = ("sv_e00_y04", "sv_e03_y04", "sv_e07_y04", "sv_e11_y04")
VALIDATION_KEYS = ("sv_e00_y00", "sv_e03_y08", "sv_e07_y00", "sv_e11_y08")
CALLS = (1, 30, 60)
LSMR_TOLERANCE = 1.0e-11
LSMR_MAX_ITERATIONS = 25_000
ALLOWED_LSMR_STOP_CODES = frozenset({1, 2})
GATE_THRESHOLDS = {
    "reference_closure_relative_l2_max": 1.0e-10,
    "canonical_reference_closure_relative_l2_max": 1.0e-8,
    "canonical_shard_increment_closure_relative_l2_max": 1.0e-4,
    "independent_canonical_field_relative_l2_max": 1.0e-5,
    "compatibility_relative_l2_max": 1.0e-10,
    "cycle_divergence_relative_l2_max": 1.0e-8,
    "canonical_full_winv_norm_ratio_max": 1.000001,
    "wall_forbidden_exchange_absolute_max": 1.0e-14,
}
GATE_ROW_METRICS = {
    "reference_closure_relative_l2_max": "reference_closure_relative_l2",
    "canonical_reference_closure_relative_l2_max": (
        "canonical_reference_closure_relative_l2"
    ),
    "canonical_shard_increment_closure_relative_l2_max": (
        "canonical_shard_increment_closure_relative_l2"
    ),
    "independent_canonical_field_relative_l2_max": (
        "independent_canonical_field_relative_l2"
    ),
    "compatibility_relative_l2_max": "compatibility_relative_l2",
    "cycle_divergence_relative_l2_max": "cycle_divergence_relative_l2",
    "canonical_full_winv_norm_ratio_max": "canonical_full_winv_norm_ratio",
    "wall_forbidden_exchange_absolute_max": ("wall_forbidden_exchange_absolute"),
}
DIRECT_GATE_THRESHOLDS = {
    "direct_compatibility_projection_relative_l2_max": 1.0e-10,
    "direct_reduced_solve_residual_relative_l2_max": 1.0e-10,
}
DIRECT_GATE_ROW_METRICS = {
    "direct_compatibility_projection_relative_l2_max": (
        "compatibility_projection_relative_l2"
    ),
    "direct_reduced_solve_residual_relative_l2_max": (
        "reduced_solve_residual_relative_l2"
    ),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--canonical-solver",
        choices=tuple(SCHEMAS),
        default="lsmr",
        help=(
            "Use the historical D046 iterative construction or the separately "
            "registered D047 fixed-mesh direct factorization."
        ),
    )
    return parser.parse_args(argv)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest_mapping(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _relative_l2(actual: np.ndarray, expected: np.ndarray) -> float:
    numerator = float(np.linalg.norm(np.asarray(actual) - np.asarray(expected)))
    denominator = float(np.linalg.norm(expected))
    if denominator > 0.0:
        return numerator / denominator
    return 0.0 if numerator == 0.0 else float("inf")


def _winv_norm(field: np.ndarray, face_weight: np.ndarray) -> float:
    values = np.asarray(field, dtype=np.float64)
    return float(np.sqrt(np.sum(values * values / face_weight[:, None])))


def _load_reference_case(
    *,
    family_root: Path,
    family_manifest: Mapping[str, Any],
    store: PCNOEuler2DShardStore,
    geometry,
    key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    artifact_path = family_root / key / "reference.npz"
    if not artifact_path.is_file():
        raise FileNotFoundError(artifact_path)
    artifact_sha256 = _sha256_file(artifact_path)
    if artifact_sha256 != str(store.entry(key).get("source_reference_sha256")):
        raise ValueError(f"source reference digest mismatch for {key}")
    required = {
        "schema",
        "physical_times",
        "conservative_states",
        "cumulative_accepted_substep_face_impulses",
        "cell_centers",
        "cell_volume",
        "face_centers",
        "face_measure",
        "face_normal",
        "face_owner",
        "face_neighbor",
        "face_boundary_tag",
        "boundary_tag_names_json",
        "family_contract_json",
    }
    with np.load(artifact_path, allow_pickle=False) as artifact:
        missing = sorted(required - set(artifact.files))
        if missing:
            raise ValueError(f"reference {key} is missing arrays: {missing}")
        if str(artifact["schema"].item()) != REFERENCE_ARTIFACT_SCHEMA:
            raise ValueError(f"reference schema mismatch for {key}")
        times = np.array(artifact["physical_times"], dtype=np.float64)
        states = np.array(artifact["conservative_states"], dtype=np.float64)
        impulses = np.array(
            artifact["cumulative_accepted_substep_face_impulses"],
            dtype=np.float64,
        )
        provenance = json.loads(str(artifact["family_contract_json"].item()))
        boundary_names = tuple(
            json.loads(str(artifact["boundary_tag_names_json"].item()))
        )
        geometry_arrays = {
            "cell_centers": np.array(artifact["cell_centers"], dtype=np.float64),
            "cell_volume": np.array(artifact["cell_volume"], dtype=np.float64),
            "face_centers": np.array(artifact["face_centers"], dtype=np.float64),
            "face_measure": np.array(artifact["face_measure"], dtype=np.float64),
            "face_normal": np.array(artifact["face_normal"], dtype=np.float64),
            "face_owner": np.array(artifact["face_owner"], dtype=np.int64),
            "face_neighbor": np.array(artifact["face_neighbor"], dtype=np.int64),
            "face_boundary_tag": np.array(
                artifact["face_boundary_tag"], dtype=np.int64
            ),
        }

    expected_geometry = {
        "cell_centers": geometry.cell_center,
        "cell_volume": geometry.cell_volume,
        "face_centers": geometry.face_center,
        "face_measure": geometry.face_measure,
        "face_normal": geometry.face_normal,
        "face_owner": geometry.face_owner,
        "face_neighbor": geometry.face_neighbor,
        "face_boundary_tag": geometry.face_boundary_tag,
    }
    for name, actual in geometry_arrays.items():
        if not np.array_equal(actual, np.asarray(expected_geometry[name])):
            raise ValueError(f"reference {key} differs in fixed geometry {name}")
    if boundary_names != geometry.boundary_tag_names:
        raise ValueError(f"reference {key} boundary names differ")
    if provenance != family_case_provenance(family_manifest, key):
        raise ValueError(f"reference {key} family provenance differs")
    if not np.array_equal(
        times,
        np.asarray(store.array(key, "physical_times"), dtype=np.float64),
    ):
        raise ValueError(f"reference and shard times differ for {key}")
    expected_state_shape = (
        times.size,
        geometry.cell_volume.size,
        4,
    )
    expected_impulse_shape = (
        times.size - 1,
        geometry.face_owner.size,
        4,
    )
    if states.shape != expected_state_shape:
        raise ValueError(f"reference {key} state shape differs")
    if impulses.shape != expected_impulse_shape:
        raise ValueError(f"reference {key} impulse shape differs")
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(impulses)):
        raise ValueError(f"reference {key} contains nonfinite values")
    return times, states, impulses, artifact_sha256


def _row_metrics(
    *,
    operators,
    geometry,
    reference_states: np.ndarray,
    shard_states: np.ndarray,
    reference_impulse: np.ndarray,
    direct_projector: DirectMinimumNormProjector | None = None,
) -> dict[str, Any]:
    volume = geometry.cell_volume[:, None]
    reference_delta = reference_states[1] - reference_states[0]
    shard_delta = shard_states[1] - shard_states[0]
    target_integral = -volume * reference_delta
    shard_target_integral = -volume * shard_delta
    decoded_reference = operators.incidence @ reference_impulse
    decomposition = decompose_interior_impulse_error(
        operators,
        reference_impulse,
        tolerance=LSMR_TOLERANCE,
        max_iterations=LSMR_MAX_ITERATIONS,
    )
    projected_reference = decomposition.divergence_active + decomposition.boundary
    if direct_projector is None:
        canonical_solution = minimum_winv_norm_face_impulse(
            operators,
            target_integral,
            reference_impulse[operators.boundary_face_indices],
            tolerance=LSMR_TOLERANCE,
            max_iterations=LSMR_MAX_ITERATIONS,
        )
    else:
        if direct_projector.operators is not operators:
            raise ValueError("direct projector and row operators differ")
        canonical_solution = direct_projector.solve(
            target_integral,
            reference_impulse[operators.boundary_face_indices],
        )
    # The preregistered canonical target is defined by the cell transition and
    # accepted boundary exchange. Projection of the full reference face field
    # is the independent reproduction check, not the target definition.
    canonical = canonical_solution.face_impulse
    decoded_canonical = canonical_solution.decoded_cell_integral

    canonical_norm = _winv_norm(canonical, operators.face_weight)
    reference_norm = _winv_norm(reference_impulse, operators.face_weight)
    independent_difference = _winv_norm(
        projected_reference - canonical,
        operators.face_weight,
    )
    interior_energy = decomposition.interior_energy
    cycle_fraction = (
        decomposition.cycle_energy / interior_energy if interior_energy > 0.0 else 0.0
    )
    wall_tags = {
        geometry.boundary_tag_names.index("y_min"),
        geometry.boundary_tag_names.index("y_max"),
    }
    wall = np.isin(geometry.face_boundary_tag, list(wall_tags))
    forbidden_components = reference_impulse[wall][:, (0, 1, 3)]
    metrics = {
        "reference_closure_relative_l2": _relative_l2(
            decoded_reference, target_integral
        ),
        "canonical_reference_closure_relative_l2": _relative_l2(
            decoded_canonical, target_integral
        ),
        "canonical_shard_increment_closure_relative_l2": _relative_l2(
            decoded_canonical, shard_target_integral
        ),
        "independent_canonical_field_relative_l2": (
            independent_difference / canonical_norm if canonical_norm > 0.0 else 0.0
        ),
        "compatibility_relative_l2": (canonical_solution.compatibility_relative_l2),
        "compatibility_max_absolute": float(
            np.max(np.abs(canonical_solution.compatibility_residual), initial=0.0)
        ),
        "cycle_divergence_relative_l2": (decomposition.cycle_divergence_relative_l2),
        "decomposition_reconstruction_relative_l2": (
            decomposition.reconstruction_relative_l2
        ),
        "winv_orthogonality_relative": decomposition.winv_orthogonality_relative,
        "canonical_full_winv_norm_ratio": (
            canonical_norm / reference_norm if reference_norm > 0.0 else 0.0
        ),
        "reference_cycle_energy_fraction": float(cycle_fraction),
        "wall_forbidden_exchange_absolute": float(
            np.max(np.abs(forbidden_components), initial=0.0)
        ),
        "projection_lsmr_stop_codes": list(decomposition.lsmr_stop_codes),
        "projection_lsmr_iterations": list(decomposition.lsmr_iterations),
        "independent_lsmr_stop_codes": list(canonical_solution.lsmr_stop_codes),
        "independent_lsmr_iterations": list(canonical_solution.lsmr_iterations),
        "canonical_solver": canonical_solution.solver,
        "compatibility_projection_relative_l2": (
            canonical_solution.compatibility_projection_relative_l2
        ),
        "reduced_solve_residual_relative_l2": (
            canonical_solution.reduced_solve_residual_relative_l2
        ),
        "direct_factorization_count": canonical_solution.factorization_count,
        "direct_anchor_cells": list(canonical_solution.anchor_cells),
    }
    scalar_values = [value for value in metrics.values() if isinstance(value, float)]
    metrics["finite"] = bool(np.all(np.isfinite(scalar_values)))
    return metrics


def _write_outputs(
    output_dir: Path, summary: Mapping[str, Any], rows: list[dict[str, Any]]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "rows.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, separators=(",", ":"))
                        if isinstance(value, (list, dict))
                        else value
                    )
                    for key, value in row.items()
                }
            )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _aggregate_gate_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    gate_thresholds: Mapping[str, float] = GATE_THRESHOLDS,
    gate_row_metrics: Mapping[str, str] = GATE_ROW_METRICS,
) -> dict[str, float]:
    if not rows:
        raise ValueError("canonical-target aggregation requires at least one row")
    if set(gate_row_metrics) != set(gate_thresholds):
        raise RuntimeError("gate metric and threshold keys differ")
    return {
        aggregate_name: max(float(row[row_name]) for row in rows)
        for aggregate_name, row_name in gate_row_metrics.items()
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    experiment_id = EXPERIMENT_IDS[args.canonical_solver]
    schema = SCHEMAS[args.canonical_solver]
    gate_thresholds = dict(GATE_THRESHOLDS)
    gate_row_metrics = dict(GATE_ROW_METRICS)
    if args.canonical_solver == "direct":
        gate_thresholds.update(DIRECT_GATE_THRESHOLDS)
        gate_row_metrics.update(DIRECT_GATE_ROW_METRICS)
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    start = perf_counter()
    store = PCNOEuler2DShardStore(args.data_dir, max_cached_trajectories=2)
    shard_manifest_digest = store.manifest_digest
    family_manifest = load_shock_vortex_family_manifest(
        args.family_root / "family_manifest.json"
    )
    cases = {str(case["case_id"]): case for case in family_manifest["cases"]}
    cohort = [(key, "train") for key in TRAIN_KEYS] + [
        (key, "validation") for key in VALIDATION_KEYS
    ]
    if len({key for key, _ in cohort}) != len(cohort):
        raise RuntimeError(
            f"registered {experiment_id} cohort contains duplicate trajectories"
        )
    for key, expected_split in cohort:
        if key not in cases or key not in store.keys:
            raise ValueError(f"registered {experiment_id} case is unavailable: {key}")
        if (
            cases[key]["split"] != expected_split
            or store.entry(key).get("split") != expected_split
        ):
            raise ValueError(f"registered {experiment_id} split mismatch for {key}")
        if expected_split == "test":
            raise RuntimeError(f"{experiment_id} cannot access the test split")

    geometry = load_fixed_fv_face_geometry(
        args.family_root,
        store,
        source_key=SOURCE_GEOMETRY_KEY,
    )
    operators = build_fv_impulse_operators(
        cell_centers=geometry.cell_center,
        cell_volume=geometry.cell_volume,
        face_centers=geometry.face_center,
        face_measure=geometry.face_measure,
        face_owner=geometry.face_owner,
        face_neighbor=geometry.face_neighbor,
        face_boundary_tag=geometry.face_boundary_tag,
    )
    direct_projector = None
    if args.canonical_solver == "direct":
        direct_projector = factorize_direct_minimum_winv_norm_projector(operators)

    rows: list[dict[str, Any]] = []
    accessed_digests: dict[str, str] = {}
    try:
        for key, split in cohort:
            times, reference_states, reference_impulses, artifact_sha256 = (
                _load_reference_case(
                    family_root=args.family_root,
                    family_manifest=family_manifest,
                    store=store,
                    geometry=geometry,
                    key=key,
                )
            )
            accessed_digests[key] = artifact_sha256
            shard_states = np.asarray(store.states(key), dtype=np.float64)
            if shard_states.shape != reference_states.shape:
                raise ValueError(f"reference and shard state shapes differ for {key}")
            for call in CALLS:
                if call < 1 or call >= times.size:
                    raise ValueError(f"registered call {call} is unavailable for {key}")
                metrics = _row_metrics(
                    operators=operators,
                    geometry=geometry,
                    reference_states=reference_states[call - 1 : call + 1],
                    shard_states=shard_states[call - 1 : call + 1],
                    reference_impulse=reference_impulses[call - 1],
                    direct_projector=direct_projector,
                )
                rows.append(
                    {
                        "case_id": key,
                        "split": split,
                        "split_group_id": cases[key]["split_group_id"],
                        "vortex_epsilon": cases[key]["parameters"]["vortex_epsilon"],
                        "vortex_y": cases[key]["parameters"]["vortex_y"],
                        "call": call,
                        "physical_time": float(times[call]),
                        **metrics,
                    }
                )
    finally:
        store.close()

    aggregates = _aggregate_gate_metrics(
        rows,
        gate_thresholds=gate_thresholds,
        gate_row_metrics=gate_row_metrics,
    )
    all_stop_codes = []
    all_iterations = []
    for row in rows:
        all_stop_codes.extend(row["projection_lsmr_stop_codes"])
        all_stop_codes.extend(row["independent_lsmr_stop_codes"])
        all_iterations.extend(row["projection_lsmr_iterations"])
        all_iterations.extend(row["independent_lsmr_iterations"])
    checks = {
        "exact_registered_cohort": len(rows) == len(cohort) * len(CALLS),
        "test_split_not_accessed": all(row["split"] != "test" for row in rows),
        "all_rows_finite": all(bool(row["finite"]) for row in rows),
        "lsmr_stop_codes_accepted": set(all_stop_codes) <= ALLOWED_LSMR_STOP_CODES,
    }
    if direct_projector is not None:
        checks.update(
            {
                "direct_factorization_count_matches_components": (
                    direct_projector.factorization_count
                    == operators.topology.num_connected_components
                ),
                "direct_factorization_reused": all(
                    row["direct_factorization_count"]
                    == direct_projector.factorization_count
                    and tuple(row["direct_anchor_cells"])
                    == direct_projector.anchor_cells
                    for row in rows
                ),
            }
        )
    for name, threshold in gate_thresholds.items():
        checks[name] = aggregates[name] <= threshold
    passed = all(checks.values())
    config = {
        "schema": schema,
        "experiment_id": experiment_id,
        "canonical_solver": args.canonical_solver,
        "source_geometry_key": SOURCE_GEOMETRY_KEY,
        "train_keys": list(TRAIN_KEYS),
        "validation_keys": list(VALIDATION_KEYS),
        "calls": list(CALLS),
        "lsmr_tolerance": LSMR_TOLERANCE,
        "lsmr_max_iterations": LSMR_MAX_ITERATIONS,
        "allowed_lsmr_stop_codes": sorted(ALLOWED_LSMR_STOP_CODES),
        "gate_thresholds": gate_thresholds,
    }
    summary = {
        "schema": schema,
        "experiment_id": experiment_id,
        "status": "passed" if passed else "failed",
        "config": config,
        "config_digest": _digest_mapping(config),
        "family_id": geometry.source_family_id,
        "family_manifest_digest": geometry.source_family_manifest_digest,
        "shard_manifest_digest": shard_manifest_digest,
        "physical_geometry_digest": geometry.physical_geometry_digest,
        "graph_geometry_digest": geometry.graph_geometry_digest,
        "topology": operators.topology.to_dict(),
        "direct_projector": (
            direct_projector.summary() if direct_projector is not None else None
        ),
        "diagnostic_weight": "W_f=face_measure*dual_width; minimum W_f^-1 norm",
        "row_count": len(rows),
        "accessed_case_splits": {
            "train": list(TRAIN_KEYS),
            "validation": list(VALIDATION_KEYS),
            "test": [],
        },
        "accessed_reference_sha256": accessed_digests,
        "aggregates": {
            **aggregates,
            "reference_cycle_energy_fraction_median": float(
                np.median([row["reference_cycle_energy_fraction"] for row in rows])
            ),
            "reference_cycle_energy_fraction_max": float(
                np.max([row["reference_cycle_energy_fraction"] for row in rows])
            ),
            "lsmr_iterations_median": float(np.median(all_iterations)),
            "lsmr_iterations_max": int(max(all_iterations)),
            "lsmr_stop_codes": sorted(set(all_stop_codes)),
        },
        "checks": checks,
        "promotion": {
            "divergence_active_four_pair_tiny_fit_authorized": passed,
            "serious_training_authorized": False,
            "full_reference_cycle_supervision_authorized": False,
            "test_split_evaluation_authorized": False,
        },
        "code_sha256": {
            "entry_point": _sha256_file(Path(__file__)),
            "fv_impulse_diagnostics": _sha256_file(
                ROOT / "utility/time_dependent_no/fv_impulse_diagnostics.py"
            ),
            "pcno_face_impulse": _sha256_file(
                ROOT / "utility/time_dependent_no/pcno_face_impulse.py"
            ),
        },
        "elapsed_seconds": perf_counter() - start,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    _write_outputs(args.output_dir, summary, rows)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    if not passed:
        raise RuntimeError(
            f"{experiment_id} canonical-target preflight failed its frozen gates"
        )


if __name__ == "__main__":
    main()
