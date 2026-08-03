#!/usr/bin/env python3
"""D073-A: test final-layer PCNO sensitivity to physical graph radius."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from scipy.fft import dctn, idctn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no.analyze_pcno_fine_grained_pathways import (
    CALLS,
    CASE_IDS,
    CURRENT_SELF_CONSISTENT_SOURCE_CONTRACT,
    D070C_DCT_TOLERANCE,
    RESOLUTIONS,
    _aux,
    _inventory_checks,
    _model_context,
    _model_state_sha256,
    _pair_key,
    _pair_label,
    _restrict,
    _verify_d067,
)
from scripts.time_dependent_no.evaluate_pcno_node_type_interventions import (
    CaseData,
    _hook_equivalence,
    _load_dynamic_cases,
    _loaded_project_source_hashes,
    _verify_common_provenance,
)
from utility.time_dependent_no.pcno_artifacts import (
    atomic_write_json_with_paths as write_json,
)
from utility.time_dependent_no.pcno_artifacts import (
    digest_array,
    git_state,
    jsonable_args,
    runtime_environment,
    sha256_file,
    write_csv_with_paths,
)
from utility.time_dependent_no.pcno_euler2d import PCNOEuler2DShardStore
from utility.time_dependent_no.pcno_resolution_pathways import (
    GraphBallOperator,
    PreparedGraphBallOperator,
    build_graph_ball_operator,
    graph_ball_device_invariants,
    graph_two_hop_physical_radius,
    prepare_graph_ball_operator,
    trace_same_hidden_physical_radius_outputs,
)
from utility.time_dependent_no.pcno_resolution_transfer import (
    build_resolution_checkpoint_model,
    conservative_admissibility_summary,
    load_resolution_checkpoint,
    parse_resolution,
    resolution_label,
)
from utility.time_dependent_no.pcno_runtime import select_device
from utility.time_dependent_no.pcno_scale_separated_drift import dct_band_masks

SCHEMA = "pcno_physical_radius_geometry_diagnostic_v1"
MANIFEST_SCHEMA = "pcno_physical_radius_geometry_manifest_v2"
MODES = ("teacher_forced", "baseline_free_input")
ARMS = ("A0", "A1", "A2")
BANDS = ("total", "large", "transition", "local")
PATHWAY_LEVELS = (
    "pre_ball_gradient",
    "post_ball_smoothed_gradient",
    "post_softsign",
    "post_gw2",
    "decoded_residual",
)
LAYER = 3
ROW_SUM_FLOAT64_TOLERANCE = 1.0e-12
EXECUTED_FLOAT32_TOLERANCE = 2.0e-6
REPLAY_TOLERANCE = 2.0e-6
A1_DENOMINATOR_MINIMUM = 1.0e-8
A0_DENOMINATOR_MINIMUM = 1.0e-8
LARGE_REDUCTION_MINIMUM = 0.20
POSITIVE_CASE_MINIMUM = 4
OTHER_BAND_RATIO_MAXIMUM = 1.10
A2_A0_LARGE_RATIO_MAXIMUM = 1.05
VISUAL_CASE_IDS = ("sv_e00_y00", "sv_e11_y08")
GATED_BANDS = ("total", "large", "local")
D070_FROZEN_ARTIFACT_ARGUMENTS = (
    "expected_checkpoint_sha256",
    "expected_normalization_sha256",
    "expected_split_sha256",
    "expected_data_manifest_digest",
    "expected_family_manifest_sha256",
    "expected_d067_summary_sha256",
    "expected_source_base_git_head",
)
D070_REQUIRED_UNCHANGED_SOURCES = (
    "pcno/pcno.py",
    "scripts/time_dependent_no/analyze_pcno_fine_grained_pathways.py",
    "scripts/time_dependent_no/evaluate_pcno_node_type_interventions.py",
    "utility/normalizer.py",
    "utility/time_dependent_no/pcno_euler2d.py",
    "utility/time_dependent_no/pcno_fv_geometry.py",
    "utility/time_dependent_no/pcno_residual_structure.py",
    "utility/time_dependent_no/pcno_resolution_transfer.py",
    "utility/time_dependent_no/pcno_runtime.py",
    "utility/time_dependent_no/pcno_scale_separated_drift.py",
    "utility/time_dependent_no/shock_vortex_family.py",
    "utility/time_dependent_no/shock_vortex_fv.py",
    "utility/time_dependent_no/shock_vortex_metrics.py",
)
D070_INTENTIONAL_SOURCE_DIFFERENCES = {
    "utility/time_dependent_no/pcno_artifacts.py": (
        "source-snapshot v5 plus backward-compatible v4 verification; no D073 math"
    ),
    "utility/time_dependent_no/pcno_resolution_pathways.py": (
        "registered D073 graph-ball pathway implementation"
    ),
}
D070_REFERENCE_BINDING_FIELDS = (
    "frozen_training_reference_sha256",
    "active_reference_artifact_sha256",
    "retained_resolution",
    "state_dtype",
    "state_shape",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(
        family="dynamic_fv",
        d067_source_contract=CURRENT_SELF_CONSISTENT_SOURCE_CONTRACT,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization-json", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--family-root", type=Path, required=True)
    parser.add_argument("--multires-reference-root", type=Path, required=True)
    parser.add_argument("--d067-summary", type=Path, required=True)
    parser.add_argument("--d067-source-contract", required=True)
    parser.add_argument("--d070-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-ids", nargs="+", default=list(CASE_IDS))
    parser.add_argument(
        "--resolutions", nargs="+", default=["125x50", "250x100", "500x200"]
    )
    parser.add_argument("--training-resolution", default="250x100")
    parser.add_argument("--rollout-calls", type=int, default=30)
    parser.add_argument("--diagnostic-calls", type=int, nargs="+", default=CALLS)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--amp", choices=("none",), default="none")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-normalization-sha256", required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--expected-data-manifest-digest", required=True)
    parser.add_argument("--expected-family-manifest-sha256", required=True)
    parser.add_argument("--expected-d067-summary-sha256", required=True)
    parser.add_argument("--expected-d070-summary-sha256", required=True)
    parser.add_argument("--expected-source-base-git-head", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument(
        "--expected-source", action="append", default=[], metavar="PATH=SHA256"
    )
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if tuple(args.case_ids) != CASE_IDS:
        raise ValueError("D073 freezes the six D070C open evaluation cases")
    if tuple(parse_resolution(value) for value in args.resolutions) != RESOLUTIONS:
        raise ValueError("D073 freezes 125x50, 250x100, and 500x200")
    if parse_resolution(args.training_resolution) != (250, 100):
        raise ValueError("D073 freezes the training grid at 250x100")
    if args.rollout_calls != 30 or tuple(sorted(set(args.diagnostic_calls))) != CALLS:
        raise ValueError("D073 freezes H30 and calls 1,5,15,30")
    if args.d067_source_contract != CURRENT_SELF_CONSISTENT_SOURCE_CONTRACT:
        raise ValueError("D073 requires the accepted current-core D070C contract")
    digest_fields = (
        "expected_checkpoint_sha256",
        "expected_normalization_sha256",
        "expected_split_sha256",
        "expected_data_manifest_digest",
        "expected_family_manifest_sha256",
        "expected_d067_summary_sha256",
        "expected_d070_summary_sha256",
        "expected_source_manifest_sha256",
    )
    for name in digest_fields:
        value = str(getattr(args, name)).lower()
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"--{name.replace('_', '-')} must be a SHA-256 digest")
    head = str(args.expected_source_base_git_head).lower()
    if len(head) != 40 or any(
        character not in "0123456789abcdef" for character in head
    ):
        raise ValueError("--expected-source-base-git-head must be a Git SHA")
    return args


def _d070_compatibility_checks(
    args: argparse.Namespace,
    summary: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    active_source_sha256: Mapping[str, str],
) -> dict[str, Any]:
    d070_args = summary.get("args", {})
    d070_sources = summary.get("provenance", {}).get("source_sha256", {})
    artifact_arguments = {
        name: str(d070_args.get(name, "")).lower() == str(getattr(args, name)).lower()
        for name in D070_FROZEN_ARTIFACT_ARGUMENTS
    }
    unchanged_sources = {
        path: (
            path in d070_sources
            and path in active_source_sha256
            and d070_sources[path] == active_source_sha256[path]
        )
        for path in D070_REQUIRED_UNCHANGED_SOURCES
    }
    intentional_source_differences = {
        path: {
            "d070_sha256": d070_sources.get(path),
            "d073_sha256": active_source_sha256.get(path),
            "difference_allowed": True,
            "reason": D070_INTENTIONAL_SOURCE_DIFFERENCES[path],
        }
        for path in D070_INTENTIONAL_SOURCE_DIFFERENCES
    }
    checks = {
        "scientific_interpretation_allowed": (
            summary.get("scientific_interpretation_allowed") is True
        ),
        "artifact_arguments": artifact_arguments,
        "normalization_mapping_digest_matches": (
            summary.get("normalization_digest")
            == checkpoint.get("normalization_digest")
        ),
        "d067_source_contract_matches": (
            summary.get("d067_binding", {}).get("source_contract")
            == args.d067_source_contract
        ),
        "required_unchanged_sources": unchanged_sources,
        "intentional_source_differences": intentional_source_differences,
    }
    checks["passed"] = bool(
        checks["scientific_interpretation_allowed"]
        and all(artifact_arguments.values())
        and checks["normalization_mapping_digest_matches"]
        and checks["d067_source_contract_matches"]
        and all(unchanged_sources.values())
        and all(
            row["d070_sha256"] is not None and row["d073_sha256"] is not None
            for row in intentional_source_differences.values()
        )
    )
    return checks


def _read_d070_reference_checks(
    args: argparse.Namespace, summary: Mapping[str, Any]
) -> list[dict[str, str]]:
    relative = "reference_checks.csv"
    expected_sha = summary.get("output_hashes", {}).get(relative)
    path = args.d070_summary.parent / relative
    if expected_sha is None or not path.is_file() or sha256_file(path) != expected_sha:
        raise ValueError("accepted D070C reference-check artifact is not exact")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_fields = {"case_id", *D070_REFERENCE_BINDING_FIELDS}
    if (
        len(rows) != len(CASE_IDS)
        or {row.get("case_id") for row in rows} != set(CASE_IDS)
        or any(not expected_fields.issubset(row) for row in rows)
    ):
        raise ValueError("accepted D070C reference-check inventory changed")
    return rows


def _canonical_reference_binding(field: str, value: Any) -> Any:
    if field == "state_shape":
        parsed = json.loads(value) if isinstance(value, str) else value
        return tuple(int(item) for item in parsed)
    return str(value)


def _reference_binding_checks(
    expected_rows: Sequence[Mapping[str, Any]],
    active_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected = {str(row["case_id"]): row for row in expected_rows}
    active = {str(row["case_id"]): row for row in active_rows}
    exact_case_inventory = (
        len(expected) == len(expected_rows)
        and len(active) == len(active_rows)
        and set(expected) == set(active) == set(CASE_IDS)
    )
    field_matches = {
        case_id: {
            field: (
                field in expected[case_id]
                and field in active[case_id]
                and _canonical_reference_binding(field, expected[case_id][field])
                == _canonical_reference_binding(field, active[case_id][field])
            )
            for field in D070_REFERENCE_BINDING_FIELDS
        }
        for case_id in sorted(set(expected) & set(active))
    }
    mismatches = {
        case_id: [field for field, passed in fields.items() if not passed]
        for case_id, fields in field_matches.items()
        if not all(fields.values())
    }
    return {
        "binding_fields": list(D070_REFERENCE_BINDING_FIELDS),
        "exact_case_inventory": exact_case_inventory,
        "field_matches": field_matches,
        "mismatches": mismatches,
        "passed": bool(exact_case_inventory and not mismatches),
    }


def _verify_d070(
    args: argparse.Namespace,
    checkpoint: Mapping[str, Any],
    *,
    active_source_sha256: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, str]]]:
    if sha256_file(args.d070_summary) != args.expected_d070_summary_sha256.lower():
        raise ValueError("D070C summary digest mismatch")
    summary = json.loads(args.d070_summary.read_text(encoding="utf-8"))
    if (
        summary.get("schema") != "pcno_fine_grained_pathway_diagnostic_v2"
        or summary.get("status") != "complete"
        or summary.get("contract_checks_passed") is not True
        or summary.get("scientific_interpretation_allowed") is not True
    ):
        raise ValueError("D070C binding is not a complete passing v2 result")
    if summary.get("checkpoint_sha256") != args.expected_checkpoint_sha256.lower():
        raise ValueError("D070C and D073 checkpoint bindings differ")
    population = summary.get("population", {})
    if population.get("case_ids") != list(CASE_IDS):
        raise ValueError("D070C case population differs from D073")
    d067_binding = summary.get("d067_binding", {})
    if d067_binding.get("summary_sha256") != args.expected_d067_summary_sha256.lower():
        raise ValueError("D070C and D073 D067 bindings differ")
    mechanism = summary.get("mechanism_selection", {})
    if mechanism.get("decision") != "multiple_supported":
        raise ValueError("D073 requires the registered D070C multiple-supported result")
    for relative, expected in summary.get("output_hashes", {}).items():
        path = args.d070_summary.parent / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"D070C output digest mismatch: {relative}")
    compatibility = _d070_compatibility_checks(
        args, summary, checkpoint, active_source_sha256
    )
    if not compatibility["passed"]:
        raise ValueError(f"D070C and D073 compatibility checks failed: {compatibility}")
    return summary, compatibility, _read_d070_reference_checks(args, summary)


def _array_sha256(*values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _operator_sha256(operator: GraphBallOperator) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray((operator.node_count,), dtype=np.int64).tobytes())
    digest.update(
        np.asarray(
            (operator.radius, operator.geometry_tolerance), dtype=np.float64
        ).tobytes()
    )
    for value in (
        operator.targets,
        operator.sources,
        operator.coefficients,
        operator.distances,
    ):
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _quantile_fields(name: str, value: np.ndarray) -> dict[str, float]:
    array = np.asarray(value, dtype=np.float64)
    quantiles = np.quantile(array, (0.0, 0.25, 0.5, 0.75, 1.0))
    return {
        f"{name}_minimum": float(quantiles[0]),
        f"{name}_q25": float(quantiles[1]),
        f"{name}_median": float(quantiles[2]),
        f"{name}_q75": float(quantiles[3]),
        f"{name}_maximum": float(quantiles[4]),
    }


def _edge_reciprocity(edges: np.ndarray) -> dict[str, Any]:
    directed = {
        (int(left), int(right))
        for left, right in np.asarray(edges, dtype=np.int64)
        if int(left) != int(right)
    }
    missing = sorted(
        (left, right) for left, right in directed if (right, left) not in directed
    )
    return {
        "directed_edge_count": len(directed),
        "missing_reverse_edge_count": len(missing),
        "reciprocal": not missing,
        "first_missing_reverse_edges": [list(value) for value in missing[:10]],
    }


def _row_slices(operator: GraphBallOperator) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(operator.targets, minlength=operator.node_count)
    offsets = np.concatenate((np.asarray((0,), dtype=np.int64), np.cumsum(counts)))
    return counts, offsets


def _operator_changed_rows(
    local: GraphBallOperator, fixed: GraphBallOperator
) -> np.ndarray:
    _, local_offsets = _row_slices(local)
    _, fixed_offsets = _row_slices(fixed)
    changed = np.zeros(local.node_count, dtype=bool)
    for target in range(local.node_count):
        local_slice = slice(local_offsets[target], local_offsets[target + 1])
        fixed_slice = slice(fixed_offsets[target], fixed_offsets[target + 1])
        changed[target] = not (
            np.array_equal(local.sources[local_slice], fixed.sources[fixed_slice])
            and np.array_equal(
                local.coefficients[local_slice], fixed.coefficients[fixed_slice]
            )
        )
    return changed


def _operator_overlap(
    local: GraphBallOperator, fixed: GraphBallOperator
) -> dict[str, float]:
    local_counts, local_offsets = _row_slices(local)
    fixed_counts, fixed_offsets = _row_slices(fixed)
    changed = _operator_changed_rows(local, fixed)
    jaccard = np.empty(local.node_count, dtype=np.float64)
    coefficient_l1 = np.empty(local.node_count, dtype=np.float64)
    for target in range(local.node_count):
        local_slice = slice(local_offsets[target], local_offsets[target + 1])
        fixed_slice = slice(fixed_offsets[target], fixed_offsets[target + 1])
        local_sources = local.sources[local_slice]
        fixed_sources = fixed.sources[fixed_slice]
        local_coefficients = local.coefficients[local_slice]
        fixed_coefficients = fixed.coefficients[fixed_slice]
        intersection = np.intersect1d(local_sources, fixed_sources).size
        union = np.union1d(local_sources, fixed_sources).size
        jaccard[target] = intersection / union
        local_mapping = dict(
            zip(local_sources.tolist(), local_coefficients.tolist(), strict=True)
        )
        fixed_mapping = dict(
            zip(fixed_sources.tolist(), fixed_coefficients.tolist(), strict=True)
        )
        coefficient_l1[target] = sum(
            abs(local_mapping.get(source, 0.0) - fixed_mapping.get(source, 0.0))
            for source in set(local_mapping) | set(fixed_mapping)
        )
    return {
        "changed_row_fraction": float(np.mean(changed)),
        "support_jaccard_median": float(np.median(jaccard)),
        "coefficient_l1_median": float(np.median(coefficient_l1)),
        "local_neighbor_count_median": float(np.median(local_counts)),
        "fixed_neighbor_count_median": float(np.median(fixed_counts)),
    }


def _operator_inventory_row(
    *,
    resolution: tuple[int, int],
    arm: str,
    operator: GraphBallOperator,
    prepared: PreparedGraphBallOperator,
    weights: np.ndarray,
    node_type: np.ndarray,
    boundary_distance: np.ndarray,
    changed_rows: np.ndarray,
    overlap: Mapping[str, float],
) -> dict[str, Any]:
    counts, _ = _row_slices(operator)
    support_weight = np.bincount(
        operator.targets,
        weights=np.asarray(weights, dtype=np.float64)[operator.sources],
        minlength=operator.node_count,
    )
    executed = graph_ball_device_invariants(prepared)
    type_values = sorted(int(value) for value in np.unique(node_type))
    type0 = np.asarray(node_type) == 0
    non_type0 = ~type0
    row_maximum_distance = np.zeros(operator.node_count, dtype=np.float64)
    np.maximum.at(row_maximum_distance, operator.targets, operator.distances)
    row = {
        "resolution": resolution_label(resolution),
        "arm": arm,
        "radius": operator.radius,
        "geometry_tolerance": operator.geometry_tolerance,
        "node_count": operator.node_count,
        "undirected_edge_count": operator.undirected_edge_count,
        "nnz": int(operator.targets.size),
        "operator_sha256": _operator_sha256(operator),
        "float64_maximum_row_sum_error": operator.maximum_row_sum_error,
        "minimum_coefficient": operator.minimum_coefficient,
        "float32_maximum_row_sum_error": executed["maximum_row_sum_error"],
        "float32_maximum_constant_error": executed["maximum_constant_error"],
        "type0_mean_neighbor_count": float(np.mean(counts[type0])),
        "non_type0_mean_neighbor_count": (
            float(np.mean(counts[non_type0])) if np.any(non_type0) else None
        ),
        **overlap,
        **_quantile_fields("neighbor_count", counts),
        **_quantile_fields("support_weight", support_weight),
        **_quantile_fields("realized_distance", operator.distances),
        **_quantile_fields("coefficient", operator.coefficients),
    }
    strata = {
        "type0": type0,
        "non_type0": non_type0,
        "boundary_distance_le_0p02": np.asarray(boundary_distance) <= 0.02,
        "boundary_distance_le_0p05": np.asarray(boundary_distance) <= 0.05,
        "interior_distance_gt_0p05": np.asarray(boundary_distance) > 0.05,
    }
    for name, selected in strata.items():
        if not np.any(selected):
            raise ValueError(f"D073 operator stratum is empty: {name}")
        row[f"{name}_node_count"] = int(np.sum(selected))
        row[f"{name}_mean_neighbor_count"] = float(np.mean(counts[selected]))
        row[f"{name}_median_maximum_distance"] = float(
            np.median(row_maximum_distance[selected])
        )
        row[f"{name}_median_support_weight"] = float(
            np.median(support_weight[selected])
        )
        row[f"{name}_changed_row_fraction"] = float(np.mean(changed_rows[selected]))
    target_type = np.asarray(node_type, dtype=np.int64)[operator.targets]
    source_type = np.asarray(node_type, dtype=np.int64)[operator.sources]
    for target_value in type_values:
        target_mask = target_type == target_value
        denominator = float(np.sum(operator.coefficients[target_mask]))
        for source_value in type_values:
            numerator = float(
                np.sum(
                    operator.coefficients[target_mask & (source_type == source_value)]
                )
            )
            row[
                f"target_type_{target_value}_source_type_{source_value}_coefficient_share"
            ] = (None if denominator == 0.0 else numerator / denominator)
    return row


def _weighted_scaled_rms(
    value: np.ndarray, *, weights: np.ndarray, scale: np.ndarray
) -> float:
    array = (
        np.asarray(value, dtype=np.float64)
        / np.asarray(scale, dtype=np.float64)[None, :]
    )
    mass = np.asarray(weights, dtype=np.float64)
    return float(np.sqrt(np.einsum("n,nc,nc->", mass, array, array) / np.sum(mass)))


def _generic_band_fields(
    field: np.ndarray,
    *,
    resolution: tuple[int, int],
    component_scale: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Project any finite channel count with the registered spatial DCT masks."""

    value = np.asarray(field, dtype=np.float64)
    scale = np.asarray(component_scale, dtype=np.float64).reshape(-1)
    nx, ny = resolution
    if (
        value.ndim != 2
        or value.shape != (nx * ny, scale.size)
        or scale.size < 1
        or np.any(scale <= 0.0)
        or not np.isfinite(value).all()
        or not np.isfinite(scale).all()
    ):
        raise ValueError("D073 DCT field, grid, and component scale must align")
    scaled = value.reshape(ny, nx, scale.size) / scale[None, None, :]
    coefficients = dctn(scaled, type=2, axes=(0, 1), norm="ortho")
    masks = dct_band_masks(resolution, domain_lengths=(2.0, 1.0))
    projected: dict[str, np.ndarray] = {"total": value}
    reconstructed = np.zeros_like(scaled)
    total_energy = float(np.sum(np.square(coefficients)))
    band_energy = 0.0
    for name, mask in masks.items():
        selected = coefficients * mask[:, :, None]
        scaled_band = idctn(selected, type=2, axes=(0, 1), norm="ortho")
        reconstructed += scaled_band
        band_energy += float(np.sum(np.square(selected)))
        projected[name] = (scaled_band * scale[None, None, :]).reshape(value.shape)
    denominator = max(total_energy, np.finfo(np.float64).tiny)
    return projected, {
        "maximum_reconstruction_abs_residual_scaled": float(
            np.max(np.abs(reconstructed - scaled))
        ),
        "maximum_instantaneous_energy_relative_closure": float(
            abs(band_energy - total_energy) / denominator
        ),
    }


def _comparison(
    value: np.ndarray,
    reference: np.ndarray,
    *,
    weights: np.ndarray,
    scale: np.ndarray,
) -> dict[str, float]:
    error = np.asarray(value, dtype=np.float64) - np.asarray(
        reference, dtype=np.float64
    )
    denominator = _weighted_scaled_rms(reference, weights=weights, scale=scale)
    numerator = _weighted_scaled_rms(error, weights=weights, scale=scale)
    return {
        "maximum_absolute_error": float(np.max(np.abs(error))),
        "error_scaled_rms": numerator,
        "reference_scaled_rms": denominator,
        "relative_scaled_rms": numerator / max(denominator, np.finfo(float).tiny),
    }


def _numpy_channel_last(value: torch.Tensor) -> np.ndarray:
    if value.ndim != 3 or value.shape[0] != 1:
        raise ValueError("D073 trace field must have shape [1,C,N]")
    return value[0].permute(1, 0).detach().float().cpu().numpy().astype(np.float64)


def _trace_level_fields(
    trace: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    residual_scale: np.ndarray,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    levels: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    scales: dict[str, np.ndarray] = {}
    names = {
        "pre_ball_gradient": ("coarse_gradient", "restricted_fine_gradient"),
        "post_ball_smoothed_gradient": (
            "coarse_smoothed_gradient",
            "restricted_fine_smoothed_gradient",
        ),
        "post_softsign": ("coarse_post_softsign", "restricted_fine_post_softsign"),
        "post_gw2": ("coarse_differential", "restricted_fine_differential"),
    }
    for arm in ARMS:
        for level, (coarse_name, fine_name) in names.items():
            levels[level][arm] = _numpy_channel_last(
                trace[arm][coarse_name] - trace[arm][fine_name]
            )
            scales[level] = np.ones(levels[level][arm].shape[1], dtype=np.float64)
        normalized_gap = (
            (trace[arm]["coarse_output"] - trace[arm]["restricted_fine_output"])[0]
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        levels["decoded_residual"][arm] = normalized_gap * residual_scale[None, :]
    scales["decoded_residual"] = np.asarray(residual_scale, dtype=np.float64)
    return dict(levels), scales


def _analyze_trace(
    *,
    case_id: str,
    mode: str,
    pair: str,
    call: int,
    resolution: tuple[int, int],
    trace: Mapping[str, Mapping[str, torch.Tensor]],
    weights: np.ndarray,
    residual_scale: np.ndarray,
    arm_rows: list[dict[str, Any]],
    component_rows: list[dict[str, Any]],
    closure_rows: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    levels, scales = _trace_level_fields(trace, residual_scale=residual_scale)
    visual: dict[str, np.ndarray] = {}
    for level in PATHWAY_LEVELS:
        arm_bands: dict[str, dict[str, np.ndarray]] = {}
        for arm in ARMS:
            bands, closure = _generic_band_fields(
                levels[level][arm],
                resolution=resolution,
                component_scale=scales[level],
            )
            arm_bands[arm] = bands
            closure_rows.append(
                {
                    "case_id": case_id,
                    "mode": mode,
                    "pair": pair,
                    "call": call,
                    "check": "dct_band_closure",
                    "pathway_level": level,
                    "arm": arm,
                    "maximum_error": max(float(value) for value in closure.values()),
                    "tolerance": D070C_DCT_TOLERANCE,
                    "passed": max(float(value) for value in closure.values())
                    <= D070C_DCT_TOLERANCE,
                }
            )
        for arm in ARMS:
            for band in BANDS:
                value = _weighted_scaled_rms(
                    arm_bands[arm][band], weights=weights, scale=scales[level]
                )
                response = _weighted_scaled_rms(
                    arm_bands[arm][band] - arm_bands["A0"][band],
                    weights=weights,
                    scale=scales[level],
                )
                response_to_a1 = _weighted_scaled_rms(
                    arm_bands[arm][band] - arm_bands["A1"][band],
                    weights=weights,
                    scale=scales[level],
                )
                arm_rows.append(
                    {
                        "case_id": case_id,
                        "mode": mode,
                        "pair": pair,
                        "call": call,
                        "pathway_level": level,
                        "arm": arm,
                        "band": band,
                        "defect_scaled_rms": value,
                        "response_to_a0_scaled_rms": response,
                        "response_to_a1_scaled_rms": response_to_a1,
                    }
                )
                if level == "decoded_residual":
                    for component in range(4):
                        component_rows.append(
                            {
                                "case_id": case_id,
                                "mode": mode,
                                "pair": pair,
                                "call": call,
                                "arm": arm,
                                "band": band,
                                "component": component,
                                "defect_scaled_rms": _weighted_scaled_rms(
                                    arm_bands[arm][band][:, component : component + 1],
                                    weights=weights,
                                    scale=residual_scale[component : component + 1],
                                ),
                            }
                        )
        if level == "decoded_residual":
            for arm in ARMS:
                for band in ("total", "large", "local"):
                    visual[f"{arm}_{band}"] = arm_bands[arm][band]
            for band in ("total", "large", "local"):
                visual[f"A2_minus_A1_{band}"] = (
                    arm_bands["A2"][band] - arm_bands["A1"][band]
                )

    a0_coarse = _comparison(
        trace["A0"]["coarse_output"][0].detach().float().cpu().numpy(),
        trace["A0"]["native_coarse_output"][0].detach().float().cpu().numpy(),
        weights=weights,
        scale=np.ones(4),
    )
    a0_fine = _comparison(
        trace["A0"]["restricted_fine_output"][0].detach().float().cpu().numpy(),
        trace["A0"]["native_restricted_fine_output"][0].detach().float().cpu().numpy(),
        weights=weights,
        scale=np.ones(4),
    )
    for name, metrics in (
        ("a0_coarse_native_reuse", a0_coarse),
        ("a0_fine_native_reuse", a0_fine),
    ):
        closure_rows.append(
            {
                "case_id": case_id,
                "mode": mode,
                "pair": pair,
                "call": call,
                "check": name,
                "pathway_level": "decoded_residual",
                "arm": "A0",
                **metrics,
                "absolute_tolerance": REPLAY_TOLERANCE,
                "relative_tolerance": REPLAY_TOLERANCE,
                "passed": metrics["maximum_absolute_error"] <= REPLAY_TOLERANCE
                and metrics["relative_scaled_rms"] <= REPLAY_TOLERANCE,
            }
        )
    anchor_prefix = "coarse_" if pair.startswith("250x100") else "restricted_fine_"
    for name in (
        "spectral",
        "pointwise",
        "gradient",
        "smoothed_gradient",
        "post_softsign",
        "differential",
        "output",
    ):
        key = f"{anchor_prefix}{name}"
        left = trace["A1"][key]
        right = trace["A2"][key]
        if key.endswith("output"):
            left_array = left[0].detach().float().cpu().numpy()
            right_array = right[0].detach().float().cpu().numpy()
        else:
            left_array = _numpy_channel_last(left)
            right_array = _numpy_channel_last(right)
        metrics = _comparison(
            left_array,
            right_array,
            weights=weights,
            scale=np.ones(left_array.shape[-1]),
        )
        closure_rows.append(
            {
                "case_id": case_id,
                "mode": mode,
                "pair": pair,
                "call": call,
                "check": "a1_a2_training_grid_output_identity",
                "pathway_level": name,
                "arm": "A1_A2",
                **metrics,
                "absolute_tolerance": EXECUTED_FLOAT32_TOLERANCE,
                "relative_tolerance": EXECUTED_FLOAT32_TOLERANCE,
                "passed": metrics["maximum_absolute_error"]
                <= EXECUTED_FLOAT32_TOLERANCE
                and metrics["relative_scaled_rms"] <= EXECUTED_FLOAT32_TOLERANCE,
            }
        )
    return visual


def _case_aggregates(
    rows: Sequence[Mapping[str, Any]], *, required_calls: Sequence[int]
) -> list[dict[str, Any]]:
    grouped: defaultdict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    fields = ("case_id", "mode", "pair", "pathway_level", "arm", "band")
    for row in rows:
        grouped[tuple(str(row[name]) for name in fields)].append(row)
    result = []
    expected_calls = {int(value) for value in required_calls}
    for key, values in sorted(grouped.items()):
        observed_calls = {int(value["call"]) for value in values}
        if observed_calls != expected_calls or len(values) != len(expected_calls):
            raise ValueError(f"incomplete D073 call aggregate: {key}")
        norms = np.asarray(
            [float(value["defect_scaled_rms"]) for value in values],
            dtype=np.float64,
        )
        result.append(
            {
                **dict(zip(fields, key, strict=True)),
                "call_count": len(values),
                "call_rms_aggregate": float(np.sqrt(np.mean(np.square(norms)))),
            }
        )
    return result


def _all_numeric_values_finite(rows: Sequence[Mapping[str, Any]]) -> bool:
    def check(value: Any) -> bool:
        if value is None or isinstance(value, (str, bool)):
            return True
        if isinstance(value, Mapping):
            return all(check(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return all(check(item) for item in value)
        if isinstance(value, (int, np.integer)):
            return True
        if isinstance(value, (float, np.floating)):
            return bool(np.isfinite(float(value)))
        return True

    return all(check(row) for row in rows)


def _visual_payload_inventory(
    payloads: Mapping[tuple[str, str, str], Mapping[str, Sequence[Any]]],
    *,
    active_case_ids: Sequence[str],
    diagnostic_calls: Sequence[int],
    grouped: Mapping[str, Mapping[tuple[int, int], CaseData]],
) -> dict[str, Any]:
    visual_cases = [
        case_id for case_id in VISUAL_CASE_IDS if case_id in set(active_case_ids)
    ]
    expected_keys = {
        (case_id, _pair_label(coarse, fine), mode)
        for case_id in visual_cases
        for coarse, fine in pairwise(RESOLUTIONS)
        for mode in MODES
    }
    expected_fields = {
        "call",
        "physical_time",
        *(
            f"{arm}_{band}"
            for arm in ("A0", "A1", "A2")
            for band in ("total", "large", "local")
        ),
        *(f"A2_minus_A1_{band}" for band in ("total", "large", "local")),
    }
    observed_keys = set(payloads)
    errors = []
    for key in sorted(expected_keys & observed_keys):
        case_id, pair, _ = key
        payload = payloads[key]
        if set(payload) != expected_fields:
            errors.append(f"field_inventory:{key}")
            continue
        calls = tuple(int(value) for value in payload["call"])
        if calls != tuple(diagnostic_calls):
            errors.append(f"call_inventory:{key}")
        if len(payload["physical_time"]) != len(diagnostic_calls):
            errors.append(f"time_inventory:{key}")
        coarse = parse_resolution(pair.split("->", maxsplit=1)[0])
        expected_shape = (coarse[0] * coarse[1], 4)
        for name in expected_fields - {"call", "physical_time"}:
            sequence = payload[name]
            if len(sequence) != len(diagnostic_calls) or any(
                np.asarray(value).shape != expected_shape
                or not np.isfinite(np.asarray(value)).all()
                for value in sequence
            ):
                errors.append(f"array_inventory:{key}:{name}")
        if coarse not in grouped[case_id]:
            errors.append(f"coarse_geometry:{key}")
    return {
        "expected_keys": [list(value) for value in sorted(expected_keys)],
        "observed_keys": [list(value) for value in sorted(observed_keys)],
        "missing_keys": [
            list(value) for value in sorted(expected_keys - observed_keys)
        ],
        "extra_keys": [list(value) for value in sorted(observed_keys - expected_keys)],
        "errors": errors,
        "passed": observed_keys == expected_keys and not errors,
    }


def _gate_summary(
    aggregates: Sequence[Mapping[str, Any]], *, smoke: bool
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if smoke:
        return [], {
            "evaluated": False,
            "mechanism_pass": None,
            "native_relevance_pass": None,
            "advance_d073b": False,
        }
    lookup = {
        (
            str(row["case_id"]),
            str(row["pair"]),
            str(row["mode"]),
            str(row["arm"]),
            str(row["band"]),
        ): float(row["call_rms_aggregate"])
        for row in aggregates
        if row["pathway_level"] == "decoded_residual"
    }
    rows = []
    for coarse, fine in pairwise(RESOLUTIONS):
        pair = _pair_label(coarse, fine)
        for mode in MODES:
            ratios: dict[str, list[float]] = {band: [] for band in GATED_BANDS}
            native_ratios: dict[str, list[float]] = {band: [] for band in GATED_BANDS}
            a1_values: dict[str, list[float]] = {band: [] for band in GATED_BANDS}
            a0_values: dict[str, list[float]] = {band: [] for band in GATED_BANDS}
            invalid_a1: dict[str, list[str]] = {band: [] for band in GATED_BANDS}
            invalid_a0: dict[str, list[str]] = {band: [] for band in GATED_BANDS}
            for case_id in CASE_IDS:
                for band in GATED_BANDS:
                    a0 = lookup[(case_id, pair, mode, "A0", band)]
                    a1 = lookup[(case_id, pair, mode, "A1", band)]
                    a2 = lookup[(case_id, pair, mode, "A2", band)]
                    a1_values[band].append(a1)
                    a0_values[band].append(a0)
                    if a1 <= A1_DENOMINATOR_MINIMUM:
                        invalid_a1[band].append(case_id)
                    else:
                        ratios[band].append(a2 / a1)
                    if a0 <= A0_DENOMINATOR_MINIMUM:
                        invalid_a0[band].append(case_id)
                    else:
                        native_ratios[band].append(a2 / a0)
            a1_denominator_valid = not any(invalid_a1.values())
            a0_denominator_valid = not any(invalid_a0.values())
            median_large_reduction = (
                float(np.median(1.0 - np.asarray(ratios["large"])))
                if a1_denominator_valid
                else None
            )
            positive_cases = (
                sum(value < 1.0 for value in ratios["large"])
                if a1_denominator_valid
                else None
            )
            median_a2_a1 = {
                band: float(np.median(ratios[band])) if a1_denominator_valid else None
                for band in GATED_BANDS
            }
            median_a2_a0 = {
                band: (
                    float(np.median(native_ratios[band]))
                    if a0_denominator_valid
                    else None
                )
                for band in GATED_BANDS
            }
            mechanism_pass = bool(
                a1_denominator_valid
                and median_large_reduction is not None
                and median_large_reduction >= LARGE_REDUCTION_MINIMUM
                and positive_cases is not None
                and positive_cases >= POSITIVE_CASE_MINIMUM
                and median_a2_a1["total"] <= OTHER_BAND_RATIO_MAXIMUM
                and median_a2_a1["local"] <= OTHER_BAND_RATIO_MAXIMUM
            )
            relevance_pass = bool(
                a0_denominator_valid
                and median_a2_a0["large"] <= A2_A0_LARGE_RATIO_MAXIMUM
                and median_a2_a0["total"] <= OTHER_BAND_RATIO_MAXIMUM
                and median_a2_a0["local"] <= OTHER_BAND_RATIO_MAXIMUM
            )
            leave_one_out = []
            if a1_denominator_valid:
                for omitted in range(len(CASE_IDS)):
                    selected = [
                        index for index in range(len(CASE_IDS)) if index != omitted
                    ]
                    leave_one_out.append(
                        float(np.median(1.0 - np.asarray(ratios["large"])[selected]))
                    )
            rows.append(
                {
                    "pair": pair,
                    "mode": mode,
                    "case_count": len(CASE_IDS),
                    **{
                        f"a1_{band}_denominator_minimum": min(a1_values[band])
                        for band in GATED_BANDS
                    },
                    **{
                        f"a0_{band}_denominator_minimum": min(a0_values[band])
                        for band in GATED_BANDS
                    },
                    "a1_invalid_denominator_cases": invalid_a1,
                    "a0_invalid_denominator_cases": invalid_a0,
                    "a1_denominator_valid": a1_denominator_valid,
                    "a0_denominator_valid": a0_denominator_valid,
                    "median_large_reduction_a2_vs_a1": median_large_reduction,
                    "positive_large_cases": positive_cases,
                    "median_total_ratio_a2_vs_a1": median_a2_a1["total"],
                    "median_local_ratio_a2_vs_a1": median_a2_a1["local"],
                    "median_large_ratio_a2_vs_a0": median_a2_a0["large"],
                    "median_total_ratio_a2_vs_a0": median_a2_a0["total"],
                    "median_local_ratio_a2_vs_a0": median_a2_a0["local"],
                    "leave_one_case_out_minimum_large_reduction": (
                        min(leave_one_out) if leave_one_out else None
                    ),
                    "mechanism_pass": mechanism_pass,
                    "native_relevance_pass": relevance_pass,
                }
            )
    return rows, {
        "evaluated": True,
        "mechanism_pass": all(row["mechanism_pass"] for row in rows),
        "native_relevance_pass": all(row["native_relevance_pass"] for row in rows),
        "advance_d073b": all(
            row["mechanism_pass"] and row["native_relevance_pass"] for row in rows
        ),
    }


def _contract_masked_decision(
    raw_decision: Mapping[str, Any], *, contract_pass: bool, smoke: bool
) -> dict[str, Any]:
    if not contract_pass:
        return {
            "evaluated": False,
            "mechanism_pass": None,
            "native_relevance_pass": None,
            "advance_d073b": False,
            "reason": "scientific_decision_suppressed_by_contract_failure",
        }
    decision = dict(raw_decision)
    decision["advance_d073b"] = bool(not smoke and raw_decision["advance_d073b"])
    return decision


def _manifest_science_result(*, contract_pass: bool, smoke: bool) -> bool:
    return bool(contract_pass and not smoke)


def _prepare_geometry_operators(
    grouped: Mapping[str, Mapping[tuple[int, int], CaseData]],
    *,
    device: torch.device,
) -> tuple[
    dict[tuple[int, int], dict[str, GraphBallOperator]],
    dict[tuple[int, int], dict[str, PreparedGraphBallOperator]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    geometry_rows = []
    radius_by_resolution: dict[tuple[int, int], float] = {}
    first_case = CASE_IDS[0]
    for resolution in RESOLUTIONS:
        case = grouped[first_case][resolution]
        expected_digest = _array_sha256(
            case.nodes, case.edges, case.weights, case.physical_node_type
        )
        for case_id in CASE_IDS[1:]:
            other = grouped[case_id][resolution]
            actual_digest = _array_sha256(
                other.nodes,
                other.edges,
                other.weights,
                other.physical_node_type,
            )
            if actual_digest != expected_digest:
                raise ValueError(f"D073 geometry changes across cases at {resolution}")
        reciprocity = _edge_reciprocity(case.edges)
        if not reciprocity["reciprocal"]:
            raise ValueError(f"D073 requires reciprocal edges: {reciprocity}")
        type0 = np.asarray(case.physical_node_type) == 0
        radius_by_resolution[resolution] = graph_two_hop_physical_radius(
            case.nodes, case.edges, node_mask=type0
        )
        row = {
            "resolution": resolution_label(resolution),
            "geometry_sha256": expected_digest,
            "nodes_sha256": digest_array(case.nodes),
            "edges_sha256": digest_array(case.edges),
            "weights_sha256": digest_array(case.weights),
            "node_type_sha256": digest_array(case.physical_node_type),
            "node_count": int(case.nodes.shape[0]),
            "directed_edge_count": int(case.edges.shape[0]),
            "type0_node_count": int(type0.sum()),
            "type0_weight_mass": float(case.weights[type0].sum()),
            "local_two_hop_radius": radius_by_resolution[resolution],
            **reciprocity,
        }
        for type_value in range(4):
            selected = np.asarray(case.physical_node_type) == type_value
            row[f"type_{type_value}_node_count"] = int(selected.sum())
            row[f"type_{type_value}_weight_mass"] = float(case.weights[selected].sum())
        geometry_rows.append(row)
    r_star = radius_by_resolution[(250, 100)]
    for row in geometry_rows:
        row["fixed_training_radius"] = r_star
    radius_ordering_pass = (
        radius_by_resolution[(125, 50)]
        > radius_by_resolution[(250, 100)]
        > radius_by_resolution[(500, 200)]
    )
    if not radius_ordering_pass:
        raise ValueError(f"D073 radius ordering failed: {radius_by_resolution}")

    operators: dict[tuple[int, int], dict[str, GraphBallOperator]] = {}
    prepared: dict[tuple[int, int], dict[str, PreparedGraphBallOperator]] = {}
    operator_rows = []
    for resolution in RESOLUTIONS:
        case = grouped[first_case][resolution]
        operators[resolution] = {
            "A1": build_graph_ball_operator(
                case.nodes,
                case.edges,
                case.weights,
                radius=radius_by_resolution[resolution],
            ),
            "A2": build_graph_ball_operator(
                case.nodes, case.edges, case.weights, radius=r_star
            ),
        }
        prepared[resolution] = {
            arm: prepare_graph_ball_operator(
                operator, device=device, dtype=torch.float32
            )
            for arm, operator in operators[resolution].items()
        }
        if resolution == (250, 100):
            anchor_arrays_equal_now = all(
                np.array_equal(
                    getattr(operators[resolution]["A1"], name),
                    getattr(operators[resolution]["A2"], name),
                )
                for name in ("targets", "sources", "coefficients", "distances")
            )
            if not anchor_arrays_equal_now:
                raise RuntimeError("D073 anchor ball operators differ")
            prepared[resolution]["A2"] = prepared[resolution]["A1"]
        overlap = _operator_overlap(
            operators[resolution]["A1"], operators[resolution]["A2"]
        )
        changed_rows = _operator_changed_rows(
            operators[resolution]["A1"], operators[resolution]["A2"]
        )
        for arm in ("A1", "A2"):
            operator_rows.append(
                _operator_inventory_row(
                    resolution=resolution,
                    arm=arm,
                    operator=operators[resolution][arm],
                    prepared=prepared[resolution][arm],
                    weights=case.weights,
                    node_type=case.physical_node_type,
                    boundary_distance=case.boundary_distance,
                    changed_rows=changed_rows,
                    overlap=overlap,
                )
            )

    anchor_local = operators[(250, 100)]["A1"]
    anchor_fixed = operators[(250, 100)]["A2"]
    anchor_arrays_equal = all(
        np.array_equal(getattr(anchor_local, name), getattr(anchor_fixed, name))
        for name in ("targets", "sources", "coefficients", "distances")
    )
    changed = {
        resolution_label(resolution): _operator_overlap(
            operators[resolution]["A1"], operators[resolution]["A2"]
        )["changed_row_fraction"]
        for resolution in RESOLUTIONS
    }
    executed_pass = all(
        float(row["float64_maximum_row_sum_error"]) <= ROW_SUM_FLOAT64_TOLERANCE
        and float(row["float32_maximum_row_sum_error"]) <= EXECUTED_FLOAT32_TOLERANCE
        and float(row["float32_maximum_constant_error"]) <= EXECUTED_FLOAT32_TOLERANCE
        and float(row["minimum_coefficient"]) >= 0.0
        for row in operator_rows
    )
    checks = {
        "radii_frozen_before_model_outputs": True,
        "radius_by_resolution": {
            resolution_label(key): value for key, value in radius_by_resolution.items()
        },
        "fixed_training_radius": r_star,
        "radius_ordering_pass": radius_ordering_pass,
        "anchor_sparse_arrays_exactly_equal": anchor_arrays_equal,
        "changed_row_fraction": changed,
        "nonanchor_rows_change": changed["125x50"] > 0.0 and changed["500x200"] > 0.0,
        "anchor_rows_unchanged": changed["250x100"] == 0.0,
        "executed_operator_invariants_pass": executed_pass,
        "all_edge_sets_reciprocal": all(row["reciprocal"] for row in geometry_rows),
    }
    checks["passed"] = bool(
        checks["radius_ordering_pass"]
        and checks["anchor_sparse_arrays_exactly_equal"]
        and checks["nonanchor_rows_change"]
        and checks["anchor_rows_unchanged"]
        and checks["executed_operator_invariants_pass"]
        and checks["all_edge_sets_reciprocal"]
    )
    if not checks["passed"]:
        raise RuntimeError(f"D073 geometry/operator contract failed: {checks}")
    return operators, prepared, geometry_rows, operator_rows, checks


def _append_visual(
    payloads: defaultdict[tuple[str, str, str], dict[str, list[Any]]],
    *,
    case_id: str,
    pair: str,
    mode: str,
    call: int,
    physical_time: float,
    visual: Mapping[str, np.ndarray],
) -> None:
    if case_id not in VISUAL_CASE_IDS:
        return
    payload = payloads[(case_id, pair, mode)]
    payload.setdefault("call", []).append(call)
    payload.setdefault("physical_time", []).append(physical_time)
    for name, value in visual.items():
        payload.setdefault(name, []).append(np.asarray(value, dtype=np.float32))


def _write_visual_payloads(
    output_dir: Path,
    payloads: Mapping[tuple[str, str, str], Mapping[str, Sequence[Any]]],
    *,
    grouped: Mapping[str, Mapping[tuple[int, int], CaseData]],
    residual_scale: np.ndarray,
) -> dict[str, str]:
    payload_dir = output_dir / "visual_payloads"
    payload_dir.mkdir()
    hashes = {}
    for (case_id, pair, mode), values in sorted(payloads.items()):
        coarse_label = pair.split("->", maxsplit=1)[0]
        coarse = parse_resolution(coarse_label)
        case = grouped[case_id][coarse]
        arrays: dict[str, np.ndarray] = {
            "schema": np.asarray("pcno_physical_radius_visual_payload_v1"),
            "case_id": np.asarray(case_id),
            "pair": np.asarray(pair),
            "mode": np.asarray(mode),
            "resolution": np.asarray(coarse_label),
            "nodes": np.asarray(case.nodes, dtype=np.float64),
            "weights": np.asarray(case.weights, dtype=np.float64),
            "node_type": np.asarray(case.physical_node_type, dtype=np.int64),
            "residual_scale": np.asarray(residual_scale, dtype=np.float64),
        }
        for name, sequence in values.items():
            arrays[name] = np.asarray(sequence)
        relative = Path("visual_payloads") / (
            f"{case_id}__{pair.replace('->', '_to_')}__{mode}.npz"
        )
        np.savez_compressed(output_dir / relative, **arrays)
        hashes[relative.as_posix()] = sha256_file(output_dir / relative)
    return hashes


def _replay_row(
    *,
    case_id: str,
    pair: str,
    call: int,
    coarse: tuple[int, int],
    fine: tuple[int, int],
    free_states: Mapping[tuple[int, int], np.ndarray],
    free_contexts: Mapping[tuple[int, int], Mapping[str, Any]],
    bundle: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    key = _pair_key(coarse, fine)
    step = call - 1
    coarse_state_error = np.max(
        np.abs(free_states[coarse] - bundle[f"free_coarse_states__{key}"][step])
    )
    fine_state_error = np.max(
        np.abs(
            _restrict(free_states[fine], fine=fine, coarse=coarse)
            - bundle[f"free_fine_states__{key}"][step]
        )
    )
    coarse_increment_error = np.max(
        np.abs(
            free_contexts[coarse]["increment"]
            - bundle[f"coarse_increment_free__{key}"][step]
        )
    )
    fine_increment_error = np.max(
        np.abs(
            _restrict(free_contexts[fine]["increment"], fine=fine, coarse=coarse)
            - bundle[f"fine_increment_free__{key}"][step]
        )
    )
    maximum = max(
        float(coarse_state_error),
        float(fine_state_error),
        float(coarse_increment_error),
        float(fine_increment_error),
    )
    return {
        "case_id": case_id,
        "pair": pair,
        "call": call,
        "coarse_state_max_abs": float(coarse_state_error),
        "restricted_fine_state_max_abs": float(fine_state_error),
        "coarse_increment_max_abs": float(coarse_increment_error),
        "restricted_fine_increment_max_abs": float(fine_increment_error),
        "maximum_abs": maximum,
        "tolerance": REPLAY_TOLERANCE,
        "passed": maximum <= REPLAY_TOLERANCE,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = perf_counter()
    device = select_device(args.device)
    store = PCNOEuler2DShardStore(args.data_dir, max_cached_trajectories=2)
    checkpoint = load_resolution_checkpoint(args.checkpoint)
    provenance = _verify_common_provenance(args, checkpoint, store)
    d067_summary, d067_binding = _verify_d067(args, checkpoint)
    d070_summary, d070_compatibility, d070_reference_checks = _verify_d070(
        args,
        checkpoint,
        active_source_sha256=provenance["source_sha256"],
    )
    model, _ = build_resolution_checkpoint_model(checkpoint, device)
    model.eval()
    if len(model.backbone.ws) - 1 != LAYER:
        raise ValueError("D073 freezes zero-indexed final layer 3")
    model_state_before = _model_state_sha256(model)
    cases, reference_checks = _load_dynamic_cases(
        args, checkpoint, store, device=device
    )
    d070_reference_compatibility = _reference_binding_checks(
        d070_reference_checks, reference_checks
    )
    if not d070_reference_compatibility["passed"]:
        raise ValueError(
            "D073 active common-source references differ from accepted D070C: "
            f"{d070_reference_compatibility}"
        )
    grouped: dict[str, dict[tuple[int, int], CaseData]] = defaultdict(dict)
    for case in cases:
        if case.resolution is None:
            raise ValueError("D073 dynamic case lacks a structured resolution")
        grouped[case.case_id][case.resolution] = case
    if set(grouped) != set(CASE_IDS) or any(
        set(grouped[case_id]) != set(RESOLUTIONS) for case_id in CASE_IDS
    ):
        raise ValueError("D073 case-resolution inventory changed")
    (
        _operators,
        prepared,
        geometry_rows,
        operator_rows,
        geometry_checks,
    ) = _prepare_geometry_operators(grouped, device=device)
    hook_equivalence = _hook_equivalence(
        model, grouped[CASE_IDS[0]][(250, 100)], device=device, amp="none"
    )

    args.output_dir.mkdir(parents=True)
    active_case_ids = CASE_IDS[:1] if args.smoke else CASE_IDS
    active_calls = 1 if args.smoke else args.rollout_calls
    diagnostic_calls = (1,) if args.smoke else CALLS
    diagnostic_call_set = set(diagnostic_calls)
    active_reference_checks = [
        row for row in reference_checks if row.get("case_id") in active_case_ids
    ]
    residual_scale = np.asarray(
        checkpoint["normalization"]["residual_scale"], dtype=np.float64
    )
    arm_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    closure_rows: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    completion_rows: list[dict[str, Any]] = []
    visual_payloads: defaultdict[tuple[str, str, str], dict[str, list[Any]]] = (
        defaultdict(dict)
    )
    maximum_raw_recurrence_identity_error = 0.0
    maximum_recurrent_input_mismatch = 0.0

    try:
        for case_index, case_id in enumerate(active_case_ids, start=1):
            case_started = perf_counter()
            by_resolution = grouped[case_id]
            free_states = {
                resolution: np.array(case.reference_states[0], copy=True)
                for resolution, case in by_resolution.items()
            }
            bundle_path = args.d067_summary.parent / "bundles" / f"{case_id}.npz"
            with np.load(bundle_path, allow_pickle=False) as loaded:
                bundle = {
                    name: np.asarray(loaded[name])
                    for name in loaded.files
                    if name != "metadata_json"
                }
            all_admissible = True
            for step in range(active_calls):
                call = step + 1
                free_contexts: dict[tuple[int, int], dict[str, Any]] = {}
                teacher_contexts: dict[tuple[int, int], dict[str, Any]] = {}
                for resolution in RESOLUTIONS:
                    case = by_resolution[resolution]
                    free_contexts[resolution] = _model_context(
                        model, case, free_states[resolution]
                    )
                    context_state = np.asarray(
                        free_contexts[resolution]["state"], dtype=np.float64
                    )
                    prediction = (
                        free_contexts[resolution]["prediction"][0]
                        .detach()
                        .float()
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                    )
                    maximum_recurrent_input_mismatch = max(
                        maximum_recurrent_input_mismatch,
                        float(
                            np.max(
                                np.abs(
                                    context_state
                                    - np.asarray(
                                        free_states[resolution], dtype=np.float64
                                    )
                                )
                            )
                        ),
                    )
                    maximum_raw_recurrence_identity_error = max(
                        maximum_raw_recurrence_identity_error,
                        float(
                            np.max(
                                np.abs(
                                    prediction
                                    - context_state
                                    - free_contexts[resolution]["increment"]
                                )
                            )
                        ),
                    )
                    admissibility = conservative_admissibility_summary(
                        prediction, gamma=case.gamma
                    )
                    all_admissible = all_admissible and bool(
                        admissibility["finite"] and admissibility["admissible"]
                    )
                    if call in diagnostic_call_set:
                        teacher_contexts[resolution] = _model_context(
                            model, case, case.reference_states[step]
                        )

                for coarse, fine in pairwise(RESOLUTIONS):
                    pair = _pair_label(coarse, fine)
                    replay_rows.append(
                        _replay_row(
                            case_id=case_id,
                            pair=pair,
                            call=call,
                            coarse=coarse,
                            fine=fine,
                            free_states=free_states,
                            free_contexts=free_contexts,
                            bundle=bundle,
                        )
                    )
                    if call not in diagnostic_call_set:
                        continue
                    for mode, contexts in (
                        ("teacher_forced", teacher_contexts),
                        ("baseline_free_input", free_contexts),
                    ):
                        fine_input = contexts[fine]["model_input"]
                        trace = trace_same_hidden_physical_radius_outputs(
                            model.backbone,
                            fine_input,
                            _aux(by_resolution[coarse]),
                            _aux(by_resolution[fine]),
                            layer=LAYER,
                            coarse_resolution=coarse,
                            fine_resolution=fine,
                            coarse_local_operator=prepared[coarse]["A1"],
                            fine_local_operator=prepared[fine]["A1"],
                            coarse_fixed_operator=prepared[coarse]["A2"],
                            fine_fixed_operator=prepared[fine]["A2"],
                        )
                        visual = _analyze_trace(
                            case_id=case_id,
                            mode=mode,
                            pair=pair,
                            call=call,
                            resolution=coarse,
                            trace=trace,
                            weights=by_resolution[coarse].weights,
                            residual_scale=residual_scale,
                            arm_rows=arm_rows,
                            component_rows=component_rows,
                            closure_rows=closure_rows,
                        )
                        _append_visual(
                            visual_payloads,
                            case_id=case_id,
                            pair=pair,
                            mode=mode,
                            call=call,
                            physical_time=float(
                                by_resolution[fine].physical_times[call]
                            ),
                            visual=visual,
                        )
                free_states = {
                    resolution: context["prediction"][0]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                    for resolution, context in free_contexts.items()
                }
            completion_rows.append(
                {
                    "case_id": case_id,
                    "calls": active_calls,
                    "all_baseline_outputs_admissible": all_admissible,
                    "seconds": perf_counter() - case_started,
                }
            )
            print(
                f"completed {case_index}/{len(active_case_ids)} {case_id} "
                f"seconds={perf_counter() - case_started:.1f}",
                flush=True,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        store.close()

    model_state_after = _model_state_sha256(model)
    aggregates = _case_aggregates(arm_rows, required_calls=diagnostic_calls)
    gate_rows, raw_gate_decision = _gate_summary(aggregates, smoke=args.smoke)
    pairs = tuple(_pair_label(coarse, fine) for coarse, fine in pairwise(RESOLUTIONS))
    arm_inventory = _inventory_checks(
        arm_rows,
        key_fields=(
            "case_id",
            "mode",
            "pair",
            "call",
            "pathway_level",
            "arm",
            "band",
        ),
        expected_keys={
            (case_id, mode, pair, call, level, arm, band)
            for case_id in active_case_ids
            for mode in MODES
            for pair in pairs
            for call in diagnostic_calls
            for level in PATHWAY_LEVELS
            for arm in ARMS
            for band in BANDS
        },
        finite_fields=(
            "defect_scaled_rms",
            "response_to_a0_scaled_rms",
            "response_to_a1_scaled_rms",
        ),
    )
    component_inventory = _inventory_checks(
        component_rows,
        key_fields=("case_id", "mode", "pair", "call", "arm", "band", "component"),
        expected_keys={
            (case_id, mode, pair, call, arm, band, component)
            for case_id in active_case_ids
            for mode in MODES
            for pair in pairs
            for call in diagnostic_calls
            for arm in ARMS
            for band in BANDS
            for component in range(4)
        },
        finite_fields=("defect_scaled_rms",),
    )
    aggregate_inventory = _inventory_checks(
        aggregates,
        key_fields=("case_id", "mode", "pair", "pathway_level", "arm", "band"),
        expected_keys={
            (case_id, mode, pair, level, arm, band)
            for case_id in active_case_ids
            for mode in MODES
            for pair in pairs
            for level in PATHWAY_LEVELS
            for arm in ARMS
            for band in BANDS
        },
        finite_fields=("call_count", "call_rms_aggregate"),
    )
    replay_inventory = _inventory_checks(
        replay_rows,
        key_fields=("case_id", "pair", "call"),
        expected_keys={
            (case_id, pair, call)
            for case_id in active_case_ids
            for pair in pairs
            for call in range(1, active_calls + 1)
        },
        finite_fields=(
            "coarse_state_max_abs",
            "restricted_fine_state_max_abs",
            "coarse_increment_max_abs",
            "restricted_fine_increment_max_abs",
            "maximum_abs",
        ),
    )
    geometry_inventory = _inventory_checks(
        geometry_rows,
        key_fields=("resolution",),
        expected_keys={(resolution_label(value),) for value in RESOLUTIONS},
        finite_fields=(
            "node_count",
            "directed_edge_count",
            "local_two_hop_radius",
            "fixed_training_radius",
        ),
    )
    operator_inventory = _inventory_checks(
        operator_rows,
        key_fields=("resolution", "arm"),
        expected_keys={
            (resolution_label(value), arm)
            for value in RESOLUTIONS
            for arm in ("A1", "A2")
        },
        finite_fields=(
            "radius",
            "nnz",
            "float64_maximum_row_sum_error",
            "float32_maximum_row_sum_error",
            "float32_maximum_constant_error",
            "changed_row_fraction",
        ),
    )
    completion_inventory = _inventory_checks(
        completion_rows,
        key_fields=("case_id",),
        expected_keys={(case_id,) for case_id in active_case_ids},
        finite_fields=("calls", "seconds"),
    )
    reference_inventory = _inventory_checks(
        active_reference_checks,
        key_fields=("case_id",),
        expected_keys={(case_id,) for case_id in active_case_ids},
        finite_fields=("restriction_crosscheck_max_abs",),
    )
    closure_expected = set()
    for case_id in active_case_ids:
        for mode in MODES:
            for pair in pairs:
                for call in diagnostic_calls:
                    for level in PATHWAY_LEVELS:
                        for arm in ARMS:
                            closure_expected.add(
                                (
                                    case_id,
                                    mode,
                                    pair,
                                    call,
                                    "dct_band_closure",
                                    level,
                                    arm,
                                )
                            )
                    for name in (
                        "a0_coarse_native_reuse",
                        "a0_fine_native_reuse",
                    ):
                        closure_expected.add(
                            (
                                case_id,
                                mode,
                                pair,
                                call,
                                name,
                                "decoded_residual",
                                "A0",
                            )
                        )
                    for level in (
                        "spectral",
                        "pointwise",
                        "gradient",
                        "smoothed_gradient",
                        "post_softsign",
                        "differential",
                        "output",
                    ):
                        closure_expected.add(
                            (
                                case_id,
                                mode,
                                pair,
                                call,
                                "a1_a2_training_grid_output_identity",
                                level,
                                "A1_A2",
                            )
                        )
    closure_inventory = _inventory_checks(
        closure_rows,
        key_fields=("case_id", "mode", "pair", "call", "check", "pathway_level", "arm"),
        expected_keys=closure_expected,
        finite_fields=(),
    )
    inventory_checks = {
        "arm_metrics": arm_inventory,
        "component_metrics": component_inventory,
        "case_aggregates": aggregate_inventory,
        "closure_checks": closure_inventory,
        "replay_metrics": replay_inventory,
        "geometry_inventory": geometry_inventory,
        "operator_checks": operator_inventory,
        "completion": completion_inventory,
        "reference_checks": reference_inventory,
    }
    inventory_pass = all(
        value["complete_and_finite"] for value in inventory_checks.values()
    )
    visual_payload_checks = _visual_payload_inventory(
        visual_payloads,
        active_case_ids=active_case_ids,
        diagnostic_calls=diagnostic_calls,
        grouped=grouped,
    )
    reference_pass = bool(
        d070_reference_compatibility["passed"]
        and len(active_reference_checks) == len(active_case_ids)
        and all(
            float(row["restriction_crosscheck_max_abs"]) <= 2.0e-12
            for row in active_reference_checks
        )
    )
    all_numeric_finite = all(
        _all_numeric_values_finite(rows)
        for rows in (
            geometry_rows,
            operator_rows,
            arm_rows,
            component_rows,
            aggregates,
            gate_rows,
            closure_rows,
            completion_rows,
            active_reference_checks,
            replay_rows,
        )
    )
    active_recurrence_contract = {
        "source_contract": CURRENT_SELF_CONSISTENT_SOURCE_CONTRACT,
        "checkpoint_raw_recurrence": checkpoint.get("raw_recurrence") is True,
        "checkpoint_boundary_mode": checkpoint.get("boundary_mode"),
        "maximum_raw_recurrence_identity_error": (
            maximum_raw_recurrence_identity_error
        ),
        "maximum_recurrent_input_mismatch": maximum_recurrent_input_mismatch,
        "absolute_tolerance": REPLAY_TOLERANCE,
    }
    active_recurrence_contract["passed"] = bool(
        active_recurrence_contract["checkpoint_raw_recurrence"]
        and active_recurrence_contract["checkpoint_boundary_mode"] == "model_all_nodes"
        and maximum_raw_recurrence_identity_error <= REPLAY_TOLERANCE
        and maximum_recurrent_input_mismatch <= REPLAY_TOLERANCE
    )
    checks = {
        "geometry_operator_contract": geometry_checks,
        "hook_equivalence_pass": bool(hook_equivalence["passed"]),
        "model_state_immutable": model_state_before == model_state_after,
        "active_recurrence_contract": active_recurrence_contract,
        "historical_d067_exact_replay_observed": all(
            row["passed"] for row in replay_rows
        ),
        "historical_d067_exact_replay_enters_contract": False,
        "maximum_historical_d067_replay_error": max(
            float(row["maximum_abs"]) for row in replay_rows
        ),
        "closure_checks_pass": all(row["passed"] for row in closure_rows),
        "maximum_dct_closure": max(
            float(row["maximum_error"])
            for row in closure_rows
            if row["check"] == "dct_band_closure"
        ),
        "reference_checks_pass": reference_pass,
        "completion_pass": all(
            row["all_baseline_outputs_admissible"] for row in completion_rows
        ),
        "exact_inventory_checks": inventory_checks,
        "exact_inventory_pass": inventory_pass,
        "visual_payload_inventory": visual_payload_checks,
        "visual_payload_inventory_pass": visual_payload_checks["passed"],
        "all_numeric_outputs_finite": all_numeric_finite,
    }
    contract_pass = all(
        (
            geometry_checks["passed"],
            bool(hook_equivalence["passed"]),
            model_state_before == model_state_after,
            active_recurrence_contract["passed"],
            all(row["passed"] for row in closure_rows),
            reference_pass,
            all(row["all_baseline_outputs_admissible"] for row in completion_rows),
            inventory_pass,
            visual_payload_checks["passed"],
            all_numeric_finite,
        )
    )
    decision = _contract_masked_decision(
        raw_gate_decision, contract_pass=contract_pass, smoke=args.smoke
    )
    status = (
        "smoke_complete"
        if args.smoke and contract_pass
        else (
            "smoke_failed"
            if args.smoke
            else "complete" if contract_pass else "failed_contract"
        )
    )

    csv_outputs: dict[str, Sequence[Mapping[str, Any]]] = {
        "geometry_inventory.csv": geometry_rows,
        "operator_checks.csv": operator_rows,
        "arm_metrics.csv": arm_rows,
        "component_metrics.csv": component_rows,
        "case_aggregates.csv": aggregates,
        "gate_summary.csv": (
            gate_rows
            if gate_rows
            else [
                {
                    "pair": "not_evaluated_in_smoke",
                    "mode": "not_evaluated_in_smoke",
                    "mechanism_pass": None,
                }
            ]
        ),
        "closure_checks.csv": closure_rows,
        "completion.csv": completion_rows,
        "reference_checks.csv": active_reference_checks,
        "replay_metrics.csv": replay_rows,
    }
    for name, rows in csv_outputs.items():
        write_csv_with_paths(args.output_dir / name, rows)
    visual_hashes = _write_visual_payloads(
        args.output_dir,
        visual_payloads,
        grouped=grouped,
        residual_scale=residual_scale,
    )
    output_hashes = {name: sha256_file(args.output_dir / name) for name in csv_outputs}
    output_hashes.update(visual_hashes)
    provenance["loaded_project_source_sha256"] = _loaded_project_source_hashes()
    summary = {
        "schema": SCHEMA,
        "status": status,
        "contract_checks_passed": contract_pass,
        "scientific_interpretation_allowed": contract_pass and not args.smoke,
        "args": jsonable_args(args),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "normalization_digest": checkpoint.get("normalization_digest"),
        "residual_scale": residual_scale.tolist(),
        "population": {
            "split": "validation_open_adaptive",
            "case_ids": list(active_case_ids),
            "sealed_populations_accessed": False,
        },
        "boundary_policy": "model_all_nodes raw recurrence; unchanged",
        "claim_boundary": {
            "adaptive": (
                "D070C selected the pathway and layer on the same open cases; "
                "D073 is exploratory, not confirmatory"
            ),
            "contrast": (
                "A2 versus A1 tests radius sensitivity within one ball-kernel "
                "family; it does not uniquely attribute the native A0 defect"
            ),
            "same_hidden": (
                "teacher and retained baseline-free inputs are same-hidden "
                "representation interventions, not modified rollouts"
            ),
            "resolution": (
                "dynamic FV only under common-source conservative restriction"
            ),
        },
        "contract": {
            "layer": LAYER,
            "arms": list(ARMS),
            "modes": list(MODES),
            "diagnostic_calls": list(diagnostic_calls),
            "bands": list(BANDS),
            "primary_norm": "physical-volume residual-component-scaled RMS",
            "call_aggregation": "RMS over calls within case before case medians",
            "a1_denominator_minimum": A1_DENOMINATOR_MINIMUM,
            "a0_denominator_minimum": A0_DENOMINATOR_MINIMUM,
            "gated_bands": list(GATED_BANDS),
            "large_reduction_minimum": LARGE_REDUCTION_MINIMUM,
            "positive_case_minimum": POSITIVE_CASE_MINIMUM,
            "other_band_ratio_maximum": OTHER_BAND_RATIO_MAXIMUM,
            "a2_a0_large_ratio_maximum": A2_A0_LARGE_RATIO_MAXIMUM,
            "visual_case_ids": list(VISUAL_CASE_IDS),
        },
        "checks": checks,
        "diagnostic_gate_before_contract_mask": raw_gate_decision,
        "decision": decision,
        "hook_equivalence": hook_equivalence,
        "d067_binding": {
            "summary_sha256": sha256_file(args.d067_summary),
            "schema": d067_summary["schema"],
            "status": d067_summary["status"],
            **d067_binding,
        },
        "d070_binding": {
            "summary_sha256": sha256_file(args.d070_summary),
            "schema": d070_summary["schema"],
            "status": d070_summary["status"],
            "mechanism_decision": d070_summary["mechanism_selection"]["decision"],
            "compatibility": d070_compatibility,
            "common_source_reference_compatibility": (d070_reference_compatibility),
        },
        "provenance": provenance,
        "runtime": runtime_environment(device),
        "git": git_state(),
        "row_counts": {name: len(rows) for name, rows in csv_outputs.items()},
        "elapsed_seconds": perf_counter() - started,
        "output_hashes": output_hashes,
    }
    write_json(args.output_dir / "summary.json", summary)
    manifest_outputs = {
        **output_hashes,
        "summary.json": sha256_file(args.output_dir / "summary.json"),
    }
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "result_schema": SCHEMA,
        "status": status,
        "science_result": _manifest_science_result(
            contract_pass=contract_pass, smoke=args.smoke
        ),
        "summary_sha256": manifest_outputs["summary.json"],
        "output_hashes": manifest_outputs,
        "output_count": len(manifest_outputs),
    }
    write_json(args.output_dir / "manifest.json", manifest)
    summary["manifest_sha256"] = sha256_file(args.output_dir / "manifest.json")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    summary = run(parse_args(argv))
    print(
        f"D073 status={summary['status']} "
        f"advance_d073b={summary['decision']['advance_d073b']}",
        flush=True,
    )
    return 0 if summary["status"] in {"complete", "smoke_complete"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
