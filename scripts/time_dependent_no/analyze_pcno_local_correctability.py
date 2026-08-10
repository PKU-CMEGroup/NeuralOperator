"""Audit causal local residual-correction directions on frozen D067 trajectories.

This is D080-A: it never re-evaluates a checkpoint and never claims a corrected
rollout. It scores target-free dissipative directions on stored native 250x100
free-rollout increments, with reference fields used only after construction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from utility.time_dependent_no.pcno_defect_corrections import (
    DissipationResult,
    controlled_graph_dissipation,
    physical_cosine_basis,
)
from utility.time_dependent_no.pcno_fv_geometry import (
    PCNOFiniteVolumeGeometry,
    build_pcno_finite_volume_geometry,
)
from utility.time_dependent_no.pcno_residual_structure import shock_vortex_regions
from utility.time_dependent_no.pcno_ripple_diagnostics import node_highpass_field
from utility.time_dependent_no.shock_vortex_fv import (
    BOUNDARY_TAG_NAMES,
    ShockVortexFVConfig,
    make_structured_fv_geometry,
)

PAIR_KEY = "250x100_to_500x200"
RESOLUTION = (250, 100)
DOMAIN_BOUNDS = (0.0, 2.0, 0.0, 1.0)
RANK = 8
GAMMA = 1.4
SENSOR_QUANTILE = 0.8
CAPS = (0.005, 0.01, 0.02, 0.05)
CASE_IDS = (
    "sv_e00_y00",
    "sv_e00_y08",
    "sv_e06_y00",
    "sv_e06_y08",
    "sv_e11_y00",
    "sv_e11_y08",
)
SUMMARY_SHA256 = "876566975da1a6ab4889cdf31f4593999bab07c38ab12e06f5a926ee1b333245"
CHECKPOINT_SHA256 = "95e6c180a3298c4b38662d8c6cb77301c0e7fd0286e9a53571bddc487b8379c9"
NORMALIZATION_DIGEST = (
    "9df7efb2f1aadf315f399dcabf3b366bf1b2f46794b026cbc5b7d80ba22e1a1a"
)
BUNDLE_SHA256 = {
    "sv_e00_y00": "66d22f83e0ed0035ebff01a9a9bf3480a74f832febbdb94f3fecabf5c7c5d80a",
    "sv_e00_y08": "a0dee720d33e769b44808b753af683795886b5bbfd1e3c7c1c8b0b14a8c50410",
    "sv_e06_y00": "1aaf33649cb2d0443a2e529756cb89032b10a697d6aaa51c8a872748b1c6c407",
    "sv_e06_y08": "29e291cfe051b97827495e1ca8de7c804b2a211279cd55c69418dac4cd1b6784",
    "sv_e11_y00": "eebf69566f2ef9925d75ca1d3c428ae56785b0404209da7fb018e21f35f1db20",
    "sv_e11_y08": "a72016916f92c4cf558057a56d1b354f31a94f05723bcde2010e6cc51971372f",
}
PATHWAYS = (
    "global_isotropic",
    "shock_isotropic",
    "shock_normal",
    "shock_tangential",
    "shock_tangential_coherent",
    "vortex_isotropic",
    "vortex_isotropic_coherent",
)
PARTITIONS = ("total", "parallel", "orthogonal", "excluded_contact")
REGIONS = {
    "boundary": "partition_boundary",
    "shock": "partition_shock",
    "vortex": "partition_vortex",
    "smooth": "partition_smooth",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        raise ValueError("metric inputs must be finite")
    if denominator <= 0.0:
        return None
    return float(numerator / denominator)


def _sqrt_ratio(numerator: float, denominator: float) -> float | None:
    ratio = _safe_ratio(max(float(numerator), 0.0), float(denominator))
    return None if ratio is None else float(np.sqrt(ratio))


def _scaled_energy(
    field: np.ndarray,
    volumes: np.ndarray,
    scale: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    values = np.asarray(field, dtype=np.float64) / scale[None, :]
    mass = volumes if mask is None else volumes[mask]
    values = values if mask is None else values[mask]
    return float(np.einsum("n,nc,nc->", mass, values, values) / volumes.sum())


def _scaled_inner(
    first: np.ndarray,
    second: np.ndarray,
    volumes: np.ndarray,
    scale: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    left = np.asarray(first, dtype=np.float64) / scale[None, :]
    right = np.asarray(second, dtype=np.float64) / scale[None, :]
    mass = volumes if mask is None else volumes[mask]
    left = left if mask is None else left[mask]
    right = right if mask is None else right[mask]
    return float(np.einsum("n,nc,nc->", mass, left, right) / volumes.sum())


@dataclass(frozen=True)
class _SubspaceProjector:
    volumes: np.ndarray
    scale: np.ndarray
    interior: np.ndarray
    square_root_mass: np.ndarray
    q_matrix: np.ndarray

    @classmethod
    def build(
        cls,
        basis: np.ndarray,
        volumes: np.ndarray,
        scale: np.ndarray,
        interior: np.ndarray,
    ) -> _SubspaceProjector:
        square_root_mass = np.sqrt(volumes[interior])
        weighted_basis = square_root_mass[:, None] * basis[interior]
        q_matrix, r_matrix = np.linalg.qr(weighted_basis, mode="reduced")
        if np.linalg.matrix_rank(r_matrix) != basis.shape[1]:
            raise ValueError("rank-eight basis is deficient on type-0 support")
        return cls(volumes, scale, interior, square_root_mass, q_matrix)

    def energies(self, field: np.ndarray) -> dict[str, float]:
        scaled = np.asarray(field, dtype=np.float64) / self.scale[None, :]
        total_mass = float(self.volumes.sum())
        weighted = self.square_root_mass[:, None] * scaled[self.interior]
        coefficients = self.q_matrix.T @ weighted
        total = _scaled_energy(field, self.volumes, self.scale)
        interior = float(np.square(weighted).sum() / total_mass)
        parallel = float(np.square(coefficients).sum() / total_mass)
        orthogonal = interior - parallel
        excluded = total - interior
        tolerance = 1.0e-11 * max(total, 1.0)
        if orthogonal < -tolerance or excluded < -tolerance:
            raise ValueError("weighted subspace energy partition is negative")
        result = {
            "total": total,
            "parallel": max(parallel, 0.0),
            "orthogonal": max(orthogonal, 0.0),
            "excluded_contact": max(excluded, 0.0),
        }
        closure = total - sum(result[name] for name in PARTITIONS[1:])
        if abs(closure) > tolerance:
            raise ValueError("weighted subspace energy partition does not close")
        return result

    def inners(self, first: np.ndarray, second: np.ndarray) -> dict[str, float]:
        left = np.asarray(first, dtype=np.float64) / self.scale[None, :]
        right = np.asarray(second, dtype=np.float64) / self.scale[None, :]
        total_mass = float(self.volumes.sum())
        left_weighted = self.square_root_mass[:, None] * left[self.interior]
        right_weighted = self.square_root_mass[:, None] * right[self.interior]
        left_coefficients = self.q_matrix.T @ left_weighted
        right_coefficients = self.q_matrix.T @ right_weighted
        total = _scaled_inner(first, second, self.volumes, self.scale)
        interior = float(np.sum(left_weighted * right_weighted) / total_mass)
        parallel = float(np.sum(left_coefficients * right_coefficients) / total_mass)
        return {
            "total": total,
            "parallel": parallel,
            "orthogonal": interior - parallel,
            "excluded_contact": total - interior,
        }


def _positive_temporal_coherence(
    current: np.ndarray, previous: np.ndarray | None
) -> np.ndarray:
    if previous is None:
        return np.zeros(current.shape[0], dtype=np.float64)
    numerator = np.einsum("nc,nc->n", current, previous)
    denominator = np.linalg.norm(current, axis=1) * np.linalg.norm(previous, axis=1)
    cosine = np.zeros_like(numerator)
    np.divide(numerator, denominator, out=cosine, where=denominator > 1.0e-30)
    return np.clip(cosine, 0.0, 1.0)


def _shock_edge_multipliers(
    nodes: np.ndarray, edges: np.ndarray, normals: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    left, right = edges.T
    direction = nodes[right] - nodes[left]
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    edge_normal = normals[left] + normals[right]
    edge_normal /= np.linalg.norm(edge_normal, axis=1, keepdims=True)
    normal = np.clip(np.square(np.einsum("ec,ec->e", direction, edge_normal)), 0.0, 1.0)
    return normal, 1.0 - normal


def _cap_scale(result: DissipationResult, cap: float) -> tuple[float, float]:
    raw_relative = result.uncapped_relative_norm
    if raw_relative == 0.0 or result.applied_scale == 0.0:
        return 0.0, 0.0
    desired_scale = min(1.0, float(cap) / raw_relative)
    return float(desired_scale / result.applied_scale), float(
        raw_relative * desired_scale
    )


def _build_geometry() -> PCNOFiniteVolumeGeometry:
    config = ShockVortexFVConfig(coarse_nx=RESOLUTION[0], coarse_ny=RESOLUTION[1])
    finite_volume = make_structured_fv_geometry(config)
    return build_pcno_finite_volume_geometry(
        cell_centers=finite_volume.cell_centers,
        cell_volume=finite_volume.cell_volume,
        face_owner=finite_volume.face_owner,
        face_neighbor=finite_volume.face_neighbor,
        face_boundary_tag=finite_volume.face_boundary_tag,
        boundary_tag_names=BOUNDARY_TAG_NAMES,
    )


def _local_directions(
    update: np.ndarray,
    state: np.ndarray,
    previous_highpass: np.ndarray | None,
    geometry: PCNOFiniteVolumeGeometry,
    scale: np.ndarray,
) -> tuple[dict[str, tuple[DissipationResult, int]], np.ndarray]:
    interior = geometry.node_type == 0
    masks, _, normals = shock_vortex_regions(
        state, geometry.nodes, resolution=RESOLUTION, gamma=GAMMA
    )
    current_highpass = node_highpass_field(update / scale[None, :], geometry.edges)
    coherence = _positive_temporal_coherence(current_highpass, previous_highpass)
    shock = masks["shock_envelope_le_0.05"] & interior
    vortex = masks["partition_vortex"] & interior
    normal_multiplier, tangential_multiplier = _shock_edge_multipliers(
        geometry.nodes, geometry.edges, normals
    )
    specifications: dict[str, tuple[np.ndarray | None, np.ndarray | None]] = {
        "global_isotropic": (None, None),
        "shock_isotropic": (shock.astype(np.float64), None),
        "shock_normal": (shock.astype(np.float64), normal_multiplier),
        "shock_tangential": (shock.astype(np.float64), tangential_multiplier),
        "shock_tangential_coherent": (
            shock.astype(np.float64) * coherence,
            tangential_multiplier,
        ),
        "vortex_isotropic": (vortex.astype(np.float64), None),
        "vortex_isotropic_coherent": (vortex.astype(np.float64) * coherence, None),
    }
    results: dict[str, tuple[DissipationResult, int]] = {}
    for pathway, (gate, multiplier) in specifications.items():
        result = controlled_graph_dissipation(
            update,
            geometry.edges,
            geometry.node_weights[:, 0],
            interior,
            component_scale=scale,
            sensor_quantile=SENSOR_QUANTILE,
            norm_cap=1.0,
            node_gate=gate,
            edge_multiplier=multiplier,
            edges_are_unique_undirected=True,
        )
        active = int(interior.sum()) if gate is None else int(np.count_nonzero(gate))
        results[pathway] = (result, active)
    return results, current_highpass


def _quadratic_energy(
    base: float, inner: float, correction: float, factor: float
) -> float:
    value = base + 2.0 * factor * inner + factor * factor * correction
    tolerance = 1.0e-11 * max(base, correction, 1.0)
    if value < -tolerance:
        raise ValueError("quadratic corrected energy is negative")
    return max(float(value), 0.0)


def _new_accumulator(shape: tuple[int, int]) -> dict[str, Any]:
    return {
        "sum_corrected_energy": 0.0,
        "sum_corrected_norm": 0.0,
        "sum_correction_energy": 0.0,
        "sum_defect_correction_inner": 0.0,
        "cumulative_defect": np.zeros(shape, dtype=np.float64),
        "partition_corrected": {name: 0.0 for name in PARTITIONS},
        "partition_correction": {name: 0.0 for name in PARTITIONS},
        "partition_inner": {name: 0.0 for name in PARTITIONS},
        "region_corrected": {name: 0.0 for name in REGIONS},
        "maximum_mean_closure_scaled": 0.0,
        "maximum_support_leakage": 0.0,
        "maximum_applied_relative_norm": 0.0,
    }


def _arm_name(pathway: str, cap: float) -> str:
    return f"{pathway}__cap{str(cap).replace('.', 'p')}"


def _validate_case_payload(
    case_id: str,
    bundle_path: Path,
    geometry: PCNOFiniteVolumeGeometry,
    expected_scale: np.ndarray,
) -> dict[str, np.ndarray]:
    with np.load(bundle_path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        if (
            metadata["case_id"] != case_id
            or metadata["schema"] != "pcno_residual_structure_diagnostic_v1"
        ):
            raise ValueError(f"{case_id}: bundle metadata identity mismatch")
        if metadata["checkpoint_sha256"] != CHECKPOINT_SHA256:
            raise ValueError(f"{case_id}: checkpoint mismatch")
        if metadata["normalization_digest"] != NORMALIZATION_DIGEST:
            raise ValueError(f"{case_id}: normalizer mismatch")
        if metadata["boundary_policy"] != "model_all_nodes raw recurrence":
            raise ValueError(f"{case_id}: boundary policy mismatch")
        if metadata["rollout_calls"] != 30 or metadata["physical_dt"] != 0.02:
            raise ValueError(f"{case_id}: recurrence contract mismatch")
        arrays = {
            "nodes": np.asarray(payload[f"nodes__{PAIR_KEY}"], dtype=np.float64),
            "volumes": np.asarray(
                payload[f"volumes__{PAIR_KEY}"], dtype=np.float64
            ).reshape(-1),
            "scale": np.asarray(payload["residual_scale"], dtype=np.float64),
            "times": np.asarray(payload["physical_times"], dtype=np.float64),
            "reference": np.asarray(
                payload[f"reference_states__{PAIR_KEY}"], dtype=np.float64
            ),
            "free_states": np.asarray(
                payload[f"free_coarse_states__{PAIR_KEY}"], dtype=np.float64
            ),
            "predicted": np.asarray(
                payload[f"coarse_increment_free__{PAIR_KEY}"], dtype=np.float64
            ),
            "truth": np.asarray(
                payload[f"true_increment__{PAIR_KEY}"], dtype=np.float64
            ),
        }
    expected_shape = (30, RESOLUTION[0] * RESOLUTION[1], 4)
    if (
        arrays["predicted"].shape != expected_shape
        or arrays["truth"].shape != expected_shape
    ):
        raise ValueError(f"{case_id}: increment shape mismatch")
    if (
        arrays["reference"].shape != (31, *expected_shape[1:])
        or arrays["free_states"].shape != arrays["reference"].shape
    ):
        raise ValueError(f"{case_id}: state shape mismatch")
    if not np.array_equal(arrays["nodes"], geometry.nodes):
        raise ValueError(f"{case_id}: native node geometry mismatch")
    if not np.array_equal(arrays["volumes"], geometry.node_measures[:, 0]):
        raise ValueError(f"{case_id}: native physical volumes mismatch")
    if not np.array_equal(arrays["scale"], expected_scale):
        raise ValueError(f"{case_id}: residual scale mismatch")
    if not np.array_equal(arrays["times"], 0.02 * np.arange(1, 31, dtype=np.float64)):
        raise ValueError(f"{case_id}: physical time mismatch")
    if not np.array_equal(arrays["predicted"], np.diff(arrays["free_states"], axis=0)):
        raise ValueError(f"{case_id}: predicted-increment replay mismatch")
    if not np.array_equal(arrays["truth"], np.diff(arrays["reference"], axis=0)):
        raise ValueError(f"{case_id}: true-increment replay mismatch")
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError(f"{case_id}: nonfinite stored trajectory")
    return arrays


def _analyze_case(
    case_id: str,
    bundle_path: Path,
    geometry: PCNOFiniteVolumeGeometry,
    expected_scale: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    arrays = _validate_case_payload(case_id, bundle_path, geometry, expected_scale)
    volumes = arrays["volumes"]
    scale = arrays["scale"]
    times = arrays["times"]
    reference = arrays["reference"]
    free_states = arrays["free_states"]
    predicted = arrays["predicted"]
    truth = arrays["truth"]
    expected_shape = predicted.shape
    interior = geometry.node_type == 0
    basis, modes = physical_cosine_basis(
        geometry.nodes, rank=RANK, domain_bounds=DOMAIN_BOUNDS
    )
    projector = _SubspaceProjector.build(basis, volumes, scale, interior)
    arm_contracts = [("zero", "zero", 0.0)] + [
        (_arm_name(pathway, cap), pathway, cap) for pathway in PATHWAYS for cap in CAPS
    ]
    accumulators = {
        arm: _new_accumulator(expected_shape[1:]) for arm, _, _ in arm_contracts
    }
    call_rows: list[dict[str, Any]] = []
    baseline_partition = {name: 0.0 for name in PARTITIONS}
    truth_partition = {name: 0.0 for name in PARTITIONS}
    baseline_region = {name: 0.0 for name in REGIONS}
    truth_region = {name: 0.0 for name in REGIONS}
    cumulative_baseline = np.zeros(expected_shape[1:], dtype=np.float64)
    cumulative_truth = np.zeros(expected_shape[1:], dtype=np.float64)
    sum_baseline_energy = 0.0
    sum_truth_energy = 0.0
    previous_highpass: np.ndarray | None = None

    for call in range(30):
        defect = predicted[call] - truth[call]
        defect_partition = projector.energies(defect)
        truth_call_partition = projector.energies(truth[call])
        reference_masks, _, _ = shock_vortex_regions(
            reference[call], geometry.nodes, resolution=RESOLUTION, gamma=GAMMA
        )
        defect_region = {
            name: _scaled_energy(defect, volumes, scale, reference_masks[mask_name])
            for name, mask_name in REGIONS.items()
        }
        truth_call_region = {
            name: _scaled_energy(
                truth[call], volumes, scale, reference_masks[mask_name]
            )
            for name, mask_name in REGIONS.items()
        }
        sum_baseline_energy += defect_partition["total"]
        sum_truth_energy += truth_call_partition["total"]
        cumulative_baseline += defect
        cumulative_truth += truth[call]
        for name in PARTITIONS:
            baseline_partition[name] += defect_partition[name]
            truth_partition[name] += truth_call_partition[name]
        for name in REGIONS:
            baseline_region[name] += defect_region[name]
            truth_region[name] += truth_call_region[name]

        local_results, current_highpass = _local_directions(
            predicted[call],
            free_states[call],
            previous_highpass,
            geometry,
            scale,
        )
        previous_highpass = current_highpass

        def record(
            arm: str,
            pathway: str,
            cap: float,
            correction: np.ndarray,
            factor: float,
            applied_relative: float,
            eligible_edges: int,
            active_nodes: int,
            correction_partition: dict[str, float],
            defect_correction_inner: dict[str, float],
            correction_region: dict[str, float],
            defect_correction_region_inner: dict[str, float],
            closure: np.ndarray,
            bound_call: int = call,
            bound_defect: np.ndarray = defect,
            bound_defect_partition: dict[str, float] = defect_partition,
            bound_truth_partition: dict[str, float] = truth_call_partition,
            bound_defect_region: dict[str, float] = defect_region,
            bound_truth_region: dict[str, float] = truth_call_region,
        ) -> None:
            corrected_partition = {
                name: _quadratic_energy(
                    bound_defect_partition[name],
                    defect_correction_inner[name],
                    correction_partition[name],
                    factor,
                )
                for name in PARTITIONS
            }
            corrected_region = {
                name: _quadratic_energy(
                    bound_defect_region[name],
                    defect_correction_region_inner[name],
                    correction_region[name],
                    factor,
                )
                for name in REGIONS
            }
            correction_total = factor * factor * correction_partition["total"]
            inner_total = factor * defect_correction_inner["total"]
            cosine = _safe_ratio(
                -inner_total,
                float(np.sqrt(bound_defect_partition["total"] * correction_total)),
            )
            row: dict[str, Any] = {
                "case_id": case_id,
                "call": bound_call + 1,
                "physical_time": float(times[bound_call]),
                "arm": arm,
                "pathway": pathway,
                "norm_cap": cap,
                "corrected_defect_norm": float(np.sqrt(corrected_partition["total"])),
                "baseline_defect_norm": float(np.sqrt(bound_defect_partition["total"])),
                "true_increment_norm": float(np.sqrt(bound_truth_partition["total"])),
                "q_res": _sqrt_ratio(
                    corrected_partition["total"], bound_truth_partition["total"]
                ),
                "rho": _sqrt_ratio(
                    corrected_partition["total"], bound_defect_partition["total"]
                ),
                "correction_to_negative_defect_cosine": cosine,
                "relative_energy_growth": _safe_ratio(
                    corrected_partition["total"] - bound_defect_partition["total"],
                    bound_defect_partition["total"],
                ),
                "applied_relative_norm": applied_relative,
                "eligible_edge_count": eligible_edges,
                "active_node_count": active_nodes,
                "maximum_mean_closure_scaled": float(
                    np.max(np.abs(factor * closure / scale))
                ),
                "maximum_non_type0_correction": float(
                    np.max(np.abs(factor * correction[~interior]))
                ),
            }
            for name in PARTITIONS:
                part_correction = factor * factor * correction_partition[name]
                part_inner = factor * defect_correction_inner[name]
                row[f"{name}_q_res"] = _sqrt_ratio(
                    corrected_partition[name], bound_truth_partition[name]
                )
                row[f"{name}_rho"] = _sqrt_ratio(
                    corrected_partition[name], bound_defect_partition[name]
                )
                row[f"{name}_cosine"] = _safe_ratio(
                    -part_inner,
                    float(np.sqrt(bound_defect_partition[name] * part_correction)),
                )
                row[f"{name}_relative_energy_growth"] = _safe_ratio(
                    corrected_partition[name] - bound_defect_partition[name],
                    bound_defect_partition[name],
                )
            for name in REGIONS:
                row[f"{name}_q_res"] = _sqrt_ratio(
                    corrected_region[name], bound_truth_region[name]
                )
                row[f"{name}_rho"] = _sqrt_ratio(
                    corrected_region[name], bound_defect_region[name]
                )
            call_rows.append(row)

            accumulator = accumulators[arm]
            accumulator["sum_corrected_energy"] += corrected_partition["total"]
            accumulator["sum_corrected_norm"] += float(
                np.sqrt(corrected_partition["total"])
            )
            accumulator["sum_correction_energy"] += correction_total
            accumulator["sum_defect_correction_inner"] += inner_total
            accumulator["cumulative_defect"] += bound_defect + factor * correction
            for name in PARTITIONS:
                accumulator["partition_corrected"][name] += corrected_partition[name]
                accumulator["partition_correction"][name] += (
                    factor * factor * correction_partition[name]
                )
                accumulator["partition_inner"][name] += (
                    factor * defect_correction_inner[name]
                )
            for name in REGIONS:
                accumulator["region_corrected"][name] += corrected_region[name]
            accumulator["maximum_mean_closure_scaled"] = max(
                accumulator["maximum_mean_closure_scaled"],
                row["maximum_mean_closure_scaled"],
            )
            accumulator["maximum_support_leakage"] = max(
                accumulator["maximum_support_leakage"],
                row["maximum_non_type0_correction"],
            )
            accumulator["maximum_applied_relative_norm"] = max(
                accumulator["maximum_applied_relative_norm"], applied_relative
            )

        zero_partition = {name: 0.0 for name in PARTITIONS}
        zero_region = {name: 0.0 for name in REGIONS}
        record(
            "zero",
            "zero",
            0.0,
            np.zeros_like(defect),
            0.0,
            0.0,
            0,
            0,
            zero_partition,
            zero_partition,
            zero_region,
            zero_region,
            np.zeros(4),
        )
        for pathway in PATHWAYS:
            result, active_nodes = local_results[pathway]
            correction_partition = projector.energies(result.correction)
            defect_correction_inner = projector.inners(defect, result.correction)
            correction_region = {
                name: _scaled_energy(
                    result.correction,
                    volumes,
                    scale,
                    reference_masks[mask_name],
                )
                for name, mask_name in REGIONS.items()
            }
            defect_correction_region_inner = {
                name: _scaled_inner(
                    defect,
                    result.correction,
                    volumes,
                    scale,
                    reference_masks[mask_name],
                )
                for name, mask_name in REGIONS.items()
            }
            for cap in CAPS:
                factor, applied_relative = _cap_scale(result, cap)
                record(
                    _arm_name(pathway, cap),
                    pathway,
                    cap,
                    result.correction,
                    factor,
                    applied_relative,
                    result.eligible_edge_count,
                    active_nodes,
                    correction_partition,
                    defect_correction_inner,
                    correction_region,
                    defect_correction_region_inner,
                    result.weighted_mean_closure,
                )

    cumulative_baseline_energy = _scaled_energy(cumulative_baseline, volumes, scale)
    cumulative_truth_energy = _scaled_energy(cumulative_truth, volumes, scale)
    case_rows: list[dict[str, Any]] = []
    contract_by_arm = {arm: (pathway, cap) for arm, pathway, cap in arm_contracts}
    for arm, accumulator in accumulators.items():
        pathway, cap = contract_by_arm[arm]
        cumulative_corrected_energy = _scaled_energy(
            accumulator["cumulative_defect"], volumes, scale
        )
        row = {
            "case_id": case_id,
            "arm": arm,
            "pathway": pathway,
            "norm_cap": cap,
            "corrected_defect_rms_numerator": float(
                np.sqrt(accumulator["sum_corrected_energy"])
            ),
            "baseline_defect_rms_denominator": float(np.sqrt(sum_baseline_energy)),
            "true_residual_rms_denominator": float(np.sqrt(sum_truth_energy)),
            "aggregate_residual_scale_error": _sqrt_ratio(
                accumulator["sum_corrected_energy"], sum_truth_energy
            ),
            "aggregate_defect_ratio": _sqrt_ratio(
                accumulator["sum_corrected_energy"], sum_baseline_energy
            ),
            "cumulative_corrected_defect_norm": float(
                np.sqrt(cumulative_corrected_energy)
            ),
            "cumulative_baseline_defect_norm": float(
                np.sqrt(cumulative_baseline_energy)
            ),
            "cumulative_true_change_norm": float(np.sqrt(cumulative_truth_energy)),
            "cumulative_residual_scale_error": _sqrt_ratio(
                cumulative_corrected_energy, cumulative_truth_energy
            ),
            "cumulative_defect_ratio": _sqrt_ratio(
                cumulative_corrected_energy, cumulative_baseline_energy
            ),
            "temporal_coherence": _safe_ratio(
                float(np.sqrt(cumulative_corrected_energy)),
                accumulator["sum_corrected_norm"],
            ),
            "correction_to_negative_defect_cosine": _safe_ratio(
                -accumulator["sum_defect_correction_inner"],
                float(
                    np.sqrt(sum_baseline_energy * accumulator["sum_correction_energy"])
                ),
            ),
            "maximum_mean_closure_scaled": accumulator["maximum_mean_closure_scaled"],
            "maximum_non_type0_correction": accumulator["maximum_support_leakage"],
            "maximum_applied_relative_norm": accumulator[
                "maximum_applied_relative_norm"
            ],
        }
        for name in PARTITIONS:
            row[f"{name}_aggregate_defect_ratio"] = _sqrt_ratio(
                accumulator["partition_corrected"][name],
                baseline_partition[name],
            )
            row[f"{name}_aggregate_residual_scale_error"] = _sqrt_ratio(
                accumulator["partition_corrected"][name], truth_partition[name]
            )
        for name in REGIONS:
            row[f"{name}_aggregate_defect_ratio"] = _sqrt_ratio(
                accumulator["region_corrected"][name], baseline_region[name]
            )
            row[f"{name}_aggregate_residual_scale_error"] = _sqrt_ratio(
                accumulator["region_corrected"][name], truth_region[name]
            )
        case_rows.append(row)

    checks = {
        "case_id": case_id,
        "bundle_sha256": BUNDLE_SHA256[case_id],
        "calls": 30,
        "node_count": int(geometry.nodes.shape[0]),
        "edge_count": int(geometry.edges.shape[0]),
        "type0_node_count": int(interior.sum()),
        "cosine_modes": [list(mode) for mode in modes],
        "maximum_mean_closure_scaled": max(
            row["maximum_mean_closure_scaled"] for row in case_rows
        ),
        "maximum_non_type0_correction": max(
            row["maximum_non_type0_correction"] for row in case_rows
        ),
        "maximum_cap_excess": max(
            max(
                0.0,
                row["maximum_applied_relative_norm"] - row["norm_cap"],
            )
            for row in case_rows
        ),
    }
    if checks["maximum_mean_closure_scaled"] > 1.0e-12:
        raise ValueError(f"{case_id}: weighted-mean closure failed")
    if checks["maximum_non_type0_correction"] != 0.0:
        raise ValueError(f"{case_id}: correction touched a contact node")
    if checks["maximum_cap_excess"] > 1.0e-12:
        raise ValueError(f"{case_id}: correction cap failed")
    return call_rows, case_rows, checks


def _routing_summary(case_rows: list[dict[str, Any]]) -> dict[str, Any]:
    arms = sorted({row["arm"] for row in case_rows})
    aggregates = []
    for arm in arms:
        rows = [row for row in case_rows if row["arm"] == arm]
        if len(rows) != len(CASE_IDS):
            raise ValueError(f"arm {arm} does not contain six case-first rows")
        sample = rows[0]

        def median(key: str, bound_rows: list[dict[str, Any]] = rows) -> float:
            return float(np.median([float(row[key]) for row in bound_rows]))

        aggregate = {
            "arm": arm,
            "pathway": sample["pathway"],
            "norm_cap": sample["norm_cap"],
            "median_aggregate_defect_ratio": median("aggregate_defect_ratio"),
            "median_cumulative_defect_ratio": median("cumulative_defect_ratio"),
            "median_shock_defect_ratio": median("shock_aggregate_defect_ratio"),
            "median_vortex_defect_ratio": median("vortex_aggregate_defect_ratio"),
            "positive_cosine_case_count": sum(
                row["correction_to_negative_defect_cosine"] is not None
                and row["correction_to_negative_defect_cosine"] > 0.0
                for row in rows
            ),
        }
        aggregate["signals"] = {
            "median_aggregate_defect_improves": (
                aggregate["median_aggregate_defect_ratio"] < 1.0
            ),
            "median_cumulative_defect_improves": (
                aggregate["median_cumulative_defect_ratio"] < 1.0
            ),
            "localized_improvement": (
                min(
                    aggregate["median_shock_defect_ratio"],
                    aggregate["median_vortex_defect_ratio"],
                )
                < 0.98
                and aggregate["median_aggregate_defect_ratio"] <= 1.10
            ),
            "subset_directional_signal": (
                aggregate["positive_cosine_case_count"] >= 2
                and aggregate["median_aggregate_defect_ratio"] <= 1.20
            ),
        }
        aggregate["routing_signal"] = any(aggregate["signals"].values())
        aggregates.append(aggregate)
    candidates = [
        row
        for row in aggregates
        if row["pathway"] not in {"zero", "global_isotropic"} and row["routing_signal"]
    ]
    candidates.sort(
        key=lambda row: (
            min(
                row["median_aggregate_defect_ratio"],
                row["median_cumulative_defect_ratio"],
                row["median_shock_defect_ratio"],
                row["median_vortex_defect_ratio"],
            ),
            row["median_aggregate_defect_ratio"],
            row["norm_cap"],
            row["arm"],
        )
    )
    return {
        "routing_authorized": bool(candidates),
        "eligible_candidates_ranked": [row["arm"] for row in candidates],
        "arm_aggregates": aggregates,
    }


def _write_csv(
    path: Path, rows: list[dict[str, Any]], leading: tuple[str, ...]
) -> None:
    fields = list(leading) + sorted(
        set().union(*(row.keys() for row in rows)) - set(leading)
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(
            "artifacts/time_dependent_no/"
            "pcno_cumulative_pathway_structure_d067_20260802a/"
            "results_d067_unified_r1d_20260803a"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "artifacts/time_dependent_no/"
            "pcno_local_correctability_d080_20260805a/"
            "d080a_d067_native_sixcase_r1"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    started = time.perf_counter()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    summary_path = input_dir / "summary.json"
    if _sha256(summary_path) != SUMMARY_SHA256:
        raise ValueError("D067 input summary SHA-256 mismatch")
    input_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        input_summary["status"] != "complete"
        or input_summary["population"]["sealed_populations_accessed"]
    ):
        raise ValueError("D067 input status or sealed-population contract mismatch")
    if tuple(input_summary["population"]["case_ids"]) != CASE_IDS:
        raise ValueError("D067 case population mismatch")
    if input_summary["checkpoint_sha256"] != CHECKPOINT_SHA256:
        raise ValueError("D067 checkpoint mismatch")
    if input_summary["normalization_digest"] != NORMALIZATION_DIGEST:
        raise ValueError("D067 normalizer mismatch")
    expected_scale = np.asarray(input_summary["residual_scale"], dtype=np.float64)
    for case_id, expected_hash in BUNDLE_SHA256.items():
        path = input_dir / "bundles" / f"{case_id}.npz"
        if _sha256(path) != expected_hash:
            raise ValueError(f"{case_id}: D067 bundle SHA-256 mismatch")
        if input_summary["output_hashes"][f"bundles/{case_id}.npz"] != expected_hash:
            raise ValueError(f"{case_id}: summary bundle binding mismatch")
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing output directory: {output_dir}"
        )

    geometry = _build_geometry()
    all_call_rows: list[dict[str, Any]] = []
    all_case_rows: list[dict[str, Any]] = []
    case_checks = []
    for case_id in CASE_IDS:
        call_rows, case_rows, checks = _analyze_case(
            case_id,
            input_dir / "bundles" / f"{case_id}.npz",
            geometry,
            expected_scale,
        )
        all_call_rows.extend(call_rows)
        all_case_rows.extend(case_rows)
        case_checks.append(checks)

    routing = _routing_summary(all_case_rows)
    output_dir.mkdir(parents=True)
    call_path = output_dir / "per_call_metrics.csv"
    case_path = output_dir / "per_case_metrics.csv"
    _write_csv(
        call_path,
        all_call_rows,
        ("case_id", "call", "physical_time", "arm", "pathway", "norm_cap"),
    )
    _write_csv(
        case_path,
        all_case_rows,
        ("case_id", "arm", "pathway", "norm_cap"),
    )
    output_hashes = {
        call_path.name: _sha256(call_path),
        case_path.name: _sha256(case_path),
    }
    repository_root = Path(__file__).resolve().parents[2]
    result = {
        "schema": "pcno_local_correctability_d080a_v1",
        "status": "complete",
        "interpretation": (
            "frozen_trajectory_directional_headroom_not_corrected_rollout"
        ),
        "input": {
            "summary_path": str(summary_path),
            "summary_sha256": SUMMARY_SHA256,
            "bundle_sha256": BUNDLE_SHA256,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "normalization_digest": NORMALIZATION_DIGEST,
            "case_ids": list(CASE_IDS),
            "sealed_populations_accessed": False,
        },
        "contract": {
            "pair_key": PAIR_KEY,
            "resolution": list(RESOLUTION),
            "calls": 30,
            "physical_dt": 0.02,
            "boundary_policy": (
                "model_all_nodes raw recurrence; correction type-0-to-type-0 only"
            ),
            "dynamic_node_type_meanings": {
                "0": "interior",
                "1": "y-symmetry contact",
                "2": "x-extrapolation contact",
                "3": "contact with both",
            },
            "component_scale": expected_scale.tolist(),
            "rank": RANK,
            "caps": list(CAPS),
            "pathways": list(PATHWAYS),
            "sensor_quantile": SENSOR_QUANTILE,
            "case_first_aggregation": True,
        },
        "case_checks": case_checks,
        "routing": routing,
        "row_counts": {
            "per_call": len(all_call_rows),
            "per_case": len(all_case_rows),
        },
        "source_hashes": {
            "scripts/time_dependent_no/analyze_pcno_local_correctability.py": (
                _sha256(Path(__file__).resolve())
            ),
            "utility/time_dependent_no/pcno_defect_corrections.py": _sha256(
                repository_root / "utility/time_dependent_no/pcno_defect_corrections.py"
            ),
        },
        "output_hashes": output_hashes,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "routing": routing,
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
