#!/usr/bin/env python3
"""Evaluate one frozen dynamic PCNO through multiresolution raw rollouts.

The primary initialization is sampled once as conservative cell averages on a
common fine grid and then block-averaged to every model grid. With an optional
multiresolution reference root, evolved targets are likewise exact conservative
restrictions of one common high-fidelity evolution, including a grid finer than
the checkpoint's training grid.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utility.time_dependent_no.euler2d_metrics import shock_centroid
from utility.time_dependent_no.pcno_artifacts import (
    atomic_write_json_with_paths as atomic_write_json,
    git_head as _git_head,
    sha256_file,
    sha256_files,
    write_csv_with_paths as write_csv,
)
from utility.time_dependent_no.pcno_euler2d import (
    PCNOEuler2DResidual,
    PCNOEuler2DShardStore,
    parameter_count,
)
from utility.time_dependent_no.pcno_resolution_transfer import (
    NODE_TYPE_PROTOCOLS,
    PHYSICAL_WAVELENGTH_BANDS,
    Resolution,
    aggregate_resolution_rollout,
    as_model_state,
    build_resolution_checkpoint_model,
    build_resolution_geometry,
    commutator_metrics,
    commutator_row,
    conservative_admissibility_summary,
    fixed_boundary_distance_mask,
    initial_state_for_model_grid,
    initial_states_from_common_source,
    load_resolution_checkpoint,
    load_resolution_reference,
    make_model_sample,
    native_geometry_audit,
    node_type_scaling_summary,
    node_types_for_protocol,
    parse_resolution,
    physical_wavelength_band_metrics,
    predict_resolution_sample,
    pressure_profile_shock_metrics,
    reference_at_resolution,
    resolution_label,
    restrict_nested_state,
    validate_resolution_rollout_contract,
    weighted_scaled_relative_l2,
    weighted_scaled_rms,
)
from utility.time_dependent_no.pcno_ripple_diagnostics import (
    conservative_to_primitive_raw,
)
from utility.time_dependent_no.pcno_runtime import (
    checkpoint_model_node_type_input,
    select_device,
)
from utility.time_dependent_no.shock_vortex_family import (
    config_for_family_case,
    family_case_provenance,
    load_shock_vortex_family_manifest,
)
from utility.time_dependent_no.shock_vortex_metrics import (
    endpoint_metrics,
    physical_call_metrics,
)

SCHEMA = "pcno_resolution_rollout_r1_v2"
DEFAULT_CASE_IDS = (
    "sv_e00_y04",
    "sv_e06_y04",
    "sv_e11_y04",
    "sv_e00_y00",
    "sv_e06_y00",
    "sv_e11_y00",
)
DEFAULT_RESOLUTIONS = ("125x50", "250x100", "500x200")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family-root", type=Path, required=True)
    parser.add_argument(
        "--multires-reference-root",
        type=Path,
        help=(
            "Optional case-root containing finer retained references from one "
            "common high-fidelity evolution."
        ),
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-ids", nargs="+", default=list(DEFAULT_CASE_IDS))
    parser.add_argument(
        "--allowed-splits",
        nargs="+",
        choices=("train", "validation"),
        default=("train", "validation"),
    )
    parser.add_argument("--resolutions", nargs="+", default=list(DEFAULT_RESOLUTIONS))
    parser.add_argument("--source-resolution", default="1000x400")
    parser.add_argument("--training-resolution", default="250x100")
    parser.add_argument(
        "--protocols",
        nargs="+",
        choices=NODE_TYPE_PROTOCOLS,
        default=("physical",),
    )
    parser.add_argument("--rollout-calls", type=int)
    parser.add_argument(
        "--endpoint-physical-times", type=float, nargs="+", default=(0.1, 0.3, 0.6)
    )
    parser.add_argument(
        "--boundary-band-widths", type=float, nargs="+", default=(0.02, 0.05)
    )
    parser.add_argument("--quadrature-order", type=int)
    parser.add_argument("--shock-quantile", type=float, default=0.9)
    parser.add_argument("--repeat-forward", type=int, default=2)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--expected-normalization-digest")
    parser.add_argument("--expected-step-stride", type=int)
    parser.add_argument("--expected-k-max", type=int, default=8)
    parser.add_argument(
        "--expected-domain-lengths", type=float, nargs=2, default=(2.0, 1.0)
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", choices=("none", "bf16", "fp16"), default="none")
    return parser.parse_args(argv)


def _source_hashes() -> dict[str, str]:
    paths = (
        Path("pcno/pcno.py"),
        Path("utility/time_dependent_no/euler2d_metrics.py"),
        Path("utility/time_dependent_no/pcno_artifacts.py"),
        Path("utility/time_dependent_no/pcno_euler2d.py"),
        Path("utility/time_dependent_no/pcno_runtime.py"),
        Path("utility/time_dependent_no/pcno_fv_geometry.py"),
        Path("utility/time_dependent_no/pcno_resolution_transfer.py"),
        Path("utility/time_dependent_no/pcno_ripple_diagnostics.py"),
        Path("utility/time_dependent_no/shock_vortex_coarse_cfd.py"),
        Path("utility/time_dependent_no/shock_vortex_family.py"),
        Path("utility/time_dependent_no/shock_vortex_fv.py"),
        Path("utility/time_dependent_no/shock_vortex_metrics.py"),
        Path(
            "scripts/time_dependent_no/generate_pcno_shock_vortex_multires_reference.py"
        ),
        Path("scripts/time_dependent_no/evaluate_pcno_resolution_rollout.py"),
    )
    return sha256_files(paths, root=ROOT)


def _expected_vortex_center(
    manifest: Mapping[str, Any],
    store: PCNOEuler2DShardStore,
    case_id: str,
    physical_time: float,
) -> tuple[float, float]:
    config = manifest["reference_config"]
    upstream_speed = float(config["shock_mach"]) * math.sqrt(float(config["gamma"]))
    arrival = (float(config["shock_x"]) - float(config["vortex_x"])) / upstream_speed
    expected_x = (
        float(config["vortex_x"]) + upstream_speed * physical_time
        if physical_time <= arrival
        else float(config["shock_x"])
        + float(config["right_u"]) * (physical_time - arrival)
    )
    return expected_x, float(store.entry(case_id)["parameters"]["vortex_y"])


def _translation_rows(
    case_id: str,
    common_states: Mapping[Resolution, np.ndarray],
    direct_states: Mapping[Resolution, np.ndarray],
    reference: Mapping[str, np.ndarray],
    *,
    resolutions: Sequence[Resolution],
    reference_resolution: Resolution,
    geometry_by_resolution: Mapping[Resolution, Any],
    state_scale: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for resolution in resolutions:
        common = np.asarray(common_states[resolution], dtype=np.float64)
        direct = np.asarray(direct_states[resolution], dtype=np.float64)
        geometry = geometry_by_resolution[resolution]
        delta = direct - common
        target = reference_at_resolution(
            reference["conservative_states"][0],
            reference_resolution=reference_resolution,
            target_resolution=resolution,
        )
        row = {
            "case_id": case_id,
            "resolution": resolution_label(resolution),
            "direct_vs_common_max_abs": float(np.max(np.abs(delta))),
            "direct_vs_common_scaled_rms": weighted_scaled_rms(
                delta,
                volumes=geometry.node_measures,
                component_scale=state_scale,
            ),
            "reference_available": target is not None,
        }
        if target is not None:
            reference_delta = as_model_state(common) - target
            row.update(
                {
                    "model_input_vs_reference_t0_max_abs": float(
                        np.max(np.abs(reference_delta))
                    ),
                    "model_input_vs_reference_t0_scaled_rms": weighted_scaled_rms(
                        reference_delta,
                        volumes=geometry.node_measures,
                        component_scale=state_scale,
                    ),
                }
            )
        rows.append(row)
    return rows


def _state_row(
    *,
    case_id: str,
    case_split: str,
    protocol: str,
    resolution: Resolution,
    call: int,
    physical_time: float,
    state: np.ndarray,
    reference_state: np.ndarray | None,
    volumes: np.ndarray,
    state_scale: np.ndarray,
    gamma: float,
    timing: Mapping[str, Any] | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "case_id": case_id,
        "case_split": case_split,
        "node_type_protocol": protocol,
        "resolution": resolution_label(resolution),
        "call": call,
        "physical_time": physical_time,
        "num_nodes": int(state.shape[0]),
        "reference_available": reference_state is not None,
        "state_scaled_rms": weighted_scaled_rms(
            state, volumes=volumes, component_scale=state_scale
        ),
        **conservative_admissibility_summary(state, gamma=gamma),
    }
    if reference_state is not None:
        error = state - reference_state
        row.update(
            {
                "error_scaled_rms": weighted_scaled_rms(
                    error, volumes=volumes, component_scale=state_scale
                ),
                "scaled_relative_l2_physical_volume": weighted_scaled_relative_l2(
                    state,
                    reference_state,
                    volumes=volumes,
                    component_scale=state_scale,
                ),
            }
        )
    if timing is not None:
        row.update(timing)
    return row


def _regional_error_rows(
    *,
    case_id: str,
    case_split: str,
    protocol: str,
    resolution: Resolution,
    call: int,
    physical_time: float,
    state: np.ndarray,
    reference_state: np.ndarray,
    geometry: Any,
    config: Any,
    physical_node_type: np.ndarray,
    boundary_band_widths: Sequence[float],
    state_scale: np.ndarray,
) -> list[dict[str, Any]]:
    regions: dict[str, np.ndarray] = {
        "boundary_touching_cells": np.asarray(physical_node_type) != 0,
    }
    for width in boundary_band_widths:
        regions[f"boundary_distance_le_{float(width):g}"] = (
            fixed_boundary_distance_mask(geometry.nodes, config, float(width))
        )
    rows = []
    for name, mask in regions.items():
        rows.append(
            {
                "case_id": case_id,
                "case_split": case_split,
                "node_type_protocol": protocol,
                "resolution": resolution_label(resolution),
                "call": call,
                "physical_time": physical_time,
                "region": name,
                "region_node_count": int(mask.sum()),
                "region_physical_volume": float(
                    np.asarray(geometry.node_measures).reshape(-1)[mask].sum()
                ),
                "error_scaled_rms": weighted_scaled_rms(
                    state - reference_state,
                    volumes=geometry.node_measures,
                    component_scale=state_scale,
                    mask=mask,
                ),
                "scaled_relative_l2_physical_volume": weighted_scaled_relative_l2(
                    state,
                    reference_state,
                    volumes=geometry.node_measures,
                    component_scale=state_scale,
                    mask=mask,
                ),
            }
        )
    return rows


def _structure_row(
    *,
    case_id: str,
    case_split: str,
    protocol: str,
    resolution: Resolution,
    call: int,
    physical_time: float,
    state: np.ndarray,
    reference_state: np.ndarray,
    initial_state: np.ndarray,
    cumulative_boundary_exchange: np.ndarray,
    geometry: Any,
    config: Any,
    manifest: Mapping[str, Any],
    store: PCNOEuler2DShardStore,
    state_scale: np.ndarray,
    gamma: float,
    shock_quantile: float,
) -> dict[str, Any]:
    prediction_primitive = conservative_to_primitive_raw(state, gamma=gamma)
    reference_primitive = conservative_to_primitive_raw(reference_state, gamma=gamma)
    prediction_centroid = np.asarray(
        shock_centroid(prediction_primitive, geometry.nodes, geometry.edges)
    ).reshape(-1)
    reference_centroid = np.asarray(
        shock_centroid(reference_primitive, geometry.nodes, geometry.edges)
    ).reshape(-1)
    spectral_fields: dict[str, Any] = {}
    domain_lengths = (
        float(config.x_max - config.x_min),
        float(config.y_max - config.y_min),
    )
    for band_index, (wavelength_min, wavelength_max) in enumerate(
        PHYSICAL_WAVELENGTH_BANDS, start=1
    ):
        metrics = physical_wavelength_band_metrics(
            state,
            reference_state,
            resolution=resolution,
            domain_lengths=domain_lengths,
            component_scale=state_scale,
            wavelength_min=wavelength_min,
            wavelength_max=wavelength_max,
        )
        for name, value in metrics.items():
            spectral_fields[f"physical_wavelength_band_{band_index}_{name}"] = value
    return {
        "case_id": case_id,
        "case_split": case_split,
        "node_type_protocol": protocol,
        "resolution": resolution_label(resolution),
        "call": call,
        "physical_time": physical_time,
        "prediction_shock_centroid": prediction_centroid,
        "reference_shock_centroid": reference_centroid,
        "shock_centroid_distance_physical": float(
            np.linalg.norm(prediction_centroid - reference_centroid)
        ),
        **pressure_profile_shock_metrics(
            state,
            reference_state,
            resolution=resolution,
            x_min=config.x_min,
            x_max=config.x_max,
            gamma=gamma,
            shock_center_x=config.shock_x,
        ),
        **endpoint_metrics(
            state,
            reference_state,
            positions=geometry.nodes,
            edges=geometry.edges,
            volumes=np.asarray(geometry.node_measures).reshape(-1),
            component_scale=state_scale,
            gamma=gamma,
            shock_quantile=shock_quantile,
            vortex_center=_expected_vortex_center(
                manifest, store, case_id, physical_time
            ),
        ),
        **physical_call_metrics(
            state,
            reference_state,
            initial_state,
            volumes=np.asarray(geometry.node_measures).reshape(-1),
            reference_cumulative_boundary_exchange=cumulative_boundary_exchange,
            component_scale=state_scale,
        ),
        **spectral_fields,
    }


def _architecture_audit(
    model: PCNOEuler2DResidual,
    geometry_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    prefactors = [
        float(layer.gw1.detach().float().cpu()) for layer in model.backbone.gws
    ]
    resolution_rows = []
    physical_rows = [
        row for row in geometry_rows if row["node_type_protocol"] == "physical"
    ]
    for row in physical_rows:
        dx = float(row["dx"])
        dy = float(row["dy"])
        resolution_rows.append(
            {
                "resolution": row["resolution"],
                "three_hop_physical_support_upper_bound": row[
                    "differential_input_support_physical_upper_bound"
                ],
                "softsign_boundary_jump_proxy_x_by_layer": [
                    abs(value) / dx / (1.0 + abs(value) / dx) for value in prefactors
                ],
                "softsign_boundary_jump_proxy_y_by_layer": [
                    abs(value) / dy / (1.0 + abs(value) / dy) for value in prefactors
                ],
            }
        )
    return {
        "paper_equation_alignment": {
            "integral": (
                "paper Eq. 13-15 uniform density: node_weights are rho(x)*cell_volume "
                "and sum to one"
            ),
            "gradient": "paper Eq. 16-19 least-squares edge pseudoinverse",
            "maintained_differences_from_paper_eq_20_21": (
                "each gradient is averaged for two fixed graph hops before a learned "
                "scalar prefactor and SoftSign; the paper states direct SoftSign(gradient)"
            ),
            "fourier": (
                "the maintained checkpoint fixes physical periods and modes; the paper "
                "describes length scales L as learnable"
            ),
        },
        "learned_gradient_softsign_prefactors": prefactors,
        "resolution_scaling": resolution_rows,
        "evaluation_padding_contract": (
            "batch size one, unpadded nodes and edges; heterogeneous padded batching is "
            "not used by this evaluator"
        ),
    }


def _run_experiment(
    args: argparse.Namespace,
    checkpoint: Mapping[str, Any],
    manifest: Mapping[str, Any],
    store: PCNOEuler2DShardStore,
    *,
    resolutions: Sequence[Resolution],
    source_resolution: Resolution,
    training_resolution: Resolution,
    stride: int,
    rollout_calls: int,
    physical_dt: float,
    endpoint_calls: set[int],
) -> dict[str, Any]:
    started_wall = perf_counter()
    protocols = list(dict.fromkeys(args.protocols))
    ordered_resolutions = sorted(resolutions, key=lambda value: value[0] * value[1])
    allowed_splits = set(args.allowed_splits)
    case_provenance: dict[str, dict[str, Any]] = {}
    case_configs = {}
    for case_id in args.case_ids:
        provenance = family_case_provenance(manifest, case_id)
        if provenance["split"] not in allowed_splits:
            raise ValueError(
                f"case {case_id} belongs to disallowed split {provenance['split']}"
            )
        if case_id not in store.keys:
            raise ValueError(f"case {case_id} is absent from checkpoint-bound shards")
        case_provenance[case_id] = provenance
        case_configs[case_id] = config_for_family_case(manifest, case_id)
    if not case_configs:
        raise ValueError("at least one physical case is required")

    device = select_device(args.device)
    if args.amp != "none" and device.type != "cuda":
        raise ValueError("mixed precision evaluation requires CUDA")
    model, normalization = build_resolution_checkpoint_model(checkpoint, device)
    first_config = next(iter(case_configs.values()))
    physical_lengths = (
        first_config.x_max - first_config.x_min,
        first_config.y_max - first_config.y_min,
    )
    if physical_lengths != tuple(float(v) for v in args.expected_domain_lengths):
        raise ValueError("physical domain and Fourier periods differ")

    geometry_by_resolution = {}
    config_by_resolution = {}
    physical_type_by_resolution = {}
    sample_by_resolution_protocol = {}
    geometry_rows: list[dict[str, Any]] = []
    for resolution in ordered_resolutions:
        build_started = perf_counter()
        config, geometry = build_resolution_geometry(first_config, resolution)
        build_seconds = perf_counter() - build_started
        geometry_by_resolution[resolution] = geometry
        config_by_resolution[resolution] = config
        physical_type = node_types_for_protocol(
            geometry,
            config,
            "physical",
            training_resolution=training_resolution,
        )
        physical_type_by_resolution[resolution] = physical_type
        for protocol in protocols:
            node_type = node_types_for_protocol(
                geometry,
                config,
                protocol,
                training_resolution=training_resolution,
            )
            sample_by_resolution_protocol[(resolution, protocol)] = make_model_sample(
                geometry,
                node_type,
                mach=first_config.shock_mach,
                device=device,
            )
            geometry_rows.append(
                {
                    "resolution": resolution_label(resolution),
                    "node_type_protocol": protocol,
                    "num_directed_edges": int(geometry.directed_edges.shape[0]),
                    "minimum_stencil_singular_value": geometry.minimum_stencil_singular_value,
                    "maximum_stencil_condition_number": geometry.maximum_stencil_condition_number,
                    "maximum_coordinate_gradient_error": geometry.maximum_coordinate_gradient_error,
                    "geometry_build_seconds": build_seconds,
                    **node_type_scaling_summary(geometry, config, node_type),
                }
            )
        print(
            f"built {resolution_label(resolution)} nodes={geometry.nodes.shape[0]} "
            f"edges={geometry.directed_edges.shape[0]} seconds={build_seconds:.3f}",
            flush=True,
        )

    native_geometry_audit_rows = native_geometry_audit(
        geometry_by_resolution[training_resolution], store, args.case_ids[0]
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)

    translation_rows: list[dict[str, Any]] = []
    reference_check_rows: list[dict[str, Any]] = []
    reference_gap_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    regional_rows: list[dict[str, Any]] = []
    structure_rows: list[dict[str, Any]] = []
    teacher_state_rows: list[dict[str, Any]] = []
    commutator_rows: list[dict[str, Any]] = []
    completion_rows: list[dict[str, Any]] = []
    reference_resolutions_seen: set[Resolution] = set()
    execution = {
        "logical_model_calls": 0,
        "actual_forward_passes_including_repeats": 0,
        "total_forward_seconds": 0.0,
        "maximum_repeat_abs_difference": 0.0,
        "maximum_peak_gpu_memory_bytes": 0,
    }

    def account(timing: Mapping[str, Any]) -> None:
        execution["logical_model_calls"] += 1
        execution["actual_forward_passes_including_repeats"] += len(
            timing["forward_seconds"]
        )
        execution["total_forward_seconds"] += float(sum(timing["forward_seconds"]))
        repeat = float(timing["repeat_max_abs"])
        if math.isfinite(repeat):
            execution["maximum_repeat_abs_difference"] = max(
                float(execution["maximum_repeat_abs_difference"]), repeat
            )
        peak = timing.get("peak_gpu_memory_bytes")
        if peak is not None:
            execution["maximum_peak_gpu_memory_bytes"] = max(
                int(execution["maximum_peak_gpu_memory_bytes"]), int(peak)
            )

    for case_index, case_id in enumerate(args.case_ids, start=1):
        provenance = case_provenance[case_id]
        case_split = str(provenance["split"])
        case_config = case_configs[case_id]
        print(
            f"case {case_index}/{len(args.case_ids)} {case_id} split={case_split}: "
            "loading reference and translating t0",
            flush=True,
        )
        reference, reference_check = load_resolution_reference(
            args.family_root,
            args.multires_reference_root,
            store,
            manifest,
            case_id,
            training_resolution=training_resolution,
        )
        reference_resolution = tuple(
            int(value) for value in reference["retained_resolution"]
        )
        reference_resolutions_seen.add(reference_resolution)
        for resolution in ordered_resolutions:
            if (
                reference_resolution[0] % resolution[0]
                or reference_resolution[1] % resolution[1]
            ):
                raise ValueError(
                    "active retained truth cannot restrict to every evaluation grid"
                )
        reference_check_rows.append(reference_check)
        _, common_states = initial_states_from_common_source(
            case_config,
            ordered_resolutions,
            source_resolution=source_resolution,
            dtype=torch.float64,
            quadrature_order=args.quadrature_order,
        )
        direct_states = {
            resolution: initial_state_for_model_grid(
                case_config,
                resolution,
                dtype=torch.float64,
                quadrature_order=args.quadrature_order,
            )
            for resolution in ordered_resolutions
        }
        stored_t0 = np.asarray(store.states(case_id)[0])
        reconstructed_native = np.asarray(
            common_states[training_resolution], dtype=np.float32
        )
        if not np.array_equal(reconstructed_native, stored_t0):
            raise ValueError(
                f"common-source native t0 does not reproduce stored float32 state: {case_id}"
            )
        translation_rows.extend(
            _translation_rows(
                case_id,
                common_states,
                direct_states,
                reference,
                resolutions=ordered_resolutions,
                reference_resolution=reference_resolution,
                geometry_by_resolution=geometry_by_resolution,
                state_scale=normalization.state_scale,
            )
        )

        reference_gap_lookup: dict[tuple[Resolution, Resolution, int], float] = {}
        for call in range(rollout_calls + 1):
            frame = call * stride
            physical_time = call * physical_dt
            if not math.isclose(
                float(reference["physical_times"][frame]),
                physical_time,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError(f"reference time mismatch for {case_id} call {call}")
            for coarse, fine in pairwise(ordered_resolutions):
                coarse_reference = reference_at_resolution(
                    reference["conservative_states"][frame],
                    reference_resolution=reference_resolution,
                    target_resolution=coarse,
                )
                fine_reference = reference_at_resolution(
                    reference["conservative_states"][frame],
                    reference_resolution=reference_resolution,
                    target_resolution=fine,
                )
                if coarse_reference is None or fine_reference is None:
                    continue
                restricted = restrict_nested_state(
                    fine_reference,
                    fine_resolution=fine,
                    coarse_resolution=coarse,
                )
                gap = weighted_scaled_rms(
                    coarse_reference - restricted,
                    volumes=geometry_by_resolution[coarse].node_measures,
                    component_scale=normalization.state_scale,
                )
                reference_gap_lookup[(coarse, fine, call)] = gap
                reference_gap_rows.append(
                    {
                        "case_id": case_id,
                        "case_split": case_split,
                        "coarse_resolution": resolution_label(coarse),
                        "fine_resolution": resolution_label(fine),
                        "call": call,
                        "physical_time": physical_time,
                        "reference_discretization_gap_scaled_rms": gap,
                        "reference_discretization_gap_relative_l2": weighted_scaled_relative_l2(
                            coarse_reference,
                            restricted,
                            volumes=geometry_by_resolution[coarse].node_measures,
                            component_scale=normalization.state_scale,
                        ),
                    }
                )

        for protocol in protocols:
            free_states = {
                resolution: as_model_state(common_states[resolution])
                for resolution in ordered_resolutions
            }
            initial_states = {
                resolution: np.array(state, copy=True)
                for resolution, state in free_states.items()
            }
            active_by_resolution = {
                resolution: True for resolution in ordered_resolutions
            }
            all_calls_admissible_by_resolution = {
                resolution: True for resolution in ordered_resolutions
            }
            completed_calls_by_resolution = {
                resolution: 0 for resolution in ordered_resolutions
            }
            first_nonfinite_call_by_resolution = {
                resolution: None for resolution in ordered_resolutions
            }

            for resolution in ordered_resolutions:
                reference_t0 = reference_at_resolution(
                    reference["conservative_states"][0],
                    reference_resolution=reference_resolution,
                    target_resolution=resolution,
                )
                geometry = geometry_by_resolution[resolution]
                state_rows.append(
                    _state_row(
                        case_id=case_id,
                        case_split=case_split,
                        protocol=protocol,
                        resolution=resolution,
                        call=0,
                        physical_time=0.0,
                        state=free_states[resolution],
                        reference_state=reference_t0,
                        volumes=geometry.node_measures,
                        state_scale=normalization.state_scale,
                        gamma=normalization.gamma,
                        timing=None,
                    )
                )
                if reference_t0 is not None:
                    regional_rows.extend(
                        _regional_error_rows(
                            case_id=case_id,
                            case_split=case_split,
                            protocol=protocol,
                            resolution=resolution,
                            call=0,
                            physical_time=0.0,
                            state=free_states[resolution],
                            reference_state=reference_t0,
                            geometry=geometry,
                            config=config_by_resolution[resolution],
                            physical_node_type=physical_type_by_resolution[resolution],
                            boundary_band_widths=args.boundary_band_widths,
                            state_scale=normalization.state_scale,
                        )
                    )

            for call in range(1, rollout_calls + 1):
                physical_time = call * physical_dt
                frame = call * stride
                input_physical_time = (call - 1) * physical_dt
                input_frame = (call - 1) * stride
                previous_states = free_states
                active_at_start = dict(active_by_resolution)
                next_states = dict(previous_states)
                timing_by_resolution = {}
                finite_by_resolution = {}
                for resolution in ordered_resolutions:
                    if not active_at_start[resolution]:
                        continue
                    prediction, timing = predict_resolution_sample(
                        model,
                        sample_by_resolution_protocol[(resolution, protocol)],
                        previous_states[resolution],
                        device=device,
                        amp=args.amp,
                        repeats=args.repeat_forward if call == 1 else 1,
                    )
                    account(timing)
                    next_states[resolution] = prediction
                    timing_by_resolution[resolution] = timing
                    finite = bool(np.isfinite(prediction).all())
                    finite_by_resolution[resolution] = finite
                    if not finite:
                        active_by_resolution[resolution] = False

                teacher_currents = {}
                teacher_predictions = {}
                teacher_finite_by_resolution = {}
                for resolution in ordered_resolutions:
                    reference_input = reference_at_resolution(
                        reference["conservative_states"][input_frame],
                        reference_resolution=reference_resolution,
                        target_resolution=resolution,
                    )
                    reference_target = reference_at_resolution(
                        reference["conservative_states"][frame],
                        reference_resolution=reference_resolution,
                        target_resolution=resolution,
                    )
                    if reference_input is None or reference_target is None:
                        raise ValueError(
                            "teacher-forced reference is unavailable on an "
                            "evaluation grid"
                        )
                    teacher_current = as_model_state(reference_input)
                    teacher_prediction, teacher_timing = predict_resolution_sample(
                        model,
                        sample_by_resolution_protocol[(resolution, protocol)],
                        teacher_current,
                        device=device,
                        amp=args.amp,
                        repeats=1,
                    )
                    account(teacher_timing)
                    teacher_currents[resolution] = teacher_current
                    teacher_predictions[resolution] = teacher_prediction
                    teacher_finite_by_resolution[resolution] = bool(
                        np.isfinite(teacher_prediction).all()
                    )
                    teacher_row = _state_row(
                        case_id=case_id,
                        case_split=case_split,
                        protocol=protocol,
                        resolution=resolution,
                        call=call,
                        physical_time=physical_time,
                        state=teacher_prediction,
                        reference_state=reference_target,
                        volumes=geometry_by_resolution[resolution].node_measures,
                        state_scale=normalization.state_scale,
                        gamma=normalization.gamma,
                        timing=teacher_timing,
                    )
                    teacher_row.update(
                        {
                            "input_call": call - 1,
                            "input_frame": input_frame,
                            "output_frame": frame,
                            "input_physical_time": input_physical_time,
                        }
                    )
                    teacher_state_rows.append(teacher_row)

                for coarse, fine in pairwise(ordered_resolutions):
                    pair_is_finite = (
                        active_at_start[coarse]
                        and active_at_start[fine]
                        and finite_by_resolution.get(coarse, False)
                        and finite_by_resolution.get(fine, False)
                    )
                    if pair_is_finite:
                        free_metrics = commutator_metrics(
                            coarse_current=previous_states[coarse],
                            coarse_prediction=next_states[coarse],
                            fine_current=previous_states[fine],
                            fine_prediction=next_states[fine],
                            coarse_resolution=coarse,
                            fine_resolution=fine,
                            coarse_volumes=geometry_by_resolution[coarse].node_measures,
                            state_scale=normalization.state_scale,
                            residual_scale=normalization.residual_scale,
                        )
                        reference_gap = reference_gap_lookup.get((coarse, fine, call))
                        commutator_rows.append(
                            commutator_row(
                                case_id=case_id,
                                case_split=case_split,
                                protocol=protocol,
                                kind="free_rollout",
                                coarse=coarse,
                                fine=fine,
                                call=call,
                                input_frame=input_frame,
                                output_frame=frame,
                                input_physical_time=input_physical_time,
                                output_physical_time=physical_time,
                                reference_gap=reference_gap,
                                metrics=free_metrics,
                            )
                        )

                        local_current = as_model_state(
                            restrict_nested_state(
                                previous_states[fine],
                                fine_resolution=fine,
                                coarse_resolution=coarse,
                            )
                        )
                        local_prediction, local_timing = predict_resolution_sample(
                            model,
                            sample_by_resolution_protocol[(coarse, protocol)],
                            local_current,
                            device=device,
                            amp=args.amp,
                            repeats=1,
                        )
                        account(local_timing)
                        if bool(np.isfinite(local_prediction).all()):
                            local_metrics = commutator_metrics(
                                coarse_current=local_current,
                                coarse_prediction=local_prediction,
                                fine_current=previous_states[fine],
                                fine_prediction=next_states[fine],
                                coarse_resolution=coarse,
                                fine_resolution=fine,
                                coarse_volumes=geometry_by_resolution[
                                    coarse
                                ].node_measures,
                                state_scale=normalization.state_scale,
                                residual_scale=normalization.residual_scale,
                            )
                            commutator_rows.append(
                                commutator_row(
                                    case_id=case_id,
                                    case_split=case_split,
                                    protocol=protocol,
                                    kind="local_restricted_fine_input",
                                    coarse=coarse,
                                    fine=fine,
                                    call=call,
                                    input_frame=input_frame,
                                    output_frame=frame,
                                    input_physical_time=input_physical_time,
                                    output_physical_time=physical_time,
                                    reference_gap=reference_gap,
                                    metrics=local_metrics,
                                )
                            )

                    teacher_pair_is_finite = (
                        teacher_finite_by_resolution[coarse]
                        and teacher_finite_by_resolution[fine]
                    )
                    if teacher_pair_is_finite:
                        teacher_metrics = commutator_metrics(
                            coarse_current=teacher_currents[coarse],
                            coarse_prediction=teacher_predictions[coarse],
                            fine_current=teacher_currents[fine],
                            fine_prediction=teacher_predictions[fine],
                            coarse_resolution=coarse,
                            fine_resolution=fine,
                            coarse_volumes=geometry_by_resolution[coarse].node_measures,
                            state_scale=normalization.state_scale,
                            residual_scale=normalization.residual_scale,
                        )
                        reference_gap = reference_gap_lookup.get((coarse, fine, call))
                        commutator_rows.append(
                            commutator_row(
                                case_id=case_id,
                                case_split=case_split,
                                protocol=protocol,
                                kind="teacher_forced_exact_reference_input",
                                coarse=coarse,
                                fine=fine,
                                call=call,
                                input_frame=input_frame,
                                output_frame=frame,
                                input_physical_time=input_physical_time,
                                output_physical_time=physical_time,
                                reference_gap=reference_gap,
                                metrics=teacher_metrics,
                            )
                        )

                free_states = next_states
                for resolution in ordered_resolutions:
                    if not active_at_start[resolution]:
                        continue
                    geometry = geometry_by_resolution[resolution]
                    target = reference_at_resolution(
                        reference["conservative_states"][frame],
                        reference_resolution=reference_resolution,
                        target_resolution=resolution,
                    )
                    state_row = _state_row(
                        case_id=case_id,
                        case_split=case_split,
                        protocol=protocol,
                        resolution=resolution,
                        call=call,
                        physical_time=physical_time,
                        state=free_states[resolution],
                        reference_state=target,
                        volumes=geometry.node_measures,
                        state_scale=normalization.state_scale,
                        gamma=normalization.gamma,
                        timing=timing_by_resolution[resolution],
                    )
                    state_rows.append(state_row)
                    finite = finite_by_resolution[resolution]
                    if finite:
                        completed_calls_by_resolution[resolution] = call
                        all_calls_admissible_by_resolution[resolution] = (
                            all_calls_admissible_by_resolution[resolution]
                            and bool(state_row["admissible"])
                        )
                    else:
                        all_calls_admissible_by_resolution[resolution] = False
                        first_nonfinite_call_by_resolution[resolution] = call
                    if target is not None and finite:
                        regional_rows.extend(
                            _regional_error_rows(
                                case_id=case_id,
                                case_split=case_split,
                                protocol=protocol,
                                resolution=resolution,
                                call=call,
                                physical_time=physical_time,
                                state=free_states[resolution],
                                reference_state=target,
                                geometry=geometry,
                                config=config_by_resolution[resolution],
                                physical_node_type=physical_type_by_resolution[
                                    resolution
                                ],
                                boundary_band_widths=args.boundary_band_widths,
                                state_scale=normalization.state_scale,
                            )
                        )
                        if call in endpoint_calls:
                            cumulative_exchange = np.sum(
                                reference["interval_boundary_exchange"][:frame], axis=0
                            )
                            structure_rows.append(
                                _structure_row(
                                    case_id=case_id,
                                    case_split=case_split,
                                    protocol=protocol,
                                    resolution=resolution,
                                    call=call,
                                    physical_time=physical_time,
                                    state=free_states[resolution],
                                    reference_state=target,
                                    initial_state=initial_states[resolution],
                                    cumulative_boundary_exchange=cumulative_exchange,
                                    geometry=geometry,
                                    config=config_by_resolution[resolution],
                                    manifest=manifest,
                                    store=store,
                                    state_scale=normalization.state_scale,
                                    gamma=normalization.gamma,
                                    shock_quantile=args.shock_quantile,
                                )
                            )
                if call == 1 or call in endpoint_calls:
                    print(
                        f"rollout case={case_id} types={protocol} call={call}/"
                        f"{rollout_calls} t={physical_time:.3f}",
                        flush=True,
                    )

            for resolution in ordered_resolutions:
                completed_calls = completed_calls_by_resolution[resolution]
                completion_rows.append(
                    {
                        "case_id": case_id,
                        "case_split": case_split,
                        "node_type_protocol": protocol,
                        "resolution": resolution_label(resolution),
                        "requested_calls": rollout_calls,
                        "completed_finite_calls": completed_calls,
                        "full_completion": completed_calls == rollout_calls,
                        "all_completed_calls_admissible": (
                            all_calls_admissible_by_resolution[resolution]
                        ),
                        "first_nonfinite_call": (
                            first_nonfinite_call_by_resolution[resolution]
                        ),
                    }
                )

        del reference, common_states, direct_states
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if len(reference_resolutions_seen) != 1:
        raise ValueError(
            "all cases must use one common retained-reference resolution contract"
        )
    active_reference_resolution = next(iter(reference_resolutions_seen))

    write_csv(output_dir / "geometry.csv", geometry_rows)
    write_csv(output_dir / "translation_preflight.csv", translation_rows)
    write_csv(output_dir / "reference_checks.csv", reference_check_rows)
    write_csv(output_dir / "reference_gaps.csv", reference_gap_rows)
    write_csv(output_dir / "state_metrics.csv", state_rows)
    write_csv(output_dir / "regional_errors.csv", regional_rows)
    write_csv(output_dir / "structure_metrics.csv", structure_rows)
    write_csv(output_dir / "teacher_forced_state_metrics.csv", teacher_state_rows)
    write_csv(output_dir / "commutators.csv", commutator_rows)
    write_csv(output_dir / "completion.csv", completion_rows)

    checkpoint_sha = sha256_file(args.checkpoint)
    source_hashes = _source_hashes()
    table_files = sorted(output_dir.glob("*.csv"))
    elapsed_wall = perf_counter() - started_wall
    execution.update(
        {
            "device": str(device),
            "amp": args.amp,
            "repeat_forward_scope": "first free call for every case/grid/protocol",
            "parameter_count": parameter_count(model),
            "teacher_forced_state_rows": len(teacher_state_rows),
            "wall_seconds_before_summary_write": elapsed_wall,
            "csv_storage_bytes": int(sum(path.stat().st_size for path in table_files)),
        }
    )
    if device.type != "cuda":
        execution["maximum_peak_gpu_memory_bytes"] = None
    return {
        "schema": SCHEMA,
        "status": "complete",
        "scientific_scope": {
            "primary_test": (
                "raw recurrent rollout of one frozen parameter vector on common-fine "
                "conservative representations"
            ),
            "teacher_forced_diagnostic": (
                "at every aligned map call, independently restrict the exact common "
                "reference input to each grid, apply the same frozen model, and compare "
                "coarse prediction with exact restriction of fine prediction"
            ),
            "restriction_consistent_reference_resolutions": [
                resolution_label(resolution)
                for resolution in ordered_resolutions
                if active_reference_resolution[0] % resolution[0] == 0
                and active_reference_resolution[1] % resolution[1] == 0
            ],
            "commutator_only_resolutions": [
                resolution_label(resolution)
                for resolution in ordered_resolutions
                if active_reference_resolution[0] % resolution[0] != 0
                or active_reference_resolution[1] % resolution[1] != 0
            ],
            "claim_boundary": (
                "Finite tests can support zero-shot discretization generalization only for "
                "the named PDE/geometry/boundary/timestep family and bound sampling, "
                "quadrature, normalization, and evaluator. They do not prove that PCNO is "
                "an operator or establish native-solver, PDE-family, geometry, or timestep transfer."
            ),
        },
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": checkpoint_sha,
            "normalization_digest": checkpoint["normalization_digest"],
            "data_manifest_digest": checkpoint["data_manifest_digest"],
            "config_digest": checkpoint["config_digest"],
            "boundary_mode": checkpoint["boundary_mode"],
            "raw_recurrence": checkpoint["raw_recurrence"],
            "model_config": checkpoint["model_config"],
            "model_node_type_input": checkpoint_model_node_type_input(checkpoint),
            "data_contract": checkpoint["data_contract"],
        },
        "source": {"git_head": _git_head(), "file_sha256": source_hashes},
        "family": {
            "root": str(args.family_root),
            "manifest_digest_sha256": manifest["manifest_digest_sha256"],
            "cases": [case_provenance[case_id] for case_id in args.case_ids],
        },
        "translation_contract": {
            "state": (
                "cell averages of conservative [rho,rho_u,rho_v,E], sampled once on "
                "the common source grid and block-averaged; no primitive interpolation"
            ),
            "source_resolution": resolution_label(source_resolution),
            "retained_reference_resolution": resolution_label(
                active_reference_resolution
            ),
            "training_resolution": resolution_label(training_resolution),
            "evaluation_resolutions": [
                resolution_label(value) for value in ordered_resolutions
            ],
            "geometry": (
                "centroids, volumes, boundary types, directed graph, and least-squares "
                "gradient weights regenerated independently on every grid"
            ),
            "node_type_protocols": protocols,
            "Fourier_periods": list(args.expected_domain_lengths),
            "k_max": args.expected_k_max,
            "normalization": "immutable checkpoint normalization",
            "reference_provenance": (
                "common high-fidelity evolution with exact conservative block "
                "restriction"
            ),
            "boundary_policy": "model_all_nodes raw recurrence",
            "teacher_forced_input": (
                "exact common-reference conservative state at the preceding aligned "
                "frame, independently restricted to each grid and rounded to float32 "
                "at the model boundary"
            ),
            "step_stride": stride,
            "physical_delta_t_per_model_call": physical_dt,
            "rollout_calls": rollout_calls,
            "physical_horizon": rollout_calls * physical_dt,
        },
        "native_geometry_audit": native_geometry_audit_rows,
        "architecture_audit": _architecture_audit(model, geometry_rows),
        "execution": execution,
        "completion": completion_rows,
        "aggregate": aggregate_resolution_rollout(
            state_rows,
            commutator_rows,
            completion_rows,
            teacher_state_rows=teacher_state_rows,
            final_call=rollout_calls,
        ),
        "artifacts": {
            path.name: {"bytes": path.stat().st_size} for path in table_files
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    resolutions = [parse_resolution(value) for value in args.resolutions]
    source_resolution = parse_resolution(args.source_resolution)
    training_resolution = parse_resolution(args.training_resolution)
    manifest_path = args.family_root / "family_manifest.json"
    manifest = load_shock_vortex_family_manifest(manifest_path)
    checkpoint = load_resolution_checkpoint(args.checkpoint)
    store = PCNOEuler2DShardStore(args.data_dir, max_cached_trajectories=2)
    try:
        stride, rollout_calls, physical_dt, endpoint_calls = (
            validate_resolution_rollout_contract(
                args,
                checkpoint,
                manifest,
                store,
                resolutions,
                source_resolution,
                training_resolution,
            )
        )
        print(
            "resolution rollout:",
            f"device={args.device}",
            f"amp={args.amp}",
            f"cases={len(args.case_ids)}",
            f"calls={rollout_calls}",
            f"grids={','.join(resolution_label(value) for value in resolutions)}",
            flush=True,
        )
        summary = _run_experiment(
            args,
            checkpoint,
            manifest,
            store,
            resolutions=resolutions,
            source_resolution=source_resolution,
            training_resolution=training_resolution,
            stride=stride,
            rollout_calls=rollout_calls,
            physical_dt=physical_dt,
            endpoint_calls=endpoint_calls,
        )
        atomic_write_json(args.output_dir / "summary.json", summary)
    finally:
        store.close()
    print(f"wrote {args.output_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
