#!/usr/bin/env python3
"""Evaluate one frozen dynamic-family PCNO across regenerated model grids.

This is an R0/R0b representation and commutator diagnostic.  It samples the
same analytic shock--vortex initial condition as conservative cell averages on
each grid, rebuilds all finite-volume PCNO geometry, and applies one immutable
checkpoint without fine-tuning.  It does not claim time-evolved PDE accuracy;
that requires common high-fidelity trajectories restricted to every model grid.
"""

from __future__ import annotations

import argparse
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

from utility.time_dependent_no.pcno_artifacts import (
    atomic_write_json_with_paths as atomic_write_json,
    git_head,
    sha256_file,
    sha256_files,
    write_csv_with_paths as write_csv,
)
from utility.time_dependent_no.pcno_euler2d import parameter_count
from utility.time_dependent_no.pcno_resolution_transfer import (
    NODE_TYPE_PROTOCOLS,
    Resolution,
    build_resolution_checkpoint_model,
    build_resolution_geometry,
    commutator_metrics,
    conservative_admissibility_summary,
    fixed_boundary_distance_mask,
    initial_state_for_model_grid,
    load_resolution_checkpoint,
    make_model_sample,
    node_type_scaling_summary,
    node_types_for_protocol,
    parse_resolution,
    resolution_label,
    weighted_scaled_rms,
)
from utility.time_dependent_no.pcno_runtime import (
    select_device,
    timed_model_call,
)
from utility.time_dependent_no.shock_vortex_family import (
    config_for_family_case,
    family_case_provenance,
    load_shock_vortex_family_manifest,
)

SCHEMA = "pcno_resolution_transfer_r0_v1"
DEFAULT_CASE_IDS = ("sv_e00_y04", "sv_e06_y04", "sv_e11_y04")
DEFAULT_RESOLUTIONS = ("125x50", "250x100", "500x200")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-ids", nargs="+", default=list(DEFAULT_CASE_IDS))
    parser.add_argument(
        "--allowed-splits",
        nargs="+",
        choices=("train", "validation"),
        default=("train",),
        help="Sealed test cases are intentionally unavailable to this evaluator.",
    )
    parser.add_argument(
        "--resolutions",
        nargs="+",
        default=list(DEFAULT_RESOLUTIONS),
        metavar="NXxNY",
    )
    parser.add_argument(
        "--training-resolution",
        default="250x100",
        metavar="NXxNY",
        help="Grid used to define the training_band diagnostic width.",
    )
    parser.add_argument(
        "--protocols",
        nargs="+",
        choices=NODE_TYPE_PROTOCOLS,
        default=list(NODE_TYPE_PROTOCOLS),
    )
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--expected-normalization-digest")
    parser.add_argument("--expected-k-max", type=int, default=8)
    parser.add_argument(
        "--expected-domain-lengths",
        type=float,
        nargs=2,
        default=(2.0, 1.0),
        metavar=("LX", "LY"),
    )
    parser.add_argument("--quadrature-order", type=int, default=None)
    parser.add_argument(
        "--boundary-band-widths",
        type=float,
        nargs="+",
        default=(0.02, 0.05),
    )
    parser.add_argument("--repeat-forward", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--amp", choices=("checkpoint", "none", "bf16", "fp16"), default="checkpoint"
    )
    return parser.parse_args(argv)


def validate_contract(
    args: argparse.Namespace,
    checkpoint: Mapping[str, Any],
    checkpoint_sha256: str,
    resolutions: Sequence[Resolution],
) -> None:
    if args.expected_checkpoint_sha256 is not None:
        expected = args.expected_checkpoint_sha256.lower()
        if checkpoint_sha256.lower() != expected:
            raise ValueError(
                "checkpoint SHA-256 mismatch: "
                f"expected {expected}, got {checkpoint_sha256}"
            )
    if (
        args.expected_normalization_digest is not None
        and checkpoint["normalization_digest"] != args.expected_normalization_digest
    ):
        raise ValueError("checkpoint normalization digest mismatch")
    config = checkpoint["model_config"]
    if int(config["k_max"]) != args.expected_k_max:
        raise ValueError("checkpoint k_max differs from the frozen physical bandwidth")
    domain_lengths = tuple(float(value) for value in config["domain_lengths"])
    expected_lengths = tuple(float(value) for value in args.expected_domain_lengths)
    if domain_lengths != expected_lengths:
        raise ValueError(
            "checkpoint Fourier periods differ from expected physical periods"
        )
    if int(config["nmeasures"]) != 1:
        raise ValueError("maintained Euler wrapper must use exactly one measure")
    if len(resolutions) < 2 or len(set(resolutions)) != len(resolutions):
        raise ValueError("resolutions must contain at least two unique grids")
    ordered = sorted(resolutions, key=lambda item: item[0] * item[1])
    for coarse, fine in pairwise(ordered):
        if fine[0] % coarse[0] or fine[1] % coarse[1]:
            raise ValueError(
                "successive resolution grids must be nested for conservative "
                f"restriction: {resolution_label(coarse)} -> "
                f"{resolution_label(fine)}"
            )
    if args.repeat_forward < 1:
        raise ValueError("repeat-forward must be positive")
    widths = np.asarray(args.boundary_band_widths, dtype=np.float64)
    if np.any(~np.isfinite(widths)) or np.any(widths <= 0.0):
        raise ValueError("boundary-band widths must be positive and finite")


def source_hashes() -> dict[str, str]:
    paths = (
        Path("pcno/pcno.py"),
        Path("utility/time_dependent_no/pcno_artifacts.py"),
        Path("utility/time_dependent_no/pcno_euler2d.py"),
        Path("utility/time_dependent_no/pcno_runtime.py"),
        Path("utility/time_dependent_no/pcno_fv_geometry.py"),
        Path("utility/time_dependent_no/pcno_resolution_transfer.py"),
        Path("utility/time_dependent_no/pcno_ripple_diagnostics.py"),
        Path("utility/time_dependent_no/shock_vortex_coarse_cfd.py"),
        Path("utility/time_dependent_no/shock_vortex_family.py"),
        Path("utility/time_dependent_no/shock_vortex_fv.py"),
        Path("scripts/time_dependent_no/evaluate_pcno_resolution_transfer.py"),
    )
    return sha256_files(paths, root=ROOT)


def aggregate_rows(
    commutator_rows: Sequence[Mapping[str, Any]],
    intervention_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    commutators: dict[str, dict[str, float | None]] = {}
    protocols = sorted({str(row["node_type_protocol"]) for row in commutator_rows})
    for protocol in protocols:
        selected = [
            row for row in commutator_rows if row["node_type_protocol"] == protocol
        ]
        prediction_values = [
            float(row["prediction_commutator_relative_l2"])
            for row in selected
            if row["prediction_commutator_relative_l2"] is not None
        ]
        update_values = [
            float(row["update_commutator_relative_to_fine_update"])
            for row in selected
            if row["update_commutator_relative_to_fine_update"] is not None
        ]
        commutators[protocol] = {
            "mean_prediction_commutator_relative_l2": (
                float(np.mean(prediction_values)) if prediction_values else None
            ),
            "maximum_prediction_commutator_relative_l2": (
                float(np.max(prediction_values)) if prediction_values else None
            ),
            "mean_update_commutator_relative_to_fine_update": (
                float(np.mean(update_values)) if update_values else None
            ),
            "maximum_update_commutator_relative_to_fine_update": (
                float(np.max(update_values)) if update_values else None
            ),
        }
    global_interventions = [
        row for row in intervention_rows if row["region"] == "global"
    ]
    return {
        "commutators_by_protocol": commutators,
        "maximum_global_node_type_intervention_relative_to_physical_update": (
            max(
                (
                    float(row["intervention_relative_to_physical_update"])
                    for row in global_interventions
                    if row["intervention_relative_to_physical_update"] is not None
                ),
                default=None,
            )
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    resolutions = [parse_resolution(value) for value in args.resolutions]
    training_resolution = parse_resolution(args.training_resolution)
    protocols = list(dict.fromkeys(args.protocols))
    checkpoint_sha256 = sha256_file(args.checkpoint)
    checkpoint = load_resolution_checkpoint(args.checkpoint)
    validate_contract(args, checkpoint, checkpoint_sha256, resolutions)
    manifest = load_shock_vortex_family_manifest(args.family_manifest)

    case_provenance = []
    case_configs = {}
    allowed_splits = set(args.allowed_splits)
    for case_id in args.case_ids:
        provenance = family_case_provenance(manifest, case_id)
        if provenance["split"] not in allowed_splits:
            raise ValueError(
                f"case {case_id} belongs to disallowed split {provenance['split']}"
            )
        case_provenance.append(provenance)
        case_configs[case_id] = config_for_family_case(manifest, case_id)
    if not case_configs:
        raise ValueError("at least one case is required")

    first_config = next(iter(case_configs.values()))
    physical_lengths = (
        first_config.x_max - first_config.x_min,
        first_config.y_max - first_config.y_min,
    )
    if physical_lengths != tuple(float(v) for v in args.expected_domain_lengths):
        raise ValueError("expected Fourier periods do not match the physical domain")
    for config in case_configs.values():
        geometry_contract = (
            config.x_min,
            config.x_max,
            config.y_min,
            config.y_max,
            config.gamma,
            config.shock_mach,
        )
        first_contract = (
            first_config.x_min,
            first_config.x_max,
            first_config.y_min,
            first_config.y_max,
            first_config.gamma,
            first_config.shock_mach,
        )
        if geometry_contract != first_contract:
            raise ValueError("selected cases do not share one geometry/PDE contract")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    device = select_device(args.device)
    amp = (
        str(checkpoint.get("training_args", {}).get("amp", "none"))
        if args.amp == "checkpoint"
        else args.amp
    )
    if amp not in {"none", "bf16", "fp16"}:
        raise ValueError(f"unsupported checkpoint AMP declaration: {amp}")
    if amp != "none" and device.type != "cuda":
        raise ValueError("mixed precision requires CUDA; use --amp none on CPU")
    model, normalization = build_resolution_checkpoint_model(checkpoint, device)

    print(
        "resolution transfer:",
        f"device={device}",
        f"amp={amp}",
        f"cases={len(case_configs)}",
        f"grids={','.join(resolution_label(r) for r in resolutions)}",
        flush=True,
    )

    geometry_by_resolution = {}
    config_by_resolution = {}
    node_type_by_resolution: dict[tuple[Resolution, str], np.ndarray] = {}
    geometry_rows: list[dict[str, Any]] = []
    geometry_seconds: dict[str, float] = {}
    for resolution in sorted(resolutions, key=lambda item: item[0] * item[1]):
        started = perf_counter()
        config, geometry = build_resolution_geometry(first_config, resolution)
        elapsed = perf_counter() - started
        label = resolution_label(resolution)
        geometry_seconds[label] = elapsed
        geometry_by_resolution[resolution] = geometry
        config_by_resolution[resolution] = config
        print(
            f"built {label}: nodes={geometry.nodes.shape[0]} "
            f"directed_edges={geometry.directed_edges.shape[0]} "
            f"seconds={elapsed:.3f}",
            flush=True,
        )
        for protocol in protocols:
            node_type = node_types_for_protocol(
                geometry,
                config,
                protocol,
                training_resolution=training_resolution,
            )
            node_type_by_resolution[(resolution, protocol)] = node_type
            summary = node_type_scaling_summary(geometry, config, node_type)
            geometry_rows.append(
                {
                    "resolution": label,
                    "node_type_protocol": protocol,
                    "num_directed_edges": int(geometry.directed_edges.shape[0]),
                    "minimum_stencil_singular_value": (
                        geometry.minimum_stencil_singular_value
                    ),
                    "maximum_stencil_condition_number": (
                        geometry.maximum_stencil_condition_number
                    ),
                    "maximum_coordinate_gradient_error": (
                        geometry.maximum_coordinate_gradient_error
                    ),
                    "geometry_build_seconds": elapsed,
                    **summary,
                }
            )

    current_by_case_resolution: dict[tuple[str, Resolution], np.ndarray] = {}
    prediction_by_key: dict[tuple[str, Resolution, str], np.ndarray] = {}
    prediction_rows: list[dict[str, Any]] = []
    maximum_repeat_difference = 0.0
    maximum_peak_gpu_memory = 0
    ordered_resolutions = sorted(resolutions, key=lambda item: item[0] * item[1])
    for resolution in ordered_resolutions:
        label = resolution_label(resolution)
        geometry = geometry_by_resolution[resolution]
        config = config_by_resolution[resolution]
        samples = {
            protocol: make_model_sample(
                geometry,
                node_type_by_resolution[(resolution, protocol)],
                mach=first_config.shock_mach,
                device=device,
            )
            for protocol in protocols
        }
        for case_id, case_config in case_configs.items():
            current = initial_state_for_model_grid(
                case_config,
                resolution,
                dtype=torch.float64,
                quadrature_order=args.quadrature_order,
            ).astype(np.float64, copy=False)
            current_by_case_resolution[(case_id, resolution)] = current
            current_tensor = torch.as_tensor(
                current.astype(np.float32),
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)
            for protocol in protocols:
                prediction, timing = timed_model_call(
                    model,
                    samples[protocol],
                    current_tensor,
                    device=device,
                    amp=amp,
                    repeats=args.repeat_forward,
                )
                prediction_by_key[(case_id, resolution, protocol)] = prediction
                maximum_repeat_difference = max(
                    maximum_repeat_difference,
                    float(timing["repeat_max_abs"]),
                )
                if timing["peak_gpu_memory_bytes"] is not None:
                    maximum_peak_gpu_memory = max(
                        maximum_peak_gpu_memory,
                        int(timing["peak_gpu_memory_bytes"]),
                    )
                update = prediction - current
                prediction_rows.append(
                    {
                        "case_id": case_id,
                        "case_split": family_case_provenance(manifest, case_id)[
                            "split"
                        ],
                        "resolution": label,
                        "node_type_protocol": protocol,
                        "num_nodes": int(current.shape[0]),
                        "state_scaled_rms": weighted_scaled_rms(
                            prediction,
                            volumes=geometry.node_measures,
                            component_scale=normalization.state_scale,
                        ),
                        "update_scaled_rms": weighted_scaled_rms(
                            update,
                            volumes=geometry.node_measures,
                            component_scale=normalization.residual_scale,
                        ),
                        **conservative_admissibility_summary(
                            prediction,
                            gamma=normalization.gamma,
                        ),
                        **timing,
                    }
                )
                print(
                    f"inferred case={case_id} grid={label} types={protocol} "
                    f"seconds={timing['mean_forward_seconds']:.3f}",
                    flush=True,
                )
            del current_tensor
        del samples
        if device.type == "cuda":
            torch.cuda.empty_cache()

    commutator_rows: list[dict[str, Any]] = []
    for coarse_resolution, fine_resolution in pairwise(ordered_resolutions):
        coarse_label = resolution_label(coarse_resolution)
        fine_label = resolution_label(fine_resolution)
        coarse_geometry = geometry_by_resolution[coarse_resolution]
        for case_id in case_configs:
            for protocol in protocols:
                metrics = commutator_metrics(
                    coarse_current=current_by_case_resolution[
                        (case_id, coarse_resolution)
                    ],
                    coarse_prediction=prediction_by_key[
                        (case_id, coarse_resolution, protocol)
                    ],
                    fine_current=current_by_case_resolution[(case_id, fine_resolution)],
                    fine_prediction=prediction_by_key[
                        (case_id, fine_resolution, protocol)
                    ],
                    coarse_resolution=coarse_resolution,
                    fine_resolution=fine_resolution,
                    coarse_volumes=coarse_geometry.node_measures,
                    state_scale=normalization.state_scale,
                    residual_scale=normalization.residual_scale,
                )
                commutator_rows.append(
                    {
                        "case_id": case_id,
                        "node_type_protocol": protocol,
                        "coarse_resolution": coarse_label,
                        "fine_resolution": fine_label,
                        **metrics,
                    }
                )

    intervention_rows: list[dict[str, Any]] = []
    for resolution in ordered_resolutions:
        geometry = geometry_by_resolution[resolution]
        config = config_by_resolution[resolution]
        physical_types = node_type_by_resolution.get((resolution, "physical"))
        if physical_types is None:
            continue
        regions: dict[str, np.ndarray] = {
            "global": np.ones(geometry.nodes.shape[0], dtype=bool),
            "physical_boundary_cells": physical_types != 0,
        }
        for width in args.boundary_band_widths:
            regions[f"boundary_distance_le_{float(width):g}"] = (
                fixed_boundary_distance_mask(geometry.nodes, config, float(width))
            )
        for case_id in case_configs:
            physical_prediction = prediction_by_key.get(
                (case_id, resolution, "physical")
            )
            if physical_prediction is None:
                continue
            current = current_by_case_resolution[(case_id, resolution)]
            physical_update = physical_prediction - current
            for protocol in protocols:
                if protocol == "physical":
                    continue
                intervention = (
                    prediction_by_key[(case_id, resolution, protocol)]
                    - physical_prediction
                )
                for region_name, mask in regions.items():
                    if not bool(mask.any()):
                        continue
                    intervention_rms = weighted_scaled_rms(
                        intervention,
                        volumes=geometry.node_measures,
                        component_scale=normalization.residual_scale,
                        mask=mask,
                    )
                    physical_update_rms = weighted_scaled_rms(
                        physical_update,
                        volumes=geometry.node_measures,
                        component_scale=normalization.residual_scale,
                        mask=mask,
                    )
                    intervention_rows.append(
                        {
                            "case_id": case_id,
                            "resolution": resolution_label(resolution),
                            "node_type_protocol": protocol,
                            "region": region_name,
                            "region_node_count": int(mask.sum()),
                            "region_physical_volume": float(
                                geometry.node_measures[:, 0][mask].sum()
                            ),
                            "intervention_scaled_rms": intervention_rms,
                            "physical_update_scaled_rms": physical_update_rms,
                            "intervention_relative_to_physical_update": (
                                intervention_rms / physical_update_rms
                                if physical_update_rms > 1.0e-30
                                else None
                            ),
                        }
                    )

    write_csv(output_dir / "geometry.csv", geometry_rows)
    write_csv(output_dir / "predictions.csv", prediction_rows)
    write_csv(output_dir / "commutators.csv", commutator_rows)
    if intervention_rows:
        write_csv(output_dir / "node_type_interventions.csv", intervention_rows)

    summary = {
        "schema": SCHEMA,
        "status": "complete",
        "scientific_scope": {
            "stage": "R0/R0b",
            "target": "analytic_initial_condition_one_step_representation_diagnostic",
            "supports_pde_accuracy_claim": False,
            "supports_rollout_accuracy_claim": False,
            "claim_boundary": (
                "The same checkpoint was applied to regenerated discretizations "
                "of the same analytic initial state. Time-evolved common-reference "
                "R1 remains required for a resolution-generalization claim."
            ),
            "commutator_pairs": "successive nested resolutions after exact block-average restriction",
        },
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": checkpoint_sha256,
            "checkpoint_schema_version": checkpoint["checkpoint_schema_version"],
            "config_digest": checkpoint["config_digest"],
            "data_manifest_digest": checkpoint["data_manifest_digest"],
            "normalization_digest": checkpoint["normalization_digest"],
            "boundary_mode": checkpoint["boundary_mode"],
            "raw_recurrence": checkpoint["raw_recurrence"],
            "inference_interventions": checkpoint["inference_interventions"],
            "model_config": checkpoint["model_config"],
            "data_contract": checkpoint["data_contract"],
        },
        "source": {
            "git_head": git_head(),
            "file_sha256": source_hashes(),
        },
        "family": {
            "manifest_path": str(args.family_manifest),
            "manifest_sha256": sha256_file(args.family_manifest),
            "manifest_digest_sha256": manifest["manifest_digest_sha256"],
            "cases": case_provenance,
        },
        "translation_contract": {
            "resolutions": [resolution_label(value) for value in ordered_resolutions],
            "training_resolution": resolution_label(training_resolution),
            "state_sampling": (
                "tensor-product Gauss-Legendre conservative cell averages with "
                "shock-crossing cells split at the physical discontinuity"
            ),
            "quadrature_order": (
                first_config.initial_quadrature_order
                if args.quadrature_order is None
                else args.quadrature_order
            ),
            "geometry": (
                "physical cell centroids and volumes; regenerated face graph and "
                "least-squares differential weights independently on every grid"
            ),
            "node_type_semantics": {
                "0": "interior",
                "1": "touches y-symmetry boundary",
                "2": "touches x-extrapolation boundary",
                "3": "touches both; corner composition has both bits",
            },
            "node_type_protocols": protocols,
            "training_band_definition": (
                "tag every target cell intersecting a boundary strip one training "
                "cell wide in the corresponding physical direction"
            ),
            "node_type_pathways": {
                "pointwise": "one-hot channels are visible at every tagged node",
                "integral": (
                    "tagged features enter Fourier integrals through normalized "
                    "physical cell-volume weights"
                ),
                "differential": (
                    "one-hop least-squares gradient followed by two fixed graph-hop "
                    "neighbor-averaging iterations in each maintained PCNO layer"
                ),
            },
            "differential_support_warning": (
                "the maintained differential branch uses three graph hops of input "
                "support in total, so its physical support shrinks under refinement"
            ),
            "normalization": (
                "immutable checkpoint normalization; never refit by resolution"
            ),
            "Fourier_periods": list(args.expected_domain_lengths),
            "k_max": args.expected_k_max,
            "boundary_policy": "model_all_nodes raw recurrence",
            "batching": (
                "one unpadded homogeneous-resolution sample per forward call; "
                "no node-count weighting across grids"
            ),
        },
        "execution": {
            "device": str(device),
            "amp": amp,
            "repeat_forward": args.repeat_forward,
            "parameter_count": parameter_count(model),
            "geometry_build_seconds": geometry_seconds,
            "maximum_repeat_abs_difference": maximum_repeat_difference,
            "maximum_peak_gpu_memory_bytes": (
                maximum_peak_gpu_memory if device.type == "cuda" else None
            ),
            "gradient_layer_softsign_prefactors": [
                float(layer.gw1.detach().float().cpu()) for layer in model.backbone.gws
            ],
        },
        "geometry": geometry_rows,
        "predictions": prediction_rows,
        "commutators": commutator_rows,
        "node_type_interventions": intervention_rows,
        "aggregate": aggregate_rows(commutator_rows, intervention_rows),
        "artifacts": {
            "geometry_csv": "geometry.csv",
            "predictions_csv": "predictions.csv",
            "commutators_csv": "commutators.csv",
            "node_type_interventions_csv": (
                "node_type_interventions.csv" if intervention_rows else None
            ),
        },
    }
    atomic_write_json(output_dir / "summary.json", summary)
    print(f"wrote {output_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
