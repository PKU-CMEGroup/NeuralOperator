#!/usr/bin/env python3
"""Analyze residual-error accumulation for one frozen multiresolution PCNO.

The runner preserves the D063 physical boundary policy and evaluates only
explicitly named open cases.  Historical model/evaluator files are verified
against a retained summary before any forward pass.  Model-independent metric
code may be supplied beside this script through ``PCNO_DIAGNOSTIC_LIB`` so an
exact historical source tree need not be modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

root_override = os.environ.get("PCNO_REPO_ROOT")
ROOT = (
    Path(root_override).resolve()
    if root_override
    else Path(__file__).resolve().parents[2]
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
diagnostic_library = os.environ.get("PCNO_DIAGNOSTIC_LIB")
if diagnostic_library and diagnostic_library not in sys.path:
    sys.path.insert(0, diagnostic_library)

try:
    from utility.time_dependent_no.pcno_residual_structure import (
        CONSERVATIVE_COMPONENTS,
        algebra_identity_closure,
        characteristic_energy,
        component_rms,
        decomposition_diagnostics,
        phase_projection,
        recurrence_diagnostics,
        region_energy_rows,
        sequence_diagnostics,
        shock_vortex_regions,
        spectral_energy_rows,
        weighted_rms,
    )
except ImportError:
    from pcno_residual_structure import (  # type: ignore[no-redef]
        CONSERVATIVE_COMPONENTS,
        algebra_identity_closure,
        characteristic_energy,
        component_rms,
        decomposition_diagnostics,
        phase_projection,
        recurrence_diagnostics,
        region_energy_rows,
        sequence_diagnostics,
        shock_vortex_regions,
        spectral_energy_rows,
        weighted_rms,
    )

from utility.time_dependent_no.pcno_euler2d import PCNOEuler2DShardStore
from utility.time_dependent_no.pcno_resolution_transfer import (
    Resolution,
    build_resolution_geometry,
    initial_states_from_common_source,
    make_model_sample,
    node_types_for_protocol,
    parse_resolution,
    resolution_label,
    restrict_nested_state,
)
from utility.time_dependent_no.shock_vortex_family import (
    config_for_family_case,
    family_case_provenance,
    load_shock_vortex_family_manifest,
)

try:
    from utility.time_dependent_no.pcno_resolution_transfer import (
        as_model_state as _as_model_state,
    )
    from utility.time_dependent_no.pcno_resolution_transfer import (
        build_resolution_checkpoint_model as _build_model,
    )
    from utility.time_dependent_no.pcno_resolution_transfer import (
        checkpoint_step_stride as _step_stride,
    )
    from utility.time_dependent_no.pcno_resolution_transfer import (
        load_resolution_checkpoint as _load_checkpoint,
    )
    from utility.time_dependent_no.pcno_resolution_transfer import (
        load_resolution_reference as _load_reference,
    )
    from utility.time_dependent_no.pcno_resolution_transfer import (
        predict_resolution_sample as _predict,
    )
    from utility.time_dependent_no.pcno_resolution_transfer import (
        reference_at_resolution as _reference_at_resolution,
    )
    from utility.time_dependent_no.pcno_resolution_transfer import (
        validate_resolution_rollout_contract as _validate_contract,
    )
    from utility.time_dependent_no.pcno_runtime import select_device as _select_device

    MODERN_API = True
except ImportError:
    from scripts.time_dependent_no.evaluate_pcno_resolution_rollout import (
        _as_model_state,
        _load_reference,
        _predict,
        _reference_at_resolution,
        _step_stride,
        _validate_contract,
    )
    from scripts.time_dependent_no.evaluate_pcno_resolution_transfer import (
        build_model as _build_model,
    )
    from scripts.time_dependent_no.evaluate_pcno_resolution_transfer import (
        load_checkpoint as _load_checkpoint,
    )
    from scripts.time_dependent_no.evaluate_pcno_resolution_transfer import (
        select_device as _select_device,
    )

    MODERN_API = False


SCHEMA = "pcno_residual_structure_diagnostic_v1"
EXPECTED_BOUNDARY_POLICY = "model_all_nodes raw recurrence"
DEFAULT_RESOLUTIONS = ("125x50", "250x100", "500x200")
DEFAULT_CASE_IDS = tuple(
    f"sv_e{epsilon:02d}_y{position:02d}" for epsilon in range(12) for position in (0, 8)
)
DEFAULT_BUNDLE_CASE_IDS = (
    "sv_e00_y00",
    "sv_e00_y08",
    "sv_e06_y00",
    "sv_e06_y08",
    "sv_e11_y00",
    "sv_e11_y08",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family-root", type=Path, required=True)
    parser.add_argument("--multires-reference-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--retained-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-ids", nargs="+", default=list(DEFAULT_CASE_IDS))
    parser.add_argument(
        "--bundle-case-ids", nargs="+", default=list(DEFAULT_BUNDLE_CASE_IDS)
    )
    parser.add_argument(
        "--bundle-payload",
        choices=("standard", "pathway", "unified"),
        default="standard",
        help=(
            "standard saves the D064 increment panels; pathway saves only exact "
            "trajectories and the mesh/state decomposition; unified saves the "
            "pathway payload plus the minimum free-rollout fields needed for a "
            "self-consistent scale and profile diagnostic"
        ),
    )
    parser.add_argument("--resolutions", nargs="+", default=list(DEFAULT_RESOLUTIONS))
    parser.add_argument("--source-resolution", default="1000x400")
    parser.add_argument("--training-resolution", default="250x100")
    parser.add_argument("--rollout-calls", type=int, default=30)
    parser.add_argument("--quadrature-order", type=int)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-normalization-digest", required=True)
    parser.add_argument("--expected-step-stride", type=int, default=2)
    parser.add_argument("--expected-k-max", type=int, default=8)
    parser.add_argument(
        "--expected-domain-lengths", type=float, nargs=2, default=(2.0, 1.0)
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--amp", choices=("none", "bf16", "fp16"), default="none")
    parser.add_argument("--progress-every", type=int, default=1)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(json_ready(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path.name}")
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serialized = {
                key: (
                    json.dumps(json_ready(value), sort_keys=True)
                    if isinstance(value, (Mapping, list, tuple, np.ndarray))
                    else value
                )
                for key, value in row.items()
            }
            writer.writerow(serialized)


def _compat_model(
    checkpoint: Mapping[str, Any], device: torch.device
) -> tuple[Any, Any]:
    return _build_model(checkpoint, device)


def _compat_reference(
    family_root: Path,
    multires_reference_root: Path,
    store: PCNOEuler2DShardStore,
    manifest: Mapping[str, Any],
    case_id: str,
    training_resolution: Resolution,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return _load_reference(
        family_root,
        multires_reference_root,
        store,
        manifest,
        case_id,
        training_resolution=training_resolution,
    )


def _compat_predict(
    model: Any,
    sample: Mapping[str, torch.Tensor],
    state: np.ndarray,
    *,
    device: torch.device,
    amp: str,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    return _predict(
        model,
        sample,
        state,
        device=device,
        amp=amp,
        repeats=1,
    )


def _contract_namespace(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        checkpoint=args.checkpoint,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        expected_normalization_digest=args.expected_normalization_digest,
        expected_step_stride=args.expected_step_stride,
        expected_k_max=args.expected_k_max,
        expected_domain_lengths=tuple(args.expected_domain_lengths),
        protocols=("physical",),
        rollout_calls=args.rollout_calls,
        endpoint_physical_times=(0.6,),
        boundary_band_widths=(0.02, 0.05),
        shock_quantile=0.9,
        repeat_forward=1,
    )


def verify_retained_source(summary_path: Path) -> dict[str, str]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = summary.get("source", {}).get("file_sha256")
    if not isinstance(expected, Mapping) or not expected:
        raise ValueError("retained summary does not bind a source manifest")
    actual: dict[str, str] = {}
    for relative, digest in expected.items():
        path = ROOT / str(relative)
        if not path.is_file():
            raise FileNotFoundError(path)
        actual[str(relative)] = sha256_file(path)
        if actual[str(relative)].lower() != str(digest).lower():
            raise ValueError(f"retained source hash mismatch: {relative}")
    return actual


def _normalization_mapping(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    value = checkpoint.get("normalization")
    if not isinstance(value, Mapping):
        raise TypeError("checkpoint normalization metadata is missing")
    return value


def _reference_frames(
    reference: Mapping[str, np.ndarray],
    *,
    reference_resolution: Resolution,
    target_resolution: Resolution,
    stride: int,
    rollout_calls: int,
) -> np.ndarray:
    frames = []
    for call in range(rollout_calls + 1):
        state = _reference_at_resolution(
            reference["conservative_states"][call * stride],
            reference_resolution=reference_resolution,
            target_resolution=target_resolution,
        )
        if state is None:
            raise ValueError("reference is unavailable on an evaluation grid")
        frames.append(np.asarray(state, dtype=np.float64))
    return np.stack(frames)


def _restrict_sequence(
    sequence: np.ndarray,
    *,
    fine_resolution: Resolution,
    coarse_resolution: Resolution,
) -> np.ndarray:
    return np.asarray(
        restrict_nested_state(
            sequence,
            fine_resolution=fine_resolution,
            coarse_resolution=coarse_resolution,
        ),
        dtype=np.float64,
    )


def _record_sequence(
    *,
    case_id: str,
    kind: str,
    target: str,
    defects: np.ndarray,
    truth: np.ndarray,
    volumes: np.ndarray,
    residual_scale: np.ndarray,
    physical_dt: float,
    time_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
) -> None:
    rows, summary = sequence_diagnostics(
        defects,
        truth,
        volumes=volumes,
        component_scale=residual_scale,
    )
    for row in rows:
        row.update(
            {
                "case_id": case_id,
                "sequence_kind": kind,
                "target": target,
                "physical_time": row["step"] * physical_dt,
            }
        )
        time_rows.append(row)
    aggregate_rows.append(
        {
            "case_id": case_id,
            "sequence_kind": kind,
            "target": target,
            **{
                key: value
                for key, value in summary.items()
                if key not in {"uncentered_pod", "centered_pod", "lag_correlations"}
            },
            "uncentered_first_mode_energy_fraction": summary["uncentered_pod"][
                "first_mode_energy_fraction"
            ],
            "uncentered_first_three_energy_fraction": summary["uncentered_pod"][
                "first_three_energy_fraction"
            ],
            "uncentered_modes_for_95_percent": summary["uncentered_pod"][
                "modes_for_95_percent"
            ],
            "centered_first_mode_energy_fraction": summary["centered_pod"][
                "first_mode_energy_fraction"
            ],
            "centered_modes_for_95_percent": summary["centered_pod"][
                "modes_for_95_percent"
            ],
            "lag_correlations": summary["lag_correlations"],
        }
    )


def _record_accumulated_change(
    *,
    case_id: str,
    trajectory_kind: str,
    target: str,
    coarse_changes: np.ndarray,
    restricted_fine_changes: np.ndarray,
    reference_denominator: np.ndarray,
    volumes: np.ndarray,
    residual_scale: np.ndarray,
    physical_dt: float,
    rows: list[dict[str, Any]],
) -> None:
    if not (
        coarse_changes.shape
        == restricted_fine_changes.shape
        == reference_denominator.shape
    ):
        raise ValueError("accumulated-change trajectories must align")
    for step, (coarse, fine, denominator) in enumerate(
        zip(coarse_changes, restricted_fine_changes, reference_denominator),
        start=1,
    ):
        numerator_rms = weighted_rms(
            coarse - fine,
            volumes=volumes,
            component_scale=residual_scale,
        )
        denominator_rms = weighted_rms(
            denominator,
            volumes=volumes,
            component_scale=residual_scale,
        )
        rows.append(
            {
                "case_id": case_id,
                "trajectory_kind": trajectory_kind,
                "target": target,
                "step": step,
                "physical_time": step * physical_dt,
                "coarse_change_rms": weighted_rms(
                    coarse,
                    volumes=volumes,
                    component_scale=residual_scale,
                ),
                "restricted_fine_change_rms": weighted_rms(
                    fine,
                    volumes=volumes,
                    component_scale=residual_scale,
                ),
                "accumulated_change_gap_rms": numerator_rms,
                "reference_denominator_rms": denominator_rms,
                "relative_to_reference_change": (
                    numerator_rms / denominator_rms if denominator_rms != 0.0 else None
                ),
            }
        )


def _region_cache(
    reference_states: np.ndarray,
    nodes: np.ndarray,
    *,
    resolution: Resolution,
    gamma: float,
) -> list[tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]]:
    return [
        shock_vortex_regions(
            reference_states[step],
            nodes,
            resolution=resolution,
            gamma=gamma,
        )
        for step in range(reference_states.shape[0] - 1)
    ]


def _record_spatial_sequence(
    *,
    case_id: str,
    kind: str,
    target: str,
    defects: np.ndarray,
    reference_currents: np.ndarray,
    volumes: np.ndarray,
    nodes: np.ndarray,
    resolution: Resolution,
    domain_lengths: tuple[float, float],
    residual_scale: np.ndarray,
    gamma: float,
    physical_dt: float,
    cache: Sequence[tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]],
    spatial_rows: list[dict[str, Any]],
    spectral_rows: list[dict[str, Any]],
    characteristic_rows: list[dict[str, Any]],
    include_characteristics: bool,
) -> None:
    for step, defect in enumerate(np.asarray(defects, dtype=np.float64)):
        masks, phase_mode, normals = cache[step]
        prefix = {
            "case_id": case_id,
            "sequence_kind": kind,
            "target": target,
            "step": step + 1,
            "physical_time": (step + 1) * physical_dt,
        }
        components = component_rms(
            defect, volumes=volumes, component_scale=residual_scale
        )
        component_energy = np.square(components)
        total_component_energy = float(component_energy.sum())
        for name, rms, energy in zip(
            CONSERVATIVE_COMPONENTS, components, component_energy
        ):
            spatial_rows.append(
                {
                    **prefix,
                    "diagnostic": "conservative_component",
                    "region": name,
                    "component_rms": float(rms),
                    "energy_share": (
                        float(energy / total_component_energy)
                        if total_component_energy != 0.0
                        else None
                    ),
                }
            )
        for row in region_energy_rows(
            defect,
            volumes=volumes,
            component_scale=residual_scale,
            masks=masks,
        ):
            spatial_rows.append({**prefix, "diagnostic": "physical_region", **row})
        spatial_rows.append(
            {
                **prefix,
                "diagnostic": "shock_phase_projection",
                "region": "shock_envelope_le_0.05",
                **phase_projection(
                    defect,
                    phase_mode,
                    shock_mask=masks["shock_envelope_le_0.05"],
                    volumes=volumes,
                    component_scale=residual_scale,
                ),
            }
        )
        for row in spectral_energy_rows(
            defect,
            resolution=resolution,
            domain_lengths=domain_lengths,
            component_scale=residual_scale,
        ):
            spectral_rows.append({**prefix, **row})
        if include_characteristics:
            characteristic_rows.append(
                {
                    **prefix,
                    **characteristic_energy(
                        defect,
                        reference_currents[step],
                        normals,
                        shock_mask=masks["shock_envelope_le_0.05"],
                        gamma=gamma,
                        component_scale=residual_scale,
                    ),
                }
            )


def _update_visual_scales(
    scales: dict[str, dict[str, np.ndarray]],
    pair_name: str,
    reference_states: np.ndarray,
    residual_scale: np.ndarray,
) -> None:
    true_increment = np.diff(reference_states, axis=0) / residual_scale[None, None, :]
    cumulative_change = (reference_states[1:] - reference_states[0]) / residual_scale[
        None, None, :
    ]
    increment = np.max(np.abs(true_increment), axis=(0, 1))
    cumulative = np.max(np.abs(cumulative_change), axis=(0, 1))
    growth = np.max(np.square(true_increment), axis=(0, 1))
    if pair_name not in scales:
        scales[pair_name] = {
            "instantaneous": increment,
            "cumulative": cumulative,
            "growth": growth,
        }
    else:
        for key, value in (
            ("instantaneous", increment),
            ("cumulative", cumulative),
            ("growth", growth),
        ):
            scales[pair_name][key] = np.maximum(scales[pair_name][key], value)


def _bundle_pair(
    arrays: dict[str, np.ndarray],
    *,
    pair_name: str,
    true_increment: np.ndarray,
    free_coarse_increment: np.ndarray,
    free_fine_increment: np.ndarray,
    teacher_coarse_increment: np.ndarray,
    teacher_fine_increment: np.ndarray,
    free_error: np.ndarray,
    free_delta: np.ndarray,
    teacher_error: np.ndarray,
    teacher_delta: np.ndarray,
    free_state_gap: np.ndarray,
) -> None:
    key = pair_name.replace("->", "_to_").replace("x", "x")
    arrays[f"true_increment__{key}"] = np.asarray(true_increment, dtype=np.float32)
    for mode, coarse, fine, error, delta in (
        (
            "free",
            free_coarse_increment,
            free_fine_increment,
            free_error,
            free_delta,
        ),
        (
            "teacher",
            teacher_coarse_increment,
            teacher_fine_increment,
            teacher_error,
            teacher_delta,
        ),
    ):
        arrays[f"coarse_increment_{mode}__{key}"] = np.asarray(coarse, dtype=np.float32)
        arrays[f"fine_increment_{mode}__{key}"] = np.asarray(fine, dtype=np.float32)
        arrays[f"coarse_error_{mode}__{key}"] = np.asarray(error, dtype=np.float32)
        arrays[f"fine_error_{mode}__{key}"] = np.asarray(
            fine - true_increment, dtype=np.float32
        )
        arrays[f"delta_{mode}__{key}"] = np.asarray(delta, dtype=np.float32)
        arrays[f"cumulative_delta_{mode}__{key}"] = np.asarray(
            np.cumsum(delta, axis=0), dtype=np.float32
        )
    growth = 2.0 * free_state_gap[:-1] * free_delta + np.square(free_delta)
    arrays[f"growth_free__{key}"] = np.asarray(growth, dtype=np.float32)
    teacher_cumulative_before = np.concatenate(
        (
            np.zeros_like(teacher_delta[:1]),
            np.cumsum(teacher_delta[:-1], axis=0),
        ),
        axis=0,
    )
    teacher_growth = 2.0 * teacher_cumulative_before * teacher_delta + np.square(
        teacher_delta
    )
    arrays[f"growth_teacher__{key}"] = np.asarray(teacher_growth, dtype=np.float32)


def _bundle_pathway_pair(
    arrays: dict[str, np.ndarray],
    *,
    pair_name: str,
    nodes: np.ndarray,
    volumes: np.ndarray,
    reference_states: np.ndarray,
    free_coarse_states: np.ndarray,
    free_fine_states: np.ndarray,
    mesh_delta: np.ndarray,
    state_delta: np.ndarray,
) -> None:
    """Save the minimum exact replay payload absent from the D064 bundles."""

    key = pair_name.replace("->", "_to_").replace("x", "x")
    arrays[f"nodes__{key}"] = np.asarray(nodes, dtype=np.float64)
    arrays[f"volumes__{key}"] = np.asarray(volumes, dtype=np.float64)
    arrays[f"reference_states__{key}"] = np.asarray(reference_states, dtype=np.float64)
    arrays[f"free_coarse_states__{key}"] = np.asarray(
        free_coarse_states, dtype=np.float64
    )
    arrays[f"free_fine_states__{key}"] = np.asarray(free_fine_states, dtype=np.float64)
    arrays[f"mesh_delta__{key}"] = np.asarray(mesh_delta, dtype=np.float64)
    arrays[f"state_delta__{key}"] = np.asarray(state_delta, dtype=np.float64)


def _bundle_unified_pair(
    arrays: dict[str, np.ndarray],
    *,
    pair_name: str,
    nodes: np.ndarray,
    volumes: np.ndarray,
    reference_states: np.ndarray,
    free_coarse_states: np.ndarray,
    free_fine_states: np.ndarray,
    mesh_delta: np.ndarray,
    state_delta: np.ndarray,
) -> None:
    """Save one self-consistent free-rollout and pathway payload."""

    _bundle_pathway_pair(
        arrays,
        pair_name=pair_name,
        nodes=nodes,
        volumes=volumes,
        reference_states=reference_states,
        free_coarse_states=free_coarse_states,
        free_fine_states=free_fine_states,
        mesh_delta=mesh_delta,
        state_delta=state_delta,
    )
    key = pair_name.replace("->", "_to_").replace("x", "x")
    true_increment = np.diff(np.asarray(reference_states, dtype=np.float64), axis=0)
    coarse_increment = np.diff(np.asarray(free_coarse_states, dtype=np.float64), axis=0)
    fine_increment = np.diff(np.asarray(free_fine_states, dtype=np.float64), axis=0)
    delta = coarse_increment - fine_increment
    arrays[f"true_increment__{key}"] = true_increment
    arrays[f"coarse_increment_free__{key}"] = coarse_increment
    arrays[f"fine_increment_free__{key}"] = fine_increment
    arrays[f"delta_free__{key}"] = delta
    arrays[f"cumulative_delta_free__{key}"] = np.cumsum(delta, axis=0)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    bundle_dir = args.output_dir / "bundles"
    bundle_dir.mkdir()

    source_hashes = verify_retained_source(args.retained_summary)
    metric_utility_path = Path(weighted_rms.__code__.co_filename).resolve()
    diagnostic_source_hashes = {
        "runner": sha256_file(Path(__file__).resolve()),
        "metric_utility": sha256_file(metric_utility_path),
    }
    checkpoint_sha = sha256_file(args.checkpoint)
    if checkpoint_sha.lower() != args.expected_checkpoint_sha256.lower():
        raise ValueError("checkpoint SHA-256 mismatch")
    checkpoint = _load_checkpoint(args.checkpoint)
    if checkpoint.get("normalization_digest") != args.expected_normalization_digest:
        raise ValueError("checkpoint normalization digest mismatch")
    recomputed_normalization_digest = canonical_digest(
        _normalization_mapping(checkpoint)
    )
    if recomputed_normalization_digest != args.expected_normalization_digest:
        raise ValueError("checkpoint normalization metadata digest does not verify")

    manifest = load_shock_vortex_family_manifest(
        args.family_root / "family_manifest.json"
    )
    store = PCNOEuler2DShardStore(args.data_dir, max_cached_trajectories=1)
    resolutions = sorted(
        [parse_resolution(value) for value in args.resolutions],
        key=lambda value: value[0] * value[1],
    )
    source_resolution = parse_resolution(args.source_resolution)
    training_resolution = parse_resolution(args.training_resolution)
    stride, rollout_calls, physical_dt, _ = _validate_contract(
        _contract_namespace(args),
        checkpoint,
        manifest,
        store,
        resolutions,
        source_resolution,
        training_resolution,
    )
    if rollout_calls != args.rollout_calls or stride != args.expected_step_stride:
        raise AssertionError("validated time contract differs from requested contract")
    if _step_stride(checkpoint) != stride:
        raise AssertionError("checkpoint stride changed after validation")
    case_ids = list(dict.fromkeys(args.case_ids))
    bundle_cases = set(args.bundle_case_ids)
    if not bundle_cases.issubset(case_ids):
        raise ValueError("every bundle case must be part of the evaluated population")
    for case_id in case_ids:
        provenance = family_case_provenance(manifest, case_id)
        if provenance["split"] != "validation":
            raise ValueError(f"case is not open validation: {case_id}")
        if case_id not in store.keys:
            raise ValueError(f"case is absent from checkpoint-bound shards: {case_id}")

    device = _select_device(args.device)
    if args.amp != "none" and device.type != "cuda":
        raise ValueError("mixed precision requires CUDA")
    model, normalization = _compat_model(checkpoint, device)
    residual_scale = np.asarray(normalization.residual_scale, dtype=np.float64)
    state_scale = np.asarray(normalization.state_scale, dtype=np.float64)
    first_config = config_for_family_case(manifest, case_ids[0])
    domain_lengths = (
        float(first_config.x_max - first_config.x_min),
        float(first_config.y_max - first_config.y_min),
    )

    geometry_by_resolution: dict[Resolution, Any] = {}
    sample_by_resolution: dict[Resolution, Mapping[str, torch.Tensor]] = {}
    for resolution in resolutions:
        config, geometry = build_resolution_geometry(first_config, resolution)
        node_type = node_types_for_protocol(
            geometry,
            config,
            "physical",
            training_resolution=training_resolution,
        )
        geometry_by_resolution[resolution] = geometry
        sample_by_resolution[resolution] = make_model_sample(
            geometry,
            node_type,
            mach=first_config.shock_mach,
            device=device,
        )

    time_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    growth_rows: list[dict[str, Any]] = []
    decomposition_rows: list[dict[str, Any]] = []
    closure_rows: list[dict[str, Any]] = []
    spatial_rows: list[dict[str, Any]] = []
    spectral_rows: list[dict[str, Any]] = []
    characteristic_rows: list[dict[str, Any]] = []
    reference_check_rows: list[dict[str, Any]] = []
    accumulated_change_rows: list[dict[str, Any]] = []
    completion_rows: list[dict[str, Any]] = []
    visual_scales: dict[str, dict[str, np.ndarray]] = {}
    started = perf_counter()

    try:
        for case_index, case_id in enumerate(case_ids, start=1):
            case_started = perf_counter()
            case_config = config_for_family_case(manifest, case_id)
            reference, reference_check = _compat_reference(
                args.family_root,
                args.multires_reference_root,
                store,
                manifest,
                case_id,
                training_resolution,
            )
            reference_check_rows.append(reference_check)
            reference_resolution = tuple(
                int(value) for value in reference["retained_resolution"]
            )
            _, common_states = initial_states_from_common_source(
                case_config,
                resolutions,
                source_resolution=source_resolution,
                dtype=torch.float64,
                quadrature_order=args.quadrature_order,
            )
            reference_by_resolution = {
                resolution: _reference_frames(
                    reference,
                    reference_resolution=reference_resolution,
                    target_resolution=resolution,
                    stride=stride,
                    rollout_calls=rollout_calls,
                )
                for resolution in resolutions
            }
            native_initial = np.asarray(
                common_states[training_resolution], dtype=np.float32
            )
            if not np.array_equal(native_initial, np.asarray(store.states(case_id)[0])):
                raise ValueError(f"checkpoint-bound t0 mismatch: {case_id}")

            free_states: dict[Resolution, np.ndarray] = {}
            teacher_increments: dict[Resolution, np.ndarray] = {}
            for resolution in resolutions:
                node_count = resolution[0] * resolution[1]
                trajectory = np.empty(
                    (rollout_calls + 1, node_count, 4), dtype=np.float32
                )
                trajectory[0] = np.asarray(
                    _as_model_state(common_states[resolution]), dtype=np.float32
                )
                for step in range(rollout_calls):
                    prediction, _ = _compat_predict(
                        model,
                        sample_by_resolution[resolution],
                        trajectory[step],
                        device=device,
                        amp=args.amp,
                    )
                    trajectory[step + 1] = np.asarray(prediction, dtype=np.float32)
                free_states[resolution] = trajectory

                increments = np.empty((rollout_calls, node_count, 4), dtype=np.float32)
                reference_states = reference_by_resolution[resolution]
                for step in range(rollout_calls):
                    teacher_current = _as_model_state(reference_states[step])
                    teacher_prediction, _ = _compat_predict(
                        model,
                        sample_by_resolution[resolution],
                        teacher_current,
                        device=device,
                        amp=args.amp,
                    )
                    increments[step] = np.asarray(
                        teacher_prediction - teacher_current, dtype=np.float32
                    )
                teacher_increments[resolution] = increments

            local_increments: dict[tuple[Resolution, Resolution], np.ndarray] = {}
            for coarse, fine in pairwise(resolutions):
                local = np.empty(
                    (rollout_calls, coarse[0] * coarse[1], 4), dtype=np.float32
                )
                for step in range(rollout_calls):
                    local_current = _as_model_state(
                        _restrict_sequence(
                            free_states[fine][step : step + 1],
                            fine_resolution=fine,
                            coarse_resolution=coarse,
                        )[0]
                    )
                    local_prediction, _ = _compat_predict(
                        model,
                        sample_by_resolution[coarse],
                        local_current,
                        device=device,
                        amp=args.amp,
                    )
                    local[step] = np.asarray(
                        local_prediction - local_current, dtype=np.float32
                    )
                local_increments[(coarse, fine)] = local

            region_cache: dict[
                Resolution,
                list[tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]],
            ] = {
                resolution: _region_cache(
                    reference_by_resolution[resolution],
                    geometry_by_resolution[resolution].nodes,
                    resolution=resolution,
                    gamma=float(normalization.gamma),
                )
                for resolution in resolutions
            }
            bundle_arrays: dict[str, np.ndarray] = {
                "physical_times": np.arange(1, rollout_calls + 1, dtype=np.float64)
                * physical_dt,
                "residual_scale": residual_scale,
                "state_scale": state_scale,
            }

            for resolution in resolutions:
                label = resolution_label(resolution)
                volumes = geometry_by_resolution[resolution].node_measures
                reference_states = reference_by_resolution[resolution]
                true_increment = np.diff(reference_states, axis=0)
                free_increment = np.diff(
                    np.asarray(free_states[resolution], dtype=np.float64), axis=0
                )
                teacher_increment = np.asarray(
                    teacher_increments[resolution], dtype=np.float64
                )
                free_error = free_increment - true_increment
                teacher_error = teacher_increment - true_increment
                state_response = free_increment - teacher_increment
                for kind, defects in (
                    ("per_grid_free_residual_error", free_error),
                    ("per_grid_teacher_residual_error", teacher_error),
                    ("per_grid_recurrence_state_response", state_response),
                ):
                    _record_sequence(
                        case_id=case_id,
                        kind=kind,
                        target=label,
                        defects=defects,
                        truth=true_increment,
                        volumes=volumes,
                        residual_scale=residual_scale,
                        physical_dt=physical_dt,
                        time_rows=time_rows,
                        aggregate_rows=aggregate_rows,
                    )
                state_decomposition_closure = (
                    free_error - teacher_error - state_response
                )
                closure_rows.append(
                    {
                        "case_id": case_id,
                        "closure_kind": "free_equals_teacher_plus_state",
                        "target": label,
                        "maximum_residual_scaled_rms": max(
                            weighted_rms(
                                value,
                                volumes=volumes,
                                component_scale=residual_scale,
                            )
                            for value in state_decomposition_closure
                        ),
                    }
                )
                state_error = (
                    np.asarray(free_states[resolution], dtype=np.float64)
                    - reference_states
                )
                growth, closure = recurrence_diagnostics(
                    state_error,
                    free_error,
                    volumes=volumes,
                    component_scale=residual_scale,
                )
                for row in growth:
                    row.update(
                        {
                            "case_id": case_id,
                            "sequence_kind": "per_grid_free_residual_error",
                            "target": label,
                            "physical_time": row["step"] * physical_dt,
                        }
                    )
                    growth_rows.append(row)
                closure_rows.append(
                    {
                        "case_id": case_id,
                        "closure_kind": "per_grid_free_recurrence",
                        "target": label,
                        **closure,
                    }
                )
                for kind, defects in (
                    ("per_grid_free_residual_error", free_error),
                    ("per_grid_teacher_residual_error", teacher_error),
                ):
                    _record_spatial_sequence(
                        case_id=case_id,
                        kind=kind,
                        target=label,
                        defects=defects,
                        reference_currents=reference_states[:-1],
                        volumes=volumes,
                        nodes=geometry_by_resolution[resolution].nodes,
                        resolution=resolution,
                        domain_lengths=domain_lengths,
                        residual_scale=residual_scale,
                        gamma=float(normalization.gamma),
                        physical_dt=physical_dt,
                        cache=region_cache[resolution],
                        spatial_rows=spatial_rows,
                        spectral_rows=spectral_rows,
                        characteristic_rows=characteristic_rows,
                        include_characteristics=False,
                    )

            for coarse, fine in pairwise(resolutions):
                coarse_label = resolution_label(coarse)
                fine_label = resolution_label(fine)
                pair_name = f"{coarse_label}->{fine_label}"
                volumes = geometry_by_resolution[coarse].node_measures
                reference_states = reference_by_resolution[coarse]
                true_increment = np.diff(reference_states, axis=0)
                free_coarse_states = np.asarray(free_states[coarse], dtype=np.float64)
                free_fine_states = _restrict_sequence(
                    free_states[fine],
                    fine_resolution=fine,
                    coarse_resolution=coarse,
                )
                reference_fine_states = _restrict_sequence(
                    reference_by_resolution[fine],
                    fine_resolution=fine,
                    coarse_resolution=coarse,
                )
                reference_coarse_change = reference_states[1:] - reference_states[0]
                reference_fine_change = (
                    reference_fine_states[1:] - reference_fine_states[0]
                )
                free_coarse_change = free_coarse_states[1:] - free_coarse_states[0]
                free_fine_change = free_fine_states[1:] - free_fine_states[0]
                _record_accumulated_change(
                    case_id=case_id,
                    trajectory_kind="reference",
                    target=pair_name,
                    coarse_changes=reference_coarse_change,
                    restricted_fine_changes=reference_fine_change,
                    reference_denominator=reference_fine_change,
                    volumes=volumes,
                    residual_scale=residual_scale,
                    physical_dt=physical_dt,
                    rows=accumulated_change_rows,
                )
                _record_accumulated_change(
                    case_id=case_id,
                    trajectory_kind="free_prediction",
                    target=pair_name,
                    coarse_changes=free_coarse_change,
                    restricted_fine_changes=free_fine_change,
                    reference_denominator=reference_fine_change,
                    volumes=volumes,
                    residual_scale=residual_scale,
                    physical_dt=physical_dt,
                    rows=accumulated_change_rows,
                )
                free_coarse_increment = np.diff(free_coarse_states, axis=0)
                free_fine_increment = np.diff(free_fine_states, axis=0)
                teacher_coarse_increment = np.asarray(
                    teacher_increments[coarse], dtype=np.float64
                )
                teacher_fine_increment = _restrict_sequence(
                    teacher_increments[fine],
                    fine_resolution=fine,
                    coarse_resolution=coarse,
                )
                free_delta = free_coarse_increment - free_fine_increment
                teacher_delta = teacher_coarse_increment - teacher_fine_increment
                mesh_delta = (
                    np.asarray(local_increments[(coarse, fine)], dtype=np.float64)
                    - free_fine_increment
                )
                state_delta = free_coarse_increment - np.asarray(
                    local_increments[(coarse, fine)], dtype=np.float64
                )
                accumulated_closure = (
                    free_coarse_change
                    - free_fine_change
                    - np.cumsum(free_delta, axis=0)
                )
                closure_rows.append(
                    {
                        "case_id": case_id,
                        "closure_kind": (
                            "free_accumulated_change_equals_cumulative_defect"
                        ),
                        "target": pair_name,
                        "maximum_residual_scaled_rms": max(
                            weighted_rms(
                                value,
                                volumes=volumes,
                                component_scale=residual_scale,
                            )
                            for value in accumulated_closure
                        ),
                    }
                )
                for kind, defects in (
                    ("free_cross_grid_defect", free_delta),
                    ("teacher_cross_grid_defect", teacher_delta),
                    ("free_mesh_defect", mesh_delta),
                    ("free_state_defect", state_delta),
                ):
                    _record_sequence(
                        case_id=case_id,
                        kind=kind,
                        target=pair_name,
                        defects=defects,
                        truth=true_increment,
                        volumes=volumes,
                        residual_scale=residual_scale,
                        physical_dt=physical_dt,
                        time_rows=time_rows,
                        aggregate_rows=aggregate_rows,
                    )
                decomposition, decomposition_closure = decomposition_diagnostics(
                    free_delta,
                    mesh_delta,
                    state_delta,
                    volumes=volumes,
                    component_scale=residual_scale,
                )
                for row in decomposition:
                    row.update(
                        {
                            "case_id": case_id,
                            "target": pair_name,
                            "physical_time": row["step"] * physical_dt,
                        }
                    )
                    decomposition_rows.append(row)
                closure_rows.append(
                    {
                        "case_id": case_id,
                        "closure_kind": "free_mesh_plus_state",
                        "target": pair_name,
                        **decomposition_closure,
                    }
                )
                free_gap = free_coarse_states - free_fine_states
                growth, recurrence_closure = recurrence_diagnostics(
                    free_gap,
                    free_delta,
                    volumes=volumes,
                    component_scale=residual_scale,
                )
                for row in growth:
                    row.update(
                        {
                            "case_id": case_id,
                            "sequence_kind": "free_cross_grid_defect",
                            "target": pair_name,
                            "physical_time": row["step"] * physical_dt,
                        }
                    )
                    growth_rows.append(row)
                closure_rows.append(
                    {
                        "case_id": case_id,
                        "closure_kind": "free_cross_grid_recurrence",
                        "target": pair_name,
                        **recurrence_closure,
                    }
                )
                algebra_max = 0.0
                teacher_algebra_max = 0.0
                teacher_current_coarse = np.asarray(
                    reference_by_resolution[coarse][:-1], dtype=np.float32
                ).astype(np.float64)
                teacher_current_fine = _restrict_sequence(
                    np.asarray(reference_by_resolution[fine][:-1], dtype=np.float32),
                    fine_resolution=fine,
                    coarse_resolution=coarse,
                )
                teacher_prediction_coarse = (
                    teacher_current_coarse + teacher_coarse_increment
                )
                teacher_prediction_fine = teacher_current_fine + teacher_fine_increment
                for step in range(rollout_calls):
                    free_closure = algebra_identity_closure(
                        free_coarse_states[step],
                        free_coarse_states[step + 1],
                        free_fine_states[step],
                        free_fine_states[step + 1],
                    )
                    teacher_closure = algebra_identity_closure(
                        teacher_current_coarse[step],
                        teacher_prediction_coarse[step],
                        teacher_current_fine[step],
                        teacher_prediction_fine[step],
                    )
                    algebra_max = max(
                        algebra_max,
                        weighted_rms(
                            free_closure,
                            volumes=volumes,
                            component_scale=residual_scale,
                        ),
                    )
                    teacher_algebra_max = max(
                        teacher_algebra_max,
                        weighted_rms(
                            teacher_closure,
                            volumes=volumes,
                            component_scale=residual_scale,
                        ),
                    )
                closure_rows.extend(
                    (
                        {
                            "case_id": case_id,
                            "closure_kind": "free_generalized_algebra_identity",
                            "target": pair_name,
                            "maximum_residual_scaled_rms": algebra_max,
                        },
                        {
                            "case_id": case_id,
                            "closure_kind": "teacher_generalized_algebra_identity",
                            "target": pair_name,
                            "maximum_residual_scaled_rms": teacher_algebra_max,
                        },
                    )
                )
                for kind, defects, include_characteristics in (
                    ("free_cross_grid_defect", free_delta, True),
                    ("teacher_cross_grid_defect", teacher_delta, False),
                    ("free_mesh_defect", mesh_delta, False),
                    ("free_state_defect", state_delta, False),
                ):
                    _record_spatial_sequence(
                        case_id=case_id,
                        kind=kind,
                        target=pair_name,
                        defects=defects,
                        reference_currents=reference_states[:-1],
                        volumes=volumes,
                        nodes=geometry_by_resolution[coarse].nodes,
                        resolution=coarse,
                        domain_lengths=domain_lengths,
                        residual_scale=residual_scale,
                        gamma=float(normalization.gamma),
                        physical_dt=physical_dt,
                        cache=region_cache[coarse],
                        spatial_rows=spatial_rows,
                        spectral_rows=spectral_rows,
                        characteristic_rows=characteristic_rows,
                        include_characteristics=include_characteristics,
                    )
                _update_visual_scales(
                    visual_scales, pair_name, reference_states, residual_scale
                )
                if case_id in bundle_cases:
                    if args.bundle_payload == "standard":
                        _bundle_pair(
                            bundle_arrays,
                            pair_name=pair_name,
                            true_increment=true_increment,
                            free_coarse_increment=free_coarse_increment,
                            free_fine_increment=free_fine_increment,
                            teacher_coarse_increment=teacher_coarse_increment,
                            teacher_fine_increment=teacher_fine_increment,
                            free_error=free_coarse_increment - true_increment,
                            free_delta=free_delta,
                            teacher_error=teacher_coarse_increment - true_increment,
                            teacher_delta=teacher_delta,
                            free_state_gap=free_gap,
                        )
                    elif args.bundle_payload == "pathway":
                        _bundle_pathway_pair(
                            bundle_arrays,
                            pair_name=pair_name,
                            nodes=geometry_by_resolution[coarse].nodes,
                            volumes=volumes,
                            reference_states=reference_states,
                            free_coarse_states=free_coarse_states,
                            free_fine_states=free_fine_states,
                            mesh_delta=mesh_delta,
                            state_delta=state_delta,
                        )
                    else:
                        _bundle_unified_pair(
                            bundle_arrays,
                            pair_name=pair_name,
                            nodes=geometry_by_resolution[coarse].nodes,
                            volumes=volumes,
                            reference_states=reference_states,
                            free_coarse_states=free_coarse_states,
                            free_fine_states=free_fine_states,
                            mesh_delta=mesh_delta,
                            state_delta=state_delta,
                        )

            if case_id in bundle_cases:
                metadata = {
                    "schema": SCHEMA,
                    "case_id": case_id,
                    "bundle_payload": args.bundle_payload,
                    "bundle_storage": (
                        "npz_uncompressed"
                        if args.bundle_payload == "unified"
                        else "npz_deflate"
                    ),
                    "boundary_policy": EXPECTED_BOUNDARY_POLICY,
                    "checkpoint_sha256": checkpoint_sha,
                    "normalization_digest": recomputed_normalization_digest,
                    "source_hashes": source_hashes,
                    "diagnostic_source_hashes": diagnostic_source_hashes,
                    "resolutions": [resolution_label(value) for value in resolutions],
                    "physical_dt": physical_dt,
                    "rollout_calls": rollout_calls,
                }
                bundle_arrays["metadata_json"] = np.asarray(
                    json.dumps(metadata, sort_keys=True, separators=(",", ":"))
                )
                bundle_path = bundle_dir / f"{case_id}.npz"
                if args.bundle_payload == "unified":
                    np.savez(bundle_path, **bundle_arrays)
                else:
                    np.savez_compressed(bundle_path, **bundle_arrays)
            completion_rows.append(
                {
                    "case_id": case_id,
                    "status": "complete",
                    "seconds": perf_counter() - case_started,
                    "bundle_written": case_id in bundle_cases,
                }
            )
            if case_index % args.progress_every == 0:
                print(
                    f"completed {case_index}/{len(case_ids)} {case_id} "
                    f"seconds={completion_rows[-1]['seconds']:.1f}",
                    flush=True,
                )
            del reference, reference_by_resolution, free_states, teacher_increments
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        store.close()

    write_csv(args.output_dir / "time_metrics.csv", time_rows)
    write_csv(args.output_dir / "aggregate_metrics.csv", aggregate_rows)
    write_csv(args.output_dir / "growth_metrics.csv", growth_rows)
    write_csv(args.output_dir / "decomposition_metrics.csv", decomposition_rows)
    write_csv(args.output_dir / "closure_metrics.csv", closure_rows)
    write_csv(args.output_dir / "spatial_metrics.csv", spatial_rows)
    write_csv(args.output_dir / "spectral_metrics.csv", spectral_rows)
    write_csv(args.output_dir / "characteristic_metrics.csv", characteristic_rows)
    write_csv(args.output_dir / "reference_checks.csv", reference_check_rows)
    write_csv(
        args.output_dir / "accumulated_change_metrics.csv", accumulated_change_rows
    )
    write_csv(args.output_dir / "completion.csv", completion_rows)
    visual_scale_payload = {
        pair: {name: value.tolist() for name, value in values.items()}
        for pair, values in visual_scales.items()
    }
    write_json(args.output_dir / "visual_scales.json", visual_scale_payload)
    output_hashes = {
        path.name: sha256_file(path) for path in sorted(args.output_dir.glob("*.csv"))
    }
    output_hashes["visual_scales.json"] = sha256_file(
        args.output_dir / "visual_scales.json"
    )
    output_hashes.update(
        {
            str(path.relative_to(args.output_dir)): sha256_file(path)
            for path in sorted(bundle_dir.glob("*.npz"))
        }
    )
    summary = {
        "schema": SCHEMA,
        "status": "complete",
        "boundary_policy": EXPECTED_BOUNDARY_POLICY,
        "bundle_payload": args.bundle_payload,
        "bundle_storage": (
            "npz_uncompressed" if args.bundle_payload == "unified" else "npz_deflate"
        ),
        "population": {
            "split": "validation",
            "case_ids": case_ids,
            "sealed_populations_accessed": False,
        },
        "checkpoint_sha256": checkpoint_sha,
        "normalization_digest": recomputed_normalization_digest,
        "state_scale": state_scale,
        "residual_scale": residual_scale,
        "source_hashes": source_hashes,
        "diagnostic_source_hashes": diagnostic_source_hashes,
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "cuda_device": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "amp": args.amp,
            "historical_api": not MODERN_API,
        },
        "time_contract": {
            "stride": stride,
            "physical_dt": physical_dt,
            "rollout_calls": rollout_calls,
        },
        "row_counts": {
            "time": len(time_rows),
            "aggregate": len(aggregate_rows),
            "growth": len(growth_rows),
            "decomposition": len(decomposition_rows),
            "closure": len(closure_rows),
            "spatial": len(spatial_rows),
            "spectral": len(spectral_rows),
            "characteristic": len(characteristic_rows),
            "accumulated_change": len(accumulated_change_rows),
        },
        "elapsed_seconds": perf_counter() - started,
        "output_hashes": output_hashes,
    }
    write_json(args.output_dir / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    summary = run(parse_args(argv))
    print(
        f"wrote residual diagnostic: cases={len(summary['population']['case_ids'])} "
        f"seconds={summary['elapsed_seconds']:.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
