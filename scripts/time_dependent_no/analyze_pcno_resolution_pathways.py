#!/usr/bin/env python3
"""Localize a frozen PCNO cross-resolution defect by internal pathway.

The experiment is teacher-forced on already-open common-source dynamic-FV
references.  Intermediate traces are descriptive; paired branch-gain responses
are frozen local-sensitivity interventions, not trained ablations or remedies.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

ROOT = Path(
    os.environ.get("PCNO_REPO_ROOT", Path(__file__).resolve().parents[2])
).resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
diagnostic_library = os.environ.get("PCNO_DIAGNOSTIC_LIB")
if diagnostic_library and diagnostic_library not in sys.path:
    sys.path.insert(0, diagnostic_library)

try:
    from utility.time_dependent_no.pcno_resolution_pathways import (
        BRANCH_NAMES,
        generic_spectral_energy_rows,
        select_dominant_pathway,
        trace_backbone_output,
        trace_resolution_pair,
    )
except ImportError:
    from pcno_resolution_pathways import (  # type: ignore[no-redef]
        BRANCH_NAMES,
        generic_spectral_energy_rows,
        select_dominant_pathway,
        trace_backbone_output,
        trace_resolution_pair,
    )

try:
    from utility.time_dependent_no.pcno_residual_structure import (
        cosine,
        region_energy_rows,
        shock_vortex_regions,
        spectral_energy_rows,
        weighted_rms,
    )
except ImportError:
    from pcno_residual_structure import (  # type: ignore[no-redef]
        cosine,
        region_energy_rows,
        shock_vortex_regions,
        spectral_energy_rows,
        weighted_rms,
    )

try:
    from scripts.time_dependent_no.analyze_pcno_residual_structure import (
        _as_model_state,
        _compat_model,
        _compat_reference,
        _contract_namespace,
        _load_checkpoint,
        _normalization_mapping,
        _reference_frames,
        _select_device,
        _step_stride,
        _validate_contract,
        canonical_digest,
        sha256_file,
        verify_retained_source,
        write_csv,
        write_json,
    )
except ImportError:
    from analyze_pcno_residual_structure import (  # type: ignore[no-redef]
        _as_model_state,
        _compat_model,
        _compat_reference,
        _contract_namespace,
        _load_checkpoint,
        _normalization_mapping,
        _reference_frames,
        _select_device,
        _step_stride,
        _validate_contract,
        canonical_digest,
        sha256_file,
        verify_retained_source,
        write_csv,
        write_json,
    )
from utility.time_dependent_no.pcno_euler2d import (
    PCNOEuler2DShardStore,
    conservative_admissibility,
)
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

SCHEMA = "pcno_resolution_pathway_diagnostic_v2"
EXPECTED_BOUNDARY_POLICY = "model_all_nodes raw recurrence"
DEFAULT_RESOLUTIONS = ("125x50", "250x100", "500x200")
DEFAULT_CASE_IDS = (
    "sv_e00_y00",
    "sv_e00_y08",
    "sv_e06_y00",
    "sv_e06_y08",
    "sv_e11_y00",
    "sv_e11_y08",
)
DEFAULT_BUNDLE_CASE_IDS = ("sv_e00_y00", "sv_e11_y08")
DEFAULT_SENSITIVITY_CALLS = (1, 5, 15, 30)
GAIN = 0.99
REPLAY_NORMALIZED_TOLERANCE = 2.0e-6
REPLAY_PHYSICAL_TOLERANCE = 1.0e-7
WRAPPER_TOLERANCE = 5.0e-7
ALGEBRA_TOLERANCE = 2.0e-5
POINTWISE_FLOAT64_COMMUTATION_TOLERANCE = 1.0e-12
POINTWISE_FLOAT32_BRANCH_TOLERANCE = 5.0e-4
POINTWISE_FLOAT32_IMPACT_TOLERANCE = 1.0e-2
COMMUTATOR_IDENTITY_TOLERANCE = 2.0e-6
PAIRED_INPUT_RESTRICTION_TOLERANCE = 2.0e-6
REFERENCE_FLOOR_TOLERANCE = 1.0e-10


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
    parser.add_argument("--resolutions", nargs="+", default=list(DEFAULT_RESOLUTIONS))
    parser.add_argument("--source-resolution", default="1000x400")
    parser.add_argument("--training-resolution", default="250x100")
    parser.add_argument("--rollout-calls", type=int, default=30)
    parser.add_argument(
        "--sensitivity-calls",
        type=int,
        nargs="+",
        default=list(DEFAULT_SENSITIVITY_CALLS),
    )
    parser.add_argument("--gain", type=float, default=GAIN)
    parser.add_argument("--quadrature-order", type=int)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-normalization-digest", required=True)
    parser.add_argument("--expected-step-stride", type=int, default=2)
    parser.add_argument("--expected-k-max", type=int, default=8)
    parser.add_argument(
        "--expected-domain-lengths", type=float, nargs=2, default=(2.0, 1.0)
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--amp", choices=("none",), default="none")
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run one preregistered case/call without mechanism selection",
    )
    return parser.parse_args(argv)


def _pair_label(coarse: Resolution, fine: Resolution) -> str:
    return f"{resolution_label(coarse)}->{resolution_label(fine)}"


def _pair_key(pair: str) -> str:
    return pair.replace("->", "_to_")


def _restrict(value: np.ndarray, *, fine: Resolution, coarse: Resolution) -> np.ndarray:
    return np.asarray(
        restrict_nested_state(
            value,
            fine_resolution=fine,
            coarse_resolution=coarse,
        ),
        dtype=np.float64,
    )


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0.0 else float(numerator / denominator)


def _sample_aux(sample: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    return (
        sample["node_mask"],
        sample["nodes"],
        sample["node_weights"],
        sample["directed_edges"],
        sample["edge_gradient_weights"],
    )


def _model_tensors(
    model: torch.nn.Module,
    sample: Mapping[str, torch.Tensor],
    state: np.ndarray,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    current = torch.as_tensor(
        np.asarray(state, dtype=np.float32), dtype=torch.float32, device=device
    ).unsqueeze(0)
    model_input = model.normalized_input(
        current,
        nodes=sample["nodes"],
        node_rhos=sample["node_rhos"],
        node_type=sample["node_type"],
        mach=sample["mach"],
    )
    return current, model_input, _sample_aux(sample)


@torch.no_grad()
def _direct_prediction(
    model: torch.nn.Module,
    sample: Mapping[str, torch.Tensor],
    current: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    captured: list[torch.Tensor] = []

    def capture_backbone_output(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        captured.append(output.detach())

    handle = model.backbone.register_forward_hook(capture_backbone_output)
    try:
        prediction = model(
            current,
            node_mask=sample["node_mask"],
            nodes=sample["nodes"],
            node_weights=sample["node_weights"],
            node_rhos=sample["node_rhos"],
            directed_edges=sample["directed_edges"],
            edge_gradient_weights=sample["edge_gradient_weights"],
            node_type=sample["node_type"],
            mach=sample["mach"],
        )
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError("the residual wrapper must call its backbone exactly once")
    return prediction, captured[0]


@torch.no_grad()
def _gain_prediction(
    model: torch.nn.Module,
    current: torch.Tensor,
    model_input: torch.Tensor,
    aux: Sequence[torch.Tensor],
    gains: Mapping[tuple[int, str], float],
) -> torch.Tensor:
    normalized = trace_backbone_output(
        model.backbone, model_input, aux, branch_gains=gains
    )
    return (current + normalized * model.residual_scale) * aux[0]


def _numpy(value: torch.Tensor) -> np.ndarray:
    return np.asarray(value.detach().cpu().numpy()[0], dtype=np.float64)


def _admissibility(prediction: torch.Tensor, *, gamma: float) -> dict[str, Any]:
    diagnostics = conservative_admissibility(prediction, gamma=gamma)
    finite_components = diagnostics.get("finite_components")
    if finite_components is None:
        finite_components = diagnostics["finite"]
    density = diagnostics.get("density")
    if density is None:
        density = diagnostics["rho"]
    return {
        "finite": bool(finite_components.all().detach().cpu()),
        "admissible": bool(diagnostics["admissible"].all().detach().cpu()),
        "minimum_density": float(density.min().detach().cpu()),
        "minimum_pressure": float(diagnostics["pressure"].min().detach().cpu()),
    }


def _metric(
    value: np.ndarray,
    *,
    volumes: np.ndarray,
    residual_scale: np.ndarray,
) -> float:
    return weighted_rms(value, volumes=volumes, component_scale=residual_scale)


def _output_row(
    *,
    case_id: str,
    pair: str,
    call: int,
    physical_time: float,
    truth: np.ndarray,
    coarse_increment: np.ndarray,
    restricted_fine_increment: np.ndarray,
    volumes: np.ndarray,
    residual_scale: np.ndarray,
    reference_floor: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    defect = coarse_increment - restricted_fine_increment
    coarse_error = coarse_increment - truth
    fine_error = restricted_fine_increment - truth
    truth_rms = _metric(truth, volumes=volumes, residual_scale=residual_scale)
    defect_rms = _metric(defect, volumes=volumes, residual_scale=residual_scale)
    return (
        {
            "case_id": case_id,
            "pair": pair,
            "call": call,
            "physical_time": physical_time,
            "truth_increment_rms": truth_rms,
            "coarse_error_rms": _metric(
                coarse_error, volumes=volumes, residual_scale=residual_scale
            ),
            "restricted_fine_error_rms": _metric(
                fine_error, volumes=volumes, residual_scale=residual_scale
            ),
            "defect_rms": defect_rms,
            "defect_relative_to_truth_increment": _safe_ratio(defect_rms, truth_rms),
            "defect_truth_cosine": cosine(
                defect,
                truth,
                volumes=volumes,
                component_scale=residual_scale,
            ),
            "reference_floor_rms": _metric(
                reference_floor, volumes=volumes, residual_scale=residual_scale
            ),
        },
        defect,
    )


def _gain_row(
    *,
    base: Mapping[str, Any],
    branch: str,
    layer: int | None,
    gain: float,
    truth: np.ndarray,
    base_defect: np.ndarray,
    coarse_increment: np.ndarray,
    restricted_fine_increment: np.ndarray,
    volumes: np.ndarray,
    residual_scale: np.ndarray,
    coarse_admissibility: Mapping[str, Any],
    fine_admissibility: Mapping[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    defect = coarse_increment - restricted_fine_increment
    response = (defect - base_defect) / (gain - 1.0)
    base_norm = float(base["defect_rms"])
    norm = _metric(defect, volumes=volumes, residual_scale=residual_scale)
    base_coarse_error = float(base["coarse_error_rms"])
    base_fine_error = float(base["restricted_fine_error_rms"])
    coarse_error = _metric(
        coarse_increment - truth,
        volumes=volumes,
        residual_scale=residual_scale,
    )
    fine_error = _metric(
        restricted_fine_increment - truth,
        volumes=volumes,
        residual_scale=residual_scale,
    )
    row = {
        "case_id": base["case_id"],
        "pair": base["pair"],
        "call": base["call"],
        "physical_time": base["physical_time"],
        "branch": branch,
        "layer": "all" if layer is None else layer,
        "gain": gain,
        "base_defect_rms": base_norm,
        "gain_defect_rms": norm,
        "defect_norm_ratio": _safe_ratio(norm, base_norm),
        "defect_norm_elasticity": (
            (_safe_ratio(norm, base_norm) - 1.0) / (gain - 1.0)
            if base_norm != 0.0
            else None
        ),
        "defect_energy_elasticity": (
            ((norm / base_norm) ** 2 - 1.0) / (gain - 1.0) if base_norm != 0.0 else None
        ),
        "response_rms_per_unit_gain": _metric(
            response, volumes=volumes, residual_scale=residual_scale
        ),
        "response_defect_cosine": cosine(
            response,
            base_defect,
            volumes=volumes,
            component_scale=residual_scale,
        ),
        "coarse_error_rms": coarse_error,
        "restricted_fine_error_rms": fine_error,
        "coarse_error_norm_ratio": _safe_ratio(coarse_error, base_coarse_error),
        "restricted_fine_error_norm_ratio": _safe_ratio(fine_error, base_fine_error),
        "coarse_finite": coarse_admissibility["finite"],
        "coarse_admissible": coarse_admissibility["admissible"],
        "coarse_minimum_density": coarse_admissibility["minimum_density"],
        "coarse_minimum_pressure": coarse_admissibility["minimum_pressure"],
        "fine_finite": fine_admissibility["finite"],
        "fine_admissible": fine_admissibility["admissible"],
        "fine_minimum_density": fine_admissibility["minimum_density"],
        "fine_minimum_pressure": fine_admissibility["minimum_pressure"],
    }
    return row, response


def _quantiles(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("summary values must be non-empty and finite")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _aggregate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_keys: Sequence[str],
    metrics: Sequence[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)
    result: list[dict[str, Any]] = []
    for key, selected in sorted(grouped.items(), key=lambda item: str(item[0])):
        common = dict(zip(group_keys, key, strict=True))
        for metric in metrics:
            values = [
                float(row[metric]) for row in selected if row.get(metric) is not None
            ]
            if values:
                result.append({**common, "metric": metric, **_quantiles(values)})
    return result


def _source_hash(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def _diagnostic_source_path(repository_relative: str, fallback_name: str) -> Path:
    repository_path = ROOT / repository_relative
    if repository_path.is_file():
        return repository_path
    if diagnostic_library:
        fallback = Path(diagnostic_library) / fallback_name
        if fallback.is_file():
            return fallback
    return repository_path


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not np.isclose(args.gain, GAIN, rtol=0.0, atol=0.0):
        raise ValueError("the preregistered diagnostic gain is exactly 0.99")
    sensitivity_calls = tuple(sorted({int(value) for value in args.sensitivity_calls}))
    if sensitivity_calls != DEFAULT_SENSITIVITY_CALLS:
        raise ValueError("the preregistered sensitivity calls are 1, 5, 15, 30")
    args.output_dir.mkdir(parents=True)
    bundle_dir = args.output_dir / "bundles"
    bundle_dir.mkdir()

    source_hashes = verify_retained_source(args.retained_summary)
    retained_summary = json.loads(args.retained_summary.read_text(encoding="utf-8"))
    retained_repeatability_floor = float(
        retained_summary["execution"]["maximum_repeat_abs_difference"]
    )
    if (
        not np.isfinite(retained_repeatability_floor)
        or retained_repeatability_floor <= 0.0
    ):
        raise ValueError(
            "retained D063 repeatability floor must be finite and positive"
        )
    checkpoint_sha = sha256_file(args.checkpoint)
    if checkpoint_sha.lower() != args.expected_checkpoint_sha256.lower():
        raise ValueError("checkpoint SHA-256 mismatch")
    checkpoint = _load_checkpoint(args.checkpoint)
    if checkpoint.get("normalization_digest") != args.expected_normalization_digest:
        raise ValueError("checkpoint normalization digest mismatch")
    normalization_digest = canonical_digest(_normalization_mapping(checkpoint))
    if normalization_digest != args.expected_normalization_digest:
        raise ValueError("checkpoint normalization metadata digest does not verify")

    manifest = load_shock_vortex_family_manifest(
        args.family_root / "family_manifest.json"
    )
    store = PCNOEuler2DShardStore(args.data_dir, max_cached_trajectories=1)
    resolutions = sorted(
        [parse_resolution(value) for value in args.resolutions],
        key=lambda value: value[0] * value[1],
    )
    if tuple(resolution_label(value) for value in resolutions) != DEFAULT_RESOLUTIONS:
        raise ValueError("the pathway contract freezes 125x50, 250x100, 500x200")
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
    if rollout_calls != 30 or stride != 2 or _step_stride(checkpoint) != stride:
        raise AssertionError("the validated time contract is not D060 stride-2 H30")
    case_ids = list(dict.fromkeys(args.case_ids))
    if tuple(case_ids) != DEFAULT_CASE_IDS:
        raise ValueError("the pathway population is the six preregistered open cases")
    requested_bundle_cases = set(args.bundle_case_ids)
    if requested_bundle_cases != set(DEFAULT_BUNDLE_CASE_IDS):
        raise ValueError("the visualization population is preregistered")
    evaluated_case_ids = case_ids[:1] if args.smoke else case_ids
    evaluated_calls = 1 if args.smoke else rollout_calls
    bundle_cases = {evaluated_case_ids[0]} if args.smoke else requested_bundle_cases
    for case_id in case_ids:
        provenance = family_case_provenance(manifest, case_id)
        if provenance["split"] != "validation":
            raise ValueError(f"case is not open validation: {case_id}")
        if case_id not in store.keys:
            raise ValueError(f"case is absent from checkpoint-bound shards: {case_id}")

    device = _select_device(args.device)
    if device.type != "cuda" and args.device == "cuda":
        raise RuntimeError("CUDA was requested but not selected")
    if device.type == "cuda":
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError(
                "deterministic CUDA pathway analysis requires "
                "CUBLAS_WORKSPACE_CONFIG=:4096:8"
            )
        torch.use_deterministic_algorithms(True)
    model, normalization = _compat_model(checkpoint, device)
    model.eval()
    residual_scale = np.asarray(normalization.residual_scale, dtype=np.float64)
    state_scale = np.asarray(normalization.state_scale, dtype=np.float64)
    gamma = float(normalization.gamma)
    layer_count = len(model.backbone.ws)
    if layer_count != 4:
        raise ValueError("the frozen D060 backbone must contain four PCNO blocks")
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

    output_rows: list[dict[str, Any]] = []
    family_gain_rows: list[dict[str, Any]] = []
    layer_gain_rows: list[dict[str, Any]] = []
    latent_rows: list[dict[str, Any]] = []
    latent_spectral_rows: list[dict[str, Any]] = []
    response_spatial_rows: list[dict[str, Any]] = []
    response_spectral_rows: list[dict[str, Any]] = []
    closure_rows: list[dict[str, Any]] = []
    completion_rows: list[dict[str, Any]] = []
    bundles: dict[str, dict[str, Any]] = {}
    started = perf_counter()

    try:
        for case_index, case_id in enumerate(evaluated_case_ids, start=1):
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
                    rollout_calls=evaluated_calls,
                )
                for resolution in resolutions
            }
            native_initial = np.asarray(
                common_states[training_resolution], dtype=np.float32
            )
            if not np.array_equal(native_initial, np.asarray(store.states(case_id)[0])):
                raise ValueError(f"checkpoint-bound t0 mismatch: {case_id}")

            if case_id in bundle_cases:
                bundles[case_id] = {
                    "schema": np.asarray(SCHEMA),
                    "case_id": np.asarray(case_id),
                    "physical_times": np.arange(1, evaluated_calls + 1) * physical_dt,
                    "residual_scale": residual_scale,
                    "checkpoint_sha256": np.asarray(checkpoint_sha),
                }
                for coarse, fine in pairwise(resolutions):
                    pair_key = _pair_key(_pair_label(coarse, fine))
                    node_count = coarse[0] * coarse[1]
                    for name in (
                        "true_increment",
                        "baseline_defect",
                        "spectral_response",
                        "pointwise_response",
                        "differential_response",
                    ):
                        bundles[case_id][f"{name}__{pair_key}"] = np.empty(
                            (evaluated_calls, node_count, 4), dtype=np.float32
                        )

            all_admissible = True
            maximum_replay = 0.0
            maximum_replay_physical = 0.0
            maximum_reference_floor = 0.0
            for step in range(evaluated_calls):
                call = step + 1
                physical_time = call * physical_dt
                current_tensors: dict[Resolution, torch.Tensor] = {}
                model_inputs: dict[Resolution, torch.Tensor] = {}
                aux_by_resolution: dict[Resolution, tuple[torch.Tensor, ...]] = {}
                direct_predictions: dict[Resolution, torch.Tensor] = {}
                direct_normalized_outputs: dict[Resolution, torch.Tensor] = {}
                wrapper_reconstruction: dict[Resolution, float] = {}
                increments: dict[Resolution, np.ndarray] = {}
                family_predictions: dict[str, dict[Resolution, torch.Tensor]] = {
                    branch: {} for branch in BRANCH_NAMES
                }
                layer_predictions: dict[
                    tuple[int, str], dict[Resolution, torch.Tensor]
                ] = {}

                for resolution in resolutions:
                    current_state = _as_model_state(
                        reference_by_resolution[resolution][step]
                    )
                    current, model_input, aux = _model_tensors(
                        model,
                        sample_by_resolution[resolution],
                        current_state,
                        device=device,
                    )
                    direct, direct_normalized = _direct_prediction(
                        model, sample_by_resolution[resolution], current
                    )
                    with torch.no_grad():
                        reconstructed = (
                            current + direct_normalized * model.residual_scale
                        ) * aux[0]
                    current_tensors[resolution] = current
                    model_inputs[resolution] = model_input
                    aux_by_resolution[resolution] = aux
                    direct_predictions[resolution] = direct
                    direct_normalized_outputs[resolution] = direct_normalized
                    wrapper_reconstruction[resolution] = float(
                        (direct - reconstructed).abs().max().detach().cpu()
                    )
                    increments[resolution] = _numpy(direct - current)
                    direct_admissibility = _admissibility(direct, gamma=gamma)
                    all_admissible = all_admissible and bool(
                        direct_admissibility["admissible"]
                    )
                    for branch in BRANCH_NAMES:
                        gains = {
                            (layer, branch): args.gain for layer in range(layer_count)
                        }
                        prediction = _gain_prediction(
                            model, current, model_input, aux, gains
                        )
                        family_predictions[branch][resolution] = prediction
                        all_admissible = all_admissible and bool(
                            _admissibility(prediction, gamma=gamma)["admissible"]
                        )
                    if call in sensitivity_calls:
                        for layer in range(layer_count):
                            for branch in BRANCH_NAMES:
                                key = (layer, branch)
                                prediction = _gain_prediction(
                                    model,
                                    current,
                                    model_input,
                                    aux,
                                    {key: args.gain},
                                )
                                layer_predictions.setdefault(key, {})[
                                    resolution
                                ] = prediction
                                all_admissible = all_admissible and bool(
                                    _admissibility(prediction, gamma=gamma)[
                                        "admissible"
                                    ]
                                )

                for coarse, fine in pairwise(resolutions):
                    pair = _pair_label(coarse, fine)
                    volumes = geometry_by_resolution[coarse].node_measures
                    true_coarse = (
                        reference_by_resolution[coarse][step + 1]
                        - reference_by_resolution[coarse][step]
                    )
                    true_fine = _restrict(
                        reference_by_resolution[fine][step + 1]
                        - reference_by_resolution[fine][step],
                        fine=fine,
                        coarse=coarse,
                    )
                    reference_floor = true_coarse - true_fine
                    reference_floor_rms = _metric(
                        reference_floor,
                        volumes=volumes,
                        residual_scale=residual_scale,
                    )
                    maximum_reference_floor = max(
                        maximum_reference_floor, reference_floor_rms
                    )
                    restricted_fine_increment = _restrict(
                        increments[fine], fine=fine, coarse=coarse
                    )
                    base_row, base_defect = _output_row(
                        case_id=case_id,
                        pair=pair,
                        call=call,
                        physical_time=physical_time,
                        truth=true_coarse,
                        coarse_increment=increments[coarse],
                        restricted_fine_increment=restricted_fine_increment,
                        volumes=volumes,
                        residual_scale=residual_scale,
                        reference_floor=reference_floor,
                    )
                    output_rows.append(base_row)

                    coarse_current = _numpy(current_tensors[coarse])
                    restricted_fine_current = _restrict(
                        _numpy(current_tensors[fine]), fine=fine, coarse=coarse
                    )
                    paired_input_gap = coarse_current - restricted_fine_current
                    state_output_commutator = _numpy(
                        direct_predictions[coarse]
                    ) - _restrict(
                        _numpy(direct_predictions[fine]), fine=fine, coarse=coarse
                    )
                    increment_output_identity_difference = (
                        base_defect - state_output_commutator
                    )
                    commutator_algebra_residual = (
                        increment_output_identity_difference + paired_input_gap
                    )
                    paired_input_gap_max_abs = float(np.max(np.abs(paired_input_gap)))
                    increment_output_identity_max_abs = float(
                        np.max(np.abs(increment_output_identity_difference))
                    )
                    commutator_algebra_residual_max_abs = float(
                        np.max(np.abs(commutator_algebra_residual))
                    )

                    collect_fields = call in sensitivity_calls
                    trace_coarse, trace_fine, pair_rows, fields = trace_resolution_pair(
                        model.backbone,
                        model_inputs[coarse],
                        aux_by_resolution[coarse],
                        model_inputs[fine],
                        aux_by_resolution[fine],
                        coarse_resolution=coarse,
                        fine_resolution=fine,
                        collect_fields=collect_fields,
                        check_pointwise_float64=(
                            case_id == evaluated_case_ids[0] and call == 1
                        ),
                    )
                    replay_coarse = float(
                        (trace_coarse - direct_normalized_outputs[coarse])
                        .abs()
                        .max()
                        .detach()
                        .cpu()
                    )
                    replay_fine = float(
                        (trace_fine - direct_normalized_outputs[fine])
                        .abs()
                        .max()
                        .detach()
                        .cpu()
                    )
                    replay_coarse_physical = float(
                        (
                            (trace_coarse - direct_normalized_outputs[coarse])
                            * model.residual_scale
                            * aux_by_resolution[coarse][0]
                        )
                        .abs()
                        .max()
                        .detach()
                        .cpu()
                    )
                    replay_fine_physical = float(
                        (
                            (trace_fine - direct_normalized_outputs[fine])
                            * model.residual_scale
                            * aux_by_resolution[fine][0]
                        )
                        .abs()
                        .max()
                        .detach()
                        .cpu()
                    )
                    maximum_replay = max(maximum_replay, replay_coarse, replay_fine)
                    maximum_replay_physical = max(
                        maximum_replay_physical,
                        replay_coarse_physical,
                        replay_fine_physical,
                    )
                    closure_rows.append(
                        {
                            "case_id": case_id,
                            "pair": pair,
                            "call": call,
                            "physical_time": physical_time,
                            "replay_coarse_max_abs_normalized": replay_coarse,
                            "replay_fine_max_abs_normalized": replay_fine,
                            "replay_coarse_max_abs_physical": (replay_coarse_physical),
                            "replay_fine_max_abs_physical": replay_fine_physical,
                            "wrapper_coarse_max_abs_physical": wrapper_reconstruction[
                                coarse
                            ],
                            "wrapper_fine_max_abs_physical": wrapper_reconstruction[
                                fine
                            ],
                            "paired_input_gap_max_abs_physical": (
                                paired_input_gap_max_abs
                            ),
                            "increment_output_identity_max_abs_physical": (
                                increment_output_identity_max_abs
                            ),
                            "commutator_algebra_residual_max_abs_physical": (
                                commutator_algebra_residual_max_abs
                            ),
                            "reference_floor_rms": reference_floor_rms,
                            "retained_reference_digest": reference_check.get(
                                "artifact_sha256"
                            ),
                        }
                    )
                    for row in pair_rows:
                        latent_rows.append(
                            {
                                "case_id": case_id,
                                "pair": pair,
                                "call": call,
                                "physical_time": physical_time,
                                **row,
                            }
                        )
                    if collect_fields:
                        for name, field in fields.items():
                            for spectral_row in generic_spectral_energy_rows(
                                field,
                                resolution=coarse,
                                domain_lengths=domain_lengths,
                            ):
                                latent_spectral_rows.append(
                                    {
                                        "case_id": case_id,
                                        "pair": pair,
                                        "call": call,
                                        "physical_time": physical_time,
                                        "latent_field": name,
                                        **spectral_row,
                                    }
                                )

                    masks, _, _ = shock_vortex_regions(
                        reference_by_resolution[coarse][step],
                        geometry_by_resolution[coarse].nodes,
                        resolution=coarse,
                        gamma=gamma,
                    )
                    for branch in BRANCH_NAMES:
                        coarse_prediction = family_predictions[branch][coarse]
                        fine_prediction = family_predictions[branch][fine]
                        coarse_increment = _numpy(
                            coarse_prediction - current_tensors[coarse]
                        )
                        fine_increment = _restrict(
                            _numpy(fine_prediction - current_tensors[fine]),
                            fine=fine,
                            coarse=coarse,
                        )
                        gain_row, response = _gain_row(
                            base=base_row,
                            branch=branch,
                            layer=None,
                            gain=args.gain,
                            truth=true_coarse,
                            base_defect=base_defect,
                            coarse_increment=coarse_increment,
                            restricted_fine_increment=fine_increment,
                            volumes=volumes,
                            residual_scale=residual_scale,
                            coarse_admissibility=_admissibility(
                                coarse_prediction, gamma=gamma
                            ),
                            fine_admissibility=_admissibility(
                                fine_prediction, gamma=gamma
                            ),
                        )
                        family_gain_rows.append(gain_row)
                        for spectral_row in spectral_energy_rows(
                            response,
                            resolution=coarse,
                            domain_lengths=domain_lengths,
                            component_scale=residual_scale,
                        ):
                            response_spectral_rows.append(
                                {
                                    "case_id": case_id,
                                    "pair": pair,
                                    "call": call,
                                    "physical_time": physical_time,
                                    "branch": branch,
                                    "layer": "all",
                                    **spectral_row,
                                }
                            )
                        for spatial_row in region_energy_rows(
                            response,
                            volumes=volumes,
                            component_scale=residual_scale,
                            masks=masks,
                        ):
                            response_spatial_rows.append(
                                {
                                    "case_id": case_id,
                                    "pair": pair,
                                    "call": call,
                                    "physical_time": physical_time,
                                    "branch": branch,
                                    "layer": "all",
                                    **spatial_row,
                                }
                            )
                        if case_id in bundle_cases:
                            pair_key = _pair_key(pair)
                            bundles[case_id][f"{branch}_response__{pair_key}"][step] = (
                                np.asarray(response, dtype=np.float32)
                            )

                    if call in sensitivity_calls:
                        for (layer, branch), predictions in layer_predictions.items():
                            coarse_prediction = predictions[coarse]
                            fine_prediction = predictions[fine]
                            gain_row, _ = _gain_row(
                                base=base_row,
                                branch=branch,
                                layer=layer,
                                gain=args.gain,
                                truth=true_coarse,
                                base_defect=base_defect,
                                coarse_increment=_numpy(
                                    coarse_prediction - current_tensors[coarse]
                                ),
                                restricted_fine_increment=_restrict(
                                    _numpy(fine_prediction - current_tensors[fine]),
                                    fine=fine,
                                    coarse=coarse,
                                ),
                                volumes=volumes,
                                residual_scale=residual_scale,
                                coarse_admissibility=_admissibility(
                                    coarse_prediction, gamma=gamma
                                ),
                                fine_admissibility=_admissibility(
                                    fine_prediction, gamma=gamma
                                ),
                            )
                            layer_gain_rows.append(gain_row)

                    if case_id in bundle_cases:
                        pair_key = _pair_key(pair)
                        bundles[case_id][f"true_increment__{pair_key}"][step] = (
                            np.asarray(true_coarse, dtype=np.float32)
                        )
                        bundles[case_id][f"baseline_defect__{pair_key}"][step] = (
                            np.asarray(base_defect, dtype=np.float32)
                        )

                del (
                    current_tensors,
                    model_inputs,
                    aux_by_resolution,
                    direct_predictions,
                    direct_normalized_outputs,
                    wrapper_reconstruction,
                    increments,
                    family_predictions,
                    layer_predictions,
                )
            completion_rows.append(
                {
                    "case_id": case_id,
                    "calls": evaluated_calls,
                    "all_direct_and_gain_outputs_admissible": all_admissible,
                    "maximum_replay_max_abs_normalized": maximum_replay,
                    "maximum_replay_max_abs_physical": maximum_replay_physical,
                    "maximum_reference_floor_rms": maximum_reference_floor,
                    "seconds": perf_counter() - case_started,
                }
            )
            if case_id in bundle_cases:
                bundle_path = bundle_dir / f"{case_id}.npz"
                np.savez_compressed(bundle_path, **bundles.pop(case_id))
            if case_index % args.progress_every == 0:
                print(
                    f"completed {case_index}/{len(evaluated_case_ids)} {case_id} "
                    f"seconds={perf_counter() - case_started:.1f}",
                    flush=True,
                )
            del reference, reference_by_resolution, common_states
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        store.close()

    family_registered = [
        row for row in family_gain_rows if int(row["call"]) in sensitivity_calls
    ]
    pair_labels = [_pair_label(*pair) for pair in pairwise(resolutions)]
    mechanism = (
        {"decision": "not_run_smoke"}
        if args.smoke
        else select_dominant_pathway(
            family_registered,
            case_ids=case_ids,
            pairs=pair_labels,
            calls=sensitivity_calls,
        )
    )
    latent_closures = [
        abs(float(row["relative_closure"]))
        for row in latent_rows
        if row.get("relative_closure") is not None
    ]
    pointwise_float32_impact = [
        abs(float(row["mesh_to_combined_native_relative"]))
        for row in latent_rows
        if row.get("record_kind") == "branch"
        and row.get("branch") == "pointwise"
        and row.get("mesh_to_combined_native_relative") is not None
    ]
    pointwise_float32_branch = [
        abs(float(row["mesh_symmetric_relative"]))
        for row in latent_rows
        if row.get("record_kind") == "branch"
        and row.get("branch") == "pointwise"
        and row.get("mesh_symmetric_relative") is not None
    ]
    pointwise_float64_commutation = [
        abs(float(row["pointwise_float64_mesh_symmetric_relative"]))
        for row in latent_rows
        if row.get("record_kind") == "branch"
        and row.get("branch") == "pointwise"
        and row.get("pointwise_float64_mesh_symmetric_relative") is not None
    ]
    checks = {
        "maximum_replay_max_abs_normalized": max(
            float(row["maximum_replay_max_abs_normalized"]) for row in completion_rows
        ),
        "maximum_replay_max_abs_physical": max(
            float(row["maximum_replay_max_abs_physical"]) for row in completion_rows
        ),
        "retained_d063_repeatability_floor_max_abs_physical": (
            retained_repeatability_floor
        ),
        "maximum_latent_relative_closure": max(latent_closures, default=0.0),
        "maximum_pointwise_float32_mesh_to_combined_native_relative": max(
            pointwise_float32_impact, default=0.0
        ),
        "maximum_pointwise_float32_mesh_symmetric_relative": max(
            pointwise_float32_branch, default=0.0
        ),
        "maximum_pointwise_float64_mesh_symmetric_relative": max(
            pointwise_float64_commutation, default=0.0
        ),
        "pointwise_float64_identity_check_count": len(pointwise_float64_commutation),
        "pointwise_float64_identity_expected_count": layer_count * len(pair_labels),
        "maximum_reference_floor_rms": max(
            float(row["maximum_reference_floor_rms"]) for row in completion_rows
        ),
        "maximum_wrapper_reconstruction_max_abs_physical": max(
            max(
                float(row["wrapper_coarse_max_abs_physical"]),
                float(row["wrapper_fine_max_abs_physical"]),
            )
            for row in closure_rows
        ),
        "maximum_paired_input_gap_max_abs_physical": max(
            float(row["paired_input_gap_max_abs_physical"]) for row in closure_rows
        ),
        "maximum_increment_output_identity_max_abs_physical": max(
            float(row["increment_output_identity_max_abs_physical"])
            for row in closure_rows
        ),
        "maximum_commutator_algebra_residual_max_abs_physical": max(
            float(row["commutator_algebra_residual_max_abs_physical"])
            for row in closure_rows
        ),
        "all_outputs_admissible": all(
            bool(row["all_direct_and_gain_outputs_admissible"])
            for row in completion_rows
        ),
    }
    checks["replay_pass"] = (
        checks["maximum_replay_max_abs_normalized"] <= REPLAY_NORMALIZED_TOLERANCE
        and checks["maximum_replay_max_abs_physical"] <= REPLAY_PHYSICAL_TOLERANCE
    )
    checks["wrapper_reconstruction_pass"] = (
        checks["maximum_wrapper_reconstruction_max_abs_physical"] <= WRAPPER_TOLERANCE
    )
    checks["paired_input_restriction_pass"] = (
        checks["maximum_paired_input_gap_max_abs_physical"]
        <= PAIRED_INPUT_RESTRICTION_TOLERANCE
    )
    checks["commutator_identity_pass"] = (
        checks["maximum_commutator_algebra_residual_max_abs_physical"]
        <= COMMUTATOR_IDENTITY_TOLERANCE
        and checks["maximum_increment_output_identity_max_abs_physical"]
        <= checks["maximum_paired_input_gap_max_abs_physical"]
        + COMMUTATOR_IDENTITY_TOLERANCE
    )
    checks["algebra_pass"] = (
        checks["maximum_latent_relative_closure"] <= ALGEBRA_TOLERANCE
    )
    checks["pointwise_commutation_pass"] = (
        checks["pointwise_float64_identity_check_count"]
        == checks["pointwise_float64_identity_expected_count"]
        and checks["maximum_pointwise_float64_mesh_symmetric_relative"]
        <= POINTWISE_FLOAT64_COMMUTATION_TOLERANCE
        and checks["maximum_pointwise_float32_mesh_symmetric_relative"]
        <= POINTWISE_FLOAT32_BRANCH_TOLERANCE
        and checks["maximum_pointwise_float32_mesh_to_combined_native_relative"]
        <= POINTWISE_FLOAT32_IMPACT_TOLERANCE
    )
    checks["reference_floor_pass"] = (
        checks["maximum_reference_floor_rms"] <= REFERENCE_FLOOR_TOLERANCE
    )
    interpretation_allowed = all(
        bool(checks[name])
        for name in (
            "replay_pass",
            "wrapper_reconstruction_pass",
            "paired_input_restriction_pass",
            "commutator_identity_pass",
            "algebra_pass",
            "pointwise_commutation_pass",
            "reference_floor_pass",
            "all_outputs_admissible",
        )
    )
    scientific_interpretation_allowed = interpretation_allowed and not args.smoke
    if args.smoke:
        status = "smoke_complete" if interpretation_allowed else "smoke_failed"
    else:
        status = "complete" if interpretation_allowed else "failed_contract"

    aggregate_rows = []
    aggregate_rows.extend(
        _aggregate_rows(
            output_rows,
            group_keys=("pair", "call"),
            metrics=(
                "defect_relative_to_truth_increment",
                "defect_rms",
                "coarse_error_rms",
                "restricted_fine_error_rms",
            ),
        )
    )
    aggregate_rows.extend(
        _aggregate_rows(
            family_gain_rows,
            group_keys=("pair", "call", "branch"),
            metrics=(
                "defect_norm_elasticity",
                "defect_norm_ratio",
                "response_rms_per_unit_gain",
                "coarse_error_norm_ratio",
                "restricted_fine_error_norm_ratio",
            ),
        )
    )
    aggregate_rows.extend(
        _aggregate_rows(
            layer_gain_rows,
            group_keys=("pair", "call", "layer", "branch"),
            metrics=(
                "defect_norm_elasticity",
                "defect_norm_ratio",
                "coarse_error_norm_ratio",
                "restricted_fine_error_norm_ratio",
            ),
        )
    )
    for aggregate_kind, selected, group_keys, metrics in (
        (
            "latent_hidden",
            [row for row in latent_rows if row.get("record_kind") == "hidden"],
            ("pair", "call", "layer", "stage"),
            ("symmetric_relative_gap",),
        ),
        (
            "latent_branch",
            [row for row in latent_rows if row.get("record_kind") == "branch"],
            ("pair", "call", "layer", "branch"),
            (
                "native_symmetric_relative",
                "mesh_symmetric_relative",
                "state_symmetric_relative",
                "combined_branch_attribution",
                "mesh_symmetric_attribution",
            ),
        ),
        (
            "input_group",
            [row for row in latent_rows if row.get("record_kind") == "input_group"],
            ("pair", "call", "input_group"),
            ("gap_rms", "symmetric_attribution"),
        ),
    ):
        aggregated = _aggregate_rows(
            selected,
            group_keys=group_keys,
            metrics=metrics,
        )
        for row in aggregated:
            row["aggregate_kind"] = aggregate_kind
        aggregate_rows.extend(aggregated)

    files_and_rows = {
        "output_metrics.csv": output_rows,
        "family_gain_metrics.csv": family_gain_rows,
        "layer_gain_metrics.csv": layer_gain_rows,
        "latent_metrics.csv": latent_rows,
        "latent_spectral_metrics.csv": latent_spectral_rows,
        "response_spatial_metrics.csv": response_spatial_rows,
        "response_spectral_metrics.csv": response_spectral_rows,
        "closure_metrics.csv": closure_rows,
        "completion.csv": completion_rows,
        "aggregate_metrics.csv": aggregate_rows,
    }
    for filename, rows in files_and_rows.items():
        write_csv(args.output_dir / filename, rows)
    visual_contract = {
        "field_units": "physical conservative increment divided by residual_scale",
        "common_symmetric_limit": 1.0,
        "response_definition": "(delta_gain_0.99 - delta_gain_1.00) / -0.01",
        "columns": [
            "true_increment",
            "baseline_cross_grid_defect",
            "spectral_shared_gain_response",
            "pointwise_shared_gain_response",
            "differential_shared_gain_response",
        ],
        "scale_source": "checkpoint residual_scale; independent of pathway outcome",
    }
    write_json(args.output_dir / "visual_contract.json", visual_contract)
    output_hashes = {
        path.name: sha256_file(path) for path in sorted(args.output_dir.glob("*.csv"))
    }
    output_hashes["visual_contract.json"] = sha256_file(
        args.output_dir / "visual_contract.json"
    )
    output_hashes.update(
        {
            str(path.relative_to(args.output_dir)): sha256_file(path)
            for path in sorted(bundle_dir.glob("*.npz"))
        }
    )
    summary = {
        "schema": SCHEMA,
        "status": status,
        "contract_checks_passed": interpretation_allowed,
        "scientific_interpretation_allowed": scientific_interpretation_allowed,
        "boundary_policy": EXPECTED_BOUNDARY_POLICY,
        "population": {
            "split": "validation",
            "case_ids": evaluated_case_ids,
            "registered_case_ids": case_ids,
            "sealed_populations_accessed": False,
        },
        "checkpoint_sha256": checkpoint_sha,
        "normalization_digest": normalization_digest,
        "state_scale": state_scale,
        "residual_scale": residual_scale,
        "source_hashes": source_hashes,
        "artifact_manifest_hashes": {
            "family_manifest": sha256_file(args.family_root / "family_manifest.json"),
            "shard_manifest": store.manifest_digest,
            "retained_d063_summary": sha256_file(args.retained_summary),
        },
        "diagnostic_source_hashes": {
            "runner": sha256_file(Path(__file__).resolve()),
            "pathway_utility": _source_hash(
                _diagnostic_source_path(
                    "utility/time_dependent_no/pcno_resolution_pathways.py",
                    "pcno_resolution_pathways.py",
                )
            ),
            "residual_utility": _source_hash(
                _diagnostic_source_path(
                    "utility/time_dependent_no/pcno_residual_structure.py",
                    "pcno_residual_structure.py",
                )
            ),
            "residual_runner_helper": _source_hash(
                _diagnostic_source_path(
                    "scripts/time_dependent_no/analyze_pcno_residual_structure.py",
                    "analyze_pcno_residual_structure.py",
                )
            ),
        },
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "cuda_device": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "amp": args.amp,
            "deterministic_algorithms": (torch.are_deterministic_algorithms_enabled()),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "contract": {
            "teacher_forced_exact_inputs": True,
            "resolutions": [resolution_label(value) for value in resolutions],
            "source_resolution": resolution_label(source_resolution),
            "training_resolution": resolution_label(training_resolution),
            "stride": stride,
            "physical_dt": physical_dt,
            "rollout_calls": rollout_calls,
            "evaluated_calls": evaluated_calls,
            "smoke": args.smoke,
            "gain": args.gain,
            "family_gain_calls": list(range(1, rollout_calls + 1)),
            "layer_gain_calls": sensitivity_calls,
            "latent_denominator": "symmetric within-layer RMS",
            "pointwise_float64_identity_scope": (
                "first registered case/call for every layer and adjacent pair; "
                "native float32 impact is measured on every call"
            ),
            "physical_output_denominator": "coarse truth increment RMS in residual_scale units",
            "aggregation": "per-case before across-case summary; no node pooling across meshes",
        },
        "tolerances": {
            "replay_max_abs_normalized": REPLAY_NORMALIZED_TOLERANCE,
            "replay_max_abs_physical": REPLAY_PHYSICAL_TOLERANCE,
            "wrapper_reconstruction": WRAPPER_TOLERANCE,
            "algebra": ALGEBRA_TOLERANCE,
            "pointwise_float64_commutation": (POINTWISE_FLOAT64_COMMUTATION_TOLERANCE),
            "pointwise_float32_impact": POINTWISE_FLOAT32_IMPACT_TOLERANCE,
            "pointwise_float32_branch": POINTWISE_FLOAT32_BRANCH_TOLERANCE,
            "commutator_identity_max_abs_physical": (COMMUTATOR_IDENTITY_TOLERANCE),
            "paired_input_restriction_max_abs_physical": (
                PAIRED_INPUT_RESTRICTION_TOLERANCE
            ),
            "reference_floor": REFERENCE_FLOOR_TOLERANCE,
        },
        "checks": checks,
        "mechanism_selection": mechanism if scientific_interpretation_allowed else None,
        "row_counts": {
            filename: len(rows) for filename, rows in files_and_rows.items()
        },
        "elapsed_seconds": perf_counter() - started,
        "output_hashes": output_hashes,
    }
    write_json(args.output_dir / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    summary = run(parse_args(argv))
    print(
        f"wrote pathway diagnostic: status={summary['status']} "
        f"seconds={summary['elapsed_seconds']:.1f}",
        flush=True,
    )
    return 0 if summary["status"] in {"complete", "smoke_complete"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
