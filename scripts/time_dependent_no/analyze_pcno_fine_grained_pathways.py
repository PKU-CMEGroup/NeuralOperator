#!/usr/bin/env python3
"""D070: causally isolate fine-grained PCNO resolution pathways."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
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
    git_state,
    jsonable_args,
    runtime_environment,
    sha256_file,
    write_csv_with_paths,
)
from utility.time_dependent_no.pcno_euler2d import PCNOEuler2DShardStore
from utility.time_dependent_no.pcno_resolution_pathways import (
    BRANCH_NAMES,
    decompose_differential_same_hidden_mesh_gap,
    decompose_spectral_same_hidden_mesh_gap,
    trace_backbone_output,
    trace_native_branch_replacement_output,
    trace_native_hidden_to_layer,
    trace_same_hidden_subpath_replacements,
)
from utility.time_dependent_no.pcno_resolution_transfer import (
    build_resolution_checkpoint_model,
    conservative_admissibility_summary,
    load_resolution_checkpoint,
    parse_resolution,
    resolution_label,
    restrict_nested_state,
)
from utility.time_dependent_no.pcno_runtime import select_device
from utility.time_dependent_no.pcno_scale_separated_drift import project_dct_bands

SCHEMA = "pcno_fine_grained_pathway_diagnostic_v2"
CASE_IDS = (
    "sv_e00_y00",
    "sv_e00_y08",
    "sv_e06_y00",
    "sv_e06_y08",
    "sv_e11_y00",
    "sv_e11_y08",
)
RESOLUTIONS = ((125, 50), (250, 100), (500, 200))
CALLS = (1, 5, 15, 30)
REPLAY_TOLERANCE = 2.0e-6
CLOSURE_TOLERANCE = 2.0e-5
D070C_ABSOLUTE_TOLERANCE = 2.0e-5
D070C_RELATIVE_TOLERANCE = 2.0e-5
D070C_TEACHER_INPUT_ABSOLUTE_TOLERANCE = 2.0e-6
D070C_DCT_TOLERANCE = 1.0e-10
D070C_TRACE_LARGE_RATIO_TOLERANCE = 0.01
D070C_SELECTOR_DENOMINATOR_MINIMUM = 1.0e-8
HISTORICAL_STATE_ABSOLUTE_LIMIT = 2.0e-5
HISTORICAL_STATE_RELATIVE_L2_LIMIT = 1.0e-7
HISTORICAL_LARGE_BAND_DRIFT_RATIO_LIMIT = 0.01
LARGE_REDUCTION_GATE = 0.20
POSITIVE_CASE_GATE = 4
OTHER_BAND_RATIO_GATE = 1.10
D067_INHERITED_SOURCE_CONTRACT = "exact_registered_d067_inherited_sources"
HISTORICAL_SOURCE_CONTRACT = "historical_output_compatibility_current_core"
CURRENT_SELF_CONSISTENT_SOURCE_CONTRACT = (
    "current_core_self_consistent_pathway_contract"
)
D067_SOURCE_CONTRACTS = (
    D067_INHERITED_SOURCE_CONTRACT,
    HISTORICAL_SOURCE_CONTRACT,
    CURRENT_SELF_CONSISTENT_SOURCE_CONTRACT,
)
SAME_HIDDEN_ARMS = (
    "fourier_quadrature",
    "fourier_subcell",
    "fourier_analysis",
    "fourier_synthesis",
    "fourier_full",
    "differential_gradient",
    "differential_fixed_hop",
    "differential_full",
    "pointwise_full",
)
PATHWAY_TERMS = {
    "fourier": (
        "quadrature_response",
        "subcell_hidden_response",
        "synthesis_restriction_response",
    ),
    "differential_pre": ("gradient_pre_response", "fixed_hop_pre_response"),
    "differential_output": (
        "gradient_output_response",
        "fixed_hop_composite_output_response",
    ),
}
D067_INHERITED_SOURCE_PATHS = (
    "pcno/pcno.py",
    "scripts/time_dependent_no/evaluate_pcno_resolution_rollout.py",
    "scripts/time_dependent_no/generate_pcno_shock_vortex_multires_reference.py",
    "utility/time_dependent_no/pcno_euler2d.py",
    "utility/time_dependent_no/pcno_fv_geometry.py",
    "utility/time_dependent_no/pcno_resolution_transfer.py",
    "utility/time_dependent_no/shock_vortex_family.py",
    "utility/time_dependent_no/shock_vortex_metrics.py",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(family="dynamic_fv")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization-json", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--family-root", type=Path, required=True)
    parser.add_argument("--multires-reference-root", type=Path, required=True)
    parser.add_argument("--d067-summary", type=Path, required=True)
    parser.add_argument(
        "--d067-source-contract",
        choices=D067_SOURCE_CONTRACTS,
        required=True,
    )
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
        raise ValueError("D070 freezes the six D067 evaluation cases")
    if tuple(parse_resolution(value) for value in args.resolutions) != RESOLUTIONS:
        raise ValueError("D070 freezes 125x50, 250x100, and 500x200")
    if parse_resolution(args.training_resolution) != (250, 100):
        raise ValueError("D070 freezes the training resolution at 250x100")
    if args.rollout_calls != 30 or tuple(sorted(set(args.diagnostic_calls))) != CALLS:
        raise ValueError("D070 freezes H30 and diagnostic calls 1,5,15,30")
    for name in ("expected_source_base_git_head", "expected_source_manifest_sha256"):
        digest = str(getattr(args, name)).lower()
        size = 40 if name == "expected_source_base_git_head" else 64
        if len(digest) != size or any(
            value not in "0123456789abcdef" for value in digest
        ):
            raise ValueError(f"--{name.replace('_', '-')} must be a {size}-digit SHA")
    return args


def _pair_label(coarse: tuple[int, int], fine: tuple[int, int]) -> str:
    return f"{resolution_label(coarse)}->{resolution_label(fine)}"


def _pair_key(coarse: tuple[int, int], fine: tuple[int, int]) -> str:
    return f"{resolution_label(coarse)}_to_{resolution_label(fine)}"


def _aux(case: CaseData) -> tuple[torch.Tensor, ...]:
    return (
        case.sample["node_mask"],
        case.sample["nodes"],
        case.sample["node_weights"],
        case.sample["directed_edges"],
        case.sample["edge_gradient_weights"],
    )


def _restrict(
    value: np.ndarray, *, fine: tuple[int, int], coarse: tuple[int, int]
) -> np.ndarray:
    return np.asarray(
        restrict_nested_state(value, fine_resolution=fine, coarse_resolution=coarse),
        dtype=np.float64,
    )


@torch.inference_mode()
def _model_context(
    model: torch.nn.Module, case: CaseData, state: np.ndarray
) -> dict[str, Any]:
    current = torch.as_tensor(
        np.asarray(state, dtype=np.float32),
        dtype=torch.float32,
        device=case.sample["nodes"].device,
    ).unsqueeze(0)
    model_input = model.normalized_input(
        current,
        nodes=case.sample["nodes"],
        node_rhos=case.sample["node_rhos"],
        node_type=case.sample["node_type"],
        mach=case.sample["mach"],
    )
    aux = _aux(case)
    normalized = model.backbone(model_input, aux)
    prediction = (current + normalized * model.residual_scale) * aux[0]
    predicted = prediction[0].detach().float().cpu().numpy().astype(np.float64)
    return {
        "state": np.asarray(state, dtype=np.float64),
        "model_input": model_input,
        "aux": aux,
        "normalized": normalized,
        "prediction": prediction,
        "increment": predicted - np.asarray(state, dtype=np.float64),
    }


def _scaled_rms(field: np.ndarray, scale: np.ndarray) -> float:
    values = np.asarray(field, dtype=np.float64) / scale[None, :]
    return float(np.sqrt(np.mean(np.sum(np.square(values), axis=-1))))


def _relative_l2(value: np.ndarray, reference: np.ndarray) -> float:
    denominator = float(np.linalg.norm(np.asarray(reference, dtype=np.float64)))
    if denominator == 0.0:
        return float("inf")
    return float(
        np.linalg.norm(
            np.asarray(value, dtype=np.float64)
            - np.asarray(reference, dtype=np.float64)
        )
        / denominator
    )


def _model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _comparison_metrics(
    value: np.ndarray,
    reference: np.ndarray,
    *,
    residual_scale: np.ndarray,
) -> dict[str, float]:
    error = np.asarray(value, dtype=np.float64) - np.asarray(
        reference, dtype=np.float64
    )
    numerator = _scaled_rms(error, residual_scale)
    denominator = _scaled_rms(reference, residual_scale)
    return {
        "max_abs_physical": float(np.max(np.abs(error))),
        "error_residual_scaled_rms": numerator,
        "reference_residual_scaled_rms": denominator,
        "relative_residual_scaled_rms": numerator
        / max(denominator, np.finfo(np.float64).tiny),
    }


def _identity_metrics(
    closure: np.ndarray, *, residual_scale: np.ndarray
) -> dict[str, float]:
    value = np.asarray(closure, dtype=np.float64)
    return {
        "closure_max_abs_physical": float(np.max(np.abs(value))),
        "closure_residual_scaled_rms": _scaled_rms(value, residual_scale),
    }


def _sorted_keys(values: Sequence[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    return sorted(values, key=lambda value: tuple(str(item) for item in value))


def _inventory_checks(
    rows: Sequence[Mapping[str, Any]],
    *,
    key_fields: Sequence[str],
    expected_keys: set[tuple[Any, ...]],
    finite_fields: Sequence[str],
) -> dict[str, Any]:
    observed_keys = [tuple(row.get(name) for name in key_fields) for row in rows]
    counts = Counter(observed_keys)
    duplicates = _sorted_keys([key for key, count in counts.items() if count > 1])
    observed_set = set(observed_keys)
    missing = _sorted_keys(list(expected_keys - observed_set))
    extra = _sorted_keys(list(observed_set - expected_keys))

    def finite(row: Mapping[str, Any], name: str) -> bool:
        try:
            return name in row and np.isfinite(float(row[name]))
        except (TypeError, ValueError):
            return False

    all_finite = all(all(finite(row, name) for name in finite_fields) for row in rows)
    matrix_complete = not duplicates and not missing and not extra
    return {
        "key_fields": list(key_fields),
        "finite_fields": list(finite_fields),
        "expected_rows": len(expected_keys),
        "observed_rows": len(rows),
        "matrix_complete": matrix_complete,
        "duplicate_keys": [list(key) for key in duplicates],
        "missing_keys": [list(key) for key in missing],
        "extra_keys": [list(key) for key in extra],
        "all_metrics_finite": all_finite,
        "complete_and_finite": matrix_complete and all_finite,
    }


def _typed_closure_row(
    *,
    case_id: str,
    mode: str,
    pair: str,
    call: int,
    layer: int | str,
    closure_kind: str,
    numerator: float,
    denominator: float,
    units: str,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    relative = float(numerator) / max(float(denominator), np.finfo(np.float64).tiny)
    passed = bool(
        np.isfinite(numerator)
        and np.isfinite(denominator)
        and np.isfinite(relative)
        and float(numerator) <= absolute_tolerance
        and relative <= relative_tolerance
    )
    return {
        "case_id": case_id,
        "mode": mode,
        "pair": pair,
        "call": call,
        "layer": layer,
        "closure_kind": closure_kind,
        "closure_numerator": float(numerator),
        "closure_denominator": float(denominator),
        "units": units,
        "absolute_closure": float(numerator),
        "relative_closure": relative,
        "absolute_tolerance": float(absolute_tolerance),
        "relative_tolerance": float(relative_tolerance),
        "passed": passed,
    }


def _historical_compatibility_checks(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_keys: set[tuple[str, str, int]],
) -> dict[str, Any]:
    required = (
        "coarse_state_max_abs_physical",
        "restricted_fine_state_max_abs_physical",
        "coarse_state_relative_l2",
        "restricted_fine_state_relative_l2",
        "coarse_increment_drift_residual_scaled_rms",
        "restricted_fine_increment_drift_residual_scaled_rms",
        "commutator_drift_residual_scaled_rms",
        "baseline_large_mesh_defect_residual_scaled_rms",
        "large_commutator_drift_residual_scaled_rms",
        "large_commutator_drift_to_baseline_ratio",
        "band_reconstruction_closure",
    )
    observed_keys = [
        (str(row["case_id"]), str(row["pair"]), int(row["call"])) for row in rows
    ]
    counts = Counter(observed_keys)
    duplicate_keys = sorted(key for key, count in counts.items() if count > 1)
    missing_keys = sorted(expected_keys - set(observed_keys))
    extra_keys = sorted(set(observed_keys) - expected_keys)
    key_matrix_complete = not duplicate_keys and not missing_keys and not extra_keys
    all_metrics_finite = all(
        all(name in row and np.isfinite(float(row[name])) for name in required)
        for row in rows
    )
    complete = key_matrix_complete and all_metrics_finite
    maximum_absolute = (
        max(
            max(
                float(row["coarse_state_max_abs_physical"]),
                float(row["restricted_fine_state_max_abs_physical"]),
            )
            for row in rows
        )
        if rows
        else float("inf")
    )
    maximum_relative = (
        max(
            max(
                float(row["coarse_state_relative_l2"]),
                float(row["restricted_fine_state_relative_l2"]),
            )
            for row in rows
        )
        if rows
        else float("inf")
    )
    maximum_science_ratio = (
        max(float(row["large_commutator_drift_to_baseline_ratio"]) for row in rows)
        if rows
        else float("inf")
    )
    maximum_band_closure = (
        max(float(row["band_reconstruction_closure"]) for row in rows)
        if rows
        else float("inf")
    )
    return {
        "expected_rows": len(expected_keys),
        "observed_rows": len(rows),
        "key_matrix_complete": key_matrix_complete,
        "duplicate_keys": [list(key) for key in duplicate_keys],
        "missing_keys": [list(key) for key in missing_keys],
        "extra_keys": [list(key) for key in extra_keys],
        "all_metrics_finite": all_metrics_finite,
        "complete_and_finite": complete,
        "maximum_state_max_abs_physical": maximum_absolute,
        "state_absolute_pass": (
            complete and maximum_absolute <= HISTORICAL_STATE_ABSOLUTE_LIMIT
        ),
        "maximum_state_relative_l2": maximum_relative,
        "state_relative_pass": (
            complete and maximum_relative <= HISTORICAL_STATE_RELATIVE_L2_LIMIT
        ),
        "maximum_large_commutator_drift_to_baseline_ratio": (maximum_science_ratio),
        "science_scale_pass": (
            complete
            and maximum_science_ratio <= HISTORICAL_LARGE_BAND_DRIFT_RATIO_LIMIT
        ),
        "maximum_band_reconstruction_closure": maximum_band_closure,
        "band_reconstruction_pass": (
            complete and maximum_band_closure <= CLOSURE_TOLERANCE
        ),
    }


def _band_fields(
    field: np.ndarray,
    *,
    resolution: tuple[int, int],
    residual_scale: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    projections, closure = project_dct_bands(
        np.asarray(field, dtype=np.float64)[None],
        resolution=resolution,
        component_scale=residual_scale,
        domain_lengths=(2.0, 1.0),
    )
    fields = {"total": np.asarray(field, dtype=np.float64)}
    fields.update({name: value[0] for name, value in projections.items()})
    return fields, {name: float(value) for name, value in closure.items()}


def _append_trace_row(
    rows: list[dict[str, Any]],
    *,
    case_id: str,
    mode: str,
    pair: str,
    call: int,
    comparison: str,
    layer: int | str,
    branch: str,
    value: np.ndarray,
    reference: np.ndarray,
    baseline_large_rms: float,
    resolution: tuple[int, int],
    residual_scale: np.ndarray,
) -> None:
    metrics = _comparison_metrics(value, reference, residual_scale=residual_scale)
    error = np.asarray(value, dtype=np.float64) - np.asarray(
        reference, dtype=np.float64
    )
    error_bands, _ = _band_fields(
        error,
        resolution=resolution,
        residual_scale=residual_scale,
    )
    large_error = _scaled_rms(error_bands["large"], residual_scale)
    large_ratio = large_error / max(baseline_large_rms, np.finfo(np.float64).tiny)
    rows.append(
        {
            "case_id": case_id,
            "mode": mode,
            "pair": pair,
            "call": call,
            "comparison": comparison,
            "layer": layer,
            "branch": branch,
            **metrics,
            "large_band_error_residual_scaled_rms": large_error,
            "baseline_large_band_residual_scaled_rms": baseline_large_rms,
            "large_band_error_to_baseline_ratio": large_ratio,
            "passed": bool(
                metrics["max_abs_physical"] <= D070C_ABSOLUTE_TOLERANCE
                and metrics["relative_residual_scaled_rms"] <= D070C_RELATIVE_TOLERANCE
                and large_ratio <= D070C_TRACE_LARGE_RATIO_TOLERANCE
            ),
        }
    )


def _append_arm_rows(
    rows: list[dict[str, Any]],
    *,
    case_id: str,
    mode: str,
    pair: str,
    call: int,
    layer: int | str,
    arm: str,
    record_kind: str,
    baseline_defect: np.ndarray,
    arm_defect: np.ndarray,
    resolution: tuple[int, int],
    residual_scale: np.ndarray,
) -> dict[str, float]:
    baseline_bands, baseline_closure = _band_fields(
        baseline_defect, resolution=resolution, residual_scale=residual_scale
    )
    arm_bands, arm_closure = _band_fields(
        arm_defect, resolution=resolution, residual_scale=residual_scale
    )
    for band in ("total", "large", "transition", "local"):
        baseline_norm = _scaled_rms(baseline_bands[band], residual_scale)
        arm_norm = _scaled_rms(arm_bands[band], residual_scale)
        ratio = None if baseline_norm == 0.0 else arm_norm / baseline_norm
        rows.append(
            {
                "record_kind": record_kind,
                "case_id": case_id,
                "mode": mode,
                "pair": pair,
                "call": call,
                "layer": layer,
                "arm": arm,
                "band": band,
                "baseline_defect_rms": baseline_norm,
                "arm_defect_rms": arm_norm,
                "defect_norm_ratio": ratio,
                "defect_reduction": None if ratio is None else 1.0 - ratio,
                "response_rms": _scaled_rms(
                    arm_bands[band] - baseline_bands[band], residual_scale
                ),
            }
        )
    return {
        name: max(float(baseline_closure[name]), float(arm_closure[name]))
        for name in baseline_closure
    }


def _latent_rms(value: torch.Tensor) -> float:
    return float(
        torch.sqrt(torch.mean(torch.sum(value.square(), dim=1))).detach().cpu()
    )


def _relative_closure(closure: torch.Tensor, reference: torch.Tensor) -> float:
    return _latent_rms(closure) / max(_latent_rms(reference), np.finfo(float).tiny)


def _analyze_pair(
    model: torch.nn.Module,
    *,
    case_id: str,
    mode: str,
    call: int,
    coarse_case: CaseData,
    fine_case: CaseData,
    coarse_context: Mapping[str, Any],
    fine_context: Mapping[str, Any],
    arm_rows: list[dict[str, Any]],
    pathway_rows: list[dict[str, Any]],
    closure_rows: list[dict[str, Any]],
    direct_output_rows: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
    residual_scale: np.ndarray,
) -> None:
    coarse = coarse_case.resolution
    fine = fine_case.resolution
    if coarse is None or fine is None:
        raise AssertionError("D070 requires structured resolutions")
    pair = _pair_label(coarse, fine)
    baseline_coarse = (
        coarse_context["normalized"][0]
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float64)
        * residual_scale[None, :]
    )
    direct_fine = (
        fine_context["normalized"][0].detach().float().cpu().numpy().astype(np.float64)
        * residual_scale[None, :]
    )
    baseline_fine = _restrict(direct_fine, fine=fine, coarse=coarse)
    baseline_native_defect = baseline_coarse - baseline_fine
    baseline_bands, baseline_dct_closure = _band_fields(
        baseline_native_defect,
        resolution=coarse,
        residual_scale=residual_scale,
    )
    baseline_large_rms = _scaled_rms(baseline_bands["large"], residual_scale)
    for output_role, physical in (
        ("coarse", baseline_coarse),
        ("restricted_fine", baseline_fine),
    ):
        output_bands, _ = _band_fields(
            physical,
            resolution=coarse,
            residual_scale=residual_scale,
        )
        direct_output_rows.append(
            {
                "case_id": case_id,
                "mode": mode,
                "pair": pair,
                "call": call,
                "output_role": output_role,
                "output_residual_scaled_rms": _scaled_rms(physical, residual_scale),
                "large_band_residual_scaled_rms": _scaled_rms(
                    output_bands["large"], residual_scale
                ),
            }
        )

    traced_coarse = (
        trace_backbone_output(
            model.backbone,
            coarse_context["model_input"],
            coarse_context["aux"],
        )[0]
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float64)
        * residual_scale[None, :]
    )
    traced_fine = (
        trace_backbone_output(
            model.backbone,
            fine_context["model_input"],
            fine_context["aux"],
        )[0]
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float64)
        * residual_scale[None, :]
    )
    _append_trace_row(
        trace_rows,
        case_id=case_id,
        mode=mode,
        pair=pair,
        call=call,
        comparison="backbone_direct_coarse",
        layer="all",
        branch="none",
        value=traced_coarse,
        reference=baseline_coarse,
        baseline_large_rms=baseline_large_rms,
        resolution=coarse,
        residual_scale=residual_scale,
    )
    _append_trace_row(
        trace_rows,
        case_id=case_id,
        mode=mode,
        pair=pair,
        call=call,
        comparison="backbone_direct_restricted_fine",
        layer="all",
        branch="none",
        value=_restrict(traced_fine, fine=fine, coarse=coarse),
        reference=baseline_fine,
        baseline_large_rms=baseline_large_rms,
        resolution=coarse,
        residual_scale=residual_scale,
    )
    maximum_dct_closure = dict(baseline_dct_closure)
    for layer in range(len(model.backbone.ws)):
        outputs = trace_same_hidden_subpath_replacements(
            model.backbone,
            fine_context["model_input"],
            coarse_context["aux"],
            fine_context["aux"],
            layer=layer,
            coarse_resolution=coarse,
            fine_resolution=fine,
        )
        physical_outputs = {
            name: value[0].detach().float().cpu().numpy().astype(np.float64)
            * residual_scale[None, :]
            for name, value in outputs.items()
        }
        restricted_fine = physical_outputs["restricted_fine"]
        baseline_defect = physical_outputs["baseline"] - restricted_fine
        _append_trace_row(
            trace_rows,
            case_id=case_id,
            mode=mode,
            pair=pair,
            call=call,
            comparison="same_hidden_restricted_fine",
            layer=layer,
            branch="none",
            value=restricted_fine,
            reference=baseline_fine,
            baseline_large_rms=baseline_large_rms,
            resolution=coarse,
            residual_scale=residual_scale,
        )
        for arm, physical in physical_outputs.items():
            if arm in {"baseline", "restricted_fine"}:
                continue
            arm_dct_closure = _append_arm_rows(
                arm_rows,
                case_id=case_id,
                mode=mode,
                pair=pair,
                call=call,
                layer=layer,
                arm=arm,
                record_kind="same_hidden_single_layer",
                baseline_defect=baseline_defect,
                arm_defect=physical - restricted_fine,
                resolution=coarse,
                residual_scale=residual_scale,
            )
            for name, value in arm_dct_closure.items():
                maximum_dct_closure[name] = max(maximum_dct_closure[name], float(value))

        fine_hidden = trace_native_hidden_to_layer(
            model.backbone,
            fine_context["model_input"],
            fine_context["aux"],
            layer=layer,
        )
        spectral = decompose_spectral_same_hidden_mesh_gap(
            model.backbone.sp_convs[layer],
            fine_hidden,
            coarse_context["aux"],
            fine_context["aux"],
            coarse_resolution=coarse,
            fine_resolution=fine,
        )
        differential = decompose_differential_same_hidden_mesh_gap(
            model.backbone.gws[layer],
            fine_hidden,
            coarse_context["aux"],
            fine_context["aux"],
            coarse_resolution=coarse,
            fine_resolution=fine,
        )
        pathway_specs = (
            (
                "fourier",
                spectral,
                PATHWAY_TERMS["fourier"],
                "mesh_gap",
                "closure",
            ),
            (
                "differential_pre",
                differential,
                PATHWAY_TERMS["differential_pre"],
                "pre_mesh_gap",
                "pre_closure",
            ),
            (
                "differential_output",
                differential,
                PATHWAY_TERMS["differential_output"],
                "mesh_gap",
                "output_closure",
            ),
        )
        for pathway, values, terms, reference_name, closure_name in pathway_specs:
            reference = values[reference_name]
            reference_rms = _latent_rms(reference)
            for term in terms:
                pathway_rows.append(
                    {
                        "case_id": case_id,
                        "mode": mode,
                        "pair": pair,
                        "call": call,
                        "layer": layer,
                        "pathway": pathway,
                        "term": term,
                        "term_rms": _latent_rms(values[term]),
                        "mesh_gap_rms": reference_rms,
                        "term_to_mesh_ratio": _latent_rms(values[term])
                        / max(reference_rms, np.finfo(float).tiny),
                    }
                )
            closure_rows.append(
                _typed_closure_row(
                    case_id=case_id,
                    mode=mode,
                    pair=pair,
                    call=call,
                    layer=layer,
                    closure_kind=f"latent_{pathway}",
                    numerator=_latent_rms(values[closure_name]),
                    denominator=reference_rms,
                    units="normalized_hidden_rms",
                    absolute_tolerance=CLOSURE_TOLERANCE,
                    relative_tolerance=CLOSURE_TOLERANCE,
                )
            )

    for branch in BRANCH_NAMES:
        outputs = trace_native_branch_replacement_output(
            model.backbone,
            coarse_context["model_input"],
            coarse_context["aux"],
            fine_context["model_input"],
            fine_context["aux"],
            branch=branch,
            coarse_resolution=coarse,
            fine_resolution=fine,
        )
        replay_outputs = trace_native_branch_replacement_output(
            model.backbone,
            coarse_context["model_input"],
            coarse_context["aux"],
            fine_context["model_input"],
            fine_context["aux"],
            branch=branch,
            coarse_resolution=coarse,
            fine_resolution=fine,
            replacement_layers=(),
        )
        hybrid = (
            outputs["hybrid"][0].detach().float().cpu().numpy().astype(np.float64)
            * residual_scale[None, :]
        )
        replay_coarse = (
            replay_outputs["hybrid"][0]
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float64)
            * residual_scale[None, :]
        )
        replay_restricted_fine = (
            replay_outputs["restricted_fine"][0]
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float64)
            * residual_scale[None, :]
        )
        _append_trace_row(
            trace_rows,
            case_id=case_id,
            mode=mode,
            pair=pair,
            call=call,
            comparison="no_replacement_coarse",
            layer="all",
            branch=branch,
            value=replay_coarse,
            reference=baseline_coarse,
            baseline_large_rms=baseline_large_rms,
            resolution=coarse,
            residual_scale=residual_scale,
        )
        _append_trace_row(
            trace_rows,
            case_id=case_id,
            mode=mode,
            pair=pair,
            call=call,
            comparison="no_replacement_restricted_fine",
            layer="all",
            branch=branch,
            value=replay_restricted_fine,
            reference=baseline_fine,
            baseline_large_rms=baseline_large_rms,
            resolution=coarse,
            residual_scale=residual_scale,
        )
        arm_dct_closure = _append_arm_rows(
            arm_rows,
            case_id=case_id,
            mode=mode,
            pair=pair,
            call=call,
            layer="all",
            arm=f"{branch}_all_layers",
            record_kind="native_all_layer_branch",
            baseline_defect=baseline_native_defect,
            arm_defect=hybrid - baseline_fine,
            resolution=coarse,
            residual_scale=residual_scale,
        )
        for name, value in arm_dct_closure.items():
            maximum_dct_closure[name] = max(maximum_dct_closure[name], float(value))
    closure_rows.extend(
        [
            _typed_closure_row(
                case_id=case_id,
                mode=mode,
                pair=pair,
                call=call,
                layer="all",
                closure_kind="dct_reconstruction",
                numerator=maximum_dct_closure[
                    "maximum_reconstruction_abs_residual_scaled"
                ],
                denominator=1.0,
                units="maximum_abs_residual_scaled",
                absolute_tolerance=D070C_DCT_TOLERANCE,
                relative_tolerance=D070C_DCT_TOLERANCE,
            ),
            _typed_closure_row(
                case_id=case_id,
                mode=mode,
                pair=pair,
                call=call,
                layer="all",
                closure_kind="dct_energy",
                numerator=maximum_dct_closure[
                    "maximum_instantaneous_energy_relative_closure"
                ],
                denominator=1.0,
                units="relative_energy",
                absolute_tolerance=D070C_DCT_TOLERANCE,
                relative_tolerance=D070C_DCT_TOLERANCE,
            ),
        ]
    )


def _commutator_row(
    *,
    case_id: str,
    mode: str,
    call: int,
    coarse: tuple[int, int],
    fine: tuple[int, int],
    coarse_context: Mapping[str, Any],
    fine_context: Mapping[str, Any],
    paired_coarse_context: Mapping[str, Any],
    residual_scale: np.ndarray,
) -> dict[str, Any]:
    restricted_fine_state = _restrict(fine_context["state"], fine=fine, coarse=coarse)
    restricted_fine_prediction = _restrict(
        fine_context["prediction"][0].detach().float().cpu().numpy().astype(np.float64),
        fine=fine,
        coarse=coarse,
    )
    restricted_fine_increment = _restrict(
        fine_context["increment"], fine=fine, coarse=coarse
    )
    coarse_prediction = (
        coarse_context["prediction"][0]
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    input_gap = coarse_context["state"] - restricted_fine_state
    output_gap = coarse_prediction - restricted_fine_prediction
    increment_gap = coarse_context["increment"] - restricted_fine_increment
    coarse_head = (
        coarse_context["normalized"][0]
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float64)
        * residual_scale[None, :]
    )
    restricted_fine_head = _restrict(
        fine_context["normalized"][0].detach().float().cpu().numpy().astype(np.float64)
        * residual_scale[None, :],
        fine=fine,
        coarse=coarse,
    )
    paired_coarse_head = (
        paired_coarse_context["normalized"][0]
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float64)
        * residual_scale[None, :]
    )
    head_defect = coarse_head - restricted_fine_head
    increment_head_bridge = increment_gap - head_defect
    generalized = output_gap - input_gap - increment_gap
    simple = output_gap - increment_gap
    delta_mesh = paired_coarse_context["increment"] - restricted_fine_increment
    delta_state = coarse_context["increment"] - paired_coarse_context["increment"]
    mesh_state_closure = increment_gap - delta_mesh - delta_state
    mesh_head_defect = paired_coarse_head - restricted_fine_head
    mesh_head_bridge = delta_mesh - mesh_head_defect
    increment_bands, _ = _band_fields(
        increment_gap, resolution=coarse, residual_scale=residual_scale
    )
    increment_bridge_bands, _ = _band_fields(
        increment_head_bridge,
        resolution=coarse,
        residual_scale=residual_scale,
    )
    mesh_bands, _ = _band_fields(
        delta_mesh, resolution=coarse, residual_scale=residual_scale
    )
    mesh_bridge_bands, _ = _band_fields(
        mesh_head_bridge,
        resolution=coarse,
        residual_scale=residual_scale,
    )
    increment_large = _scaled_rms(increment_bands["large"], residual_scale)
    increment_bridge_large = _scaled_rms(
        increment_bridge_bands["large"], residual_scale
    )
    mesh_large = _scaled_rms(mesh_bands["large"], residual_scale)
    mesh_bridge_large = _scaled_rms(mesh_bridge_bands["large"], residual_scale)
    input_metrics = _identity_metrics(input_gap, residual_scale=residual_scale)
    generalized_metrics = _identity_metrics(generalized, residual_scale=residual_scale)
    simple_metrics = _identity_metrics(simple, residual_scale=residual_scale)
    mesh_state_metrics = _identity_metrics(
        mesh_state_closure, residual_scale=residual_scale
    )
    paired_input_applicable = bool(
        mode == "teacher_forced"
        and input_metrics["closure_max_abs_physical"]
        <= D070C_TEACHER_INPUT_ABSOLUTE_TOLERANCE
        and input_metrics["closure_residual_scaled_rms"] <= D070C_RELATIVE_TOLERANCE
    )
    return {
        "case_id": case_id,
        "mode": mode,
        "pair": _pair_label(coarse, fine),
        "call": call,
        "input_gap_max_abs_physical": input_metrics["closure_max_abs_physical"],
        "input_gap_residual_scaled_rms": input_metrics["closure_residual_scaled_rms"],
        "output_gap_residual_scaled_rms": _scaled_rms(output_gap, residual_scale),
        "increment_gap_residual_scaled_rms": _scaled_rms(increment_gap, residual_scale),
        "head_defect_residual_scaled_rms": _scaled_rms(head_defect, residual_scale),
        "increment_head_bridge_max_abs_physical": float(
            np.max(np.abs(increment_head_bridge))
        ),
        "increment_head_bridge_residual_scaled_rms": _scaled_rms(
            increment_head_bridge, residual_scale
        ),
        "increment_defect_large_band_residual_scaled_rms": increment_large,
        "increment_head_bridge_large_band_residual_scaled_rms": (
            increment_bridge_large
        ),
        "increment_head_bridge_large_band_ratio": increment_bridge_large
        / max(increment_large, np.finfo(np.float64).tiny),
        "generalized_closure_max_abs_physical": generalized_metrics[
            "closure_max_abs_physical"
        ],
        "generalized_closure_residual_scaled_rms": generalized_metrics[
            "closure_residual_scaled_rms"
        ],
        "paired_input_identity_applicable": paired_input_applicable,
        "paired_identity_closure_max_abs_physical": simple_metrics[
            "closure_max_abs_physical"
        ],
        "paired_identity_closure_residual_scaled_rms": simple_metrics[
            "closure_residual_scaled_rms"
        ],
        "delta_mesh_residual_scaled_rms": _scaled_rms(delta_mesh, residual_scale),
        "mesh_head_defect_residual_scaled_rms": _scaled_rms(
            mesh_head_defect, residual_scale
        ),
        "mesh_head_bridge_max_abs_physical": float(np.max(np.abs(mesh_head_bridge))),
        "mesh_head_bridge_residual_scaled_rms": _scaled_rms(
            mesh_head_bridge, residual_scale
        ),
        "delta_mesh_large_band_residual_scaled_rms": mesh_large,
        "mesh_head_bridge_large_band_residual_scaled_rms": mesh_bridge_large,
        "mesh_head_bridge_large_band_ratio": mesh_bridge_large
        / max(mesh_large, np.finfo(np.float64).tiny),
        "delta_state_residual_scaled_rms": _scaled_rms(delta_state, residual_scale),
        "mesh_state_closure_max_abs_physical": mesh_state_metrics[
            "closure_max_abs_physical"
        ],
        "mesh_state_closure_residual_scaled_rms": mesh_state_metrics[
            "closure_residual_scaled_rms"
        ],
    }


def _recurrence_row(
    *,
    case_id: str,
    call: int,
    coarse: tuple[int, int],
    fine: tuple[int, int],
    coarse_state: np.ndarray,
    fine_state: np.ndarray,
    coarse_context: Mapping[str, Any],
    fine_context: Mapping[str, Any],
    residual_scale: np.ndarray,
) -> dict[str, Any]:
    error_before = np.asarray(coarse_state, dtype=np.float64) - _restrict(
        fine_state, fine=fine, coarse=coarse
    )
    coarse_prediction = (
        coarse_context["prediction"][0]
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    fine_prediction = (
        fine_context["prediction"][0].detach().float().cpu().numpy().astype(np.float64)
    )
    error_after = coarse_prediction - _restrict(
        fine_prediction, fine=fine, coarse=coarse
    )
    delta = coarse_context["increment"] - _restrict(
        fine_context["increment"], fine=fine, coarse=coarse
    )
    closure = error_after - error_before - delta
    metrics = _identity_metrics(closure, residual_scale=residual_scale)
    return {
        "case_id": case_id,
        "pair": _pair_label(coarse, fine),
        "call": call,
        "error_before_residual_scaled_rms": _scaled_rms(error_before, residual_scale),
        "delta_residual_scaled_rms": _scaled_rms(delta, residual_scale),
        "error_after_residual_scaled_rms": _scaled_rms(error_after, residual_scale),
        "closure_max_abs_physical": metrics["closure_max_abs_physical"],
        "closure_residual_scaled_rms": metrics["closure_residual_scaled_rms"],
    }


def _d070c_contract_checks(
    *,
    active_case_ids: Sequence[str],
    active_calls: int,
    diagnostic_calls: Sequence[int],
    layer_count: int,
    arm_rows: Sequence[Mapping[str, Any]],
    pathway_rows: Sequence[Mapping[str, Any]],
    closure_rows: Sequence[Mapping[str, Any]],
    direct_output_rows: Sequence[Mapping[str, Any]],
    commutator_rows: Sequence[Mapping[str, Any]],
    recurrence_rows: Sequence[Mapping[str, Any]],
    trace_rows: Sequence[Mapping[str, Any]],
    completion_rows: Sequence[Mapping[str, Any]],
    reference_checks: Sequence[Mapping[str, Any]],
    model_state_before: str,
    model_state_after: str,
) -> dict[str, Any]:
    pairs = tuple(_pair_label(*value) for value in pairwise(RESOLUTIONS))
    modes = ("teacher_forced", "free_rollout")
    bands = ("total", "large", "transition", "local")
    contexts = {
        (case_id, mode, pair, int(call))
        for case_id in active_case_ids
        for mode in modes
        for pair in pairs
        for call in diagnostic_calls
    }
    arm_expected = {
        (
            "same_hidden_single_layer",
            case_id,
            mode,
            pair,
            call,
            layer,
            arm,
            band,
        )
        for case_id, mode, pair, call in contexts
        for layer in range(layer_count)
        for arm in SAME_HIDDEN_ARMS
        for band in bands
    } | {
        (
            "native_all_layer_branch",
            case_id,
            mode,
            pair,
            call,
            "all",
            f"{branch}_all_layers",
            band,
        )
        for case_id, mode, pair, call in contexts
        for branch in BRANCH_NAMES
        for band in bands
    }
    pathway_expected = {
        (case_id, mode, pair, call, layer, pathway, term)
        for case_id, mode, pair, call in contexts
        for layer in range(layer_count)
        for pathway, terms in PATHWAY_TERMS.items()
        for term in terms
    }
    closure_expected = {
        (case_id, mode, pair, call, layer, f"latent_{pathway}")
        for case_id, mode, pair, call in contexts
        for layer in range(layer_count)
        for pathway in PATHWAY_TERMS
    } | {
        (case_id, mode, pair, call, "all", closure_kind)
        for case_id, mode, pair, call in contexts
        for closure_kind in ("dct_reconstruction", "dct_energy")
    }
    direct_expected = {
        (case_id, mode, pair, call, output_role)
        for case_id, mode, pair, call in contexts
        for output_role in ("coarse", "restricted_fine")
    }
    commutator_expected = set(contexts)
    recurrence_expected = {
        (case_id, pair, call)
        for case_id in active_case_ids
        for pair in pairs
        for call in range(1, active_calls + 1)
    }
    trace_expected = (
        {
            (case_id, mode, pair, call, comparison, "all", "none")
            for case_id, mode, pair, call in contexts
            for comparison in (
                "backbone_direct_coarse",
                "backbone_direct_restricted_fine",
            )
        }
        | {
            (case_id, mode, pair, call, comparison, "all", branch)
            for case_id, mode, pair, call in contexts
            for comparison in (
                "no_replacement_coarse",
                "no_replacement_restricted_fine",
            )
            for branch in BRANCH_NAMES
        }
        | {
            (
                case_id,
                mode,
                pair,
                call,
                "same_hidden_restricted_fine",
                layer,
                "none",
            )
            for case_id, mode, pair, call in contexts
            for layer in range(layer_count)
        }
    )
    completion_expected = {(case_id,) for case_id in active_case_ids}
    reference_expected = {(case_id,) for case_id in active_case_ids}

    inventories = {
        "arm_metrics": _inventory_checks(
            arm_rows,
            key_fields=(
                "record_kind",
                "case_id",
                "mode",
                "pair",
                "call",
                "layer",
                "arm",
                "band",
            ),
            expected_keys=arm_expected,
            finite_fields=(
                "baseline_defect_rms",
                "arm_defect_rms",
                "defect_norm_ratio",
                "defect_reduction",
                "response_rms",
            ),
        ),
        "pathway_terms": _inventory_checks(
            pathway_rows,
            key_fields=(
                "case_id",
                "mode",
                "pair",
                "call",
                "layer",
                "pathway",
                "term",
            ),
            expected_keys=pathway_expected,
            finite_fields=("term_rms", "mesh_gap_rms", "term_to_mesh_ratio"),
        ),
        "typed_closures": _inventory_checks(
            closure_rows,
            key_fields=(
                "case_id",
                "mode",
                "pair",
                "call",
                "layer",
                "closure_kind",
            ),
            expected_keys=closure_expected,
            finite_fields=(
                "closure_numerator",
                "closure_denominator",
                "absolute_closure",
                "relative_closure",
                "absolute_tolerance",
                "relative_tolerance",
            ),
        ),
        "direct_outputs": _inventory_checks(
            direct_output_rows,
            key_fields=("case_id", "mode", "pair", "call", "output_role"),
            expected_keys=direct_expected,
            finite_fields=(
                "output_residual_scaled_rms",
                "large_band_residual_scaled_rms",
            ),
        ),
        "commutators": _inventory_checks(
            commutator_rows,
            key_fields=("case_id", "mode", "pair", "call"),
            expected_keys=commutator_expected,
            finite_fields=(
                "input_gap_max_abs_physical",
                "input_gap_residual_scaled_rms",
                "output_gap_residual_scaled_rms",
                "increment_gap_residual_scaled_rms",
                "head_defect_residual_scaled_rms",
                "increment_head_bridge_max_abs_physical",
                "increment_head_bridge_residual_scaled_rms",
                "increment_defect_large_band_residual_scaled_rms",
                "increment_head_bridge_large_band_residual_scaled_rms",
                "increment_head_bridge_large_band_ratio",
                "generalized_closure_max_abs_physical",
                "generalized_closure_residual_scaled_rms",
                "paired_identity_closure_max_abs_physical",
                "paired_identity_closure_residual_scaled_rms",
                "delta_mesh_residual_scaled_rms",
                "mesh_head_defect_residual_scaled_rms",
                "mesh_head_bridge_max_abs_physical",
                "mesh_head_bridge_residual_scaled_rms",
                "delta_mesh_large_band_residual_scaled_rms",
                "mesh_head_bridge_large_band_residual_scaled_rms",
                "mesh_head_bridge_large_band_ratio",
                "delta_state_residual_scaled_rms",
                "mesh_state_closure_max_abs_physical",
                "mesh_state_closure_residual_scaled_rms",
            ),
        ),
        "recurrence": _inventory_checks(
            recurrence_rows,
            key_fields=("case_id", "pair", "call"),
            expected_keys=recurrence_expected,
            finite_fields=(
                "error_before_residual_scaled_rms",
                "delta_residual_scaled_rms",
                "error_after_residual_scaled_rms",
                "closure_max_abs_physical",
                "closure_residual_scaled_rms",
            ),
        ),
        "trace_replay": _inventory_checks(
            trace_rows,
            key_fields=(
                "case_id",
                "mode",
                "pair",
                "call",
                "comparison",
                "layer",
                "branch",
            ),
            expected_keys=trace_expected,
            finite_fields=(
                "max_abs_physical",
                "error_residual_scaled_rms",
                "reference_residual_scaled_rms",
                "relative_residual_scaled_rms",
                "large_band_error_residual_scaled_rms",
                "baseline_large_band_residual_scaled_rms",
                "large_band_error_to_baseline_ratio",
            ),
        ),
        "completion": _inventory_checks(
            completion_rows,
            key_fields=("case_id",),
            expected_keys=completion_expected,
            finite_fields=("calls", "seconds"),
        ),
        "reference_checks": _inventory_checks(
            reference_checks,
            key_fields=("case_id",),
            expected_keys=reference_expected,
            finite_fields=("restriction_crosscheck_max_abs",),
        ),
    }
    inventories_pass = all(
        value["complete_and_finite"] for value in inventories.values()
    )
    selector_rows = [
        row
        for row in arm_rows
        if row.get("record_kind") == "same_hidden_single_layer"
        and row.get("band") in {"total", "large", "local"}
    ]
    selector_minimum = min(
        (float(row["baseline_defect_rms"]) for row in selector_rows),
        default=0.0,
    )
    selector_denominators_pass = bool(
        selector_rows and selector_minimum > D070C_SELECTOR_DENOMINATOR_MINIMUM
    )
    generalized_identity_pass = bool(
        commutator_rows
        and all(
            float(row["generalized_closure_max_abs_physical"])
            <= D070C_ABSOLUTE_TOLERANCE
            and float(row["generalized_closure_residual_scaled_rms"])
            <= D070C_RELATIVE_TOLERANCE
            for row in commutator_rows
        )
    )
    mesh_state_identity_pass = bool(
        commutator_rows
        and all(
            float(row["mesh_state_closure_max_abs_physical"])
            <= D070C_ABSOLUTE_TOLERANCE
            and float(row["mesh_state_closure_residual_scaled_rms"])
            <= D070C_RELATIVE_TOLERANCE
            for row in commutator_rows
        )
    )
    residual_head_bridge_pass = bool(
        commutator_rows
        and all(
            float(row["increment_head_bridge_max_abs_physical"])
            <= D070C_ABSOLUTE_TOLERANCE
            and float(row["increment_head_bridge_residual_scaled_rms"])
            <= D070C_RELATIVE_TOLERANCE
            and float(row["increment_defect_large_band_residual_scaled_rms"])
            > D070C_SELECTOR_DENOMINATOR_MINIMUM
            and float(row["increment_head_bridge_large_band_ratio"])
            <= D070C_TRACE_LARGE_RATIO_TOLERANCE
            and float(row["mesh_head_bridge_max_abs_physical"])
            <= D070C_ABSOLUTE_TOLERANCE
            and float(row["mesh_head_bridge_residual_scaled_rms"])
            <= D070C_RELATIVE_TOLERANCE
            and float(row["delta_mesh_large_band_residual_scaled_rms"])
            > D070C_SELECTOR_DENOMINATOR_MINIMUM
            and float(row["mesh_head_bridge_large_band_ratio"])
            <= D070C_TRACE_LARGE_RATIO_TOLERANCE
            for row in commutator_rows
        )
    )
    teacher_rows = [
        row for row in commutator_rows if row.get("mode") == "teacher_forced"
    ]
    teacher_input_floor_pass = bool(
        teacher_rows
        and all(
            bool(row["paired_input_identity_applicable"])
            and float(row["input_gap_max_abs_physical"])
            <= D070C_TEACHER_INPUT_ABSOLUTE_TOLERANCE
            and float(row["input_gap_residual_scaled_rms"]) <= D070C_RELATIVE_TOLERANCE
            for row in teacher_rows
        )
    )
    paired_identity_pass = bool(
        teacher_input_floor_pass
        and all(
            float(row["paired_identity_closure_max_abs_physical"])
            <= D070C_ABSOLUTE_TOLERANCE
            and float(row["paired_identity_closure_residual_scaled_rms"])
            <= D070C_RELATIVE_TOLERANCE
            for row in teacher_rows
        )
    )
    recurrence_identity_pass = bool(
        recurrence_rows
        and all(
            float(row["closure_max_abs_physical"]) <= D070C_ABSOLUTE_TOLERANCE
            and float(row["closure_residual_scaled_rms"]) <= D070C_RELATIVE_TOLERANCE
            for row in recurrence_rows
        )
    )
    trace_replay_pass = bool(
        trace_rows and all(bool(row["passed"]) for row in trace_rows)
    )
    typed_closures_pass = bool(
        closure_rows and all(bool(row["passed"]) for row in closure_rows)
    )
    reference_binding_pass = bool(
        reference_checks
        and all(
            float(row["restriction_crosscheck_max_abs"])
            <= D070C_TEACHER_INPUT_ABSOLUTE_TOLERANCE
            for row in reference_checks
        )
    )
    return {
        "inventories": inventories,
        "inventories_pass": inventories_pass,
        "selector_minimum_baseline_residual_scaled_rms": selector_minimum,
        "selector_denominators_pass": selector_denominators_pass,
        "generalized_commutator_identity_pass": generalized_identity_pass,
        "mesh_state_identity_pass": mesh_state_identity_pass,
        "residual_head_to_increment_bridge_pass": residual_head_bridge_pass,
        "teacher_input_restriction_floor_pass": teacher_input_floor_pass,
        "paired_input_commutator_identity_pass": paired_identity_pass,
        "free_recurrence_identity_pass": recurrence_identity_pass,
        "trace_replay_pass": trace_replay_pass,
        "typed_closures_pass": typed_closures_pass,
        "reference_binding_pass": reference_binding_pass,
        "model_state_sha256_before": model_state_before,
        "model_state_sha256_after": model_state_after,
        "model_state_immutable_pass": model_state_before == model_state_after,
    }


def _contract_checks_pass(
    source_contract: str,
    checks: Mapping[str, Any],
    d070c_checks: Mapping[str, Any],
) -> bool:
    if source_contract == CURRENT_SELF_CONSISTENT_SOURCE_CONTRACT:
        return all(
            bool(d070c_checks[name])
            for name in (
                "inventories_pass",
                "selector_denominators_pass",
                "generalized_commutator_identity_pass",
                "mesh_state_identity_pass",
                "residual_head_to_increment_bridge_pass",
                "teacher_input_restriction_floor_pass",
                "paired_input_commutator_identity_pass",
                "free_recurrence_identity_pass",
                "trace_replay_pass",
                "typed_closures_pass",
                "reference_binding_pass",
                "model_state_immutable_pass",
            )
        ) and all(
            bool(checks[name])
            for name in (
                "all_baseline_outputs_admissible",
                "hook_equivalence_pass",
            )
        )
    return all(
        bool(checks[name])
        for name in (
            "d067_source_compatibility_pass",
            "pathway_closure_pass",
            "all_baseline_outputs_admissible",
            "hook_equivalence_pass",
        )
    )


def _per_case_medians(
    rows: Sequence[Mapping[str, Any]], *, band: str, field: str
) -> list[float]:
    aggregated = []
    for case_id in CASE_IDS:
        values = [
            float(row[field])
            for row in rows
            if row["band"] == band
            and row["case_id"] == case_id
            and row[field] is not None
        ]
        if values:
            aggregated.append(float(np.median(values)))
    return aggregated


def _mechanism_selection(
    arm_rows: Sequence[Mapping[str, Any]], pairs: Sequence[str]
) -> dict[str, Any]:
    primary_rows = [
        row for row in arm_rows if row.get("record_kind") == "same_hidden_single_layer"
    ]
    candidates = sorted({(str(row["arm"]), str(row["layer"])) for row in primary_rows})
    summaries = []
    for arm, layer in candidates:
        pair_summaries = []
        qualifies = True
        for pair in pairs:
            mode_summaries = []
            for mode in ("teacher_forced", "free_rollout"):
                selected = [
                    row
                    for row in primary_rows
                    if str(row["arm"]) == arm
                    and str(row["layer"]) == layer
                    and str(row["pair"]) == pair
                    and str(row["mode"]) == mode
                ]
                expected = {
                    (case_id, call, band)
                    for case_id in CASE_IDS
                    for call in CALLS
                    for band in ("total", "large", "transition", "local")
                }
                observed = [
                    (str(row["case_id"]), int(row["call"]), str(row["band"]))
                    for row in selected
                ]
                matrix_complete = (
                    len(observed) == len(expected) and set(observed) == expected
                )
                case_large = _per_case_medians(
                    selected, band="large", field="defect_reduction"
                )
                large_median = None if not case_large else float(np.median(case_large))
                positive_cases = sum(value > 0.0 for value in case_large)
                ratios = {}
                ratio_case_counts = {}
                for band in ("total", "local"):
                    values = _per_case_medians(
                        selected, band=band, field="defect_norm_ratio"
                    )
                    ratio_case_counts[band] = len(values)
                    ratios[band] = None if not values else float(np.median(values))
                mode_pass = bool(
                    matrix_complete
                    and large_median is not None
                    and len(case_large) == len(CASE_IDS)
                    and large_median >= LARGE_REDUCTION_GATE
                    and positive_cases >= POSITIVE_CASE_GATE
                    and ratio_case_counts["total"] == len(CASE_IDS)
                    and ratios["total"] is not None
                    and ratios["total"] <= OTHER_BAND_RATIO_GATE
                    and ratio_case_counts["local"] == len(CASE_IDS)
                    and ratios["local"] is not None
                    and ratios["local"] <= OTHER_BAND_RATIO_GATE
                )
                mode_summaries.append(
                    {
                        "mode": mode,
                        "matrix_complete": matrix_complete,
                        "median_large_reduction": large_median,
                        "large_band_case_count": len(case_large),
                        "positive_case_count": positive_cases,
                        "median_total_ratio": ratios["total"],
                        "total_band_case_count": ratio_case_counts["total"],
                        "median_local_ratio": ratios["local"],
                        "local_band_case_count": ratio_case_counts["local"],
                        "passes": mode_pass,
                    }
                )
            pair_pass = all(value["passes"] for value in mode_summaries)
            pair_large = [
                value["median_large_reduction"]
                for value in mode_summaries
                if value["median_large_reduction"] is not None
            ]
            qualifies = qualifies and pair_pass
            pair_summaries.append(
                {
                    "pair": pair,
                    "median_large_reduction": (
                        None if not pair_large else float(np.mean(pair_large))
                    ),
                    "modes": mode_summaries,
                    "passes": pair_pass,
                }
            )
        summaries.append(
            {
                "arm": arm,
                "layer": layer,
                "qualifies": qualifies,
                "pairs": pair_summaries,
            }
        )
    qualified = [row for row in summaries if row["qualifies"]]
    if not qualified:
        return {
            "decision": "composite_or_unresolved",
            "primary_record_kind": "same_hidden_single_layer",
            "candidates": summaries,
            "qualified": [],
        }
    if len(qualified) > 1:
        return {
            "decision": "multiple_supported",
            "primary_record_kind": "same_hidden_single_layer",
            "candidates": summaries,
            "qualified": qualified,
        }
    return {
        "decision": "selected",
        "primary_record_kind": "same_hidden_single_layer",
        "selected": qualified[0],
        "candidates": summaries,
        "qualified": qualified,
    }


def _verify_d067(
    args: argparse.Namespace, checkpoint: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, str]]:
    if sha256_file(args.d067_summary) != args.expected_d067_summary_sha256.lower():
        raise ValueError("D067 summary digest mismatch")
    summary = json.loads(args.d067_summary.read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise ValueError("D067 retained summary is not complete")
    if summary.get("checkpoint_sha256") != args.expected_checkpoint_sha256.lower():
        raise ValueError("D067 and D070 checkpoints differ")
    if summary.get("normalization_digest") != checkpoint.get("normalization_digest"):
        raise ValueError("D067 and D070 normalizers differ")
    registered_sources = summary.get("source_hashes")
    if not isinstance(registered_sources, dict) or set(registered_sources) != set(
        D067_INHERITED_SOURCE_PATHS
    ):
        raise ValueError("D067 inherited source inventory mismatch")
    active_sources = {
        relative: sha256_file(ROOT / relative)
        for relative in D067_INHERITED_SOURCE_PATHS
    }
    mismatched_sources = {
        relative: {
            "registered": str(registered_sources[relative]).lower(),
            "active": active_sources[relative],
        }
        for relative in D067_INHERITED_SOURCE_PATHS
        if active_sources[relative] != str(registered_sources[relative]).lower()
    }
    if (
        args.d067_source_contract == D067_INHERITED_SOURCE_CONTRACT
        and mismatched_sources
    ):
        raise ValueError(f"D067 inherited source hash mismatch: {mismatched_sources}")
    for case_id in CASE_IDS:
        relative = f"bundles/{case_id}.npz"
        if (
            sha256_file(args.d067_summary.parent / relative)
            != summary["output_hashes"][relative]
        ):
            raise ValueError(f"D067 bundle digest mismatch: {case_id}")
    return summary, {
        "source_contract": args.d067_source_contract,
        "exact_replay_claimed": (
            args.d067_source_contract == D067_INHERITED_SOURCE_CONTRACT
        ),
        "cross_version_comparability_claimed": (
            args.d067_source_contract == D067_INHERITED_SOURCE_CONTRACT
        ),
        "registered_source_sha256": {
            relative: str(registered_sources[relative]).lower()
            for relative in D067_INHERITED_SOURCE_PATHS
        },
        "active_source_sha256": active_sources,
        "source_mismatches": mismatched_sources,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = perf_counter()
    device = select_device(args.device)
    store = PCNOEuler2DShardStore(args.data_dir, max_cached_trajectories=2)
    checkpoint = load_resolution_checkpoint(args.checkpoint)
    provenance = _verify_common_provenance(args, checkpoint, store)
    d067_summary, d067_source_binding = _verify_d067(args, checkpoint)
    model, _ = build_resolution_checkpoint_model(checkpoint, device)
    model.eval()
    model_state_before = _model_state_sha256(model)
    cases, reference_checks = _load_dynamic_cases(
        args, checkpoint, store, device=device
    )
    grouped: dict[str, dict[tuple[int, int], CaseData]] = defaultdict(dict)
    for case in cases:
        if case.resolution is None:
            raise AssertionError("D070 loaded an unstructured case")
        grouped[case.case_id][case.resolution] = case
    if set(grouped) != set(CASE_IDS):
        raise ValueError("D070 case population changed")
    hook_equivalence = _hook_equivalence(
        model,
        grouped[CASE_IDS[0]][RESOLUTIONS[1]],
        device=device,
        amp="none",
    )
    args.output_dir.mkdir(parents=True)
    arm_rows: list[dict[str, Any]] = []
    pathway_rows: list[dict[str, Any]] = []
    closure_rows: list[dict[str, Any]] = []
    direct_output_rows: list[dict[str, Any]] = []
    commutator_rows: list[dict[str, Any]] = []
    recurrence_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    completion_rows: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    maximum_d067_replay = 0.0
    active_case_ids = CASE_IDS[:1] if args.smoke else CASE_IDS
    active_calls = 1 if args.smoke else args.rollout_calls
    diagnostic_calls = {1} if args.smoke else set(CALLS)
    active_reference_checks = [
        row for row in reference_checks if row.get("case_id") in active_case_ids
    ]
    residual_scale = np.asarray(
        checkpoint["normalization"]["residual_scale"], dtype=np.float64
    )

    try:
        for case_index, case_id in enumerate(active_case_ids, start=1):
            case_started = perf_counter()
            by_resolution = grouped[case_id]
            free_states = {
                resolution: np.array(case.reference_states[0], copy=True)
                for resolution, case in by_resolution.items()
            }
            bundle_path = args.d067_summary.parent / "bundles" / f"{case_id}.npz"
            with np.load(bundle_path, allow_pickle=False) as bundle:
                bundle_arrays = {
                    name: np.asarray(bundle[name])
                    for name in bundle.files
                    if not name.endswith("metadata_json")
                }
            all_admissible = True
            for step in range(active_calls):
                call = step + 1
                teacher_contexts: dict[tuple[int, int], dict[str, Any]] = {}
                free_contexts: dict[tuple[int, int], dict[str, Any]] = {}
                for resolution in RESOLUTIONS:
                    case = by_resolution[resolution]
                    if call in diagnostic_calls:
                        teacher_contexts[resolution] = _model_context(
                            model, case, case.reference_states[step]
                        )
                    free_contexts[resolution] = _model_context(
                        model, case, free_states[resolution]
                    )
                    prediction = (
                        free_contexts[resolution]["prediction"][0]
                        .detach()
                        .float()
                        .cpu()
                        .numpy()
                    )
                    admissibility = conservative_admissibility_summary(
                        prediction, gamma=case.gamma
                    )
                    all_admissible = all_admissible and bool(
                        admissibility["finite"] and admissibility["admissible"]
                    )

                for coarse, fine in pairwise(RESOLUTIONS):
                    key = _pair_key(coarse, fine)
                    recurrence_rows.append(
                        _recurrence_row(
                            case_id=case_id,
                            call=call,
                            coarse=coarse,
                            fine=fine,
                            coarse_state=free_states[coarse],
                            fine_state=free_states[fine],
                            coarse_context=free_contexts[coarse],
                            fine_context=free_contexts[fine],
                            residual_scale=residual_scale,
                        )
                    )
                    current_coarse_state = (
                        free_contexts[coarse]["prediction"][0]
                        .detach()
                        .float()
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                    )
                    current_restricted_fine_state = _restrict(
                        free_contexts[fine]["prediction"][0]
                        .detach()
                        .float()
                        .cpu()
                        .numpy()
                        .astype(np.float64),
                        fine=fine,
                        coarse=coarse,
                    )
                    registered_coarse_state = bundle_arrays[
                        f"free_coarse_states__{key}"
                    ][step + 1]
                    registered_restricted_fine_state = bundle_arrays[
                        f"free_fine_states__{key}"
                    ][step + 1]
                    current_coarse_increment = free_contexts[coarse]["increment"]
                    current_restricted_fine_increment = _restrict(
                        free_contexts[fine]["increment"],
                        fine=fine,
                        coarse=coarse,
                    )
                    registered_coarse_increment = bundle_arrays[
                        f"coarse_increment_free__{key}"
                    ][step]
                    registered_restricted_fine_increment = bundle_arrays[
                        f"fine_increment_free__{key}"
                    ][step]
                    replay_values = (
                        np.max(
                            np.abs(
                                free_states[coarse]
                                - bundle_arrays[f"free_coarse_states__{key}"][step]
                            )
                        ),
                        np.max(
                            np.abs(
                                _restrict(free_states[fine], fine=fine, coarse=coarse)
                                - bundle_arrays[f"free_fine_states__{key}"][step]
                            )
                        ),
                        np.max(
                            np.abs(
                                current_coarse_increment - registered_coarse_increment
                            )
                        ),
                        np.max(
                            np.abs(
                                current_restricted_fine_increment
                                - registered_restricted_fine_increment
                            )
                        ),
                    )
                    maximum_d067_replay = max(
                        maximum_d067_replay,
                        *(float(value) for value in replay_values),
                    )
                    current_commutator = (
                        current_coarse_increment - current_restricted_fine_increment
                    )
                    registered_commutator = (
                        registered_coarse_increment
                        - registered_restricted_fine_increment
                    )
                    commutator_drift = current_commutator - registered_commutator
                    current_bands, current_band_closure = _band_fields(
                        current_commutator,
                        resolution=coarse,
                        residual_scale=residual_scale,
                    )
                    drift_bands, drift_band_closure = _band_fields(
                        commutator_drift,
                        resolution=coarse,
                        residual_scale=residual_scale,
                    )
                    baseline_large_rms = _scaled_rms(
                        current_bands["large"],
                        residual_scale,
                    )
                    drift_large_rms = _scaled_rms(
                        drift_bands["large"],
                        residual_scale,
                    )
                    replay_rows.append(
                        {
                            "case_id": case_id,
                            "pair": _pair_label(coarse, fine),
                            "call": call,
                            "coarse_state_max_abs_physical": float(
                                np.max(
                                    np.abs(
                                        current_coarse_state - registered_coarse_state
                                    )
                                )
                            ),
                            "restricted_fine_state_max_abs_physical": float(
                                np.max(
                                    np.abs(
                                        current_restricted_fine_state
                                        - registered_restricted_fine_state
                                    )
                                )
                            ),
                            "coarse_state_relative_l2": _relative_l2(
                                current_coarse_state,
                                registered_coarse_state,
                            ),
                            "restricted_fine_state_relative_l2": _relative_l2(
                                current_restricted_fine_state,
                                registered_restricted_fine_state,
                            ),
                            "coarse_increment_drift_residual_scaled_rms": (
                                _scaled_rms(
                                    current_coarse_increment
                                    - registered_coarse_increment,
                                    residual_scale,
                                )
                            ),
                            "restricted_fine_increment_drift_residual_scaled_rms": (
                                _scaled_rms(
                                    current_restricted_fine_increment
                                    - registered_restricted_fine_increment,
                                    residual_scale,
                                )
                            ),
                            "commutator_drift_residual_scaled_rms": _scaled_rms(
                                commutator_drift,
                                residual_scale,
                            ),
                            "baseline_large_mesh_defect_residual_scaled_rms": (
                                baseline_large_rms
                            ),
                            "large_commutator_drift_residual_scaled_rms": (
                                drift_large_rms
                            ),
                            "large_commutator_drift_to_baseline_ratio": (
                                drift_large_rms / baseline_large_rms
                                if baseline_large_rms > 0.0
                                else float("inf")
                            ),
                            "band_reconstruction_closure": max(
                                *current_band_closure.values(),
                                *drift_band_closure.values(),
                            ),
                        }
                    )
                    if call in diagnostic_calls:
                        for mode, contexts in (
                            ("teacher_forced", teacher_contexts),
                            ("free_rollout", free_contexts),
                        ):
                            paired_coarse_context = _model_context(
                                model,
                                by_resolution[coarse],
                                _restrict(
                                    contexts[fine]["state"],
                                    fine=fine,
                                    coarse=coarse,
                                ),
                            )
                            commutator_rows.append(
                                _commutator_row(
                                    case_id=case_id,
                                    mode=mode,
                                    call=call,
                                    coarse=coarse,
                                    fine=fine,
                                    coarse_context=contexts[coarse],
                                    fine_context=contexts[fine],
                                    paired_coarse_context=paired_coarse_context,
                                    residual_scale=residual_scale,
                                )
                            )
                            _analyze_pair(
                                model,
                                case_id=case_id,
                                mode=mode,
                                call=call,
                                coarse_case=by_resolution[coarse],
                                fine_case=by_resolution[fine],
                                coarse_context=contexts[coarse],
                                fine_context=contexts[fine],
                                arm_rows=arm_rows,
                                pathway_rows=pathway_rows,
                                closure_rows=closure_rows,
                                direct_output_rows=direct_output_rows,
                                trace_rows=trace_rows,
                                residual_scale=residual_scale,
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
    maximum_closure = max(
        (abs(float(row["relative_closure"])) for row in closure_rows), default=0.0
    )
    compatibility_checks = _historical_compatibility_checks(
        replay_rows,
        expected_keys={
            (case_id, _pair_label(coarse, fine), call)
            for case_id in active_case_ids
            for coarse, fine in pairwise(RESOLUTIONS)
            for call in range(1, active_calls + 1)
        },
    )
    historical_compatibility_pass = all(
        bool(compatibility_checks[name])
        for name in (
            "complete_and_finite",
            "state_absolute_pass",
            "state_relative_pass",
            "science_scale_pass",
            "band_reconstruction_pass",
        )
    )
    exact_replay_pass = maximum_d067_replay <= REPLAY_TOLERANCE
    if args.d067_source_contract == D067_INHERITED_SOURCE_CONTRACT:
        source_compatibility_pass: bool | None = exact_replay_pass
    elif args.d067_source_contract == HISTORICAL_SOURCE_CONTRACT:
        source_compatibility_pass = historical_compatibility_pass
    else:
        source_compatibility_pass = None
    d070c_checks = _d070c_contract_checks(
        active_case_ids=active_case_ids,
        active_calls=active_calls,
        diagnostic_calls=sorted(diagnostic_calls),
        layer_count=len(model.backbone.ws),
        arm_rows=arm_rows,
        pathway_rows=pathway_rows,
        closure_rows=closure_rows,
        direct_output_rows=direct_output_rows,
        commutator_rows=commutator_rows,
        recurrence_rows=recurrence_rows,
        trace_rows=trace_rows,
        completion_rows=completion_rows,
        reference_checks=active_reference_checks,
        model_state_before=model_state_before,
        model_state_after=model_state_after,
    )
    checks = {
        "d067_native_replay_max_abs_physical": maximum_d067_replay,
        "d067_native_replay_pass": exact_replay_pass,
        "historical_output_compatibility": compatibility_checks,
        "d067_source_compatibility_pass": source_compatibility_pass,
        "maximum_pathway_or_dct_closure": maximum_closure,
        "pathway_closure_pass": maximum_closure <= CLOSURE_TOLERANCE,
        "all_baseline_outputs_admissible": all(
            row["all_baseline_outputs_admissible"] for row in completion_rows
        ),
        "hook_equivalence_pass": bool(hook_equivalence["passed"]),
        "current_core_self_consistent": d070c_checks,
    }
    contract_pass = _contract_checks_pass(
        args.d067_source_contract, checks, d070c_checks
    )
    if args.smoke:
        status = "smoke_complete" if contract_pass else "smoke_failed"
    else:
        status = "complete" if contract_pass else "failed_contract"
    pairs = [_pair_label(*value) for value in pairwise(RESOLUTIONS)]
    mechanism = (
        None
        if args.smoke or not contract_pass
        else _mechanism_selection(arm_rows, pairs)
    )
    outputs = {
        "arm_metrics.csv": arm_rows,
        "pathway_terms.csv": pathway_rows,
        "closure_metrics.csv": closure_rows,
        "direct_output_metrics.csv": direct_output_rows,
        "commutator_metrics.csv": commutator_rows,
        "recurrence_metrics.csv": recurrence_rows,
        "trace_replay_metrics.csv": trace_rows,
        "completion.csv": completion_rows,
        "replay_metrics.csv": replay_rows,
        "reference_checks.csv": active_reference_checks,
    }
    for name, rows in outputs.items():
        write_csv_with_paths(args.output_dir / name, rows)
    output_hashes = {name: sha256_file(args.output_dir / name) for name in outputs}
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
            "split": "validation",
            "case_ids": list(active_case_ids),
            "sealed_populations_accessed": False,
        },
        "boundary_policy": "model_all_nodes raw recurrence; unchanged",
        "claim_boundary": {
            "resolution": "dynamic FV only with common-source conservative restriction",
            "current_core": (
                "causal pathway support within one frozen current evaluator; "
                "not exact D067 replay or cross-version attribution"
            ),
            "same_hidden": "single-layer representation interventions, not deployable models",
            "all_layer": "native-fine branch replacement is a causal diagnostic, not training",
            "quadrature": (
                "controlled coarse-hidden injection response; not exact continuum "
                "quadrature error"
            ),
            "subcell": "fine hidden content lost by coarse restriction",
            "fixed_hop": "decoded fixed-hop response includes nonlinear noncommutation",
        },
        "contract": {
            "diagnostic_calls": sorted(diagnostic_calls),
            "modes": ["teacher_forced", "free_rollout"],
            "bands": ["large>=0.125", "transition=0.05..0.125", "local<0.05"],
            "selection": {
                "primary_record_kind": "same_hidden_single_layer",
                "required_modes": ["teacher_forced", "free_rollout"],
                "required_calls": list(CALLS),
                "mode_gate": "passes separately in each required mode",
                "median_large_reduction_minimum": LARGE_REDUCTION_GATE,
                "positive_cases_per_pair_minimum": POSITIVE_CASE_GATE,
                "maximum_total_or_local_ratio": OTHER_BAND_RATIO_GATE,
                "unique_decision": "selected only when exactly one arm qualifies",
            },
            "aggregation": "within case before across-case; no node pooling",
            "current_core_self_consistent": {
                "enabled": (
                    args.d067_source_contract == CURRENT_SELF_CONSISTENT_SOURCE_CONTRACT
                ),
                "identity_max_abs_physical_limit": D070C_ABSOLUTE_TOLERANCE,
                "identity_residual_scaled_rms_limit": D070C_RELATIVE_TOLERANCE,
                "teacher_input_max_abs_physical_limit": (
                    D070C_TEACHER_INPUT_ABSOLUTE_TOLERANCE
                ),
                "trace_max_abs_physical_limit": D070C_ABSOLUTE_TOLERANCE,
                "trace_relative_residual_scaled_rms_limit": (D070C_RELATIVE_TOLERANCE),
                "trace_large_band_to_baseline_ratio_limit": (
                    D070C_TRACE_LARGE_RATIO_TOLERANCE
                ),
                "residual_head_bridge_max_abs_and_scaled_rms_limit": (
                    D070C_ABSOLUTE_TOLERANCE
                ),
                "residual_head_bridge_large_band_ratio_limit": (
                    D070C_TRACE_LARGE_RATIO_TOLERANCE
                ),
                "selector_target": (
                    "decoded residual-head defect linked to F=N-U only by "
                    "the passing native and paired-mesh bridges"
                ),
                "selector_denominator_minimum": (D070C_SELECTOR_DENOMINATOR_MINIMUM),
                "latent_closure_absolute_and_relative_limit": CLOSURE_TOLERANCE,
                "dct_closure_limit": D070C_DCT_TOLERANCE,
                "d067_metrics_enter_promotion": False,
            },
            "d067_output_compatibility": {
                "source_contract": args.d067_source_contract,
                "exact_replay_claimed": d067_source_binding["exact_replay_claimed"],
                "cross_version_comparability_claimed": d067_source_binding[
                    "cross_version_comparability_claimed"
                ],
                "state_max_abs_physical_limit": (HISTORICAL_STATE_ABSOLUTE_LIMIT),
                "state_relative_l2_limit": HISTORICAL_STATE_RELATIVE_L2_LIMIT,
                "large_commutator_drift_to_baseline_ratio_limit": (
                    HISTORICAL_LARGE_BAND_DRIFT_RATIO_LIMIT
                ),
                "increment_drift_denominator": "frozen residual component scale",
            },
        },
        "checks": checks,
        "mechanism_selection": mechanism,
        "hook_equivalence": hook_equivalence,
        "d067_binding": {
            "summary_sha256": sha256_file(args.d067_summary),
            "schema": d067_summary["schema"],
            "status": d067_summary["status"],
            **d067_source_binding,
        },
        "provenance": provenance,
        "runtime": runtime_environment(device),
        "git": git_state(),
        "row_counts": {name: len(rows) for name, rows in outputs.items()},
        "elapsed_seconds": perf_counter() - started,
        "output_hashes": output_hashes,
    }
    write_json(args.output_dir / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    summary = run(parse_args(argv))
    print(f"D070 status={summary['status']}", flush=True)
    return 0 if summary["status"] in {"complete", "smoke_complete"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
