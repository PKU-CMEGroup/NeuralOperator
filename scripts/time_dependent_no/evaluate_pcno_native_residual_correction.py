#!/usr/bin/env python3
"""D074-A/D075: select and evaluate native persistent corrections."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no.evaluate_pcno_defect_corrections import (
    _predict,
    _run_hook_equivalence,
    _safe_admissibility_summary,
    _weighted_rms,
)
from scripts.time_dependent_no.evaluate_pcno_node_type_interventions import (
    CaseData,
    _bump_replay_gate,
    _load_bump_cases,
    _load_dynamic_cases,
    _loaded_project_source_hashes,
    _region_masks,
    _verify_common_provenance,
)
from utility.time_dependent_no.pcno_artifacts import (
    atomic_write_json_with_paths as write_json,
)
from utility.time_dependent_no.pcno_artifacts import (
    digest_array,
    git_head,
    git_state,
    jsonable_args,
    runtime_environment,
    sha256_file,
    write_csv_with_paths,
)
from utility.time_dependent_no.pcno_defect_corrections import (
    fit_error_coefficient_sequence,
    mean_calibration_coefficients,
    physical_cosine_basis,
    reconstruct_coefficient_sequence,
    weighted_subspace_decomposition,
)
from utility.time_dependent_no.pcno_euler2d import PCNOEuler2DShardStore
from utility.time_dependent_no.pcno_residual_structure import (
    cosine,
    recurrence_diagnostics,
    sequence_diagnostics,
    weighted_inner,
)
from utility.time_dependent_no.pcno_resolution_transfer import (
    build_resolution_checkpoint_model,
    build_resolution_geometry,
    load_resolution_checkpoint,
    node_types_for_protocol,
)
from utility.time_dependent_no.pcno_ripple_diagnostics import node_highpass_field
from utility.time_dependent_no.pcno_runtime import (
    build_checkpoint_model,
    load_checkpoint_payload,
    select_device,
)
from utility.time_dependent_no.shock_vortex_fv import ShockVortexFVConfig

SCHEMA = "pcno_native_residual_correction_diagnostic_v1"
RANKS = (2, 4, 8, 16)
GAINS = (0.25, 0.5, 0.75, 1.0)
D075_GAINS = (0.125, 0.25, 0.5, 1.0)
EXPERIMENT_CONTRACTS = ("d074_raw", "d075_integral_neutral")
CORRECTION_POLICIES = (
    "raw",
    "energy_integral_neutral",
    "all_integrals_neutral",
)
CORRECTION_POLICY_COMPLEXITY = {
    "raw": 0,
    "energy_integral_neutral": 1,
    "all_integrals_neutral": 2,
}
INTEGRAL_CLOSURE_LIMIT = 1.0e-12
COMPONENTS = ("rho", "rho_u", "rho_v", "energy")
DYNAMIC_CALIBRATION_CASES = tuple(
    f"sv_e{energy:02d}_{offset}"
    for energy in (1, 2, 3, 4, 5, 7, 8, 9, 10)
    for offset in ("y00", "y08")
)
DYNAMIC_EVALUATION_CASES = tuple(
    f"sv_e{energy:02d}_{offset}" for energy in (0, 6, 11) for offset in ("y00", "y08")
)
BUMP_CALIBRATION_CASES = tuple(
    str(value)
    for value in (
        7,
        16,
        18,
        23,
        47,
        54,
        60,
        72,
        82,
        101,
        103,
        112,
        120,
        126,
        141,
        145,
        150,
        188,
        190,
        211,
        227,
        233,
        235,
        251,
        287,
        296,
    )
)
BUMP_EVALUATION_CASES = ("128", "172", "58", "187")
REQUIRED_ARTIFACTS = {
    "dynamic_fv": {
        "checkpoint": "95e6c180a3298c4b38662d8c6cb77301c0e7fd0286e9a53571bddc487b8379c9",
        "normalization": "4c931c813d318f9a3012814c9803cf85fbb30ce4c68739535047b2aff6f0faf4",
        "split": "1be17494eaac3902159763a9e9a6c562d39c31957739899f50c28e37b48a921d",
        "data_manifest": "f8d228ae3c6e08fe6697f37df21476fe621abc354d0b0a6eed2a623ed2b9f96c",
        "family_manifest": "2c0d0c516d38dc826c23edda76e1a0ef668ff4692caf72f1bd710150629c7dc8",
    },
    "bump": {
        "checkpoint": "2bb5ee3ca831a6ffc498f01e309ae7ea957b2df0f411023fb3812919a6732964",
        "normalization": "717a948a1f219af9eabd6aafeacbdb8e2d00a362133e6a2b6b4408e60b71b5af",
        "split": "ba648aa0bf404f61f5f8ceabf2be963efda9935ca325908435c3c52bad88519f",
        "data_manifest": "5d5373fdcc682544bf330fba6d54fe65509dabe936baee7443c71a8c0c8d9fa7",
        "family_manifest": None,
    },
}
RATIO_DENOMINATOR_FLOOR = 1.0e-8
CALIBRATION_NO_HARM_LIMIT = 1.0
CONTROL_NO_HARM_LIMIT = 1.05
FROZEN_SHOCK_QUANTILE = 0.9
NODE_TYPE_MEANINGS = {
    "dynamic_fv": {
        0: "interior",
        1: "y_symmetry_contact",
        2: "x_extrapolation_contact",
        3: "x_and_y_contact",
    },
    "bump": {0: "normal", 1: "wall", 2: "outflow", 3: "inflow"},
}
DYNAMIC_NATIVE_RESOLUTION = (250, 100)
DYNAMIC_NATIVE_ARRAY_SHA256 = {
    "nodes": "037849fa9a4262e8854ad9b77743a0ea88a68c2e3e2d1afde910e62c2c74d55f",
    "directed_edges": "6d607018c23c02d41e2aad35fb9be91b6d3d1ff3d768cd48e945a105df1ca22c",
    "physical_measures": "dbac18bd60a64ce0dc58027a4e5ee1b3e53a68e606c06182103f6b79fe93bc46",
    "normalized_weights": "c78af1b04fa606ff4bf45bc5f702227abe66befd781b1416b67c64928ea26e8d",
    "node_rhos": "32df19385e99207360c66ac3f4d9245011126f0f0370ad3729d5dd0d45594817",
    "node_type": "7a1e2e2dc6d34bb4f92f25a71a339e52699a45229254be4bac653048af16f4c9",
}
STRUCTURE_CONTROL_REGIONS = {
    "dynamic_fv": (
        "shock",
        "vortex",
        "boundary_distance_le_0.05",
    ),
    "bump": ("shock", "boundary_distance_le_0.10"),
}


def _global_control_prefix(family: str) -> str:
    if family == "dynamic_fv":
        return "physical_volume_integral_state_scale"
    if family == "bump":
        return "proxy_weighted_mean_state_scale"
    raise ValueError(f"unsupported D074 family: {family}")


def expected_control_keys(family: str) -> tuple[str, ...]:
    """Return the exact family-local primary no-harm control inventory."""

    prefix = _global_control_prefix(family)
    keys = [
        *(f"endpoint_state__{region}" for region in STRUCTURE_CONTROL_REGIONS[family]),
        "endpoint_smooth_highpass",
        *(f"component_residual_rms__{name}" for name in COMPONENTS),
        *(f"{prefix}_rms__{name}" for name in COMPONENTS),
        *(f"{prefix}_final_abs__{name}" for name in COMPONENTS),
    ]
    return tuple(sorted(keys))


@dataclass(frozen=True, order=True)
class Candidate:
    """One unique rank/gain correction candidate."""

    key: str
    rank: int
    gain: float
    correction_policy: str = "raw"

    @property
    def is_zero(self) -> bool:
        return self.rank == 0 and self.gain == 0.0


@dataclass
class RolloutResult:
    """One case/arm recurrent payload before case-level aggregation."""

    candidate: Candidate
    complete: bool
    valid_length: int
    states: np.ndarray
    base_defects: np.ndarray
    corrections: np.ndarray
    defects: np.ndarray
    truth_increments: np.ndarray
    admissibility_rows: list[dict[str, Any]]
    summary: dict[str, Any]


@dataclass(frozen=True)
class SelectorRolloutSummary:
    """Compact calibration payload retained until candidate selection."""

    complete: bool
    summary: dict[str, Any]


def _selector_rollout_summary(result: RolloutResult) -> SelectorRolloutSummary:
    """Drop all recurrent fields after extracting selector-required scalars."""

    return SelectorRolloutSummary(
        complete=bool(result.complete),
        summary={
            "final_state_error": result.summary.get("final_state_error"),
            "residual_rms": result.summary.get("residual_rms"),
            "controls": dict(result.summary.get("controls", {})),
            "correction_integral_audit": dict(
                result.summary.get("correction_integral_audit", {})
            ),
        },
    )


def _gain_key(gain: float) -> str:
    return str(float(gain)).replace(".", "p")


def candidate_inventory(
    *,
    smoke: bool = False,
    experiment_contract: str = "d074_raw",
) -> tuple[Candidate, ...]:
    """Return the canonical zero plus unique positive-gain candidates."""

    zero = Candidate(key="zero", rank=0, gain=0.0)
    if experiment_contract == "d074_raw":
        if smoke:
            return (zero, Candidate(key="rank8_gain1", rank=8, gain=1.0))
        positive = tuple(
            Candidate(
                key=f"rank{rank}_gain{str(gain).replace('.', 'p')}",
                rank=rank,
                gain=gain,
            )
            for rank in RANKS
            for gain in GAINS
        )
    elif experiment_contract == "d075_integral_neutral":
        active_gains = (0.5,) if smoke else D075_GAINS
        positive = tuple(
            Candidate(
                key=f"rank8_gain{_gain_key(gain)}_{policy}",
                rank=8,
                gain=gain,
                correction_policy=policy,
            )
            for policy in CORRECTION_POLICIES
            for gain in active_gains
        )
    else:
        raise ValueError(f"unsupported experiment contract: {experiment_contract}")
    candidates = (zero, *positive)
    if len({candidate.key for candidate in candidates}) != len(candidates):
        raise AssertionError("candidate keys are not unique")
    return candidates


def _family_cases(family: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if family == "dynamic_fv":
        return DYNAMIC_CALIBRATION_CASES, DYNAMIC_EVALUATION_CASES
    if family == "bump":
        return BUMP_CALIBRATION_CASES, BUMP_EVALUATION_CASES
    raise ValueError(f"unsupported D074 family: {family}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-contract",
        choices=EXPERIMENT_CONTRACTS,
        default="d074_raw",
    )
    parser.add_argument("--family", choices=("dynamic_fv", "bump"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization-json", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--amp", choices=("none",), default="none")
    parser.add_argument("--shock-quantile", type=float, default=0.9)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-normalization-sha256", required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--expected-data-manifest-digest", required=True)
    parser.add_argument("--expected-source-base-git-head", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument(
        "--expected-source", action="append", default=[], metavar="PATH=SHA256"
    )
    parser.add_argument("--family-root", type=Path)
    parser.add_argument("--multires-reference-root", type=Path)
    parser.add_argument("--expected-family-manifest-sha256")
    parser.add_argument("--bump-replay-root", type=Path)
    parser.add_argument("--bump-replay-cases", nargs="*", default=())
    parser.add_argument("--bump-replay-calls", type=int, default=20)
    parser.add_argument("--bump-replay-absolute-limit", type=float)
    parser.add_argument("--bump-replay-relative-limit", type=float)
    parser.add_argument("--visualization-cases", nargs="*", default=())
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.shock_quantile != FROZEN_SHOCK_QUANTILE:
        raise ValueError("D074 freezes --shock-quantile=0.9")
    source_head = args.expected_source_base_git_head.lower()
    if len(source_head) != 40 or any(
        character not in "0123456789abcdef" for character in source_head
    ):
        raise ValueError("D074 source base must be one 40-digit hexadecimal commit")
    if args.smoke and args.family != "dynamic_fv":
        raise ValueError("the registered D074 smoke is dynamic-FV only")
    if args.experiment_contract == "d075_integral_neutral" and args.family != "dynamic_fv":
        raise ValueError("D075 is dynamic-FV only")
    contract = REQUIRED_ARTIFACTS[args.family]
    observed = {
        "checkpoint": args.expected_checkpoint_sha256.lower(),
        "normalization": args.expected_normalization_sha256.lower(),
        "split": args.expected_split_sha256.lower(),
        "data_manifest": args.expected_data_manifest_digest.lower(),
        "family_manifest": (
            None
            if args.expected_family_manifest_sha256 is None
            else args.expected_family_manifest_sha256.lower()
        ),
    }
    if observed != contract:
        raise ValueError(f"D074 family artifact contract changed: {observed}")
    if args.family == "dynamic_fv":
        if args.family_root is None or args.multires_reference_root is None:
            raise ValueError("dynamic D074 requires both common-source roots")
        if args.bump_replay_root is not None or args.bump_replay_cases:
            raise ValueError("bump replay arguments are invalid for dynamic D074")
    else:
        if args.family_root is not None or args.multires_reference_root is not None:
            raise ValueError("dynamic roots are invalid for bump D074")
        if args.bump_replay_root is None or not args.bump_replay_cases:
            raise ValueError("bump D074 requires the D041 replay gate")
        if (
            args.bump_replay_absolute_limit is None
            or args.bump_replay_relative_limit is None
        ):
            raise ValueError("bump D074 requires both replay limits")
    args.rollout_calls = 2 if args.smoke else 30 if args.family == "dynamic_fv" else 20
    args.training_resolution = "250x100"
    args.resolutions = ["250x100"]
    return args


def _domain_bounds(family: str) -> tuple[float, float, float, float]:
    return (0.0, 2.0, 0.0, 1.0) if family == "dynamic_fv" else (0.0, 6.0, 0.0, 2.0)


@lru_cache(maxsize=1)
def _dynamic_native_geometry_contract() -> tuple[Any, Any, np.ndarray, dict[str, str]]:
    """Regenerate and bind the exact dynamic 250x100 model geometry once."""

    config, geometry = build_resolution_geometry(
        ShockVortexFVConfig(), DYNAMIC_NATIVE_RESOLUTION
    )
    node_type = node_types_for_protocol(
        geometry,
        config,
        "physical",
        training_resolution=DYNAMIC_NATIVE_RESOLUTION,
    )
    physical_measures = np.asarray(geometry.node_measures).reshape(-1)
    normalized_weights = np.asarray(geometry.node_weights).reshape(-1)
    node_rhos = np.asarray(geometry.node_rhos).reshape(-1)
    observed_sha256 = {
        "nodes": digest_array(geometry.nodes),
        "directed_edges": digest_array(geometry.directed_edges),
        "physical_measures": digest_array(physical_measures),
        "normalized_weights": digest_array(normalized_weights),
        "node_rhos": digest_array(node_rhos),
        "node_type": digest_array(node_type),
    }
    if observed_sha256 != DYNAMIC_NATIVE_ARRAY_SHA256:
        raise RuntimeError(
            "regenerated dynamic native geometry differs from the frozen D074 contract"
        )
    node_count = DYNAMIC_NATIVE_RESOLUTION[0] * DYNAMIC_NATIVE_RESOLUTION[1]
    if not np.array_equal(
        geometry.mesh_cell_to_graph_node, np.arange(node_count, dtype=np.int64)
    ):
        raise RuntimeError("dynamic native mesh-to-graph ordering is not identity")
    if not np.array_equal(
        np.bincount(node_type, minlength=4), np.asarray([24304, 496, 196, 4])
    ):
        raise RuntimeError("dynamic native node-type inventory changed")
    if not np.array_equal(physical_measures, np.full(node_count, 8.0e-5)):
        raise RuntimeError("dynamic native physical cell measures changed")
    if float(physical_measures.sum()) != 2.0:
        raise RuntimeError("dynamic native physical volume changed")
    return config, geometry, node_type, observed_sha256


def _validate_dynamic_native_geometry(case: CaseData) -> dict[str, Any]:
    """Require exact regenerated arrays in both CaseData and the PCNO sample."""

    config, geometry, node_type, geometry_sha256 = _dynamic_native_geometry_contract()
    physical_measures = np.asarray(geometry.node_measures).reshape(-1)
    if not np.array_equal(case.nodes, geometry.nodes):
        raise ValueError(
            "dynamic nodes differ from canonical regenerated 250x100 nodes"
        )
    if not np.array_equal(case.edges, geometry.directed_edges):
        raise ValueError(
            "dynamic directed edges differ from canonical regenerated 250x100 edges"
        )
    if not np.array_equal(case.weights, physical_measures):
        raise ValueError(
            "dynamic physical cell measures differ from canonical regenerated weights"
        )
    if not np.array_equal(case.physical_node_type, node_type):
        raise ValueError(
            "dynamic node types differ from canonical regenerated contact types"
        )

    expected_sample = {
        "nodes": np.asarray(geometry.nodes, dtype=np.float32)[None, ...],
        "node_measures": np.asarray(geometry.node_measures, dtype=np.float32)[
            None, ...
        ],
        "node_weights": np.asarray(geometry.node_weights, dtype=np.float32)[None, ...],
        "node_rhos": np.asarray(geometry.node_rhos, dtype=np.float32)[None, ...],
        "directed_edges": np.asarray(geometry.directed_edges, dtype=np.int64)[
            None, ...
        ],
        "edge_gradient_weights": np.asarray(
            geometry.edge_gradient_weights, dtype=np.float32
        )[None, ...],
        "node_type": np.asarray(node_type, dtype=np.int64)[None, ...],
    }
    if not isinstance(case.sample, Mapping):
        raise TypeError("dynamic model sample is not a mapping")
    for key, expected in expected_sample.items():
        tensor = case.sample.get(key)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"dynamic model sample is missing tensor {key}")
        actual = tensor.detach().cpu().numpy()
        if not np.array_equal(actual, expected):
            raise ValueError(
                f"dynamic model sample tensor {key} differs from regenerated geometry"
            )
    node_mask = case.sample.get("node_mask")
    if not isinstance(node_mask, torch.Tensor) or not np.array_equal(
        node_mask.detach().cpu().numpy(),
        np.ones((1, geometry.nodes.shape[0], 1), dtype=np.float32),
    ):
        raise ValueError(
            "dynamic model sample node_mask differs from the native contract"
        )
    mach = case.sample.get("mach")
    if not isinstance(mach, torch.Tensor) or not np.array_equal(
        mach.detach().cpu().numpy(), np.asarray([config.shock_mach], dtype=np.float32)
    ):
        raise ValueError(
            "dynamic model sample Mach channel differs from the native contract"
        )
    return {
        "geometry_contract": "regenerated_dynamic_fv_250x100_v1",
        "num_directed_edges": int(geometry.directed_edges.shape[0]),
        "mesh_cell_to_graph_identity": True,
        "model_sample_geometry_exact": True,
        **{f"{key}_sha256": value for key, value in geometry_sha256.items()},
    }


def _validate_case_contract(case: CaseData) -> dict[str, Any]:
    """Fail closed on family-local geometry, weight, time, and type semantics."""

    if case.family not in NODE_TYPE_MEANINGS:
        raise ValueError(f"unsupported D074 family: {case.family}")
    node_count = int(case.nodes.shape[0])
    if case.nodes.shape != (node_count, 2) or node_count < 1:
        raise ValueError("case nodes must have shape [N,2]")
    if case.reference_states.ndim != 3 or case.reference_states.shape[1:] != (
        node_count,
        len(COMPONENTS),
    ):
        raise ValueError("reference trajectory does not align with case nodes")
    if case.weights.shape != (node_count,) or not np.all(np.isfinite(case.weights)):
        raise ValueError("case weights do not align with case nodes")
    if np.any(case.weights <= 0.0):
        raise ValueError("case weights must be positive")
    if case.physical_node_type.shape != (node_count,):
        raise ValueError("physical node types do not align with case nodes")
    observed_types = {int(value) for value in np.unique(case.physical_node_type)}
    allowed_types = set(NODE_TYPE_MEANINGS[case.family])
    if not observed_types <= allowed_types:
        raise ValueError(
            f"{case.family} node types violate family-local meanings: {observed_types}"
        )
    if 0 not in observed_types or not np.any(case.physical_node_type != 0):
        raise ValueError("D074 requires both type-0 fit support and excluded support")
    if case.state_scale.shape != (len(COMPONENTS),) or case.residual_scale.shape != (
        len(COMPONENTS),
    ):
        raise ValueError("state and residual scales must contain four components")
    if (
        not np.all(np.isfinite(case.state_scale))
        or not np.all(np.isfinite(case.residual_scale))
        or np.any(case.state_scale <= 0.0)
        or np.any(case.residual_scale <= 0.0)
    ):
        raise ValueError("state and residual scales must be finite and positive")
    if case.family == "dynamic_fv" and case.resolution_name != "250x100":
        raise ValueError("D074-A dynamic evaluation is frozen to 250x100")
    if case.family == "bump" and case.resolution_name != "native_graph":
        raise ValueError("D074-A bump evaluation is native-graph only")
    geometry_contract = (
        _validate_dynamic_native_geometry(case)
        if case.family == "dynamic_fv"
        else {
            "geometry_contract": "family_local_native_graph",
            "num_directed_edges": int(case.edges.shape[0]),
            "mesh_cell_to_graph_identity": None,
            "model_sample_geometry_exact": None,
            **{f"{key}_sha256": None for key in DYNAMIC_NATIVE_ARRAY_SHA256},
        }
    )
    expected_dt = 0.02 if case.family == "dynamic_fv" else 0.025
    expected_times = expected_dt * np.arange(case.reference_states.shape[0])
    if case.physical_times.shape != expected_times.shape or not np.allclose(
        case.physical_times, expected_times, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("case physical times differ from the frozen D074 clock")
    physical_cosine_basis(
        case.nodes,
        rank=1,
        domain_bounds=_domain_bounds(case.family),
    )
    counts = {
        str(value): int(np.count_nonzero(case.physical_node_type == value))
        for value in sorted(observed_types)
    }
    masses = {
        str(value): float(case.weights[case.physical_node_type == value].sum())
        for value in sorted(observed_types)
    }
    return {
        "family": case.family,
        "case_id": case.case_id,
        "resolution": case.resolution_name,
        "node_type_meanings": {
            str(key): value for key, value in NODE_TYPE_MEANINGS[case.family].items()
        },
        "node_type_counts": counts,
        "node_type_weight_masses": masses,
        "weight_semantics": (
            "physical_cell_volume"
            if case.family == "dynamic_fv"
            else "frozen_proxy_weight_nonphysical"
        ),
        "total_weight": float(case.weights.sum()),
        "time_step": expected_dt,
        "calls": int(case.reference_states.shape[0] - 1),
        **geometry_contract,
    }


def _fit_case_coefficients(
    model: torch.nn.Module,
    case: CaseData,
    ranks: Sequence[int],
) -> dict[int, np.ndarray]:
    errors = []
    for call in range(1, case.reference_states.shape[0]):
        current = case.reference_states[call - 1]
        target = case.reference_states[call]
        prediction = _predict(model, case, current)
        errors.append((prediction - current) - (target - current))
    error_sequence = np.asarray(errors, dtype=np.float64)
    interior = case.physical_node_type == 0
    result = {}
    for rank in sorted({int(value) for value in ranks}):
        basis, _ = physical_cosine_basis(
            case.nodes, rank=rank, domain_bounds=_domain_bounds(case.family)
        )
        result[rank] = fit_error_coefficient_sequence(
            error_sequence[:, interior],
            basis[interior],
            case.weights[interior],
            component_scale=case.residual_scale,
        )
    return result


def _bias_sequence(
    case: CaseData,
    coefficients: np.ndarray,
    *,
    rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    basis, _ = physical_cosine_basis(
        case.nodes, rank=rank, domain_bounds=_domain_bounds(case.family)
    )
    bias = reconstruct_coefficient_sequence(
        coefficients,
        basis,
        component_scale=case.residual_scale,
    )
    bias[:, case.physical_node_type != 0] = 0.0
    return basis, bias


def _neutralized_component_indices(correction_policy: str) -> tuple[int, ...]:
    if correction_policy == "raw":
        return ()
    if correction_policy == "energy_integral_neutral":
        return (COMPONENTS.index("energy"),)
    if correction_policy == "all_integrals_neutral":
        return tuple(range(len(COMPONENTS)))
    raise ValueError(f"unsupported correction policy: {correction_policy}")


def _apply_integral_policy(
    sequence: np.ndarray,
    case: CaseData,
    *,
    correction_policy: str,
) -> np.ndarray:
    """Remove selected physical-volume means on type-0 support."""

    values = np.asarray(sequence, dtype=np.float64)
    if (
        values.ndim != 3
        or values.shape[1:] != (case.nodes.shape[0], len(COMPONENTS))
        or not np.isfinite(values).all()
    ):
        raise ValueError("correction sequence must be finite [calls,N,4]")
    components = _neutralized_component_indices(correction_policy)
    if not components:
        return values
    if case.family != "dynamic_fv":
        raise ValueError("physical-volume integral policies are dynamic-FV only")
    interior = np.asarray(case.physical_node_type == 0, dtype=bool)
    if not np.any(interior):
        raise ValueError("integral policy requires nonempty type-0 support")
    if np.max(np.abs(values[:, ~interior])) > 1.0e-12:
        raise ValueError("integral policy input is nonzero outside type-0 support")
    weights = np.asarray(case.weights, dtype=np.float64)
    interior_mass = float(weights[interior].sum())
    if not np.isfinite(interior_mass) or interior_mass <= 0.0:
        raise ValueError("integral policy has invalid type-0 physical volume")
    means = (
        np.einsum(
            "n,tnc->tc",
            weights[interior],
            values[:, interior],
            optimize=True,
        )
        / interior_mass
    )
    projected = np.array(values, copy=True)
    interior_indices = np.flatnonzero(interior)
    for component in components:
        projected[:, interior_indices, component] -= means[:, component, None]
    projected[:, ~interior] = 0.0
    return projected


def _candidate_bias_sequence(
    case: CaseData,
    coefficients: np.ndarray,
    candidate: Candidate,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return basis, policy-adjusted physical bias, and adjusted coefficients."""

    basis, raw_bias = _bias_sequence(case, coefficients, rank=candidate.rank)
    components = _neutralized_component_indices(candidate.correction_policy)
    adjusted_coefficients = np.asarray(coefficients, dtype=np.float64)
    if not components:
        return basis, raw_bias, adjusted_coefficients
    projected = _apply_integral_policy(
        raw_bias,
        case,
        correction_policy=candidate.correction_policy,
    )
    adjusted_coefficients = np.array(adjusted_coefficients, copy=True)
    interior = case.physical_node_type == 0
    interior_mass = float(case.weights[interior].sum())
    raw_means = (
        np.einsum(
            "n,tnc->tc",
            case.weights[interior],
            raw_bias[:, interior],
            optimize=True,
        )
        / interior_mass
    )
    for component in components:
        adjusted_coefficients[:, 0, component] -= (
            raw_means[:, component] / case.residual_scale[component]
        )
    reconstructed = reconstruct_coefficient_sequence(
        adjusted_coefficients,
        basis,
        component_scale=case.residual_scale,
    )
    reconstructed[:, ~interior] = 0.0
    if np.max(np.abs(reconstructed - projected)) > 1.0e-10:
        raise ValueError("integral-neutral coefficient reconstruction failed")
    repeated = _apply_integral_policy(
        projected,
        case,
        correction_policy=candidate.correction_policy,
    )
    if np.max(np.abs(repeated - projected)) > INTEGRAL_CLOSURE_LIMIT:
        raise ValueError("integral-neutral projection is not idempotent")
    if candidate.correction_policy == "energy_integral_neutral" and not np.array_equal(
        projected[:, :, :3], raw_bias[:, :, :3]
    ):
        raise ValueError("energy-only policy changed another conservative component")
    return basis, projected, adjusted_coefficients


def _correction_integral_audit(
    correction_sequence: np.ndarray,
    case: CaseData,
    candidate: Candidate,
) -> dict[str, Any]:
    values = np.asarray(correction_sequence, dtype=np.float64)
    required = _neutralized_component_indices(candidate.correction_policy)
    if values.ndim != 3 or values.shape[0] < 1:
        maxima = {name: None for name in COMPONENTS}
        required_maximum = None if required else 0.0
        passed = not required
    else:
        normalized_mean = (
            np.einsum("n,tnc->tc", case.weights, values, optimize=True)
            / float(case.weights.sum())
            / case.residual_scale[None, :]
        )
        maxima = {
            name: float(np.max(np.abs(normalized_mean[:, component])))
            for component, name in enumerate(COMPONENTS)
        }
        required_maximum = (
            max(maxima[COMPONENTS[component]] for component in required)
            if required
            else 0.0
        )
        passed = bool(
            not required
            or (
                np.isfinite(required_maximum)
                and required_maximum <= INTEGRAL_CLOSURE_LIMIT
            )
        )
    return {
        "correction_policy": candidate.correction_policy,
        "neutralized_components": [COMPONENTS[index] for index in required],
        "maximum_normalized_physical_volume_mean_by_component": maxima,
        "maximum_required_integral_closure": required_maximum,
        "passed": passed,
    }


def _correction_integral_rows(
    case: CaseData,
    result: RolloutResult,
    *,
    arm: str,
    policy_adjustment: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    corrections = np.asarray(result.corrections, dtype=np.float64)
    if corrections.shape != result.defects.shape:
        raise ValueError("correction integral rows require one correction per defect")
    if policy_adjustment is None:
        adjustment = np.zeros_like(corrections)
    else:
        adjustment = np.asarray(policy_adjustment, dtype=np.float64)
        if adjustment.shape != corrections.shape or not np.isfinite(adjustment).all():
            raise ValueError("policy adjustment does not match the correction sequence")
    neutralized = set(
        _neutralized_component_indices(result.candidate.correction_policy)
    )
    total_volume = float(case.weights.sum())
    rows = []
    for call, (correction, removed) in enumerate(
        zip(corrections, adjustment, strict=True),
        start=1,
    ):
        raw_integral = np.einsum(
            "n,nc->c", case.weights, correction, optimize=True
        )
        normalized_mean = raw_integral / total_volume / case.residual_scale
        correction_rms = np.sqrt(
            np.einsum(
                "n,nc->c",
                case.weights,
                np.square(correction / case.residual_scale[None, :]),
                optimize=True,
            )
            / total_volume
        )
        adjustment_rms = np.sqrt(
            np.einsum(
                "n,nc->c",
                case.weights,
                np.square(removed / case.residual_scale[None, :]),
                optimize=True,
            )
            / total_volume
        )
        for component, name in enumerate(COMPONENTS):
            required = component in neutralized
            rows.append(
                {
                    "family": case.family,
                    "case_id": case.case_id,
                    "resolution": case.resolution_name,
                    "arm": arm,
                    "candidate": result.candidate.key,
                    "correction_policy": result.candidate.correction_policy,
                    "call": call,
                    "physical_time": float(case.physical_times[call]),
                    "component": name,
                    "raw_physical_volume_integral": float(raw_integral[component]),
                    "residual_scaled_physical_volume_mean": float(
                        normalized_mean[component]
                    ),
                    "correction_residual_scaled_rms": float(
                        correction_rms[component]
                    ),
                    "policy_adjustment_residual_scaled_rms": float(
                        adjustment_rms[component]
                    ),
                    "integral_neutralization_required": required,
                    "integral_closure_pass": bool(
                        not required
                        or abs(float(normalized_mean[component]))
                        <= INTEGRAL_CLOSURE_LIMIT
                    ),
                }
            )
    return rows


def _sequence_component_rms(
    sequence: np.ndarray,
    case: CaseData,
) -> dict[str, float]:
    result = {}
    for component, name in enumerate(COMPONENTS):
        values = [
            _weighted_rms(
                frame[:, component : component + 1],
                weights=case.weights,
                scale=case.residual_scale[component : component + 1],
            )
            for frame in sequence
        ]
        if any(value is None for value in values):
            raise AssertionError("full-domain component RMS is undefined")
        result[f"component_residual_rms__{name}"] = float(
            np.sqrt(np.mean(np.square(values)))
        )
    return result


def _global_quantity_metrics(
    state_errors: np.ndarray,
    case: CaseData,
) -> tuple[dict[str, float], dict[str, Any]]:
    weighted_sum = np.einsum("n,tnc->tc", case.weights, state_errors, optimize=True)
    if case.family == "dynamic_fv":
        raw_quantity = weighted_sum
        semantics = "signed physical-volume integral"
    elif case.family == "bump":
        raw_quantity = weighted_sum / float(case.weights.sum())
        semantics = "signed proxy-weighted mean; nonphysical"
    else:
        raise ValueError(f"unsupported D074 family: {case.family}")
    scaled_quantity = raw_quantity / case.state_scale[None, :]
    prefix = _global_control_prefix(case.family)
    controls = {}
    budgets: dict[str, Any] = {"semantics": semantics}
    for component, name in enumerate(COMPONENTS):
        controls[f"{prefix}_rms__{name}"] = float(
            np.sqrt(np.mean(np.square(scaled_quantity[:, component])))
        )
        controls[f"{prefix}_final_abs__{name}"] = float(
            abs(scaled_quantity[-1, component])
        )
        budgets[f"raw_rms__{name}"] = float(
            np.sqrt(np.mean(np.square(raw_quantity[:, component])))
        )
        budgets[f"raw_final_signed__{name}"] = float(raw_quantity[-1, component])
        budgets[f"state_scaled_final_signed__{name}"] = float(
            scaled_quantity[-1, component]
        )
    return controls, budgets


def _summarize_rollout(
    case: CaseData,
    *,
    states: np.ndarray,
    defects: np.ndarray,
    complete: bool,
    shock_quantile: float,
) -> dict[str, Any]:
    steps = defects.shape[0]
    if steps < 1 or states.shape[0] != steps + 1:
        raise ValueError("rollout states and defects do not align")
    target_states = case.reference_states[1 : steps + 1]
    state_errors = states[1:] - target_states
    final_target = target_states[-1]
    final_error = state_errors[-1]
    residual_values = [
        _weighted_rms(
            frame,
            weights=case.weights,
            scale=case.residual_scale,
        )
        for frame in defects
    ]
    if any(value is None for value in residual_values):
        raise AssertionError("full-domain residual RMS is undefined")
    final_state_error = _weighted_rms(
        final_error,
        weights=case.weights,
        scale=case.state_scale,
    )
    if final_state_error is None:
        raise AssertionError("full-domain state RMS is undefined")
    controls: dict[str, float] = {}
    masks = _region_masks(case, final_target, shock_quantile=shock_quantile)
    required_regions = (*STRUCTURE_CONTROL_REGIONS[case.family], "smooth")
    mask_inventory = {}
    for region in required_regions:
        if region not in masks:
            raise ValueError(f"required {case.family} mask is missing: {region}")
        selected = np.asarray(masks[region], dtype=bool)
        if selected.shape != (case.nodes.shape[0],) or not np.any(selected):
            raise ValueError(f"required {case.family} mask is empty: {region}")
        mass = float(case.weights[selected].sum())
        if not np.isfinite(mass) or mass <= 0.0:
            raise ValueError(f"required {case.family} mask has no weight: {region}")
        mask_inventory[region] = {
            "node_count": int(selected.sum()),
            "weight_mass": mass,
        }
    for region in STRUCTURE_CONTROL_REGIONS[case.family]:
        selected = np.asarray(masks[region], dtype=bool)
        value = _weighted_rms(
            final_error,
            weights=case.weights,
            scale=case.state_scale,
            mask=selected,
        )
        if value is None:
            raise AssertionError(f"required state control is undefined: {region}")
        controls[f"endpoint_state__{region}"] = value
    scaled_highpass = node_highpass_field(
        final_error / case.state_scale[None, :], case.edges
    )
    smooth_highpass = _weighted_rms(
        scaled_highpass,
        weights=case.weights,
        scale=np.ones(final_error.shape[1]),
        mask=masks["smooth"],
    )
    if smooth_highpass is None:
        raise AssertionError("required smooth high-pass control is undefined")
    controls["endpoint_smooth_highpass"] = smooth_highpass
    controls.update(_sequence_component_rms(defects, case))
    global_controls, global_budgets = _global_quantity_metrics(state_errors, case)
    controls.update(global_controls)
    expected_controls = set(expected_control_keys(case.family))
    if set(controls) != expected_controls:
        raise AssertionError(
            "primary control inventory differs from the frozen family contract"
        )
    return {
        "family": case.family,
        "complete": bool(complete),
        "valid_length": int(steps if complete else states.shape[0] - 1),
        "final_state_error": float(final_state_error),
        "residual_rms": float(np.sqrt(np.mean(np.square(residual_values)))),
        "controls": controls,
        "mask_inventory": mask_inventory,
        "global_budgets": global_budgets,
    }


def _rollout_candidate(
    model: torch.nn.Module,
    case: CaseData,
    candidate: Candidate,
    *,
    bias_sequence: np.ndarray | None,
    shock_quantile: float,
) -> RolloutResult:
    calls = case.reference_states.shape[0] - 1
    if candidate.is_zero:
        if bias_sequence is not None:
            raise ValueError("zero candidate must not receive a bias sequence")
    elif (
        bias_sequence is None or bias_sequence.shape != case.reference_states[1:].shape
    ):
        raise ValueError("nonzero candidate bias does not match the rollout")
    current = np.array(case.reference_states[0], copy=True)
    states = [current.copy()]
    base_defects = []
    corrections = []
    defects = []
    truth_increments = []
    admissibility_rows = []
    valid_length = 0
    for call in range(1, calls + 1):
        base_prediction = _predict(model, case, current)
        base_summary = _safe_admissibility_summary(base_prediction, gamma=case.gamma)
        if not base_summary["finite"]:
            admissibility_rows.append(
                {
                    "call": call,
                    "accepted": False,
                    "failure_stage": "base_prediction_nonfinite",
                    **base_summary,
                }
            )
            break
        correction = (
            np.zeros_like(base_prediction)
            if candidate.is_zero
            else -candidate.gain * bias_sequence[call - 1]
        )
        prediction = base_prediction + correction
        truth_increment = case.reference_states[call] - case.reference_states[call - 1]
        base_defect = (base_prediction - current) - truth_increment
        defect = base_defect + correction
        admissibility = _safe_admissibility_summary(prediction, gamma=case.gamma)
        accepted = bool(admissibility["finite"] and admissibility["admissible"])
        admissibility_rows.append(
            {
                "call": call,
                "accepted": accepted,
                "failure_stage": None if accepted else "proposal_inadmissible",
                **admissibility,
            }
        )
        if not admissibility["finite"]:
            break
        states.append(np.asarray(prediction, dtype=np.float64))
        base_defects.append(np.asarray(base_defect, dtype=np.float64))
        corrections.append(np.asarray(correction, dtype=np.float64))
        defects.append(np.asarray(defect, dtype=np.float64))
        truth_increments.append(np.asarray(truth_increment, dtype=np.float64))
        if not accepted:
            break
        current = prediction
        valid_length = call
    state_array = np.asarray(states, dtype=np.float64)
    base_array = np.asarray(base_defects, dtype=np.float64)
    correction_array = np.asarray(corrections, dtype=np.float64)
    defect_array = np.asarray(defects, dtype=np.float64)
    truth_array = np.asarray(truth_increments, dtype=np.float64)
    complete = bool(valid_length == calls)
    if defect_array.shape[0] < 1:
        summary = {
            "complete": False,
            "valid_length": valid_length,
            "final_state_error": None,
            "residual_rms": None,
            "controls": {},
        }
    else:
        summary = _summarize_rollout(
            case,
            states=state_array,
            defects=defect_array,
            complete=complete,
            shock_quantile=shock_quantile,
        )
        summary["valid_length"] = valid_length
    summary["correction_integral_audit"] = _correction_integral_audit(
        correction_array,
        case,
        candidate,
    )
    return RolloutResult(
        candidate=candidate,
        complete=complete,
        valid_length=valid_length,
        states=state_array,
        base_defects=base_array,
        corrections=correction_array,
        defects=defect_array,
        truth_increments=truth_array,
        admissibility_rows=admissibility_rows,
        summary=summary,
    )


def _required_ratio(numerator: float | None, denominator: float | None) -> float:
    if denominator is None or not np.isfinite(denominator):
        raise ValueError("required baseline denominator is missing or nonfinite")
    if denominator <= RATIO_DENOMINATOR_FLOOR:
        raise ValueError("required baseline denominator is numerically unresolved")
    if numerator is None or not np.isfinite(numerator):
        raise ValueError("required candidate numerator is missing or nonfinite")
    return float(numerator / denominator)


def select_candidate(
    candidates: Sequence[Candidate],
    case_ids: Sequence[str],
    rollouts: Mapping[str, Mapping[str, RolloutResult | SelectorRolloutSummary]],
    *,
    family: str | None = None,
) -> tuple[
    Candidate,
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Apply the complete-case lexicographic D074/D075 calibration selector."""

    candidate_keys = tuple(candidate.key for candidate in candidates)
    expected_cases = tuple(sorted(str(value) for value in case_ids))
    if len(set(candidate_keys)) != len(candidate_keys):
        raise ValueError("candidate inventory contains duplicate keys")
    if "zero" not in candidate_keys:
        raise ValueError("candidate inventory lacks the canonical zero arm")
    if set(rollouts) != set(candidate_keys):
        raise ValueError("rollout candidate inventory differs from selector contract")
    zero_by_case = rollouts["zero"]
    if set(zero_by_case) != set(expected_cases):
        raise ValueError("zero-arm case inventory differs from selector contract")

    rows = []
    summaries = []
    metric_rows = []
    eligible_candidates = []
    for candidate in candidates:
        if candidate.correction_policy not in CORRECTION_POLICY_COMPLEXITY:
            raise ValueError(
                f"unsupported candidate correction policy: "
                f"{candidate.correction_policy}"
            )
        by_case = rollouts[candidate.key]
        if set(by_case) != set(expected_cases):
            raise ValueError(f"incomplete case inventory for {candidate.key}")
        endpoint_ratios = []
        residual_ratios = []
        candidate_eligible = True
        for case_id in expected_cases:
            baseline = zero_by_case[case_id]
            result = by_case[case_id]
            if not baseline.complete:
                raise ValueError(f"baseline calibration rollout failed for {case_id}")
            baseline_controls = baseline.summary.get("controls", {})
            required_controls = (
                set(expected_control_keys(family))
                if family is not None
                else set(baseline_controls)
            )
            if set(baseline_controls) != required_controls:
                raise ValueError("baseline control inventory differs from contract")
            baseline_endpoint = baseline.summary.get("final_state_error")
            baseline_residual = baseline.summary.get("residual_rms")
            _required_ratio(baseline_endpoint, baseline_endpoint)
            _required_ratio(baseline_residual, baseline_residual)
            for value in baseline_controls.values():
                _required_ratio(value, value)
            complete = bool(result.complete)
            candidate_controls = result.summary.get("controls", {})
            if complete and set(candidate_controls) != required_controls:
                raise ValueError("candidate control inventory differs from contract")
            candidate_endpoint = (
                result.summary.get("final_state_error") if complete else None
            )
            candidate_residual = (
                result.summary.get("residual_rms") if complete else None
            )
            endpoint_ratio = (
                _required_ratio(candidate_endpoint, baseline_endpoint)
                if complete
                else None
            )
            residual_ratio = (
                _required_ratio(candidate_residual, baseline_residual)
                if complete
                else None
            )
            control_ratios = (
                {
                    name: _required_ratio(
                        candidate_controls[name], baseline_controls[name]
                    )
                    for name in sorted(required_controls)
                }
                if complete
                else {}
            )
            maximum_control_ratio = (
                max(control_ratios.values()) if control_ratios else None
            )
            correction_audit = result.summary.get("correction_integral_audit", {})
            correction_contract_pass = bool(
                candidate.correction_policy == "raw"
                or correction_audit.get("passed") is True
            )
            maximum_integral_closure = correction_audit.get(
                "maximum_required_integral_closure"
            )
            row_eligible = bool(
                complete
                and endpoint_ratio is not None
                and residual_ratio is not None
                and maximum_control_ratio is not None
                and endpoint_ratio <= CALIBRATION_NO_HARM_LIMIT
                and residual_ratio <= CALIBRATION_NO_HARM_LIMIT
                and maximum_control_ratio <= CONTROL_NO_HARM_LIMIT
                and correction_contract_pass
            )
            candidate_eligible = candidate_eligible and row_eligible
            if endpoint_ratio is not None:
                endpoint_ratios.append(endpoint_ratio)
            if residual_ratio is not None:
                residual_ratios.append(residual_ratio)
            rows.append(
                {
                    "candidate": candidate.key,
                    "rank": candidate.rank,
                    "gain": candidate.gain,
                    "correction_policy": candidate.correction_policy,
                    "case_id": case_id,
                    "complete": complete,
                    "endpoint_state_numerator": candidate_endpoint,
                    "endpoint_state_denominator": baseline_endpoint,
                    "endpoint_state_ratio": endpoint_ratio,
                    "residual_rms_numerator": candidate_residual,
                    "residual_rms_denominator": baseline_residual,
                    "residual_rms_ratio": residual_ratio,
                    "maximum_control_ratio": maximum_control_ratio,
                    "maximum_required_integral_closure": maximum_integral_closure,
                    "correction_integral_contract_pass": (
                        correction_contract_pass
                    ),
                    "control_ratios_json": json.dumps(control_ratios, sort_keys=True),
                    "eligible": row_eligible,
                }
            )
            metric_specs = [
                (
                    "final_state_error",
                    candidate_endpoint,
                    baseline_endpoint,
                    endpoint_ratio,
                    CALIBRATION_NO_HARM_LIMIT,
                ),
                (
                    "residual_rms",
                    candidate_residual,
                    baseline_residual,
                    residual_ratio,
                    CALIBRATION_NO_HARM_LIMIT,
                ),
                *(
                    (
                        f"control::{name}",
                        candidate_controls.get(name) if complete else None,
                        baseline_controls[name],
                        control_ratios.get(name),
                        CONTROL_NO_HARM_LIMIT,
                    )
                    for name in sorted(required_controls)
                ),
            ]
            for metric, numerator, denominator, ratio, threshold in metric_specs:
                metric_rows.append(
                    {
                        "candidate": candidate.key,
                        "rank": candidate.rank,
                        "gain": candidate.gain,
                        "correction_policy": candidate.correction_policy,
                        "case_id": case_id,
                        "metric": metric,
                        "complete_horizon": complete,
                        "numerator": numerator,
                        "denominator": denominator,
                        "denominator_valid": True,
                        "ratio": ratio,
                        "threshold": threshold,
                        "passed": bool(
                            complete and ratio is not None and ratio <= threshold
                        ),
                    }
                )
        summary = {
            "candidate": candidate.key,
            "rank": candidate.rank,
            "gain": candidate.gain,
            "correction_policy": candidate.correction_policy,
            "case_count": len(expected_cases),
            "median_endpoint_state_ratio": (
                float(np.median(endpoint_ratios)) if candidate_eligible else None
            ),
            "worst_endpoint_state_ratio": (
                float(np.max(endpoint_ratios)) if candidate_eligible else None
            ),
            "median_residual_rms_ratio": (
                float(np.median(residual_ratios)) if candidate_eligible else None
            ),
            "eligible": candidate_eligible,
        }
        if candidate_eligible:
            summary["score_tuple"] = [
                summary["median_endpoint_state_ratio"],
                summary["worst_endpoint_state_ratio"],
                summary["median_residual_rms_ratio"],
                candidate.rank,
                candidate.gain,
                CORRECTION_POLICY_COMPLEXITY[candidate.correction_policy],
                candidate.key,
            ]
        summaries.append(summary)
        if candidate_eligible:
            eligible_candidates.append((candidate, summary))
    if not eligible_candidates:
        raise ValueError("canonical zero candidate unexpectedly failed eligibility")
    selected, _ = min(
        eligible_candidates,
        key=lambda item: (
            item[1]["median_endpoint_state_ratio"],
            item[1]["worst_endpoint_state_ratio"],
            item[1]["median_residual_rms_ratio"],
            item[0].rank,
            item[0].gain,
            CORRECTION_POLICY_COMPLEXITY[item[0].correction_policy],
            item[0].key,
        ),
    )
    return selected, rows, summaries, metric_rows


def _crossfit_calibration(
    model: torch.nn.Module,
    cases: Sequence[CaseData],
    candidates: Sequence[Candidate],
    *,
    shock_quantile: float,
    smoke: bool,
) -> tuple[
    Candidate,
    dict[str, dict[int, np.ndarray]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    case_by_id = {case.case_id: case for case in cases}
    if len(case_by_id) != len(cases):
        raise ValueError("calibration cases are not unique")
    all_case_ids = tuple(sorted(case_by_id))
    fold_case_ids = all_case_ids[:1] if smoke else all_case_ids
    required_ranks = tuple(
        sorted({candidate.rank for candidate in candidates if not candidate.is_zero})
    )
    coefficients_by_case = {
        case_id: _fit_case_coefficients(model, case_by_id[case_id], required_ranks)
        for case_id in all_case_ids
    }
    rollouts: dict[str, dict[str, SelectorRolloutSummary]] = {
        candidate.key: {} for candidate in candidates
    }
    for case_id in fold_case_ids:
        case = case_by_id[case_id]
        rollouts["zero"][case_id] = _selector_rollout_summary(
            _rollout_candidate(
                model,
                case,
                candidates[0],
                bias_sequence=None,
                shock_quantile=shock_quantile,
            )
        )
        training_ids = [value for value in all_case_ids if value != case_id]
        if not training_ids:
            raise ValueError("cross-fit fold has no remaining calibration case")
        for candidate in candidates[1:]:
            frozen = mean_calibration_coefficients(
                np.asarray(
                    [
                        coefficients_by_case[training_id][candidate.rank]
                        for training_id in training_ids
                    ]
                )
            )
            _, bias, _ = _candidate_bias_sequence(case, frozen, candidate)
            rollouts[candidate.key][case_id] = _selector_rollout_summary(
                _rollout_candidate(
                    model,
                    case,
                    candidate,
                    bias_sequence=bias,
                    shock_quantile=shock_quantile,
                )
            )
    selected, rows, summaries, metric_rows = select_candidate(
        candidates,
        fold_case_ids,
        rollouts,
        family=cases[0].family,
    )
    return selected, coefficients_by_case, rows, summaries, metric_rows


def _frozen_selected_coefficients(
    selected: Candidate,
    coefficients_by_case: Mapping[str, Mapping[int, np.ndarray]],
) -> np.ndarray | None:
    if selected.is_zero:
        return None
    return mean_calibration_coefficients(
        np.asarray(
            [
                by_rank[selected.rank]
                for _, by_rank in sorted(coefficients_by_case.items())
            ]
        )
    )


def _weighted_quantity(field: np.ndarray, case: CaseData) -> tuple[np.ndarray, str]:
    weighted_sum = np.einsum("n,nc->c", case.weights, field, optimize=True)
    if case.family == "dynamic_fv":
        return weighted_sum, "signed physical-volume integral"
    if case.family == "bump":
        return (
            weighted_sum / float(case.weights.sum()),
            "signed proxy-weighted mean; nonphysical",
        )
    raise ValueError(f"unsupported D074 family: {case.family}")


def _sequence_artifacts(
    case: CaseData,
    *,
    arm: str,
    defect_kind: str,
    defects: np.ndarray,
    truth_increments: np.ndarray,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    raw_call_rows, summary = sequence_diagnostics(
        defects,
        truth_increments,
        volumes=case.weights,
        component_scale=case.residual_scale,
    )
    identity = {
        "family": case.family,
        "case_id": case.case_id,
        "resolution": case.resolution_name,
        "arm": arm,
        "defect_kind": defect_kind,
    }
    call_rows = [{**identity, **row} for row in raw_call_rows]
    summary_row = {
        **identity,
        **{
            name: value
            for name, value in summary.items()
            if name not in {"lag_correlations", "uncentered_pod", "centered_pod"}
        },
    }
    lag_rows = [{**identity, **row} for row in summary["lag_correlations"]]
    pod_rows = []
    for centering in ("uncentered", "centered"):
        pod = summary[f"{centering}_pod"]
        pod_rows.append(
            {
                **identity,
                "centering": centering,
                "first_mode_energy_fraction": pod["first_mode_energy_fraction"],
                "first_three_energy_fraction": pod["first_three_energy_fraction"],
                "modes_for_95_percent": pod["modes_for_95_percent"],
                "energy_fractions_json": json.dumps(pod["energy_fractions"]),
            }
        )
    return call_rows, summary_row, lag_rows, pod_rows


def _correction_mode_parts(
    case: CaseData,
    correction: np.ndarray,
    basis: np.ndarray,
    applied_coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    coefficients = np.asarray(applied_coefficients, dtype=np.float64)
    if coefficients.shape != (basis.shape[1], len(COMPONENTS)):
        raise ValueError("applied correction coefficients do not match the basis")
    constant = (
        np.einsum("nr,rc->nc", basis[:, :1], coefficients[:1], optimize=True)
        * case.residual_scale[None, :]
    )
    nonconstant = (
        np.einsum("nr,rc->nc", basis[:, 1:], coefficients[1:], optimize=True)
        * case.residual_scale[None, :]
        if basis.shape[1] > 1
        else np.zeros_like(correction)
    )
    type0 = case.physical_node_type == 0
    constant[~type0] = 0.0
    nonconstant[~type0] = 0.0
    closure = float(np.max(np.abs(correction - constant - nonconstant)))
    return constant, nonconstant, closure


def _same_input_rows(
    case: CaseData,
    result: RolloutResult,
    *,
    arm: str,
    basis: np.ndarray | None,
    applied_coefficients: np.ndarray | None = None,
) -> dict[str, list[dict[str, Any]]]:
    rows = []
    projection_rows = []
    projection_component_rows = []
    budget_rows = []
    errors = result.states - case.reference_states[: result.states.shape[0]]
    (
        sequence_rows,
        sequence_summary,
        lag_rows,
        pod_rows,
    ) = _sequence_artifacts(
        case,
        arm=arm,
        defect_kind="corrected_defect_on_own_recurrent_inputs",
        defects=result.defects,
        truth_increments=result.truth_increments,
    )
    sequence_time_rows = list(sequence_rows)
    sequence_summary_rows = [sequence_summary]
    if basis is not None:
        if applied_coefficients is None or applied_coefficients.shape[0] != len(
            result.defects
        ):
            raise ValueError("selected projection requires per-call coefficients")
        base_call_rows, base_summary, base_lags, base_pods = _sequence_artifacts(
            case,
            arm=arm,
            defect_kind="base_defect_on_same_corrected_inputs",
            defects=result.base_defects,
            truth_increments=result.truth_increments,
        )
        sequence_time_rows.extend(base_call_rows)
        sequence_summary_rows.append(base_summary)
        lag_rows.extend(base_lags)
        pod_rows.extend(base_pods)
    recurrence_rows, _ = recurrence_diagnostics(
        errors,
        result.defects,
        volumes=case.weights,
        component_scale=case.residual_scale,
    )
    for index, (sequence_row, recurrence_row) in enumerate(
        zip(sequence_rows, recurrence_rows, strict=True)
    ):
        base = result.base_defects[index]
        correction = result.corrections[index]
        corrected = result.defects[index]
        correction_energy = weighted_inner(
            correction,
            correction,
            volumes=case.weights,
            component_scale=case.residual_scale,
        )
        twice_cross = 2.0 * weighted_inner(
            correction,
            base,
            volumes=case.weights,
            component_scale=case.residual_scale,
        )
        base_energy = weighted_inner(
            base,
            base,
            volumes=case.weights,
            component_scale=case.residual_scale,
        )
        corrected_energy = weighted_inner(
            corrected,
            corrected,
            volumes=case.weights,
            component_scale=case.residual_scale,
        )
        state_error = errors[index + 1]
        state_error_rms = _weighted_rms(
            state_error,
            weights=case.weights,
            scale=case.state_scale,
        )
        correction_cosine = cosine(
            correction,
            base,
            volumes=case.weights,
            component_scale=case.residual_scale,
        )
        defect_closure = float(np.max(np.abs(corrected - base - correction)))
        physical_component_fields = {}
        for component, name in enumerate(COMPONENTS):
            one = slice(component, component + 1)
            unit_scale = np.ones(1, dtype=np.float64)
            physical_component_fields[f"instant_truth_physical_rms__{name}"] = (
                _weighted_rms(
                    result.truth_increments[index, :, one],
                    weights=case.weights,
                    scale=unit_scale,
                )
            )
            physical_component_fields[f"instant_defect_physical_rms__{name}"] = (
                _weighted_rms(
                    corrected[:, one],
                    weights=case.weights,
                    scale=unit_scale,
                )
            )
        rows.append(
            {
                "family": case.family,
                "case_id": case.case_id,
                "resolution": case.resolution_name,
                "arm": arm,
                "call": index + 1,
                "physical_time": float(case.physical_times[index + 1]),
                "state_error_rms": state_error_rms,
                **sequence_row,
                **recurrence_row,
                "base_defect_energy_same_input": base_energy,
                "correction_energy": correction_energy,
                "twice_correction_base_inner": twice_cross,
                "corrected_minus_base_energy": corrected_energy - base_energy,
                "same_input_energy_closure": (
                    corrected_energy - base_energy - twice_cross - correction_energy
                ),
                "same_input_pointwise_closure_max": defect_closure,
                "correction_base_cosine": correction_cosine,
                "correction_base_cosine_valid": correction_cosine is not None,
                **physical_component_fields,
            }
        )
        cumulative_correction = result.corrections[: index + 1].sum(axis=0)
        budget_fields = {
            "base_defect_same_corrected_input": (base, case.residual_scale),
            "correction": (correction, case.residual_scale),
            "corrected_defect": (corrected, case.residual_scale),
            "cumulative_correction": (cumulative_correction, case.residual_scale),
            "state_error": (state_error, case.state_scale),
        }
        for field_name, (field, scale) in budget_fields.items():
            quantity, semantics = _weighted_quantity(field, case)
            for component, name in enumerate(COMPONENTS):
                budget_rows.append(
                    {
                        "family": case.family,
                        "case_id": case.case_id,
                        "resolution": case.resolution_name,
                        "arm": arm,
                        "call": index + 1,
                        "physical_time": float(case.physical_times[index + 1]),
                        "field": field_name,
                        "component": name,
                        "raw_signed_quantity": float(quantity[component]),
                        "scaled_signed_quantity": float(
                            quantity[component] / scale[component]
                        ),
                        "quantity_semantics": semantics,
                    }
                )
        if basis is not None:
            type0 = case.physical_node_type == 0
            constant, nonconstant, coefficient_closure = _correction_mode_parts(
                case,
                correction,
                basis,
                applied_coefficients[index],
            )
            for field_name, field in (
                ("correction_constant_mode", constant),
                ("correction_nonconstant_modes", nonconstant),
            ):
                quantity, semantics = _weighted_quantity(field, case)
                for component, name in enumerate(COMPONENTS):
                    component_field = field[:, component : component + 1]
                    energy = _weighted_rms(
                        component_field,
                        weights=case.weights,
                        scale=case.residual_scale[component : component + 1],
                    )
                    budget_rows.append(
                        {
                            "family": case.family,
                            "case_id": case.case_id,
                            "resolution": case.resolution_name,
                            "arm": arm,
                            "call": index + 1,
                            "physical_time": float(case.physical_times[index + 1]),
                            "field": field_name,
                            "component": name,
                            "raw_signed_quantity": float(quantity[component]),
                            "scaled_signed_quantity": float(
                                quantity[component] / case.residual_scale[component]
                            ),
                            "scaled_energy": float(energy**2),
                            "energy_additivity_requires_cross_term": True,
                            "quantity_semantics": semantics,
                        }
                    )
            scaled_constant = constant / case.residual_scale[None, :]
            scaled_nonconstant = nonconstant / case.residual_scale[None, :]
            scaled_correction = correction / case.residual_scale[None, :]
            total_weight = float(case.weights.sum())
            constant_component_energy = (
                np.einsum(
                    "n,nc->c", case.weights, np.square(scaled_constant), optimize=True
                )
                / total_weight
            )
            nonconstant_component_energy = (
                np.einsum(
                    "n,nc->c",
                    case.weights,
                    np.square(scaled_nonconstant),
                    optimize=True,
                )
                / total_weight
            )
            correction_component_energy = (
                np.einsum(
                    "n,nc->c", case.weights, np.square(scaled_correction), optimize=True
                )
                / total_weight
            )
            twice_mode_cross_component = (
                2.0
                * np.einsum(
                    "n,nc,nc->c",
                    case.weights,
                    scaled_constant,
                    scaled_nonconstant,
                    optimize=True,
                )
                / total_weight
            )
            mode_energy_closure = (
                correction_component_energy
                - constant_component_energy
                - nonconstant_component_energy
                - twice_mode_cross_component
            )
            before = weighted_subspace_decomposition(
                base,
                basis,
                case.weights,
                type0,
                component_scale=case.residual_scale,
            )
            after = weighted_subspace_decomposition(
                corrected,
                basis,
                case.weights,
                type0,
                component_scale=case.residual_scale,
            )
            correction_projection = weighted_subspace_decomposition(
                correction,
                basis,
                case.weights,
                type0,
                component_scale=case.residual_scale,
            )
            non_type0 = ~type0
            projection_rows.append(
                {
                    "family": case.family,
                    "case_id": case.case_id,
                    "resolution": case.resolution_name,
                    "arm": arm,
                    "call": index + 1,
                    "physical_time": float(case.physical_times[index + 1]),
                    "projection_status": "applied",
                    "effective_rank": before.effective_rank,
                    "type0_fit_support_count": int(type0.sum()),
                    "excluded_non_type0_count": int(non_type0.sum()),
                    "before_total_energy": before.total_energy,
                    "before_parallel_energy": before.parallel_energy,
                    "before_orthogonal_energy": before.orthogonal_energy,
                    "before_excluded_non_type0_energy": (
                        before.excluded_non_type0_energy
                    ),
                    "after_total_energy": after.total_energy,
                    "after_parallel_energy": after.parallel_energy,
                    "after_orthogonal_energy": after.orthogonal_energy,
                    "after_excluded_non_type0_energy": (
                        after.excluded_non_type0_energy
                    ),
                    "correction_parallel_energy": (
                        correction_projection.parallel_energy
                    ),
                    "correction_orthogonal_energy": (
                        correction_projection.orthogonal_energy
                    ),
                    "correction_excluded_non_type0_energy": (
                        correction_projection.excluded_non_type0_energy
                    ),
                    "maximum_parallel_orthogonal_inner": max(
                        abs(before.parallel_orthogonal_inner),
                        abs(after.parallel_orthogonal_inner),
                        abs(correction_projection.parallel_orthogonal_inner),
                    ),
                    "maximum_energy_closure": max(
                        abs(before.energy_closure),
                        abs(after.energy_closure),
                        abs(correction_projection.energy_closure),
                    ),
                    "maximum_reconstruction_error": max(
                        before.maximum_reconstruction_error,
                        after.maximum_reconstruction_error,
                        correction_projection.maximum_reconstruction_error,
                    ),
                    "maximum_non_type0_correction": (
                        float(np.max(np.abs(correction[non_type0])))
                        if np.any(non_type0)
                        else 0.0
                    ),
                    "maximum_coefficient_reconstruction_error": coefficient_closure,
                    "maximum_orthogonal_partition_change": float(
                        np.max(np.abs(after.orthogonal - before.orthogonal))
                    ),
                    "maximum_excluded_partition_change": float(
                        np.max(
                            np.abs(after.excluded_non_type0 - before.excluded_non_type0)
                        )
                    ),
                }
            )
            for component, name in enumerate(COMPONENTS):
                projection_component_rows.append(
                    {
                        "family": case.family,
                        "case_id": case.case_id,
                        "resolution": case.resolution_name,
                        "arm": arm,
                        "call": index + 1,
                        "physical_time": float(case.physical_times[index + 1]),
                        "component": name,
                        "before_total_energy": before.component_total_energy[component],
                        "before_parallel_energy": (
                            before.component_parallel_energy[component]
                        ),
                        "before_orthogonal_energy": (
                            before.component_orthogonal_energy[component]
                        ),
                        "before_excluded_non_type0_energy": (
                            before.component_excluded_non_type0_energy[component]
                        ),
                        "before_parallel_orthogonal_inner": (
                            before.component_parallel_orthogonal_inner[component]
                        ),
                        "before_energy_closure": (
                            before.component_total_energy[component]
                            - before.component_parallel_energy[component]
                            - before.component_orthogonal_energy[component]
                            - before.component_excluded_non_type0_energy[component]
                        ),
                        "after_total_energy": after.component_total_energy[component],
                        "after_parallel_energy": (
                            after.component_parallel_energy[component]
                        ),
                        "after_orthogonal_energy": (
                            after.component_orthogonal_energy[component]
                        ),
                        "after_excluded_non_type0_energy": (
                            after.component_excluded_non_type0_energy[component]
                        ),
                        "after_parallel_orthogonal_inner": (
                            after.component_parallel_orthogonal_inner[component]
                        ),
                        "after_energy_closure": (
                            after.component_total_energy[component]
                            - after.component_parallel_energy[component]
                            - after.component_orthogonal_energy[component]
                            - after.component_excluded_non_type0_energy[component]
                        ),
                        "correction_total_energy": (
                            correction_projection.component_total_energy[component]
                        ),
                        "correction_constant_mode_energy": (
                            constant_component_energy[component]
                        ),
                        "correction_nonconstant_modes_energy": (
                            nonconstant_component_energy[component]
                        ),
                        "twice_constant_nonconstant_inner": (
                            twice_mode_cross_component[component]
                        ),
                        "constant_nonconstant_energy_closure": (
                            mode_energy_closure[component]
                        ),
                        "correction_parallel_energy": (
                            correction_projection.component_parallel_energy[component]
                        ),
                        "correction_orthogonal_energy": (
                            correction_projection.component_orthogonal_energy[component]
                        ),
                        "correction_excluded_non_type0_energy": (
                            correction_projection.component_excluded_non_type0_energy[
                                component
                            ]
                        ),
                        "correction_parallel_orthogonal_inner": (
                            correction_projection.component_parallel_orthogonal_inner[
                                component
                            ]
                        ),
                        "correction_energy_closure": (
                            correction_projection.component_total_energy[component]
                            - correction_projection.component_parallel_energy[component]
                            - correction_projection.component_orthogonal_energy[
                                component
                            ]
                            - correction_projection.component_excluded_non_type0_energy[
                                component
                            ]
                        ),
                    }
                )
    return {
        "call_rows": rows,
        "projection_rows": projection_rows,
        "projection_component_rows": projection_component_rows,
        "budget_rows": budget_rows,
        "sequence_summary_rows": sequence_summary_rows,
        "sequence_time_rows": sequence_time_rows,
        "lag_rows": lag_rows,
        "pod_rows": pod_rows,
    }


def _evaluation_gate(
    family: str,
    baseline_by_case: Mapping[str, RolloutResult],
    selected_by_case: Mapping[str, RolloutResult],
    selected: Candidate,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    expected_ids = tuple(sorted(baseline_by_case))
    if set(selected_by_case) != set(expected_ids):
        raise ValueError("evaluation arm case inventories differ")
    rows = []
    state_ratios = []
    residual_ratios = []
    control_maxima = []
    for case_id in expected_ids:
        baseline = baseline_by_case[case_id]
        result = selected_by_case[case_id]
        if not baseline.complete:
            raise ValueError(f"evaluation baseline failed for {case_id}")
        if not result.complete:
            raise ValueError(
                f"selected evaluation rollout is incomplete for {case_id}; "
                "no unequal-horizon ratio is defined"
            )
        baseline_state = baseline.summary.get("final_state_error")
        selected_state = result.summary.get("final_state_error")
        baseline_residual = baseline.summary.get("residual_rms")
        selected_residual = result.summary.get("residual_rms")
        state_ratio = _required_ratio(
            selected_state,
            baseline_state,
        )
        residual_ratio = _required_ratio(
            selected_residual,
            baseline_residual,
        )
        baseline_controls = baseline.summary.get("controls", {})
        selected_controls = result.summary.get("controls", {})
        required_controls = set(expected_control_keys(family))
        if (
            set(baseline_controls) != required_controls
            or set(selected_controls) != required_controls
        ):
            raise ValueError("evaluation control inventory differs from contract")
        control_ratios = {
            name: _required_ratio(selected_controls[name], baseline_controls[name])
            for name in sorted(baseline_controls)
        }
        maximum_control = max(control_ratios.values(), default=1.0)
        rows.append(
            {
                "family": family,
                "case_id": case_id,
                "candidate": selected.key,
                "correction_policy": selected.correction_policy,
                "complete": result.complete,
                "final_state_numerator": selected_state,
                "final_state_denominator": baseline_state,
                "final_state_ratio": state_ratio,
                "residual_rms_numerator": selected_residual,
                "residual_rms_denominator": baseline_residual,
                "residual_rms_ratio": residual_ratio,
                "maximum_control_ratio": maximum_control,
                "control_ratios_json": json.dumps(control_ratios, sort_keys=True),
            }
        )
        state_ratios.append(state_ratio)
        residual_ratios.append(residual_ratio)
        control_maxima.append(maximum_control)
    if family == "dynamic_fv":
        passed = bool(
            not selected.is_zero
            and all(result.complete for result in selected_by_case.values())
            and float(np.median(state_ratios)) <= 0.92
            and max(state_ratios) <= 1.0
            and max(residual_ratios) <= 1.0
            and max(control_maxima) <= CONTROL_NO_HARM_LIMIT
        )
        required_case_wins = len(state_ratios)
    else:
        required_case_wins = 3
        passed = bool(
            not selected.is_zero
            and all(result.complete for result in selected_by_case.values())
            and float(np.median(state_ratios)) <= 0.95
            and sum(value <= 1.0 for value in state_ratios) >= required_case_wins
            and max(state_ratios) <= CONTROL_NO_HARM_LIMIT
            and max(residual_ratios) <= CONTROL_NO_HARM_LIMIT
            and max(control_maxima) <= CONTROL_NO_HARM_LIMIT
        )
    return (
        {
            "passed": passed,
            "selected_candidate": selected.key,
            "selected_correction_policy": selected.correction_policy,
            "median_final_state_ratio": float(np.median(state_ratios)),
            "maximum_final_state_ratio": float(np.max(state_ratios)),
            "maximum_residual_rms_ratio": float(np.max(residual_ratios)),
            "maximum_control_ratio": float(np.max(control_maxima)),
            "case_wins": int(sum(value <= 1.0 for value in state_ratios)),
            "required_case_wins": required_case_wins,
            "adaptive_open_population": True,
        },
        rows,
    )


def _save_visual_payload(
    path: Path,
    case: CaseData,
    baseline: RolloutResult,
    selected: RolloutResult,
) -> dict[str, Any]:
    if not baseline.complete or not selected.complete:
        raise ValueError("visual payload requires complete baseline and selected arms")
    calls = case.reference_states.shape[0] - 1
    if baseline.defects.shape[0] != calls or selected.defects.shape[0] != calls:
        raise ValueError("visual payload arm length differs from the frozen horizon")
    errors = selected.states - case.reference_states[: selected.states.shape[0]]
    cumulative_closure = errors[1:] - np.cumsum(selected.defects, axis=0)
    maximum_cumulative_closure = float(np.max(np.abs(cumulative_closure)))
    if maximum_cumulative_closure > 2.0e-5:
        raise ValueError("visual cumulative-error replay failed")
    previous = errors[:-1] / case.residual_scale[None, None, :]
    defect = selected.defects / case.residual_scale[None, None, :]
    signed_growth_density = 2.0 * previous * defect + np.square(defect)
    signed_growth_contribution = (
        case.weights[None, :, None] * signed_growth_density / float(case.weights.sum())
    )
    squared_growth = np.sum(signed_growth_contribution, axis=(1, 2))
    direct_growth = np.asarray(
        [
            weighted_inner(
                errors[index + 1],
                errors[index + 1],
                volumes=case.weights,
                component_scale=case.residual_scale,
            )
            - weighted_inner(
                errors[index],
                errors[index],
                volumes=case.weights,
                component_scale=case.residual_scale,
            )
            for index in range(calls)
        ]
    )
    maximum_growth_replay = float(np.max(np.abs(squared_growth - direct_growth)))
    if maximum_growth_replay > 1.0e-10:
        raise ValueError("visual signed-growth replay failed")
    field_semantics = {
        "true_increment": "reference U[n+1]-U[n] in physical conservative units",
        "baseline_defect": "raw baseline-path defect on baseline recurrent inputs",
        "corrected_input_base_defect": (
            "uncorrected model defect b on the selected arm's corrected input"
        ),
        "correction": "applied offline call-indexed correction C",
        "corrected_defect": "same-input delta=b+C on selected recurrent inputs",
        "cumulative_error": "selected state minus native reference state",
        "signed_growth_density_scaled": (
            "pointwise component density (2 e delta + delta^2)/D_r^2; "
            "weights are not included"
        ),
        "signed_growth_contribution": (
            "node-weighted contribution whose node/component sum is squared-error growth"
        ),
    }
    np.savez_compressed(
        path,
        schema=np.asarray(SCHEMA),
        family=np.asarray(case.family),
        case_id=np.asarray(case.case_id),
        resolution=np.asarray(case.resolution_name),
        selected_candidate=np.asarray(selected.candidate.key),
        selected_rank=np.asarray(selected.candidate.rank, dtype=np.int64),
        selected_gain=np.asarray(selected.candidate.gain, dtype=np.float64),
        selected_correction_policy=np.asarray(
            selected.candidate.correction_policy
        ),
        expected_calls=np.asarray(calls, dtype=np.int64),
        baseline_complete=np.asarray(baseline.complete),
        selected_complete=np.asarray(selected.complete),
        component_names=np.asarray(COMPONENTS),
        weight_semantics=np.asarray(
            "physical_cell_volume"
            if case.family == "dynamic_fv"
            else "frozen_proxy_weight_nonphysical"
        ),
        field_semantics_json=np.asarray(json.dumps(field_semantics, sort_keys=True)),
        nodes=case.nodes.astype(np.float64),
        weights=case.weights.astype(np.float64),
        node_type=case.physical_node_type.astype(np.int64),
        physical_times=case.physical_times[1:].astype(np.float64),
        residual_scale=case.residual_scale.astype(np.float64),
        true_increment=selected.truth_increments.astype(np.float32),
        baseline_defect=baseline.defects.astype(np.float32),
        corrected_input_base_defect=selected.base_defects.astype(np.float32),
        correction=selected.corrections.astype(np.float32),
        corrected_defect=selected.defects.astype(np.float32),
        cumulative_error=errors[1:].astype(np.float32),
        signed_growth_density_scaled=signed_growth_density.astype(np.float32),
        signed_growth_contribution=signed_growth_contribution.astype(np.float32),
    )
    return {
        "family": case.family,
        "case_id": case.case_id,
        "resolution": case.resolution_name,
        "relative_path": path.name,
        "calls": calls,
        "selected_candidate": selected.candidate.key,
        "selected_correction_policy": selected.candidate.correction_policy,
        "maximum_cumulative_closure": maximum_cumulative_closure,
        "maximum_signed_growth_replay": maximum_growth_replay,
    }


def _load_cases(
    args: argparse.Namespace,
    checkpoint: Mapping[str, Any],
    store: PCNOEuler2DShardStore,
    case_ids: Sequence[str],
    *,
    device: torch.device,
) -> tuple[list[CaseData], list[dict[str, Any]]]:
    args.case_ids = list(case_ids)
    if args.family == "dynamic_fv":
        return _load_dynamic_cases(args, checkpoint, store, device=device)
    return _load_bump_cases(args, checkpoint, store, device=device), []


def _serialize_coefficients(
    path: Path,
    *,
    family: str,
    selected: Candidate,
    coefficients: np.ndarray | None,
) -> None:
    np.savez_compressed(
        path,
        schema=np.asarray(SCHEMA),
        family=np.asarray(family),
        candidate=np.asarray(selected.key),
        rank=np.asarray(selected.rank, dtype=np.int64),
        gain=np.asarray(selected.gain, dtype=np.float64),
        correction_policy=np.asarray(selected.correction_policy),
        coefficients=(
            np.empty((0, 0, 0), dtype=np.float64)
            if coefficients is None
            else np.asarray(coefficients, dtype=np.float64)
        ),
    )


def _case_summary_row(
    case: CaseData,
    arm: str,
    result: RolloutResult,
) -> dict[str, Any]:
    row = {
        "family": case.family,
        "case_id": case.case_id,
        "resolution": case.resolution_name,
        "arm": arm,
        "candidate": result.candidate.key,
        "correction_policy": result.candidate.correction_policy,
        "complete": result.complete,
        "valid_length": result.valid_length,
        "final_state_error": result.summary.get("final_state_error"),
        "residual_rms": result.summary.get("residual_rms"),
        "controls_json": json.dumps(result.summary.get("controls", {}), sort_keys=True),
        "mask_inventory_json": json.dumps(
            result.summary.get("mask_inventory", {}), sort_keys=True
        ),
        "global_budgets_json": json.dumps(
            result.summary.get("global_budgets", {}), sort_keys=True
        ),
        "correction_integral_audit_json": json.dumps(
            result.summary.get("correction_integral_audit", {}), sort_keys=True
        ),
    }
    if result.defects.shape[0]:
        _, sequence_summary = sequence_diagnostics(
            result.defects,
            result.truth_increments,
            volumes=case.weights,
            component_scale=case.residual_scale,
        )
        for name in (
            "aggregate_relative_residual_energy",
            "path_sum_rms",
            "net_defect_rms",
            "net_truth_change_rms",
            "net_relative_to_truth_change",
            "temporal_coherence",
            "net_defect_truth_cosine",
        ):
            row[name] = sequence_summary[name]
    return row


def _assert_selector_bundle_unchanged(
    selector_path: Path,
    selector_sha256: str,
    artifact_sha256: Mapping[str, str],
) -> None:
    if not selector_path.is_file() or sha256_file(selector_path) != selector_sha256:
        raise ValueError("frozen selector artifact is missing or changed")
    if not artifact_sha256:
        raise ValueError("frozen selector bundle has no artifact inventory")
    root = selector_path.parent.resolve()
    for relative_name, expected_digest in artifact_sha256.items():
        candidate = (root / relative_name).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise ValueError(
                "selector bundle path escapes its output directory"
            ) from error
        if not candidate.is_file() or sha256_file(candidate) != expected_digest:
            raise ValueError(
                f"frozen selector bundle artifact changed: {relative_name}"
            )


def _assert_selector_frozen_before_evaluation(
    selector_path: Path,
    selector_sha256: str,
    artifact_sha256: Mapping[str, str],
    phase_rows: Sequence[Mapping[str, Any]],
) -> None:
    _assert_selector_bundle_unchanged(selector_path, selector_sha256, artifact_sha256)
    if not phase_rows or phase_rows[-1].get("phase") != "selector_frozen":
        raise ValueError("evaluation targets cannot load before selector freeze")
    if phase_rows[-1].get("evaluation_targets_loaded") is not False:
        raise ValueError("selector freeze phase already reports evaluation targets")


def _row_inventory_exact(
    rows: Sequence[Mapping[str, Any]],
    expected: set[tuple[Any, ...]],
    fields: Sequence[str],
) -> bool:
    observed = [tuple(row.get(field) for field in fields) for row in rows]
    return (
        len(observed) == len(expected)
        and len(set(observed)) == len(observed)
        and set(observed) == expected
    )


def _maximum_absolute(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    if not rows:
        return float("inf")
    values = [row.get(field) for row in rows]
    if any(value is None or not np.isfinite(value) for value in values):
        return float("inf")
    return max(abs(float(value)) for value in values)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = perf_counter()
    device = select_device(args.device)
    calibration_ids, evaluation_ids = _family_cases(args.family)
    split = json.loads(args.split_json.read_text(encoding="utf-8"))
    validation_ids = tuple(str(value) for value in split.get("val_keys", ()))
    if set(validation_ids) != set(calibration_ids) | set(evaluation_ids):
        raise ValueError("D074 calibration/evaluation IDs do not partition validation")
    active_calibration_ids = calibration_ids[:2] if args.smoke else calibration_ids
    active_evaluation_ids = evaluation_ids[:1] if args.smoke else evaluation_ids
    candidates = candidate_inventory(
        smoke=args.smoke,
        experiment_contract=args.experiment_contract,
    )
    store = PCNOEuler2DShardStore(args.data_dir, max_cached_trajectories=2)
    if args.family == "dynamic_fv":
        checkpoint = load_resolution_checkpoint(args.checkpoint)
        model, _ = build_resolution_checkpoint_model(checkpoint, device)
    else:
        checkpoint = load_checkpoint_payload(args.checkpoint)
        model, _ = build_checkpoint_model(
            checkpoint, device, model_node_type_input="physical"
        )
    model.eval()
    provenance = _verify_common_provenance(args, checkpoint, store)
    observed_git_head = git_head()
    if observed_git_head != args.expected_source_base_git_head.lower():
        raise ValueError(
            "active repository HEAD differs from the frozen D074 source base"
        )
    phase_rows: list[dict[str, Any]] = []
    case_contract_rows: list[dict[str, Any]] = []
    try:
        calibration_cases, calibration_reference_checks = _load_cases(
            args,
            checkpoint,
            store,
            active_calibration_ids,
            device=device,
        )
        case_contract_rows.extend(
            _validate_case_contract(case) for case in calibration_cases
        )
        phase_rows.append(
            {
                "phase": "calibration_targets_loaded",
                "case_count": len(calibration_cases),
                "evaluation_targets_loaded": False,
            }
        )
        calibration_hook = _run_hook_equivalence(
            model, calibration_cases[0], device=device
        )
        (
            selected,
            coefficients_by_case,
            selector_rows,
            selector_summary_rows,
            selector_metric_rows,
        ) = _crossfit_calibration(
            model,
            calibration_cases,
            candidates,
            shock_quantile=args.shock_quantile,
            smoke=args.smoke,
        )
        frozen_coefficients = _frozen_selected_coefficients(
            selected, coefficients_by_case
        )

        args.output_dir.mkdir(parents=True)
        visual_dir = args.output_dir / "visual_payloads"
        visual_dir.mkdir()
        coefficient_payload = {
            "schema": np.asarray(SCHEMA),
            "family": np.asarray(args.family),
            "experiment_contract": np.asarray(args.experiment_contract),
        }
        for case_id, by_rank in sorted(coefficients_by_case.items()):
            for rank, values in sorted(by_rank.items()):
                coefficient_payload[f"case__{case_id}__rank__{rank}"] = values
        calibration_coefficients_path = (
            args.output_dir / "calibration_coefficients_by_case.npz"
        )
        np.savez_compressed(calibration_coefficients_path, **coefficient_payload)
        frozen_path = args.output_dir / "frozen_selected_coefficients.npz"
        _serialize_coefficients(
            frozen_path,
            family=args.family,
            selected=selected,
            coefficients=frozen_coefficients,
        )
        selector_rows_path = args.output_dir / "selector_rows.csv"
        selector_summary_path = args.output_dir / "selector_summary.csv"
        selector_metrics_path = args.output_dir / "selector_metrics.csv"
        write_csv_with_paths(selector_rows_path, selector_rows)
        write_csv_with_paths(selector_summary_path, selector_summary_rows)
        write_csv_with_paths(selector_metrics_path, selector_metric_rows)
        selected_summary = next(
            row for row in selector_summary_rows if row["candidate"] == selected.key
        )
        selector_record = {
            "schema": SCHEMA,
            "family": args.family,
            "experiment_contract": args.experiment_contract,
            "smoke": args.smoke,
            "candidate_inventory": [candidate.__dict__ for candidate in candidates],
            "fold_case_ids": (
                [active_calibration_ids[0]]
                if args.smoke
                else sorted(active_calibration_ids)
            ),
            "fit_case_ids": sorted(active_calibration_ids),
            "selected": selected.__dict__,
            "selected_score_tuple": selected_summary["score_tuple"],
            "selection_order": [
                "median_endpoint_state_ratio",
                "worst_endpoint_state_ratio",
                "median_residual_rms_ratio",
                "lower_rank",
                "lower_gain",
                "lower_correction_policy_complexity",
                "candidate_key",
            ],
            "evaluation_targets_loaded_before_freeze": False,
            "artifact_sha256": {
                "calibration_coefficients_by_case.npz": sha256_file(
                    calibration_coefficients_path
                ),
                "frozen_selected_coefficients.npz": sha256_file(frozen_path),
                "selector_rows.csv": sha256_file(selector_rows_path),
                "selector_summary.csv": sha256_file(selector_summary_path),
                "selector_metrics.csv": sha256_file(selector_metrics_path),
            },
        }
        selector_path = args.output_dir / "selector.json"
        write_json(selector_path, selector_record)
        selector_sha = sha256_file(selector_path)
        phase_rows.append(
            {
                "phase": "selector_frozen",
                "case_count": len(active_calibration_ids),
                "evaluation_targets_loaded": False,
                "selector_sha256": selector_sha,
            }
        )

        _assert_selector_frozen_before_evaluation(
            selector_path,
            selector_sha,
            selector_record["artifact_sha256"],
            phase_rows,
        )

        evaluation_cases, evaluation_reference_checks = _load_cases(
            args,
            checkpoint,
            store,
            active_evaluation_ids,
            device=device,
        )
        case_contract_rows.extend(
            _validate_case_contract(case) for case in evaluation_cases
        )
        phase_rows.append(
            {
                "phase": "evaluation_targets_loaded",
                "case_count": len(evaluation_cases),
                "evaluation_targets_loaded": True,
                "frozen_selector_sha256": selector_sha,
            }
        )
        evaluation_hook = _run_hook_equivalence(
            model, evaluation_cases[0], device=device
        )
        cases_by_id = {case.case_id: case for case in evaluation_cases}
        bump_replay = (
            _bump_replay_gate(args, model, cases_by_id, device=device)
            if args.family == "bump"
            else None
        )

        baseline_by_case: dict[str, RolloutResult] = {}
        selected_by_case: dict[str, RolloutResult] = {}
        case_summary_rows = []
        call_rows = []
        projection_rows = []
        projection_component_rows = []
        projection_status_rows = []
        budget_rows = []
        correction_integral_rows = []
        sequence_summary_rows = []
        sequence_time_rows = []
        lag_rows = []
        pod_rows = []
        completion_rows = []
        visual_inventory_rows = []
        default_visual_ids = (
            {active_evaluation_ids[0], active_evaluation_ids[-1]}
            if args.family == "dynamic_fv"
            else {active_evaluation_ids[0]}
        )
        visual_ids = (
            set(args.visualization_cases)
            if args.visualization_cases
            else default_visual_ids
        )
        unknown_visual_ids = visual_ids - set(active_evaluation_ids)
        if unknown_visual_ids:
            raise ValueError(
                f"visualization cases are outside active evaluation: "
                f"{sorted(unknown_visual_ids)}"
            )
        for index, case in enumerate(evaluation_cases, start=1):
            baseline = _rollout_candidate(
                model,
                case,
                candidates[0],
                bias_sequence=None,
                shock_quantile=args.shock_quantile,
            )
            baseline_by_case[case.case_id] = baseline
            if selected.is_zero:
                selected_result = baseline
                selected_basis = None
                selected_applied_coefficients = None
                selected_policy_adjustment = np.zeros_like(baseline.corrections)
            else:
                _, raw_selected_bias = _bias_sequence(
                    case,
                    frozen_coefficients,
                    rank=selected.rank,
                )
                (
                    selected_basis,
                    selected_bias,
                    selected_policy_coefficients,
                ) = _candidate_bias_sequence(
                    case,
                    frozen_coefficients,
                    selected,
                )
                selected_applied_coefficients = (
                    -selected.gain * selected_policy_coefficients
                )
                selected_policy_adjustment = -selected.gain * (
                    selected_bias - raw_selected_bias
                )
                selected_result = _rollout_candidate(
                    model,
                    case,
                    selected,
                    bias_sequence=selected_bias,
                    shock_quantile=args.shock_quantile,
                )
            selected_by_case[case.case_id] = selected_result
            for arm, result, basis, arm_coefficients, policy_adjustment in (
                (
                    "baseline",
                    baseline,
                    None,
                    None,
                    np.zeros_like(baseline.corrections),
                ),
                (
                    "selected",
                    selected_result,
                    selected_basis,
                    selected_applied_coefficients,
                    selected_policy_adjustment,
                ),
            ):
                case_summary_rows.append(_case_summary_row(case, arm, result))
                correction_integral_rows.extend(
                    _correction_integral_rows(
                        case,
                        result,
                        arm=arm,
                        policy_adjustment=policy_adjustment,
                    )
                )
                if result.defects.shape[0]:
                    artifacts = _same_input_rows(
                        case,
                        result,
                        arm=arm,
                        basis=basis,
                        applied_coefficients=arm_coefficients,
                    )
                    call_rows.extend(artifacts["call_rows"])
                    projection_rows.extend(artifacts["projection_rows"])
                    projection_component_rows.extend(
                        artifacts["projection_component_rows"]
                    )
                    budget_rows.extend(artifacts["budget_rows"])
                    sequence_summary_rows.extend(artifacts["sequence_summary_rows"])
                    sequence_time_rows.extend(artifacts["sequence_time_rows"])
                    lag_rows.extend(artifacts["lag_rows"])
                    pod_rows.extend(artifacts["pod_rows"])
                for row in result.admissibility_rows:
                    completion_rows.append(
                        {
                            "family": case.family,
                            "case_id": case.case_id,
                            "resolution": case.resolution_name,
                            "arm": arm,
                            **row,
                        }
                    )
            projection_status_rows.append(
                {
                    "family": case.family,
                    "case_id": case.case_id,
                    "resolution": case.resolution_name,
                    "selected_candidate": selected.key,
                    "status": (
                        "not_applicable_zero"
                        if selected.is_zero
                        else (
                            "applied_complete"
                            if selected_result.complete
                            else "failed_incomplete"
                        )
                    ),
                }
            )
            if case.case_id in visual_ids:
                visual_path = (
                    visual_dir
                    / f"{case.family}_{case.case_id}_{case.resolution_name}.npz"
                )
                visual_row = _save_visual_payload(
                    visual_path, case, baseline, selected_result
                )
                visual_row["relative_path"] = str(
                    visual_path.relative_to(args.output_dir)
                ).replace("\\", "/")
                visual_row["sha256"] = sha256_file(visual_path)
                visual_inventory_rows.append(visual_row)
            print(
                f"evaluated {index}/{len(evaluation_cases)} "
                f"{case.case_id} {case.resolution_name}",
                flush=True,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        store.close()

    promotion, evaluation_ratio_rows = _evaluation_gate(
        args.family, baseline_by_case, selected_by_case, selected
    )
    output_rows = {
        "phase_order.csv": phase_rows,
        "case_contracts.csv": case_contract_rows,
        "evaluation_case_summary.csv": case_summary_rows,
        "evaluation_ratios.csv": evaluation_ratio_rows,
        "evaluation_call_metrics.csv": call_rows,
        "sequence_summaries.csv": sequence_summary_rows,
        "sequence_time_metrics.csv": sequence_time_rows,
        "lag_correlations.csv": lag_rows,
        "pod_summaries.csv": pod_rows,
        "signed_component_budgets.csv": budget_rows,
        "correction_integral_audit.csv": correction_integral_rows,
        "projection_status.csv": projection_status_rows,
        "visual_payload_inventory.csv": visual_inventory_rows,
        "completion.csv": completion_rows,
    }
    if calibration_reference_checks:
        output_rows["calibration_reference_checks.csv"] = calibration_reference_checks
    if evaluation_reference_checks:
        output_rows["evaluation_reference_checks.csv"] = evaluation_reference_checks
    if projection_rows:
        output_rows["projection_metrics.csv"] = projection_rows
    if projection_component_rows:
        output_rows["projection_component_metrics.csv"] = projection_component_rows
    for name, rows in output_rows.items():
        write_csv_with_paths(args.output_dir / name, rows)

    _assert_selector_bundle_unchanged(
        selector_path,
        selector_sha,
        selector_record["artifact_sha256"],
    )
    selector_bundle_revalidated = True

    evaluation_case_ids = tuple(sorted(cases_by_id))
    arms = ("baseline", "selected")
    calls = tuple(range(1, args.rollout_calls + 1))
    expected_call_keys = {
        (case_id, arm, call)
        for case_id in evaluation_case_ids
        for arm in arms
        for call in calls
    }
    call_inventory_pass = _row_inventory_exact(
        call_rows, expected_call_keys, ("case_id", "arm", "call")
    )
    completion_inventory_pass = _row_inventory_exact(
        completion_rows, expected_call_keys, ("case_id", "arm", "call")
    ) and all(bool(row["accepted"]) for row in completion_rows)
    expected_integral_keys = {
        (case_id, arm, call, component)
        for case_id in evaluation_case_ids
        for arm in arms
        for call in calls
        for component in COMPONENTS
    }
    correction_integral_inventory_pass = _row_inventory_exact(
        correction_integral_rows,
        expected_integral_keys,
        ("case_id", "arm", "call", "component"),
    )
    expected_case_summary_keys = {
        (case_id, arm) for case_id in evaluation_case_ids for arm in arms
    }
    case_summary_inventory_pass = _row_inventory_exact(
        case_summary_rows,
        expected_case_summary_keys,
        ("case_id", "arm"),
    )
    sequence_kinds = [
        ("baseline", "corrected_defect_on_own_recurrent_inputs"),
        ("selected", "corrected_defect_on_own_recurrent_inputs"),
    ]
    if not selected.is_zero:
        sequence_kinds.append(("selected", "base_defect_on_same_corrected_inputs"))
    expected_sequence_keys = {
        (case_id, arm, kind)
        for case_id in evaluation_case_ids
        for arm, kind in sequence_kinds
    }
    sequence_inventory_pass = _row_inventory_exact(
        sequence_summary_rows,
        expected_sequence_keys,
        ("case_id", "arm", "defect_kind"),
    )
    expected_sequence_time_keys = {
        (case_id, arm, kind, call)
        for case_id in evaluation_case_ids
        for arm, kind in sequence_kinds
        for call in calls
    }
    sequence_time_inventory_pass = _row_inventory_exact(
        sequence_time_rows,
        expected_sequence_time_keys,
        ("case_id", "arm", "defect_kind", "step"),
    )
    expected_lag_keys = {
        (case_id, arm, kind, lag)
        for case_id in evaluation_case_ids
        for arm, kind in sequence_kinds
        for lag in range(1, min(10, args.rollout_calls - 1) + 1)
    }
    lag_inventory_pass = _row_inventory_exact(
        lag_rows,
        expected_lag_keys,
        ("case_id", "arm", "defect_kind", "lag"),
    )
    expected_pod_keys = {
        (case_id, arm, kind, centering)
        for case_id in evaluation_case_ids
        for arm, kind in sequence_kinds
        for centering in ("uncentered", "centered")
    }
    pod_inventory_pass = _row_inventory_exact(
        pod_rows,
        expected_pod_keys,
        ("case_id", "arm", "defect_kind", "centering"),
    )
    base_budget_fields = (
        "base_defect_same_corrected_input",
        "correction",
        "corrected_defect",
        "cumulative_correction",
        "state_error",
    )
    expected_budget_keys = {
        (case_id, arm, call, field, component)
        for case_id in evaluation_case_ids
        for arm in arms
        for call in calls
        for field in base_budget_fields
        for component in COMPONENTS
    }
    if not selected.is_zero:
        expected_budget_keys.update(
            {
                (case_id, "selected", call, field, component)
                for case_id in evaluation_case_ids
                for call in calls
                for field in (
                    "correction_constant_mode",
                    "correction_nonconstant_modes",
                )
                for component in COMPONENTS
            }
        )
    budget_inventory_pass = _row_inventory_exact(
        budget_rows,
        expected_budget_keys,
        ("case_id", "arm", "call", "field", "component"),
    )
    expected_projection_keys = (
        set()
        if selected.is_zero
        else {
            (case_id, "selected", call)
            for case_id in evaluation_case_ids
            for call in calls
        }
    )
    projection_inventory_pass = _row_inventory_exact(
        projection_rows,
        expected_projection_keys,
        ("case_id", "arm", "call"),
    )
    expected_projection_component_keys = {
        (*key, component)
        for key in expected_projection_keys
        for component in COMPONENTS
    }
    projection_component_inventory_pass = _row_inventory_exact(
        projection_component_rows,
        expected_projection_component_keys,
        ("case_id", "arm", "call", "component"),
    )
    projection_status_pass = _row_inventory_exact(
        projection_status_rows,
        {(case_id,) for case_id in evaluation_case_ids},
        ("case_id",),
    ) and all(
        row["status"]
        == ("not_applicable_zero" if selected.is_zero else "applied_complete")
        for row in projection_status_rows
    )
    visual_inventory_pass = _row_inventory_exact(
        visual_inventory_rows,
        {(case_id,) for case_id in visual_ids},
        ("case_id",),
    )
    expected_case_contract_ids = set(active_calibration_ids) | set(
        active_evaluation_ids
    )
    case_contract_inventory_pass = _row_inventory_exact(
        case_contract_rows,
        {(case_id,) for case_id in expected_case_contract_ids},
        ("case_id",),
    )
    fold_ids = (
        (active_calibration_ids[0],)
        if args.smoke
        else tuple(sorted(active_calibration_ids))
    )
    selector_matrix_inventory_pass = _row_inventory_exact(
        selector_rows,
        {(candidate.key, case_id) for candidate in candidates for case_id in fold_ids},
        ("candidate", "case_id"),
    )
    selector_metric_names = {
        "final_state_error",
        "residual_rms",
        *(f"control::{name}" for name in expected_control_keys(args.family)),
    }
    selector_metric_inventory_pass = _row_inventory_exact(
        selector_metric_rows,
        {
            (candidate.key, case_id, metric)
            for candidate in candidates
            for case_id in fold_ids
            for metric in selector_metric_names
        },
        ("candidate", "case_id", "metric"),
    )
    selector_integral_contract_pass = all(
        bool(row.get("correction_integral_contract_pass")) for row in selector_rows
    )
    required_integral_rows = [
        row
        for row in correction_integral_rows
        if bool(row["integral_neutralization_required"])
    ]
    maximum_required_integral_closure = (
        max(
            abs(float(row["residual_scaled_physical_volume_mean"]))
            for row in required_integral_rows
        )
        if required_integral_rows
        else 0.0
    )
    correction_integral_contract_pass = all(
        bool(row["integral_closure_pass"]) for row in correction_integral_rows
    )

    maximum_recurrence_closure = _maximum_absolute(call_rows, "recurrence_closure_rms")
    maximum_growth_closure = _maximum_absolute(call_rows, "growth_closure")
    maximum_same_input_closure = _maximum_absolute(
        call_rows, "same_input_energy_closure"
    )
    maximum_pointwise_same_input_closure = _maximum_absolute(
        call_rows, "same_input_pointwise_closure_max"
    )
    if selected.is_zero:
        maximum_projection_inner = 0.0
        maximum_projection_closure = 0.0
        maximum_projection_reconstruction = 0.0
        maximum_non_type0_correction = 0.0
        maximum_coefficient_reconstruction = 0.0
        maximum_orthogonal_partition_change = 0.0
        maximum_excluded_partition_change = 0.0
        maximum_correction_projection_leakage = 0.0
        maximum_component_projection_inner = 0.0
        maximum_component_energy_closure = 0.0
        maximum_constant_nonconstant_energy_closure = 0.0
    else:
        maximum_projection_inner = _maximum_absolute(
            projection_rows, "maximum_parallel_orthogonal_inner"
        )
        maximum_projection_closure = _maximum_absolute(
            projection_rows, "maximum_energy_closure"
        )
        maximum_projection_reconstruction = _maximum_absolute(
            projection_rows, "maximum_reconstruction_error"
        )
        maximum_non_type0_correction = _maximum_absolute(
            projection_rows, "maximum_non_type0_correction"
        )
        maximum_coefficient_reconstruction = _maximum_absolute(
            projection_rows, "maximum_coefficient_reconstruction_error"
        )
        maximum_orthogonal_partition_change = _maximum_absolute(
            projection_rows, "maximum_orthogonal_partition_change"
        )
        maximum_excluded_partition_change = _maximum_absolute(
            projection_rows, "maximum_excluded_partition_change"
        )
        maximum_correction_projection_leakage = max(
            _maximum_absolute(projection_rows, "correction_orthogonal_energy"),
            _maximum_absolute(projection_rows, "correction_excluded_non_type0_energy"),
        )
        maximum_component_projection_inner = max(
            _maximum_absolute(
                projection_component_rows, "before_parallel_orthogonal_inner"
            ),
            _maximum_absolute(
                projection_component_rows, "after_parallel_orthogonal_inner"
            ),
            _maximum_absolute(
                projection_component_rows, "correction_parallel_orthogonal_inner"
            ),
        )
        maximum_component_energy_closure = max(
            _maximum_absolute(projection_component_rows, "before_energy_closure"),
            _maximum_absolute(projection_component_rows, "after_energy_closure"),
            _maximum_absolute(projection_component_rows, "correction_energy_closure"),
        )
        maximum_constant_nonconstant_energy_closure = _maximum_absolute(
            projection_component_rows,
            "constant_nonconstant_energy_closure",
        )
    maximum_visual_cumulative_closure = _maximum_absolute(
        visual_inventory_rows, "maximum_cumulative_closure"
    )
    maximum_visual_growth_replay = _maximum_absolute(
        visual_inventory_rows, "maximum_signed_growth_replay"
    )
    reference_binding_pass = bool(
        args.family != "dynamic_fv"
        or (
            len(calibration_reference_checks) == len(active_calibration_ids)
            and len(evaluation_reference_checks) == len(active_evaluation_ids)
        )
    )
    replay_pass = bool(bump_replay is None or bump_replay.get("passed"))
    zero_identity_pass = bool(
        not selected.is_zero
        or all(
            selected_by_case[key] is baseline_by_case[key] for key in baseline_by_case
        )
    )
    row_inventories_pass = bool(
        call_inventory_pass
        and completion_inventory_pass
        and correction_integral_inventory_pass
        and case_summary_inventory_pass
        and sequence_inventory_pass
        and sequence_time_inventory_pass
        and lag_inventory_pass
        and pod_inventory_pass
        and budget_inventory_pass
        and projection_inventory_pass
        and projection_component_inventory_pass
        and projection_status_pass
        and visual_inventory_pass
        and case_contract_inventory_pass
        and selector_matrix_inventory_pass
        and selector_metric_inventory_pass
        and selector_integral_contract_pass
    )
    core_contract_pass = bool(
        calibration_hook["passed"]
        and evaluation_hook["passed"]
        and reference_binding_pass
        and replay_pass
        and selector_bundle_revalidated
        and zero_identity_pass
        and row_inventories_pass
        and maximum_recurrence_closure <= 2.0e-5
        and maximum_growth_closure <= 2.0e-5
        and maximum_same_input_closure <= 1.0e-10
        and maximum_pointwise_same_input_closure <= 1.0e-12
        and maximum_projection_inner <= 1.0e-10
        and maximum_component_projection_inner <= 1.0e-10
        and maximum_projection_closure <= 1.0e-10
        and maximum_component_energy_closure <= 1.0e-10
        and maximum_constant_nonconstant_energy_closure <= 1.0e-10
        and maximum_projection_reconstruction <= 1.0e-10
        and maximum_non_type0_correction <= 1.0e-12
        and maximum_coefficient_reconstruction <= 1.0e-10
        and maximum_orthogonal_partition_change <= 1.0e-10
        and maximum_excluded_partition_change <= 1.0e-12
        and maximum_correction_projection_leakage <= 1.0e-20
        and maximum_visual_cumulative_closure <= 2.0e-5
        and maximum_visual_growth_replay <= 1.0e-10
        and correction_integral_contract_pass
        and maximum_required_integral_closure <= INTEGRAL_CLOSURE_LIMIT
        and all(result.complete for result in baseline_by_case.values())
        and all(result.complete for result in selected_by_case.values())
    )
    status = (
        "smoke_complete"
        if args.smoke and core_contract_pass
        else (
            "smoke_failed"
            if args.smoke
            else "complete" if core_contract_pass else "failed_contract"
        )
    )
    provenance["loaded_project_source_sha256"] = _loaded_project_source_hashes()
    output_hashes = {
        str(path.relative_to(args.output_dir)).replace("\\", "/"): sha256_file(path)
        for path in sorted(args.output_dir.rglob("*"))
        if path.is_file() and path.name != "summary.json"
    }
    summary = {
        "schema": SCHEMA,
        "experiment_contract": args.experiment_contract,
        "experiment_id": (
            "D075"
            if args.experiment_contract == "d075_integral_neutral"
            else "D074-A"
        ),
        "status": status,
        "contract_checks_passed": core_contract_pass,
        "scientific_interpretation_allowed": bool(
            core_contract_pass and not args.smoke
        ),
        "family": args.family,
        "args": jsonable_args(args),
        "population": {
            "split": "validation",
            "calibration_case_ids": list(active_calibration_ids),
            "evaluation_case_ids": list(active_evaluation_ids),
            "sealed_populations_accessed": False,
            "adaptive_reuse_of_d071_evaluation": True,
            "adaptive_reuse_of_open_validation": True,
        },
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "normalization_digest": checkpoint.get("normalization_digest"),
        "boundary_policy": "model_all_nodes raw recurrence; base policy unchanged",
        "selector": selector_record,
        "selector_sha256": selector_sha,
        "promotion": promotion,
        "checks": {
            "calibration_hook_equivalence_pass": calibration_hook["passed"],
            "evaluation_hook_equivalence_pass": evaluation_hook["passed"],
            "reference_binding_pass": reference_binding_pass,
            "bump_replay_pass": replay_pass,
            "selector_bundle_revalidated": selector_bundle_revalidated,
            "row_inventories_pass": row_inventories_pass,
            "selector_matrix_inventory_pass": selector_matrix_inventory_pass,
            "selector_metric_inventory_pass": selector_metric_inventory_pass,
            "selector_integral_contract_pass": selector_integral_contract_pass,
            "evaluation_call_inventory_pass": call_inventory_pass,
            "completion_inventory_pass": completion_inventory_pass,
            "correction_integral_inventory_pass": (
                correction_integral_inventory_pass
            ),
            "correction_integral_contract_pass": (
                correction_integral_contract_pass
            ),
            "maximum_required_integral_closure": (
                maximum_required_integral_closure
            ),
            "case_summary_inventory_pass": case_summary_inventory_pass,
            "sequence_inventory_pass": sequence_inventory_pass,
            "sequence_time_inventory_pass": sequence_time_inventory_pass,
            "lag_inventory_pass": lag_inventory_pass,
            "pod_inventory_pass": pod_inventory_pass,
            "budget_inventory_pass": budget_inventory_pass,
            "projection_inventory_pass": projection_inventory_pass,
            "projection_component_inventory_pass": (
                projection_component_inventory_pass
            ),
            "projection_status_pass": projection_status_pass,
            "visual_inventory_pass": visual_inventory_pass,
            "case_contract_inventory_pass": case_contract_inventory_pass,
            "maximum_recurrence_closure_rms": maximum_recurrence_closure,
            "maximum_growth_closure_absolute": maximum_growth_closure,
            "maximum_same_input_energy_closure": maximum_same_input_closure,
            "maximum_same_input_pointwise_closure": (
                maximum_pointwise_same_input_closure
            ),
            "maximum_projection_orthogonality": maximum_projection_inner,
            "maximum_component_projection_orthogonality": (
                maximum_component_projection_inner
            ),
            "maximum_projection_energy_closure": maximum_projection_closure,
            "maximum_component_energy_closure": maximum_component_energy_closure,
            "maximum_constant_nonconstant_energy_closure": (
                maximum_constant_nonconstant_energy_closure
            ),
            "maximum_projection_reconstruction_error": (
                maximum_projection_reconstruction
            ),
            "maximum_non_type0_correction": maximum_non_type0_correction,
            "maximum_coefficient_reconstruction_error": (
                maximum_coefficient_reconstruction
            ),
            "maximum_orthogonal_partition_change": (
                maximum_orthogonal_partition_change
            ),
            "maximum_excluded_partition_change": (maximum_excluded_partition_change),
            "maximum_correction_projection_leakage": (
                maximum_correction_projection_leakage
            ),
            "maximum_visual_cumulative_closure": (maximum_visual_cumulative_closure),
            "maximum_visual_signed_growth_replay": maximum_visual_growth_replay,
            "zero_reuses_baseline_object": zero_identity_pass,
            "core_contract_pass": core_contract_pass,
        },
        "claim_boundary": {
            "method": (
                "offline call-indexed low-rank bias with optional target-free "
                "correction-integral projection; not state feedback"
            ),
            "dynamic": "audited physical volumes; base PCNO is not conservative",
            "bump": "native graph and proxy weights only; no resolution claim",
            "assimilation": "not implemented",
            "confirmation": "adaptive open-population development evidence only",
        },
        "provenance": provenance,
        "bump_open_validation_replay": bump_replay,
        "calibration_hook_equivalence": calibration_hook,
        "evaluation_hook_equivalence": evaluation_hook,
        "runtime": runtime_environment(device),
        "git": git_state(),
        "row_counts": {
            "selector_rows.csv": len(selector_rows),
            "selector_summary.csv": len(selector_summary_rows),
            "selector_metrics.csv": len(selector_metric_rows),
            **{name: len(rows) for name, rows in output_rows.items()},
        },
        "elapsed_seconds": perf_counter() - started,
        "output_hashes": output_hashes,
    }
    write_json(args.output_dir / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    summary = run(parse_args(argv))
    print(
        f"{summary['experiment_id']} {summary['family']} "
        f"status={summary['status']} "
        f"selected={summary['selector']['selected']['key']} "
        f"promoted={summary['promotion']['passed']}",
        flush=True,
    )
    return 0 if summary["status"] in {"complete", "smoke_complete"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
