#!/usr/bin/env python3
"""Evaluate the serious conservative-residual PCNO on held-out Euler shards.

The evaluator is deliberately narrower than D013.  It runs the frozen model on
every selected trajectory under the checkpoint's native autonomous recurrence
and compares it with persistence.  Historical ``model_all_nodes`` checkpoints
may additionally run exactly one predeclared diagnostic counterfactual.  A
checkpoint trained with the causal nodal boundary contract is evaluated only
under that same native contract; it is not relabeled as a raw recurrence.
The evaluator never trains or edits the checkpoint.

Reconstructed vertex weights are reported only as quadrature proxies.  The CPG
bump shards do not contain the finite-volume geometry needed for physical
conservation or flux claims.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utility.time_dependent_no.cpg_mesh_contract import (  # noqa: E402
    BoundaryStencil,
    apply_torch_boundary_policy,
    build_torch_boundary_policy,
)
from utility.time_dependent_no.cpg_release import release_rollout_metrics  # noqa: E402
from utility.time_dependent_no.euler2d import PRIMITIVE_NAMES  # noqa: E402
from utility.time_dependent_no.euler2d_metrics import (  # noqa: E402
    front_centroid_distance,
    front_distance_metrics,
    front_overlap_metrics,
    median_edge_length,
    shock_front_masks,
    shock_smearing_metrics,
)
from utility.time_dependent_no.pcno_euler2d import (  # noqa: E402
    Euler2DNormalization,
    PCNOEuler2DResidual,
    PCNOEuler2DShardStore,
    apply_causal_boundary_conservative_batch,
    build_graph_causal_boundary_policy,
    conservative_to_primitive_torch,
    parameter_count,
    primitive_to_conservative_torch,
)
from utility.time_dependent_no.pcno_ripple_diagnostics import (  # noqa: E402
    conservative_to_primitive_raw,
    induced_subgraph,
    node_highpass_amplitude,
    normalized_node_weights,
    raw_admissibility_summary,
    trace_pcno_branches,
    weighted_relative_l2_numpy,
)

EVALUATION_SCHEMA = "pcno_euler2d_official_rollout_v1"
CHECKPOINT_SCHEMA_VERSION = 4
NO_COUNTERFACTUAL = "none"
POINTWISE_TAIL_GAIN = 0.75
POINTWISE_COUNTERFACTUAL = "pointwise_tail_gain_0p75"
CAUSAL_BOUNDARY_COUNTERFACTUAL = "causal_nodal_boundary"
FIXED_INFLOW_COUNTERFACTUAL = "fixed_freestream_inflow"
NATIVE_CAUSAL_BOUNDARY_SCOPE = "native_causal_nodal_physical"
CAUSAL_BOUNDARY_MODE = "causal_nodal_physical"
MINIMUM_CHANGE_BOUNDARY_MODE = "minimum_change_nodal_physical"
POINTWISE_VARIANT = "pcno_pointwise_tail_gain_0p75"
CAUSAL_BOUNDARY_VARIANT = "pcno_causal_nodal_boundary_sensitivity"
FIXED_INFLOW_VARIANT = "pcno_fixed_freestream_inflow_sensitivity"
DEFAULT_ENDPOINT_CALLS = (1, 5, 10, 20, 40, 79)
CORE_MANIFEST_FIELDS = (
    "schema_version",
    "gamma",
    "dt",
    "state_convention",
    "coordinate_convention",
    "weight_provenance",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--training-data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trajectory-keys", nargs="*", default=None)
    parser.add_argument("--expected-trajectory-count", type=int, default=20)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=79)
    parser.add_argument(
        "--endpoint-calls",
        type=int,
        nargs="+",
        default=list(DEFAULT_ENDPOINT_CALLS),
    )
    parser.add_argument("--shock-quantile", type=float, default=0.9)
    parser.add_argument(
        "--counterfactual",
        choices=(
            NO_COUNTERFACTUAL,
            POINTWISE_COUNTERFACTUAL,
            CAUSAL_BOUNDARY_COUNTERFACTUAL,
            FIXED_INFLOW_COUNTERFACTUAL,
        ),
        default=POINTWISE_COUNTERFACTUAL,
        help="Run no counterfactual or exactly one frozen diagnostic counterfactual.",
    )
    parser.add_argument(
        "--boundary-mesh-audit",
        type=Path,
        default=None,
        help=(
            "Line-2 mesh-audit summary required by the causal nodal boundary "
            "counterfactual."
        ),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(
            f"output directory already exists; choose a new directory: {args.output_dir}"
        )
    if args.expected_trajectory_count < 1:
        raise ValueError("--expected-trajectory-count must be positive")
    if args.start_frame < 0 or args.num_steps < 1:
        raise ValueError("start frame must be nonnegative and num steps positive")
    endpoint_calls = sorted(set(int(value) for value in args.endpoint_calls))
    if not endpoint_calls or endpoint_calls[0] < 1:
        raise ValueError("endpoint calls must be positive")
    args.endpoint_calls = [value for value in endpoint_calls if value <= args.num_steps]
    if not args.endpoint_calls:
        raise ValueError("no endpoint call lies inside the requested rollout")
    if not 0.0 < args.shock_quantile < 1.0:
        raise ValueError("--shock-quantile must lie in (0,1)")
    if args.counterfactual in {
        CAUSAL_BOUNDARY_COUNTERFACTUAL,
        FIXED_INFLOW_COUNTERFACTUAL,
    }:
        if args.boundary_mesh_audit is None:
            raise ValueError(
                "--boundary-mesh-audit is required for boundary counterfactuals"
            )
    elif args.boundary_mesh_audit is not None:
        raise ValueError(
            "--boundary-mesh-audit is only valid for boundary counterfactuals"
        )


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    serialized_rows = []
    for row in rows:
        serialized = {}
        for key, value in row.items():
            safe_value = json_safe(value)
            if isinstance(safe_value, (dict, list)):
                safe_value = json.dumps(
                    safe_value, sort_keys=True, separators=(",", ":")
                )
            serialized[str(key)] = safe_value
        serialized_rows.append(serialized)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(serialized_rows)


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    required = {
        "checkpoint_schema_version",
        "model_state",
        "model_config",
        "normalization",
        "data_manifest_digest",
        "step_stride",
        "config_digest",
        "boundary_mode",
        "raw_recurrence",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"checkpoint is missing required fields: {missing}")
    if int(checkpoint["checkpoint_schema_version"]) != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported PCNO checkpoint schema")
    boundary_mode = str(checkpoint["boundary_mode"])
    if boundary_mode not in {
        "model_all_nodes",
        CAUSAL_BOUNDARY_MODE,
        MINIMUM_CHANGE_BOUNDARY_MODE,
    }:
        raise ValueError(f"unsupported checkpoint boundary mode: {boundary_mode}")
    expected_raw = boundary_mode == "model_all_nodes"
    if checkpoint["raw_recurrence"] is not expected_raw:
        raise ValueError("checkpoint recurrence declaration contradicts boundary mode")
    if boundary_mode in {CAUSAL_BOUNDARY_MODE, MINIMUM_CHANGE_BOUNDARY_MODE}:
        boundary_contract = checkpoint.get("boundary_contract")
        if not isinstance(boundary_contract, Mapping):
            raise ValueError("hard-boundary checkpoint lacks its boundary contract")
        if str(boundary_contract.get("mode")) != boundary_mode:
            raise ValueError("checkpoint boundary contract mode is inconsistent")
    return dict(checkpoint)


def build_model(
    checkpoint: Mapping[str, Any], device: torch.device
) -> PCNOEuler2DResidual:
    config = checkpoint["model_config"]
    model = PCNOEuler2DResidual(
        normalization=Euler2DNormalization.from_mapping(checkpoint["normalization"]),
        k_max=int(config["k_max"]),
        domain_lengths=config["domain_lengths"],
        layers=config["layers"],
        fc_dim=int(config["fc_dim"]),
        nmeasures=int(config["nmeasures"]),
        zero_initialize=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model


def _trajectory_metadata(
    store: PCNOEuler2DShardStore, key: str
) -> dict[str, Any] | None:
    path = store.root / str(store.entry(key)["folder"]) / "metadata.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def preprocessing_contract_audit(
    checkpoint: Mapping[str, Any],
    training_store: PCNOEuler2DShardStore,
    test_store: PCNOEuler2DShardStore,
) -> dict[str, Any]:
    """Close fields that can be verified and name legacy omissions explicitly."""

    incompatible: list[str] = []
    if checkpoint["data_manifest_digest"] != training_store.manifest_digest:
        incompatible.append("checkpoint_training_manifest_digest")
    for field in CORE_MANIFEST_FIELDS:
        if training_store.manifest.get(field) != test_store.manifest.get(field):
            incompatible.append(f"train_test_{field}")
    normalization = checkpoint["normalization"]
    if not math.isclose(
        float(normalization["gamma"]),
        float(test_store.manifest["gamma"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        incompatible.append("checkpoint_test_gamma")

    # Train/test key strings can overlap, so collect each store explicitly.
    metadata = [
        value
        for store in (training_store, test_store)
        for key in store.keys
        if (value := _trajectory_metadata(store, key)) is not None
    ]
    gradient_rconds = sorted(
        {
            float(value["gradient_rcond"])
            for value in metadata
            if "gradient_rcond" in value
        }
    )
    if len(gradient_rconds) > 1:
        incompatible.append("gradient_rcond_varies_across_shards")
    if incompatible:
        raise ValueError(f"incompatible PCNO preprocessing contract: {incompatible}")

    return {
        "status": "compatible_with_declared_legacy_gaps",
        "verified": {
            "checkpoint_training_manifest_digest": training_store.manifest_digest,
            "train_test_manifest_fields": list(CORE_MANIFEST_FIELDS),
            "state_and_target": (
                "conservative current state and dataset next state at checkpoint stride"
            ),
            "normalization": "fixed checkpoint training-set normalization",
            "positivity_transform": "none",
            "inverse_transform": "analytic conservative-to-primitive with checkpoint gamma",
            "mesh_node_order": test_store.manifest["coordinate_convention"],
            "quadrature": test_store.manifest["weight_provenance"],
            "boundary_mode": checkpoint["boundary_mode"],
            "temporal_stride": int(checkpoint["step_stride"]),
            "model_config": checkpoint["model_config"],
            "gradient_rcond": gradient_rconds[0] if len(gradient_rconds) == 1 else None,
            "target_source": "held-out shard states, never another model output",
        },
        "inferred": {
            "dataset_split": (
                "caller-designated held-out shard directory; per-trajectory state and "
                "geometry digests are present"
            ),
            "reconstruction_request": (
                "legacy manifest records each realized reconstruction method but not "
                "the original auto/face/cycles CLI choice"
            ),
        },
        "missing": [
            "physical units",
            "raw source-HDF5 content digest",
            "original reconstruction min/max-face-node CLI values",
            "validated physical control volumes",
            "physical face measures and outward normals",
            "oriented physical face connectivity and boundary exchange",
            "cumulative accepted-substep reference face impulses",
        ],
        "incompatible": [],
    }


def select_trajectory_keys(
    args: argparse.Namespace, store: PCNOEuler2DShardStore
) -> list[str]:
    keys = (
        store.keys
        if not args.trajectory_keys
        else [str(k) for k in args.trajectory_keys]
    )
    missing = sorted(set(keys) - set(store.keys))
    if missing:
        raise KeyError(f"test shards are missing trajectory keys: {missing}")
    if len(keys) != args.expected_trajectory_count:
        raise ValueError(
            f"selected {len(keys)} trajectories; expected {args.expected_trajectory_count}"
        )
    return keys


def load_boundary_mesh_audit(path: Path, keys: Sequence[str]) -> dict[str, Any]:
    """Load the frozen Line-2 mesh contract without reconstructing geometry."""

    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema": "cpg_bump_mesh_provenance_audit_v1",
        "status": "valid",
        "boundary_geometry": "verified_all_selected_cases",
        "graph_mesh_identity": "verified_all_selected_cases",
        "legal_boundary_counterfactual": (
            "enabled_causal_nodal_projection_not_exact_dg_boundary_replay"
        ),
    }
    for field, expected in required.items():
        if payload.get(field) != expected:
            raise ValueError(f"mesh audit {field} does not equal {expected!r}")
    cases = {str(case["trajectory_key"]): case for case in payload.get("cases", [])}
    missing = sorted(set(keys) - set(cases))
    if missing:
        raise ValueError(f"mesh audit is missing trajectories: {missing}")
    payload["_path"] = path
    payload["_sha256"] = sha256_file(path)
    payload["_cases_by_key"] = cases
    return payload


def _canonical_edges(edges: np.ndarray) -> np.ndarray:
    canonical = np.sort(np.asarray(edges, dtype=np.int64).reshape(-1, 2), axis=1)
    order = np.lexsort((canonical[:, 1], canonical[:, 0]))
    return canonical[order]


def build_validated_boundary_policy(
    audit: Mapping[str, Any],
    store: PCNOEuler2DShardStore,
    key: str,
    *,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind one audited Line-2 boundary artifact to one PCNO shard exactly."""

    record = audit["_cases_by_key"][key]
    audit_path = Path(audit["_path"]).resolve()
    geometry_path = (audit_path.parent / str(record["geometry_artifact"])).resolve()
    try:
        geometry_path.relative_to(audit_path.parent)
    except ValueError as exc:
        raise ValueError(
            "mesh-audit geometry artifact escapes its audit directory"
        ) from exc
    if not geometry_path.is_file():
        raise FileNotFoundError(geometry_path)

    required_arrays = {
        "schema",
        "trajectory_key",
        "raw_case_id",
        "mach",
        "pos",
        "primal_edges",
        "node_type",
        "node_normal",
        "node_boundary_normal_coherence",
        "stencil_target_nodes",
        "stencil_target_rows",
        "stencil_source_nodes",
        "stencil_weights",
    }
    with np.load(geometry_path, allow_pickle=False) as arrays:
        missing = sorted(required_arrays - set(arrays.files))
        if missing:
            raise ValueError(f"boundary geometry artifact is missing arrays: {missing}")
        if arrays["schema"].item() != "cpg_bump_mesh_contract_v1":
            raise ValueError("unsupported boundary geometry schema")
        if str(arrays["trajectory_key"].item()) != key:
            raise ValueError("boundary geometry trajectory key does not match shard")
        if int(arrays["raw_case_id"].item()) != int(record["raw_case_id"]):
            raise ValueError("boundary geometry raw case does not match mesh audit")

        positions = np.array(store.array(key, "nodes"), copy=False)
        edges = np.array(store.array(key, "edges"), copy=False)
        node_type = np.array(store.array(key, "node_type"), copy=False).reshape(-1)
        if arrays["pos"].shape != positions.shape or not np.allclose(
            arrays["pos"], positions, rtol=0.0, atol=5.0e-6
        ):
            raise ValueError(
                "boundary geometry positions do not match PCNO shard order"
            )
        if not np.array_equal(
            _canonical_edges(arrays["primal_edges"]), _canonical_edges(edges)
        ):
            raise ValueError("boundary geometry edges do not match PCNO shard")
        if not np.array_equal(arrays["node_type"].reshape(-1), node_type):
            raise ValueError("boundary geometry node types do not match PCNO shard")
        mach = float(store.entry(key)["mach"])
        if not math.isclose(
            float(arrays["mach"].item()), mach, rel_tol=0.0, abs_tol=2.0e-10
        ):
            raise ValueError("boundary geometry Mach does not match PCNO shard")

        stencil = BoundaryStencil(
            target_nodes=np.array(arrays["stencil_target_nodes"], copy=True),
            target_rows=np.array(arrays["stencil_target_rows"], copy=True),
            source_nodes=np.array(arrays["stencil_source_nodes"], copy=True),
            weights=np.array(arrays["stencil_weights"], copy=True),
            fallback_target_count=int(
                record["boundary"]["boundary_stencil_fallback_target_count"]
            ),
        )
        if stencil.target_nodes.size != int(
            record["boundary"]["boundary_stencil_target_count"]
        ) or stencil.source_nodes.size != int(
            record["boundary"]["boundary_stencil_entry_count"]
        ):
            raise ValueError("boundary stencil sizes do not match mesh audit")
        config = dict(record["config"])
        if not math.isclose(
            float(config["gamma"]),
            float(store.manifest["gamma"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("boundary-policy gamma does not match PCNO shard")
        policy = build_torch_boundary_policy(
            torch=torch,
            device=device,
            node_type=node_type,
            node_normal=np.array(arrays["node_normal"], copy=True),
            wall_normal_coherence=np.array(
                arrays["node_boundary_normal_coherence"], copy=True
            ),
            stencil=stencil,
            mach=mach,
            config=config,
        )

    metadata = {
        "trajectory_key": key,
        "raw_case_id": int(record["raw_case_id"]),
        "geometry_artifact": str(record["geometry_artifact"]),
        "geometry_artifact_sha256": sha256_file(geometry_path),
        "mesh_audit_sha256": audit["_sha256"],
        "target_count": int(stencil.target_nodes.size),
        "entry_count": int(stencil.source_nodes.size),
        "fallback_target_count": int(stencil.fallback_target_count),
        "sharp_wall_corner_count": int(record["boundary"]["sharp_wall_corner_count"]),
        "policy": (
            "fixed freestream inflow, current-interior slip wall, and "
            "current-interior supersonic outflow"
        ),
        "exact_dg_boundary_replay": False,
        "future_reference_boundary_values": False,
    }
    return policy, metadata


def apply_causal_boundary_conservative(
    state: torch.Tensor,
    policy: Mapping[str, Any],
    *,
    gamma: float,
    scope: str,
) -> torch.Tensor:
    """Apply a registered primitive nodal policy to one conservative state."""

    if state.ndim != 3 or state.shape[0] != 1 or state.shape[-1] != 4:
        raise ValueError("causal boundary policy expects state shape [1,N,4]")
    if scope == NATIVE_CAUSAL_BOUNDARY_SCOPE:
        return apply_causal_boundary_conservative_batch(state, policy, gamma=gamma)
    primitive = conservative_to_primitive_torch(state[0], gamma=gamma)
    if scope == CAUSAL_BOUNDARY_COUNTERFACTUAL:
        bounded = apply_torch_boundary_policy(torch, primitive, policy)
    elif scope == FIXED_INFLOW_COUNTERFACTUAL:
        bounded = primitive.clone()
        bounded[policy["inflow_nodes"]] = policy["freestream"].to(dtype=bounded.dtype)
    else:
        raise ValueError(f"unsupported boundary scope: {scope}")
    return primitive_to_conservative_torch(bounded, gamma=gamma).unsqueeze(0)


def branch_gain_profiles(model: PCNOEuler2DResidual) -> tuple[dict, dict]:
    layer_count = len(model.backbone.ws)
    identity = {
        (layer, branch): 1.0
        for layer in range(layer_count)
        for branch in ("spectral", "pointwise", "differential")
    }
    candidate = {
        (layer, "pointwise"): POINTWISE_TAIL_GAIN for layer in range(1, layer_count)
    }
    return identity, candidate


@torch.no_grad()
def model_call(
    model: PCNOEuler2DResidual,
    sample: Mapping[str, torch.Tensor],
    current: torch.Tensor,
    *,
    branch_gains: Mapping[tuple[int, str], float] | None = None,
) -> torch.Tensor:
    if branch_gains is None:
        return model(
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
    model_input = model.normalized_input(
        current,
        nodes=sample["nodes"],
        node_rhos=sample["node_rhos"],
        node_type=sample["node_type"],
        mach=sample["mach"],
    )
    normalized_residual, _ = trace_pcno_branches(
        model.backbone,
        model_input,
        (
            sample["node_mask"],
            sample["nodes"],
            sample["node_weights"],
            sample["directed_edges"],
            sample["edge_gradient_weights"],
        ),
        branch_gains=branch_gains,
        collect_summaries=False,
    )
    return (current + normalized_residual * model.residual_scale) * sample["node_mask"]


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def rollout_variant(
    model: PCNOEuler2DResidual,
    store: PCNOEuler2DShardStore,
    key: str,
    *,
    start_frame: int,
    num_steps: int,
    step_stride: int,
    device: torch.device,
    branch_gains: Mapping[tuple[int, str], float] | None,
    boundary_policy: Mapping[str, Any] | None = None,
    boundary_scope: str = CAUSAL_BOUNDARY_COUNTERFACTUAL,
) -> dict[str, Any]:
    states = store.states(key)
    final_index = start_frame + num_steps * step_stride
    if final_index >= states.shape[0]:
        raise ValueError(
            f"trajectory {key} has {states.shape[0]} frames; target index "
            f"{final_index} is unavailable"
        )
    sample = store.tensor_sample(
        key, start_frame, step_stride=step_stride, device=device
    )
    current = sample["current"]
    predictions: list[np.ndarray] = []
    call_seconds: list[float] = []
    failure_cause = "completed"
    failure_call: int | None = None
    failed_proposal: np.ndarray | None = None
    minimums = {
        "min_density": math.inf,
        "min_internal_energy": math.inf,
        "min_pressure": math.inf,
    }
    for call_index in range(1, num_steps + 1):
        synchronize(device)
        start = perf_counter()
        model_current = current
        if boundary_policy is not None:
            model_current = apply_causal_boundary_conservative(
                current, boundary_policy, gamma=model.gamma, scope=boundary_scope
            )
        proposal = model_call(model, sample, model_current, branch_gains=branch_gains)
        if boundary_policy is not None:
            proposal = apply_causal_boundary_conservative(
                proposal, boundary_policy, gamma=model.gamma, scope=boundary_scope
            )
        synchronize(device)
        call_seconds.append(perf_counter() - start)
        proposal_np = proposal[0].float().cpu().numpy()
        admissibility = raw_admissibility_summary(proposal_np, gamma=model.gamma)
        for name in minimums:
            value = admissibility[name]
            if value is not None:
                minimums[name] = min(minimums[name], float(value))
        if not admissibility["all_finite"]:
            failure_cause = "nonfinite_state"
        elif not admissibility["all_admissible"]:
            failure_cause = "inadmissible_state"
        if failure_cause != "completed":
            failure_call = call_index
            failed_proposal = proposal_np.copy()
            break
        predictions.append(proposal_np.copy())
        current = proposal
    valid_length = len(predictions)
    prediction_array = (
        np.asarray(predictions, dtype=np.float32)
        if predictions
        else np.empty((0, states.shape[1], states.shape[2]), dtype=np.float32)
    )
    return {
        "predictions": prediction_array,
        "valid_length": valid_length,
        "completed": valid_length == num_steps,
        "failure_cause": failure_cause,
        "failure_call": failure_call,
        "failed_proposal": failed_proposal,
        "call_seconds": call_seconds,
        **{
            name: None if not math.isfinite(value) else float(value)
            for name, value in minimums.items()
        },
    }


def _dilate_mask(mask: np.ndarray, edges: np.ndarray, hops: int = 1) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    edge_index = np.asarray(edges, dtype=np.int64)
    for _ in range(hops):
        expanded = result.copy()
        left = edge_index[:, 0]
        right = edge_index[:, 1]
        np.logical_or.at(expanded, left, result[right])
        np.logical_or.at(expanded, right, result[left])
        result = expanded
    return result


def failure_location_fields(
    failed_proposal: np.ndarray | None,
    node_type: np.ndarray,
    *,
    gamma: float,
) -> dict[str, Any]:
    """Locate the strongest raw Euler-admissibility violation, if one exists."""

    empty = {
        "failure_node_index": None,
        "failure_node_type": None,
        "failure_quantity": None,
        "failure_quantity_value": None,
    }
    if failed_proposal is None:
        return empty
    state = np.asarray(failed_proposal, dtype=np.float64)
    rho = state[:, 0]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        internal = state[:, 3] - 0.5 * (state[:, 1] ** 2 + state[:, 2] ** 2) / rho
        pressure = (gamma - 1.0) * internal
    quantities = {
        "density": rho,
        "internal_energy": internal,
        "pressure": pressure,
    }
    nonfinite = np.flatnonzero(~np.isfinite(state).all(axis=1))
    if nonfinite.size:
        index = int(nonfinite[0])
        quantity = "nonfinite_components"
        value = None
    else:
        candidates = [
            (float(np.min(values)), name, int(np.argmin(values)))
            for name, values in quantities.items()
        ]
        value, quantity, index = min(candidates, key=lambda item: item[0])
    types = np.asarray(node_type).reshape(-1)
    return {
        "failure_node_index": index,
        "failure_node_type": int(types[index]),
        "failure_quantity": quantity,
        "failure_quantity_value": value,
    }


def _scalar(value: Any) -> float | None:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("expected a scalar diagnostic")
    numeric = float(array.reshape(-1)[0])
    return numeric if math.isfinite(numeric) else None


def endpoint_diagnostics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    positions: np.ndarray,
    edges: np.ndarray,
    node_type: np.ndarray,
    proxy_weights: np.ndarray,
    component_scale: np.ndarray,
    gamma: float,
    shock_quantile: float,
) -> dict[str, Any]:
    pred_primitive = conservative_to_primitive_raw(prediction, gamma=gamma)
    target_primitive = conservative_to_primitive_raw(target, gamma=gamma)
    interior = np.asarray(node_type).reshape(-1) == 0
    if np.count_nonzero(interior) < 3:
        interior = np.ones_like(interior)
    fronts = shock_front_masks(
        pred_primitive,
        target_primitive,
        edges,
        quantile=shock_quantile,
        node_mask=interior,
    )
    overlap = front_overlap_metrics(fronts["prediction_mask"], fronts["target_mask"])
    distance = front_distance_metrics(
        fronts["prediction_mask"], fronts["target_mask"], positions
    )
    pred_interior, interior_edges, _ = induced_subgraph(
        pred_primitive, edges, np.ones(prediction.shape[0]), interior
    )
    target_interior, _, _ = induced_subgraph(
        target_primitive, edges, np.ones(prediction.shape[0]), interior
    )
    smearing = shock_smearing_metrics(
        pred_interior, target_interior, interior_edges, scalar_index=3
    )

    target_front = _dilate_mask(fronts["target_mask"], edges, hops=1)
    smooth = interior & ~target_front
    smooth_contract = "target_pressure_front_dilated_one_hop"
    if not np.any(smooth):
        smooth = interior & ~fronts["target_mask"]
        smooth_contract = "target_pressure_front_without_dilation_fallback"
    scaled_error = (prediction - target) / component_scale.reshape(1, -1)
    highpass = node_highpass_amplitude(scaled_error, edges)
    proxy_mass = normalized_node_weights(proxy_weights, name="endpoint high-pass")
    equal_mass = np.full(prediction.shape[0], 1.0 / prediction.shape[0])

    def smooth_energy(mass: np.ndarray) -> float:
        selected = mass * smooth
        return float(np.sum(selected * np.square(highpass)) / np.sum(selected))

    thickness_ratio = _scalar(smearing["thickness_ratio"])
    strength_ratio = _scalar(smearing["strength_ratio"])

    def log_error(value: float | None) -> float | None:
        if value is None or value <= 0.0:
            return None
        return abs(math.log(value))

    error_energy = proxy_mass * np.sum(np.square(scaled_error), axis=-1)
    return {
        "scaled_relative_l2_reconstructed_weight_proxy": weighted_relative_l2_numpy(
            prediction, target, proxy_weights, component_scale
        ),
        "scaled_relative_l2_equal_node_proxy": weighted_relative_l2_numpy(
            prediction, target, np.ones(prediction.shape[0]), component_scale
        ),
        "smooth_region_contract": smooth_contract,
        "smooth_region_node_fraction": float(np.mean(smooth)),
        "smooth_highpass_energy_reconstructed_weight_proxy": smooth_energy(proxy_mass),
        "smooth_highpass_energy_equal_node_proxy": smooth_energy(equal_mass),
        "scaled_error_energy_fraction_target_front_dilated_proxy": float(
            error_energy[target_front].sum() / max(float(error_energy.sum()), 1e-30)
        ),
        "front_iou": _scalar(overlap["iou"]),
        "front_symmetric_chamfer": _scalar(distance["symmetric_chamfer_mean"]),
        "front_centroid_distance": _scalar(
            front_centroid_distance(
                fronts["prediction_mask"], fronts["target_mask"], positions
            )
        ),
        "shock_thickness_ratio": thickness_ratio,
        "shock_strength_ratio": strength_ratio,
        "shock_thickness_log_error": log_error(thickness_ratio),
        "shock_strength_log_error": log_error(strength_ratio),
    }


def _primitive_rmse_rows(
    prediction: np.ndarray,
    target: np.ndarray,
    node_type: np.ndarray,
) -> list[dict[str, float | None]]:
    masks = {
        "all": np.ones(prediction.shape[1], dtype=bool),
        "normal": np.asarray(node_type).reshape(-1) == 0,
        "boundary": np.asarray(node_type).reshape(-1) != 0,
    }
    rows = [{} for _ in range(prediction.shape[0])]
    for mask_name, mask in masks.items():
        if not np.any(mask):
            for row in rows:
                for name in PRIMITIVE_NAMES:
                    row[f"{mask_name}_{name}_rmse"] = None
            continue
        mse = np.mean(np.square(prediction[:, mask] - target[:, mask]), axis=1)
        for call_index, values in enumerate(np.sqrt(mse)):
            for name, value in zip(PRIMITIVE_NAMES, values, strict=True):
                rows[call_index][f"{mask_name}_{name}_rmse"] = float(value)
    return rows


def trajectory_metrics(
    variant: str,
    result: Mapping[str, Any],
    targets: np.ndarray,
    *,
    key: str,
    positions: np.ndarray,
    edges: np.ndarray,
    node_type: np.ndarray,
    proxy_weights: np.ndarray,
    component_scale: np.ndarray,
    gamma: float,
    dt: float,
    start_frame: int,
    step_stride: int,
    endpoint_calls: Sequence[int],
    shock_quantile: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    valid_length = int(result["valid_length"])
    predictions = np.asarray(result["predictions"])
    valid_targets = targets[:valid_length]
    proxy_curve: list[float] = []
    equal_curve: list[float] = []
    call_rows: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []
    release_metrics = None
    if valid_length:
        primitive_prediction = conservative_to_primitive_raw(predictions, gamma=gamma)
        primitive_target = conservative_to_primitive_raw(valid_targets, gamma=gamma)
        release_metrics = release_rollout_metrics(
            primitive_prediction, primitive_target, node_type
        )
        primitive_rows = _primitive_rmse_rows(
            primitive_prediction, primitive_target, node_type
        )
        for index, (prediction, target) in enumerate(
            zip(predictions, valid_targets, strict=True), start=1
        ):
            proxy_error = weighted_relative_l2_numpy(
                prediction, target, proxy_weights, component_scale
            )
            equal_error = weighted_relative_l2_numpy(
                prediction, target, np.ones(prediction.shape[0]), component_scale
            )
            proxy_curve.append(proxy_error)
            equal_curve.append(equal_error)
            call_rows.append(
                {
                    "variant": variant,
                    "trajectory": key,
                    "call_index": index,
                    "target_frame": start_frame + index * step_stride,
                    "physical_time": (start_frame + index * step_stride) * dt,
                    "scaled_relative_l2_reconstructed_weight_proxy": proxy_error,
                    "scaled_relative_l2_equal_node_proxy": equal_error,
                    "forward_seconds": float(result["call_seconds"][index - 1]),
                    **primitive_rows[index - 1],
                }
            )
            if index in endpoint_calls:
                endpoint_rows.append(
                    {
                        "variant": variant,
                        "trajectory": key,
                        "call_index": index,
                        "physical_time": (start_frame + index * step_stride) * dt,
                        **endpoint_diagnostics(
                            prediction,
                            target,
                            positions=positions,
                            edges=edges,
                            node_type=node_type,
                            proxy_weights=proxy_weights,
                            component_scale=component_scale,
                            gamma=gamma,
                            shock_quantile=shock_quantile,
                        ),
                    }
                )
    total_seconds = float(np.sum(result["call_seconds"]))
    summary = {
        "variant": variant,
        "trajectory": key,
        "requested_steps": int(targets.shape[0]),
        "valid_length": valid_length,
        "completed": bool(result["completed"]),
        "failure_cause": result["failure_cause"],
        "failure_call": result["failure_call"],
        "survival_fraction": valid_length / targets.shape[0],
        "final_scaled_relative_l2_reconstructed_weight_proxy": (
            proxy_curve[-1] if proxy_curve else None
        ),
        "mean_prefix_scaled_relative_l2_reconstructed_weight_proxy": (
            float(np.mean(proxy_curve)) if proxy_curve else None
        ),
        "final_scaled_relative_l2_equal_node_proxy": (
            equal_curve[-1] if equal_curve else None
        ),
        "scaled_relative_l2_reconstructed_weight_proxy_by_call": proxy_curve,
        "total_forward_seconds": total_seconds,
        "mean_forward_seconds": (
            total_seconds / len(result["call_seconds"])
            if result["call_seconds"]
            else None
        ),
        "calls_per_second": (
            len(result["call_seconds"]) / total_seconds if total_seconds > 0.0 else None
        ),
        "min_density": result["min_density"],
        "min_internal_energy": result["min_internal_energy"],
        "min_pressure": result["min_pressure"],
        "release_style_primitive_rmse": release_metrics,
        **failure_location_fields(result["failed_proposal"], node_type, gamma=gamma),
    }
    return summary, call_rows, endpoint_rows


def persistence_result(
    initial: np.ndarray,
    num_steps: int,
    *,
    gamma: float,
) -> dict[str, Any]:
    predictions = np.repeat(initial[None], num_steps, axis=0)
    admissibility = raw_admissibility_summary(predictions, gamma=gamma)
    return {
        "predictions": predictions.astype(np.float32),
        "valid_length": num_steps,
        "completed": True,
        "failure_cause": "completed",
        "failure_call": None,
        "failed_proposal": None,
        "call_seconds": [0.0] * num_steps,
        "min_density": admissibility["min_density"],
        "min_internal_energy": admissibility["min_internal_energy"],
        "min_pressure": admissibility["min_pressure"],
    }


def characteristic_travel_summary(
    reference_states: np.ndarray,
    positions: np.ndarray,
    edges: np.ndarray,
    *,
    gamma: float,
    delta_t: float,
) -> dict[str, float | str | None]:
    primitive = conservative_to_primitive_raw(reference_states, gamma=gamma)
    speed = np.linalg.norm(primitive[..., 1:3], axis=-1)
    sound = np.sqrt(gamma * primitive[..., 3] / primitive[..., 0])
    characteristic = speed + sound
    edge_length = median_edge_length(positions, edges)
    travel = characteristic * delta_t / edge_length if edge_length > 0.0 else None
    return {
        "contract": (
            "reference |velocity|+sound-speed travel per macro call divided by "
            "median graph-edge length; geometry proxy, not a solver CFL"
        ),
        "median_edge_length": edge_length,
        "p99_characteristic_travel_edges": (
            None if travel is None else float(np.quantile(travel, 0.99))
        ),
        "max_characteristic_travel_edges": (
            None if travel is None else float(np.max(travel))
        ),
    }


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if denominator == 0.0:
        return 1.0 if numerator == 0.0 else math.inf
    return float(numerator / denominator)


def aggregate_variant(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = list(rows)
    if not valid:
        raise ValueError("cannot aggregate an empty variant")
    completed = [row for row in valid if row["completed"]]
    mixed = [
        float(row["final_scaled_relative_l2_reconstructed_weight_proxy"])
        for row in valid
        if row["final_scaled_relative_l2_reconstructed_weight_proxy"] is not None
    ]
    common_call = min(int(row["valid_length"]) for row in valid)
    common_values = []
    if common_call > 0:
        common_values = [
            float(
                row["scaled_relative_l2_reconstructed_weight_proxy_by_call"][
                    common_call - 1
                ]
            )
            for row in valid
        ]
    completed_values = [
        float(row["final_scaled_relative_l2_reconstructed_weight_proxy"])
        for row in completed
    ]
    one_step = [
        float(row["scaled_relative_l2_reconstructed_weight_proxy_by_call"][0])
        for row in valid
        if row["valid_length"] > 0
    ]
    return {
        "trajectories": len(valid),
        "completed": len(completed),
        "completion_rate": len(completed) / len(valid),
        "mean_survival_fraction": float(
            np.mean([row["survival_fraction"] for row in valid])
        ),
        "one_step_entry_gate_mean_relative_l2_proxy": (
            float(np.mean(one_step)) if one_step else None
        ),
        "mixed_prefix_mean_final_relative_l2_proxy": (
            float(np.mean(mixed)) if mixed else None
        ),
        "completed_case_mean_final_relative_l2_proxy": (
            float(np.mean(completed_values)) if completed_values else None
        ),
        "common_endpoint_call": common_call,
        "common_endpoint_mean_relative_l2_proxy": (
            float(np.mean(common_values)) if common_values else None
        ),
        "mean_forward_seconds_per_trajectory": float(
            np.mean([row["total_forward_seconds"] for row in valid])
        ),
        "mean_batch1_call_seconds": float(
            np.mean(
                [
                    row["mean_forward_seconds"]
                    for row in valid
                    if row["mean_forward_seconds"] is not None
                ]
            )
        ),
    }


def grouped_evaluation(
    trajectory_rows: Sequence[Mapping[str, Any]],
    test_store: PCNOEuler2DShardStore,
    training_store: PCNOEuler2DShardStore,
) -> list[dict[str, Any]]:
    train_mach = np.asarray(
        [float(training_store.entry(key)["mach"]) for key in training_store.keys]
    )
    train_nodes = np.asarray(
        [int(training_store.entry(key)["num_nodes"]) for key in training_store.keys]
    )
    cutpoints = {
        "mach": np.quantile(train_mach, [1 / 3, 2 / 3]),
        "node_count": np.quantile(train_nodes, [1 / 3, 2 / 3]),
    }
    result: list[dict[str, Any]] = []
    for field, cuts in cutpoints.items():
        for bin_index, label in enumerate(("low", "middle", "high")):
            keys = []
            for key in test_store.keys:
                value = (
                    float(test_store.entry(key)["mach"])
                    if field == "mach"
                    else int(test_store.entry(key)["num_nodes"])
                )
                if int(np.searchsorted(cuts, value, side="right")) == bin_index:
                    keys.append(key)
            for variant in sorted({str(row["variant"]) for row in trajectory_rows}):
                selected = [
                    row
                    for row in trajectory_rows
                    if row["variant"] == variant and row["trajectory"] in keys
                ]
                if not selected:
                    continue
                final_values = [
                    row["final_scaled_relative_l2_reconstructed_weight_proxy"]
                    for row in selected
                    if row["final_scaled_relative_l2_reconstructed_weight_proxy"]
                    is not None
                ]
                result.append(
                    {
                        "group_field": field,
                        "group": label,
                        "variant": variant,
                        "trajectory_count": len(selected),
                        "completion_rate": float(
                            np.mean([row["completed"] for row in selected])
                        ),
                        "mean_survival_fraction": float(
                            np.mean([row["survival_fraction"] for row in selected])
                        ),
                        "mixed_prefix_mean_final_relative_l2_proxy": (
                            float(np.mean(final_values)) if final_values else None
                        ),
                    }
                )
    return result


def diagnostic_gate(
    aggregates: Mapping[str, Mapping[str, Any]],
    endpoint_rows: Sequence[Mapping[str, Any]],
    gain_one_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    baseline = aggregates["pcno_baseline"]
    candidate = aggregates["pcno_pointwise_tail_gain_0p75"]
    paired = {}
    for row in endpoint_rows:
        paired[(row["trajectory"], row["call_index"], row["variant"])] = row
    ratios: dict[str, list[float]] = {
        "state": [],
        "smooth_highpass": [],
        "front_position": [],
        "shock_thickness": [],
        "shock_strength": [],
    }
    joint_nonworse = []
    for trajectory, call_index, variant in list(paired):
        if variant != "pcno_baseline":
            continue
        base = paired[(trajectory, call_index, variant)]
        candidate_row = paired.get(
            (trajectory, call_index, "pcno_pointwise_tail_gain_0p75")
        )
        if candidate_row is None:
            continue
        fields = {
            "state": "scaled_relative_l2_reconstructed_weight_proxy",
            "smooth_highpass": "smooth_highpass_energy_reconstructed_weight_proxy",
            "front_position": "front_centroid_distance",
            "shock_thickness": "shock_thickness_log_error",
            "shock_strength": "shock_strength_log_error",
        }
        row_ratios = {
            name: _safe_ratio(candidate_row[field], base[field])
            for name, field in fields.items()
        }
        for name, value in row_ratios.items():
            if value is not None:
                ratios[name].append(value)
        if (
            row_ratios["state"] is not None
            and row_ratios["smooth_highpass"] is not None
        ):
            joint_nonworse.append(
                row_ratios["state"] <= 1.0 and row_ratios["smooth_highpass"] <= 1.0
            )
    medians = {
        name: (float(np.median(values)) if values else None)
        for name, values in ratios.items()
    }
    max_gain_one_relative = max(
        (float(row["relative_l2"]) for row in gain_one_rows), default=math.inf
    )
    max_gain_one_absolute = max(
        (float(row["max_abs"]) for row in gain_one_rows), default=math.inf
    )
    checks = {
        "gain_one_replay_relative_l2_at_most_1e-5": (max_gain_one_relative <= 1e-5),
        "gain_one_replay_max_abs_at_most_1e-5": max_gain_one_absolute <= 1e-5,
        "completion_noninferior": candidate["completion_rate"]
        >= baseline["completion_rate"],
        "survival_drop_at_most_0p02": (
            candidate["mean_survival_fraction"]
            >= baseline["mean_survival_fraction"] - 0.02
        ),
        "median_endpoint_state_error_reduction_at_least_5_percent": (
            medians["state"] is not None and medians["state"] <= 0.95
        ),
        "median_smooth_highpass_reduction_at_least_10_percent": (
            medians["smooth_highpass"] is not None
            and medians["smooth_highpass"] <= 0.90
        ),
        "joint_state_highpass_nonworse_fraction_at_least_0p75": (
            bool(joint_nonworse) and float(np.mean(joint_nonworse)) >= 0.75
        ),
        "anti_smearing_median_front_position_at_most_1p05": (
            medians["front_position"] is not None and medians["front_position"] <= 1.05
        ),
        "anti_smearing_median_thickness_error_at_most_1p05": (
            medians["shock_thickness"] is not None
            and medians["shock_thickness"] <= 1.05
        ),
        "anti_smearing_median_strength_error_at_most_1p05": (
            medians["shock_strength"] is not None and medians["shock_strength"] <= 1.05
        ),
    }
    passed = all(checks.values())
    return {
        "status": "passed" if passed else "failed",
        "checks": checks,
        "median_candidate_to_baseline_ratios": medians,
        "joint_nonworse_fraction": (
            float(np.mean(joint_nonworse)) if joint_nonworse else None
        ),
        "max_gain_one_replay_relative_l2": max_gain_one_relative,
        "max_gain_one_replay_absolute_error": max_gain_one_absolute,
        "interpretation": (
            "frozen counterfactual supports the pointwise-amplification hypothesis"
            if passed
            else "frozen counterfactual does not pass the pointwise-amplification gate"
        ),
        "learned_method_authorized": False,
        "authorization_boundary": (
            "the bump artifact cannot satisfy the physical conservative-correction "
            "oracle; the dynamic finite-volume benchmark remains mandatory"
        ),
    }


def boundary_sensitivity_gate(
    aggregates: Mapping[str, Mapping[str, Any]],
    endpoint_rows: Sequence[Mapping[str, Any]],
    trajectory_rows: Sequence[Mapping[str, Any]],
    *,
    candidate_variant: str,
    maximum_allowed_ratio: float,
) -> dict[str, Any]:
    """Route one frozen boundary sensitivity without calling it a fair baseline."""

    baseline = aggregates["pcno_baseline"]
    candidate = aggregates[candidate_variant]
    paired = {
        (str(row["trajectory"]), int(row["call_index"]), str(row["variant"])): row
        for row in endpoint_rows
    }
    common_calls = sorted(
        {
            call
            for trajectory, call, variant in paired
            if variant == "pcno_baseline"
            and (trajectory, call, candidate_variant) in paired
        }
    )
    gate_call = (
        20 if 20 in common_calls else (max(common_calls) if common_calls else None)
    )
    fields = {
        "state": "scaled_relative_l2_reconstructed_weight_proxy",
        "smooth_highpass": "smooth_highpass_energy_reconstructed_weight_proxy",
        "front_position": "front_centroid_distance",
        "shock_thickness": "shock_thickness_log_error",
        "shock_strength": "shock_strength_log_error",
    }
    ratios: dict[str, list[float]] = {name: [] for name in fields}
    if gate_call is not None:
        for trajectory, call, variant in paired:
            if variant != "pcno_baseline" or call != gate_call:
                continue
            base = paired[(trajectory, call, variant)]
            causal = paired.get((trajectory, call, candidate_variant))
            if causal is None:
                continue
            for name, field in fields.items():
                ratio = _safe_ratio(causal[field], base[field])
                if ratio is not None:
                    ratios[name].append(ratio)
    medians = {
        name: (float(np.median(values)) if values else None)
        for name, values in ratios.items()
    }

    rows = {
        (str(row["trajectory"]), str(row["variant"])): row for row in trajectory_rows
    }
    inflow_failures = []
    for (trajectory, variant), base in rows.items():
        if variant != "pcno_baseline" or base["failure_node_type"] != 3:
            continue
        causal = rows[(trajectory, candidate_variant)]
        required_extension = max(5, math.ceil(0.1 * int(base["requested_steps"])))
        inflow_failures.append(
            {
                "trajectory": trajectory,
                "baseline_valid_length": int(base["valid_length"]),
                "causal_valid_length": int(causal["valid_length"]),
                "valid_length_extension": int(causal["valid_length"])
                - int(base["valid_length"]),
                "required_extension": required_extension,
                "causal_completed": bool(causal["completed"]),
            }
        )
    checks = {
        "completion_noninferior": candidate["completion_rate"]
        >= baseline["completion_rate"],
        "mean_survival_noninferior": candidate["mean_survival_fraction"]
        >= baseline["mean_survival_fraction"],
        "gate_endpoint_state_ratio_within_tolerance": medians["state"] is not None
        and medians["state"] <= maximum_allowed_ratio,
        "gate_endpoint_smooth_highpass_ratio_within_tolerance": (
            medians["smooth_highpass"] is not None
            and medians["smooth_highpass"] <= maximum_allowed_ratio
        ),
        "anti_smearing_front_position_ratio_within_tolerance": (
            medians["front_position"] is not None
            and medians["front_position"] <= maximum_allowed_ratio
        ),
        "anti_smearing_thickness_ratio_within_tolerance": (
            medians["shock_thickness"] is not None
            and medians["shock_thickness"] <= maximum_allowed_ratio
        ),
        "anti_smearing_strength_ratio_within_tolerance": (
            medians["shock_strength"] is not None
            and medians["shock_strength"] <= maximum_allowed_ratio
        ),
        "every_inflow_failure_extends_by_predeclared_margin": bool(inflow_failures)
        and all(
            row["valid_length_extension"] >= row["required_extension"]
            for row in inflow_failures
        ),
    }
    informative = bool(inflow_failures)
    passed = informative and all(checks.values())
    return {
        "status": (
            "supports_matched_boundary_contract_repair"
            if passed
            else (
                "does_not_support_matched_boundary_contract_repair"
                if informative
                else "uninformative_no_baseline_inflow_failure"
            )
        ),
        "checks": checks,
        "candidate_variant": candidate_variant,
        "maximum_allowed_ratio": maximum_allowed_ratio,
        "gate_endpoint_call": gate_call,
        "median_causal_to_baseline_ratios": medians,
        "baseline_inflow_failures": inflow_failures,
        "matched_boundary_training_authorized": passed,
        "learned_stabilization_method_authorized": False,
        "interpretation": (
            "checkpoint-mismatch causal boundary sensitivity only; a fair autonomous "
            "baseline requires the same boundary policy during training and rollout"
        ),
        "authorization_boundary": (
            "even a pass authorizes only matched PCNO baseline closure; the bump "
            "artifact still cannot authorize a conservative learned corrector"
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    device = select_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint)
    training_store = PCNOEuler2DShardStore(args.training_data_dir)
    test_store = PCNOEuler2DShardStore(args.data_dir)
    contract = preprocessing_contract_audit(checkpoint, training_store, test_store)
    keys = select_trajectory_keys(args, test_store)
    model = build_model(checkpoint, device)
    identity_gains, candidate_gains = branch_gain_profiles(model)
    native_boundary_mode = str(checkpoint["boundary_mode"])
    if native_boundary_mode == CAUSAL_BOUNDARY_MODE and (
        args.counterfactual != NO_COUNTERFACTUAL
    ):
        raise ValueError(
            "causal-boundary checkpoints require --counterfactual none for native "
            "evaluation"
        )
    candidate_variant = None
    candidate_boundary_mode = None
    candidate_prediction_field = None
    boundary_audit = None
    if args.counterfactual == POINTWISE_COUNTERFACTUAL:
        candidate_variant = POINTWISE_VARIANT
        candidate_boundary_mode = "model_all_nodes"
        candidate_prediction_field = "pointwise_gain_predictions_conservative"
    elif args.counterfactual == CAUSAL_BOUNDARY_COUNTERFACTUAL:
        candidate_variant = CAUSAL_BOUNDARY_VARIANT
        candidate_boundary_mode = CAUSAL_BOUNDARY_MODE
        candidate_prediction_field = "causal_boundary_predictions_conservative"
        boundary_audit = load_boundary_mesh_audit(args.boundary_mesh_audit, keys)
    elif args.counterfactual == FIXED_INFLOW_COUNTERFACTUAL:
        candidate_variant = FIXED_INFLOW_VARIANT
        candidate_boundary_mode = "fixed_freestream_inflow_model_other_nodes"
        candidate_prediction_field = "fixed_inflow_predictions_conservative"
        boundary_audit = load_boundary_mesh_audit(args.boundary_mesh_audit, keys)
    step_stride = int(checkpoint["step_stride"])
    dt = float(test_store.manifest["dt"])
    delta_t = dt * step_stride
    checkpoint_digest = sha256_file(args.checkpoint)

    args.output_dir.mkdir(parents=True)
    artifact_dir = args.output_dir / "trajectories"
    artifact_dir.mkdir()
    trajectory_rows: list[dict[str, Any]] = []
    call_rows: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []
    gain_one_rows: list[dict[str, Any]] = []
    trajectory_contracts: list[dict[str, Any]] = []
    training_mach_values = [
        float(training_store.entry(item)["mach"]) for item in training_store.keys
    ]
    training_node_counts = [
        int(training_store.entry(item)["num_nodes"]) for item in training_store.keys
    ]

    for trajectory_index, key in enumerate(keys):
        states = test_store.states(key)
        target_indices = args.start_frame + step_stride * np.arange(
            1, args.num_steps + 1
        )
        targets = np.array(states[target_indices], copy=True)
        initial = np.array(states[args.start_frame], copy=True)
        positions = np.array(test_store.array(key, "nodes"), copy=True)
        edges = np.array(test_store.array(key, "edges"), copy=True)
        node_type = np.array(test_store.array(key, "node_type"), copy=True).reshape(-1)
        proxy_weights = np.array(test_store.array(key, "node_weights"), copy=True).sum(
            axis=-1
        )
        sample = test_store.tensor_sample(
            key, args.start_frame, step_stride=step_stride, device=device
        )
        if native_boundary_mode == "model_all_nodes":
            standard = model_call(model, sample, sample["current"])
            replay = model_call(
                model, sample, sample["current"], branch_gains=identity_gains
            )
            replay_delta = (
                replay[0].float().cpu().numpy() - standard[0].float().cpu().numpy()
            )
            gain_one_rows.append(
                {
                    "trajectory": key,
                    "max_abs": float(np.max(np.abs(replay_delta))),
                    "relative_l2": weighted_relative_l2_numpy(
                        replay[0].float().cpu().numpy(),
                        standard[0].float().cpu().numpy(),
                        proxy_weights,
                        np.asarray(checkpoint["normalization"]["state_scale"]),
                    ),
                }
            )

        native_boundary_policy = None
        native_boundary_metadata = None
        if native_boundary_mode == CAUSAL_BOUNDARY_MODE:
            native_contract = checkpoint["boundary_contract"]
            native_boundary_policy, native_boundary_metadata = (
                build_graph_causal_boundary_policy(
                    test_store,
                    key,
                    device=device,
                    max_source_hops=int(native_contract["max_source_hops"]),
                    rho_inf=float(native_contract["rho_inf"]),
                    p_inf=float(native_contract["p_inf"]),
                )
            )
            expected_policy_digests = native_contract.get("policy_digests", {})
            if key in expected_policy_digests and (
                native_boundary_metadata["policy_digest"]
                != expected_policy_digests[key]
            ):
                raise ValueError(
                    f"trajectory {key} native boundary policy digest changed"
                )
            native_boundary_metadata["applied_scope"] = NATIVE_CAUSAL_BOUNDARY_SCOPE
            native_boundary_metadata["source"] = "checkpoint_native_contract"

        candidate_boundary_policy = None
        candidate_boundary_metadata = None
        if boundary_audit is not None:
            candidate_boundary_policy, candidate_boundary_metadata = (
                build_validated_boundary_policy(
                    boundary_audit, test_store, key, device=device
                )
            )
            candidate_boundary_metadata["applied_scope"] = args.counterfactual
            candidate_boundary_metadata["policy"] = (
                "fixed freestream inflow, current-interior slip wall, and "
                "current-interior supersonic outflow"
                if args.counterfactual == CAUSAL_BOUNDARY_COUNTERFACTUAL
                else "fixed freestream inflow; model-predicted wall and outflow nodes"
            )

        results = {
            "persistence": persistence_result(
                initial,
                args.num_steps,
                gamma=model.gamma,
            ),
            "pcno_baseline": rollout_variant(
                model,
                test_store,
                key,
                start_frame=args.start_frame,
                num_steps=args.num_steps,
                step_stride=step_stride,
                device=device,
                branch_gains=None,
                boundary_policy=native_boundary_policy,
                boundary_scope=NATIVE_CAUSAL_BOUNDARY_SCOPE,
            ),
        }
        if candidate_variant is not None:
            results[candidate_variant] = rollout_variant(
                model,
                test_store,
                key,
                start_frame=args.start_frame,
                num_steps=args.num_steps,
                step_stride=step_stride,
                device=device,
                branch_gains=(
                    candidate_gains
                    if args.counterfactual == POINTWISE_COUNTERFACTUAL
                    else None
                ),
                boundary_policy=candidate_boundary_policy,
                boundary_scope=args.counterfactual,
            )
        for variant, result in results.items():
            summary, variant_calls, variant_endpoints = trajectory_metrics(
                variant,
                result,
                targets,
                key=key,
                positions=positions,
                edges=edges,
                node_type=node_type,
                proxy_weights=proxy_weights,
                component_scale=np.asarray(checkpoint["normalization"]["state_scale"]),
                gamma=model.gamma,
                dt=dt,
                start_frame=args.start_frame,
                step_stride=step_stride,
                endpoint_calls=args.endpoint_calls,
                shock_quantile=args.shock_quantile,
            )
            summary["mach"] = float(test_store.entry(key)["mach"])
            summary["num_nodes"] = int(test_store.entry(key)["num_nodes"])
            summary["mach_outside_training_range"] = not (
                min(training_mach_values)
                <= summary["mach"]
                <= max(training_mach_values)
            )
            summary["node_count_outside_training_range"] = not (
                min(training_node_counts)
                <= summary["num_nodes"]
                <= max(training_node_counts)
            )
            trajectory_rows.append(summary)
            call_rows.extend(variant_calls)
            endpoint_rows.extend(variant_endpoints)

        artifact_arrays: dict[str, Any] = {
            "schema": np.asarray(EVALUATION_SCHEMA),
            "trajectory_key": np.asarray(key),
            "trajectory_index": np.asarray(trajectory_index, dtype=np.int64),
            "initial_conservative": initial.astype(np.float32),
            "reference_targets_conservative": targets.astype(np.float32),
            "pcno_baseline_predictions_conservative": results["pcno_baseline"][
                "predictions"
            ],
            "positions": positions.astype(np.float32),
            "edges": edges.astype(np.int64),
            "node_type": node_type.astype(np.int64),
            "reconstructed_node_weights_proxy": proxy_weights.astype(np.float32),
            "equal_node_weights_proxy": np.ones(positions.shape[0], dtype=np.float32),
            "physical_target_times": target_indices.astype(np.float64) * dt,
            "physical_delta_t": np.asarray(delta_t, dtype=np.float64),
            "mach": np.asarray(test_store.entry(key)["mach"], dtype=np.float64),
            "boundary_mode": np.asarray(native_boundary_mode),
            "baseline_boundary_mode": np.asarray(native_boundary_mode),
            "coordinate_convention": np.asarray(
                test_store.manifest["coordinate_convention"]
            ),
            "state_convention": np.asarray(test_store.manifest["state_convention"]),
            "checkpoint_sha256": np.asarray(checkpoint_digest),
            "checkpoint_config_digest": np.asarray(checkpoint["config_digest"]),
            "test_manifest_digest": np.asarray(test_store.manifest_digest),
            "training_manifest_digest": np.asarray(training_store.manifest_digest),
            "baseline_valid_length": np.asarray(
                results["pcno_baseline"]["valid_length"], dtype=np.int64
            ),
            "baseline_failure_cause": np.asarray(
                results["pcno_baseline"]["failure_cause"]
            ),
            "baseline_failure_call": np.asarray(
                (
                    -1
                    if results["pcno_baseline"]["failure_call"] is None
                    else results["pcno_baseline"]["failure_call"]
                ),
                dtype=np.int64,
            ),

            "weight_provenance": np.asarray(
                "reconstructed_vertex_lumped_proxy_and_equal_node_proxy"
            ),
        }
        if native_boundary_metadata is not None:
            artifact_arrays["native_boundary_metadata"] = np.asarray(
                json.dumps(native_boundary_metadata, sort_keys=True)
            )
        if candidate_variant is not None:
            artifact_arrays["candidate_boundary_mode"] = np.asarray(
                candidate_boundary_mode
            )
            artifact_arrays["candidate_variant"] = np.asarray(candidate_variant)
            artifact_arrays["candidate_valid_length"] = np.asarray(
                results[candidate_variant]["valid_length"], dtype=np.int64
            )
            artifact_arrays["candidate_failure_cause"] = np.asarray(
                results[candidate_variant]["failure_cause"]
            )
            artifact_arrays["candidate_failure_call"] = np.asarray(
                (
                    -1
                    if results[candidate_variant]["failure_call"] is None
                    else results[candidate_variant]["failure_call"]
                ),
                dtype=np.int64,
            )
            artifact_arrays[candidate_prediction_field] = results[candidate_variant][
                "predictions"
            ]
            if args.counterfactual == POINTWISE_COUNTERFACTUAL:
                artifact_arrays["pointwise_layer_gains"] = np.asarray(
                    [1.0] + [POINTWISE_TAIL_GAIN] * (len(model.backbone.ws) - 1),
                    dtype=np.float32,
                )
            else:
                metadata_field = (
                    "causal_boundary_metadata"
                    if args.counterfactual == CAUSAL_BOUNDARY_COUNTERFACTUAL
                    else "fixed_inflow_metadata"
                )
                artifact_arrays[metadata_field] = np.asarray(
                    json.dumps(candidate_boundary_metadata, sort_keys=True)
                )
        if results["pcno_baseline"]["failed_proposal"] is not None:
            artifact_arrays["baseline_failed_proposal"] = results["pcno_baseline"][
                "failed_proposal"
            ].astype(np.float32)
        if candidate_variant is not None and (
            results[candidate_variant]["failed_proposal"] is not None
        ):
            artifact_arrays["candidate_failed_proposal"] = results[candidate_variant][
                "failed_proposal"
            ].astype(np.float32)
        artifact_name = f"trajectory_{key}.npz"
        np.savez_compressed(artifact_dir / artifact_name, **artifact_arrays)
        trajectory_contracts.append(
            {
                "trajectory": key,
                "artifact": f"trajectories/{artifact_name}",
                "characteristic_travel": characteristic_travel_summary(
                    np.concatenate((initial[None], targets), axis=0),
                    positions,
                    edges,
                    gamma=model.gamma,
                    delta_t=delta_t,
                ),
                "native_boundary": native_boundary_metadata,
                "counterfactual_boundary": candidate_boundary_metadata,
            }
        )
        progress = {
            "trajectory": key,
            "baseline_valid_length": results["pcno_baseline"]["valid_length"],
            "candidate_variant": candidate_variant,
        }
        if candidate_variant is not None:
            progress["candidate_valid_length"] = results[candidate_variant][
                "valid_length"
            ]
        print(json.dumps(progress, sort_keys=True), flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    variant_names = ["persistence", "pcno_baseline"]
    if candidate_variant is not None:
        variant_names.append(candidate_variant)
    aggregates = {
        variant: aggregate_variant(
            [row for row in trajectory_rows if row["variant"] == variant]
        )
        for variant in variant_names
    }
    grouped = grouped_evaluation(trajectory_rows, test_store, training_store)
    if args.counterfactual == NO_COUNTERFACTUAL:
        native_metadata = [
            row["native_boundary"]
            for row in trajectory_contracts
            if row["native_boundary"] is not None
        ]
        gate = {
            "status": "descriptive_native_evaluation",
            "checks": {
                "all_requested_trajectories_evaluated": (
                    aggregates["pcno_baseline"]["trajectories"] == len(keys)
                ),
                "native_boundary_policy_count_matches": (
                    native_boundary_mode != CAUSAL_BOUNDARY_MODE
                    or len(native_metadata) == len(keys)
                ),
                "native_boundary_policies_use_no_fallback": all(
                    int(item["fallback_target_count"]) == 0
                    for item in native_metadata
                ),
            },
            "learned_method_authorized": False,
            "interpretation": "checkpoint-native autonomous evaluation only",
        }
    elif args.counterfactual == POINTWISE_COUNTERFACTUAL:
        gate = diagnostic_gate(aggregates, endpoint_rows, gain_one_rows)
    else:
        gate = boundary_sensitivity_gate(
            aggregates,
            endpoint_rows,
            trajectory_rows,
            candidate_variant=candidate_variant,
            maximum_allowed_ratio=(
                1.10 if args.counterfactual == CAUSAL_BOUNDARY_COUNTERFACTUAL else 1.05
            ),
        )
    candidate_description = None
    if args.counterfactual == NO_COUNTERFACTUAL:
        verified_claim = (
            "frozen-checkpoint behavior on dataset ground truth under its declared "
            "native autonomous boundary and recurrence contract"
        )
        plausible_claim = "none; this is a descriptive native evaluation"
    elif args.counterfactual == POINTWISE_COUNTERFACTUAL:
        candidate_description = {
            "kind": "single frozen diagnostic counterfactual, not trained method",
            "pointwise_layer_gains": [1.0]
            + [POINTWISE_TAIL_GAIN] * (len(model.backbone.ws) - 1),
            "spectral_layer_gains": [1.0] * len(model.backbone.ws),
            "differential_layer_gains": [1.0] * len(model.backbone.ws),
            "boundary_mode": "model_all_nodes",
        }
        verified_claim = (
            "frozen-checkpoint behavior on dataset ground truth under raw autonomous "
            "recurrence and one exact predeclared branch-gain counterfactual"
        )
        plausible_claim = (
            "whether the pointwise branch amplifies shock-seeded local roughness"
        )
    else:
        applied_policy = (
            "fixed freestream inflow, current-interior slip wall, and "
            "current-interior supersonic outflow"
            if args.counterfactual == CAUSAL_BOUNDARY_COUNTERFACTUAL
            else "fixed freestream inflow; model-predicted wall and outflow nodes"
        )
        candidate_description = {
            "kind": (
                "single frozen checkpoint-mismatch boundary sensitivity; not a "
                "fairly trained autonomous baseline"
            ),
            "boundary_mode": candidate_boundary_mode,
            "input_and_output_policy": applied_policy,
            "mesh_audit_path_name": Path(args.boundary_mesh_audit).name,
            "mesh_audit_sha256": boundary_audit["_sha256"],
            "exact_dg_boundary_replay": False,
            "future_reference_boundary_values": False,
        }
        verified_claim = (
            "frozen-checkpoint boundary sensitivity on dataset ground truth under "
            "raw recurrence, using Line 2's validated mesh-to-graph map"
        )
        plausible_claim = (
            "whether model-all-nodes boundary drift materially contributes to the "
            "observed PCNO admissibility failures"
        )
    variants: dict[str, Any] = {
        "persistence": "repeat the initial state without model calls",
        "pcno_baseline": {
            "kind": "unmodified frozen checkpoint under its native recurrence",
            "boundary_mode": native_boundary_mode,
            "raw_recurrence": bool(checkpoint["raw_recurrence"]),
            "future_reference_boundary_values": False,
        },
    }
    if candidate_variant is not None:
        variants[candidate_variant] = candidate_description
    summary = {
        "schema": EVALUATION_SCHEMA,
        "status": "complete",
        "checkpoint": {
            "path_name": args.checkpoint.name,
            "sha256": checkpoint_digest,
            "epoch": checkpoint.get("epoch"),
            "best_epoch": checkpoint.get("best_epoch"),
            "config_digest": checkpoint["config_digest"],
            "data_manifest_digest": checkpoint["data_manifest_digest"],
            "model_config": checkpoint["model_config"],
            "normalization": checkpoint["normalization"],
            "parameter_count": parameter_count(model),
            "selection_rule": checkpoint.get("best_selection"),
            "boundary_mode": native_boundary_mode,
            "raw_recurrence": bool(checkpoint["raw_recurrence"]),
            "boundary_contract": checkpoint.get("boundary_contract"),
        },
        "preprocessing_contract": contract,
        "evaluation": {
            "trajectory_keys": keys,
            "start_frame": args.start_frame,
            "num_steps": args.num_steps,
            "step_stride": step_stride,
            "physical_delta_t": delta_t,
            "physical_horizon": args.num_steps * delta_t,
            "endpoint_calls": args.endpoint_calls,
            "device": str(device),
            "baseline_boundary_mode": native_boundary_mode,
            "candidate_boundary_mode": candidate_boundary_mode,
            "counterfactual": args.counterfactual,
            "raw_recurrence": bool(checkpoint["raw_recurrence"]),
            "boundary_decode_reencode_closure": (
                native_boundary_mode == CAUSAL_BOUNDARY_MODE
            ),
            "future_reference_boundary_values": False,
            "clipping_floors_smoothing_limiter": False,
            "ground_truth": "held-out test shard states",
            "latency_contract": (
                "synchronized batch-1 device forward, including basis construction; "
                "excludes CPU copy, metrics, and artifact compression"
            ),
        },
        "variants": variants,
        "aggregates": aggregates,
        "grouped_geometry_parameter_evaluation": grouped,
        "gain_one_replay": gain_one_rows,
        "diagnostic_gate": gate,
        "trajectory_artifacts": trajectory_contracts,
        "artifact_schema": {
            "trajectory_npz": (
                "initial state, held-out targets, baseline/candidate predictions, "
                "geometry, node types, proxy weights, time/Mach, boundary/failure "
                "contract, checkpoint/config/manifests, and resolved counterfactual"
            ),
            "trajectory_metrics_csv": (
                "mixed-prefix/completed metadata and release-style primitive RMSE"
            ),
            "call_metrics_csv": "per-call conservative proxy errors and primitive RMSE",
            "endpoint_metrics_csv": (
                "graph-native smooth high-pass and shock position/strength/thickness"
            ),
        },
        "claim_boundary": {
            "verified": verified_claim,
            "plausible_only": plausible_claim,
            "unsupported": (
                "physical conservation or flux, paper-faithful CPGNet, a learned "
                "stabilization method, or generalization beyond this test distribution"
            ),
            "result_to_claim_review": (
                "local evidence gate; independent Codex review pending because private "
                "unpublished results were not transmitted externally"
            ),
            "coarse_cfd_error_cost": "pending matched solver/runtime contract",
            "amortized_throughput": "not measured by this batch-1 evaluator",
        },
    }
    atomic_write_json(args.output_dir / "summary.json", summary)
    write_csv(args.output_dir / "trajectory_metrics.csv", trajectory_rows)
    write_csv(args.output_dir / "call_metrics.csv", call_rows)
    write_csv(args.output_dir / "endpoint_metrics.csv", endpoint_rows)
    write_csv(args.output_dir / "grouped_metrics.csv", grouped)
    if gain_one_rows:
        write_csv(args.output_dir / "gain_one_replay.csv", gain_one_rows)
    training_store.close()
    test_store.close()
    print(json.dumps(json_safe(gate), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
