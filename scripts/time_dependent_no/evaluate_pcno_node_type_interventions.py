#!/usr/bin/env python3
"""Evaluate frozen PCNO node-type interventions without changing boundary policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utility.time_dependent_no.euler2d_metrics import shock_indicator
from utility.time_dependent_no.pcno_artifacts import (
    atomic_write_json_with_paths as atomic_write_json,
)
from utility.time_dependent_no.pcno_artifacts import (
    git_state,
    jsonable_args,
    runtime_environment,
    sha256_file,
)
from utility.time_dependent_no.pcno_artifacts import (
    write_csv_with_paths as write_csv,
)
from utility.time_dependent_no.pcno_euler2d import (
    PCNOEuler2DResidual,
    PCNOEuler2DShardStore,
)
from utility.time_dependent_no.pcno_node_type_interpretability import (
    FAMILY_NODE_TYPE_NAMES,
    PCNOActivationRecorder,
    activation_differences,
    node_type_intervention_name,
    replace_node_types_with_normal,
    tensor_equivalence_metrics,
    type_lift_delta,
)
from utility.time_dependent_no.pcno_residual_structure import shock_vortex_regions
from utility.time_dependent_no.pcno_resolution_transfer import (
    Resolution,
    as_model_state,
    build_resolution_checkpoint_model,
    build_resolution_geometry,
    checkpoint_step_stride,
    conservative_admissibility_summary,
    load_resolution_checkpoint,
    load_resolution_reference,
    make_model_sample,
    node_types_for_protocol,
    parse_resolution,
    reference_at_resolution,
    resolution_label,
)
from utility.time_dependent_no.pcno_ripple_diagnostics import (
    conservative_to_primitive_raw,
    graph_distance_to_mask,
    node_highpass_field,
)
from utility.time_dependent_no.pcno_runtime import (
    autocast_context,
    build_checkpoint_model,
    load_checkpoint_payload,
    select_device,
    synchronize,
)
from utility.time_dependent_no.shock_vortex_family import (
    config_for_family_case,
    family_case_provenance,
    load_shock_vortex_family_manifest,
)

SCHEMA = "pcno_node_type_interventions_v1"
INTERVENTIONS = ("correct", "all_normal", "type1", "type2", "type3")
MODES = ("teacher_forced", "free_rollout")
ABS_REPEAT_LIMIT = 2.0e-5
REL_REPEAT_LIMIT = 2.0e-8
PHYSICAL_WAVELENGTH_BANDS = ((0.05, 0.125), (0.125, 0.25))


@dataclass(frozen=True)
class Intervention:
    name: str
    source_type: int | None
    all_normal: bool = False


@dataclass
class CaseData:
    family: str
    case_id: str
    resolution: Resolution | None
    sample: dict[str, torch.Tensor]
    reference_states: np.ndarray
    physical_times: np.ndarray
    nodes: np.ndarray
    edges: np.ndarray
    weights: np.ndarray
    physical_node_type: np.ndarray
    boundary_distance: np.ndarray
    state_scale: np.ndarray
    residual_scale: np.ndarray
    gamma: float
    provenance: dict[str, Any]

    @property
    def resolution_name(self) -> str:
        return (
            "native_graph"
            if self.resolution is None
            else resolution_label(self.resolution)
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("bump", "dynamic_fv"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization-json", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument(
        "--interventions", nargs="+", choices=INTERVENTIONS, default=INTERVENTIONS
    )
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--rollout-calls", type=int, required=True)
    parser.add_argument("--trace-cases", nargs="*", default=())
    parser.add_argument("--trace-calls", nargs="*", type=int, default=())
    parser.add_argument(
        "--trace-interventions",
        nargs="+",
        choices=INTERVENTIONS,
        default=("all_normal",),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", choices=("none", "bf16", "fp16"), default="none")
    parser.add_argument(
        "--native-repeat-absolute-limit",
        type=float,
        default=ABS_REPEAT_LIMIT,
    )
    parser.add_argument(
        "--native-repeat-relative-limit",
        type=float,
        default=REL_REPEAT_LIMIT,
    )
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-normalization-sha256", required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--expected-data-manifest-digest", required=True)
    parser.add_argument("--expected-source-base-git-head", required=True)
    parser.add_argument(
        "--expected-source",
        action="append",
        default=[],
        metavar="PATH=SHA256",
        help="Repeat for every frozen inference source file.",
    )
    parser.add_argument("--bump-replay-root", type=Path)
    parser.add_argument("--bump-replay-cases", nargs="*", default=())
    parser.add_argument("--bump-replay-calls", type=int, default=20)
    parser.add_argument("--bump-replay-absolute-limit", type=float)
    parser.add_argument("--bump-replay-relative-limit", type=float)
    parser.add_argument("--family-root", type=Path)
    parser.add_argument("--multires-reference-root", type=Path)
    parser.add_argument("--expected-family-manifest-sha256")
    parser.add_argument(
        "--resolutions", nargs="+", default=("125x50", "250x100", "500x200")
    )
    parser.add_argument("--training-resolution", default="250x100")
    parser.add_argument("--shock-quantile", type=float, default=0.9)
    parser.add_argument("--animation-max-nodes", type=int, default=25000)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(
            f"output directory already exists; choose a new path: {args.output_dir}"
        )
    if args.rollout_calls < 1:
        raise ValueError("--rollout-calls must be positive")
    if not 0.0 < args.shock_quantile < 1.0:
        raise ValueError("--shock-quantile must lie in (0,1)")
    if args.animation_max_nodes < 1:
        raise ValueError("--animation-max-nodes must be positive")
    if (
        args.native_repeat_absolute_limit <= 0.0
        or args.native_repeat_relative_limit <= 0.0
    ):
        raise ValueError("native repeat limits must be positive")
    source_head = args.expected_source_base_git_head.lower()
    if len(source_head) != 40 or any(
        character not in "0123456789abcdef" for character in source_head
    ):
        raise ValueError("--expected-source-base-git-head must be a 40-digit SHA")
    invalid_trace_calls = [
        value for value in args.trace_calls if value < 1 or value > args.rollout_calls
    ]
    if invalid_trace_calls:
        raise ValueError(f"trace calls lie outside the rollout: {invalid_trace_calls}")
    if "correct" in args.trace_interventions:
        raise ValueError("correct is implicit in every paired activation trace")
    if args.family == "bump":
        if args.bump_replay_root is None or not args.bump_replay_cases:
            raise ValueError("bump evaluation requires the open-validation replay gate")
        if (
            args.bump_replay_absolute_limit is None
            or args.bump_replay_relative_limit is None
            or args.bump_replay_absolute_limit <= 0.0
            or args.bump_replay_relative_limit <= 0.0
        ):
            raise ValueError("bump evaluation requires positive replay limits")
        if args.family_root is not None or args.multires_reference_root is not None:
            raise ValueError("dynamic reference roots are not valid for bump")
    else:
        if args.family_root is None or args.multires_reference_root is None:
            raise ValueError("dynamic_fv requires family and multires reference roots")
        if args.expected_family_manifest_sha256 is None:
            raise ValueError("dynamic_fv requires the frozen family-manifest digest")
        if args.bump_replay_root is not None or args.bump_replay_cases:
            raise ValueError("bump replay options are not valid for dynamic_fv")
        if (
            args.bump_replay_absolute_limit is not None
            or args.bump_replay_relative_limit is not None
        ):
            raise ValueError("bump replay limits are not valid for dynamic_fv")
    return args


def _canonical_mapping_digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _parse_source_bindings(values: Sequence[str]) -> dict[str, str]:
    if not values:
        raise ValueError("at least one --expected-source PATH=SHA256 is required")
    result = {}
    for value in values:
        path, separator, digest = str(value).partition("=")
        if not separator or not path or len(digest) != 64:
            raise ValueError(f"invalid source binding: {value!r}")
        if path in result:
            raise ValueError(f"duplicate source binding: {path}")
        result[path] = digest.lower()
    return result


def _loaded_project_source_hashes() -> dict[str, str]:
    root = ROOT.resolve()
    paths = set()
    for module in tuple(sys.modules.values()):
        raw_path = getattr(module, "__file__", None)
        if raw_path is None:
            continue
        path = Path(raw_path)
        if not path.is_absolute() or path.suffix != ".py":
            continue
        try:
            resolved = path.resolve()
            relative = resolved.relative_to(root)
        except (OSError, ValueError):
            continue
        if not resolved.is_file():
            continue
        paths.add(relative.as_posix())
    return {relative: sha256_file(ROOT / relative) for relative in sorted(paths)}


SOURCE_SNAPSHOT_SCHEMA = "pcno_isolated_source_manifest_v1"
SOURCE_SNAPSHOT_ROOTS = (
    "pcno",
    "utility",
    "scripts/time_dependent_no",
)


def _snapshot_project_sources() -> tuple[str, ...]:
    paths = []
    for relative_root in SOURCE_SNAPSHOT_ROOTS:
        root = ROOT / relative_root
        paths.extend(
            path.relative_to(ROOT).as_posix()
            for path in root.rglob("*.py")
            if "__pycache__" not in path.parts
        )
    return tuple(sorted(paths))


def _verify_source_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = getattr(args, "source_manifest", None)
    expected_manifest_sha = getattr(args, "expected_source_manifest_sha256", None)
    if manifest_path is None and expected_manifest_sha is None:
        source_bindings = _parse_source_bindings(args.expected_source)
        source_hashes = {}
        for relative_name, expected_sha in source_bindings.items():
            path = ROOT / relative_name
            if not path.is_file():
                raise FileNotFoundError(path)
            source_hashes[relative_name] = sha256_file(path)
            if source_hashes[relative_name] != expected_sha:
                raise ValueError(f"source hash mismatch: {relative_name}")
        return {
            "source_sha256": source_hashes,
            "source_manifest_sha256": None,
        }
    if manifest_path is None or expected_manifest_sha is None:
        raise ValueError("source manifest path and digest must be supplied together")
    actual_manifest_sha = sha256_file(manifest_path)
    if actual_manifest_sha != str(expected_manifest_sha).lower():
        raise ValueError("isolated source-manifest digest mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SOURCE_SNAPSHOT_SCHEMA:
        raise ValueError("unexpected isolated source-manifest schema")
    expected_head = str(args.expected_source_base_git_head).lower()
    if str(manifest.get("source_base_git_head", "")).lower() != expected_head:
        raise ValueError("source manifest and registered base git head differ")
    raw_hashes = manifest.get("source_sha256")
    if not isinstance(raw_hashes, dict) or not raw_hashes:
        raise ValueError("source manifest contains no source hashes")
    source_bindings = {
        str(relative): str(digest).lower() for relative, digest in raw_hashes.items()
    }
    discovered = _snapshot_project_sources()
    if set(source_bindings) != set(discovered):
        missing = sorted(set(discovered) - set(source_bindings))
        extra = sorted(set(source_bindings) - set(discovered))
        raise ValueError(
            f"isolated source inventory mismatch: missing={missing}, extra={extra}"
        )
    if args.expected_source:
        repeated_bindings = _parse_source_bindings(args.expected_source)
        if repeated_bindings != source_bindings:
            raise ValueError("--expected-source bindings differ from source manifest")
    source_hashes = {relative: sha256_file(ROOT / relative) for relative in discovered}
    mismatched = {
        relative: {"expected": source_bindings[relative], "actual": actual}
        for relative, actual in source_hashes.items()
        if actual != source_bindings[relative]
    }
    if mismatched:
        raise ValueError(f"isolated source hash mismatch: {mismatched}")
    return {
        "source_sha256": source_hashes,
        "source_manifest_sha256": actual_manifest_sha,
        "source_manifest_schema": SOURCE_SNAPSHOT_SCHEMA,
        "source_inventory_roots": list(SOURCE_SNAPSHOT_ROOTS),
    }


def _verify_common_provenance(
    args: argparse.Namespace,
    checkpoint: Mapping[str, Any],
    store: PCNOEuler2DShardStore,
) -> dict[str, Any]:
    checkpoint_sha = sha256_file(args.checkpoint)
    normalization_sha = sha256_file(args.normalization_json)
    split_sha = sha256_file(args.split_json)
    expected = {
        "checkpoint": args.expected_checkpoint_sha256.lower(),
        "normalization": args.expected_normalization_sha256.lower(),
        "split": args.expected_split_sha256.lower(),
    }
    actual = {
        "checkpoint": checkpoint_sha,
        "normalization": normalization_sha,
        "split": split_sha,
    }
    if actual != expected:
        raise ValueError(f"frozen artifact digest mismatch: {actual} != {expected}")
    if checkpoint.get("boundary_mode") != "model_all_nodes":
        raise ValueError("node-type intervention freezes model_all_nodes policy")
    if checkpoint.get("raw_recurrence") is not True:
        raise ValueError("checkpoint does not declare raw recurrence")
    if checkpoint.get("data_manifest_digest") != args.expected_data_manifest_digest:
        raise ValueError("checkpoint data-manifest digest differs from contract")
    if store.manifest_digest != args.expected_data_manifest_digest:
        raise ValueError("active shard manifest differs from checkpoint contract")

    normalization = json.loads(args.normalization_json.read_text(encoding="utf-8"))
    if normalization != checkpoint["normalization"]:
        raise ValueError("normalization file and checkpoint mapping differ")
    split = json.loads(args.split_json.read_text(encoding="utf-8"))
    if split.get("data_manifest_digest") != args.expected_data_manifest_digest:
        raise ValueError("split and data-manifest bindings differ")

    source_snapshot = _verify_source_snapshot(args)
    return {
        "artifact_sha256": actual,
        "data_manifest_digest": store.manifest_digest,
        "normalization_mapping_digest": _canonical_mapping_digest(normalization),
        **source_snapshot,
        "source_base_git_head": args.expected_source_base_git_head.lower(),
        "split": split,
    }


def _intervention_specs(family: str, requested: Sequence[str]) -> list[Intervention]:
    unique = list(dict.fromkeys(requested))
    if "correct" not in unique:
        unique.insert(0, "correct")
    specs = []
    for name in unique:
        if name == "correct":
            specs.append(Intervention(name="correct", source_type=None))
        elif name == "all_normal":
            specs.append(
                Intervention(name="all_normal", source_type=None, all_normal=True)
            )
        else:
            source_type = int(name[-1])
            specs.append(
                Intervention(
                    name=node_type_intervention_name(
                        family=family, source_type=source_type
                    ),
                    source_type=source_type,
                )
            )
    return specs


def _node_type_for_spec(
    physical: torch.Tensor,
    *,
    family: str,
    spec: Intervention,
) -> torch.Tensor:
    if spec.name == "correct":
        return physical
    return replace_node_types_with_normal(
        physical,
        family=family,
        source_type=None if spec.all_normal else spec.source_type,
    )


def _intervention_contract_rows(
    model: PCNOEuler2DResidual,
    case: CaseData,
    specs: Sequence[Intervention],
) -> list[dict[str, Any]]:
    correct_node_type = case.sample["node_type"]
    physical = correct_node_type.detach().cpu().numpy().reshape(-1)
    total_mass = float(case.weights.sum())
    semantics = FAMILY_NODE_TYPE_NAMES[case.family]
    common = {
        "family": case.family,
        "case_id": case.case_id,
        "resolution": case.resolution_name,
        "num_nodes": int(case.nodes.shape[0]),
        "num_directed_edges": int(case.edges.shape[0]),
        "mach": float(case.sample["mach"].detach().cpu().reshape(-1)[0]),
        "total_weight_mass": total_mass,
    }
    for code, name in semantics.items():
        mask = physical == code
        common[f"type_{code}_meaning"] = name
        common[f"type_{code}_node_count"] = int(mask.sum())
        common[f"type_{code}_weight_mass"] = float(case.weights[mask].sum())

    rows = []
    for spec in specs:
        intervened = _node_type_for_spec(
            correct_node_type,
            family=case.family,
            spec=spec,
        )
        affected = correct_node_type.detach().cpu().numpy().reshape(
            -1
        ) != intervened.detach().cpu().numpy().reshape(-1)
        lift_delta = (
            type_lift_delta(model.backbone, correct_node_type, intervened)[0]
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        per_node_energy = np.square(lift_delta).mean(axis=-1)
        affected_mass = float(case.weights[affected].sum())
        weighted_lift_rms = float(
            np.sqrt(np.dot(case.weights, per_node_energy) / total_mass)
        )
        affected_lift_rms = (
            None
            if affected_mass <= 0.0
            else float(
                np.sqrt(
                    np.dot(case.weights[affected], per_node_energy[affected])
                    / affected_mass
                )
            )
        )
        rows.append(
            {
                **common,
                "intervention": spec.name,
                "source_type": (
                    "all_nonnormal" if spec.all_normal else spec.source_type
                ),
                "target_type": 0 if spec.name != "correct" else None,
                "target_meaning": (semantics[0] if spec.name != "correct" else None),
                "affected_node_count": int(affected.sum()),
                "affected_node_fraction": float(affected.mean()),
                "affected_weight_mass": affected_mass,
                "affected_weight_fraction": affected_mass / total_mass,
                "weighted_type_lift_rms": weighted_lift_rms,
                "affected_node_type_lift_rms": affected_lift_rms,
            }
        )
    return rows


@torch.inference_mode()
def _model_call(
    model: PCNOEuler2DResidual,
    case: CaseData,
    current: np.ndarray | torch.Tensor,
    node_type: torch.Tensor,
    *,
    device: torch.device,
    amp: str,
    trace: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
    if isinstance(current, np.ndarray):
        current_tensor = torch.as_tensor(
            np.asarray(current, dtype=np.float32), dtype=torch.float32, device=device
        ).unsqueeze(0)
    else:
        current_tensor = current
    recorder = PCNOActivationRecorder(model.backbone) if trace else None
    synchronize(device)
    with autocast_context(device, amp):
        if recorder is None:
            prediction = model(
                current_tensor,
                node_mask=case.sample["node_mask"],
                nodes=case.sample["nodes"],
                node_weights=case.sample["node_weights"],
                node_rhos=case.sample["node_rhos"],
                directed_edges=case.sample["directed_edges"],
                edge_gradient_weights=case.sample["edge_gradient_weights"],
                node_type=node_type,
                mach=case.sample["mach"],
            )
        else:
            with recorder:
                prediction = model(
                    current_tensor,
                    node_mask=case.sample["node_mask"],
                    nodes=case.sample["nodes"],
                    node_weights=case.sample["node_weights"],
                    node_rhos=case.sample["node_rhos"],
                    directed_edges=case.sample["directed_edges"],
                    edge_gradient_weights=case.sample["edge_gradient_weights"],
                    node_type=node_type,
                    mach=case.sample["mach"],
                )
    synchronize(device)
    snapshot = None if recorder is None else recorder.snapshot(device="cpu")
    return prediction, snapshot


def _hook_equivalence(
    model: PCNOEuler2DResidual,
    case: CaseData,
    *,
    device: torch.device,
    amp: str,
    absolute_limit: float = ABS_REPEAT_LIMIT,
    relative_limit: float = REL_REPEAT_LIMIT,
) -> dict[str, Any]:
    current = case.reference_states[0]
    node_type = case.sample["node_type"]
    unhooked_0, _ = _model_call(
        model, case, current, node_type, device=device, amp=amp, trace=False
    )
    hooked_0, trace_0 = _model_call(
        model, case, current, node_type, device=device, amp=amp, trace=True
    )
    hooked_1, trace_1 = _model_call(
        model, case, current, node_type, device=device, amp=amp, trace=True
    )
    unhooked_1, _ = _model_call(
        model, case, current, node_type, device=device, amp=amp, trace=False
    )
    if trace_0 is None or trace_1 is None:
        raise RuntimeError("hook equivalence did not produce activation traces")

    comparisons = {
        "native_repeat": tensor_equivalence_metrics(unhooked_0, unhooked_1),
        "unhooked0_hooked0": tensor_equivalence_metrics(unhooked_0, hooked_0),
        "hooked0_hooked1": tensor_equivalence_metrics(hooked_0, hooked_1),
        "hooked1_unhooked1": tensor_equivalence_metrics(hooked_1, unhooked_1),
    }
    native = comparisons["native_repeat"]
    output_scale = max(1.0, float(unhooked_0.detach().abs().max().cpu()))
    numerical_floor = 32.0 * torch.finfo(torch.float32).eps * output_scale
    hook_abs_limit = max(
        float(absolute_limit),
        1.25 * max(float(native["max_abs"]), numerical_floor),
    )
    passed = (
        bool(native["same_shape"])
        and bool(native["same_dtype"])
        and float(native["max_abs"]) <= absolute_limit
        and float(native["relative_l2"]) <= relative_limit
    )
    for name in (
        "unhooked0_hooked0",
        "hooked0_hooked1",
        "hooked1_unhooked1",
    ):
        metric = comparisons[name]
        passed = (
            passed
            and bool(metric["same_shape"])
            and bool(metric["same_dtype"])
            and float(metric["max_abs"]) <= hook_abs_limit
            and float(metric["relative_l2"]) <= relative_limit
        )
    if device.type == "cpu":
        passed = passed and all(
            bool(value["exact_equal"]) for value in comparisons.values()
        )
    result = {
        "family": case.family,
        "case_id": case.case_id,
        "resolution": case.resolution_name,
        "amp": amp,
        "absolute_native_repeat_limit": absolute_limit,
        "relative_repeat_limit": relative_limit,
        "hook_absolute_limit": hook_abs_limit,
        "comparisons": comparisons,
        "passed": passed,
    }
    if not passed:
        raise RuntimeError(f"hook equivalence failed: {json.dumps(result)}")
    return result


def _scaled_rms(
    value: np.ndarray,
    *,
    weights: np.ndarray,
    scale: np.ndarray,
    mask: np.ndarray,
) -> float | None:
    selected = np.asarray(mask, dtype=bool)
    if not np.any(selected):
        return None
    mass = float(np.asarray(weights)[selected].sum())
    if mass <= 0.0:
        return None
    array = np.asarray(value, dtype=np.float64)[selected]
    squared = np.square(array / np.asarray(scale)[None, :]).sum(axis=-1)
    return float(np.sqrt(np.dot(np.asarray(weights)[selected], squared) / mass))


def _scaled_relative_l2(
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    weights: np.ndarray,
    scale: np.ndarray,
    mask: np.ndarray,
) -> float | None:
    selected = np.asarray(mask, dtype=bool)
    if not np.any(selected):
        return None
    weight = np.asarray(weights, dtype=np.float64)[selected]
    error = (
        np.asarray(prediction, dtype=np.float64)[selected]
        - np.asarray(reference, dtype=np.float64)[selected]
    ) / np.asarray(scale)[None, :]
    baseline = (
        np.asarray(reference, dtype=np.float64)[selected] / np.asarray(scale)[None, :]
    )
    numerator = float(np.einsum("n,nc,nc->", weight, error, error))
    denominator = float(np.einsum("n,nc,nc->", weight, baseline, baseline))
    return None if denominator <= 1.0e-30 else float(np.sqrt(numerator / denominator))


def _latent_rms(
    value: np.ndarray,
    *,
    weights: np.ndarray,
    mask: np.ndarray,
) -> float:
    selected = np.asarray(mask, dtype=bool)
    weight = np.asarray(weights, dtype=np.float64)[selected]
    energy = np.square(np.asarray(value, dtype=np.float64)[selected]).mean(axis=-1)
    return float(np.sqrt(np.dot(weight, energy) / weight.sum()))


def _dynamic_boundary_distance(nodes: np.ndarray, resolution: Resolution) -> np.ndarray:
    nx, ny = resolution
    positions = np.asarray(nodes, dtype=np.float64)
    x = positions[:, 0].reshape(ny, nx)
    y = positions[:, 1].reshape(ny, nx)
    dx = float(np.median(np.diff(x[0])))
    dy = float(np.median(np.diff(y[:, 0])))
    x_min, x_max = float(x[0, 0] - 0.5 * dx), float(x[0, -1] + 0.5 * dx)
    y_min, y_max = float(y[0, 0] - 0.5 * dy), float(y[-1, 0] + 0.5 * dy)
    return np.minimum.reduce(
        (
            positions[:, 0] - x_min,
            x_max - positions[:, 0],
            positions[:, 1] - y_min,
            y_max - positions[:, 1],
        )
    )


def _region_masks(
    case: CaseData,
    reference: np.ndarray,
    *,
    shock_quantile: float,
) -> dict[str, np.ndarray]:
    if case.family == "dynamic_fv":
        if case.resolution is None:
            raise ValueError("dynamic case is missing its resolution")
        masks, _, _ = shock_vortex_regions(
            reference,
            case.nodes,
            resolution=case.resolution,
            gamma=case.gamma,
            boundary_width=0.05,
            shock_core_width=0.02,
            shock_envelope_width=0.05,
            vortex_radius=0.18,
        )
        return {
            "all": np.ones(case.nodes.shape[0], dtype=bool),
            "boundary_nodes": case.physical_node_type != 0,
            "boundary_distance_le_0.02": masks["boundary_le_0.02"],
            "boundary_distance_le_0.05": masks["boundary_le_0.05"],
            "shock": masks["partition_shock"],
            "vortex": masks["partition_vortex"],
            "smooth": masks["partition_smooth"],
        }

    primitive = conservative_to_primitive_raw(reference, gamma=case.gamma)
    shock_seed = shock_indicator(
        primitive,
        case.edges,
        quantile=shock_quantile,
    )
    shock_distance = graph_distance_to_mask(case.nodes, case.edges, shock_seed)
    boundary_005 = case.boundary_distance <= 0.05
    boundary_010 = case.boundary_distance <= 0.10
    shock = (shock_distance <= 0.05) & ~boundary_010
    smooth = ~(boundary_010 | shock)
    return {
        "all": np.ones(case.nodes.shape[0], dtype=bool),
        "boundary_nodes": case.physical_node_type != 0,
        "boundary_distance_le_0.05": boundary_005,
        "boundary_distance_le_0.10": boundary_010,
        "shock": shock,
        "smooth": smooth,
    }


def _activation_rows(
    backbone: torch.nn.Module,
    case: CaseData,
    correct: Mapping[str, torch.Tensor],
    intervened: Mapping[str, torch.Tensor],
    *,
    correct_node_type: torch.Tensor,
    intervened_node_type: torch.Tensor,
    intervention: str,
    mode: str,
    call: int,
    regions: Mapping[str, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    differences = activation_differences(correct, intervened)
    rows = []
    maps = {}
    for field, difference_tensor in differences.items():
        difference = difference_tensor[0].float().numpy().astype(np.float64)
        correct_value = correct[field][0].float().numpy().astype(np.float64)
        intervened_value = intervened[field][0].float().numpy().astype(np.float64)
        maps[field.replace(".", "_") + "_difference_norm"] = np.linalg.norm(
            difference, axis=-1
        ).astype(np.float32)
        for region_name, mask in regions.items():
            selected = np.asarray(mask, dtype=bool)
            if not np.any(selected):
                continue
            weight = case.weights[selected]
            correct_rms = _latent_rms(
                correct_value, weights=case.weights, mask=selected
            )
            intervened_rms = _latent_rms(
                intervened_value, weights=case.weights, mask=selected
            )
            difference_rms = _latent_rms(
                difference, weights=case.weights, mask=selected
            )
            rows.append(
                {
                    "family": case.family,
                    "case_id": case.case_id,
                    "resolution": case.resolution_name,
                    "mode": mode,
                    "call": call,
                    "physical_time": float(case.physical_times[call]),
                    "intervention": intervention,
                    "field": field,
                    "region": region_name,
                    "node_count": int(selected.sum()),
                    "weight_mass": float(weight.sum()),
                    "correct_rms": correct_rms,
                    "intervened_rms": intervened_rms,
                    "difference_rms": difference_rms,
                    "difference_to_correct_rms": difference_rms
                    / max(correct_rms, 1.0e-30),
                    "interpretation": (
                        "branch and hidden norms are diagnostics, not causal shares"
                    ),
                }
            )
    correct_input = correct["model_input"].float()
    intervened_input = intervened["model_input"].float()
    input_difference = correct_input - intervened_input
    fc0 = getattr(backbone, "fc0", None)
    if not isinstance(fc0, torch.nn.Linear):
        raise TypeError("PCNO backbone must expose a linear fc0 lifting layer")
    weight = fc0.weight.detach().float().cpu()
    analytical_full = torch.matmul(input_difference, weight.transpose(0, 1))
    analytical_type = (
        type_lift_delta(
            backbone,
            correct_node_type,
            intervened_node_type,
        )
        .detach()
        .float()
        .cpu()
    )
    analytical_nontype = analytical_full - analytical_type
    actual_lift_difference = differences["post_lift"].float()
    full_lift_error = actual_lift_difference - analytical_full
    type_only_error = actual_lift_difference - analytical_type
    nontype_input_difference = input_difference.clone()
    nontype_input_difference[..., 7:11] = 0.0
    same_nontype_input = bool(torch.count_nonzero(nontype_input_difference) == 0)

    maps["analytical_type_lift_component_norm"] = (
        torch.linalg.vector_norm(analytical_type[0], dim=-1).numpy().astype(np.float32)
    )
    maps["analytical_nontype_lift_component_norm"] = (
        torch.linalg.vector_norm(analytical_nontype[0], dim=-1)
        .numpy()
        .astype(np.float32)
    )
    maps["analytical_full_lift_error_norm"] = (
        torch.linalg.vector_norm(full_lift_error[0], dim=-1).numpy().astype(np.float32)
    )
    rows.append(
        {
            "family": case.family,
            "case_id": case.case_id,
            "resolution": case.resolution_name,
            "mode": mode,
            "call": call,
            "physical_time": float(case.physical_times[call]),
            "intervention": intervention,
            "field": "analytical_full_post_lift_check",
            "region": "all",
            "node_count": int(case.nodes.shape[0]),
            "weight_mass": float(case.weights.sum()),
            "correct_rms": None,
            "intervened_rms": None,
            "difference_rms": float(torch.sqrt(torch.mean(full_lift_error.square()))),
            "difference_to_correct_rms": None,
            "maximum_absolute_error": float(full_lift_error.abs().max()),
            "same_nontype_input": same_nontype_input,
            "interpretation": (
                "full post-lift difference equals W_fc0 times the full input "
                "difference; numerical error reflects execution precision"
            ),
        }
    )
    rows.append(
        {
            "family": case.family,
            "case_id": case.case_id,
            "resolution": case.resolution_name,
            "mode": mode,
            "call": call,
            "physical_time": float(case.physical_times[call]),
            "intervention": intervention,
            "field": "analytical_type_only_post_lift_check",
            "region": "all",
            "node_count": int(case.nodes.shape[0]),
            "weight_mass": float(case.weights.sum()),
            "correct_rms": None,
            "intervened_rms": None,
            "difference_rms": (
                float(torch.sqrt(torch.mean(type_only_error.square())))
                if same_nontype_input
                else None
            ),
            "difference_to_correct_rms": None,
            "maximum_absolute_error": (
                float(type_only_error.abs().max()) if same_nontype_input else None
            ),
            "same_nontype_input": same_nontype_input,
            "interpretation": (
                "when non-type inputs match, correct minus intervened post-lift "
                "equals W_type times the one-hot difference; otherwise this row "
                "is intentionally not evaluated"
            ),
        }
    )
    return rows, maps


def _outcome_rows(
    case: CaseData,
    *,
    mode: str,
    call: int,
    spec: Intervention,
    current: np.ndarray,
    prediction: np.ndarray,
    correct_current: np.ndarray,
    correct_prediction: np.ndarray,
    target: np.ndarray,
    regions: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows = []
    state_gap = prediction - correct_prediction
    increment_gap = (prediction - current) - (correct_prediction - correct_current)
    correct_increment = correct_prediction - correct_current
    for region_name, mask in regions.items():
        rows.append(
            {
                "family": case.family,
                "case_id": case.case_id,
                "resolution": case.resolution_name,
                "mode": mode,
                "call": call,
                "physical_time": float(case.physical_times[call]),
                "intervention": spec.name,
                "region": region_name,
                "node_count": int(np.asarray(mask).sum()),
                "weight_mass": float(case.weights[np.asarray(mask)].sum()),
                "prediction_error_rms": _scaled_rms(
                    prediction - target,
                    weights=case.weights,
                    scale=case.state_scale,
                    mask=mask,
                ),
                "prediction_relative_l2": _scaled_relative_l2(
                    prediction,
                    target,
                    weights=case.weights,
                    scale=case.state_scale,
                    mask=mask,
                ),
                "state_gap_to_correct_rms": _scaled_rms(
                    state_gap,
                    weights=case.weights,
                    scale=case.state_scale,
                    mask=mask,
                ),
                "increment_gap_to_correct_rms": _scaled_rms(
                    increment_gap,
                    weights=case.weights,
                    scale=case.residual_scale,
                    mask=mask,
                ),
                "correct_increment_rms": _scaled_rms(
                    correct_increment,
                    weights=case.weights,
                    scale=case.residual_scale,
                    mask=mask,
                ),
            }
        )
    return rows


def _physical_wavelength_band_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    resolution: Resolution,
    domain_lengths: tuple[float, float],
    component_scale: Sequence[float] | np.ndarray,
    wavelength_min: float,
    wavelength_max: float,
    denominator_epsilon: float = 1.0e-30,
) -> dict[str, float | int | None]:
    """Measure error on a fixed physical-wavelength band of a regular grid."""

    nx, ny = resolution
    lx, ly = (float(value) for value in domain_lengths)
    if (
        nx < 2
        or ny < 2
        or lx <= 0.0
        or ly <= 0.0
        or not np.isfinite(wavelength_min)
        or not np.isfinite(wavelength_max)
        or wavelength_min <= 0.0
        or wavelength_max <= wavelength_min
    ):
        raise ValueError("invalid grid, domain, or physical wavelength band")
    prediction_array = np.asarray(prediction, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    if (
        prediction_array.ndim != 2
        or prediction_array.shape[0] != nx * ny
        or reference_array.shape != prediction_array.shape
    ):
        raise ValueError("prediction/reference must be flattened row-major grid fields")
    scale = np.asarray(component_scale, dtype=np.float64)
    if scale.shape != (prediction_array.shape[-1],) or np.any(scale <= 0.0):
        raise ValueError("component_scale must be positive and match components")

    reshape = (ny, nx, -1)
    component_shape = (1, 1, -1)
    scaled_prediction = prediction_array.reshape(reshape) / scale.reshape(
        component_shape
    )
    scaled_reference = reference_array.reshape(reshape) / scale.reshape(component_shape)
    error_spectrum = np.fft.rfftn(
        scaled_prediction - scaled_reference,
        axes=(0, 1),
        norm="ortho",
    )
    reference_spectrum = np.fft.rfftn(
        scaled_reference,
        axes=(0, 1),
        norm="ortho",
    )
    frequency_x = np.fft.rfftfreq(nx, d=lx / nx)
    frequency_y = np.fft.fftfreq(ny, d=ly / ny)
    radial_frequency = np.hypot(frequency_y[:, None], frequency_x[None, :])
    mask = (radial_frequency >= 1.0 / wavelength_max) & (
        radial_frequency < 1.0 / wavelength_min
    )
    mode_count = int(np.count_nonzero(mask))
    if mode_count == 0:
        raise ValueError("the physical wavelength band contains no Fourier modes")
    error_energy = float(np.sum(np.abs(error_spectrum[mask]) ** 2))
    reference_energy = float(np.sum(np.abs(reference_spectrum[mask]) ** 2))
    return {
        "wavelength_min": float(wavelength_min),
        "wavelength_max": float(wavelength_max),
        "mode_count": mode_count,
        "error_spectral_l2": float(np.sqrt(error_energy)),
        "reference_spectral_l2": float(np.sqrt(reference_energy)),
        "relative_spectral_l2": (
            float(np.sqrt(error_energy / reference_energy))
            if reference_energy > denominator_epsilon
            else None
        ),
    }


def _frequency_row(
    case: CaseData,
    *,
    mode: str,
    call: int,
    intervention: str,
    correct_increment: np.ndarray,
    intervened_increment: np.ndarray,
) -> list[dict[str, Any]]:
    base = {
        "family": case.family,
        "case_id": case.case_id,
        "resolution": case.resolution_name,
        "mode": mode,
        "call": call,
        "physical_time": float(case.physical_times[call]),
        "intervention": intervention,
    }
    if case.family == "dynamic_fv":
        if case.resolution is None:
            raise ValueError("dynamic case is missing resolution")
        rows = []
        for wavelength_min, wavelength_max in PHYSICAL_WAVELENGTH_BANDS:
            row = _physical_wavelength_band_metrics(
                intervened_increment,
                correct_increment,
                resolution=case.resolution,
                domain_lengths=(2.0, 1.0),
                component_scale=case.residual_scale,
                wavelength_min=wavelength_min,
                wavelength_max=wavelength_max,
            )
            rows.append(
                {
                    **base,
                    "metric_kind": "physical_wavelength_band",
                    **row,
                }
            )
        return rows

    difference_highpass = node_highpass_field(
        intervened_increment - correct_increment, case.edges
    )
    correct_highpass = node_highpass_field(correct_increment, case.edges)
    mask = np.ones(case.nodes.shape[0], dtype=bool)
    difference_rms = _scaled_rms(
        difference_highpass,
        weights=case.weights,
        scale=case.residual_scale,
        mask=mask,
    )
    correct_rms = _scaled_rms(
        correct_highpass,
        weights=case.weights,
        scale=case.residual_scale,
        mask=mask,
    )
    return [
        {
            **base,
            "metric_kind": "graph_highpass_proxy_not_physical_frequency",
            "difference_highpass_rms": difference_rms,
            "correct_highpass_rms": correct_rms,
            "difference_to_correct_highpass_rms": (
                None
                if difference_rms is None or correct_rms is None
                else difference_rms / max(correct_rms, 1.0e-30)
            ),
        }
    ]


def _visualization_indices(case: CaseData, maximum_nodes: int) -> np.ndarray:
    count = case.nodes.shape[0]
    if count <= maximum_nodes:
        return np.arange(count, dtype=np.int64)
    if case.resolution is None:
        stride = math.ceil(count / maximum_nodes)
        return np.arange(0, count, stride, dtype=np.int64)
    nx, ny = case.resolution
    stride = math.ceil(math.sqrt(count / maximum_nodes))
    indices = np.arange(count, dtype=np.int64).reshape(ny, nx)
    return indices[::stride, ::stride].reshape(-1)


def _save_trace_npz(
    path: Path,
    case: CaseData,
    *,
    mode: str,
    call: int,
    intervention: str,
    correct_current: np.ndarray,
    intervened_current: np.ndarray,
    correct_prediction: np.ndarray,
    intervened_prediction: np.ndarray,
    target: np.ndarray,
    regions: Mapping[str, np.ndarray],
    maps: Mapping[str, np.ndarray],
) -> None:
    arrays: dict[str, Any] = {
        "schema": np.asarray(SCHEMA),
        "family": np.asarray(case.family),
        "case_id": np.asarray(case.case_id),
        "resolution": np.asarray(case.resolution_name),
        "mode": np.asarray(mode),
        "call": np.asarray(call, dtype=np.int64),
        "physical_time": np.asarray(case.physical_times[call]),
        "intervention": np.asarray(intervention),
        "nodes": case.nodes.astype(np.float32),
        "physical_node_type": case.physical_node_type.astype(np.int64),
        "weights": case.weights.astype(np.float64),
        "boundary_distance": case.boundary_distance.astype(np.float64),
        "correct_current": correct_current.astype(np.float32),
        "intervened_current": intervened_current.astype(np.float32),
        "target": target.astype(np.float32),
        "correct_prediction": correct_prediction.astype(np.float32),
        "intervened_prediction": intervened_prediction.astype(np.float32),
        "correct_residual": (correct_prediction - correct_current).astype(np.float32),
        "intervened_residual": (intervened_prediction - intervened_current).astype(
            np.float32
        ),
        "difference_interpretation": np.asarray(
            "teacher-forced differences are same-state node-type interventions; "
            "free-rollout differences after call 1 include recurrent state divergence"
        ),
    }
    arrays.update({f"region_{name}": value for name, value in regions.items()})
    arrays.update(maps)
    np.savez_compressed(path, **arrays)


def _evaluate_mode(
    args: argparse.Namespace,
    model: PCNOEuler2DResidual,
    case: CaseData,
    specs: Sequence[Intervention],
    *,
    mode: str,
    device: torch.device,
    outcome_rows: list[dict[str, Any]],
    completion_rows: list[dict[str, Any]],
    frequency_rows: list[dict[str, Any]],
    activation_rows: list[dict[str, Any]],
    trace_files: list[Path],
) -> dict[str, list[np.ndarray]]:
    free_states = {
        spec.name: np.array(case.reference_states[0], copy=True) for spec in specs
    }
    active = {spec.name: True for spec in specs}
    visual = defaultdict(list)
    trace_case = case.case_id in set(args.trace_cases)
    trace_calls = set(args.trace_calls)
    trace_requested = {
        spec.name
        for spec in _intervention_specs(case.family, args.trace_interventions)
        if spec.name != "correct"
    }

    for call in range(1, args.rollout_calls + 1):
        target = np.asarray(case.reference_states[call], dtype=np.float64)
        regions = _region_masks(
            case,
            target,
            shock_quantile=args.shock_quantile,
        )
        currents = {}
        predictions = {}
        traces: dict[str, dict[str, torch.Tensor]] = {}
        node_types = {}
        for spec in specs:
            if mode == "free_rollout" and not active[spec.name]:
                continue
            current = (
                np.asarray(case.reference_states[call - 1], dtype=np.float64)
                if mode == "teacher_forced"
                else free_states[spec.name]
            )
            node_type = _node_type_for_spec(
                case.sample["node_type"], family=case.family, spec=spec
            )
            prediction_tensor, _ = _model_call(
                model,
                case,
                current,
                node_type,
                device=device,
                amp=args.amp,
                trace=False,
            )
            prediction = (
                prediction_tensor[0].detach().float().cpu().numpy().astype(np.float64)
            )
            currents[spec.name] = np.asarray(current, dtype=np.float64)
            predictions[spec.name] = prediction
            node_types[spec.name] = node_type
            admissibility = conservative_admissibility_summary(
                prediction, gamma=case.gamma
            )
            completion_rows.append(
                {
                    "family": case.family,
                    "case_id": case.case_id,
                    "resolution": case.resolution_name,
                    "mode": mode,
                    "call": call,
                    "physical_time": float(case.physical_times[call]),
                    "intervention": spec.name,
                    **admissibility,
                }
            )
            if mode == "free_rollout":
                if admissibility["finite"] and admissibility["admissible"]:
                    free_states[spec.name] = prediction
                else:
                    active[spec.name] = False

            if (
                trace_case
                and call in trace_calls
                and (spec.name == "correct" or spec.name in trace_requested)
            ):
                traced_tensor, trace = _model_call(
                    model,
                    case,
                    current,
                    node_type,
                    device=device,
                    amp=args.amp,
                    trace=True,
                )
                trace_equivalence = tensor_equivalence_metrics(
                    prediction_tensor, traced_tensor
                )
                if (
                    not trace_equivalence["same_shape"]
                    or not trace_equivalence["same_dtype"]
                    or float(trace_equivalence["max_abs"])
                    > args.native_repeat_absolute_limit
                    or float(trace_equivalence["relative_l2"])
                    > args.native_repeat_relative_limit
                ):
                    raise RuntimeError(
                        f"traced call perturbed output: {trace_equivalence}"
                    )
                if trace is None:
                    raise RuntimeError("traced call did not return activations")
                traces[spec.name] = trace

        if "correct" not in predictions:
            raise RuntimeError("correct rollout terminated before comparison")
        correct_prediction = predictions["correct"]
        correct_current = currents["correct"]
        for spec in specs:
            if spec.name not in predictions:
                continue
            outcome_rows.extend(
                _outcome_rows(
                    case,
                    mode=mode,
                    call=call,
                    spec=spec,
                    current=currents[spec.name],
                    prediction=predictions[spec.name],
                    correct_current=correct_current,
                    correct_prediction=correct_prediction,
                    target=target,
                    regions=regions,
                )
            )
            if spec.name != "correct":
                frequency_rows.extend(
                    _frequency_row(
                        case,
                        mode=mode,
                        call=call,
                        intervention=spec.name,
                        correct_increment=correct_prediction - correct_current,
                        intervened_increment=(
                            predictions[spec.name] - currents[spec.name]
                        ),
                    )
                )
            if trace_case and spec.name in trace_requested and spec.name in traces:
                rows, maps = _activation_rows(
                    model.backbone,
                    case,
                    traces["correct"],
                    traces[spec.name],
                    correct_node_type=node_types["correct"],
                    intervened_node_type=node_types[spec.name],
                    intervention=spec.name,
                    mode=mode,
                    call=call,
                    regions=regions,
                )
                activation_rows.extend(rows)
                trace_path = (
                    args.output_dir
                    / "traces"
                    / (
                        f"{case.family}_{case.case_id}_{case.resolution_name}_"
                        f"{mode}_call{call:03d}_{spec.name}.npz"
                    )
                )
                _save_trace_npz(
                    trace_path,
                    case,
                    mode=mode,
                    call=call,
                    intervention=spec.name,
                    correct_current=correct_current,
                    intervened_current=currents[spec.name],
                    correct_prediction=correct_prediction,
                    intervened_prediction=predictions[spec.name],
                    target=target,
                    regions=regions,
                    maps=maps,
                )
                trace_files.append(trace_path)

        if trace_case:
            visual["reference"].append(target.astype(np.float32))
            for name in ("correct", "all_normal"):
                if name in predictions:
                    visual[f"{name}_prediction"].append(
                        predictions[name].astype(np.float32)
                    )
                    visual[f"{name}_residual"].append(
                        (predictions[name] - currents[name]).astype(np.float32)
                    )
            if "correct" in predictions and "all_normal" in predictions:
                visual["correct_minus_all_normal_prediction"].append(
                    (predictions["correct"] - predictions["all_normal"]).astype(
                        np.float32
                    )
                )
                visual["correct_minus_all_normal_residual"].append(
                    (
                        (predictions["correct"] - currents["correct"])
                        - (predictions["all_normal"] - currents["all_normal"])
                    ).astype(np.float32)
                )
    return dict(visual)


def _save_animation_payload(
    args: argparse.Namespace,
    case: CaseData,
    mode_payloads: Mapping[str, Mapping[str, Sequence[np.ndarray]]],
) -> Path:
    indices = _visualization_indices(case, args.animation_max_nodes)
    arrays: dict[str, Any] = {
        "schema": np.asarray(SCHEMA),
        "family": np.asarray(case.family),
        "case_id": np.asarray(case.case_id),
        "resolution": np.asarray(case.resolution_name),
        "visualization_only_subsampling": np.asarray(True),
        "visualization_node_indices": indices,
        "physical_times": case.physical_times[1:].astype(np.float64),
        "nodes": case.nodes[indices].astype(np.float32),
        "physical_node_type": case.physical_node_type[indices].astype(np.int64),
        "boundary_distance": case.boundary_distance[indices].astype(np.float64),
    }
    for mode, payload in mode_payloads.items():
        for name, values in payload.items():
            if values:
                arrays[f"{mode}_{name}"] = np.asarray(values)[:, indices]
    path = (
        args.output_dir
        / "animations"
        / f"{case.family}_{case.case_id}_{case.resolution_name}.npz"
    )
    np.savez_compressed(path, **arrays)
    return path


def _bump_replay_gate(
    args: argparse.Namespace,
    model: PCNOEuler2DResidual,
    cases_by_key: Mapping[str, CaseData],
    *,
    device: torch.device,
) -> dict[str, Any]:
    rows = []
    for key in args.bump_replay_cases:
        if key not in cases_by_key:
            raise ValueError(f"bump replay case is outside loaded validation: {key}")
        case = cases_by_key[key]
        path = args.bump_replay_root / f"trajectory_{key}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as artifact:
            if artifact["checkpoint_sha256"].item() != args.expected_checkpoint_sha256:
                raise ValueError(f"bump replay checkpoint mismatch: {key}")
            if artifact["boundary_mode"].item() != "model_all_nodes":
                raise ValueError(f"bump replay boundary policy mismatch: {key}")
            currents = np.asarray(artifact["rollout_currents"])
            expected = np.asarray(artifact["predictions"])
        count = min(args.bump_replay_calls, currents.shape[0])
        predictions = []
        for index in range(count):
            prediction, _ = _model_call(
                model,
                case,
                currents[index],
                case.sample["node_type"],
                device=device,
                amp="none",
                trace=False,
            )
            predictions.append(prediction[0].detach().float().cpu().numpy())
        actual = np.asarray(predictions)
        difference = actual - expected[:count]
        max_abs = float(np.max(np.abs(difference)))
        relative_l2 = float(
            np.linalg.norm(difference.astype(np.float64).ravel())
            / max(
                np.linalg.norm(expected[:count].astype(np.float64).ravel()),
                1.0e-30,
            )
        )
        passed = (
            max_abs <= args.bump_replay_absolute_limit
            and relative_l2 <= args.bump_replay_relative_limit
        )
        rows.append(
            {
                "case_id": key,
                "calls": count,
                "replay_artifact_sha256": sha256_file(path),
                "max_abs": max_abs,
                "relative_l2": relative_l2,
                "passed": passed,
            }
        )
    if not all(row["passed"] for row in rows):
        raise RuntimeError(f"D041 open-validation replay gate failed: {rows}")
    return {
        "precision": "fp32",
        "absolute_limit": args.bump_replay_absolute_limit,
        "relative_l2_limit": args.bump_replay_relative_limit,
        "rows": rows,
        "passed": True,
    }


def _load_bump_cases(
    args: argparse.Namespace,
    checkpoint: Mapping[str, Any],
    store: PCNOEuler2DShardStore,
    *,
    device: torch.device,
) -> list[CaseData]:
    split = json.loads(args.split_json.read_text(encoding="utf-8"))
    validation_keys = [str(value) for value in split.get("val_keys", [])]
    requested = validation_keys if args.case_ids is None else list(args.case_ids)
    if not requested or any(key not in validation_keys for key in requested):
        raise ValueError(
            "bump cases must be drawn only from the frozen validation split"
        )
    stride = int(checkpoint["step_stride"])
    dt = float(store.manifest["dt"]) * stride
    cases = []
    for key in requested:
        states = np.asarray(store.states(key))
        indices = stride * np.arange(args.rollout_calls + 1)
        if indices[-1] >= states.shape[0]:
            raise ValueError(f"bump trajectory {key} lacks the requested horizon")
        sample = store.tensor_sample(key, 0, step_stride=stride, device=device)
        # CaseData outlives the shard store's small mmap cache. Own these arrays
        # so loading a third native graph cannot invalidate an earlier case.
        nodes = np.array(store.array(key, "nodes"), dtype=np.float64, copy=True)
        edges = np.array(store.array(key, "directed_edges"), dtype=np.int64, copy=True)
        physical_type = np.array(
            store.array(key, "node_type"), dtype=np.int64, copy=True
        ).reshape(-1)
        weights = np.asarray(store.array(key, "node_weights"), dtype=np.float64).sum(
            axis=-1
        )
        boundary_distance = graph_distance_to_mask(nodes, edges, physical_type != 0)
        cases.append(
            CaseData(
                family="bump",
                case_id=key,
                resolution=None,
                sample=sample,
                reference_states=np.asarray(states[indices], dtype=np.float64),
                physical_times=dt * np.arange(args.rollout_calls + 1),
                nodes=nodes,
                edges=edges,
                weights=weights,
                physical_node_type=physical_type,
                boundary_distance=boundary_distance,
                state_scale=np.asarray(
                    checkpoint["normalization"]["state_scale"], dtype=np.float64
                ),
                residual_scale=np.asarray(
                    checkpoint["normalization"]["residual_scale"], dtype=np.float64
                ),
                gamma=float(checkpoint["normalization"]["gamma"]),
                provenance={"split": "validation", "trajectory": key},
            )
        )
    return cases


def _load_dynamic_cases(
    args: argparse.Namespace,
    checkpoint: Mapping[str, Any],
    store: PCNOEuler2DShardStore,
    *,
    device: torch.device,
) -> tuple[list[CaseData], list[dict[str, Any]]]:
    manifest_path = args.family_root / "family_manifest.json"
    if sha256_file(manifest_path) != args.expected_family_manifest_sha256:
        raise ValueError("dynamic family-manifest digest mismatch")
    manifest = load_shock_vortex_family_manifest(manifest_path)
    split = json.loads(args.split_json.read_text(encoding="utf-8"))
    validation_keys = [str(value) for value in split.get("val_keys", [])]
    requested = validation_keys if args.case_ids is None else list(args.case_ids)
    if not requested or any(key not in validation_keys for key in requested):
        raise ValueError("dynamic cases must be drawn only from frozen validation")
    case_configs = {}
    for key in requested:
        if family_case_provenance(manifest, key)["split"] != "validation":
            raise ValueError(f"dynamic case is not validation: {key}")
        case_configs[key] = config_for_family_case(manifest, key)
    resolutions = [parse_resolution(value) for value in args.resolutions]
    training_resolution = parse_resolution(args.training_resolution)
    stride = checkpoint_step_stride(checkpoint)
    first_config = case_configs[requested[0]]
    first_contract = (
        first_config.x_min,
        first_config.x_max,
        first_config.y_min,
        first_config.y_max,
        first_config.gamma,
        first_config.shock_mach,
    )
    for config in case_configs.values():
        contract = (
            config.x_min,
            config.x_max,
            config.y_min,
            config.y_max,
            config.gamma,
            config.shock_mach,
        )
        if contract != first_contract:
            raise ValueError("dynamic cases do not share one geometry/PDE contract")
    if float(checkpoint["normalization"]["gamma"]) != float(first_config.gamma):
        raise ValueError("dynamic checkpoint and reference gamma differ")
    geometry_by_resolution = {}
    physical_type_by_resolution = {}
    sample_by_resolution = {}
    for resolution in resolutions:
        config, geometry = build_resolution_geometry(first_config, resolution)
        physical_type = node_types_for_protocol(
            geometry,
            config,
            "physical",
            training_resolution=training_resolution,
        )
        geometry_by_resolution[resolution] = geometry
        physical_type_by_resolution[resolution] = physical_type
        sample_by_resolution[resolution] = make_model_sample(
            geometry,
            physical_type,
            mach=first_config.shock_mach,
            device=device,
        )

    cases = []
    reference_checks = []
    for key in requested:
        reference, check = load_resolution_reference(
            args.family_root,
            args.multires_reference_root,
            store,
            manifest,
            key,
            training_resolution=training_resolution,
        )
        reference_checks.append(check)
        retained_resolution = tuple(
            int(value) for value in reference["retained_resolution"]
        )
        frame_indices = stride * np.arange(args.rollout_calls + 1)
        if frame_indices[-1] >= reference["conservative_states"].shape[0]:
            raise ValueError(f"dynamic case {key} lacks the requested horizon")
        times = np.asarray(reference["physical_times"])[frame_indices]
        for resolution in resolutions:
            state_list = [
                reference_at_resolution(
                    reference["conservative_states"][frame],
                    reference_resolution=retained_resolution,
                    target_resolution=resolution,
                )
                for frame in frame_indices
            ]
            if any(value is None for value in state_list):
                raise ValueError("dynamic reference cannot restrict to active grid")
            states = np.asarray(state_list, dtype=np.float64)
            geometry = geometry_by_resolution[resolution]
            physical_type = np.asarray(
                physical_type_by_resolution[resolution], dtype=np.int64
            ).reshape(-1)
            cases.append(
                CaseData(
                    family="dynamic_fv",
                    case_id=key,
                    resolution=resolution,
                    sample=sample_by_resolution[resolution],
                    reference_states=np.asarray(
                        [as_model_state(value) for value in states], dtype=np.float64
                    ),
                    physical_times=times,
                    nodes=np.asarray(geometry.nodes, dtype=np.float64),
                    edges=np.asarray(geometry.directed_edges, dtype=np.int64),
                    weights=np.asarray(
                        geometry.node_measures, dtype=np.float64
                    ).reshape(-1),
                    physical_node_type=physical_type,
                    boundary_distance=_dynamic_boundary_distance(
                        geometry.nodes, resolution
                    ),
                    state_scale=np.asarray(
                        checkpoint["normalization"]["state_scale"], dtype=np.float64
                    ),
                    residual_scale=np.asarray(
                        checkpoint["normalization"]["residual_scale"],
                        dtype=np.float64,
                    ),
                    gamma=float(checkpoint["normalization"]["gamma"]),
                    provenance=family_case_provenance(manifest, key),
                )
            )
    return cases, reference_checks


def _final_summary_rows(
    rows: Sequence[Mapping[str, Any]], final_call: int
) -> list[dict[str, Any]]:
    groups: defaultdict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["call"] != final_call or row["region"] != "all":
            continue
        value = row.get("prediction_relative_l2")
        if value is not None:
            groups[(row["resolution"], row["mode"], row["intervention"])].append(
                float(value)
            )
    result = []
    for (resolution, mode, intervention), values in sorted(groups.items()):
        result.append(
            {
                "resolution": resolution,
                "mode": mode,
                "intervention": intervention,
                "case_count": len(values),
                "mean_final_relative_l2": float(np.mean(values)),
                "median_final_relative_l2": float(np.median(values)),
            }
        )
    return result


def _completion_summary_rows(
    rows: Sequence[Mapping[str, Any]],
    final_call: int,
) -> list[dict[str, Any]]:
    groups: defaultdict[tuple[str, str, str, str, str], list[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    for row in rows:
        key = (
            str(row["family"]),
            str(row["case_id"]),
            str(row["resolution"]),
            str(row["mode"]),
            str(row["intervention"]),
        )
        groups[key].append(row)
    result = []
    for key, values in sorted(groups.items()):
        ordered = sorted(values, key=lambda value: int(value["call"]))
        inadmissible = [
            int(value["call"])
            for value in ordered
            if not bool(value["finite"]) or not bool(value["admissible"])
        ]
        last_call = int(ordered[-1]["call"])
        result.append(
            {
                "family": key[0],
                "case_id": key[1],
                "resolution": key[2],
                "mode": key[3],
                "intervention": key[4],
                "calls_evaluated": len(ordered),
                "last_call_evaluated": last_call,
                "horizon_output_produced": last_call == final_call,
                "fully_admissible": not inadmissible,
                "admissible_rollout_completed": (
                    last_call == final_call and not inadmissible
                ),
                "first_inadmissible_call": (
                    None if not inadmissible else inadmissible[0]
                ),
                "minimum_density": min(
                    float(value["minimum_density"]) for value in ordered
                ),
                "minimum_pressure": min(
                    float(value["minimum_pressure"]) for value in ordered
                ),
            }
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    device = select_device(args.device)
    store = PCNOEuler2DShardStore(args.data_dir, max_cached_trajectories=2)
    if args.family == "dynamic_fv":
        checkpoint = load_resolution_checkpoint(args.checkpoint)
        model, _ = build_resolution_checkpoint_model(checkpoint, device)
    else:
        checkpoint = load_checkpoint_payload(args.checkpoint)
        if int(checkpoint.get("checkpoint_schema_version", -1)) != 4:
            raise ValueError("unsupported bump checkpoint schema")
        model, _ = build_checkpoint_model(
            checkpoint, device, model_node_type_input="physical"
        )
    provenance = _verify_common_provenance(args, checkpoint, store)

    reference_checks: list[dict[str, Any]] = []
    if args.family == "dynamic_fv":
        cases, reference_checks = _load_dynamic_cases(
            args, checkpoint, store, device=device
        )
    else:
        cases = _load_bump_cases(args, checkpoint, store, device=device)
    cases_by_key = {case.case_id: case for case in cases}

    bump_replay = None
    if args.family == "bump":
        bump_replay = _bump_replay_gate(args, model, cases_by_key, device=device)

    equivalence_cases = [
        case
        for case in cases
        if not args.trace_cases or case.case_id in set(args.trace_cases)
    ]
    if not equivalence_cases:
        equivalence_cases = [cases[0]]
    equivalence = [
        _hook_equivalence(
            model,
            case,
            device=device,
            amp=args.amp,
            absolute_limit=args.native_repeat_absolute_limit,
            relative_limit=args.native_repeat_relative_limit,
        )
        for case in equivalence_cases
    ]

    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "traces").mkdir()
    (args.output_dir / "animations").mkdir()
    outcome_rows: list[dict[str, Any]] = []
    completion_rows: list[dict[str, Any]] = []
    frequency_rows: list[dict[str, Any]] = []
    activation_rows: list[dict[str, Any]] = []
    trace_files: list[Path] = []
    animation_files: list[Path] = []
    specs = _intervention_specs(args.family, args.interventions)
    intervention_contract_rows = [
        row for case in cases for row in _intervention_contract_rows(model, case, specs)
    ]

    for index, case in enumerate(cases, start=1):
        print(
            f"case {index}/{len(cases)} {case.case_id} {case.resolution_name}",
            flush=True,
        )
        mode_payloads = {}
        for mode in args.modes:
            mode_payloads[mode] = _evaluate_mode(
                args,
                model,
                case,
                specs,
                mode=mode,
                device=device,
                outcome_rows=outcome_rows,
                completion_rows=completion_rows,
                frequency_rows=frequency_rows,
                activation_rows=activation_rows,
                trace_files=trace_files,
            )
        if case.case_id in set(args.trace_cases):
            animation_files.append(_save_animation_payload(args, case, mode_payloads))

    write_csv(args.output_dir / "outcome_metrics.csv", outcome_rows)
    write_csv(args.output_dir / "completion.csv", completion_rows)
    write_csv(args.output_dir / "frequency_metrics.csv", frequency_rows)
    write_csv(
        args.output_dir / "intervention_contract.csv",
        intervention_contract_rows,
    )
    if activation_rows:
        write_csv(args.output_dir / "activation_metrics.csv", activation_rows)
    if reference_checks:
        write_csv(args.output_dir / "reference_checks.csv", reference_checks)
    completion_summary_rows = _completion_summary_rows(
        completion_rows,
        args.rollout_calls,
    )
    write_csv(
        args.output_dir / "completion_summary.csv",
        completion_summary_rows,
    )
    final_rows = _final_summary_rows(outcome_rows, args.rollout_calls)
    write_csv(args.output_dir / "final_summary.csv", final_rows)
    provenance["loaded_project_source_sha256"] = _loaded_project_source_hashes()
    summary = {
        "schema": SCHEMA,
        "status": "complete",
        "family": args.family,
        "args": jsonable_args(args),
        "claim_boundary": {
            "causal": (
                "teacher-forced same-state rows change only model-facing node type"
            ),
            "free_rollout": (
                "free rows combine the initial type intervention with recurrently "
                "diverged state inputs"
            ),
            "branches": (
                "branch and hidden norms are diagnostics, not causal attribution"
            ),
            "bump_frequency": (
                "bump graph high-pass is not a physical-frequency measurement"
            ),
            "boundary": (
                "physical boundary conditions and model_all_nodes policy are frozen"
            ),
            "resolution": (
                "dynamic_fv uses regenerated physical grids; bump uses only each "
                "case's native graph and is not a PDE resolution-transfer test"
            ),
        },
        "input_contract": {
            "columns_0_1": "physical coordinates",
            "column_2": "quadrature density",
            "columns_3_6": "normalized conservative state",
            "columns_7_10": "family-local node-type one-hot",
            "column_11": "normalized Mach",
            "type_lift_identity": (
                "correct minus k-to-0 post-lift signal is "
                "W_type times (e_k - e_0) when all non-type inputs match"
            ),
            "node_type_semantics": FAMILY_NODE_TYPE_NAMES[args.family],
        },
        "provenance": provenance,
        "bump_open_validation_replay": bump_replay,
        "hook_equivalence": equivalence,
        "runtime": runtime_environment(device),
        "git": git_state(),
        "case_count": len(cases),
        "physical_case_count": len({case.case_id for case in cases}),
        "interventions": [spec.name for spec in specs],
        "modes": list(args.modes),
        "row_counts": {
            "outcome": len(outcome_rows),
            "completion": len(completion_rows),
            "completion_summary": len(completion_summary_rows),
            "frequency": len(frequency_rows),
            "activation": len(activation_rows),
            "intervention_contract": len(intervention_contract_rows),
            "reference_checks": len(reference_checks),
        },
        "final_summary": final_rows,
        "completion_summary": completion_summary_rows,
        "artifacts": {
            "trace_files": [path.name for path in trace_files],
            "animation_files": [path.name for path in animation_files],
        },
    }
    atomic_write_json(args.output_dir / "summary.json", summary)
    close = getattr(store, "close", None)
    if callable(close):
        close()
    print(f"wrote {args.output_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
