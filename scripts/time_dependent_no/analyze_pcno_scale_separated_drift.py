#!/usr/bin/env python3
"""Post-process saved PCNO cross-grid defects by spatial and temporal scale.

No model is loaded or executed.  Every input bundle must be present in, and
match the SHA-256 recorded by, one completed D064 or D065 summary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import scipy

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utility.time_dependent_no.pcno_scale_separated_drift import (
    band_contract,
    pathway_scale_region_diagnostics,
    scale_component_diagnostics,
    scale_separated_diagnostics,
    shock_profile_diagnostics,
)

SCHEMA = "pcno_scale_separated_drift_diagnostic_v2"
EXPECTED_BOUNDARY_POLICY = "model_all_nodes raw recurrence"
SUPPORTED_INPUT_SCHEMAS = {
    "pcno_residual_structure_diagnostic_v1",
    "pcno_resolution_pathway_diagnostic_v2",
}
PAIR_PATTERN = re.compile(
    r"^(?P<coarse_x>\d+)x(?P<coarse_y>\d+)_to_" r"(?P<fine_x>\d+)x(?P<fine_y>\d+)$"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-summary", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pathway-summary", type=Path)
    parser.add_argument("--pathway-bundle-dir", type=Path)
    parser.add_argument("--domain-lengths", type=float, nargs=2, default=(2.0, 1.0))
    parser.add_argument("--max-lag", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=1.4)
    parser.add_argument(
        "--replay-increment-scaled-rms-tolerance", type=float, default=2.0e-2
    )
    parser.add_argument(
        "--replay-cumulative-scaled-rms-tolerance", type=float, default=1.0e-1
    )
    parser.add_argument("--replay-relative-rms-tolerance", type=float, default=5.0e-2)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_ready(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path.name}")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(_json_ready(value), sort_keys=True)
                        if isinstance(value, (Mapping, list, tuple, np.ndarray))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def _scalar_text(value: np.ndarray) -> str:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError("expected a scalar string bundle field")
    return str(array.item())


def _load_input_summary(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise ValueError("input summary is not complete")
    if summary.get("schema") not in SUPPORTED_INPUT_SCHEMAS:
        raise ValueError("unsupported input diagnostic schema")
    if summary.get("boundary_policy") != EXPECTED_BOUNDARY_POLICY:
        raise ValueError("the frozen physical boundary policy changed")
    population = summary.get("population", {})
    if population.get("split") != "validation":
        raise ValueError("only the already-open validation population is allowed")
    if population.get("sealed_populations_accessed") is not False:
        raise ValueError("sealed-population provenance is not acceptable")
    return summary


def _verify_bundles(
    summary: Mapping[str, Any], bundle_dir: Path
) -> tuple[list[Path], dict[str, str]]:
    registered = {
        Path(name).name: str(digest).lower()
        for name, digest in summary.get("output_hashes", {}).items()
        if str(name).replace("\\", "/").startswith("bundles/")
    }
    if not registered:
        raise ValueError("input summary does not register any field bundle")
    present = {path.name: path for path in bundle_dir.glob("*.npz")}
    missing = sorted(set(registered) - set(present))
    unexpected = sorted(set(present) - set(registered))
    if missing:
        raise FileNotFoundError(
            "registered field bundles are missing: " + ", ".join(missing)
        )
    if unexpected:
        raise ValueError(
            "unregistered field bundles are present: " + ", ".join(unexpected)
        )
    verified: dict[str, str] = {}
    for name in sorted(registered):
        digest = sha256_file(present[name])
        if digest.lower() != registered[name]:
            raise ValueError(f"field bundle SHA-256 mismatch: {name}")
        verified[name] = digest
    return [present[name] for name in sorted(registered)], verified


def _bundle_contract(
    bundle: Mapping[str, np.ndarray],
    *,
    input_schema: str,
) -> tuple[str, str, np.ndarray, np.ndarray, dict[str, str]]:
    if input_schema == "pcno_residual_structure_diagnostic_v1":
        metadata = json.loads(_scalar_text(bundle["metadata_json"]))
        case_id = str(metadata["case_id"])
        if str(metadata["schema"]) != input_schema:
            raise ValueError("bundle/input summary schema mismatch")
        checkpoint = str(metadata["checkpoint_sha256"])
        if metadata.get("bundle_payload", "standard") == "unified":
            sequence_prefixes = {"free_cross_grid_defect": "delta_free__"}
        else:
            sequence_prefixes = {
                "free_cross_grid_defect": "delta_free__",
                "teacher_exact_input_cross_grid_defect": "delta_teacher__",
            }
    else:
        case_id = _scalar_text(bundle["case_id"])
        if _scalar_text(bundle["schema"]) != input_schema:
            raise ValueError("bundle/input summary schema mismatch")
        checkpoint = _scalar_text(bundle["checkpoint_sha256"])
        sequence_prefixes = {
            "teacher_exact_input_cross_grid_defect": "baseline_defect__"
        }
    return (
        case_id,
        checkpoint,
        np.asarray(bundle["physical_times"], dtype=np.float64),
        np.asarray(bundle["residual_scale"], dtype=np.float64),
        sequence_prefixes,
    )


def _pair_keys(
    bundle: Mapping[str, np.ndarray], prefixes: Mapping[str, str]
) -> list[str]:
    keys = {
        key[len(prefix) :]
        for prefix in prefixes.values()
        for key in bundle
        if key.startswith(prefix)
    }
    if not keys:
        raise ValueError("bundle contains no supported cross-grid defect fields")
    for key in keys:
        if PAIR_PATTERN.fullmatch(key) is None:
            raise ValueError(f"invalid pair key: {key}")
        if f"true_increment__{key}" not in bundle:
            raise ValueError(f"bundle lacks the paired true increment: {key}")
    return sorted(keys)


def _pair_contract(key: str) -> tuple[str, tuple[int, int]]:
    match = PAIR_PATTERN.fullmatch(key)
    if match is None:
        raise ValueError(f"invalid pair key: {key}")
    coarse = (int(match["coarse_x"]), int(match["coarse_y"]))
    target = (
        f"{match['coarse_x']}x{match['coarse_y']}->{match['fine_x']}x{match['fine_y']}"
    )
    return target, coarse


def _pathway_pair_keys(bundle: Mapping[str, np.ndarray]) -> list[str]:
    keys = sorted(
        key.removeprefix("mesh_delta__")
        for key in bundle
        if key.startswith("mesh_delta__")
    )
    if not keys:
        raise ValueError("pathway bundle contains no mesh/state decomposition")
    required_prefixes = (
        "nodes__",
        "volumes__",
        "reference_states__",
        "free_coarse_states__",
        "free_fine_states__",
        "mesh_delta__",
        "state_delta__",
    )
    for key in keys:
        if PAIR_PATTERN.fullmatch(key) is None:
            raise ValueError(f"invalid pathway pair key: {key}")
        missing = [
            prefix for prefix in required_prefixes if f"{prefix}{key}" not in bundle
        ]
        if missing:
            raise ValueError(f"incomplete pathway bundle for {key}: {missing}")
    return keys


def _replay_row(
    parent: np.ndarray,
    replay: np.ndarray,
    *,
    field: str,
    residual_scale: np.ndarray,
) -> dict[str, Any]:
    parent_array = np.asarray(parent, dtype=np.float64)
    replay_array = np.asarray(replay, dtype=np.float64)
    if parent_array.shape != replay_array.shape:
        raise ValueError(f"parent/replay shape mismatch for {field}")
    difference = replay_array - parent_array
    scaled_difference = difference / residual_scale[None, None, :]
    scaled_parent = parent_array / residual_scale[None, None, :]
    step_rms = np.sqrt(np.mean(np.sum(np.square(scaled_difference), axis=2), axis=1))
    difference_energy = float(np.sum(np.square(scaled_difference)))
    parent_energy = float(np.sum(np.square(scaled_parent)))
    return {
        "field": field,
        "maximum_absolute_difference": float(np.max(np.abs(difference))),
        "maximum_step_residual_scaled_rms": float(np.max(step_rms)),
        "aggregate_residual_scaled_rms": float(
            np.sqrt(np.mean(np.square(scaled_difference)))
        ),
        "relative_residual_scaled_rms": (
            float(np.sqrt(difference_energy / parent_energy))
            if parent_energy > 0.0
            else (0.0 if difference_energy == 0.0 else None)
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.max_lag < 1:
        raise ValueError("max_lag must be positive")
    if (args.pathway_summary is None) != (args.pathway_bundle_dir is None):
        raise ValueError(
            "pathway summary and bundle directory must be supplied together"
        )
    if args.gamma <= 1.0:
        raise ValueError("gamma must exceed one")
    replay_tolerances = (
        args.replay_increment_scaled_rms_tolerance,
        args.replay_cumulative_scaled_rms_tolerance,
        args.replay_relative_rms_tolerance,
    )
    if any(value <= 0.0 for value in replay_tolerances):
        raise ValueError("replay tolerances must be positive")
    domain_lengths = tuple(float(value) for value in args.domain_lengths)
    if any(value <= 0.0 for value in domain_lengths):
        raise ValueError("domain lengths must be positive")
    input_summary = _load_input_summary(args.input_summary)
    bundle_paths, verified_bundles = _verify_bundles(input_summary, args.bundle_dir)
    pathway_summary: dict[str, Any] | None = None
    pathway_bundle_paths: list[Path] = []
    verified_pathway_bundles: dict[str, str] = {}
    if args.pathway_summary is not None and args.pathway_bundle_dir is not None:
        pathway_summary = _load_input_summary(args.pathway_summary)
        if pathway_summary.get("bundle_payload") not in {"pathway", "unified"}:
            raise ValueError("the extension summary is not a pathway payload")
        pathway_bundle_paths, verified_pathway_bundles = _verify_bundles(
            pathway_summary, args.pathway_bundle_dir
        )
    args.output_dir.mkdir(parents=True)

    input_schema = str(input_summary["schema"])
    expected_checkpoint = str(input_summary["checkpoint_sha256"])
    expected_scale = np.asarray(input_summary["residual_scale"], dtype=np.float64)
    registered_cases = set(input_summary["population"]["case_ids"])
    time_contract = (
        input_summary["time_contract"]
        if input_schema == "pcno_residual_structure_diagnostic_v1"
        else input_summary["contract"]
    )
    expected_dt = float(time_contract["physical_dt"])
    if pathway_summary is not None:
        if pathway_summary.get("schema") != "pcno_residual_structure_diagnostic_v1":
            raise ValueError("pathway replay must use the D064 residual schema")
        for field in (
            "checkpoint_sha256",
            "normalization_digest",
            "boundary_policy",
            "source_hashes",
            "time_contract",
        ):
            if pathway_summary.get(field) != input_summary.get(field):
                raise ValueError(f"parent/pathway contract mismatch: {field}")
        pathway_cases = set(pathway_summary["population"]["case_ids"])
        if not pathway_cases.issubset(registered_cases):
            raise ValueError(
                "pathway replay includes a case outside the parent population"
            )
    time_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    component_time_rows: list[dict[str, Any]] = []
    component_aggregate_rows: list[dict[str, Any]] = []
    closure_rows: list[dict[str, Any]] = []
    evaluated_cases: list[str] = []
    sequence_kinds: set[str] = set()
    pathway_time_rows: list[dict[str, Any]] = []
    pathway_aggregate_rows: list[dict[str, Any]] = []
    pathway_closure_rows: list[dict[str, Any]] = []
    shock_time_rows: list[dict[str, Any]] = []
    shock_aggregate_rows: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []

    for bundle_path in bundle_paths:
        with np.load(bundle_path, allow_pickle=False) as bundle:
            case_id, checkpoint, physical_times, residual_scale, prefixes = (
                _bundle_contract(bundle, input_schema=input_schema)
            )
            if case_id != bundle_path.stem:
                raise ValueError(f"bundle filename/case mismatch: {bundle_path.name}")
            if case_id not in registered_cases:
                raise ValueError(
                    f"bundle case is outside the registered population: {case_id}"
                )
            if checkpoint != expected_checkpoint:
                raise ValueError(f"checkpoint mismatch in bundle: {bundle_path.name}")
            if not np.array_equal(residual_scale, expected_scale):
                raise ValueError(
                    f"residual-scale mismatch in bundle: {bundle_path.name}"
                )
            if physical_times.ndim != 1 or np.any(np.diff(physical_times) <= 0.0):
                raise ValueError(
                    f"invalid physical times in bundle: {bundle_path.name}"
                )
            expected_times = (
                np.arange(1, physical_times.size + 1, dtype=np.float64) * expected_dt
            )
            if not np.array_equal(physical_times, expected_times):
                raise ValueError(f"physical-time contract mismatch: {bundle_path.name}")
            evaluated_cases.append(case_id)
            for pair_key in _pair_keys(bundle, prefixes):
                target, resolution = _pair_contract(pair_key)
                truth = np.asarray(
                    bundle[f"true_increment__{pair_key}"], dtype=np.float64
                )
                if truth.shape[0] != physical_times.size:
                    raise ValueError("truth sequence and physical times do not align")
                for sequence_kind, prefix in prefixes.items():
                    field_key = f"{prefix}{pair_key}"
                    if field_key not in bundle:
                        raise ValueError(f"incomplete sequence family: {field_key}")
                    defects = np.asarray(bundle[field_key], dtype=np.float64)
                    rows, aggregates, closure, projections = (
                        scale_separated_diagnostics(
                            defects,
                            truth,
                            resolution=resolution,
                            component_scale=residual_scale,
                            domain_lengths=domain_lengths,
                            max_lag=args.max_lag,
                            return_projections=True,
                        )
                    )
                    if projections is None:
                        raise AssertionError("scale projections were not returned")
                    component_rows, component_aggregates, component_closure = (
                        scale_component_diagnostics(
                            defects,
                            truth,
                            projections,
                            component_scale=residual_scale,
                            max_lag=args.max_lag,
                        )
                    )
                    closure.update(component_closure)
                    sequence_kinds.add(sequence_kind)
                    for row in rows:
                        step = int(row["step"])
                        time_rows.append(
                            {
                                "case_id": case_id,
                                "sequence_kind": sequence_kind,
                                "target": target,
                                "physical_time": float(physical_times[step - 1]),
                                **row,
                            }
                        )
                    for row in aggregates:
                        aggregate_rows.append(
                            {
                                "case_id": case_id,
                                "sequence_kind": sequence_kind,
                                "target": target,
                                **row,
                            }
                        )
                    for row in component_rows:
                        step = int(row["step"])
                        component_time_rows.append(
                            {
                                "case_id": case_id,
                                "sequence_kind": sequence_kind,
                                "target": target,
                                "physical_time": float(physical_times[step - 1]),
                                **row,
                            }
                        )
                    for row in component_aggregates:
                        component_aggregate_rows.append(
                            {
                                "case_id": case_id,
                                "sequence_kind": sequence_kind,
                                "target": target,
                                **row,
                            }
                        )
                    closure_rows.append(
                        {
                            "case_id": case_id,
                            "sequence_kind": sequence_kind,
                            "target": target,
                            **closure,
                        }
                    )

    if pathway_summary is not None:
        parent_by_case = {path.stem: path for path in bundle_paths}
        for pathway_path in pathway_bundle_paths:
            if pathway_path.stem not in parent_by_case:
                raise ValueError(
                    f"pathway bundle lacks a registered parent: {pathway_path.name}"
                )
            parent_path = parent_by_case[pathway_path.stem]
            with (
                np.load(parent_path, allow_pickle=False) as parent,
                np.load(pathway_path, allow_pickle=False) as pathway,
            ):
                parent_case, parent_checkpoint, parent_times, parent_scale, prefixes = (
                    _bundle_contract(parent, input_schema=input_schema)
                )
                pathway_case, pathway_checkpoint, pathway_times, pathway_scale, _ = (
                    _bundle_contract(
                        pathway,
                        input_schema="pcno_residual_structure_diagnostic_v1",
                    )
                )
                pathway_metadata = json.loads(_scalar_text(pathway["metadata_json"]))
                if pathway_metadata.get("bundle_payload") not in {
                    "pathway",
                    "unified",
                }:
                    raise ValueError(f"bundle is not a pathway payload: {pathway_path}")
                if parent_case != pathway_case or pathway_case != pathway_path.stem:
                    raise ValueError("parent/pathway case identity mismatch")
                if parent_checkpoint != pathway_checkpoint:
                    raise ValueError("parent/pathway checkpoint mismatch")
                if not np.array_equal(parent_times, pathway_times):
                    raise ValueError("parent/pathway physical times mismatch")
                if not np.array_equal(parent_scale, pathway_scale):
                    raise ValueError("parent/pathway residual scale mismatch")
                parent_pairs = set(_pair_keys(parent, prefixes))
                pathway_pairs = set(_pathway_pair_keys(pathway))
                if pathway_pairs != parent_pairs:
                    raise ValueError("parent/pathway mesh-pair sets differ")

                for pair_key in sorted(pathway_pairs):
                    target, resolution = _pair_contract(pair_key)
                    nodes = np.asarray(pathway[f"nodes__{pair_key}"], dtype=np.float64)
                    volumes = np.asarray(
                        pathway[f"volumes__{pair_key}"], dtype=np.float64
                    )
                    reference_states = np.asarray(
                        pathway[f"reference_states__{pair_key}"], dtype=np.float64
                    )
                    coarse_states = np.asarray(
                        pathway[f"free_coarse_states__{pair_key}"], dtype=np.float64
                    )
                    fine_states = np.asarray(
                        pathway[f"free_fine_states__{pair_key}"], dtype=np.float64
                    )
                    mesh_delta = np.asarray(
                        pathway[f"mesh_delta__{pair_key}"], dtype=np.float64
                    )
                    state_delta = np.asarray(
                        pathway[f"state_delta__{pair_key}"], dtype=np.float64
                    )
                    replay_true = np.diff(reference_states, axis=0)
                    replay_coarse = np.diff(coarse_states, axis=0)
                    replay_fine = np.diff(fine_states, axis=0)
                    replay_total = replay_coarse - replay_fine
                    replay_cumulative = np.cumsum(replay_total, axis=0)
                    replay_fields = (
                        (
                            "true_increment",
                            parent[f"true_increment__{pair_key}"],
                            replay_true,
                        ),
                        (
                            "coarse_increment_free",
                            parent[f"coarse_increment_free__{pair_key}"],
                            replay_coarse,
                        ),
                        (
                            "fine_increment_free",
                            parent[f"fine_increment_free__{pair_key}"],
                            replay_fine,
                        ),
                        (
                            "delta_free",
                            parent[f"delta_free__{pair_key}"],
                            replay_total,
                        ),
                        (
                            "cumulative_delta_free",
                            parent[f"cumulative_delta_free__{pair_key}"],
                            replay_cumulative,
                        ),
                    )
                    for field, parent_value, replay_value in replay_fields:
                        replay_rows.append(
                            {
                                "case_id": pathway_case,
                                "target": target,
                                **_replay_row(
                                    parent_value,
                                    replay_value,
                                    field=field,
                                    residual_scale=parent_scale,
                                ),
                            }
                        )

                    rows, aggregates, closure, _ = pathway_scale_region_diagnostics(
                        replay_total,
                        mesh_delta,
                        state_delta,
                        reference_states,
                        nodes,
                        volumes,
                        resolution=resolution,
                        component_scale=parent_scale,
                        gamma=args.gamma,
                        domain_lengths=domain_lengths,
                    )
                    for row in rows:
                        step = int(row["step"])
                        pathway_time_rows.append(
                            {
                                "case_id": pathway_case,
                                "sequence_kind": "free_cross_grid_defect",
                                "target": target,
                                "physical_time": float(parent_times[step - 1]),
                                **row,
                            }
                        )
                    for row in aggregates:
                        pathway_aggregate_rows.append(
                            {
                                "case_id": pathway_case,
                                "sequence_kind": "free_cross_grid_defect",
                                "target": target,
                                **row,
                            }
                        )
                    pathway_closure_rows.append(
                        {
                            "case_id": pathway_case,
                            "sequence_kind": "free_cross_grid_defect",
                            "target": target,
                            **closure,
                        }
                    )
                    profile_rows, profile_aggregates = shock_profile_diagnostics(
                        reference_states,
                        coarse_states,
                        fine_states,
                        nodes,
                        volumes,
                        resolution=resolution,
                        gamma=args.gamma,
                        component_scale=parent_scale,
                    )
                    for row in profile_rows:
                        shock_time_rows.append(
                            {
                                "case_id": pathway_case,
                                "target": target,
                                "physical_time": float(row["frame"] * expected_dt),
                                **row,
                            }
                        )
                    for row in profile_aggregates:
                        shock_aggregate_rows.append(
                            {"case_id": pathway_case, "target": target, **row}
                        )

    _write_csv(args.output_dir / "scale_time_metrics.csv", time_rows)
    _write_csv(args.output_dir / "scale_aggregate_metrics.csv", aggregate_rows)
    _write_csv(
        args.output_dir / "scale_component_time_metrics.csv", component_time_rows
    )
    _write_csv(
        args.output_dir / "scale_component_aggregate_metrics.csv",
        component_aggregate_rows,
    )
    _write_csv(args.output_dir / "scale_closure_metrics.csv", closure_rows)
    if pathway_summary is not None:
        _write_csv(
            args.output_dir / "pathway_scale_region_time_metrics.csv",
            pathway_time_rows,
        )
        _write_csv(
            args.output_dir / "pathway_scale_region_aggregate_metrics.csv",
            pathway_aggregate_rows,
        )
        _write_csv(
            args.output_dir / "pathway_scale_region_closure_metrics.csv",
            pathway_closure_rows,
        )
        _write_csv(args.output_dir / "shock_profile_time_metrics.csv", shock_time_rows)
        _write_csv(
            args.output_dir / "shock_profile_aggregate_metrics.csv",
            shock_aggregate_rows,
        )
        _write_csv(args.output_dir / "replay_consistency_metrics.csv", replay_rows)
    output_hashes = {
        path.name: sha256_file(path) for path in sorted(args.output_dir.glob("*.csv"))
    }
    reconstruction_max = max(
        max(
            float(row["maximum_reconstruction_abs_residual_scaled"]),
            float(row["maximum_component_band_reconstruction_abs_residual_scaled"]),
        )
        for row in closure_rows
    )
    energy_max = max(
        max(
            float(row["maximum_instantaneous_energy_relative_closure"]),
            float(row["maximum_component_band_instantaneous_energy_relative_closure"]),
        )
        for row in closure_rows
    )
    cumulative_energy_max = max(
        max(
            float(row["maximum_cumulative_energy_relative_closure"]),
            float(row["maximum_component_band_cumulative_energy_relative_closure"]),
        )
        for row in closure_rows
    )
    growth_max = max(
        float(value)
        for row in closure_rows
        for key, value in row.items()
        if key.endswith("signed_growth_absolute_closure")
    )
    tolerances = {
        "reconstruction_abs_residual_scaled": 5.0e-12,
        "energy_relative_closure": 5.0e-12,
        "signed_growth_absolute_closure": 5.0e-10,
    }
    checks = {
        "maximum_reconstruction_abs_residual_scaled": reconstruction_max,
        "maximum_instantaneous_energy_relative_closure": energy_max,
        "maximum_cumulative_energy_relative_closure": cumulative_energy_max,
        "maximum_signed_growth_absolute_closure": growth_max,
    }
    base_checks_passed = (
        reconstruction_max <= tolerances["reconstruction_abs_residual_scaled"]
        and energy_max <= tolerances["energy_relative_closure"]
        and cumulative_energy_max <= tolerances["energy_relative_closure"]
        and growth_max <= tolerances["signed_growth_absolute_closure"]
    )
    pathway_checks: dict[str, float | bool | None] = {
        "maximum_total_mesh_state_reconstruction_abs_residual_scaled": None,
        "maximum_band_energy_relative_closure": None,
        "maximum_pathway_or_partition_absolute_closure": None,
        "maximum_reference_replay_step_residual_scaled_rms": None,
        "maximum_increment_replay_step_residual_scaled_rms": None,
        "maximum_cumulative_replay_step_residual_scaled_rms": None,
        "maximum_replay_relative_residual_scaled_rms": None,
        "pathway_checks_passed": True,
        "replay_checks_passed": True,
    }
    if pathway_summary is not None:
        pathway_reconstruction = max(
            float(row["maximum_total_mesh_state_reconstruction_abs_residual_scaled"])
            for row in pathway_closure_rows
        )
        pathway_relative = max(
            max(
                float(row["maximum_band_instantaneous_energy_relative_closure"]),
                float(row["maximum_band_cumulative_energy_relative_closure"]),
            )
            for row in pathway_closure_rows
        )
        pathway_absolute = max(
            max(
                float(row["maximum_pathway_instantaneous_energy_absolute_closure"]),
                float(row["maximum_pathway_cumulative_energy_absolute_closure"]),
                float(row["maximum_pathway_signed_growth_absolute_closure"]),
                float(row["maximum_physical_partition_absolute_closure"]),
                float(row["maximum_band_signed_growth_absolute_closure"]),
            )
            for row in pathway_closure_rows
        )
        reference_replay = max(
            float(row["maximum_step_residual_scaled_rms"])
            for row in replay_rows
            if row["field"] == "true_increment"
        )
        increment_replay = max(
            float(row["maximum_step_residual_scaled_rms"])
            for row in replay_rows
            if row["field"] not in {"true_increment", "cumulative_delta_free"}
        )
        cumulative_replay = max(
            float(row["maximum_step_residual_scaled_rms"])
            for row in replay_rows
            if row["field"] == "cumulative_delta_free"
        )
        replay_relative = max(
            float(row["relative_residual_scaled_rms"])
            for row in replay_rows
            if row["relative_residual_scaled_rms"] is not None
        )
        pathway_passed = (
            pathway_reconstruction <= 5.0e-5
            and pathway_relative <= 5.0e-10
            and pathway_absolute <= 5.0e-8
        )
        replay_passed = (
            reference_replay <= 1.0e-5
            and increment_replay <= args.replay_increment_scaled_rms_tolerance
            and cumulative_replay <= args.replay_cumulative_scaled_rms_tolerance
            and replay_relative <= args.replay_relative_rms_tolerance
        )
        pathway_checks = {
            "maximum_total_mesh_state_reconstruction_abs_residual_scaled": (
                pathway_reconstruction
            ),
            "maximum_band_energy_relative_closure": pathway_relative,
            "maximum_pathway_or_partition_absolute_closure": pathway_absolute,
            "maximum_reference_replay_step_residual_scaled_rms": reference_replay,
            "maximum_increment_replay_step_residual_scaled_rms": increment_replay,
            "maximum_cumulative_replay_step_residual_scaled_rms": cumulative_replay,
            "maximum_replay_relative_residual_scaled_rms": replay_relative,
            "pathway_checks_passed": pathway_passed,
            "replay_checks_passed": replay_passed,
        }
    checks_passed = (
        base_checks_passed
        and bool(pathway_checks["pathway_checks_passed"])
        and bool(pathway_checks["replay_checks_passed"])
    )
    utility_path = REPO_ROOT / "utility/time_dependent_no/pcno_scale_separated_drift.py"
    summary = {
        "schema": SCHEMA,
        "status": "complete" if checks_passed else "failed_closure",
        "scientific_interpretation_allowed": checks_passed,
        "input": {
            "schema": input_schema,
            "summary_sha256": sha256_file(args.input_summary),
            "bundle_sha256": verified_bundles,
            "checkpoint_sha256": expected_checkpoint,
            "normalization_digest": input_summary.get("normalization_digest"),
            "pathway_replay": (
                None
                if pathway_summary is None
                else {
                    "summary_sha256": sha256_file(args.pathway_summary),
                    "bundle_sha256": verified_pathway_bundles,
                    "diagnostic_source_hashes": pathway_summary.get(
                        "diagnostic_source_hashes"
                    ),
                }
            ),
        },
        "population": {
            "split": "validation",
            "case_ids": evaluated_cases,
            "sealed_populations_accessed": False,
            "aggregation": "per case and mesh pair before across-case summaries",
        },
        "boundary_policy": EXPECTED_BOUNDARY_POLICY,
        "contract": {
            "transform": "orthonormal cell-centered DCT-II; reflective/nonperiodic",
            "domain_lengths": domain_lengths,
            "bands": band_contract(),
            "component_scale": "frozen checkpoint residual_scale",
            "component_decomposition": {
                "components": ["rho", "rho_u", "rho_v", "energy"],
                "energy": "uniform cell mean of squared residual-scaled field",
                "shares": (
                    "per case and mesh pair over the full band-component partition"
                ),
            },
            "uniform_cell_measure": True,
            "sequence_kinds": sorted(sequence_kinds),
            "max_lag": args.max_lag,
            "gamma": args.gamma,
            "pathway_decomposition": {
                "identity": "delta_free = delta_mesh + delta_state",
                "cross_term": "allocated symmetrically to mesh and state",
                "regions": [
                    "fixed physical boundary distance <= 0.05",
                    "reference shock envelope <= 0.05 excluding boundary",
                    "reference vortex core <= 0.18 excluding boundary and shock",
                    "remaining smooth region",
                ],
                "regional_growth_semantics": (
                    "input-frame mask localizes signed growth density; it is not "
                    "a finite difference of a moving-region energy"
                ),
            },
            "shock_profile": {
                "position": "pressure-gradient total-variation centroid in a fixed 0.08 reference-centered window",
                "strength": "net pressure jump in the same window",
                "profile_variation": "total pressure variation in the same window",
                "thickness": "pressure-gradient second-moment width",
                "modes": [
                    "negative reference x derivative (translation)",
                    "reference-centered derivative dilation",
                    "reference plateau-centered amplitude",
                ],
                "projection_component_scale": "frozen checkpoint residual_scale",
            },
            "teacher_cumulative_semantics": (
                "formal signed sum of exact-input defects, not a free-rollout state gap"
            ),
        },
        "checks": {**checks, **pathway_checks},
        "tolerances": {
            **tolerances,
            "pathway_reconstruction_abs_residual_scaled": 5.0e-5,
            "pathway_band_energy_relative_closure": 5.0e-10,
            "pathway_absolute_closure": 5.0e-8,
            "reference_replay_step_residual_scaled_rms": 1.0e-5,
            "increment_replay_step_residual_scaled_rms": (
                args.replay_increment_scaled_rms_tolerance
            ),
            "cumulative_replay_step_residual_scaled_rms": (
                args.replay_cumulative_scaled_rms_tolerance
            ),
            "replay_relative_residual_scaled_rms": (args.replay_relative_rms_tolerance),
        },
        "checks_passed": checks_passed,
        "diagnostic_source_hashes": {
            "runner": sha256_file(Path(__file__).resolve()),
            "utility": sha256_file(utility_path),
        },
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "device": "cpu",
        },
        "row_counts": {
            "time": len(time_rows),
            "aggregate": len(aggregate_rows),
            "component_time": len(component_time_rows),
            "component_aggregate": len(component_aggregate_rows),
            "closure": len(closure_rows),
            "pathway_time": len(pathway_time_rows),
            "pathway_aggregate": len(pathway_aggregate_rows),
            "pathway_closure": len(pathway_closure_rows),
            "shock_profile_time": len(shock_time_rows),
            "shock_profile_aggregate": len(shock_aggregate_rows),
            "replay_consistency": len(replay_rows),
        },
        "output_hashes": output_hashes,
    }
    _write_json(args.output_dir / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    summary = run(parse_args(argv))
    print(json.dumps(_json_ready(summary), indent=2, sort_keys=True))
    return 0 if summary["checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
