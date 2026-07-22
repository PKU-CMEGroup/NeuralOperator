"""Audit released CPGNet dependency support against Euler characteristic reach.

The input is an enriched HDF5 rollout written by evaluate_cpg_release.py.  The
diagnostic is frozen-checkpoint and rollout-free: it traces the pinned model
architecture, validates the dataset-to-message-graph identity, and compares the
resulting 13-hop support with endpoint-sampled characteristic travel on each
selected reference interval.

If the directed characteristic cone exceeds model support, the script may save
a pointwise-admissible graph-state pair.  That pair is only an input candidate;
the causal label must come from the validated reference solver under legal
boundaries after graph nodes have been mapped to solver degrees of freedom.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from utility.time_dependent_no.cpg_reach import (  # noqa: E402
    DEFAULT_GAMMA,
    REACH_SCHEMA,
    build_incoming_travel_adjacency,
    compare_characteristic_and_model_reach,
    construct_pointwise_admissible_causal_pair,
    cpg_dependency_trace,
    euler_characteristic_travel_times,
    graph_hop_distances,
    minimum_predecessor_times,
    node_pressure_jump_scores,
    select_reach_targets,
    validate_release_graph_mapping,
)
from utility.time_dependent_no.cpg_release import (  # noqa: E402
    CPG_REFERENCE_COMMIT,
    sha256_file,
)


CANONICAL_FRAMES = (0, 20, 40, 58, 78)
MESSAGE_PASSING_LAYERS = 12


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        action="append",
        required=True,
        help="Enriched release result HDF5, or a directory containing result/*.h5",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--frames",
        type=int,
        nargs="+",
        help="Reference interval indices; defaults to 0,20,40,58,78 when present",
    )
    parser.add_argument(
        "--target-node",
        type=int,
        action="append",
        help="Explicit target node; otherwise deterministic region targets are used",
    )
    parser.add_argument("--targets-per-region", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--max-causal-pairs", type=int, default=4)
    parser.add_argument("--causal-pair-amplitude", type=float, default=0.01)
    parser.add_argument("--causal-pair-patch-hops", type=int, default=0)
    return parser.parse_args(argv)


def artifact_paths(values: Sequence[Path]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        if value.is_file():
            paths.append(value)
            continue
        result_dir = value / "result" if (value / "result").is_dir() else value
        paths.extend(
            sorted(
                result_dir.glob("*.h5"),
                key=lambda path: (
                    0 if path.stem.isdigit() else 1,
                    int(path.stem) if path.stem.isdigit() else path.name,
                ),
            )
        )
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    if not unique:
        raise FileNotFoundError("no HDF5 artifacts were found")
    return unique


def selected_frames(total_steps: int, requested: Sequence[int] | None) -> list[int]:
    frames = list(CANONICAL_FRAMES if requested is None else requested)
    if any(frame < 0 for frame in frames):
        raise ValueError("frame indices must be nonnegative")
    if requested is not None and any(frame >= total_steps for frame in frames):
        raise ValueError(f"requested frame lies outside a {total_steps}-step artifact")
    frames = list(dict.fromkeys(frame for frame in frames if frame < total_steps))
    if not frames:
        raise ValueError(f"no requested frame lies in a {total_steps}-step artifact")
    return frames


def _read_artifact(path: Path) -> dict[str, Any]:
    required = (
        "reference_current",
        "targets",
        "pos",
        "edges",
        "node_type",
        "Mach",
        "injection_mask",
        "model_pos",
        "directed_edges",
        "edge_attr_before_model",
    )
    with h5py.File(path, "r") as handle:
        missing = [name for name in required if name not in handle]
        if missing:
            raise KeyError(f"{path} is missing enriched datasets {missing}")
        arrays = {name: np.asarray(handle[name]) for name in required}
        attributes = {
            str(name): _json_scalar(value) for name, value in handle.attrs.items()
        }

    current = np.asarray(arrays["reference_current"], dtype=np.float64)
    target = np.asarray(arrays["targets"], dtype=np.float64)
    if current.shape != target.shape or current.ndim != 3 or current.shape[-1] != 4:
        raise ValueError(
            f"reference_current and targets need matching (T,N,4) shape, got "
            f"{current.shape} and {target.shape}"
        )
    nodes = current.shape[1]
    node_type = np.asarray(arrays["node_type"]).reshape(-1).astype(np.int64)
    injection = np.asarray(arrays["injection_mask"]).reshape(-1).astype(bool)
    mach = np.asarray(arrays["Mach"], dtype=np.float64).reshape(-1)
    if any(value.shape != (nodes,) for value in (node_type, injection, mach)):
        raise ValueError("node_type, injection_mask, and Mach must match node count")
    if not np.array_equal(injection, node_type != 0):
        raise ValueError("injection mask does not match every non-normal node")
    reference_commit = str(attributes.get("reference_commit", ""))
    if reference_commit != CPG_REFERENCE_COMMIT:
        raise ValueError(
            f"artifact reference commit {reference_commit!r} is not pinned commit "
            f"{CPG_REFERENCE_COMMIT}"
        )
    if str(attributes.get("boundary_mode", "")) != "oracle_next_reference":
        raise ValueError("artifact does not use the audited release boundary contract")
    macro_dt = float(attributes.get("dt", 0.0))
    if macro_dt <= 0.0:
        raise ValueError("artifact must record a positive dt attribute")

    mapping = validate_release_graph_mapping(
        raw_pos=arrays["pos"],
        raw_edges=arrays["edges"],
        model_pos=arrays["model_pos"],
        directed_edges=arrays["directed_edges"],
        edge_attr_before_model=arrays["edge_attr_before_model"],
    )
    return {
        "reference_current": current,
        "targets": target,
        "pos": np.asarray(arrays["pos"], dtype=np.float64)[:, :2],
        "edges": np.asarray(arrays["edges"], dtype=np.int64),
        "node_type": node_type,
        "injection_mask": injection,
        "Mach": mach,
        "attributes": attributes,
        "macro_dt": macro_dt,
        "mapping": mapping,
    }


def _boundary_distance(edges: np.ndarray, injection_mask: np.ndarray) -> np.ndarray:
    nodes = injection_mask.size
    sources = np.flatnonzero(injection_mask)
    if sources.size == 0:
        return np.full(nodes, nodes + 1, dtype=np.int64)
    distance = graph_hop_distances(edges, nodes, sources)
    distance[distance < 0] = nodes + 1
    return distance


def _target_records(
    *,
    data: dict[str, Any],
    frame: int,
    explicit_nodes: Sequence[int] | None,
    targets_per_region: int,
) -> list[dict[str, Any]]:
    state = data["reference_current"][frame]
    nodes = state.shape[0]
    boundary_distance = _boundary_distance(data["edges"], data["injection_mask"])
    if explicit_nodes:
        score = node_pressure_jump_scores(state, data["edges"])
        records = []
        for node in dict.fromkeys(explicit_nodes):
            if node < 0 or node >= nodes:
                raise ValueError(
                    f"target node {node} lies outside a {nodes}-node graph"
                )
            records.append(
                {
                    "target_node": int(node),
                    "region": "explicit",
                    "pressure_jump_score": float(score[node]),
                    "boundary_graph_distance": int(boundary_distance[node]),
                }
            )
        return records
    return select_reach_targets(
        pos=data["pos"],
        edges=data["edges"],
        primitive=state,
        normal_mask=data["node_type"] == 0,
        boundary_distance=boundary_distance,
        dependency_radius=cpg_dependency_trace(
            MESSAGE_PASSING_LAYERS
        ).final_output_radius,
        count_per_region=targets_per_region,
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty reach table")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _save_pair(
    *,
    pair_dir: Path,
    artifact_stem: str,
    frame: int,
    data: dict[str, Any],
    comparison_arrays: dict[str, np.ndarray],
    predecessor_time: np.ndarray,
    next_hop: np.ndarray,
    pair: dict[str, Any],
) -> dict[str, Any]:
    pair_dir.mkdir(exist_ok=True)
    name = f"{artifact_stem}_frame{frame}_target{pair['target_node']}.npz"
    path = pair_dir / name
    np.savez_compressed(
        path,
        state_a=pair["state_a"].astype(np.float32),
        state_b=pair["state_b"].astype(np.float32),
        pos=data["pos"].astype(np.float32),
        edges=data["edges"].astype(np.int64),
        node_type=data["node_type"].astype(np.int64),
        Mach=data["Mach"].astype(np.float32),
        dependency_mask=comparison_arrays["dependency_mask"],
        perturbation_mask=pair["perturbation_mask"],
        predecessor_time=predecessor_time,
        next_hop=next_hop,
    )
    metadata = {
        key: value
        for key, value in pair.items()
        if key not in {"state_a", "state_b", "perturbation_mask"}
    }
    metadata.update(
        {
            "file": str(Path("causal_pairs") / name),
            "sha256": sha256_file(path),
            "frame": int(frame),
        }
    )
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.targets_per_region < 1:
        raise ValueError("targets-per-region must be positive")
    if args.max_causal_pairs < 0:
        raise ValueError("max-causal-pairs must be nonnegative")
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    trace = cpg_dependency_trace(MESSAGE_PASSING_LAYERS)
    paths = artifact_paths(args.artifact)
    rows: list[dict[str, Any]] = []
    artifact_records: list[dict[str, Any]] = []
    pair_records: list[dict[str, Any]] = []

    for artifact_index, path in enumerate(paths):
        data = _read_artifact(path)
        frames = selected_frames(data["targets"].shape[0], args.frames)
        artifact_record = {
            "artifact": str(path),
            "artifact_sha256": sha256_file(path),
            "trajectory_key": str(data["attributes"].get("trajectory_key", "")),
            "frames": frames,
            "macro_dt": data["macro_dt"],
            "mapping": data["mapping"],
        }
        artifact_records.append(artifact_record)

        for frame in frames:
            samples = np.stack(
                (data["reference_current"][frame], data["targets"][frame]), axis=0
            )
            travel = euler_characteristic_travel_times(
                data["pos"], data["edges"], samples, gamma=args.gamma
            )
            directed_incoming = build_incoming_travel_adjacency(
                travel["directed_edges"],
                travel["plus_travel_time"],
                samples.shape[1],
            )
            symmetric_incoming = build_incoming_travel_adjacency(
                travel["directed_edges"],
                travel["symmetric_travel_time"],
                samples.shape[1],
            )
            target_records = _target_records(
                data=data,
                frame=frame,
                explicit_nodes=args.target_node,
                targets_per_region=args.targets_per_region,
            )
            forbidden_fraction = float(
                np.mean(~np.isfinite(travel["plus_travel_time"]))
            )
            for target_record in target_records:
                target_node = target_record["target_node"]
                directed_time, next_hop = minimum_predecessor_times(
                    directed_incoming, target_node, cutoff=data["macro_dt"]
                )
                symmetric_time, _ = minimum_predecessor_times(
                    symmetric_incoming, target_node, cutoff=data["macro_dt"]
                )
                comparison, comparison_arrays = compare_characteristic_and_model_reach(
                    edges=data["edges"],
                    num_nodes=samples.shape[1],
                    target_node=target_node,
                    directed_predecessor_time=directed_time,
                    symmetric_predecessor_time=symmetric_time,
                    macro_dt=data["macro_dt"],
                    injected_mask=data["injection_mask"],
                    message_passing_layers=MESSAGE_PASSING_LAYERS,
                )
                row = {
                    "artifact_index": artifact_index,
                    "artifact_name": path.name,
                    "trajectory_key": artifact_record["trajectory_key"],
                    "frame": frame,
                    "region": target_record["region"],
                    "pressure_jump_score": target_record["pressure_jump_score"],
                    "macro_dt": data["macro_dt"],
                    "directed_arc_forbidden_fraction": forbidden_fraction,
                    **comparison,
                }
                rows.append(row)

                if (
                    comparison["directed_uncovered_count"] > 0
                    and len(pair_records) < args.max_causal_pairs
                ):
                    pair = construct_pointwise_admissible_causal_pair(
                        base_primitive=samples[0],
                        pos=data["pos"],
                        edges=data["edges"],
                        target_node=target_node,
                        dependency_mask=comparison_arrays["dependency_mask"],
                        predecessor_time=directed_time,
                        next_hop=next_hop,
                        macro_dt=data["macro_dt"],
                        normal_mask=data["node_type"] == 0,
                        amplitude=args.causal_pair_amplitude,
                        patch_hops=args.causal_pair_patch_hops,
                        gamma=args.gamma,
                    )
                    if pair is not None:
                        pair_records.append(
                            {
                                "artifact_index": artifact_index,
                                "artifact_name": path.name,
                                **_save_pair(
                                    pair_dir=args.output_dir / "causal_pairs",
                                    artifact_stem=path.stem,
                                    frame=frame,
                                    data=data,
                                    comparison_arrays=comparison_arrays,
                                    predecessor_time=directed_time,
                                    next_hop=next_hop,
                                    pair=pair,
                                ),
                            }
                        )

    _write_csv(args.output_dir / "reach_rows.csv", rows)
    uncovered_rows = sum(row["directed_uncovered_count"] > 0 for row in rows)
    summary = {
        "schema": REACH_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_scope": (
            "pinned-release graph dependency and endpoint-sampled characteristic "
            "reach; not a paper reproduction or autonomous-solver validation"
        ),
        "source_sha256": {
            "script": sha256_file(Path(__file__)),
            "utility": sha256_file(
                ROOT / "utility" / "time_dependent_no" / "cpg_reach.py"
            ),
        },
        "dependency_trace": trace.to_dict(),
        "characteristic_contract": {
            "directed_direction": "source node to target node",
            "families": "v_dot_n-c, v_dot_n, v_dot_n+c",
            "causal_predecessor_speed": "max positive v_dot_n+c",
            "nonpositive_traversal": "disallowed",
            "edge_speed_sampling": (
                "maximum over both edge endpoints at reference current and next "
                "reference saved states"
            ),
            "directed_travel_time": "edge_length / positive sampled speed maximum",
            "symmetric_envelope": "edge_length / max(abs(v_dot_n)+c)",
            "symmetric_role": "comparison envelope only, not directed causality",
            "integration_caveat": (
                "saved endpoints do not bound unresolved within-interval DG substeps; "
                "these are endpoint-sampled optimistic reach diagnostics"
            ),
            "dg_substeps": "not treated as graph distance",
        },
        "artifacts": artifact_records,
        "row_count": len(rows),
        "rows_with_directed_cone_outside_support": uncovered_rows,
        "causal_pairs": pair_records,
        "causal_test_status": (
            "input_candidates_only_reference_target_labels_missing"
            if pair_records
            else (
                "sampled_directed_cones_within_model_support"
                if uncovered_rows == 0
                else "cone_exceeds_support_but_no_normal_pair_was_constructed"
            )
        ),
        "required_before_causal_claim": [
            "map graph nodes to reference-solver degrees of freedom",
            "bound or sample characteristic speeds over accepted DG substeps",
            "run both admissible states with the same legal physical boundaries",
            "measure the paired reference target difference",
        ],
        "conservation_claim_status": (
            "not_tested_control_volumes_faces_measures_and_orientations_unverified"
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _json_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == "__main__":
    raise SystemExit(main())
