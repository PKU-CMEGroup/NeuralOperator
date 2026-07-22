"""Build a strict provenance manifest for CPGNet release rollout artifacts.

This command never infers trajectory identity from ``N.h5 -> N-1`` alone.  It
hashes the stored ground-truth targets and matches them to the exact float32
target sequence implied by each HDF5 trajectory.  It also records the dataset,
checkpoint, reference-source, evaluator, mask, and metric contracts needed to
distinguish a release replay from a paper-table reproduction.

Example
-------

.. code-block:: powershell

   python scripts/time_dependent_no/audit_cpg_release_provenance.py `
     --train-file <train.h5> --test-file <test.h5> `
     --result-dir <run/result> --run-manifest <run/run_manifest.json> `
     --checkpoint <simulator.pth> `
     --reference-repo <cpggnspdes> --output-dir <ignored-artifact-dir>
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utility.time_dependent_no.cpg_release import (  # noqa: E402
    CPG_EVALUATOR_SOURCE_FILES,
    CPG_REFERENCE_COMMIT,
    CPGTrajectoryFingerprint,
    cpg_evaluator_source_manifest,
    cpg_graph_frame_metadata,
    cpg_mach_range,
    cpg_trajectory_dimensions,
    fingerprint_cpg_trajectory,
    graph_distance_from_sources,
    hash_rollout_targets,
    release_rollout_metrics,
    rollout_rmse_by_graph_distance,
    sha256_file,
    validate_cpg_reference_source,
)
from utility.time_dependent_no.euler2d import (  # noqa: E402
    EulerNodeType,
    PRIMITIVE_NAMES,
)


MANIFEST_SCHEMA = "cpg_release_provenance_v1"
EVALUATOR_SCHEMA = "cpg_frozen_release_evaluation_v1"
RELEASE_BOUNDARY_MODE = "oracle_next_reference"
REQUIRED_INSTRUMENTATION = (
    "raw_predicteds",
    "reference_current",
    "inputs_before_boundary",
    "model_inputs",
    "model_mach",
    "injection_mask",
    "model_pos",
    "directed_edges",
    "edge_attr_before_model",
    "pos",
    "edges",
    "node_type",
    "Mach",
    "boundary_graph_distance",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-results",
        type=int,
        default=20,
        help="Required number of result files; set explicitly for a sanity subset.",
    )
    parser.add_argument("--expected-split", default="test")
    parser.add_argument("--expected-dt", type=float, default=0.025)
    parser.add_argument("--expected-seed", type=int, default=0)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.expected_results < 1:
        raise ValueError("--expected-results must be positive")
    if args.expected_dt <= 0.0:
        raise ValueError("--expected-dt must be positive")
    return args


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_manifest() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "h5py": _package_version("h5py"),
        "torch": _package_version("torch"),
        "torch_geometric": _package_version("torch-geometric"),
    }


def dataset_inventory(
    path: Path,
    *,
    fingerprint_targets: bool,
) -> tuple[dict[str, Any], list[CPGTrajectoryFingerprint]]:
    try:
        import h5py  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError("h5py is required for the CPG provenance audit") from exc

    fingerprints: list[CPGTrajectoryFingerprint] = []
    trajectories: list[dict[str, Any]] = []
    global_mach_min = np.inf
    global_mach_max = -np.inf
    with h5py.File(path, "r") as handle:
        raw_keys = [str(key) for key in handle.keys()]
        for key in raw_keys:
            group = handle[key]
            time_steps, num_nodes, num_edges = cpg_trajectory_dimensions(group)
            mach_min, mach_max = cpg_mach_range(group)
            global_mach_min = min(global_mach_min, mach_min)
            global_mach_max = max(global_mach_max, mach_max)
            item: dict[str, Any] = {
                "trajectory_key": key,
                "num_time_steps": time_steps,
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "mach_min": mach_min,
                "mach_max": mach_max,
            }
            if fingerprint_targets:
                fingerprint = fingerprint_cpg_trajectory(
                    group,
                    trajectory_key=key,
                )
                fingerprints.append(fingerprint)
                item.update(
                    {
                        "target_steps": fingerprint.target_steps,
                        "target_sha256": fingerprint.target_sha256,
                        "graph_frame_sha256": fingerprint.graph_frame_sha256,
                    }
                )
            trajectories.append(item)

    if not trajectories:
        raise ValueError(f"dataset contains no trajectories: {path}")
    return (
        {
            "file_name": path.name,
            "file_size_bytes": path.stat().st_size,
            "file_sha256": sha256_file(path),
            "hdf5_iteration_keys": [item["trajectory_key"] for item in trajectories],
            "num_trajectories": len(trajectories),
            "mach_min": float(global_mach_min),
            "mach_max": float(global_mach_max),
            "trajectories": trajectories,
        },
        fingerprints,
    )


def _numeric_result_files(path: Path) -> list[Path]:
    files = list(path.glob("*.h5"))
    invalid = [file.name for file in files if not file.stem.isdigit()]
    if invalid:
        raise ValueError(f"result directory has nonnumeric HDF5 names: {invalid}")
    return sorted(files, key=lambda file: int(file.stem))


def _json_attribute(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def audit_results(
    result_dir: Path,
    test_file: Path,
    fingerprints: list[CPGTrajectoryFingerprint],
    checkpoint_sha256: str,
    *,
    expected_results: int,
    expected_attributes: Mapping[str, Any] | None = None,
    required_datasets: Sequence[str] = (),
) -> tuple[list[dict[str, Any]], list[str]]:
    try:
        import h5py  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError("h5py is required for the CPG provenance audit") from exc

    violations: list[str] = []
    expected_attributes = {} if expected_attributes is None else expected_attributes
    result_files = _numeric_result_files(result_dir)
    if len(result_files) != expected_results:
        violations.append(
            f"expected {expected_results} result files, found {len(result_files)}"
        )

    by_target_hash: dict[str, list[CPGTrajectoryFingerprint]] = {}
    for fingerprint in fingerprints:
        by_target_hash.setdefault(fingerprint.target_sha256, []).append(fingerprint)

    rows: list[dict[str, Any]] = []
    matched_keys: list[str] = []
    with h5py.File(test_file, "r") as dataset:
        for result_file in result_files:
            with h5py.File(result_file, "r") as result:
                missing = [
                    key for key in ("predicteds", "targets") if key not in result
                ]
                if missing:
                    violations.append(f"{result_file.name} is missing {missing}")
                    continue
                missing_instrumentation = [
                    key for key in required_datasets if key not in result
                ]
                if missing_instrumentation:
                    violations.append(
                        f"{result_file.name} is missing instrumentation "
                        f"{missing_instrumentation}"
                    )
                predictions = np.asarray(result["predicteds"])
                targets = np.asarray(result["targets"])
                if predictions.shape != targets.shape:
                    violations.append(
                        f"{result_file.name} prediction/target shape mismatch"
                    )
                    continue

                target_sha256 = hash_rollout_targets(targets)
                candidates = by_target_hash.get(target_sha256, [])
                if len(candidates) != 1:
                    violations.append(
                        f"{result_file.name} target hash matched {len(candidates)} trajectories"
                    )
                    continue
                fingerprint = candidates[0]
                trajectory_key = fingerprint.trajectory_key
                matched_keys.append(trajectory_key)
                attrs = {
                    str(key): _json_attribute(value)
                    for key, value in result.attrs.items()
                }
                for name, expected in expected_attributes.items():
                    if name not in attrs:
                        violations.append(f"{result_file.name} has no {name} attribute")
                    elif attrs[name] != expected:
                        violations.append(
                            f"{result_file.name} {name}={attrs[name]!r}, "
                            f"expected {expected!r}"
                        )
                reference_verification = attrs.get("reference_verification")
                if reference_verification not in {
                    "pinned_runtime_file_hashes",
                    "git_commit_and_pinned_runtime_file_hashes",
                }:
                    violations.append(
                        f"{result_file.name} has invalid reference_verification "
                        f"{reference_verification!r}"
                    )

                declared_key = attrs.get("trajectory_key")
                if declared_key is not None and str(declared_key) != trajectory_key:
                    violations.append(
                        f"{result_file.name} declares trajectory {declared_key!r} "
                        f"but targets match {trajectory_key!r}"
                    )
                declared_checkpoint = attrs.get("checkpoint_sha256")
                checkpoint_bound = declared_checkpoint == checkpoint_sha256
                if declared_checkpoint is None:
                    violations.append(
                        f"{result_file.name} has no checkpoint_sha256 attribute"
                    )
                elif not checkpoint_bound:
                    violations.append(
                        f"{result_file.name} checkpoint hash does not match the audited checkpoint"
                    )

                pos, edges, node_type, _ = cpg_graph_frame_metadata(
                    dataset[trajectory_key], frame=0
                )
                if targets.shape[0] != fingerprint.target_steps:
                    violations.append(
                        f"{result_file.name} contains {targets.shape[0]} steps; "
                        f"trajectory {trajectory_key!r} requires {fingerprint.target_steps}"
                    )
                if targets.shape[1] != fingerprint.num_nodes:
                    violations.append(
                        f"{result_file.name} node count does not match {trajectory_key!r}"
                    )
                    continue

                metrics = release_rollout_metrics(predictions, targets, node_type)
                boundary_mask = node_type != int(EulerNodeType.NORMAL)
                distance = graph_distance_from_sources(edges, boundary_mask)
                rows.append(
                    {
                        "result_file": result_file.name,
                        "result_sha256": sha256_file(result_file),
                        "trajectory_key": trajectory_key,
                        "target_sha256": target_sha256,
                        "checkpoint_bound": checkpoint_bound,
                        "attributes": attrs,
                        "graph_frame_sha256": fingerprint.graph_frame_sha256,
                        "metrics": metrics,
                        "boundary_graph_distance": {
                            "max_finite_distance": int(np.max(distance)),
                            "unreachable_nodes": int(np.count_nonzero(distance < 0)),
                            "rmse": rollout_rmse_by_graph_distance(
                                predictions, targets, distance
                            ),
                        },
                    }
                )

    duplicates = sorted(key for key in set(matched_keys) if matched_keys.count(key) > 1)
    if duplicates:
        violations.append(f"multiple result files map to trajectories {duplicates}")
    return rows, violations


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    output: dict[str, Any] = {"num_trajectories": len(rows)}
    for mask in ("all", "normal", "boundary", "wall", "outflow", "inflow"):
        values = [row["metrics"][mask]["rollout_rmse"] for row in rows]
        present = [value for value in values if value is not None]
        output[mask] = {
            "mean_per_trajectory_rollout_rmse": (
                np.mean(np.asarray(present, dtype=np.float64), axis=0).tolist()
                if present
                else None
            )
        }
    return output


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "result_file",
        "trajectory_key",
        "target_sha256",
        "checkpoint_bound",
        "num_steps",
        "num_nodes",
        "max_boundary_graph_distance",
    ]
    for mask in ("all", "normal", "boundary"):
        fieldnames.extend(f"{mask}_{name}_rollout_rmse" for name in PRIMITIVE_NAMES)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in rows:
            metrics = item["metrics"]
            row: dict[str, Any] = {
                "result_file": item["result_file"],
                "trajectory_key": item["trajectory_key"],
                "target_sha256": item["target_sha256"],
                "checkpoint_bound": item["checkpoint_bound"],
                "num_steps": metrics["num_steps"],
                "num_nodes": metrics["num_nodes"],
                "max_boundary_graph_distance": item["boundary_graph_distance"][
                    "max_finite_distance"
                ],
            }
            for mask in ("all", "normal", "boundary"):
                values = metrics[mask]["rollout_rmse"]
                for index, name in enumerate(PRIMITIVE_NAMES):
                    row[f"{mask}_{name}_rollout_rmse"] = (
                        None if values is None else values[index]
                    )
            writer.writerow(row)


def audit_run_manifest(
    path: Path,
    *,
    reference: Mapping[str, Any],
    evaluator_sources: Mapping[str, str],
    checkpoint_sha256: str,
    dataset: Mapping[str, Any],
    expected_attributes: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    violations: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return (
            {
                "file_name": path.name,
                "file_sha256": sha256_file(path),
                "cross_checked": False,
            },
            [f"run manifest could not be parsed: {exc}"],
        )

    def expect(label: str, actual: Any, expected: Any) -> None:
        if actual != expected:
            violations.append(f"run manifest {label}={actual!r}, expected {expected!r}")

    expect("schema", payload.get("schema"), EVALUATOR_SCHEMA)
    expect(
        "boundary_mode",
        payload.get("boundary_mode"),
        expected_attributes["boundary_mode"],
    )
    expect("seed", payload.get("seed"), expected_attributes["seed"])

    run_reference = payload.get("reference", {})
    expect(
        "reference.expected_commit",
        run_reference.get("expected_commit"),
        CPG_REFERENCE_COMMIT,
    )
    expect(
        "reference.runtime_file_sha256",
        run_reference.get("runtime_file_sha256"),
        reference["runtime_file_sha256"],
    )

    run_evaluator = payload.get("evaluator", {})
    expect(
        "evaluator.source_sha256",
        run_evaluator.get("source_sha256"),
        dict(evaluator_sources),
    )

    run_checkpoint = payload.get("checkpoint", {})
    expect(
        "checkpoint.sha256",
        run_checkpoint.get("sha256"),
        checkpoint_sha256,
    )
    expect(
        "checkpoint.binding",
        run_checkpoint.get("binding"),
        "explicit Simulator.load_checkpoint argument",
    )

    run_dataset = payload.get("dataset", {})
    expect("dataset.sha256", run_dataset.get("sha256"), dataset["file_sha256"])
    expect("dataset.split", run_dataset.get("split"), expected_attributes["split"])
    expect(
        "dataset.hdf5_iteration_keys",
        run_dataset.get("hdf5_iteration_keys"),
        dataset["hdf5_iteration_keys"],
    )
    expected_keys = [str(result["trajectory_key"]) for result in results]
    expect("dataset.selected_keys", run_dataset.get("selected_keys"), expected_keys)

    model = payload.get("model_configuration", {})
    expect("model.dt", model.get("dt"), expected_attributes["dt"])
    expect("model.message_passing_num", model.get("message_passing_num"), 12)
    expect("model.node_input_size", model.get("node_input_size"), 6)
    expect("model.edge_input_size", model.get("edge_input_size"), 5)

    records = payload.get("trajectories", [])
    if not isinstance(records, list):
        violations.append("run manifest trajectories is not a list")
        records = []
    declared_by_file: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            violations.append("run manifest has a non-object trajectory record")
            continue
        result_file = str(record.get("result_file"))
        if result_file in declared_by_file:
            violations.append(
                f"run manifest repeats trajectory result file {result_file!r}"
            )
        declared_by_file[result_file] = record

    expected_files = [str(result["result_file"]) for result in results]
    expect("trajectory result files", sorted(declared_by_file), sorted(expected_files))
    for result in results:
        result_file = str(result["result_file"])
        record = declared_by_file.get(result_file)
        if record is None:
            continue
        expect(
            f"{result_file}.result_sha256",
            record.get("result_sha256"),
            result["result_sha256"],
        )
        expect(
            f"{result_file}.trajectory_key",
            record.get("trajectory_key"),
            result["trajectory_key"],
        )

    aggregate = payload.get("aggregate", {})
    audited_aggregate = aggregate_metrics(list(results))
    if audited_aggregate is not None:
        for run_key, audit_key in (
            ("post_all_rollout_rmse", "all"),
            ("post_normal_rollout_rmse", "normal"),
        ):
            actual = np.asarray(aggregate.get(run_key), dtype=np.float64)
            expected = np.asarray(
                audited_aggregate[audit_key]["mean_per_trajectory_rollout_rmse"],
                dtype=np.float64,
            )
            if actual.shape != expected.shape or not np.allclose(
                actual,
                expected,
                rtol=1e-12,
                atol=1e-12,
            ):
                violations.append(
                    f"run manifest aggregate {run_key} does not match audited results"
                )

    return (
        {
            "file_name": path.name,
            "file_sha256": sha256_file(path),
            "schema": payload.get("schema"),
            "trajectory_records": len(records),
            "cross_checked": not violations,
        },
        violations,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    for path in (
        args.train_file,
        args.test_file,
        args.checkpoint,
        args.run_manifest,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.result_dir.is_dir():
        raise FileNotFoundError(args.result_dir)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    violations: list[str] = []

    reference = validate_cpg_reference_source(args.reference_repo)
    evaluator_sources = cpg_evaluator_source_manifest(REPO_ROOT)
    train, _ = dataset_inventory(args.train_file, fingerprint_targets=False)
    test, fingerprints = dataset_inventory(args.test_file, fingerprint_targets=True)
    checkpoint_sha256 = sha256_file(args.checkpoint)
    expected_attributes = {
        "schema": EVALUATOR_SCHEMA,
        "split": args.expected_split,
        "boundary_mode": RELEASE_BOUNDARY_MODE,
        "checkpoint_sha256": checkpoint_sha256,
        "dataset_sha256": test["file_sha256"],
        "evaluator_sha256": evaluator_sources[CPG_EVALUATOR_SOURCE_FILES[0]],
        "reference_commit": CPG_REFERENCE_COMMIT,
        "dt": args.expected_dt,
        "seed": args.expected_seed,
        "primitive_order": json.dumps(list(PRIMITIVE_NAMES)),
    }
    results, result_violations = audit_results(
        args.result_dir,
        args.test_file,
        fingerprints,
        checkpoint_sha256,
        expected_results=args.expected_results,
        expected_attributes=expected_attributes,
        required_datasets=REQUIRED_INSTRUMENTATION,
    )
    violations.extend(result_violations)
    run_manifest, run_manifest_violations = audit_run_manifest(
        args.run_manifest,
        reference=reference,
        evaluator_sources=evaluator_sources,
        checkpoint_sha256=checkpoint_sha256,
        dataset=test,
        expected_attributes=expected_attributes,
        results=results,
    )
    violations.extend(run_manifest_violations)

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "valid": not violations,
        "violations": violations,
        "claim_scope": (
            "pinned-release evaluator artifact audit; paper dataset/checkpoint "
            "identity requires independent author evidence"
        ),
        "reference": reference,
        "runtime": runtime_manifest(),
        "tooling": {
            "audit_script_sha256": sha256_file(Path(__file__)),
            "evaluator_source_sha256": evaluator_sources,
        },
        "run_manifest": run_manifest,
        "dataset": {"train": train, "test": test},
        "checkpoint": {
            "file_name": args.checkpoint.name,
            "file_size_bytes": args.checkpoint.stat().st_size,
            "sha256": checkpoint_sha256,
        },
        "evaluator_contract": {
            "target_source": "dataset ground truth frames 1..T-1",
            "primitive_order": list(PRIMITIVE_NAMES),
            "release_boundary_mode": "next-reference injection and output clamp",
            "release_metric": (
                "per-trajectory sqrt(mean over rollout time and all nodes), "
                "then uniform mean across trajectories"
            ),
            "required_result_attributes": expected_attributes,
            "required_instrumentation": list(REQUIRED_INSTRUMENTATION),
            "diagnostic_masks": [
                "all",
                "normal",
                "boundary",
                "wall",
                "outflow",
                "inflow",
                "graph_distance_from_boundary",
            ],
        },
        "results": results,
        "aggregate_metrics": aggregate_metrics(results),
    }
    manifest_path = output_dir / "provenance_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_metrics_csv(output_dir / "trajectory_metrics.csv", results)

    print(
        json.dumps(
            {
                "valid": manifest["valid"],
                "violations": len(violations),
                "matched_results": len(results),
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )
    return 0 if manifest["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
