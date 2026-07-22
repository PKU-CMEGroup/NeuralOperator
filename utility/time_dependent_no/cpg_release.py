"""Provenance and metric helpers for frozen CPGNet release evaluation.

The public CPGNet evaluator does not persist trajectory identifiers, checkpoint
digests, or enough geometry to verify a saved rollout after the fact.  This
module provides a small NumPy-only contract for doing those checks without
vendoring or importing the public model implementation.

Target hashes use the exact float32 primitive ordering consumed by the release:
``[rho, v1, v2, pres]`` at frames ``1..T-1``.  A rollout therefore matches a
trajectory only when its stored ground-truth targets match that logical array,
not merely when its filename has the expected numeric index.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np

from utility.time_dependent_no.euler2d import EulerNodeType, PRIMITIVE_KEYS


ArrayLike = Any
GroupLike = Mapping[str, Any]
_HASH_CHUNK_BYTES = 8 * 1024 * 1024
_TARGET_HASH_SCHEMA = "cpg_float32_rollout_targets_v1"
_GEOMETRY_HASH_SCHEMA = "cpg_release_graph_frame_v1"

CPG_REFERENCE_COMMIT = "b127e5e9489dc8e218f8f9b94147153e6998ea89"
CPG_MODEL_DT = 0.025
CPG_LEGAL_BOUNDARY_MAX_SOURCE_HOPS = 3
CPG_TERMINATION_ACCOUNTING_SCHEMA = "cpg_admissible_prefix_v1"
CPG_MESH_AUDIT_SCHEMA = "cpg_bump_mesh_provenance_audit_v2"
CPG_LEGACY_MESH_AUDIT_SCHEMA = "cpg_bump_mesh_provenance_audit_v1"
CPG_REFERENCE_EVALUATION_RUNTIME_PIN_SCHEMA = (
    "cpggnspdes_b127e5_evaluation_runtime_v1"
)
CPG_REFERENCE_RUNTIME_PIN_SCHEMA = "cpggnspdes_b127e5_runtime_v2"
CPG_LEGACY_TRAINING_PIN_OMISSIONS = frozenset(
    {"utils/lossCompute.py", "utils/noise.py"}
)
CPG_REFERENCE_RUNTIME_SHA256 = {
    "dataset/__init__.py": "9b7786ae561bdf0e4feaa67924ca4481aef2a375f67b84293efa83750f9d2bed",
    "dataset/fpc.py": "a64a4362acf947ffa78a0805790d2437c6095ce9ac09cec319343d5b9be83f70",
    "dataset/fpcMulti.py": "af7426b3739fe66b17b585e6d4b214cd015400a744b013e2699a2175670f8c99",
    "modelEdgeUpd/simulator.py": "cc3bfd025d7756fac0c66e9e10d418c96751a5e0d92121cfedad9a40a6cabb5e",
    "modelEdgeUpd/modelEU.py": "5a5103b61f0a1aa1e0e0d89004f268f3d712d4093e2752c9849f87f3396ba454",
    "modelEdgeUpd/EdgeEncoder.py": "58bdac82056e2bb503449de08c7e94c727678e2c6f44e8121d85564cedbdd610",
    "modelEdgeUpd/convFlow.py": "24b882d8168b0fa53fd37b3bcf2d64c87d592bdff8fa1aadd2d77d56af566272",
    "modelEdgeUpd/convReconstruct.py": "1528e16a0ffc01294b8f6e051e6f0b710ecc5b509f8b3d47047ef06c8174035a",
    "modelEdgeUpd/conserveUpd.py": "ead8aadfa261bf135301c35cd38ef2397b5c03a4f72bc086f932c50eec266571",
    "modelEdgeUpd/dissFactor.py": "d6a88ee21b50e089cd9785708576802d165404f6e2f81a85c3ff57397610bfd3",
    "modelEdgeUpd/limiting.py": "2a999b7e1fc490e8425431574f7a60d8b770e712dc596dcfb5a7dffc2c9ff912",
    "utils/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "utils/normalization.py": "6726d74f5ffdbcfb2c5673f0b0202871cea4734983cd195df94e4518fd448fde",
    "utils/GeometricEstimates.py": "819daf84ef5116a601613261925efa588b0f852f6be310dd55c096bab8ce555b",
    "utils/lossCompute.py": "935a1025787d723d9980375275b3c4167ea931ad95b6c4c70b7694fe702ecc42",
    "utils/noise.py": "67269a162bf812e3d43da988bc9ce41feba3b083ef204863144b22f59f6ffcb2",
    "utils/to_undirected.py": "7a5724998484492c9898d926cfe1099cdf8474bc53b454780500f2b0048ba9d7",
    "utils/utils.py": "6a05f67a6632cd6fba5cbec58aefe3168d8277feb2f6e167ef5a23dd7970ba9e",
    "utils/ghost_marker.py": "c649401f1ed7bc53a1076b1b725661fd2e1867cd313d3967ac61c7f4c6abe6e8",
    "utilsEulerEqs/eulerEqs.py": "b5df645d0268ea78ec605ad26691959a37037d9839816dc71eb5a0ca43083ed8",
    "utilsEulerEqs/fluxLLFCons.py": "f70515291877af578b409aa539e7c279c5e5578503f744c6d70ea02ac16a22ad",
    "utilsEulerEqs/fluxLocalLaxFriedrichs.py": "a885fb308415b6ea0c28a08c27ac91b80b9a92a228faa83fd57a0390415b6428",
    "utilsPreservation/firstOrderFVloss.py": "b43ec5aaafe5de00bbd6ef9c8bd69388f6003b004ca22f35fae21eca82e3cc3a",
    "rollout.py": "31f79f370bd6e8ff6483b51db057e51b817063526b8bddaee905f0525fc27cfa",
    "trainMulti.py": "26b84f12af37f97ae5f60799a9e354a19a6d958f0d912ad024c48e136bde0d8f",
    "trainMultiScheduled.py": "dd6332c1870c0616d8cb4e75c767b56312be9c3c91ab9865e5227ac44687294a",
}
CPG_REFERENCE_EVALUATION_RUNTIME_SHA256 = {
    path: digest
    for path, digest in CPG_REFERENCE_RUNTIME_SHA256.items()
    if path not in CPG_LEGACY_TRAINING_PIN_OMISSIONS
}
CPG_LOCAL_TRAINING_SOURCE_SCHEMA = "cpg_local_training_source_v1"
CPG_LOCAL_TRAINING_SOURCE_FILES = {
    "trainer": "scripts/time_dependent_no/train_cpg_legal_boundary.py",
    "boundary_utility": "utility/time_dependent_no/cpg_mesh_contract.py",
    "provenance_utility": "utility/time_dependent_no/cpg_release.py",
    "euler2d_utility": "utility/time_dependent_no/euler2d.py",
}
CPG_LEGACY_LOCAL_TRAINING_SOURCE_COMMIT = (
    "4654225e93a1566e7fb5fbc46ba81825a040b30b"
)
CPG_ARCHIVAL_LOCAL_TRAINING_SOURCE_SHA256 = {
    "trainer": "510be9f9705328ea83c580d641f6b452aa94318713282a118128d588cf07b682",
    "boundary_utility": "9e0069a0798f74738af242b7bacbf0324fb9d6ad18e9a12de77e4e83dd577abf",
    "provenance_utility": "5fd0132678d44152039107feca30320b927e262997d7227b0767ffdf8fc5b372",
    "euler2d_utility": "75ad2ccc09e51ed914ed5fe81f19fefe8019ab93ca307a7e33a0df77c39652d2",
}
CPG_LEGACY_LOCAL_TRAINING_SOURCE_SHA256 = {
    name: CPG_ARCHIVAL_LOCAL_TRAINING_SOURCE_SHA256[name]
    for name in ("trainer", "boundary_utility")
}
CPG_EVALUATOR_SOURCE_FILES = (
    "scripts/time_dependent_no/evaluate_cpg_release.py",
    "utility/time_dependent_no/cpg_release.py",
    "utility/time_dependent_no/euler2d.py",
)


@dataclass(frozen=True)
class CPGTrajectoryFingerprint:
    """Canonical identity fields for one CPG HDF5 trajectory."""

    trajectory_key: str
    num_time_steps: int
    num_nodes: int
    num_edges: int
    target_steps: int
    target_sha256: str
    graph_frame_sha256: str
    mach_min: float
    mach_max: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: str | Path, *, chunk_bytes: int = _HASH_CHUNK_BYTES) -> str:
    """Return a streaming SHA-256 digest for ``path``."""

    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)

    digest = sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def validate_cpg_model_dt(value: float) -> float:
    """Require the macro timestep hardcoded by the released CPGNet model."""

    try:
        dt = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"released CPGNet hardcodes dt={CPG_MODEL_DT}; got {value!r}"
        ) from exc
    if not np.isfinite(dt) or dt != CPG_MODEL_DT:
        raise ValueError(
            f"released CPGNet hardcodes dt={CPG_MODEL_DT}; got {value!r}"
        )
    return dt


def cpg_evaluator_source_manifest(repo_root: str | Path) -> dict[str, str]:
    """Hash the complete local source surface used by the release evaluator."""

    root = Path(repo_root)
    return {
        relative_path: sha256_file(root / Path(relative_path))
        for relative_path in CPG_EVALUATOR_SOURCE_FILES
    }


def cpg_local_training_source_manifest(repo_root: str | Path) -> dict[str, Any]:
    """Bind every local training dependency to a clean Git commit."""

    root = Path(repo_root)
    commit = _git_output(root, "rev-parse", "HEAD")
    relative_paths = list(CPG_LOCAL_TRAINING_SOURCE_FILES.values())
    status = _git_output(root, "status", "--short", "--", *relative_paths)
    if status:
        raise RuntimeError(
            "local CPG training source files must be committed before training"
        )
    file_sha256 = {
        label: _git_blob_sha256(root, commit, relative_path)
        for label, relative_path in CPG_LOCAL_TRAINING_SOURCE_FILES.items()
    }
    working_tree_sha256 = {
        label: sha256_file(root / relative_path)
        for label, relative_path in CPG_LOCAL_TRAINING_SOURCE_FILES.items()
    }
    return {
        "schema": CPG_LOCAL_TRAINING_SOURCE_SCHEMA,
        "git_commit": commit,
        "tracked_files_clean": True,
        "file_hash_semantics": "git_blob_sha256",
        "file_sha256": file_sha256,
        "working_tree_sha256": working_tree_sha256,
        "relative_paths": dict(CPG_LOCAL_TRAINING_SOURCE_FILES),
    }


def validate_cpg_local_training_source_manifest(
    manifest: Mapping[str, Any],
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Verify a training source manifest against immutable local Git blobs."""

    if manifest.get("schema") != CPG_LOCAL_TRAINING_SOURCE_SCHEMA:
        raise ValueError("local training source manifest has an unsupported schema")
    commit = manifest.get("git_commit")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or commit != commit.lower()
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise ValueError("local training source manifest has an invalid Git commit")
    if manifest.get("tracked_files_clean") is not True:
        raise ValueError("local training source manifest was not clean")
    if manifest.get("file_hash_semantics") != "git_blob_sha256":
        raise ValueError("local training source manifest has ambiguous hash semantics")
    if manifest.get("relative_paths") != CPG_LOCAL_TRAINING_SOURCE_FILES:
        raise ValueError("local training source paths differ from the required closure")
    file_sha256 = manifest.get("file_sha256")
    if not isinstance(file_sha256, Mapping) or set(file_sha256) != set(
        CPG_LOCAL_TRAINING_SOURCE_FILES
    ):
        raise ValueError("local training source hash set is incomplete")
    working_tree_sha256 = manifest.get("working_tree_sha256")
    if working_tree_sha256 is not None and (
        not isinstance(working_tree_sha256, Mapping)
        or set(working_tree_sha256) != set(CPG_LOCAL_TRAINING_SOURCE_FILES)
    ):
        raise ValueError("local training working-tree hash set is incomplete")

    root = Path(repo_root)
    for label, relative_path in CPG_LOCAL_TRAINING_SOURCE_FILES.items():
        digest = file_sha256[label]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or digest != digest.lower()
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("local training source manifest has an invalid SHA256")
        if _git_blob_sha256(root, commit, relative_path) != digest:
            raise ValueError(
                f"local training source {relative_path} does not match Git commit"
            )
        if working_tree_sha256 is not None:
            working_digest = working_tree_sha256[label]
            if (
                not isinstance(working_digest, str)
                or len(working_digest) != 64
                or working_digest != working_digest.lower()
                or any(
                    character not in "0123456789abcdef"
                    for character in working_digest
                )
            ):
                raise ValueError(
                    "local training source manifest has an invalid working-tree SHA256"
                )
    normalized = {
        "schema": CPG_LOCAL_TRAINING_SOURCE_SCHEMA,
        "git_commit": commit,
        "tracked_files_clean": True,
        "file_hash_semantics": "git_blob_sha256",
        "file_sha256": dict(file_sha256),
        "relative_paths": dict(CPG_LOCAL_TRAINING_SOURCE_FILES),
        "completeness": "git_commit_and_full_local_source_closure",
    }
    if working_tree_sha256 is not None:
        normalized["working_tree_sha256"] = dict(working_tree_sha256)
    return normalized


def validate_cpg_reference_source(
    repo: str | Path,
    *,
    include_training_dependencies: bool = False,
) -> dict[str, Any]:
    """Validate the exact public CPGNet source files used by the evaluator.

    A copied runtime tree can legitimately lack Git metadata. In that case the
    returned manifest says that commit identity is unavailable and binds every
    imported/evaluation source file to its digest at the pinned public commit.
    A real Git checkout must additionally be at that commit with no tracked
    modifications.
    """

    source = Path(repo)
    if not source.is_dir():
        raise FileNotFoundError(f"reference source directory does not exist: {source}")

    expected_hashes = (
        CPG_REFERENCE_RUNTIME_SHA256
        if include_training_dependencies
        else CPG_REFERENCE_EVALUATION_RUNTIME_SHA256
    )
    pin_schema = (
        CPG_REFERENCE_RUNTIME_PIN_SCHEMA
        if include_training_dependencies
        else CPG_REFERENCE_EVALUATION_RUNTIME_PIN_SCHEMA
    )
    actual_hashes: dict[str, str] = {}
    failures: list[str] = []
    for relative_path, expected_sha256 in expected_hashes.items():
        file_path = source / Path(relative_path)
        if not file_path.is_file():
            failures.append(f"missing {relative_path}")
            continue
        actual_sha256 = sha256_file(file_path)
        actual_hashes[relative_path] = actual_sha256
        if actual_sha256 != expected_sha256:
            failures.append(
                f"{relative_path}: expected {expected_sha256}, got {actual_sha256}"
            )
    if failures:
        details = "\n".join(f"  - {failure}" for failure in failures)
        raise RuntimeError(f"reference source identity mismatch:\n{details}")

    manifest: dict[str, Any] = {
        "expected_commit": CPG_REFERENCE_COMMIT,
        "runtime_pin_schema": pin_schema,
        "runtime_scope": (
            "evaluation_and_training"
            if include_training_dependencies
            else "evaluation_only"
        ),
        "git_commit": None,
        "git_commit_verified": False,
        "tracked_clean": None,
        "untracked_file_count": None,
        "runtime_files_verified": True,
        "runtime_file_sha256": actual_hashes,
        "verification": "pinned_runtime_file_hashes",
    }
    if not (source / ".git").exists():
        return manifest

    commit = _git_output(source, "rev-parse", "HEAD")
    if commit != CPG_REFERENCE_COMMIT:
        raise RuntimeError(
            f"reference commit {commit} does not match {CPG_REFERENCE_COMMIT}"
        )
    tracked_status = _git_output(source, "status", "--short", "--untracked-files=no")
    if tracked_status:
        raise RuntimeError("reference checkout has tracked modifications")
    untracked = _git_output(source, "ls-files", "--others", "--exclude-standard")
    manifest.update(
        {
            "git_commit": commit,
            "git_commit_verified": True,
            "tracked_clean": True,
            "untracked_file_count": len(untracked.splitlines()) if untracked else 0,
            "verification": "git_commit_and_pinned_runtime_file_hashes",
        }
    )
    return manifest


def hash_rollout_targets(
    targets: ArrayLike,
    *,
    chunk_steps: int = 1,
) -> str:
    """Hash a ``(steps, nodes, 4)`` rollout target array canonically."""

    shape = _shape(targets)
    _validate_rollout_shape(shape, "targets")
    if chunk_steps <= 0:
        raise ValueError("chunk_steps must be positive")

    digest = _target_digest(shape)
    for start in range(0, shape[0], chunk_steps):
        stop = min(start + chunk_steps, shape[0])
        _update_array_bytes(digest, np.asarray(targets[start:stop]), np.dtype("<f4"))
    return digest.hexdigest()


def hash_cpg_group_targets(
    group: GroupLike,
    *,
    steps: int | None = None,
    chunk_steps: int = 1,
) -> str:
    """Hash the release targets implied by a CPG trajectory group.

    The public loader casts each primitive to float32 and uses frames ``1``
    through ``steps`` as rollout targets.  This routine reproduces that logical
    array without materializing the full trajectory in memory.
    """

    time_steps, num_nodes = _primitive_shape(group)
    target_steps = time_steps - 1 if steps is None else int(steps)
    if target_steps < 1 or target_steps > time_steps - 1:
        raise ValueError(f"steps must be in [1, {time_steps - 1}], got {target_steps}")
    if chunk_steps <= 0:
        raise ValueError("chunk_steps must be positive")

    shape = (target_steps, num_nodes, len(PRIMITIVE_KEYS))
    digest = _target_digest(shape)
    for offset in range(0, target_steps, chunk_steps):
        count = min(chunk_steps, target_steps - offset)
        start = 1 + offset
        stop = start + count
        arrays = [
            _as_time_node_scalar(np.asarray(group[key][start:stop]), key)
            for key in PRIMITIVE_KEYS
        ]
        primitive = np.concatenate(arrays, axis=-1)
        _update_array_bytes(digest, primitive, np.dtype("<f4"))
    return digest.hexdigest()


def fingerprint_cpg_trajectory(
    group: GroupLike,
    *,
    trajectory_key: str,
    target_steps: int | None = None,
) -> CPGTrajectoryFingerprint:
    """Return a canonical release-facing fingerprint for one trajectory."""

    time_steps, num_nodes = _primitive_shape(group)
    steps = time_steps - 1 if target_steps is None else int(target_steps)
    if steps < 1 or steps > time_steps - 1:
        raise ValueError(f"invalid target_steps={steps} for {time_steps} frames")

    pos, edges, node_type, mach = cpg_graph_frame_metadata(group, frame=0)
    if pos.shape[0] != num_nodes:
        raise ValueError("trajectory position and primitive node counts differ")

    mach_min, mach_max = _finite_min_max(group["Mach"])
    return CPGTrajectoryFingerprint(
        trajectory_key=str(trajectory_key),
        num_time_steps=time_steps,
        num_nodes=num_nodes,
        num_edges=int(edges.shape[0]),
        target_steps=steps,
        target_sha256=hash_cpg_group_targets(group, steps=steps),
        graph_frame_sha256=hash_cpg_graph_frame(pos, edges, node_type, mach),
        mach_min=mach_min,
        mach_max=mach_max,
    )


def cpg_trajectory_dimensions(group: GroupLike) -> tuple[int, int, int]:
    """Return (time_steps, nodes, edges) after strict shape validation."""

    time_steps, num_nodes = _primitive_shape(group)
    _, edges, _, _ = cpg_graph_frame_metadata(group, frame=0)
    return time_steps, num_nodes, int(edges.shape[0])


def cpg_mach_range(group: GroupLike) -> tuple[float, float]:
    """Return the Mach range after requiring every entry to be finite."""

    if "Mach" not in group:
        raise KeyError("trajectory is missing Mach")
    return _finite_min_max(group["Mach"])


def cpg_graph_frame_metadata(
    group: GroupLike,
    *,
    frame: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load release-facing graph metadata for one trajectory frame."""

    time_steps, num_nodes = _primitive_shape(group)
    if frame < 0 or frame >= time_steps:
        raise IndexError(f"frame must be in [0, {time_steps}), got {frame}")

    pos = np.asarray(_frame_or_static(group["pos"], frame, time_steps, "pos"))
    raw_edges = np.asarray(
        _frame_or_static(group["edges"], frame, time_steps, "edges")
    )
    if not np.issubdtype(raw_edges.dtype, np.integer) and (
        not np.all(np.isfinite(raw_edges))
        or not np.all(raw_edges == np.floor(raw_edges))
    ):
        raise ValueError("edges must contain finite integer indices")
    edges = raw_edges.astype(np.int64, copy=False)
    raw_node_type = np.asarray(
        _frame_or_static(group["node_type"], frame, time_steps, "node_type")
    ).reshape(-1)
    if not np.issubdtype(raw_node_type.dtype, np.integer) and (
        not np.all(np.isfinite(raw_node_type))
        or not np.all(raw_node_type == np.floor(raw_node_type))
    ):
        raise ValueError("node_type must contain finite integer codes")
    node_type = raw_node_type.astype(np.int64, copy=False)
    mach = np.asarray(
        _frame_or_static(group["Mach"], frame, time_steps, "Mach"),
        dtype=np.float32,
    ).reshape(-1)

    if pos.ndim != 2 or pos.shape[0] != num_nodes or pos.shape[1] < 2:
        raise ValueError(f"pos must have shape ({num_nodes}, 2+), got {pos.shape}")
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"edges must have shape (num_edges, 2), got {edges.shape}")
    if node_type.shape != (num_nodes,):
        raise ValueError(
            f"node_type must contain one value per node, got {node_type.shape}"
        )
    if mach.shape != (num_nodes,):
        raise ValueError(f"Mach must contain one value per node, got {mach.shape}")
    if not np.all(np.isfinite(pos[:, :2])):
        raise ValueError("pos contains nonfinite coordinates")
    if not np.all(np.isfinite(mach)):
        raise ValueError("Mach contains nonfinite values")
    _validate_edges(edges, num_nodes)
    return pos[:, :2].astype(np.float32), edges, node_type, mach


def validate_cpg_static_graph(
    group: GroupLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Require release graph metadata to remain identical at every frame."""

    time_steps, _ = _primitive_shape(group)
    reference = cpg_graph_frame_metadata(group, frame=0)
    for frame in range(1, time_steps):
        current = cpg_graph_frame_metadata(group, frame=frame)
        for name, expected, actual in zip(
            ("pos", "edges", "node_type", "Mach"),
            reference,
            current,
            strict=True,
        ):
            if not np.array_equal(expected, actual):
                raise ValueError(
                    f"{name} changes within one HDF trajectory at frame {frame}"
                )
    return reference


def hash_cpg_graph_frame(
    pos: ArrayLike,
    edges: ArrayLike,
    node_type: ArrayLike,
    mach: ArrayLike,
) -> str:
    """Hash graph metadata in the dtypes used by the public evaluator."""

    values = (
        ("pos", pos, np.dtype("<f4")),
        ("edges", edges, np.dtype("<i8")),
        ("node_type", node_type, np.dtype("<i8")),
        ("Mach", mach, np.dtype("<f4")),
    )
    digest = sha256()
    _update_header(digest, {"schema": _GEOMETRY_HASH_SCHEMA})
    for name, value, dtype in values:
        array = np.asarray(value)
        _update_header(
            digest,
            {"name": name, "shape": list(array.shape), "dtype": dtype.str},
        )
        _update_array_bytes(digest, array, dtype)
    return digest.hexdigest()


def release_rollout_metrics(
    predictions: ArrayLike,
    targets: ArrayLike,
    node_type: ArrayLike,
) -> dict[str, Any]:
    """Compute exact release-style and mask-stratified primitive RMSE."""

    pred = np.asarray(predictions, dtype=np.float64)
    truth = np.asarray(targets, dtype=np.float64)
    if pred.shape != truth.shape:
        raise ValueError(
            f"predictions and targets must share shape, got {pred.shape} and {truth.shape}"
        )
    _validate_rollout_shape(pred.shape, "predictions")
    if not np.all(np.isfinite(pred)) or not np.all(np.isfinite(truth)):
        raise ValueError("predictions and targets must contain only finite values")

    types = np.asarray(node_type, dtype=np.int64).reshape(-1)
    if types.shape != (pred.shape[1],):
        raise ValueError("node_type must contain one value per rollout node")

    result: dict[str, Any] = {
        "num_steps": int(pred.shape[0]),
        "num_nodes": int(pred.shape[1]),
        "all": _masked_rollout_metrics(pred, truth, np.ones_like(types, dtype=bool)),
    }
    masks = {
        "normal": types == int(EulerNodeType.NORMAL),
        "wall": types == int(EulerNodeType.WALL),
        "outflow": types == int(EulerNodeType.OUTFLOW),
        "inflow": types == int(EulerNodeType.INFLOW),
    }
    masks["boundary"] = ~masks["normal"]
    for name, mask in masks.items():
        result[name] = _masked_rollout_metrics(pred, truth, mask)
    return result


def graph_distance_from_sources(
    edges: ArrayLike,
    source_mask: ArrayLike,
) -> np.ndarray:
    """Return strict undirected shortest-hop distance from every source node."""

    sources = np.asarray(source_mask, dtype=bool).reshape(-1)
    edge_array = np.asarray(edges, dtype=np.int64)
    _validate_edges(edge_array, sources.size)
    if not np.any(sources):
        raise ValueError("source_mask must select at least one node")

    adjacency: list[list[int]] = [[] for _ in range(sources.size)]
    for left, right in edge_array:
        left_i = int(left)
        right_i = int(right)
        adjacency[left_i].append(right_i)
        adjacency[right_i].append(left_i)

    distance = np.full(sources.size, -1, dtype=np.int64)
    queue: deque[int] = deque()
    for source in np.flatnonzero(sources):
        source_i = int(source)
        distance[source_i] = 0
        queue.append(source_i)

    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            if distance[neighbor] < 0:
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)
    return distance


def rollout_rmse_by_graph_distance(
    predictions: ArrayLike,
    targets: ArrayLike,
    distance: ArrayLike,
) -> list[dict[str, Any]]:
    """Report per-variable rollout RMSE at every finite graph distance."""

    pred = np.asarray(predictions, dtype=np.float64)
    truth = np.asarray(targets, dtype=np.float64)
    if pred.shape != truth.shape:
        raise ValueError("predictions and targets must share shape")
    _validate_rollout_shape(pred.shape, "predictions")
    if not np.all(np.isfinite(pred)) or not np.all(np.isfinite(truth)):
        raise ValueError("predictions and targets must contain only finite values")
    distances = np.asarray(distance, dtype=np.int64).reshape(-1)
    if distances.shape != (pred.shape[1],):
        raise ValueError("distance must contain one value per rollout node")

    rows: list[dict[str, Any]] = []
    for value in sorted(int(item) for item in np.unique(distances) if item >= 0):
        metrics = _masked_rollout_metrics(pred, truth, distances == value)
        rows.append({"distance": value, **metrics})
    return rows


def _masked_rollout_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    count = int(np.count_nonzero(mask))
    if count == 0:
        return {
            "num_nodes": 0,
            "rollout_rmse": None,
            "final_step_rmse": None,
            "cumulative_rmse": None,
            "max_abs_error": None,
        }

    error = predictions[:, mask, :] - targets[:, mask, :]
    mse_per_step = np.mean(error * error, axis=1)
    cumulative_mse = (
        np.cumsum(mse_per_step, axis=0) / np.arange(1, error.shape[0] + 1)[:, None]
    )
    return {
        "num_nodes": count,
        "rollout_rmse": np.sqrt(np.mean(mse_per_step, axis=0)).tolist(),
        "final_step_rmse": np.sqrt(mse_per_step[-1]).tolist(),
        "cumulative_rmse": np.sqrt(cumulative_mse).tolist(),
        "max_abs_error": float(np.max(np.abs(error))),
    }


def _target_digest(shape: tuple[int, ...]) -> Any:
    digest = sha256()
    _update_header(
        digest,
        {
            "schema": _TARGET_HASH_SCHEMA,
            "shape": list(shape),
            "dtype": np.dtype("<f4").str,
        },
    )
    return digest


def _update_header(digest: Any, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)


def _update_array_bytes(digest: Any, value: ArrayLike, dtype: np.dtype) -> None:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    digest.update(array.tobytes(order="C"))


def _primitive_shape(group: GroupLike) -> tuple[int, int]:
    shapes: list[tuple[int, int]] = []
    missing = [key for key in PRIMITIVE_KEYS if key not in group]
    if missing:
        raise KeyError(f"missing primitive arrays: {missing}")
    for key in PRIMITIVE_KEYS:
        shape = _shape(group[key])
        if len(shape) == 2:
            time_steps, num_nodes = shape
        elif len(shape) == 3 and shape[-1] == 1:
            time_steps, num_nodes = shape[:2]
        else:
            raise ValueError(f"{key} must have shape (T,N) or (T,N,1), got {shape}")
        shapes.append((time_steps, num_nodes))
    if len(set(shapes)) != 1:
        raise ValueError(f"primitive arrays have inconsistent shapes: {shapes}")
    return shapes[0]


def _as_time_node_scalar(value: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 2:
        array = array[..., None]
    if array.ndim != 3 or array.shape[-1] != 1:
        raise ValueError(f"{name} must have shape (T,N) or (T,N,1)")
    return array


def _frame_or_static(
    value: ArrayLike,
    frame: int,
    time_steps: int,
    name: str,
) -> np.ndarray:
    shape = _shape(value)
    temporal_matrix = name in {"node_type", "Mach"} and len(shape) == 2
    if shape and shape[0] == time_steps and (len(shape) >= 3 or temporal_matrix):
        return np.asarray(value[frame])
    return np.asarray(value)


def _finite_min_max(value: ArrayLike) -> tuple[float, float]:
    shape = _shape(value)
    if not shape:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(array)):
            raise ValueError("Mach contains nonfinite values")
        if not array.size:
            raise ValueError("Mach contains no values")
        return float(np.min(array)), float(np.max(array))

    minimum = np.inf
    maximum = -np.inf
    for index in range(shape[0]):
        array = np.asarray(value[index], dtype=np.float64)
        if not np.all(np.isfinite(array)):
            raise ValueError("Mach contains nonfinite values")
        if array.size:
            minimum = min(minimum, float(np.min(array)))
            maximum = max(maximum, float(np.max(array)))
    if not np.isfinite(minimum) or not np.isfinite(maximum):
        raise ValueError("Mach contains no values")
    return minimum, maximum


def _validate_rollout_shape(shape: tuple[int, ...], name: str) -> None:
    if len(shape) != 3 or shape[-1] != len(PRIMITIVE_KEYS):
        raise ValueError(f"{name} must have shape (steps, nodes, 4), got {shape}")
    if shape[0] < 1 or shape[1] < 1:
        raise ValueError(f"{name} must contain at least one step and node")


def _validate_edges(edges: np.ndarray, num_nodes: int) -> None:
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"edges must have shape (num_edges, 2), got {edges.shape}")
    if edges.size and (np.min(edges) < 0 or np.max(edges) >= num_nodes):
        raise ValueError("edges contain node indices outside the graph")


def _shape(value: ArrayLike) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        shape = np.asarray(value).shape
    return tuple(int(item) for item in shape)


def _git_output(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _git_blob_sha256(repo: Path, commit: str, relative_path: str) -> str:
    try:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "cat-file",
                "blob",
                f"{commit}:{relative_path}",
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        raise ValueError(
            f"Git commit {commit} does not contain {relative_path}"
        ) from exc
    return sha256(completed.stdout).hexdigest()
