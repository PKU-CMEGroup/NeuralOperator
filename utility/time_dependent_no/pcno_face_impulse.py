"""Global PCNO features with a shared physical-face impulse decoder.

This module implements the first structured-target row on the validated
shock--vortex finite-volume testbed.  A PCNO produces one latent vector per
cell.  A small shared face network then emits one owner-oriented cumulative
impulse per physical face, and a fixed finite-volume decoder advances the cell
state.  Interior impulses are orientation antisymmetric by construction;
boundary impulses are causal functions of the current state and geometry.

The model is trained through decoded next-state loss.  It does not use or claim
reference-impulse supervision, and the fixed-geometry constructor does not by
itself establish mesh or resolution transfer.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from utility.time_dependent_no.pcno_euler2d import (
    Euler2DNormalization,
    NUM_EULER_COMPONENTS,
    PCNOEuler2DResidual,
    PCNOEuler2DShardStore,
)
from utility.time_dependent_no.pcno_fv_geometry import (
    build_pcno_finite_volume_geometry,
)
from utility.time_dependent_no.shock_vortex_family import (
    REFERENCE_ARTIFACT_SCHEMA,
    family_case_by_id,
    family_case_provenance,
    load_shock_vortex_family_manifest,
)

TARGET_KIND = "state_loss_only_shared_face_impulse"
DIVERGENCE_ACTIVE_TARGET_KIND = "divergence_active_shared_face_impulse"
FACE_TARGET_KINDS = frozenset({TARGET_KIND, DIVERGENCE_ACTIVE_TARGET_KIND})
REQUIRED_BOUNDARY_NAMES = ("interior", "x_min", "x_max", "y_min", "y_max")


@dataclass(frozen=True)
class FixedFVFaceGeometry:
    """Validated physical-face data for the single-mesh structured target."""

    cell_center: np.ndarray
    cell_volume: np.ndarray
    face_center: np.ndarray
    face_measure: np.ndarray
    face_normal: np.ndarray
    face_owner: np.ndarray
    face_neighbor: np.ndarray
    face_boundary_tag: np.ndarray
    boundary_tag_names: tuple[str, ...]
    physical_geometry_digest: str
    graph_geometry_digest: str
    source_family_id: str
    source_family_manifest_digest: str
    source_key: str
    source_reference_sha256: str
    fixed_delta_t: float

    def target_contract(
        self,
        *,
        target_kind: str = TARGET_KIND,
        supervision: str = "decoded_next_state_loss_only",
        reference_impulse_supervision: bool = False,
    ) -> dict[str, Any]:
        """Return provenance without exposing machine-specific source paths."""

        if target_kind not in FACE_TARGET_KINDS:
            raise ValueError(f"unsupported shared-face target kind: {target_kind}")

        return {
            "target_kind": target_kind,
            "supervision": supervision,
            "reference_impulse_supervision": bool(reference_impulse_supervision),
            "decoder": "fixed_owner_oriented_finite_volume_divergence",
            "interior_shared_face": True,
            "interior_orientation_antisymmetric": True,
            "boundary_exchange": (
                "predicted_from_current_state_and_geometry_only; "
                "y walls exchange y momentum only"
            ),
            "fixed_geometry_only": True,
            "physical_geometry_digest": self.physical_geometry_digest,
            "graph_geometry_digest": self.graph_geometry_digest,
            "source_family_id": self.source_family_id,
            "source_family_manifest_digest": self.source_family_manifest_digest,
            "source_geometry_key": self.source_key,
            "source_reference_sha256": self.source_reference_sha256,
            "fixed_delta_t": self.fixed_delta_t,
            "physical_geometry_arrays_loaded": [
                "cell_centers",
                "cell_volume",
                "face_centers",
                "face_measure",
                "face_normal",
                "face_owner",
                "face_neighbor",
                "face_boundary_tag",
                "boundary_tag_names_json",
                "physical_times",
            ],
            "reference_face_impulse_arrays_loaded": bool(reference_impulse_supervision),
        }


@dataclass(frozen=True)
class FVReferenceFaceTrajectory:
    """Validated reference states and accepted macro-step face impulses."""

    case_id: str
    split: str
    physical_times: np.ndarray
    conservative_states: np.ndarray
    cumulative_face_impulses: np.ndarray
    source_reference_sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_arrays(*values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def load_fixed_fv_face_geometry(
    family_root: str | Path,
    store: PCNOEuler2DShardStore,
    *,
    source_key: str,
) -> FixedFVFaceGeometry:
    """Close the physical-face-to-PCNO contract using one training case.

    The frozen family manifest establishes common geometry across cases. This
    loader independently rebuilds the graph tensors from one training-case
    physical mesh and compares them with the published shards. It never
    materializes the reference face-impulse field.
    """

    if store.manifest.get("dataset") != "shock_vortex_fv_family":
        raise ValueError("shared-face training requires shock_vortex_fv_family")
    if store.manifest.get("mesh_to_graph_map") != (
        "identity_FV_cell_index_to_PCNO_node_index"
    ):
        raise ValueError("shared-face training requires the identity FV map")
    if store.manifest.get("weight_provenance") != (
        "validated_physical_cell_volume_normalized"
    ):
        raise ValueError("shared-face training requires physical cell volumes")

    family_root = Path(family_root)
    family_manifest = load_shock_vortex_family_manifest(
        family_root / "family_manifest.json"
    )
    family_digest = str(family_manifest["manifest_digest_sha256"])
    family_id = str(family_manifest["family_id"])
    if family_digest != store.manifest.get("source_family_manifest_digest"):
        raise ValueError("source family and shard manifest digests differ")
    if family_id != store.manifest.get("source_family_id"):
        raise ValueError("source family and shard family identifiers differ")

    cases = {str(case["case_id"]): case for case in family_manifest["cases"]}
    for key in store.keys:
        if key not in cases:
            raise ValueError(f"shard trajectory is absent from source family: {key}")
        entry = store.entry(key)
        case = cases[key]
        for name in ("split", "split_group_id", "parameters"):
            if entry.get(name) != case.get(name):
                raise ValueError(f"source family and shard {name} differ for {key}")

    graph_digests = {
        str(store.entry(key).get("geometry_digest", "")) for key in store.keys
    }
    if len(graph_digests) != 1 or "" in graph_digests:
        raise ValueError("shared-face target requires one fixed graph geometry")
    graph_geometry_digest = next(iter(graph_digests))
    source_key = str(source_key)
    if source_key not in store.keys:
        raise ValueError("source geometry key is absent from the shard store")
    source_case = family_case_by_id(family_manifest, source_key)
    if (
        source_case["split"] != "train"
        or store.entry(source_key).get("split") != "train"
    ):
        raise ValueError("source face geometry must come from a training case")

    artifact_path = family_root / source_key / "reference.npz"
    if not artifact_path.is_file():
        raise FileNotFoundError(artifact_path)
    source_reference_sha256 = _sha256_file(artifact_path)
    if source_reference_sha256 != store.entry(source_key).get(
        "source_reference_sha256"
    ):
        raise ValueError("source reference digest differs from the shard contract")

    required = {
        "schema",
        "physical_times",
        "cell_centers",
        "cell_volume",
        "face_centers",
        "face_measure",
        "face_normal",
        "face_owner",
        "face_neighbor",
        "face_boundary_tag",
        "boundary_tag_names_json",
        "family_contract_json",
    }
    with np.load(artifact_path, allow_pickle=False) as artifact:
        missing = sorted(required - set(artifact.files))
        if missing:
            raise ValueError(f"source reference is missing geometry arrays: {missing}")
        if str(artifact["schema"].item()) != REFERENCE_ARTIFACT_SCHEMA:
            raise ValueError("unexpected source reference schema")
        times = np.array(artifact["physical_times"], dtype=np.float64)
        centers = np.array(artifact["cell_centers"], dtype=np.float64)
        volume = np.array(artifact["cell_volume"], dtype=np.float64)
        face_center = np.array(artifact["face_centers"], dtype=np.float64)
        face_measure = np.array(artifact["face_measure"], dtype=np.float64)
        face_normal = np.array(artifact["face_normal"], dtype=np.float64)
        owner = np.array(artifact["face_owner"], dtype=np.int64)
        neighbor = np.array(artifact["face_neighbor"], dtype=np.int64)
        boundary_tag = np.array(artifact["face_boundary_tag"], dtype=np.int64)
        boundary_names = tuple(
            json.loads(str(artifact["boundary_tag_names_json"].item()))
        )
        provenance = json.loads(str(artifact["family_contract_json"].item()))

    if provenance != family_case_provenance(family_manifest, source_key):
        raise ValueError("source reference family provenance mismatch")
    if boundary_names != REQUIRED_BOUNDARY_NAMES:
        raise ValueError("source boundary tags differ from the face decoder")
    if (
        times.ndim != 1
        or times.size < 2
        or not np.all(np.isfinite(times))
        or not np.all(np.diff(times) > 0.0)
    ):
        raise ValueError("source physical times are invalid")
    deltas = np.diff(times)
    if not np.allclose(deltas, deltas[0], rtol=0.0, atol=1.0e-14):
        raise ValueError("shared-face training requires a fixed saved delta t")
    if not np.array_equal(
        np.asarray(store.array(source_key, "physical_times"), dtype=np.float64),
        times,
    ):
        raise ValueError("source and shard physical times differ")
    declared_dt = float(store.manifest.get("dt", float("nan")))
    if not np.isclose(declared_dt, deltas[0], rtol=0.0, atol=1.0e-14):
        raise ValueError("source and shard delta t differ")

    rebuilt = build_pcno_finite_volume_geometry(
        cell_centers=centers,
        cell_volume=volume,
        face_owner=owner,
        face_neighbor=neighbor,
        face_boundary_tag=boundary_tag,
        boundary_tag_names=boundary_names,
        gradient_rcond=float(store.manifest.get("gradient_rcond", 1.0e-12)),
    )
    num_faces = owner.size
    if (
        face_center.shape != (num_faces, 2)
        or face_measure.shape != (num_faces,)
        or face_normal.shape != (num_faces, 2)
    ):
        raise ValueError("physical face arrays have incompatible shapes")
    if (
        not np.all(np.isfinite(face_center))
        or not np.all(np.isfinite(face_measure))
        or not np.all(np.isfinite(face_normal))
        or np.any(face_measure <= 0.0)
    ):
        raise ValueError("physical face geometry must be positive and finite")
    interior = neighbor >= 0
    if not np.any(interior) or not np.any(~interior):
        raise ValueError("physical mesh must contain interior and boundary faces")
    unit_error = np.max(np.abs(np.linalg.norm(face_normal, axis=1) - 1.0))
    interior_orientation = np.einsum(
        "ij,ij->i",
        centers[neighbor[interior]] - centers[owner[interior]],
        face_normal[interior],
    )
    boundary_orientation = np.einsum(
        "ij,ij->i",
        face_center[~interior] - centers[owner[~interior]],
        face_normal[~interior],
    )
    if (
        unit_error > 1.0e-12
        or np.any(interior_orientation <= 0.0)
        or np.any(boundary_orientation <= 0.0)
    ):
        raise ValueError("face normal or owner-orientation contract failed")

    rebuilt_arrays = {
        "nodes": rebuilt.nodes,
        "edges": rebuilt.edges,
        "node_type": rebuilt.node_type,
        "node_measures": rebuilt.node_measures,
        "node_weights": rebuilt.node_weights,
        "node_rhos": rebuilt.node_rhos,
        "directed_edges": rebuilt.directed_edges,
        "edge_gradient_weights": rebuilt.edge_gradient_weights,
        "mesh_cell_to_graph_node": rebuilt.mesh_cell_to_graph_node,
        "face_to_directed_edge": rebuilt.face_to_directed_edge,
    }
    for name, expected in rebuilt_arrays.items():
        actual = np.asarray(store.array(source_key, name))
        expected = np.asarray(expected, dtype=actual.dtype)
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            raise ValueError(f"source physical geometry and shard {name} differ")
    if _sha256_arrays(*rebuilt_arrays.values()) != graph_geometry_digest:
        raise ValueError("rebuilt graph digest differs from the shard contract")

    return FixedFVFaceGeometry(
        cell_center=centers,
        cell_volume=volume,
        face_center=face_center,
        face_measure=face_measure,
        face_normal=face_normal,
        face_owner=owner,
        face_neighbor=neighbor,
        face_boundary_tag=boundary_tag,
        boundary_tag_names=boundary_names,
        physical_geometry_digest=_sha256_arrays(
            centers,
            volume,
            face_center,
            face_measure,
            face_normal,
            owner,
            neighbor,
            boundary_tag,
        ),
        graph_geometry_digest=graph_geometry_digest,
        source_family_id=family_id,
        source_family_manifest_digest=family_digest,
        source_key=source_key,
        source_reference_sha256=source_reference_sha256,
        fixed_delta_t=float(deltas[0]),
    )


def load_fv_reference_face_trajectory(
    family_root: str | Path,
    store: PCNOEuler2DShardStore,
    geometry: FixedFVFaceGeometry,
    *,
    key: str,
) -> FVReferenceFaceTrajectory:
    """Load one digest-bound reference trajectory with explicit face labels.

    This function is intentionally separate from ``load_fixed_fv_face_geometry``:
    state-loss-only training must never call it.  Callers are responsible for
    enforcing their train/validation/test access policy before materialization.
    """

    key = str(key)
    if key not in store.keys:
        raise ValueError(f"reference trajectory is absent from the shard store: {key}")
    family_root = Path(family_root)
    family_manifest = load_shock_vortex_family_manifest(
        family_root / "family_manifest.json"
    )
    if str(family_manifest["manifest_digest_sha256"]) != (
        geometry.source_family_manifest_digest
    ):
        raise ValueError("reference family digest differs from fixed geometry")
    if str(family_manifest["family_id"]) != geometry.source_family_id:
        raise ValueError("reference family identifier differs from fixed geometry")
    case = family_case_by_id(family_manifest, key)
    if case["split"] != store.entry(key).get("split"):
        raise ValueError(f"reference and shard split differ for {key}")

    artifact_path = family_root / key / "reference.npz"
    if not artifact_path.is_file():
        raise FileNotFoundError(artifact_path)
    artifact_sha256 = _sha256_file(artifact_path)
    if artifact_sha256 != str(store.entry(key).get("source_reference_sha256")):
        raise ValueError(f"source reference digest differs for {key}")
    required = {
        "schema",
        "physical_times",
        "conservative_states",
        "cumulative_accepted_substep_face_impulses",
        "cell_centers",
        "cell_volume",
        "face_centers",
        "face_measure",
        "face_normal",
        "face_owner",
        "face_neighbor",
        "face_boundary_tag",
        "boundary_tag_names_json",
        "family_contract_json",
    }
    with np.load(artifact_path, allow_pickle=False) as artifact:
        missing = sorted(required - set(artifact.files))
        if missing:
            raise ValueError(f"reference {key} is missing arrays: {missing}")
        if str(artifact["schema"].item()) != REFERENCE_ARTIFACT_SCHEMA:
            raise ValueError(f"reference schema differs for {key}")
        times = np.array(artifact["physical_times"], dtype=np.float64)
        states = np.array(artifact["conservative_states"], dtype=np.float64)
        impulses = np.array(
            artifact["cumulative_accepted_substep_face_impulses"],
            dtype=np.float64,
        )
        provenance = json.loads(str(artifact["family_contract_json"].item()))
        boundary_names = tuple(
            json.loads(str(artifact["boundary_tag_names_json"].item()))
        )
        geometry_arrays = {
            "cell_centers": np.array(artifact["cell_centers"], dtype=np.float64),
            "cell_volume": np.array(artifact["cell_volume"], dtype=np.float64),
            "face_centers": np.array(artifact["face_centers"], dtype=np.float64),
            "face_measure": np.array(artifact["face_measure"], dtype=np.float64),
            "face_normal": np.array(artifact["face_normal"], dtype=np.float64),
            "face_owner": np.array(artifact["face_owner"], dtype=np.int64),
            "face_neighbor": np.array(artifact["face_neighbor"], dtype=np.int64),
            "face_boundary_tag": np.array(
                artifact["face_boundary_tag"], dtype=np.int64
            ),
        }

    expected_geometry = {
        "cell_centers": geometry.cell_center,
        "cell_volume": geometry.cell_volume,
        "face_centers": geometry.face_center,
        "face_measure": geometry.face_measure,
        "face_normal": geometry.face_normal,
        "face_owner": geometry.face_owner,
        "face_neighbor": geometry.face_neighbor,
        "face_boundary_tag": geometry.face_boundary_tag,
    }
    for name, actual in geometry_arrays.items():
        if not np.array_equal(actual, np.asarray(expected_geometry[name])):
            raise ValueError(f"reference {key} differs in fixed geometry {name}")
    if boundary_names != geometry.boundary_tag_names:
        raise ValueError(f"reference {key} boundary names differ")
    if provenance != family_case_provenance(family_manifest, key):
        raise ValueError(f"reference {key} family provenance differs")
    shard_times = np.asarray(store.array(key, "physical_times"), dtype=np.float64)
    if not np.array_equal(times, shard_times):
        raise ValueError(f"reference and shard times differ for {key}")
    expected_state_shape = (times.size, geometry.cell_volume.size, 4)
    expected_impulse_shape = (times.size - 1, geometry.face_owner.size, 4)
    if states.shape != expected_state_shape:
        raise ValueError(f"reference {key} state shape differs")
    if impulses.shape != expected_impulse_shape:
        raise ValueError(f"reference {key} impulse shape differs")
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(impulses)):
        raise ValueError(f"reference {key} contains nonfinite values")
    return FVReferenceFaceTrajectory(
        case_id=key,
        split=str(case["split"]),
        physical_times=times,
        conservative_states=states,
        cumulative_face_impulses=impulses,
        source_reference_sha256=artifact_sha256,
    )


def decode_owner_oriented_face_impulse_torch(
    face_impulse: torch.Tensor,
    *,
    cell_volume: torch.Tensor,
    face_owner: torch.Tensor,
    face_neighbor: torch.Tensor,
) -> torch.Tensor:
    """Decode cumulative owner-oriented face impulses into cell increments.

    Interior impulses are subtracted from the owner and added to the neighbor.
    Boundary impulses are subtracted from the owner only.  Consequently,
    ``sum(V * delta_U)`` is exactly the negative predicted boundary exchange,
    up to floating-point scatter accumulation.
    """

    if face_impulse.ndim != 3 or face_impulse.shape[-1] != NUM_EULER_COMPONENTS:
        raise ValueError("face_impulse must have shape [batch, faces, 4]")
    batch_size, num_faces, num_components = face_impulse.shape
    volume = torch.as_tensor(
        cell_volume, dtype=face_impulse.dtype, device=face_impulse.device
    )
    if volume.ndim == 1:
        volume = volume.unsqueeze(0).expand(batch_size, -1)
    if volume.ndim == 3 and volume.shape[-1] == 1:
        volume = volume[..., 0]
    if volume.ndim != 2 or volume.shape[0] != batch_size:
        raise ValueError(
            "cell_volume must have shape [cells], [B,cells], or [B,cells,1]"
        )
    if not bool(torch.isfinite(volume).all()) or bool((volume <= 0.0).any()):
        raise ValueError("cell_volume must be positive and finite")

    owner = torch.as_tensor(face_owner, dtype=torch.int64, device=face_impulse.device)
    neighbor = torch.as_tensor(
        face_neighbor, dtype=torch.int64, device=face_impulse.device
    )
    if owner.shape != (num_faces,) or neighbor.shape != (num_faces,):
        raise ValueError("face_owner and face_neighbor must have shape [faces]")
    num_cells = volume.shape[1]
    if bool((owner < 0).any()) or bool((owner >= num_cells).any()):
        raise ValueError("face owner lies outside the cell axis")
    if bool((neighbor < -1).any()) or bool((neighbor >= num_cells).any()):
        raise ValueError("face neighbor lies outside the cell axis")
    interior = neighbor >= 0
    if bool((owner[interior] == neighbor[interior]).any()):
        raise ValueError("an interior face cannot connect a cell to itself")

    integrated = face_impulse.new_zeros((batch_size, num_cells, num_components))
    owner_index = owner.reshape(1, -1, 1).expand(batch_size, -1, num_components)
    integrated.scatter_add_(1, owner_index, -face_impulse)
    if bool(interior.any()):
        neighbor_index = (
            neighbor[interior].reshape(1, -1, 1).expand(batch_size, -1, num_components)
        )
        integrated.scatter_add_(1, neighbor_index, face_impulse[:, interior])
    return integrated / volume.unsqueeze(-1)


def _activation(name: str) -> nn.Module:
    normalized = str(name).lower()
    if normalized == "gelu":
        return nn.GELU()
    if normalized in {"silu", "swish"}:
        return nn.SiLU()
    raise ValueError("face decoder activation must be gelu or silu")


def _face_mlp(in_features: int, hidden_features: int, act: str) -> nn.Sequential:
    if hidden_features < 1:
        raise ValueError("face_hidden_dim must be positive")
    return nn.Sequential(
        nn.Linear(in_features, hidden_features),
        _activation(act),
        nn.Linear(hidden_features, hidden_features),
        _activation(act),
        nn.Linear(hidden_features, NUM_EULER_COMPONENTS),
    )


class PCNOEuler2DSharedFaceImpulse(PCNOEuler2DResidual):
    """PCNO latent field followed by a shared finite-volume face decoder.

    Geometry is a frozen model buffer for the current single-mesh dynamic
    benchmark.  The full geometry digest is part of ``model_config`` and must be
    checked before checkpoint replay.  This is an honest fixed-geometry row,
    not yet a cross-mesh adapter.
    """

    def __init__(
        self,
        *,
        normalization: Euler2DNormalization,
        cell_volume: np.ndarray | torch.Tensor,
        face_center: np.ndarray | torch.Tensor,
        face_measure: np.ndarray | torch.Tensor,
        face_normal: np.ndarray | torch.Tensor,
        face_owner: np.ndarray | torch.Tensor,
        face_neighbor: np.ndarray | torch.Tensor,
        face_boundary_tag: np.ndarray | torch.Tensor,
        boundary_tag_names: Sequence[str],
        geometry_digest: str,
        fixed_delta_t: float,
        graph_geometry_digest: str | None = None,
        target_kind: str = TARGET_KIND,
        supervision: str = "decoded_next_state_loss_only",
        reference_impulse_supervision: bool = False,
        k_max: int = 8,
        domain_lengths: Sequence[float] = (2.0, 1.0),
        layers: Sequence[int] = (128, 128, 128, 128, 128),
        fc_dim: int = 128,
        latent_dim: int = 32,
        face_hidden_dim: int = 128,
        nmeasures: int = 1,
        act: str = "gelu",
        zero_initialize: bool = True,
    ) -> None:
        if latent_dim < NUM_EULER_COMPONENTS:
            raise ValueError("latent_dim must be at least four")
        if not geometry_digest:
            raise ValueError("geometry_digest must be nonempty")
        if not np.isfinite(fixed_delta_t) or fixed_delta_t <= 0.0:
            raise ValueError("fixed_delta_t must be positive and finite")
        if target_kind not in FACE_TARGET_KINDS:
            raise ValueError(f"unsupported shared-face target kind: {target_kind}")
        if not supervision:
            raise ValueError("supervision must be nonempty")
        names = tuple(map(str, boundary_tag_names))
        if names != REQUIRED_BOUNDARY_NAMES:
            raise ValueError(
                "boundary_tag_names must be interior, x_min, x_max, y_min, y_max"
            )

        volume = torch.as_tensor(cell_volume, dtype=torch.float32).reshape(-1)
        centers = torch.as_tensor(face_center, dtype=torch.float32)
        measure = torch.as_tensor(face_measure, dtype=torch.float32).reshape(-1)
        normals = torch.as_tensor(face_normal, dtype=torch.float32)
        owner = torch.as_tensor(face_owner, dtype=torch.int64).reshape(-1)
        neighbor = torch.as_tensor(face_neighbor, dtype=torch.int64).reshape(-1)
        tags = torch.as_tensor(face_boundary_tag, dtype=torch.int64).reshape(-1)
        num_faces = owner.numel()
        if (
            volume.numel() < 1
            or not bool(torch.isfinite(volume).all())
            or bool((volume <= 0.0).any())
        ):
            raise ValueError("cell_volume must be a nonempty positive finite vector")
        if centers.shape != (num_faces, 2) or normals.shape != (num_faces, 2):
            raise ValueError("face centers and normals must have shape [faces,2]")
        if (
            measure.shape != (num_faces,)
            or neighbor.shape != (num_faces,)
            or tags.shape != (num_faces,)
        ):
            raise ValueError("all face arrays must share their face dimension")
        if not bool(torch.isfinite(centers).all()) or not bool(
            torch.isfinite(normals).all()
        ):
            raise ValueError("face centers and normals must be finite")
        if not bool(torch.isfinite(measure).all()) or bool((measure <= 0.0).any()):
            raise ValueError("face measures must be positive and finite")
        normal_error = torch.max(
            torch.abs(torch.linalg.vector_norm(normals, dim=1) - 1.0)
        )
        if float(normal_error) > 1.0e-6:
            raise ValueError("face normals must be unit length")
        num_cells = volume.numel()
        if bool((owner < 0).any()) or bool((owner >= num_cells).any()):
            raise ValueError("face owner lies outside the cell axis")
        if bool((neighbor < -1).any()) or bool((neighbor >= num_cells).any()):
            raise ValueError("face neighbor lies outside the cell axis")
        interior = neighbor >= 0
        if bool((owner[interior] == neighbor[interior]).any()):
            raise ValueError("an interior face cannot connect a cell to itself")
        if bool((tags < 0).any()) or bool((tags >= len(names)).any()):
            raise ValueError("face boundary tag lies outside boundary_tag_names")
        if bool((tags[interior] != 0).any()) or bool((tags[~interior] == 0).any()):
            raise ValueError("face connectivity and boundary tags disagree")

        super().__init__(
            normalization=normalization,
            k_max=k_max,
            domain_lengths=domain_lengths,
            layers=layers,
            fc_dim=fc_dim,
            nmeasures=nmeasures,
            act=act,
            zero_initialize=False,
        )
        self.backbone.fc2 = nn.Linear(self.backbone.fc2.in_features, int(latent_dim))
        self.backbone.out_dim = int(latent_dim)
        self.latent_dim = int(latent_dim)
        self.face_hidden_dim = int(face_hidden_dim)
        self.geometry_digest = str(geometry_digest)
        self.graph_geometry_digest = (
            None if graph_geometry_digest is None else str(graph_geometry_digest)
        )
        self.fixed_delta_t = float(fixed_delta_t)
        self.boundary_tag_names = names
        self.target_kind = str(target_kind)
        self.supervision = str(supervision)
        self.reference_impulse_supervision = bool(reference_impulse_supervision)

        mean_volume = torch.mean(volume)
        degree = torch.bincount(owner, minlength=num_cells).to(torch.float32)
        if bool(interior.any()):
            degree = degree + torch.bincount(
                neighbor[interior], minlength=num_cells
            ).to(torch.float32)
        if bool((degree <= 0.0).any()):
            raise ValueError("every cell must touch at least one physical face")
        share = volume[owner] / degree[owner]
        share[interior] = 0.5 * (
            volume[owner[interior]] / degree[owner[interior]]
            + volume[neighbor[interior]] / degree[neighbor[interior]]
        )
        impulse_scale = (
            share[:, None]
            * torch.as_tensor(normalization.residual_scale, dtype=torch.float32)[
                None, :
            ]
        )

        self.register_buffer("cell_volume", volume)
        self.register_buffer("face_center", centers)
        self.register_buffer("face_measure", measure)
        self.register_buffer("face_normal", normals)
        self.register_buffer("face_owner", owner)
        self.register_buffer("face_neighbor", neighbor)
        self.register_buffer("face_boundary_tag", tags)
        self.register_buffer("face_is_interior", interior)
        self.register_buffer("face_impulse_scale", impulse_scale)
        self.register_buffer("mean_cell_volume", mean_volume.reshape(()))

        # Oriented interior features contain both endpoint latents/states plus
        # owner-to-neighbor geometry.  Shared face features are repeated in the
        # reversed call, while normals and endpoint ordering are reversed.
        interior_features = 2 * self.latent_dim + 2 * NUM_EULER_COMPONENTS + 8
        boundary_features = self.latent_dim + NUM_EULER_COMPONENTS + 7 + len(names)
        self.interior_head = _face_mlp(interior_features, self.face_hidden_dim, act)
        self.boundary_head = _face_mlp(boundary_features, self.face_hidden_dim, act)
        if zero_initialize:
            self.zero_initialize_update_head()

    def zero_initialize_update_head(self) -> None:
        """Initialize every decoded face impulse to zero exactly."""

        for head in (self.interior_head, self.boundary_head):
            final = head[-1]
            if not isinstance(final, nn.Linear):
                raise TypeError("face decoder must end in a linear layer")
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def model_config(self) -> dict[str, Any]:
        config = super().model_config()
        config.update(
            {
                "model": "PCNOEuler2DSharedFaceImpulse",
                "target_kind": self.target_kind,
                "out_dim": self.latent_dim,
                "latent_dim": self.latent_dim,
                "face_hidden_dim": self.face_hidden_dim,
                "num_physical_faces": int(self.face_owner.numel()),
                "num_interior_faces": int(self.face_is_interior.sum()),
                "geometry_digest": self.geometry_digest,
                "graph_geometry_digest": self.graph_geometry_digest,
                "fixed_delta_t": self.fixed_delta_t,
                "boundary_tag_names": list(self.boundary_tag_names),
                "interior_orientation": (
                    "0.5*(phi(owner,neighbor,normal)-phi(neighbor,owner,-normal))"
                ),
                "boundary_contract": (
                    "current_state_and_geometry_only; y walls exchange y momentum only"
                ),
                "supervision": self.supervision,
                "reference_impulse_supervision": (self.reference_impulse_supervision),
            }
        )
        return config

    def _shared_geometry_features(
        self, face_index: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        center_scale = self.face_center.new_tensor(self.domain_lengths)
        centers = self.face_center[face_index] / center_scale
        measures = self.face_measure[face_index, None] / torch.sqrt(
            self.mean_cell_volume
        )
        owner = self.face_owner[face_index]
        owner_volume = self.cell_volume[owner, None] / self.mean_cell_volume
        return centers, measures, owner_volume

    def _interior_normalized_impulse(
        self, latent: torch.Tensor, normalized_state: torch.Tensor
    ) -> torch.Tensor:
        face_index = torch.nonzero(self.face_is_interior, as_tuple=False).flatten()
        owner = self.face_owner[face_index]
        neighbor = self.face_neighbor[face_index]
        owner_latent = latent[:, owner]
        neighbor_latent = latent[:, neighbor]
        owner_state = normalized_state[:, owner]
        neighbor_state = normalized_state[:, neighbor]
        normal = (
            self.face_normal[face_index].unsqueeze(0).expand(latent.shape[0], -1, -1)
        )
        centers, measures, owner_volume = self._shared_geometry_features(face_index)
        centers = centers.unsqueeze(0).expand(latent.shape[0], -1, -1)
        measures = measures.unsqueeze(0).expand(latent.shape[0], -1, -1)
        owner_volume = owner_volume.unsqueeze(0).expand(latent.shape[0], -1, -1)
        neighbor_volume = (
            (self.cell_volume[neighbor, None] / self.mean_cell_volume)
            .unsqueeze(0)
            .expand(latent.shape[0], -1, -1)
        )
        forward_features = torch.cat(
            (
                owner_latent,
                neighbor_latent,
                owner_state,
                neighbor_state,
                normal,
                centers,
                measures,
                owner_volume,
                neighbor_volume,
                torch.ones_like(measures),
            ),
            dim=-1,
        )
        reverse_features = torch.cat(
            (
                neighbor_latent,
                owner_latent,
                neighbor_state,
                owner_state,
                -normal,
                centers,
                measures,
                neighbor_volume,
                owner_volume,
                torch.ones_like(measures),
            ),
            dim=-1,
        )
        return 0.5 * (
            self.interior_head(forward_features) - self.interior_head(reverse_features)
        )

    def _boundary_normalized_impulse(
        self, latent: torch.Tensor, normalized_state: torch.Tensor
    ) -> torch.Tensor:
        face_index = torch.nonzero(~self.face_is_interior, as_tuple=False).flatten()
        owner = self.face_owner[face_index]
        owner_latent = latent[:, owner]
        owner_state = normalized_state[:, owner]
        normal = (
            self.face_normal[face_index].unsqueeze(0).expand(latent.shape[0], -1, -1)
        )
        centers, measures, owner_volume = self._shared_geometry_features(face_index)
        centers = centers.unsqueeze(0).expand(latent.shape[0], -1, -1)
        measures = measures.unsqueeze(0).expand(latent.shape[0], -1, -1)
        owner_volume = owner_volume.unsqueeze(0).expand(latent.shape[0], -1, -1)
        tags = F.one_hot(
            self.face_boundary_tag[face_index], num_classes=len(self.boundary_tag_names)
        ).to(dtype=latent.dtype)
        tags = tags.unsqueeze(0).expand(latent.shape[0], -1, -1)
        features = torch.cat(
            (
                owner_latent,
                owner_state,
                normal,
                centers,
                measures,
                owner_volume,
                torch.ones_like(measures),
                tags,
            ),
            dim=-1,
        )
        output = self.boundary_head(features)
        wall_tags = {
            self.boundary_tag_names.index("y_min"),
            self.boundary_tag_names.index("y_max"),
        }
        wall = torch.zeros(face_index.numel(), dtype=torch.bool, device=latent.device)
        for tag in wall_tags:
            wall |= self.face_boundary_tag[face_index] == tag
        component_mask = torch.ones(
            (face_index.numel(), NUM_EULER_COMPONENTS),
            dtype=latent.dtype,
            device=latent.device,
        )
        component_mask[wall] = 0.0
        component_mask[wall, 2] = 1.0
        return output * component_mask.unsqueeze(0)

    def predict_face_impulse(
        self,
        current_conservative: torch.Tensor,
        *,
        node_mask: torch.Tensor,
        nodes: torch.Tensor,
        node_weights: torch.Tensor,
        node_rhos: torch.Tensor,
        directed_edges: torch.Tensor,
        edge_gradient_weights: torch.Tensor,
        node_type: torch.Tensor,
        mach: torch.Tensor,
    ) -> torch.Tensor:
        if current_conservative.shape[1] != self.cell_volume.numel():
            raise ValueError(
                "current state cell count differs from fixed face geometry"
            )
        model_input = self.normalized_input(
            current_conservative,
            nodes=nodes,
            node_rhos=node_rhos,
            node_type=node_type,
            mach=mach,
        )
        latent = self.backbone(
            model_input,
            (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights),
        )
        latent = latent * node_mask
        normalized_state = (current_conservative - self.state_mean) / self.state_scale
        normalized_impulse = latent.new_zeros(
            (latent.shape[0], self.face_owner.numel(), NUM_EULER_COMPONENTS)
        )
        interior_impulse = self._interior_normalized_impulse(
            latent, normalized_state
        ).to(dtype=normalized_impulse.dtype)
        boundary_impulse = self._boundary_normalized_impulse(
            latent, normalized_state
        ).to(dtype=normalized_impulse.dtype)
        normalized_impulse[:, self.face_is_interior] = interior_impulse
        normalized_impulse[:, ~self.face_is_interior] = boundary_impulse
        return normalized_impulse * self.face_impulse_scale.unsqueeze(0)

    def forward_with_face_impulse(
        self,
        current_conservative: torch.Tensor,
        **kwargs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        impulse = self.predict_face_impulse(current_conservative, **kwargs)
        update = decode_owner_oriented_face_impulse_torch(
            impulse,
            cell_volume=self.cell_volume,
            face_owner=self.face_owner,
            face_neighbor=self.face_neighbor,
        )
        prediction = (current_conservative + update) * kwargs["node_mask"]
        return prediction, impulse

    def forward(
        self,
        current_conservative: torch.Tensor,
        *,
        node_mask: torch.Tensor,
        nodes: torch.Tensor,
        node_weights: torch.Tensor,
        node_rhos: torch.Tensor,
        directed_edges: torch.Tensor,
        edge_gradient_weights: torch.Tensor,
        node_type: torch.Tensor,
        mach: torch.Tensor,
    ) -> torch.Tensor:
        prediction, _ = self.forward_with_face_impulse(
            current_conservative,
            node_mask=node_mask,
            nodes=nodes,
            node_weights=node_weights,
            node_rhos=node_rhos,
            directed_edges=directed_edges,
            edge_gradient_weights=edge_gradient_weights,
            node_type=node_type,
            mach=mach,
        )
        return prediction


def winv_face_loss_and_relative_l2(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    face_weight: torch.Tensor | np.ndarray,
    face_mask: torch.Tensor | np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return component-balanced loss and global relative `W_f^-1` error."""

    predicted = prediction.float()
    expected = target.float()
    if predicted.shape != expected.shape or predicted.ndim != 3:
        raise ValueError("prediction and target must share shape [batch, faces, vars]")
    if predicted.shape[-1] != NUM_EULER_COMPONENTS:
        raise ValueError("face impulse must have four conservative components")
    weight = torch.as_tensor(
        face_weight,
        dtype=predicted.dtype,
        device=predicted.device,
    ).reshape(-1)
    if weight.shape != (predicted.shape[1],):
        raise ValueError("face_weight and face impulse axes differ")
    if not bool(torch.isfinite(weight).all()) or bool((weight <= 0.0).any()):
        raise ValueError("face_weight must be positive and finite")
    if face_mask is not None:
        mask = torch.as_tensor(
            face_mask,
            dtype=torch.bool,
            device=predicted.device,
        ).reshape(-1)
        if mask.shape != weight.shape or not bool(mask.any()):
            raise ValueError("face_mask must select at least one physical face")
        predicted = predicted[:, mask]
        expected = expected[:, mask]
        weight = weight[mask]
    if not bool(torch.isfinite(predicted).all()) or not bool(
        torch.isfinite(expected).all()
    ):
        raise ValueError("face predictions and targets must be finite")

    inv_weight = torch.reciprocal(weight).reshape(1, -1, 1)
    error_energy = torch.sum((predicted - expected).square() * inv_weight, dim=1)
    target_energy = torch.sum(expected.square() * inv_weight, dim=1)
    if bool((target_energy <= 0.0).any()) or not bool(
        torch.isfinite(target_energy).all()
    ):
        raise ValueError("every supervised face component needs positive target energy")
    component_relative_squared = error_energy / target_energy
    loss = torch.mean(component_relative_squared)
    relative_l2 = torch.sqrt(torch.sum(error_energy) / torch.sum(target_energy))
    return loss, relative_l2


def impulse_balance_residual(
    state_update: torch.Tensor,
    face_impulse: torch.Tensor,
    *,
    cell_volume: torch.Tensor,
    face_neighbor: torch.Tensor,
) -> torch.Tensor:
    """Return ``sum(V*dU) + sum(boundary impulse)`` per batch/component."""

    volume = torch.as_tensor(
        cell_volume, dtype=state_update.dtype, device=state_update.device
    ).reshape(1, -1, 1)
    if state_update.ndim != 3 or state_update.shape[-1] != NUM_EULER_COMPONENTS:
        raise ValueError("state_update must have shape [batch,cells,4]")
    if volume.shape[1] != state_update.shape[1]:
        raise ValueError("cell_volume and state_update cell axes differ")
    neighbor = torch.as_tensor(
        face_neighbor, dtype=torch.int64, device=face_impulse.device
    )
    if neighbor.shape != (face_impulse.shape[1],):
        raise ValueError("face_neighbor and face_impulse axes differ")
    boundary_exchange = face_impulse[:, neighbor < 0].sum(dim=1)
    return (volume * state_update).sum(dim=1) + boundary_exchange


def face_contract_summary(model: PCNOEuler2DSharedFaceImpulse) -> Mapping[str, Any]:
    """Return the fixed decoder contract without serializing tensor values."""

    return {
        "target_kind": model.target_kind,
        "geometry_digest": model.geometry_digest,
        "graph_geometry_digest": model.graph_geometry_digest,
        "num_cells": int(model.cell_volume.numel()),
        "num_faces": int(model.face_owner.numel()),
        "num_interior_faces": int(model.face_is_interior.sum()),
        "num_boundary_faces": int((~model.face_is_interior).sum()),
        "boundary_tag_names": list(model.boundary_tag_names),
        "interior_shared_face": True,
        "interior_orientation_antisymmetric": True,
        "decoded_state_conservative_up_to_predicted_boundary_exchange": True,
        "reference_impulse_supervision": model.reference_impulse_supervision,
        "supervision": model.supervision,
        "fixed_geometry_only": True,
        "fixed_delta_t": model.fixed_delta_t,
    }
