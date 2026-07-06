from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv


@dataclass(frozen=True)
class TargetSpec:
    location: str
    name: str


DATASET_TARGETS = {
    "AirCraft": TargetSpec("auto", "Cp"),
    "BlendedNet": TargetSpec("auto", "Cp"),
    "NACA-CRM": TargetSpec("auto", "Cp"),
    "DrivAerNet++": TargetSpec("auto", "Cp"),
    "DrivAerML": TargetSpec("auto", "Cp"),
    "DrivAerStar": TargetSpec("auto", "Cp"),
}


def parse_dataset_list(value: str) -> list[str]:
    datasets = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [name for name in datasets if name not in DATASET_TARGETS]
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}. Known: {sorted(DATASET_TARGETS)}")
    return datasets


def list_dataset_files(data_root: Path, dataset: str) -> list[Path]:
    candidate_dirs = [
        data_root / f"{dataset}_20000",
        data_root / dataset / "decimated_mesh_20000",
    ]
    mesh_dir = next((path for path in candidate_dirs if path.is_dir()), candidate_dirs[0])
    files = sorted(mesh_dir.glob("*.vtp"))
    if not files:
        candidates = ", ".join(str(path) for path in candidate_dirs)
        raise FileNotFoundError(f"No .vtp files found. Checked: {candidates}")
    return files


def _triangle_elems(mesh: pv.PolyData) -> np.ndarray:
    if mesh.faces.size == 0:
        raise ValueError("VTP mesh has no polygon faces")
    faces = mesh.faces.reshape((-1, 4))
    if not np.all(faces[:, 0] == 3):
        mesh = mesh.triangulate()
        faces = mesh.faces.reshape((-1, 4))
        if not np.all(faces[:, 0] == 3):
            raise ValueError("Failed to convert all faces to triangles")
    elems = np.empty((faces.shape[0], 4), dtype=np.int64)
    elems[:, 0] = 2
    elems[:, 1:] = faces[:, 1:4]
    return elems


def _normalize_points(points: np.ndarray) -> np.ndarray:
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    center = 0.5 * (lower + upper)
    scale = 0.5 * float(np.max(upper - lower))
    if scale <= 0:
        scale = 1.0
    return (points - center) / scale


def _cell_normals(points: np.ndarray, elems: np.ndarray) -> np.ndarray:
    tri = elems[:, 1:4]
    v0 = points[tri[:, 0]]
    v1 = points[tri[:, 1]]
    v2 = points[tri[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    bad = norm[:, 0] <= 1.0e-12
    norm[bad] = 1.0
    normals = normals / norm
    normals[bad] = 0.0
    return normals.astype(np.float32)


def _vertex_normals(points: np.ndarray, elems: np.ndarray) -> np.ndarray:
    tri = elems[:, 1:4]
    v0 = points[tri[:, 0]]
    v1 = points[tri[:, 1]]
    v2 = points[tri[:, 2]]
    area_normals = np.cross(v1 - v0, v2 - v0)
    normals = np.zeros_like(points, dtype=np.float64)
    for local_idx in range(3):
        np.add.at(normals, tri[:, local_idx], area_normals)
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    bad = norm[:, 0] <= 1.0e-12
    norm[bad] = 1.0
    normals = normals / norm
    normals[bad] = 0.0
    return normals.astype(np.float32)


def _cell_targets(mesh: pv.PolyData, elems: np.ndarray, spec: TargetSpec) -> np.ndarray:
    if spec.location in {"cell", "auto"} and spec.name in mesh.cell_data:
        target = np.asarray(mesh.cell_data[spec.name])
    elif spec.location in {"point", "auto"} and spec.name in mesh.point_data:
        point_target = np.asarray(mesh.point_data[spec.name])
        target = point_target[elems[:, 1:4]].mean(axis=1)
    else:
        raise KeyError(f"Missing {spec.name} in point_data/cell_data")

    target = np.asarray(target).reshape((elems.shape[0], -1))
    if target.shape[1] != 1:
        target = target[:, :1]
    return target.astype(np.float32)


def _point_targets(mesh: pv.PolyData, spec: TargetSpec) -> np.ndarray:
    if spec.location in {"point", "auto"} and spec.name in mesh.point_data:
        target = np.asarray(mesh.point_data[spec.name]).reshape((mesh.n_points, -1))
    elif spec.location in {"cell", "auto"} and spec.name in mesh.cell_data:
        converted = mesh.cell_data_to_point_data(pass_cell_data=True)
        if spec.name not in converted.point_data:
            raise KeyError(f"Missing point field {spec.name}")
        target = np.asarray(converted.point_data[spec.name]).reshape((mesh.n_points, -1))
    else:
        raise KeyError(f"Missing {spec.name} in point_data/cell_data")
    if target.shape[1] != 1:
        target = target[:, :1]
    return target.astype(np.float32)


def load_vtp_as_cell_sample(path: Path, spec: TargetSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mesh = pv.read(path)
    elems = _triangle_elems(mesh)
    points = _normalize_points(np.asarray(mesh.points, dtype=np.float32))
    normals = _cell_normals(points, elems)
    target = _cell_targets(mesh, elems, spec)
    if not np.isfinite(target).all():
        raise ValueError(f"Non-finite target values in {path}")
    features = np.concatenate([normals, target], axis=1)
    return points.astype(np.float32), elems, features


def load_vtp_as_vertex_sample(path: Path, spec: TargetSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mesh = pv.read(path)
    elems = _triangle_elems(mesh)
    points = _normalize_points(np.asarray(mesh.points, dtype=np.float32))
    normals = _vertex_normals(points, elems)
    target = _point_targets(mesh, spec)
    if not np.isfinite(target).all():
        raise ValueError(f"Non-finite target values in {path}")
    features = np.concatenate([normals, target], axis=1)
    return points.astype(np.float32), elems, features


def load_hifi3d_cell_data(
    data_root: Path,
    datasets: list[str],
    n_each: int | None,
    seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[str]]:
    rng = np.random.default_rng(seed)
    vertices_list: list[np.ndarray] = []
    elems_list: list[np.ndarray] = []
    features_list: list[np.ndarray] = []
    names: list[str] = []

    for dataset in datasets:
        files = list_dataset_files(data_root, dataset)
        if n_each is not None and n_each > 0:
            selected = files[: min(n_each, len(files))]
        else:
            selected = files
        for path in selected:
            vertices, elems, features = load_vtp_as_cell_sample(path, DATASET_TARGETS[dataset])
            vertices_list.append(vertices)
            elems_list.append(elems)
            features_list.append(features)
            names.append(f"{dataset}-{path.stem}")

    order = np.arange(len(names))
    rng.shuffle(order)
    vertices_list = [vertices_list[i] for i in order]
    elems_list = [elems_list[i] for i in order]
    features_list = [features_list[i] for i in order]
    names = [names[i] for i in order]
    return vertices_list, elems_list, features_list, names


def load_hifi3d_data(
    data_root: Path,
    datasets: list[str],
    n_each: int | None,
    seed: int,
    mesh_type: str,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[str]]:
    if mesh_type == "cell_centered":
        loader = load_vtp_as_cell_sample
    elif mesh_type == "vertex_centered":
        loader = load_vtp_as_vertex_sample
    else:
        raise ValueError(f"Unsupported mesh_type: {mesh_type}")

    rng = np.random.default_rng(seed)
    vertices_list: list[np.ndarray] = []
    elems_list: list[np.ndarray] = []
    features_list: list[np.ndarray] = []
    names: list[str] = []

    for dataset in datasets:
        files = list_dataset_files(data_root, dataset)
        if n_each is not None and n_each > 0:
            selected = files[: min(n_each, len(files))]
        else:
            selected = files
        for path in selected:
            vertices, elems, features = loader(path, DATASET_TARGETS[dataset])
            vertices_list.append(vertices)
            elems_list.append(elems)
            features_list.append(features)
            names.append(f"{dataset}-{path.stem}")

    order = np.arange(len(names))
    rng.shuffle(order)
    vertices_list = [vertices_list[i] for i in order]
    elems_list = [elems_list[i] for i in order]
    features_list = [features_list[i] for i in order]
    names = [names[i] for i in order]
    return vertices_list, elems_list, features_list, names
