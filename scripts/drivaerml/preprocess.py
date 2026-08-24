"""
本文件处理 drivaerml 数据， 数据存在 ../../data/drivaerml/
我们关注车体表面数据 boundary_1.vtp, boundary_2.vtp, ......
每个数据存了 points, cells, 以及存在cell 上的数据包括
    CpMeanTrim              float32    (nc,)
    pMeanTrim               float32    (nc,)
    pPrime2MeanTrim         float32    (nc,)
    wallShearStressMeanTrim float32    (nc, 3)

处理后的数据存放在 ../../data/drivaerml/preprocess/
包括 node_data_1.npy,  网格信息 格点中心坐标，格子面积，和外法向量， nc by 7 numpy array 
    CpMeanTrim_1.npy, pMeanTrim_1.npy, pPrime2MeanTrim_1.npy,  wallShearStressMeanTrim_1.npy ......
    metadata_1.json, 包括数组描述、几何统计量和每个 y 通道的统计量
"""
import json
from pathlib import Path
from timeit import default_timer

import numpy as np
import pyvista as pv
from vtk.util.numpy_support import vtk_to_numpy

def _channel_statistics(
    values: np.ndarray, chunk_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """分块计算每个通道的 mean、M2、min 和 max，避免创建大型临时数组。"""
    values_2d = values[:, None] if values.ndim == 1 else values
    count = 0
    mean = np.zeros(values_2d.shape[1], dtype=np.float64)
    m2 = np.zeros_like(mean)
    minimum = np.full_like(mean, np.inf)
    maximum = np.full_like(mean, -np.inf)

    for start in range(0, values_2d.shape[0], chunk_size):
        chunk = np.asarray(values_2d[start : start + chunk_size], dtype=np.float64)
        if not np.isfinite(chunk).all():
            raise ValueError(f"Y data contains NaN/Inf in rows starting at {start}")
        chunk_count = chunk.shape[0]
        chunk_mean = chunk.mean(axis=0, dtype=np.float64)
        chunk_m2 = np.square(chunk - chunk_mean).sum(axis=0, dtype=np.float64)
        minimum = np.minimum(minimum, chunk.min(axis=0))
        maximum = np.maximum(maximum, chunk.max(axis=0))

        if count == 0:
            mean = chunk_mean
            m2 = chunk_m2
            count = chunk_count
            continue
        delta = chunk_mean - mean
        combined_count = count + chunk_count
        mean += delta * (chunk_count / combined_count)
        m2 += chunk_m2 + np.square(delta) * count * chunk_count / combined_count
        count = combined_count

    return mean, m2, minimum, maximum



def _preprocess_polygon_chunk(
    points: np.ndarray,
    offsets: np.ndarray,
    connectivity: np.ndarray,
    start: int,
    end: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute centers, areas, and normals for a range of ragged polygons."""
    
    output_dtype = np.float64
    cell_ids = np.arange(start, end, dtype=np.int64)
    polygon_sizes = offsets[cell_ids + 1] - offsets[cell_ids]
    centers = np.empty((end - start, 3), dtype=output_dtype)
    measures = np.empty((end - start, 1), dtype=output_dtype)
    normals = np.empty((end - start, 3), dtype=output_dtype)

    for polygon_size in np.unique(polygon_sizes):
        if polygon_size < 3:
            raise ValueError(f"Found a polygon with only {polygon_size} vertices")
        local_rows = np.flatnonzero(polygon_sizes == polygon_size)
        selected_cells = cell_ids[local_rows]
        connectivity_positions = offsets[selected_cells, None] + np.arange(polygon_size, dtype=np.int64)[None, :]
        vertex_ids = connectivity[connectivity_positions]
        vertices = np.asarray(points[vertex_ids], dtype=output_dtype)

        centers[local_rows] = vertices.mean(axis=1)

        # Vectorized form of compute_measure_per_elem_ for a batch of polygons
        # with the same vertex count.  Keeping this batched avoids millions of
        # Python calls during full DrivAerML preprocessing.
        edge_a = vertices[:, 1:-1, :] - vertices[:, :1, :]
        edge_b = vertices[:, 2:, :] - vertices[:, :1, :]
        area_vectors = 0.5 * np.cross(edge_a, edge_b).sum(axis=1)
        areas = np.linalg.norm(area_vectors, axis=1)
        polygon_normals = np.zeros_like(area_vectors, dtype=output_dtype)
        np.divide(
            area_vectors,
            areas[:, None],
            out=polygon_normals,
            where=areas[:, None] > 0,
        )
        normals[local_rows] = polygon_normals
        measures[local_rows, 0] = areas.astype(output_dtype, copy=False)

    return centers, measures, normals


def preprocess_vtp(
    source: str | Path,
    preprocess_root: str | Path,
    y_fields: list[str],
    chunk_size: int,
) -> dict:
    """Preprocess one large VTP into NumPy arrays and save them with np.save."""
    source = Path(source).expanduser().resolve()
    preprocess_root = Path(preprocess_root).expanduser().resolve()

    start_time = default_timer()
    print(f"Reading {source}", flush=True)
    mesh = pv.read(source)
    mesh.points = np.asarray(mesh.points, dtype=np.float64)

    print(
        f"Orienting polygon connectivity with PyVista for {source.name}",
        flush=True,
    )
    mesh.compute_normals(
        cell_normals=True,
        point_normals=False,
        split_vertices=False,
        flip_normals=False,
        consistent_normals=True,
        auto_orient_normals=True,
        non_manifold_traversal=False,
        inplace=True,
        progress_bar=True,
    )

    polygons = mesh.GetPolys()
    n_cells = int(polygons.GetNumberOfCells())
    if n_cells <= 0:
        raise ValueError(f"{source} contains no polygon cells")
    vtk_cell_normals = np.asarray(mesh.cell_data["Normals"], dtype=np.float64)

    points = np.asarray(mesh.points, dtype=np.float64)
    # 第 i 个多边形的顶点索引为 connectivity[offsets[i] : offsets[i+1]]
    offsets = vtk_to_numpy(polygons.GetOffsetsArray()).astype(np.int64, copy=False)
    connectivity = vtk_to_numpy(polygons.GetConnectivityArray()).astype(np.int64, copy=False)

    preprocess_root.mkdir(parents=True, exist_ok=True)
    case_id = (
        source.stem.removeprefix("boundary_")
        if source.stem.startswith("boundary_")
        else source.stem
    )
    node_data_path = preprocess_root / f"node_data_{case_id}.npy"
    metadata_path  = preprocess_root / f"metadata_{case_id}.json"
    metadata_path.unlink(missing_ok=True)

    # Columns: center_x, center_y, center_z, area, normal_x, normal_y, normal_z.
    node_data = np.empty((n_cells, 7), dtype=np.float64)

    y_metadata: dict[str, dict] = {}
    for y_field in y_fields:
        if y_field not in mesh.cell_data:
            raise KeyError(
                f"Cell field {y_field!r} is absent from {source.name}; "
                f"available fields: {list(mesh.cell_data.keys())}"
            )
        values = np.asarray(mesh.cell_data[y_field], dtype=np.float64)
        if values.ndim not in (1, 2) or values.shape[0] != n_cells:
            raise ValueError(
                f"Cell field {y_field!r} must have shape [n_cells] or "
                f"[n_cells, channels]; got {values.shape}"
            )
        feature_path = preprocess_root / f"{y_field}_{case_id}.npy"
        np.save(feature_path, values)
        y_mean, y_m2, y_min, y_max = _channel_statistics(values, chunk_size)
        y_metadata[y_field] = {
            "file": feature_path.name,
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "width": 1 if values.ndim == 1 else int(values.shape[1]),
            "mean": y_mean.tolist(),
            "m2": y_m2.tolist(),
            "min": y_min.tolist(),
            "max": y_max.tolist(),
        }

    total_measure = 0.0
    coordinate_min = np.full(3, np.inf, dtype=np.float64)
    coordinate_max = np.full(3, -np.inf, dtype=np.float64)
    min_cell_area = np.inf
    max_cell_area = -np.inf
    for start in range(0, n_cells, chunk_size):
        end = min(start + chunk_size, n_cells)
        centers, measures, _ = _preprocess_polygon_chunk(points, offsets, connectivity, start, end)
        if not (
            np.isfinite(centers).all()
            and np.isfinite(measures).all()
            and np.isfinite(vtk_cell_normals[start:end]).all()
        ):
            raise ValueError(
                f"Geometry contains NaN/Inf in cells [{start}, {end})"
            )
        node_data[start:end, :3] = centers
        node_data[start:end, 3:4] = measures
        node_data[start:end, 4:7] = vtk_cell_normals[start:end]

        total_measure += float(measures.sum(dtype=np.float64))
        coordinate_min = np.minimum(coordinate_min, centers.min(axis=0))
        coordinate_max = np.maximum(coordinate_max, centers.max(axis=0))
        min_cell_area = min(min_cell_area, float(measures.min()))
        max_cell_area = max(max_cell_area, float(measures.max()))

    if not np.isfinite(total_measure) or total_measure <= 0:
        raise ValueError(f"Invalid total cell area: {total_measure}")

    print(
        f"  cell area range: min={min_cell_area:.8e}, max={max_cell_area:.8e}",
        flush=True,
    )
    np.save(node_data_path, node_data)

    metadata = {
        "case_id": case_id,
        "source_file": source.name,
        "n_points": int(mesh.n_points),
        "n_cells": n_cells,
        "node_data": {
            "file": node_data_path.name,
            "shape": list(node_data.shape),
            "dtype": str(node_data.dtype),
            "columns": [
                "center_x",
                "center_y",
                "center_z",
                "area",
                "normal_x",
                "normal_y",
                "normal_z",
            ],
        },
        "geometry": {
            "total_measure": total_measure,
            "coordinate_min": coordinate_min.tolist(),
            "coordinate_max": coordinate_max.tolist(),
            "cell_area_min": min_cell_area,
            "cell_area_max": max_cell_area,
        },
        "y_fields": y_metadata,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    elapsed = default_timer() - start_time
    print(
        f"Saved {n_cells:,} cells from {source.name} to {preprocess_root} "
        f"in {elapsed:.2f}s",
        flush=True,
    )



    # print(
    #     "VTK normal error =",
    #     np.sqrt(normal_error_squared),
    #     normal_error_max,
    #     normal_error_min,
    # )
    # # Save a VTK PolyData file for visual inspection.  It retains the mesh
    # # geometry/connectivity, while its only point/cell data array is the
    # # outward cell normal computed above.
    # normal_vtp_path = preprocess_root / f"{source.stem}_normals.vtp"
    # mesh.point_data.clear()
    # mesh.cell_data.clear()
    # mesh.field_data.clear()
    # mesh.cell_data["Normals"] = node_data[:, 4:7]
    # mesh.cell_data["Areas"] = node_data[:, 3]
    # mesh.cell_data.active_normals_name = "Normals"
    # mesh.save(normal_vtp_path, binary=True)
    # print(f"Saved cell-normal VTP to {normal_vtp_path}", flush=True)

    return metadata




def preprocess_data(
    chunk_size: int = 200_000,
    y_fields: list[str] | None = None,
) -> None:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if y_fields is None:
        y_fields = [
            "CpMeanTrim",
            "pMeanTrim",
            "wallShearStressMeanTrim",
        ]

    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data" / "drivaerml"
    preprocess_root = data_dir / "preprocess"
    files = sorted(data_dir.glob("boundary_*.vtp"))
    if not files:
        raise FileNotFoundError(f"No boundary_*.vtp files found in {data_dir}")

    for source in files:
        preprocess_vtp(source, preprocess_root, y_fields, chunk_size)


if __name__ == "__main__":
    preprocess_data()
