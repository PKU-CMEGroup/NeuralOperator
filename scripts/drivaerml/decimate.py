#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

try:
    import pymeshlab
except ImportError as exc:
    raise SystemExit("Error: pymeshlab is required.") from exc

try:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
except ImportError as exc:
    raise SystemExit("Error: vtk is required.") from exc


DEFAULT_CP_ALIASES = ["Cp", "cp", "PressureCoefficient", "CpMeanTrim"]
OUTPUT_CP_NAME = "Cp"


def read_vtp_polydata(path: str):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    polydata = reader.GetOutput()
    if polydata is None or polydata.GetNumberOfPoints() == 0:
        raise ValueError(f"VTP file has no valid points: {path}")
    return polydata


def write_vtp_polydata(polydata, path: str) -> None:
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(path)
    writer.SetInputData(polydata)
    writer.Write()


def find_cp_point_values(polydata, cp_aliases: list[str] | None = None) -> tuple[np.ndarray, str]:
    aliases = list(cp_aliases or DEFAULT_CP_ALIASES)

    point_data = polydata.GetPointData()
    for name in aliases:
        array = point_data.GetArray(name)
        if array is not None:
            values = vtk_to_numpy(array).reshape((-1,))
            return values.astype(np.float64), f"point:{name}"

    cell_data = polydata.GetCellData()
    for name in aliases:
        array = cell_data.GetArray(name)
        if array is None:
            continue
        converter = vtk.vtkCellDataToPointData()
        converter.SetInputData(polydata)
        converter.PassCellDataOn()
        converter.Update()
        converted = converter.GetOutput()
        converted_array = converted.GetPointData().GetArray(name)
        if converted_array is not None:
            values = vtk_to_numpy(converted_array).reshape((-1,))
            return values.astype(np.float64), f"cell:{name}"

    checked = ", ".join(aliases)
    raise ValueError(f"No Cp array found. Checked: {checked}")


def convert_vtp_to_ply(vtp_file: str, ply_file: str) -> None:
    polydata = read_vtp_polydata(vtp_file)
    triangle_filter = vtk.vtkTriangleFilter()
    triangle_filter.SetInputData(polydata)
    triangle_filter.Update()
    triangulated = triangle_filter.GetOutput()

    writer = vtk.vtkPLYWriter()
    writer.SetFileName(ply_file)
    writer.SetInputData(triangulated)
    writer.Write()


def simplify_mesh(input_ply_file: str, output_ply_file: str, target_vertices: int) -> tuple[int, int]:
    ms = pymeshlab.MeshSet()
    print("     loading...")
    ms.load_new_mesh(input_ply_file)
    original_vertices = ms.current_mesh().vertex_number()
    original_faces = ms.current_mesh().face_number()

    if target_vertices <= 0:
        raise ValueError("target_vertices must be positive")

    if target_vertices < original_vertices:
        targetperc = target_vertices / original_vertices
        print("     apply filter 0")
        ms.apply_filter(
            "meshing_decimation_quadric_edge_collapse",
            targetperc=targetperc,
            preservenormal=True,
            preserveboundary=True,
            preservetopology=True,
            planarquadric=True,
            optimalplacement=False,
        )
    print("     apply filter 1")
    ms.apply_filter("meshing_remove_duplicate_faces")
    print("     apply filter 2")
    ms.apply_filter("meshing_remove_duplicate_vertices")
    print("     apply filter 3")
    ms.apply_filter("meshing_remove_unreferenced_vertices")
    print("     apply filter 4")
    ms.apply_filter("meshing_remove_null_faces")
    print("     save")
    ms.save_current_mesh(output_ply_file)
    return ms.current_mesh().vertex_number(), ms.current_mesh().face_number()


def transfer_cp_vtp_to_ply(
    input_vtp_file: str,
    ply_mesh_file: str,
    output_vtp_file: str,
    cp_aliases: list[str] | None = None,
    output_cp_name: str = OUTPUT_CP_NAME,
) -> None:
    original_mesh = read_vtp_polydata(input_vtp_file)
    orig_cp_values, cp_source = find_cp_point_values(original_mesh, cp_aliases=cp_aliases)

    orig_points = original_mesh.GetPoints()
    num_orig_points = orig_points.GetNumberOfPoints()
    orig_coords = np.array([orig_points.GetPoint(i) for i in range(num_orig_points)], dtype=np.float64)

    ply_reader = vtk.vtkPLYReader()
    ply_reader.SetFileName(ply_mesh_file)
    ply_reader.Update()
    ply_mesh = ply_reader.GetOutput()

    ply_points = ply_mesh.GetPoints()
    num_ply_points = ply_points.GetNumberOfPoints()
    ply_coords = np.array([ply_points.GetPoint(i) for i in range(num_ply_points)], dtype=np.float64)

    tree = cKDTree(orig_coords)
    distances, indices = tree.query(ply_coords, k=1)
    mapped_cp = orig_cp_values[indices].astype(np.float32)

    cp_point_array = numpy_to_vtk(mapped_cp, deep=True)
    cp_point_array.SetName(output_cp_name)
    ply_mesh.GetPointData().SetScalars(cp_point_array)

    point_to_cell = vtk.vtkPointDataToCellData()
    point_to_cell.SetInputData(ply_mesh)
    point_to_cell.PassPointDataOn()
    point_to_cell.Update()
    output_mesh = point_to_cell.GetOutput()

    normals_filter = vtk.vtkPolyDataNormals()
    normals_filter.SetInputData(output_mesh)
    normals_filter.SetConsistency(True)
    normals_filter.SetAutoOrientNormals(True)
    normals_filter.SetNonManifoldTraversal(True)
    normals_filter.SetSplitting(False)
    normals_filter.SetFlipNormals(False)
    normals_filter.Update()
    final_mesh = normals_filter.GetOutput()

    write_vtp_polydata(final_mesh, output_vtp_file)

    print(f"Cp source: {cp_source}")
    print(f"Output Cp name: {output_cp_name}")
    print(f"Original points: {num_orig_points}")
    print(f"Decimated points: {num_ply_points}")
    print(f"Nearest-neighbor distance max: {distances.max():.6f}")
    print(f"Nearest-neighbor distance mean: {distances.mean():.6f}")


def preprocess_single(
    input_vtp_file: str,
    output_vtp_file: str,
    target_vertices: int,
    cp_aliases: list[str] | None = None,
    input_ply_dir: str | None = None,
    output_ply_dir: str | None = None,
) -> None:
    input_path = Path(input_vtp_file)
    output_path = Path(output_vtp_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if input_ply_dir is None:
        input_ply_path = input_path.with_suffix(".ply")
    else:
        input_ply_root = Path(input_ply_dir)
        input_ply_root.mkdir(parents=True, exist_ok=True)
        input_ply_path = input_ply_root / f"{input_path.stem}.ply"

    if output_ply_dir is None:
        output_ply_path = output_path.with_suffix(".ply")
    else:
        output_ply_root = Path(output_ply_dir)
        output_ply_root.mkdir(parents=True, exist_ok=True)
        output_ply_path = output_ply_root / f"{output_path.stem}.ply"

    print("Start converting vtp to ply")
    print(f"  input ply: {input_ply_path}")
    print(f"  output ply: {output_ply_path}")
    convert_vtp_to_ply(str(input_path), str(input_ply_path))
    print("Starting Simplifing ply mesh")
    final_vertices, final_faces = simplify_mesh(str(input_ply_path), str(output_ply_path), target_vertices)
    print(f"  target vertices: {target_vertices}")
    print(f"  simplified mesh: vertices={final_vertices} faces={final_faces}")
    print("Starting transfering cp to ply")
    transfer_cp_vtp_to_ply(
        str(input_path),
        str(output_ply_path),
        str(output_path),
        cp_aliases=cp_aliases,
        output_cp_name=OUTPUT_CP_NAME,
    )
    print("Finished")


def preprocess_folder(
    input_vtp_folder: str,
    input_ply_folder: str,
    output_ply_folder: str,
    output_vtp_folder: str,
    target_vertices: int,
    cp_aliases: list[str] | None = None,
) -> None:
    os.makedirs(input_ply_folder, exist_ok=True)
    os.makedirs(output_ply_folder, exist_ok=True)
    os.makedirs(output_vtp_folder, exist_ok=True)

    for dirpath, _, filenames in os.walk(input_vtp_folder):
        for filename in filenames:
            if not filename.endswith(".vtp"):
                continue

            base = os.path.splitext(filename)[0]
            input_vtp_file = os.path.join(dirpath, filename)
            input_ply_file = os.path.join(input_ply_folder, base + ".ply")
            output_ply_file = os.path.join(output_ply_folder, base + ".ply")
            output_vtp_file = os.path.join(output_vtp_folder, base + ".vtp")

            convert_vtp_to_ply(input_vtp_file, input_ply_file)
            simplify_mesh(input_ply_file, output_ply_file, target_vertices)
            transfer_cp_vtp_to_ply(
                input_vtp_file,
                output_ply_file,
                output_vtp_file,
                cp_aliases=cp_aliases,
                output_cp_name=OUTPUT_CP_NAME,
            )


def main() -> None:
    argc = len(sys.argv)

    if argc in {4, 5}:
        input_vtp_file = sys.argv[1]
        output_vtp_file = sys.argv[2]
        target_vertices = int(sys.argv[3])
        cp_aliases = [sys.argv[4]] if argc == 5 else None
        preprocess_single(input_vtp_file, output_vtp_file, target_vertices, cp_aliases=cp_aliases)
        return

    if argc in {6, 7} and str(sys.argv[1]).lower().endswith(".vtp"):
        input_vtp_file = sys.argv[1]
        output_vtp_file = sys.argv[2]
        target_vertices = int(sys.argv[3])
        cp_aliases = [sys.argv[4]] if argc >= 5 else None
        input_ply_dir = sys.argv[5] if argc >= 6 else None
        output_ply_dir = sys.argv[6] if argc == 7 else None
        preprocess_single(
            input_vtp_file,
            output_vtp_file,
            target_vertices,
            cp_aliases=cp_aliases,
            input_ply_dir=input_ply_dir,
            output_ply_dir=output_ply_dir,
        )
        return

    if argc in {6, 7}:
        input_vtp_folder = sys.argv[1]
        input_ply_folder = sys.argv[2]
        output_ply_folder = sys.argv[3]
        output_vtp_folder = sys.argv[4]
        target_vertices = int(sys.argv[5])
        cp_aliases = [sys.argv[6]] if argc == 7 else None
        preprocess_folder(
            input_vtp_folder,
            input_ply_folder,
            output_ply_folder,
            output_vtp_folder,
            target_vertices,
            cp_aliases=cp_aliases,
        )
        return

    print("Single file usage:")
    print("  python scripts/drivaerml/decimate.py input.vtp output.vtp target_vertices [source_cp_name]")
    print("  python scripts/drivaerml/decimate.py input.vtp output.vtp target_vertices source_cp_name input_ply_dir [output_ply_dir]")
    print("Folder usage:")
    print("  python scripts/drivaerml/decimate.py input_vtp_dir input_ply_dir output_ply_dir output_vtp_dir target_vertices [source_cp_name]")
    print("Output field name is always written as 'Cp'.")
    sys.exit(1)


if __name__ == "__main__":
    main()
