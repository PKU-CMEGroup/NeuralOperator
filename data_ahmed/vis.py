import os
import glob
import torch
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


dataset_folder = r"train"

ply_files = sorted(glob.glob(os.path.join(dataset_folder, "*.ply")))
pt_files = sorted(glob.glob(os.path.join(dataset_folder, "*.pt")))
npy_files = sorted(glob.glob(os.path.join(dataset_folder, "*.npy")))


if len(ply_files) != len(pt_files) or len(ply_files) != len(npy_files):
    raise ValueError("The number of files varies for each type!")
print(len(ply_files))

for i in range(3):
    ply_file = ply_files[i]
    pt_file = pt_files[i]
    npy_file = npy_files[i]

    info = torch.load(pt_file, weights_only=True)
    mesh = o3d.io.read_triangle_mesh(ply_file)
    pressure = np.load(npy_file)

    print(f"Mesh{i+1}: ", mesh, "Info:", info)

    num_triangles = len(mesh.triangles)
    if pressure.shape[0] != num_triangles:
        raise ValueError(f"The length of the pressure does not match the triangles!")

    pressure_min = np.min(pressure)
    pressure_max = np.max(pressure)
    normalized_pressure = (pressure - pressure_min) / (pressure_max - pressure_min)

    colors = plt.get_cmap("jet")(normalized_pressure)[:, :3]
    vertex_colors = np.zeros((len(mesh.vertices), 3))

    for idx, triangle in enumerate(mesh.triangles):
        pressure_value = colors[idx]
        for vertex_index in triangle:
            vertex_colors[vertex_index] = pressure_value[:3]

    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    o3d.visualization.draw_geometries(
        [mesh], window_name=f"Mesh with Triangle Pressure Visualization - {i + 1}"
    )
