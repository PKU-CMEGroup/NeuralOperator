import os

import sys
import numpy as np

from timeit import default_timer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
np.random.seed(0)

from tqdm import tqdm
data_path = "../../data/car_shapenet/car_shapenet_original"


def load_data(data_path):
    ndata = 611
    nodes_list, elems_list, features_list = [], [], []
    for i in range(ndata):    
        nodes_list.append(np.load(data_path+"/nodes_%05d"%(i)+".npy"))
        elems_list.append(np.load(data_path+"/elems_%05d"%(i)+".npy"))
        features_list.append(np.load(data_path+"/features_%05d"%(i)+".npy"))
    return nodes_list, elems_list, features_list 

nodes_list, elems_list, features_list  = load_data(data_path = data_path)


def compute_normals_with_open3d(nodes, elems):
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError("open3d is not installed. Please install it to compute normals.")
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(nodes)
    mesh.triangles = o3d.utility.Vector3iVector(elems)
    mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals)
    return normals

# 提供可视化
def visualize_mesh_with_normals(nodes, elems, normals):
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError("open3d is not installed. Please install it to compute normals.")
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(nodes)
    mesh.triangles = o3d.utility.Vector3iVector(elems)
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
    
    # 创建法线箭头
    line_set = o3d.geometry.LineSet()
    points = []
    lines = []
    colors = []
    for i, (vertex, normal) in enumerate(zip(nodes, normals)):
        points.append(vertex)
        points.append(vertex + normal * 0.1)  # 箭头长度
        lines.append([2*i, 2*i+1])
        colors.append([1, 0, 0])  # 红色箭头
    
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    o3d.visualization.draw_geometries([mesh, line_set])

normals_list = []
for i in tqdm((range(len(nodes_list)))):
    nodes = nodes_list[i]
    elems = elems_list[i][...,1:]
    features = features_list[i]
    
    normals = compute_normals_with_open3d(nodes, elems)
    
    # visualize_mesh_with_normals(nodes, elems, normals)

    normals_list.append(normals)


normals_array = np.array(normals_list)
print("normals_array shape:", normals_array.shape)
np.savez_compressed(data_path+"/car_shapenet_normals.npz", normals=normals_array)


    