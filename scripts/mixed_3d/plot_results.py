import meshio
import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
plt.rcParams['font.family'] = 'Times New Roman'
from pcno_geo_mixed_3d_test import random_shuffle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utility.normalizer import UnitGaussianNormalizer
from utility.losses import LpLoss
from pcno.pcno import compute_Fourier_modes, PCNO
FONTSIZE = 17
def get_median_index(arr):
    # 确保输入是一个 NumPy 数组
    arr = np.asarray(arr)
    # 获取排序后的索引
    sorted_indices = np.argsort(arr)
    # 计算中位数的索引
    mid_index = len(arr) // 2
    
    if len(arr) % 2 == 1:
        # 如果是奇数长度，返回中间元素的原始索引
        median_index = sorted_indices[mid_index]
    else:
        # 如果是偶数长度，返回中间两个元素的原始索引
        median_index_1 = sorted_indices[mid_index - 1]
        median_index_2 = sorted_indices[mid_index]
        # 通常我们不会为偶数长度的数组返回单个索引，因为中位数是两个值的平均。
        # 但是，如果你需要，你可以选择返回这两个索引或仅其中一个。
        # 这里我们简单地返回一个元组
        median_index = median_index_1
    
    return median_index




def plot_results(vertices, elems, vertex_data, elem_data, file_name):
    
    """
    Save mesh data to a VTK file using meshio.
    
    Parameters:
    -----------
    vertices : numpy array (n_vertices, 3)
        Vertex coordinates
    elems : numpy array (n_elements, 4)
        Element connectivity (assuming triangle elements with 3 nodes + 1 element dimension)
    vertex_data : dict
        Data defined at vertices (e.g., {"displacement": vertex_displacements})
    elem_data : dict
        Data defined at elements (e.g., {"stress": element_stresses})
    file_name : str
        Output file name without extension
    """
 
    cells = []
    cells.append(("triangle", elems[:,1:]))

    # Create the mesh
    mesh = meshio.Mesh(
        points=vertices,
        cells=cells,
        point_data=vertex_data,
        cell_data=elem_data,
    )

    # Save to an Exodus II file
    meshio.write(file_name+ ".vtk", mesh)

# visualize raw data
folder = "../../data/mixed_3d_add_elem_features"
category = "Plane"
subcategory = "boeing737"
data_id = 10
data_file = folder + "/" + category + "/" + subcategory + "/" + str(data_id).zfill(4) + ".npz"
data = np.load(data_file)
vertices = data["nodes_list"]
elems = data["elems_list"]

print("number of vertices : ", vertices.shape[0])
print("number of elems : ", elems.shape[0])

vertex_data = {"vertex_Cp":  data["features_list"][:,0]}
elem_data   = {"element_Cp": [data["elem_features_list"][:,0]]}
file_name = category + subcategory + str(data_id).zfill(4)
plot_results(vertices, elems, vertex_data, elem_data, file_name)



mesh_type = "vertex_centered"
# visualize postprocessed data
data = np.load(folder+"/pcno_mixed_3d_"+mesh_type+".npz")
names_array = np.load(folder+"/pcno_mixed_3d_names_list.npy", allow_pickle=True)
# random shuffle, and keep only n_train + n_test data
print("ndata = ", data["nodes"].shape[0])
assert(data["nodes"].shape[0] == names_array.shape[0])
n_train, n_test = 10, 10
data, names_array = random_shuffle(data, names_array, n_train, n_test, seed=42)
data_id = 14 # from n_train + n_test data
nnodes, node_mask, nodes, features = data["nnodes"][data_id], data["node_mask"][data_id], data["nodes"][data_id], data["features"][data_id]
print("nnodes = ", nnodes, " total nnodes = ", node_mask.shape[0], flush = True)

raw_data_category, raw_data_subcategory, raw_data_id = names_array[data_id].split('-')
raw_data_id = int(raw_data_id)
print("Visualize ", raw_data_category, " ", raw_data_subcategory, " ", raw_data_id)
raw_data_file = folder + "/" + raw_data_category + "/" + raw_data_subcategory + "/" + str(raw_data_id).zfill(4) + ".npz"
raw_data = np.load(raw_data_file)
elems = raw_data["elems_list"]
vertices = raw_data["nodes_list"]

print("number of vertices : ", vertices.shape[0])
print("number of elems : ", elems.shape[0])

if mesh_type == "cell_centered":
    vertex_data = {"vertex_Cp":  raw_data["features_list"][:,0]}
    elem_data   = {"element_Cp": [raw_data["elem_features_list"][:,0]], "post_element_Cp": [features[0:nnodes,3]], "post_element_normal": [features[0:nnodes,0:3]]}
    
elif mesh_type == "vertex_centered":
    vertex_data = {"vertex_Cp":  raw_data["features_list"][:,0], "post_vertex_Cp": features[0:nnodes,3], "post_vertex_normal": features[0:nnodes,0:3]}
    elem_data   = {"element_Cp": [raw_data["elem_features_list"][:,0]]}
    
else:
    raise NotImplementedError(
                    f"Unsupported mesh_type={mesh_type}"
                ) 
    
file_name = "post_" + mesh_type + "_" + raw_data_category + raw_data_subcategory + str(raw_data_id).zfill(4)
plot_results(vertices, elems, vertex_data, elem_data, file_name)