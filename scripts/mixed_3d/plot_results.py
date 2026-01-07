import meshio
import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
plt.rcParams['font.family'] = 'Times New Roman'

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

folder = "/Users/huang/Desktop/Code/NeuralOperator/data/mixed_3d/mixed_3d_add_elem_features"
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