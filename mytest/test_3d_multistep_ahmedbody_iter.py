import os
import glob
import random
import torch
import sys
import numpy as np
import math
from timeit import default_timer
import matplotlib.pyplot as plt
import open3d as o3d
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from pcno.geo_utility import preprocess_data, compute_node_weights
from pcno.pcno_normfix_res import compute_Fourier_modes, compute_Fourier_bases

def truncate_batch(nx, x, y, modes, sigma, k_max, L, batch_size=1000):
    """
    分批处理modes以节省内存
    nx: 法向量
    x: 节点坐标 (用于计算Fourier bases)
    y: 特征值 (需要变换的数据)
    """
    num_modes = ((2*k_max+1)**3 - 1) // 2
    
    # 初始化结果
    y_result = torch.zeros_like(y)
    
    # 分批处理
    for start_idx in range(0, num_modes, batch_size):
        end_idx = min(start_idx + batch_size, num_modes)
        
        # 当前批次的modes
        modes_batch = modes[start_idx:end_idx]
        print(f"Batch {start_idx//batch_size + 1}: modes {start_idx}-{end_idx}")
        
        # 计算当前批次的bases (基于节点坐标x)
        bases_c_batch, bases_s_batch, bases_0_batch = compute_Fourier_bases(x, modes_batch)
        
        # 应用权重
        wbases_c_batch = torch.einsum("bxkw,bxw->bxkw", bases_c_batch, node_weights)
        wbases_s_batch = torch.einsum("bxkw,bxw->bxkw", bases_s_batch, node_weights)
        wbases_0_batch = torch.einsum("bxkw,bxw->bxkw", bases_0_batch, node_weights)
        
        # 计算修正因子
        temp = torch.einsum("bxd,kdw->bxkw", nx, modes_batch)
        correction = pow(2*torch.pi*sigma**2, 1/2)*torch.exp(-sigma**2/2*temp**2)
        
        # 应用修正
        wbases_c_fix = wbases_c_batch * correction
        wbases_s_fix = wbases_s_batch * correction
        wbases_0_fix = wbases_0_batch * pow(2*torch.pi*sigma**2, 1/2)
        
        # 计算系数 (对特征值y进行变换)
        y_c_hat = torch.einsum("bix,bxkw->bikw", y, wbases_c_fix)
        y_s_hat = -torch.einsum("bix,bxkw->bikw", y, wbases_s_fix)
        y_0_hat = torch.einsum("bix,bxkw->bikw", y, wbases_0_fix)
        
        # 重建并累加到结果
        if start_idx == 0:  # 第一批包含0频率项
            y_batch = torch.einsum("bokw,bxkw->box", y_0_hat, bases_0_batch) + \
                     2*torch.einsum("bokw,bxkw->box", y_c_hat, bases_c_batch) - \
                     2*torch.einsum("bokw,bxkw->box", y_s_hat, bases_s_batch)
        else:  # 其他批次不包含0频率项
            y_batch = 2*torch.einsum("bokw,bxkw->box", y_c_hat, bases_c_batch) - \
                     2*torch.einsum("bokw,bxkw->box", y_s_hat, bases_s_batch)
        
        y_result += y_batch
        
        # 清理内存
        del bases_c_batch, bases_s_batch, bases_0_batch
        del wbases_c_batch, wbases_s_batch, wbases_0_batch
        del y_c_hat, y_s_hat, y_0_hat, y_batch
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return y_result

def direct_batch(x, y, modes, k_max, L, batch_size=1000):
    """
    分批处理的直接计算方法
    x: 节点坐标 (用于计算Fourier bases)
    y: 特征值 (需要变换的数据)
    """
    num_modes = ((2*k_max+1)**3 - 1) // 2
    
    # 初始化结果
    y_result = torch.zeros_like(y)
    
    # 分批处理
    for start_idx in range(0, num_modes, batch_size):
        end_idx = min(start_idx + batch_size, num_modes)
        
        # 当前批次的modes
        modes_batch = modes[start_idx:end_idx]
        print(f"Batch {start_idx//batch_size + 1}: modes {start_idx}-{end_idx}")
        # 计算当前批次的bases (基于节点坐标x)
        bases_c_batch, bases_s_batch, bases_0_batch = compute_Fourier_bases(x, modes_batch)
        
        # 应用权重
        wbases_c_batch = torch.einsum("bxkw,bxw->bxkw", bases_c_batch, node_weights)
        wbases_s_batch = torch.einsum("bxkw,bxw->bxkw", bases_s_batch, node_weights)
        wbases_0_batch = torch.einsum("bxkw,bxw->bxkw", bases_0_batch, node_weights)
        
        # 计算系数 (对特征值y进行变换)
        y_c_hat = torch.einsum("bix,bxkw->bikw", y, wbases_c_batch)
        y_s_hat = -torch.einsum("bix,bxkw->bikw", y, wbases_s_batch)
        y_0_hat = torch.einsum("bix,bxkw->bikw", y, wbases_0_batch)
        
        # 重建并累加到结果
        if start_idx == 0:  # 第一批包含0频率项
            y_batch = torch.einsum("bokw,bxkw->box", y_0_hat, bases_0_batch) + \
                     2*torch.einsum("bokw,bxkw->box", y_c_hat, bases_c_batch) - \
                     2*torch.einsum("bokw,bxkw->box", y_s_hat, bases_s_batch)
        else:  # 其他批次不包含0频率项
            y_batch = 2*torch.einsum("bokw,bxkw->box", y_c_hat, bases_c_batch) - \
                     2*torch.einsum("bokw,bxkw->box", y_s_hat, bases_s_batch)
        
        y_result += y_batch
        
        # 清理内存
        del bases_c_batch, bases_s_batch, bases_0_batch
        del wbases_c_batch, wbases_s_batch, wbases_0_batch
        del y_c_hat, y_s_hat, y_0_hat, y_batch
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return y_result

def normfix_onestep_batch(x, y, nx, sigma, k, L, batch_size=1000):
    """
    分批处理的单步normfix
    """
    # print(f"normfix_onestep_batch: y.shape = {y.shape}")
    y_normfix = truncate_batch(nx, x, y.transpose(2,1), modes, sigma, k, L, batch_size).transpose(2,1)
    
    # 归一化系数
    coffe = torch.norm(y[0,:,0])/torch.norm(y_normfix[0,:,0])
    y_normfix = y_normfix * coffe
    
    return y_normfix

def normfix_multistep_batch(x, y, nx, sigma_list, k_max_list, L, batch_size=1000):
    """
    分批处理的多步normfix
    """
    for i,(sigma,k) in enumerate(zip(sigma_list, k_max_list)):
        if i==0:
            y_normfix = normfix_onestep_batch(x, y, nx, sigma, k, L, batch_size)
        else:
            y_normfix_new = normfix_onestep_batch(x, y - y_normfix, nx, sigma, k, L, batch_size)
            y_normfix = y_normfix + y_normfix_new
    return y_normfix

# 主程序
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_path = "../data/ahmed_body/"
equal_weights = False
i = 1

nodes = np.load(data_path + "/nodes_%05d" % (i) + ".npy")
elems = np.load(data_path + "/elems_%05d" % (i) + ".npy")
features = np.load(data_path + "/features_%05d" % (i) + ".npy")

N_ref = nodes.shape[0]
ymin = np.amin(nodes[:, 1])
print('ymin: ',ymin)
nodes_reflect, elems_reflect , features_reflect = nodes.copy(), elems.copy(), features.copy()
nodes_reflect[:, 1] = 2*ymin - nodes_reflect[:, 1]
elems_reflect[:,1:] = elems_reflect[:,1:] + N_ref

nodes = np.concatenate([nodes, nodes_reflect], axis = 0)
elems = np.concatenate([elems, elems_reflect], axis = 0)
features = np.concatenate([features, features_reflect], axis = 0)

# Create Open3D TriangleMesh
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(nodes)
mesh.triangles = o3d.utility.Vector3iVector(elems[:,1:].astype(np.int32))
mesh.compute_vertex_normals()
normals = np.asarray(mesh.vertex_normals)  # shape (N, 3)

nodes_list = [nodes]
elems_list = [elems]
features_list = [np.concatenate([features, normals], axis=-1)]
nnodes, node_mask, nodes, node_measures_raw, features, directed_edges, edge_gradient_weights = preprocess_data(
    nodes_list, elems_list, features_list)
node_measures, node_weights = compute_node_weights(
    nnodes, node_measures_raw, equal_measure=False)
node_equal_measures, node_equal_weights = compute_node_weights(
    nnodes, node_measures_raw, equal_measure=True)

x = torch.from_numpy(nodes).to(device)  # 节点坐标
nx = torch.from_numpy(features[:, :, -3:]).to(device)  # 法向量
y = torch.from_numpy(features[:, :, 0:1]).to(device)  # 特征值
node_weights = torch.from_numpy(node_weights).to(device)
N = x.shape[1]  # number of nodes

print('\n',np.max(nodes, axis = 1), np.min(nodes, axis = 1))

# 现在可以使用更大的k_max
k_max = 50  # 增大k_max
batch_size = 500  # 批次大小，可以根据内存情况调整

sigma = 0.05
Lx = 0.0004795 - (-1.34399998)
Ly = 0.25450477 - 0
Lz = 0.43050185 - 0
Ly = Ly*2
L = [Lx, Ly, Lz]
print(f'sigma = {sigma}, k_max = {k_max}, L = {L}, batch_size = {batch_size}')

# 计算modes（一次性计算，只是存储索引）
modes = torch.tensor(compute_Fourier_modes(3, [k_max,k_max,k_max], L)).to(device)
print(f'Total modes: {modes.shape[0]}')

sigma_list = [0.1, 0.05]
k_max_list = [k_max, k_max]

print("Starting batch computations...")
print(f"x.shape = {x.shape}, y.shape = {y.shape}, nx.shape = {nx.shape}")

t1 = default_timer()
y_normfix_multi = normfix_multistep_batch(x, y, nx, sigma_list, k_max_list, L, batch_size)
t2 = default_timer()

y_normfix_single = normfix_onestep_batch(x, y, nx, sigma, k_max, L, batch_size)
t3 = default_timer()

y_direct = direct_batch(x, y.transpose(2,1), modes, k_max, L, batch_size).transpose(2,1)
t4 = default_timer()

# 归一化直接计算结果
coffe = torch.dot(y[0,:,0], y_direct[0,:,0])/torch.norm(y_direct[0,:,0])**2
y_direct = y_direct * coffe

print('loss(normfix multi): ', (torch.norm(y - y_normfix_multi)/torch.norm(y)).item())
print('loss(normfix single): ', (torch.norm(y - y_normfix_single)/torch.norm(y)).item())
print('loss(direct): ', (torch.norm(y - y_direct)/torch.norm(y)).item())
print('\n')
print('time(normfix multi): ', t2-t1)
print('time(normfix single): ', t3-t2)
print('time(direct): ', t4-t3)

# 移动到CPU进行可视化
x, y, y_normfix_single, y_normfix_multi, y_direct = x.to('cpu'), y.to('cpu'), y_normfix_single.to('cpu'), y_normfix_multi.to('cpu'), y_direct.to('cpu')

# Prepare color maps for each result
def get_vertex_colors(values):
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    values = values[0, :, 0]  # shape (N,)
    norm = plt.Normalize(vmin=np.min(values), vmax=np.max(values))
    cmap = plt.get_cmap("jet")
    colors = cmap(norm(values))[:, :3]  # shape (N, 3)
    return colors.astype(np.float64)

# Create a copy of the mesh for each result
meshes = []
titles = ['y_ref', 'y_normfix_single', 'y_normfix_multi', 'y_direct']
fields = [y, y_normfix_single, y_normfix_multi, y_direct]

for field, title in zip(fields, titles):
    mesh_copy = o3d.geometry.TriangleMesh(mesh)  # deep copy
    mesh_copy.vertex_colors = o3d.utility.Vector3dVector(get_vertex_colors(field[:, :N, :]))
    mesh_copy.compute_vertex_normals()
    mesh_copy = mesh_copy.translate((0, 0, 0))  # no translation, but can be used for arrangement
    meshes.append(mesh_copy)

# Optionally, arrange meshes in a row for easier comparison
for i, m in enumerate(meshes):
    m.translate((i * (L[0] + 0.2), 0, 0))  # shift each mesh along x-axis

# Visualize all meshes together with coordinate axes
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="3D Mesh Comparison", width=1600, height=600)
for m in meshes:
    vis.add_geometry(m)

# Add coordinate axes
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
vis.add_geometry(axis)

# Enable back face rendering for meshes
opt = vis.get_render_option()
opt.mesh_show_back_face = True

vis.run()
vis.destroy_window()