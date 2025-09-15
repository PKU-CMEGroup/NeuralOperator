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
from pcno.pcno_normfix_res import truncate, compute_Fourier_modes, compute_Fourier_bases, direct


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

# print(elems)

x = torch.from_numpy(nodes).to(device)
nx = torch.from_numpy(features[:, :, -3:]).to(device)
y = torch.from_numpy(features[:, :, 0:1]).to(device)
node_weights = torch.from_numpy(node_weights).to(device)
N = x.shape[1]  # number of nodes

# ymax = torch.amax(x[:, :, 1:2], dim = 1, keepdims=True)  # b,1,1
# print(ymax)
# x_reflect, nx_reflect, y_reflect, node_weight_reflect = x.clone(), nx.clone(), y.clone(), node_weights.clone()
# x_reflect[:, :, 1] = 2*ymax - x_reflect[:, :, 1]
# nx_reflect[:, :, 1] = -nx_reflect[:, :, 1]
# x = torch.cat([x, x_reflect], axis = 1).to(device)
# nx = torch.cat([nx, nx_reflect], axis = 1).to(device)
# y = torch.cat([y, y_reflect], axis = 1).to(device)
# node_weights = torch.cat([node_weights, node_weight_reflect], axis = 1).to(device)

print(np.max(nodes, axis = 1), np.min(nodes, axis = 1))
k_max = 4

sigma = 0.05
Lx = 0.0004795 - (-1.34399998)
Ly = 0.25450477 - 0
Lz = 0.43050185 - 0
# Lx = Lx*2
Ly = Ly*2
# Lz = Lz*2
L = [Lx, Ly, Lz]
print(f'sigma = {sigma}, k_max = {k_max}, L = {L}')



modes = torch.tensor(compute_Fourier_modes(3, [k_max,k_max,k_max], L)).to(device)  # nmodes, 2, 1
bases_c,  bases_s,  bases_0  = compute_Fourier_bases(x, modes)  # (1, n, nmodes, 1) , (1, n, nmodes, 1), (1, n, 1, 1)
wbases_c = torch.einsum("bxkw,bxw->bxkw", bases_c, node_weights)
wbases_s = torch.einsum("bxkw,bxw->bxkw", bases_s, node_weights)
wbases_0 = torch.einsum("bxkw,bxw->bxkw", bases_0, node_weights)


def normfix_onestep(x, y, nx, sigma, k, L):

    num_modes = ((2*k+1)**3- 1 )//2
    # t = default_timer()
    y_normfix = truncate(nx, y.transpose(2,1), wbases_c[:,:,:num_modes,:], wbases_s[:,:,:num_modes,:], wbases_0, 
                         bases_c[:,:,:num_modes,:], bases_s[:,:,:num_modes,:], bases_0, modes[:num_modes], sigma).transpose(2,1)
    # s = default_timer()
    # print(f'time for k_max = {k_max}: ',s-t)

    # coffe = torch.dot(y[0,:,0], y_normfix[0,:,0])/torch.norm(y_normfix[0,:,0])**2
    coffe = torch.norm(y[0,:,0])/torch.norm(y_normfix[0,:,0])
    y_normfix = y_normfix * coffe
    
    return y_normfix

def normfix_multistep(x, y, nx, sigma_list, k_max_list, L):
    for i,(sigma,k) in enumerate(zip(sigma_list, k_max_list)):
        if i==0:
            y_normfix = normfix_onestep(x, y, nx, sigma, k, L)
        else:
            y_normfix_new = normfix_onestep(x, y - y_normfix, nx, sigma, k, L)
            y_normfix = y_normfix + y_normfix_new
    return y_normfix

sigma_list = [0.1,0.05]
k_max_list = [4,4]

y_normfix_multi = normfix_multistep(x, y, nx, sigma_list, k_max_list, L)
# t1 = default_timer()
y_normfix_single = normfix_onestep(x, y, nx, sigma, k_max, L)
# t2 = default_timer()

# t3 = default_timer()

y_direct = direct(y.transpose(2,1), wbases_c, wbases_s, wbases_0, bases_c, bases_s, bases_0).transpose(2,1)

# t4 = default_timer()


# argmin_c ||y - c*y_out|| = <y, y_out>/||y_out||^2
# print(y.shape,y_out.shape)
# coffe1 = torch.dot(y[0,:,0], y_out1[0,:,0])/torch.norm(y_out1[0,:,0])**2
# y_out1 = y_out1 * coffe1

coffe = torch.dot(y[0,:,0], y_direct[0,:,0])/torch.norm(y_direct[0,:,0])**2
y_direct = y_direct * coffe

print('loss(normfix multi): ', (torch.norm(y - y_normfix_multi)/torch.norm(y)).item())
print('loss(normfix single): ', (torch.norm(y - y_normfix_single)/torch.norm(y)).item())
print('loss(direct): ', (torch.norm(y - y_direct)/torch.norm(y)).item())
# print('\n')
# print('time(normfix multi): ',t3-t2)
# print('time(direct): ',t4-t3)
# print('time(normfix single): ',t2-t1)




x, y, y_normfix_single, y_normfix_multi, y_direct = x.to('cpu'), y.to('cpu'), y_normfix_single.to('cpu'), y_normfix_multi.to('cpu') ,y_direct.to('cpu')
# fig, ax = plt.subplots(2, 4,figsize=(20,14), subplot_kw={'projection': '3d'})

# Prepare color maps for each result
def get_vertex_colors(values):
    # values: (1, N, 1) torch tensor or numpy array
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

# surf1 = ax[0,0].scatter(x[0,:,0], x[0,:,1], x[0,:,2], c=y[0,:,0], s=10)
# fig.colorbar(surf1 , ax=ax[0,0], shrink=0.5, aspect=10, pad=0.1)
# ax[0,0].view_init(elev=-50, azim=50)
# ax[0,0].set_title('y_ref', pad=30)

# surf2 = ax[0,1].scatter(x[0,:,0], x[0,:,1], x[0,:,2], c=y_normfix_single[0,:,0], s=10)
# fig.colorbar(surf2 , ax=ax[0,1], shrink=0.5, aspect=10, pad=0.1)
# ax[0,1].view_init(elev=-50, azim=50)
# ax[0,1].set_title('y_normfix_single', pad=30)

# surf3 = ax[0,2].scatter(x[0,:,0], x[0,:,1], x[0,:,2], c=y_normfix_multi[0,:,0], s=10)
# fig.colorbar(surf3 , ax=ax[0,2], shrink=0.5, aspect=10, pad=0.1)
# ax[0,2].view_init(elev=-50, azim=50)
# ax[0,2].set_title('y_normfix_multi', pad=30)

# surf4 = ax[0,3].scatter(x[0,:,0], x[0,:,1], x[0,:,2], c=y_direct[0,:,0], s=10)
# fig.colorbar(surf4 , ax=ax[0,3], shrink=0.5, aspect=10, pad=0.1)
# ax[0,3].view_init(elev=-50, azim=50)
# ax[0,3].set_title('y_direct', pad=30)



# surf21 = ax[1,1].scatter(x[0,:,0], x[0,:,1], x[0,:,2], c = torch.abs(y_normfix_single[0,:,0] - y[0,:,0]), s=10)
# fig.colorbar(surf21 , ax=ax[1,1], shrink=0.5, aspect=10, pad=0.1)
# ax[1,1].view_init(elev=-50, azim=50)
# ax[1,1].set_title('error_normfix_single', pad=30)

# surf31 = ax[1,2].scatter(x[0,:,0], x[0,:,1], x[0,:,2], c = torch.abs(y_normfix_multi[0,:,0] - y[0,:,0]), s=10)
# fig.colorbar(surf31 , ax=ax[1,2], shrink=0.5, aspect=10, pad=0.1)
# ax[1,2].view_init(elev=-50, azim=50)
# ax[1,2].set_title('error_normfix_multi', pad=30)

# surf41 = ax[1,3].scatter(x[0,:,0], x[0,:,1], x[0,:,2], c = torch.abs(y_direct[0,:,0] - y[0,:,0]), s=10)
# fig.colorbar(surf41 , ax=ax[1,3], shrink=0.5, aspect=10, pad=0.1)
# ax[1,3].view_init(elev=-50, azim=50)
# ax[1,3].set_title('error_direct', pad=30)
# plt.show()