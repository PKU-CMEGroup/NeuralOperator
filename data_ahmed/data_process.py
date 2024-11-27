import os
import glob
import torch
import open3d as o3d
import numpy as np
from pre import ahmed_data


elem_dim = 2

nodes_list = []
elems_list = []
elemfeats_list = []
infos_list = []

dataset_folder = "train"
ply_files = sorted(glob.glob(os.path.join(dataset_folder, "*.ply")))
pt_files = sorted(glob.glob(os.path.join(dataset_folder, "*.pt")))
npy_files = sorted(glob.glob(os.path.join(dataset_folder, "*.npy")))

if len(ply_files) != len(pt_files) or len(ply_files) != len(npy_files):
    raise ValueError("The number of files varies for each type!")
n_train = len(ply_files)
print(n_train)


max_nnodes = 0
for i in range(n_train):

    mesh = o3d.io.read_triangle_mesh(ply_files[i])

    vertices = np.asarray(mesh.vertices)
    max_nnodes = max(max_nnodes, vertices.shape[0])
    nodes_list.append(vertices)

    elem = np.asarray(mesh.triangles)
    elems_list.append(np.concatenate(
        (np.full((elem.shape[0], 1), elem_dim, dtype=int), elem), axis=1))

    elemfeats_list.append(np.load(npy_files[i])[:, np.newaxis])
    infos_list.append(torch.load(pt_files[i], weights_only=True))


dataset_folder = "test"
ply_files = sorted(glob.glob(os.path.join(dataset_folder, "*.ply")))
pt_files = sorted(glob.glob(os.path.join(dataset_folder, "*.pt")))
npy_files = sorted(glob.glob(os.path.join(dataset_folder, "*.npy")))

if len(ply_files) != len(pt_files) or len(ply_files) != len(npy_files):
    raise ValueError("The number of files varies for each type!")
n_test = len(ply_files)
print(n_test)

for i in range(n_test):

    mesh = o3d.io.read_triangle_mesh(ply_files[i])

    vertices = np.asarray(mesh.vertices)
    max_nnodes = max(max_nnodes, vertices.shape[0])
    nodes_list.append(vertices)

    elem = np.asarray(mesh.triangles)
    elems_list.append(np.concatenate(
        (np.full((elem.shape[0], 1), elem_dim, dtype=int), elem), axis=1))

    elemfeats_list.append(np.load(npy_files[i])[:, np.newaxis])
    infos_list.append(torch.load(pt_files[i], weights_only=True))

print(nodes_list[0].shape)
print(elems_list[0].shape)
print(elemfeats_list[0].shape)
print(infos_list[0])

print(nodes_list[-1].shape)
print(elems_list[-1].shape)
print(elemfeats_list[-1].shape)
print(infos_list[-1], flush=True)

nnodes, node_mask, nodes, node_weights, features, directed_edges, edge_gradient_weights = ahmed_data(
    nodes_list, elems_list, elemfeats_list, infos_list, node_weight_type=None)

print(f"nodes{nodes.shape}, directed_edges{directed_edges.shape}, features{features.shape}")

np.savez("geokno_triangle_equal_weight_data.npz", nnodes=nodes, node_mask=node_mask, nodes=nodes,
         node_weights=node_weights, features=features, directed_edges=directed_edges,
         edge_gradient_weights=edge_gradient_weights)
