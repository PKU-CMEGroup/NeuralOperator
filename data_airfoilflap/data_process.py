import numpy as np
import os
from pre import preprocess_data

elem_dim = 2
nodes_list = []
elems_list = []
feats_list = []

path1 = "Airfoil_flap_data/fluid_mesh"
nodes_file_list1 = sorted([f for f in os.listdir(path1)
                          if f.startswith("nodes_") and f.endswith(".npy")])[:10]
elems_file_list1 = sorted([f for f in os.listdir(path1)
                          if f.startswith("elems_") and f.endswith(".npy")])[:10]
feats_file_list1 = sorted([f for f in os.listdir(path1)
                          if f.startswith("features_") and f.endswith(".npy")])[:10]
for i, file in enumerate(nodes_file_list1):
    file_path = os.path.join(path1, file)
    nodes = np.load(file_path)
    nodes_list.append(nodes)
for i, file in enumerate(elems_file_list1):
    file_path = os.path.join(path1, file)
    elems = np.load(file_path)
    elems_list.append(np.concatenate(
        (np.full((elems.shape[0], 1), elem_dim, dtype=int), elems), axis=1))
for i, file in enumerate(feats_file_list1):
    file_path = os.path.join(path1, file)
    feats = np.load(file_path)
    feats_list.append(feats)


path2 = "Airfoil_data/fluid_mesh"
nodes_file_list2 = sorted([f for f in os.listdir(path2)
                          if f.startswith("nodes_") and f.endswith(".npy")])[:10]
elems_file_list2 = sorted([f for f in os.listdir(path2)
                          if f.startswith("elems_") and f.endswith(".npy")])[:10]
feats_file_list2 = sorted([f for f in os.listdir(path2)
                          if f.startswith("features_") and f.endswith(".npy")])[:10]
for i, file in enumerate(nodes_file_list2):
    file_path = os.path.join(path2, file)
    nodes = np.load(file_path)
    nodes_list.append(nodes)
for i, file in enumerate(elems_file_list2):
    file_path = os.path.join(path2, file)
    elems = np.load(file_path)
    elems_list.append(np.concatenate(
        (np.full((elems.shape[0], 1), elem_dim, dtype=int), elems), axis=1))
for i, file in enumerate(feats_file_list2):
    file_path = os.path.join(path2, file)
    feats = np.load(file_path)
    feats_list.append(feats)


print(nodes_list[0].shape, elems_list[0].shape, feats_list[0].shape)
print(nodes_list[1].shape, elems_list[1].shape, feats_list[1].shape)
print(nodes_list[-1].shape, elems_list[-1].shape, feats_list[-1].shape)
print(nodes_list[-2].shape, elems_list[-2].shape, feats_list[-2].shape)

print(
    f"Total files loaded: node{len(nodes_list)},elem{len(elems_list)},feats{len(feats_list)}")

nnodes, node_mask, nodes, node_weights, features, directed_edges, edge_gradient_weights = preprocess_data(
    nodes_list, elems_list, feats_list, node_weight_type=None)
np.savez("geokno_triangle_equal_weight_data.npz", nnodes=nodes, node_mask=node_mask, nodes=nodes,
         node_weights=node_weights, features=features, directed_edges=directed_edges,
         edge_gradient_weights=edge_gradient_weights)

nnodes, node_mask, nodes, node_weights, features, directed_edges, edge_gradient_weights = preprocess_data(
    nodes_list, elems_list, feats_list, node_weight_type="area")
np.savez("geokno_triangle_data.npz", nnodes=nodes, node_mask=node_mask, nodes=nodes,
         node_weights=node_weights, features=features, directed_edges=directed_edges,
         edge_gradient_weights=edge_gradient_weights)
