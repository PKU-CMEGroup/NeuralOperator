import open3d as o3d
import os
import glob
import random
import torch
import sys
import numpy as np
import math
from timeit import default_timer
sys.path.append("../")

from geo_utility import preprocess_data
from geokno import compute_Fourier_modes, GeoKNO, GeoKNO_train

torch.set_printoptions(precision=16)


torch.manual_seed(0)
np.random.seed(0)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


###################################
# load data
###################################

CONVERT_DATA = False
if CONVERT_DATA:
    print("Loading data")
    data_path = "..\data\car\car_data.npz"
    data = np.load(data_path)

    all_points = data["points"]
    all_triangles = data["triangles"]
    all_normals = data["normals"]
    all_pressures = data["pressures"]


    all_triangles = np.concatenate((3*np.ones((all_triangles.shape[0],all_triangles.shape[1],1),dtype = int),all_triangles),axis = -1)
    nodes_list = [all_points[i] for i in range(all_points.shape[0])]
    elems_list = [all_triangles[i] for i in range(all_triangles.shape[0])]
    features_list = [all_normals[i] for i in range(all_normals.shape[0])]
    nnodes, node_mask, nodes, node_weights, features, directed_edges, edge_gradient_weights = preprocess_data(nodes_list, elems_list, features_list, node_weight_type="area")
    np.savez("../data/car/geokno_triangle_data.npz", nnodes=nodes, node_mask=node_mask, nodes=nodes, node_weights=node_weights, features=features, directed_edges=directed_edges, edge_gradient_weights=edge_gradient_weights)
    exit()
else:
    data = np.load("../data/car/geokno_triangle_data.npz")
    nnodes, node_mask, nodes, node_weights, features, directed_edges, edge_gradient_weights = data["nnodes"], data["node_mask"], data["nodes"], data["node_weights"], data["features"], data["directed_edges"], data["edge_gradient_weights"]



print("Casting to tensor")
nnodes = torch.from_numpy(nnodes)
node_mask = torch.from_numpy(node_mask)
nodes = torch.from_numpy(nodes.astype(np.float32))
node_weights = torch.from_numpy(node_weights.astype(np.float32))
features = torch.from_numpy(features.astype(np.float32))
directed_edges = torch.from_numpy(directed_edges)
edge_gradient_weights = torch.from_numpy(edge_gradient_weights.astype(np.float32))

data_in, data_out = nodes, features

n_train, n_test = 500, 100


x_train, x_test = nodes[:n_train,...], nodes[-n_test:,...]
aux_train       = (node_mask[0:n_train,...], nodes[0:n_train,...], node_weights[0:n_train,...], directed_edges[0:n_train,...], edge_gradient_weights[0:n_train,...])
aux_test        = (node_mask[-n_test:,...],  nodes[-n_test:,...],  node_weights[-n_test:,...],  directed_edges[-n_test:,...],  edge_gradient_weights[-n_test:,...])
y_train, y_test = features[:n_train,...],     features[-n_test:,...]


k_max = 4
ndim = 3
kernel_modes = 32
modes = compute_Fourier_modes(ndim, [k_max,k_max,k_max], [1.0,1.0,1.0])
print('modes.shape:',modes.shape)
modes = torch.tensor(modes, dtype=torch.float).to(device)
model = GeoKNO(ndim, modes,kernel_modes,
               layers=[64,64,64,64,64],
               fc_dim=128,
               in_dim=3, out_dim=3,
               act='gelu').to(device)



epochs = 500
base_lr = 0.001
scheduler = "OneCycleLR"
weight_decay = 1.0e-4
batch_size= 10

normalization_x = True
normalization_y = True
normalization_dim = []




config = {"train" : {"base_lr": base_lr, "weight_decay": weight_decay, "epochs": epochs, "scheduler": scheduler,  "batch_size": batch_size, 
                     "normalization_x": normalization_x,"normalization_y": normalization_y, "normalization_dim": normalization_dim}}

train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = GeoKNO_train(
    x_train, aux_train, y_train, x_test, aux_test, y_test, config, model, save_model_name="./GeoKNO_car_model"
)







# kx_max, ky_max = 32, 16
# ndim = 2
# pad_ratio = 0.05
# # Lx, Ly = (1.0+pad_ratio)*(grid_ds[0:n_train,...,0].max()-grid_ds[0:n_train,...,0].min()), (1.0+pad_ratio)*(grid_ds[0:n_train,...,1].max()-grid_ds[0:n_train,...,1].min())
# Lx = Ly = 4.0
# print("Lx, Ly = ", Lx, Ly)
# modes = compute_Fourier_modes(ndim, [kx_max,ky_max], [Lx, Ly])
# modes = torch.tensor(modes, dtype=torch.float).to(device)
# model = GeoFNO(ndim, modes,
#                layers=[128,128,128,128,128],
#                fc_dim=128,
#                in_dim=2, out_dim=1,
#                act='gelu').to(device)


# epochs = 500
# base_lr = 0.001
# scheduler = "OneCycleLR"
# weight_decay = 1.0e-4
# batch_size=20

# normalization_x = True
# normalization_y = True
# normalization_dim = []

# config = {"train" : {"base_lr": base_lr, "weight_decay": weight_decay, "epochs": epochs, "scheduler": scheduler,  "batch_size": batch_size, 
#                      "normalization_x": normalization_x,"normalization_y": normalization_y, "normalization_dim": normalization_dim}}

# train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = GeoFNO_train(
#     x_train, y_train, x_test, y_test, config, model, save_model_name="./GeoFNO_darcy_model"
# )





