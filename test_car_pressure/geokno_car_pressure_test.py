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

from baselines.geo_utility import preprocess_data
from baselines.geokno import compute_Fourier_modes, GeoKNO, GeoKNO_train
from baselines.geofno import GeoFNO, GeoFNO_train

torch.set_printoptions(precision=16)


torch.manual_seed(0)
np.random.seed(0)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def read_mesh_indices(file_path):
    with open(file_path, "r") as f:
        indices = f.read().splitlines()
    return [int(index) - 1 for index in indices]

def load_data(data_path = "../data/car"):
    dataset_folder_mesh = data_path+"/data/mesh/"
    dataset_folder_pressure = data_path+"/data/press/"
    valid_indices = read_mesh_indices(data_path+"/watertight_meshes.txt")
    ply_files = sorted(glob.glob(os.path.join(dataset_folder_mesh, "*.ply")))
    npy_pressure_files = sorted(glob.glob(os.path.join(dataset_folder_pressure, "*.npy")))
    nodes, max_nnodes = [], 0
    elems, elem_dim = [], 2
    features = []

    for index, i in enumerate(valid_indices):
        ply_file = ply_files[i]
        mesh = o3d.io.read_triangle_mesh(ply_file)
        vertices = np.asarray(mesh.vertices)
        max_nnodes = max(max_nnodes, vertices.shape[0])
        nodes.append(vertices)
        elem = np.asarray(mesh.triangles)
        elems.append(np.concatenate((np.full((elem.shape[0], 1), elem_dim, dtype=int), elem), axis=1))
        press = np.load(npy_pressure_files[i])
        press = np.concatenate((press[0:16], press[112:]), axis=0)[:,np.newaxis]
        features.append(press)
    return max_nnodes, nodes, elems, features 
###################################
# load data
###################################

CONVERT_DATA = False
if CONVERT_DATA:
    print("Loading data")
    data_path = "../data/car_pressure"
    max_nnodes, nodes_list, elems_list, features_list  = load_data(data_path = data_path)

    print("Preprocessing data")

    nnodes, node_mask, nodes, node_weights, features, directed_edges, edge_gradient_weights = preprocess_data(nodes_list, elems_list, features_list, node_weight_type=None)
    np.savez("../data/car_pressure/geokno_triangle_equal_weight_data.npz", nnodes=nnodes, node_mask=node_mask, nodes=nodes, node_weights=node_weights, features=features, directed_edges=directed_edges, edge_gradient_weights=edge_gradient_weights)

    nnodes, node_mask, nodes, node_weights, features, directed_edges, edge_gradient_weights = preprocess_data(nodes_list, elems_list, features_list, node_weight_type="area")
    np.savez("../data/car_pressure/geokno_triangle_data.npz", nnodes=nnodes, node_mask=node_mask, nodes=nodes, node_weights=node_weights, features=features, directed_edges=directed_edges, edge_gradient_weights=edge_gradient_weights)
    exit()
else:
    data = np.load("../data/car_pressure/geokno_triangle_data.npz")
    nnodes, node_mask, nodes, node_weights, features, directed_edges, edge_gradient_weights = data["nnodes"], data["node_mask"], data["nodes"], data["node_weights"], data["features"], data["directed_edges"], data["edge_gradient_weights"]
    


print("Casting to tensor")
nnodes = torch.from_numpy(nnodes)
node_mask = torch.from_numpy(node_mask)
nodes = torch.from_numpy(nodes.astype(np.float32))
node_weights = torch.from_numpy(node_weights.astype(np.float32))
features = torch.from_numpy(features.astype(np.float32))
directed_edges = torch.from_numpy(directed_edges).to(torch.int64)
edge_gradient_weights = torch.from_numpy(edge_gradient_weights.astype(np.float32))

data_in, data_out = nodes, features

n_train, n_test = 500, 100


x_train, x_test = nodes[:n_train,...], nodes[-n_test:,...]
aux_train       = (node_mask[0:n_train,...], nodes[0:n_train,...], node_weights[0:n_train,...], directed_edges[0:n_train,...], edge_gradient_weights[0:n_train,...])
aux_test        = (node_mask[-n_test:,...],  nodes[-n_test:,...],  node_weights[-n_test:,...],  directed_edges[-n_test:,...],  edge_gradient_weights[-n_test:,...])
y_train, y_test = features[:n_train,...],     features[-n_test:,...]


k_max = 8
ndim = 3
modes = compute_Fourier_modes(ndim, [k_max,k_max,k_max], [2.0,2.0,5.0])
modes = torch.tensor(modes, dtype=torch.float).to(device)
model = GeoKNO(ndim, modes,
               layers=[32,32,32,32,32],
               fc_dim=128,
               in_dim=3, out_dim=1,
               act='gelu').to(device)



epochs = 500
base_lr = 0.001
scheduler = "OneCycleLR"
weight_decay = 1.0e-4
batch_size=8

normalization_x = True
normalization_y = True
normalization_dim = []
x_aux_dim = 0
y_aux_dim = 0


config = {"train" : {"base_lr": base_lr, "weight_decay": weight_decay, "epochs": epochs, "scheduler": scheduler,  "batch_size": batch_size, 
                     "normalization_x": normalization_x,"normalization_y": normalization_y, "normalization_dim": normalization_dim, 
                     "x_aux_dim": x_aux_dim, "y_aux_dim": y_aux_dim},
        'plot': {'plot_hidden_layers_num': 3, 'save_figure_hidden': 'figure/car_pressure/test/'}
        
        }


train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = GeoKNO_train(
    x_train, aux_train, y_train, x_test, aux_test, y_test, config, model, save_model_name="./GeoKNO_car_pressure_model"
)






