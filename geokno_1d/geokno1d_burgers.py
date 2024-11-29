import random
import torch
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
from timeit import default_timer
from scipy.io import loadmat
from geo_utility import convert_structured_data, preprocess_data
from geokno import compute_Fourier_modes, GeoKNO, GeoKNO_train

torch.set_printoptions(precision=16)

torch.manual_seed(0)
np.random.seed(10)

downsample_ratio = 8
n_train = 1000
n_test = 200
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

CONVERT_DATA = True
if CONVERT_DATA:
    ###################################
    # load data
    ###################################
    dataloader = loadmat("datasets\\burgers\\burgers_data_R10.mat")
    data_in = np.array(dataloader.get('a'))
    data_out = np.array(dataloader.get('u'))

    print("data_in.shape " , data_in.shape)
    print("data_out.shape", data_out.shape)

    Np_ref = data_in.shape[1]
    Np = 1 + (Np_ref -  1)//downsample_ratio
    L = 1.0
    grid_1d = np.linspace(0, L, Np)

    data_in_ds = data_in[:, 0::downsample_ratio]
    data_out_ds = data_out[:, 0::downsample_ratio]
    ndata = data_in_ds.shape[0]
    features = np.stack((data_in_ds, data_out_ds), axis = 2)

    nodes_list, elems_list, features_list = convert_structured_data([np.tile(grid_1d, (ndata, 1))], features, nnodes_per_elem = 2, feature_include_coords = False)
    nnodes, node_mask, nodes, node_weights, features, directed_edges, edge_gradient_weights = preprocess_data(nodes_list, elems_list, features_list, node_weight_type=None)
    np.savez_compressed("datasets/burgers/geokno_quad_equal_weight_data.npz", nnodes=nodes, node_mask=node_mask, nodes=nodes, node_weights=node_weights, features=features, directed_edges=directed_edges, edge_gradient_weights=edge_gradient_weights)
    exit()
else:
    data = np.load("datasets/burgers/geokno_quad_equal_weight_data.npz")
    nnodes, node_mask, nodes, node_weights, features, directed_edges, edge_gradient_weights = data["nnodes"], data["node_mask"], data["nodes"], data["node_weights"], data["features"], data["directed_edges"], data["edge_gradient_weights"]

nnodes = torch.from_numpy(nnodes)
node_mask = torch.from_numpy(node_mask)
nodes = torch.from_numpy(nodes.astype(np.float32))
node_weights = torch.from_numpy(node_weights.astype(np.float32))
features = torch.from_numpy(features.astype(np.float32))
directed_edges = torch.from_numpy(directed_edges)
edge_gradient_weights = torch.from_numpy(edge_gradient_weights.astype(np.float32))

n_train = 1000
n_test = 200

x_train, x_test = torch.cat((features[:n_train, :, [0]],nodes[:n_train, ...]),-1), torch.cat((features[-n_test:, :, [0]],nodes[-n_test:, ...]),-1)
aux_train       = (node_mask[0:n_train,...], nodes[0:n_train,...], node_weights[0:n_train,...], directed_edges[0:n_train,...], edge_gradient_weights[0:n_train,...])
aux_test        = (node_mask[-n_test:,...],  nodes[-n_test:,...],  node_weights[-n_test:,...],  directed_edges[-n_test:,...],  edge_gradient_weights[-n_test:,...])
y_train, y_test = features[:n_train, :, [1]],       features[-n_test:, :, [1]]
print("x_train.shape: ", x_train.shape)
print("y_train.shape: ", y_train.shape)

k_max = 32
ndim = 1
modes = compute_Fourier_modes(ndim, k_max, 1.0)
modes = torch.tensor(modes, dtype=torch.float).to(device)
model = GeoKNO(ndim, modes,
               layers=[128,128,128,128,128],
               fc_dim=128,
               in_dim=2, out_dim=1,
               act='gelu').to(device)


epochs       = 1000
base_lr      = 0.001
scheduler    = "OneCycleLR"
weight_decay = 1.0e-4
batch_size   = 20

normalization_x = True
normalization_y = True
normalization_dim = []
x_aux_dim = 1
y_aux_dim = 0


config = {"train" : {"base_lr": base_lr, "weight_decay": weight_decay, "epochs": epochs, "scheduler": scheduler,  "batch_size": batch_size, 
                     "normalization_x": normalization_x,"normalization_y": normalization_y, "normalization_dim": normalization_dim, 
                     "x_aux_dim": x_aux_dim, "y_aux_dim": y_aux_dim}}

train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = GeoKNO_train(
    x_train, aux_train, y_train, x_test, aux_test, y_test, config, model, save_model_name="./GeoKNO_darcy_model"
)