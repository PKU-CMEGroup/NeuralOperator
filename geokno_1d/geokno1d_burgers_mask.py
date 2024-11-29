import random
import torch
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
from timeit import default_timer
from scipy.io import loadmat
from geokno import compute_Fourier_modes, GeoKNO, GeoKNO_train

torch.set_printoptions(precision=16)

torch.manual_seed(0)
np.random.seed(10)

n_train = 1000
n_test = 200
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_data = np.load("datasets/burgers/geokno_uniform_coord.npz")
test_data  = np.load("datasets/burgers/geokno_uniform_coord.npz")

train_nnodes = train_data["nnodes"]
train_node_mask = train_data["node_mask"]
train_nodes = train_data["nodes"]
train_node_weights = train_data["node_weights"]
train_features = train_data["features"]
train_directed_edges = train_data["directed_edges"]
train_edge_gradient_weights = train_data["edge_gradient_weights"]

train_nnodes = torch.from_numpy(train_nnodes)
train_node_mask = torch.from_numpy(train_node_mask)
train_nodes = torch.from_numpy(train_nodes.astype(np.float32))
train_node_weights = torch.from_numpy(train_node_weights.astype(np.float32))
train_features = torch.from_numpy(train_features.astype(np.float32))
train_directed_edges = torch.from_numpy(train_directed_edges)
train_edge_gradient_weights = torch.from_numpy(train_edge_gradient_weights.astype(np.float32))

test_nnodes = test_data["nnodes"]
test_node_mask = test_data["node_mask"]
test_nodes = test_data["nodes"]
test_node_weights = test_data["node_weights"]
test_features = test_data["features"]
test_directed_edges = test_data["directed_edges"]
test_edge_gradient_weights = test_data["edge_gradient_weights"]

test_nnodes = torch.from_numpy(test_nnodes)
test_node_mask = torch.from_numpy(test_node_mask)
test_nodes = torch.from_numpy(test_nodes.astype(np.float32))
test_node_weights = torch.from_numpy(test_node_weights.astype(np.float32))
test_features = torch.from_numpy(test_features.astype(np.float32))
test_directed_edges = torch.from_numpy(test_directed_edges)
test_edge_gradient_weights = torch.from_numpy(test_edge_gradient_weights.astype(np.float32))

n_train = 1000
n_test = 200

x_train   = torch.cat((train_features[:n_train, :, [0]], train_nodes[:n_train, ...]), -1)
x_test    = torch.cat((test_features[-n_test:, :, [0]], test_nodes[-n_test:, ...]), -1)
aux_train = (train_node_mask[0:n_train,...], train_nodes[0:n_train,...], train_node_weights[0:n_train,...], train_directed_edges[0:n_train,...], train_edge_gradient_weights[0:n_train,...])
aux_test  = (test_node_mask[-n_test:,...], test_nodes[-n_test:,...], test_node_weights[-n_test:,...], test_directed_edges[-n_test:,...], test_edge_gradient_weights[-n_test:,...])
y_train   = train_features[:n_train, :, [1]]
y_test    = test_features[-n_test:, :, [1]]

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


epochs       = 2000
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
    x_train, aux_train, y_train, x_test, aux_test, y_test, config, model, save_model_name="GeoKNO_burgers_model"
)