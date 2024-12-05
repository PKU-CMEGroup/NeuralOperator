import random
import torch
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
from timeit import default_timer
from scipy.io import loadmat

sys.path.append("../")

from baselines.geo_utility import preprocess_data
from baselines.geokno import compute_Fourier_modes, GeoKNO, GeoKNO_train
from baselines.geofno import GeoFNO, GeoFNO_train

torch.set_printoptions(precision=16)

torch.manual_seed(0)
np.random.seed(10)

downsample_ratio = 1
n_train = 1000
n_test = 200
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

k_str='-1e+02'   # '0e+00,1e+00,1e+01,1e+02,-1e+00,+1e+01,-1e+02'
equal_weight = True
print('k: '+k_str)
print('equal_weight: ',equal_weight)

CONVERT_DATA = True
if CONVERT_DATA:
    ###################################
    # load data
    ###################################
    dataloader = loadmat(f"../data/burgers/burgers_k{k_str}_N2048.mat")
    nodes_list = np.array(dataloader.get('nodes_list'))[...,np.newaxis]
    elems_list = np.array(dataloader.get('elems_list'))
    elems_list[:,:,1:] = elems_list[:,:,1:]-1  #python index starts from 0
    features_list = np.array(dataloader.get('features_list'))
    print('nodes_list.shape',nodes_list.shape)
    print('elems_list.shape',elems_list.shape)
    print('features_list.shape',features_list.shape)

    nnodes, node_mask, nodes, node_weights, features, directed_edges, edge_gradient_weights = preprocess_data(nodes_list, elems_list, features_list, node_weight_type=None)
    np.savez_compressed(f"../data/burgers/burgers_k{k_str}_N2048_equal_weight_data.npz", nnodes=nnodes, node_mask=node_mask, nodes=nodes, node_weights=node_weights, features=features, directed_edges=directed_edges, edge_gradient_weights=edge_gradient_weights)
    nnodes, node_mask, nodes, node_weights, features, directed_edges, edge_gradient_weights = preprocess_data(nodes_list, elems_list, features_list, node_weight_type='area')
    np.savez_compressed(f"../data/burgers/burgers_k{k_str}_N2048_data.npz", nnodes=nnodes, node_mask=node_mask, nodes=nodes, node_weights=node_weights, features=features, directed_edges=directed_edges, edge_gradient_weights=edge_gradient_weights)
    exit()
else:
    if equal_weight:
        data_path = f"../data/burgers/burgers_k{k_str}_N2048_equal_weight_data.npz"
        model_path = f"./model/k{k_str}_equal_weight.pth"
    else:
        data_path = f"../data/burgers/burgers_k{k_str}_N2048_data.npz"
        model_path = f"./model/k{k_str}.pth"
    data = np.load(data_path)
    print('load data from' + data_path)
    nnodes, node_mask, nodes, node_weights, features, directed_edges, edge_gradient_weights = data["nnodes"], data["node_mask"], data["nodes"], data["node_weights"], data["features"], data["directed_edges"], data["edge_gradient_weights"]

nnodes = torch.from_numpy(nnodes)
node_mask = torch.from_numpy(node_mask)
nodes = torch.from_numpy(nodes.astype(np.float32))
node_weights = torch.from_numpy(node_weights.astype(np.float32))
features = torch.from_numpy(features.astype(np.float32))
directed_edges = torch.from_numpy(directed_edges).to(torch.int64)
edge_gradient_weights = torch.from_numpy(edge_gradient_weights.astype(np.float32))
print("features.shape: ", features.shape)
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


epochs       = 500
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
    x_train, aux_train, y_train, x_test, aux_test, y_test, config, model, save_model_name = model_path
)