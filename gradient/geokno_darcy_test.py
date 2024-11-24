import random
import torch
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
from timeit import default_timer
from scipy.io import loadmat
import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.append("../")


from gradient.geo_utility import preprocess_data, convert_structured_data
from gradient.geokno import compute_Fourier_modes, GeoKNO, GeoKNO_train


torch.set_printoptions(precision=16)

torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


CONVERT_DATA = True

if CONVERT_DATA:
    ###################################
    # load data
    ###################################
    data_path = "../data/darcy_2d/piececonst_r421_N1024_smooth1"
    data1 = loadmat(data_path)
    data_path = "../data/darcy_2d/piececonst_r421_N1024_smooth2"
    data2 = loadmat(data_path)
    downsample_ratio = 2
    data_in = np.vstack((data1["coeff"], data2["coeff"]))[
        :, 0::downsample_ratio, 0::downsample_ratio]  # shape: 2048,421,421
    data_out = np.vstack((data1["sol"], data2["sol"]))[
        :, 0::downsample_ratio, 0::downsample_ratio]     # shape: 2048,421,421
    features = np.stack((data_in, data_out), axis=3)
    ndata = data_in.shape[0]

    Np = data_in.shape[1]
    L = 1.0
    grid_1d = np.linspace(0, L, Np)
    grid_x, grid_y = np.meshgrid(grid_1d, grid_1d)
    grid_x, grid_y = grid_x.T, grid_y.T

    nodes_list, elems_list, features_list = convert_structured_data([np.tile(grid_x, (ndata, 1, 1)), np.tile(
        grid_y, (ndata, 1, 1))], features, nnodes_per_elem=4, feature_include_coords=False)
    # uniform weights
    nnodes, node_mask, nodes, node_weights, features, directed_edges, edge_gradient_weights = preprocess_data(
        nodes_list, elems_list, features_list, node_weight_type=None)
    np.savez_compressed("../data/darcy_2d/geokno_quad_equal_weight_data.npz", nnodes=nodes, node_mask=node_mask, nodes=nodes,
                        node_weights=node_weights, features=features, directed_edges=directed_edges, edge_gradient_weights=edge_gradient_weights)
    nnodes, node_mask, nodes, node_weights, features, directed_edges, edge_gradient_weights = preprocess_data(
        nodes_list, elems_list, features_list, node_weight_type="area")
    np.savez_compressed("../data/darcy_2d/geokno_quad_data.npz", nnodes=nodes, node_mask=node_mask, nodes=nodes,
                        node_weights=node_weights, features=features, directed_edges=directed_edges, edge_gradient_weights=edge_gradient_weights)
    exit()
else:
    data = np.load("../data/darcy_2d/geokno_quad_equal_weight_data.npz")
    nnodes, node_mask, nodes, node_weights, features, directed_edges, edge_gradient_weights = data["nnodes"], data[
        "node_mask"], data["nodes"], data["node_weights"], data["features"], data["directed_edges"], data["edge_gradient_weights"]


print("Casting to tensor")
nnodes = torch.from_numpy(nnodes)
node_mask = torch.from_numpy(node_mask)
nodes = torch.from_numpy(nodes.astype(np.float32))
node_weights = torch.from_numpy(node_weights.astype(np.float32))
features = torch.from_numpy(features.astype(np.float32))
directed_edges = torch.from_numpy(directed_edges)
edge_gradient_weights = torch.from_numpy(
    edge_gradient_weights.astype(np.float32))

n_train = 1000
n_test = 200

x_train, x_test = torch.cat((features[:n_train, :, [0]], nodes[:n_train, ...]), -1), torch.cat(
    (features[-n_test:, :, [0]], nodes[-n_test:, ...]), -1)
aux_train = (node_mask[0:n_train, ...], nodes[0:n_train, ...], node_weights[0:n_train, ...],
             directed_edges[0:n_train, ...], edge_gradient_weights[0:n_train, ...])
aux_test = (node_mask[-n_test:, ...], nodes[-n_test:, ...], node_weights[-n_test:, ...],
            directed_edges[-n_test:, ...], edge_gradient_weights[-n_test:, ...])
y_train, y_test = features[:n_train, :, [1]], features[-n_test:, :, [1]]


k_max = 16
ndim = 2
modes = compute_Fourier_modes(ndim, [k_max, k_max], [1.0, 1.0])
modes = torch.tensor(modes, dtype=torch.float).to(device)
model = GeoKNO(ndim, modes,
               layers=[128, 128, 128, 128, 128],
               fc_dim=128,
               in_dim=3, out_dim=1,
               act='gelu').to(device)


epochs = 1000
base_lr = 0.001
scheduler = "OneCycleLR"
weight_decay = 1.0e-4
batch_size = 8

normalization_x = True
normalization_y = True
normalization_dim = []
x_aux_dim = 2
y_aux_dim = 0


config = {"train": {"base_lr": base_lr, "weight_decay": weight_decay, "epochs": epochs, "scheduler": scheduler, "batch_size": batch_size,
                    "normalization_x": normalization_x, "normalization_y": normalization_y, "normalization_dim": normalization_dim,
                    "x_aux_dim": x_aux_dim, "y_aux_dim": y_aux_dim}}

train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = GeoKNO_train(
    x_train, aux_train, y_train, x_test, aux_test, y_test, config, model, save_model_name="./GeoKNO_darcy_model"
)
