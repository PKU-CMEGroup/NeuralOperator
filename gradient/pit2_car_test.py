import torch
import sys
import numpy as np
import os


os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.append("../")

from gradient.pit2 import PhyGaussNO
from gradient.geokno import GeoKNO_train, compute_Fourier_modes

torch.set_printoptions(precision=16)
torch.manual_seed(0)
np.random.seed(0)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


data = np.load("../data/car/geokno_triangle_data.npz")
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

data_in, data_out = nodes, features

n_train, n_test = 500, 100


x_train, x_test = nodes[:n_train, ...], nodes[-n_test:, ...]
aux_train = (node_mask[0:n_train, ...], nodes[0:n_train, ...], node_weights[0:n_train, ...],
             directed_edges[0:n_train, ...], edge_gradient_weights[0:n_train, ...])
aux_test = (node_mask[-n_test:, ...], nodes[-n_test:, ...], node_weights[-n_test:, ...],
            directed_edges[-n_test:, ...], edge_gradient_weights[-n_test:, ...])
y_train, y_test = features[:n_train, ...], features[-n_test:, ...]


k_max = 8
ndim = 3
modes = compute_Fourier_modes(ndim, [k_max, k_max, k_max], [1.0, 1.0, 1.0])
modes = torch.tensor(modes, dtype=torch.float).to(device)
model = PhyGaussNO(ndim, modes,
                   layers=[128, 128, 128, 128, 128],
                   fc_dim=128,
                   in_dim=3, out_dim=3,
                   act='gelu').to(device)


epochs = 1000
base_lr = 0.001
scheduler = "OneCycleLR"
weight_decay = 1.0e-4
batch_size = 8

normalization_x = True
normalization_y = True
normalization_dim = []
x_aux_dim = 0
y_aux_dim = 0


config = {"train": {"base_lr": base_lr, "weight_decay": weight_decay, "epochs": epochs, "scheduler": scheduler, "batch_size": batch_size,
                    "normalization_x": normalization_x, "normalization_y": normalization_y, "normalization_dim": normalization_dim,
                    "x_aux_dim": x_aux_dim, "y_aux_dim": y_aux_dim}}


train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = GeoKNO_train(
    x_train, aux_train, y_train, x_test, aux_test, y_test, config, model
)
