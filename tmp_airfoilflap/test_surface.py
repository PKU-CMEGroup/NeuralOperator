import os
import sys
import torch
import numpy as np


os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.append("../")

from tmp_airfoilflap.geokno_adjust import compute_Fourier_modes, GeoKNO, GeoKNO_train_surface


torch.set_printoptions(precision=16)
torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
###################################
# load data
###################################
data = np.load("../data/airfoil_flap/geokno_triangle_equal_weight_data.npz")
nnodes, node_mask, nodes, node_weights, features, directed_edges, edge_gradient_weights = data["nnodes"], data[
    "node_mask"], data["nodes"], data["node_weights"], data["features"], data["directed_edges"], data["edge_gradient_weights"]


###################################
# prepare data
###################################
ndata = nnodes.shape[0]

print("Casting to tensor")
nnodes = torch.from_numpy(nnodes)
node_mask = torch.from_numpy(node_mask)
nodes = torch.from_numpy(nodes.astype(np.float32))
node_weights = torch.from_numpy(node_weights.astype(np.float32))
features = torch.from_numpy(features.astype(np.float32))
directed_edges = torch.from_numpy(directed_edges)
edge_gradient_weights = torch.from_numpy(
    edge_gradient_weights.astype(np.float32))

data_in = nodes.clone()

n_train, n_test = 1000, 500
n1_train, n1_test = n_train // 2, n_test // 2

train_index = torch.cat([torch.arange(n1_train),
                        torch.arange(ndata - n1_train, ndata)], dim=0)
test_index = torch.cat([torch.arange(n1_train, n1_train + n1_test),
                       torch.arange(ndata - n1_train - n1_test, ndata - n1_train)], dim=0)


x_train, x_test = data_in[train_index, ...], data_in[test_index, ...]
aux_train = (node_mask[train_index, ...], nodes[train_index, ...], node_weights[train_index, ...],
             directed_edges[train_index, ...], edge_gradient_weights[train_index, ...])
aux_test = (node_mask[test_index, ...], nodes[test_index, ...], node_weights[test_index, ...],
            directed_edges[test_index, ...], edge_gradient_weights[test_index, ...])
y_train, y_test = features[train_index, :, [1]], features[test_index, :, [1]]
s_train, s_test = features[train_index, :, [2]], features[test_index, :, [2]]
print(x_train.shape)
print(y_train.shape)
print(s_train.shape)
###################################
# train
###################################
kx_max, ky_max = 32, 16
ndim = 2
Lx = 1.25
Ly = 0.5
print("Lx, Ly = ", Lx, Ly, flush=True)
modes = compute_Fourier_modes(ndim, [kx_max, ky_max], [Lx, Ly])
modes = torch.tensor(modes, dtype=torch.float).to(device)
model = GeoKNO(ndim, modes,
               layers=[128, 128, 128, 128, 128],
               should_learn_L=True,
               fc_dim=128,
               in_dim=2, out_dim=1,
               act='gelu').to(device)

epochs = 500
base_lr = 0.001
scheduler = "OneCycleLR"
weight_decay = 1.0e-4
batch_size = 20

normalization_x = True
normalization_y = True
normalization_dim = []
x_aux_dim = 0
y_aux_dim = 0


config = {"train": {"base_lr": base_lr, "weight_decay": weight_decay, "epochs": epochs, "scheduler": scheduler, "batch_size": batch_size,
                    "normalization_x": normalization_x, "normalization_y": normalization_y, "normalization_dim": normalization_dim,
                    "x_aux_dim": x_aux_dim, "y_aux_dim": y_aux_dim}}


train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = GeoKNO_train_surface(
     x_train, aux_train, y_train, s_train, x_test, aux_test, y_test, s_test, config, model, should_print_L=True
)
