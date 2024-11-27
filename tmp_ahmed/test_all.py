import os
import torch
import sys
import numpy as np

sys.path.append("../")

from tmp_ahmed.geokno import compute_Fourier_modes, GeoKNO, GeoKNO_train

torch.set_printoptions(precision=16)
torch.manual_seed(0)
np.random.seed(0)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data = np.load(
    "../data/ahmed/geokno_triangle_equal_weight_data.npz",
    # "../data/ahmed/geokno_triangle_data.npz",
    # allow_pickle=True
)

nnodes, node_mask, nodes, node_weights, features, directed_edges, edge_gradient_weights = data["nnodes"], data["node_mask"], data[
    "nodes"], data["node_weights"], data["features"], data["directed_edges"], data["edge_gradient_weights"]



print("Casting to tensor")
nnodes = torch.from_numpy(nnodes)
node_mask = torch.from_numpy(node_mask)
nodes = torch.from_numpy(nodes.astype(np.float32))
node_weights = torch.from_numpy(node_weights.astype(np.float32))

# 0 mean pressure of neighbour triangles,
# 1 length, 2 width, 3 height, 4 clearance, 5 slant, 6 radius, 7 velocity, 8 reynolds number
# Normalize the infos to the range [0,1]
normalization_infos = True
if normalization_infos:
    eps = 1e-06
    with open("../data/ahmed/info_bounds.txt", "r") as fp:
        min_bounds = fp.readline().split(" ")
        max_bounds = fp.readline().split(" ")

        min_bounds = [float(a) - eps for a in min_bounds]
        max_bounds = [float(a) + eps for a in max_bounds]

    for i in range(8):
        features[..., i + 1] = (features[..., i + 1]
                                - min_bounds[i]) / (max_bounds[i] - min_bounds[i])
features = torch.from_numpy(features.astype(np.float32))

directed_edges = torch.from_numpy(directed_edges)
edge_gradient_weights = torch.from_numpy(
    edge_gradient_weights.astype(np.float32))

data_in, data_out = torch.cat([nodes,features[...,1:]], dim=-1), features[...,:1]
print(f"data in:{data_in.shape}, data out:{data_out.shape}")
n_train, n_test = 500, 50


x_train, x_test = data_in[:n_train, ...], data_in[-n_test:, ...]
aux_train = (node_mask[0:n_train, ...], nodes[0:n_train, ...], node_weights[0:n_train, ...],
             directed_edges[0:n_train, ...], edge_gradient_weights[0:n_train, ...])
aux_test = (node_mask[-n_test:, ...], nodes[-n_test:, ...], node_weights[-n_test:, ...],
            directed_edges[-n_test:, ...], edge_gradient_weights[-n_test:, ...])
y_train, y_test = data_out[:n_train, ...], data_out[-n_test:, ...]

print(f"x train:{x_train.shape}, y train:{y_train.shape}", flush=True)

k_max = 8
ndim = 3

Lx = 0.0004795 - (-1.34399998)
Ly = 0.25450477 - 0
Lz = 0.43050185 - 0
# Lx = Ly = Lz = 1

modes = compute_Fourier_modes(ndim, [k_max, k_max, k_max], [Lx, Ly, Lz])
modes = torch.tensor(modes, dtype=torch.float).to(device)
model = GeoKNO(ndim, modes,
               layers=[128, 128, 128, 128, 128],
               fc_dim=128,
               in_dim=11, out_dim=1,
               act='gelu').to(device)


epochs = 500
base_lr = 0.001
scheduler = "OneCycleLR"
weight_decay = 1.0e-4
batch_size = 5

normalization_x = True
normalization_y = True
normalization_dim = []
x_aux_dim = 8
y_aux_dim = 0


config = {"train": {"base_lr": base_lr, "weight_decay": weight_decay, "epochs": epochs, "scheduler": scheduler, "batch_size": batch_size,
                    "normalization_x": normalization_x, "normalization_y": normalization_y, "normalization_dim": normalization_dim,
                    "x_aux_dim": x_aux_dim, "y_aux_dim": y_aux_dim}}


train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = GeoKNO_train(
    x_train, aux_train, y_train, x_test, aux_test, y_test, config, model
)
