import torch
import sys
import os
import numpy as np
import argparse
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


from pcno.geo_utility import (
    preprocess_data,
    convert_structured_data,
    compute_node_weights,
)
from pcno.pcno import compute_Fourier_modes, PCNO, PCNO_train


torch.set_printoptions(precision=16)

torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


PREPROCESS_DATA = False
parser = argparse.ArgumentParser(description='Train model with different types.')
parser.add_argument('--train_type', type=str, default='mixed', choices=['standard', 'flap', 'mixed'],
                    help='Type of training (standard, flap, mixed)')
parser.add_argument('--feature_type', type=str, default='pressure', choices=['pressure', 'mach', 'both'],
                    help='Type of training (pressure, mach, both)')
parser.add_argument('--n_train', type=int, default=1000,
                    help='training datasize (500,1000,1500)')
parser.add_argument('--train_sp_L', type=str, default='independently', choices=['False' , 'together' , 'independently'],
                    help='type of train_sp_L (False, together, independently )')
parser.add_argument('--Lx', type=float, default=1, help='Initial value for Lx')
parser.add_argument('--Ly', type=float, default=0.5, help='Initial value for Ly')
parser.add_argument('--lr_ratio', type=float, default=10, help='lr_ratio')
parser.add_argument('--rho', type=str, default='True', help='should add rho')


args = parser.parse_args()
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
save_model_name = f"PCNO_airfoilplap_{args.train_type}_{args.feature_type}_n{args.n_train}_Lx{args.Lx}_Ly{args.Ly}_rho_{args.rho}_{current_time}"
print(save_model_name, flush=True)


###################################
# load data
###################################
data_path = "../../data/airfoil_flap/"

print("Loading data")
ndata1 = 1931
ndata2 = 1932
equal_weights = True

data = np.load(data_path + "pcno_triangle_data.npz")
nnodes, node_mask, nodes = data["nnodes"], data["node_mask"], data["nodes"]

node_equal_weights = data["node_equal_weights"] 
node_measures = data["node_measures"]
node_measures_raw = data["node_measures_raw"]
indices = np.isfinite(node_measures_raw)
node_rhos = np.copy(node_equal_weights)
node_rhos[indices] = node_rhos[indices]/node_measures[indices]

directed_edges, edge_gradient_weights = data["directed_edges"], data["edge_gradient_weights"]
features = data["features"]


###################################
# prepare data
###################################
print("Casting to tensor")
nnodes = torch.from_numpy(nnodes)
node_mask = torch.from_numpy(node_mask)
nodes = torch.from_numpy(nodes.astype(np.float32))
node_equal_weights = torch.from_numpy(node_equal_weights.astype(np.float32))
node_rhos = torch.from_numpy(node_rhos.astype(np.float32))
features = torch.from_numpy(features.astype(np.float32))
directed_edges = torch.from_numpy(directed_edges)
edge_gradient_weights = torch.from_numpy(edge_gradient_weights.astype(np.float32))

# This is important
nodes_input = nodes.clone()
in_dim = 2
if args.rho == "True":
    nodes_input = torch.cat([nodes_input, node_rhos], dim=-1)
    in_dim = 3

ndata = nodes_input.shape[0]

print(ndata)

n_train, n_test = args.n_train, 500
m_train, m_test = n_train // 2, n_test // 2

train_type = args.train_type
if train_type == "flap":
    train_index = torch.arange(n_train)
elif train_type  == "standard":
    train_index = torch.arange(ndata - n_train, ndata) 
elif train_type == "mixed":
    train_index = torch.cat(
        [torch.arange(m_train), torch.arange(ndata - m_train, ndata)], dim=0
    ) 

test_index = torch.cat(
    [torch.arange(ndata1 - m_test, ndata1), torch.arange(ndata1, ndata1 + m_test)],
    dim=0,
)


x_train, x_test = nodes_input[train_index, ...], nodes_input[test_index, ...]
aux_train = (
    node_mask[train_index, ...],
    nodes[train_index, ...],
    node_equal_weights[train_index, ...],
    directed_edges[train_index, ...],
    edge_gradient_weights[train_index, ...],
)
aux_test = (
    node_mask[test_index, ...],
    nodes[test_index, ...],
    node_equal_weights[test_index, ...],
    directed_edges[test_index, ...],
    edge_gradient_weights[test_index, ...],
)
feature_type = args.feature_type
if feature_type == "mach":
    feature_type_index = 1
    out_dim=1
elif feature_type == "pressure":
    feature_type_index = 0
    out_dim=1
elif feature_type == "both":
    feature_type_index = (0, 1)
    out_dim=2
y_train, y_test = (
    features[train_index, ...][...,feature_type_index],
    features[test_index, ...][...,feature_type_index],
)
print(
    f"x train:{x_train.shape}, y train:{y_train.shape}, x test:{x_test.shape}, y test:{y_test.shape}",
    flush=True,
)

###################################
# train
###################################
kx_max, ky_max = 32, 16
ndim = 2

if args.train_sp_L == 'False':
    args.train_sp_L = False
train_sp_L = args.train_sp_L

Lx, Ly = args.Lx, args.Ly
print("Lx, Ly = ", Lx, Ly)

modes = compute_Fourier_modes(ndim, [kx_max, ky_max], [Lx, Ly])
modes = torch.tensor(modes, dtype=torch.float).to(device)
model = PCNO(
    ndim,
    modes,
    nmeasures=1,
    layers=[128, 128, 128, 128, 128],
    fc_dim=128,
    in_dim=in_dim,
    out_dim=out_dim,
    train_sp_L=train_sp_L,
    act="gelu",
).to(device)

epochs = 500
base_lr = 0.001
scheduler = "OneCycleLR"
weight_decay = 1.0e-4
batch_size = 20
lr_ratio = args.lr_ratio

normalization_x = True
normalization_y = True
normalization_dim_x = []
normalization_dim_y = []
non_normalized_dim_x = 0
non_normalized_dim_y = 0


config = {
    "train": {
        "base_lr": base_lr,
        "lr_ratio": lr_ratio,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "scheduler": scheduler,
        "batch_size": batch_size,
        "normalization_x": normalization_x,
        "normalization_y": normalization_y,
        "normalization_dim_x": normalization_dim_x,
        "normalization_dim_y": normalization_dim_y,
        "non_normalized_dim_x": non_normalized_dim_x,
        "non_normalized_dim_y": non_normalized_dim_y,
    }
}


train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = PCNO_train(
    x_train,
    aux_train,
    y_train,
    x_test,
    aux_test,
    y_test,
    config,
    model,
    save_model_name=save_model_name,
)

# print("Generating Figure Data", flush=True)
# # model.train()
# with torch.no_grad():

#     n_train = 10
#     n_test = 10
#     m_train, m_test = n_train // 2, n_test // 2

#     train_index = torch.cat(
#         [torch.arange(m_train), torch.arange(ndata - m_train, ndata)], dim=0
#     ) 
#     test_index = torch.cat([torch.arange(ndata1 - m_test, ndata1),
#                         torch.arange(ndata1, ndata1 + m_test)], dim=0)


#     x_train, x_test = nodes_input[train_index, ...], nodes_input[test_index, ...]
#     aux_train = (node_mask[train_index, ...], nodes[train_index, ...], node_equal_weights[train_index, ...],
#                 directed_edges[train_index, ...], edge_gradient_weights[train_index, ...])
#     aux_test = (node_mask[test_index, ...], nodes[test_index, ...], node_equal_weights[test_index, ...],
#                 directed_edges[test_index, ...], edge_gradient_weights[test_index, ...])
#     y_train, y_test = features[train_index, :, :], features[test_index, :, :]
#     print(f"x train:{x_train.shape}, y train:{y_train.shape}, x test:{x_test.shape}, y test:{y_test.shape}", flush=True)


#     node_mask_train, nodes_train, node_weights_train, directed_edges_train, edge_gradient_weights_train = aux_train
#     x_train, node_mask_train, nodes_train, node_weights_train, directed_edges_train, edge_gradient_weights_train = x_train.to(device), node_mask_train.to(device), nodes_train.to(device), node_weights_train.to(device), directed_edges_train.to(device), edge_gradient_weights_train.to(device)
#     out_train = model(x_train, (node_mask_train, nodes_train, node_weights_train, directed_edges_train, edge_gradient_weights_train))
#     out_train=out_train * node_mask_train

#     node_mask_test, nodes_test, node_weights_test, directed_edges_test, edge_gradient_weights_test = aux_test
#     x_test, node_mask_test, nodes_test, node_weights_test, directed_edges_test, edge_gradient_weights_test = x_test.to(device), node_mask_test.to(device), nodes_test.to(device), node_weights_test.to(device), directed_edges_test.to(device), edge_gradient_weights_test.to(device)
#     out_test = model(x_test, (node_mask_test, nodes_test, node_weights_test, directed_edges_test, edge_gradient_weights_test))
#     out_test=out_test * node_mask_test


#     np.savez_compressed("./figs/" + save_model_name + "_fig.npz", 
#                         x_train=x_train.cpu().numpy(), 
#                         y_train=y_train.cpu().numpy(),
#                         out_train=out_train.cpu().numpy(), 
#                         node_mask_train=node_mask_train.cpu().numpy(),
#                         x_test=x_test.cpu().numpy(), 
#                         y_test=y_test.cpu().numpy(),
#                         out_test=out_test.cpu().numpy(),
#                         node_mask_test=node_mask_test.cpu().numpy(),
#                         )
