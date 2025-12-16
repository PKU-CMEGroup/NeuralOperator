import os
import torch
import sys
import argparse

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from timeit import default_timer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from pcno.geo_utility import preprocess_data_mesh, compute_node_weights
from pcno.pcno_geo import compute_Fourier_modes, PCNO, PCNO_train
from pcno.modes_discretization import discrete_half_ball_modes
torch.set_printoptions(precision=16)


torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Train model with different configurations and options.')

parser.add_argument('--grad', type=str, default='True', choices=['True', 'False'])
parser.add_argument('--geo', type=str, default='False', choices=['True', 'False'])
parser.add_argument('--lap', type=str, default='False', choices=['True', 'False'])
parser.add_argument('--geo_dims', type=int, nargs='+', default=None)
parser.add_argument('--k_max', type=int, default=16)
parser.add_argument('--n_train', type=int, default=900)
parser.add_argument('--n_test', type=int, default=100)
parser.add_argument('--act', type=str, default="none")
parser.add_argument('--normal_prod', type=str, default='False', choices=['True', 'False'])
parser.add_argument('--kernel_type', type=str, default='sp_laplace', choices=['sp_laplace', 'dp_laplace', 'adjoint_dp_laplace', 'stokes', 'modified_dp_laplace', 'fredholm_laplace', 'exterior_laplace_neumann'])
parser.add_argument('--two_circles_test', type=str, default="False", choices=['True', 'False'])
parser.add_argument('--single_mixed', type=str, default="False", choices=['True', 'False'])
parser.add_argument('--two_circles_train', type=str, default="False", choices=['True', 'False'])
parser.add_argument("--layer_sizes", type=str, default="128,128")
args = parser.parse_args()

###################################
# load data
###################################
data_path = "../../data/curve"

# load data 
data_file_path = data_path+f"/pcno_curve_data_1_1_5_2d_{args.kernel_type}_panel"+("_single_mixed" if args.single_mixed == "True" else "")+("_two_circles" if args.two_circles_train == "True" else "")+".npz"
print("Loading train data from ", data_file_path, flush = True)
data = np.load(data_file_path)
nnodes, node_mask, nodes = data["nnodes"], data["node_mask"], data["nodes"]
print(nnodes.shape,node_mask.shape,nodes.shape,flush = True)
# node_weights = data["node_equal_weights"] if equal_weights else data["node_weights"]
node_weights = data["node_measures_raw"]
# print('use node_weight')
to_divide = np.amax(np.sum(node_weights, axis = 1))
node_weights = node_weights/to_divide
print('use normalized raw measures')
node_measures = data["node_measures"]
directed_edges, edge_gradient_weights = data["directed_edges"], data["edge_gradient_weights"]
features = data["features"]

node_measures_raw = data["node_measures_raw"]
indices = np.isfinite(node_measures_raw)
node_rhos = np.copy(node_weights)
node_rhos[indices] = node_rhos[indices]/node_measures[indices]
print("Two circles test:", args.two_circles_test, flush = True)

data_file_path2 = data_path+f"/pcno_curve_data_1_1_5_2d_{args.kernel_type}_panel"+("_two_circles" if args.two_circles_test == "True" else "")+".npz"
print("Loading test data from ", data_file_path2, flush = True)
data2 = np.load(data_file_path2)
nnodes2, node_mask2, nodes2 = data2["nnodes"], data2["node_mask"], data2["nodes"]
print(nnodes2.shape,node_mask2.shape,nodes2.shape,flush = True)
# node_weights = data["node_equal_weights"] if equal_weights else data["node_weights"]
node_weights2 = data2["node_measures_raw"]
# print('use node_weight')
node_weights2 = node_weights2/to_divide
print('use normalized raw measures')
node_measures2 = data2["node_measures"]
directed_edges2, edge_gradient_weights2 = data2["directed_edges"], data2["edge_gradient_weights"]
features2 = data2["features"]
node_measures_raw2 = data2["node_measures_raw"]
indices2 = np.isfinite(node_measures_raw2)
node_rhos2 = np.copy(node_weights2)
node_rhos2[indices2] = node_rhos2[indices2]/node_measures2[indices2]




layer_selection = {'grad': args.grad.lower() == "true", 'geo': args.geo.lower() == "true", 'lap': args.lap.lower() == "true"}
normal_prod = args.normal_prod.lower() == "true"

print("Casting to tensor",flush = True)
nnodes = torch.from_numpy(nnodes)
node_mask = torch.from_numpy(node_mask)
nodes = torch.from_numpy(nodes.astype(np.float32))
node_weights = torch.from_numpy(node_weights.astype(np.float32))
node_rhos = torch.from_numpy(node_rhos.astype(np.float32))
features = torch.from_numpy(features.astype(np.float32))
directed_edges = torch.from_numpy(directed_edges.astype(np.int64))
edge_gradient_weights = torch.from_numpy(edge_gradient_weights.astype(np.float32))

nnodes2 = torch.from_numpy(nnodes2)
node_mask2 = torch.from_numpy(node_mask2)
nodes2 = torch.from_numpy(nodes2.astype(np.float32))
node_weights2 = torch.from_numpy(node_weights2.astype(np.float32))
node_rhos2 = torch.from_numpy(node_rhos2.astype(np.float32))
features2 = torch.from_numpy(features2.astype(np.float32))
directed_edges2 = torch.from_numpy(directed_edges2.astype(np.int64))
edge_gradient_weights2 = torch.from_numpy(edge_gradient_weights2.astype(np.float32))


nodes_input = nodes.clone()
nodes_input2 = nodes2.clone()

n_train, n_test = args.n_train, args.n_test

f_in_dim = 2 if args.kernel_type in ['stokes'] else 1
f_out_dim = 2 if args.kernel_type in ['modified_dp_laplace','stokes'] else 1


aux_train       = (node_mask[0:n_train,...], nodes[0:n_train,...], node_weights[0:n_train,...], directed_edges[0:n_train,...], edge_gradient_weights[0:n_train,...])
aux_test        = (node_mask2[-n_test:,...],  nodes2[-n_test:,...],  node_weights2[-n_test:,...],  directed_edges2[-n_test:,...],  edge_gradient_weights2[-n_test:,...])

if normal_prod:
    x_train = torch.cat((features[:n_train, ...][...,:f_in_dim+2],
                        features[:n_train, ...][...,f_in_dim:f_in_dim+1]*features[:n_train, ...][...,:f_in_dim],
                        features[:n_train, ...][...,f_in_dim+1:f_in_dim+2]*features[:n_train, ...][...,:f_in_dim],
                            nodes_input[:n_train, ...], node_rhos[:n_train, ...]), -1)
    x_test  = torch.cat((features2[-n_test:, ...][...,:f_in_dim+2],
                        features2[-n_test:, ...][...,f_in_dim:f_in_dim+1]*features2[-n_test:, ...][...,:f_in_dim],
                        features2[-n_test:, ...][...,f_in_dim+1:f_in_dim+2]*features2[-n_test:, ...][...,:f_in_dim],
                            nodes_input2[-n_test:, ...], node_rhos2[-n_test:, ...]), -1)
else:
    x_train = torch.cat((features[:n_train, ...][...,:f_in_dim+2],
                          nodes_input[:n_train, ...], node_rhos[:n_train, ...]), -1)
    x_test  = torch.cat((features2[-n_test:, ...][...,:f_in_dim+2],
                          nodes_input2[-n_test:, ...], node_rhos2[-n_test:, ...]), -1)

y_train = features[:n_train, ...][...,-f_out_dim:]
y_test = features2[-n_test:, ...][...,-f_out_dim:]

print(f'x_train shape {x_train.shape}, y_train shape {y_train.shape}')
print('length of each dim: ',torch.amax(nodes_input, dim = [0,1]) - torch.amin(nodes_input, dim = [0,1]), flush = True)




##########################################
train_inv_L_scale = False
k_max = args.k_max
ndim = 2
L = 10
scale = 0
layers = [int(size) for size in args.layer_sizes.split(",")]
geo_dims = args.geo_dims if args.geo_dims is not None else [f_in_dim, f_in_dim+1, 3*f_in_dim+2, 3*f_in_dim+3] if normal_prod else [f_in_dim, f_in_dim+1, f_in_dim+2, f_in_dim+3]
act = args.act
###########################################




modes = compute_Fourier_modes(ndim, [k_max,k_max], [L,L])

def nonlinear_scale(modes, scale=1.0):
    norms = np.linalg.norm(modes, axis=1)
    max_norm = norms.max()
    scaled_norms = (norms / max_norm) ** scale
    modes_scaled = modes * scaled_norms[:, np.newaxis]
    return modes_scaled

modes = nonlinear_scale(modes, scale=scale)

print(f'kmax = {k_max}')
print(f'L = {L}')
print(f'use cube modes, scale = {scale}', modes.shape)
print(f'normal_prod = {normal_prod}')
print(f'geo_dims = {geo_dims}')
print(f'layer_selection = {layer_selection}')
print(f'layers = {layers}')
print(f'activation = {act}')


modes = torch.tensor(modes, dtype=torch.float).to(device)
model = PCNO(ndim, modes, nmeasures=1, geo_dims=geo_dims, 
               layer_selection = layer_selection,
               layers=layers,
               fc_dim=128,
               in_dim=x_train.shape[-1], out_dim=y_train.shape[-1],
               inv_L_scale_hyper = [train_inv_L_scale, 0.5, 2.0],
                act = act
               ).to(device)



epochs = 500
base_lr = 5e-4 #0.001
lr_ratio = 10
scheduler = "OneCycleLR"
weight_decay = 1.0e-4
batch_size = 8

normalization_x = False
normalization_y = True
normalization_dim_x = []
normalization_dim_y = []
non_normalized_dim_x = 4
non_normalized_dim_y = 0


config = {"train" : {"base_lr": base_lr, 'lr_ratio': lr_ratio, "weight_decay": weight_decay, "epochs": epochs, "scheduler": scheduler,  "batch_size": batch_size, 
                     "normalization_x": normalization_x,"normalization_y": normalization_y, 
                     "normalization_dim_x": normalization_dim_x, "normalization_dim_y": normalization_dim_y, 
                     "non_normalized_dim_x": non_normalized_dim_x, "non_normalized_dim_y": non_normalized_dim_y}
                     }


train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = PCNO_train(
    x_train, aux_train, y_train, x_test, aux_test, y_test, config, model,
     save_model_name = None,#"model/1_1_5_5_grad_deformed/k8_L10_normal_prod_layer2_geo2_softsign_new"
)
