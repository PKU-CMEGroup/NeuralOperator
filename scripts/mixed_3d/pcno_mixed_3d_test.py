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

torch.set_printoptions(precision=16)


torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


parser = argparse.ArgumentParser(description='Train model with different configurations and options.')

parser.add_argument('--grad', type=str, default='True', choices=['True', 'False'])
parser.add_argument('--geo', type=str, default='True', choices=['True', 'False'])
parser.add_argument('--geointegral', type=str, default='False', choices=['True', 'False'])

parser.add_argument('--num_grad', type=int, default=3)
parser.add_argument('--k_max', type=int, default=8)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--scale', type=float, default=0.0)
parser.add_argument('--layers', type=int, nargs='+', default=[128, 128, 128, 128])
parser.add_argument('--normal_prod', type=str, default='False', choices=['True', 'False'])
parser.add_argument('--to_divide_factor', type=float, default=1.0)

parser.add_argument('--preprocess_data', type=str, default='False', choices=['True', 'False'])

args = parser.parse_args()


def _load_data(file_path, nodes_list, elems_list, features_list):
    data = np.load(file_path)
    nodes_list.append(data["nodes_list"])
    elems = data["elems_list"]
    elems[:, 0] = 2
    elems_list.append(elems)
    features_list.append(data["features_list"])


def load_data(data_path, Plane_datasets, DrivAerNet_datasets, n_train_each, n_test_each):
    nodes_list, elems_list, features_list = [], [], []

    Plane_dir = os.path.join(data_path, "Plane")
    DrivAerNet_dir = os.path.join(data_path, "DrivAerNet")

    # load train data
    for subdir in Plane_datasets:
        for i in range(n_train_each):
            file_path = os.path.join(Plane_dir, subdir, "%04d"%(i+1)+".npz")
            _load_data(file_path, nodes_list, elems_list, features_list)

    for subdir in DrivAerNet_datasets:
        for i in range(n_train_each):
            file_path = os.path.join(DrivAerNet_dir, subdir, "%04d"%(i+1)+".npz")
            _load_data(file_path, nodes_list, elems_list, features_list)

    # load test data
    for subdir in Plane_datasets:
        for i in range(n_train_each, n_train_each + n_test_each):
            file_path = os.path.join(Plane_dir, subdir, "%04d"%(i+1)+".npz")
            _load_data(file_path, nodes_list, elems_list, features_list)

    for subdir in DrivAerNet_datasets:
        for i in range(n_train_each, n_train_each + n_test_each):
            file_path = os.path.join(DrivAerNet_dir, subdir, "%04d"%(i+1)+".npz")
            _load_data(file_path, nodes_list, elems_list, features_list)

    return nodes_list, elems_list, features_list 


PREPROCESS_DATA = args.preprocess_data.lower() == "true"


Plane_datasets = [
    "Airbus-Maveric",
    "boeing737",
    "erj",
    "J20",
    "P180"
]
DrivAerNet_datasets = [
    "E_S_WW_WM",
    "E_S_WWC_WM",
    "F_D_WM_WW",
    "F_S_WWC_WM",
    "F_S_WWS_WM",
    "N_S_WW_WM",
    "N_S_WWC_WM",
    "N_S_WWS_WM"
]
n_train_each, n_test_each = 90, 10

###################################
# load data
###################################
data_path = "../../data/mixed_3d"
if PREPROCESS_DATA:
    print("Loading data")
    print("Plane datasets: ", Plane_datasets)
    print("DrivAerNet datasets: ", DrivAerNet_datasets)
    nodes_list, elems_list, features_list  =  load_data(data_path, 
                                                        Plane_datasets,
                                                        DrivAerNet_datasets,
                                                        n_train_each, 
                                                        n_test_each)

    ndata = len(nodes_list)
    n_subdirs = ndata // (n_train_each + n_test_each)
    assert n_subdirs == (len(Plane_datasets) + len(DrivAerNet_datasets))
    
    n_train = n_subdirs * n_train_each
    n_test  = n_subdirs * n_test_each
    print(f"ndata: {ndata}, number of subdirs: {n_subdirs}, n_train: {n_train}, n_test: {n_test}", flush=True)
    
    print("Preprocessing data")
    nnodes, node_mask, nodes, node_measures_raw, features, directed_edges, edge_gradient_weights = preprocess_data_mesh(nodes_list, elems_list, features_list, mesh_type="vertex_centered", adjacent_type="edge")
    node_measures, node_weights = compute_node_weights(nnodes,  node_measures_raw,  equal_measure = False)
    node_equal_measures, node_equal_weights = compute_node_weights(nnodes,  node_measures_raw,  equal_measure = True)
    np.savez_compressed(data_path+"/pcno_mixed_3d.npz", \
                        nnodes=nnodes, node_mask=node_mask, nodes=nodes, \
                        node_measures_raw=node_measures_raw, \
                        node_measures=node_measures, node_weights=node_weights, \
                        node_equal_measures=node_equal_measures, node_equal_weights=node_equal_weights, \
                        features=features, \
                        directed_edges=directed_edges, edge_gradient_weights=edge_gradient_weights) 
    exit()
else:
    # load data 
    equal_weights = False
    data = np.load(data_path+"/pcno_mixed_3d.npz")
    nnodes, node_mask, nodes = data["nnodes"], data["node_mask"], data["nodes"]

    # node_weights = data["node_equal_weights"] if equal_weights else data["node_weights"]
    node_weights = data["node_weights"]
    to_divide_factor = args.to_divide_factor
    print('Node weights are devided by factor ', to_divide_factor)
    to_divide = to_divide_factor * np.amax(np.sum(node_weights, axis=1))
    node_weights = node_weights / to_divide

    node_measures = data["node_measures"]
    directed_edges, edge_gradient_weights = data["directed_edges"], data["edge_gradient_weights"]
    features = data["features"]

    node_measures_raw = data["node_measures_raw"]
    indices = np.isfinite(node_measures_raw)
    node_rhos = np.copy(node_weights)
    node_rhos[indices] = node_rhos[indices] / node_measures[indices]

print(args)

layer_selection = {'grad': args.grad.lower() == "true", 'geo': args.geo.lower() == "true", 'geointegral': args.geointegral.lower() == "true"}
normal_prod = args.normal_prod.lower() == "true"


print("Casting to tensor", flush=True)
nnodes = torch.from_numpy(nnodes)
node_mask = torch.from_numpy(node_mask)
nodes = torch.from_numpy(nodes.astype(np.float32))
node_weights = torch.from_numpy(node_weights.astype(np.float32))
node_rhos = torch.from_numpy(node_rhos.astype(np.float32))
features = torch.from_numpy(features.astype(np.float32))
directed_edges = torch.from_numpy(directed_edges.astype(np.int64))
edge_gradient_weights = torch.from_numpy(edge_gradient_weights.astype(np.float32))


nodes_input = nodes.clone()


ndata = nodes_input.shape[0]
n_subdirs = ndata // (n_train_each + n_test_each)
assert n_subdirs == (len(Plane_datasets) + len(DrivAerNet_datasets))

n_train = n_subdirs * n_train_each
n_test  = n_subdirs * n_test_each
print(f"ndata: {ndata}, n_subdirs: {n_subdirs}, n_train: {n_train}, n_test: {n_test}", flush=True)


x_train, x_test = torch.cat((nodes_input[:n_train, ...], node_rhos[:n_train, ...]), -1), torch.cat((nodes_input[-n_test:, ...], node_rhos[-n_test:, ...]),-1)

aux_train       = (node_mask[0:n_train,...], nodes[0:n_train,...], node_weights[0:n_train,...], directed_edges[0:n_train,...], edge_gradient_weights[0:n_train,...])
aux_test        = (node_mask[-n_test:,...],  nodes[-n_test:,...],  node_weights[-n_test:,...],  directed_edges[-n_test:,...],  edge_gradient_weights[-n_test:,...])

y_train, y_test = features[:n_train, :, :1],     features[-n_test:, :, :1]


geo_dims = [0, 1, 2, 3]
train_inv_L_scale = False
k_max = args.k_max
ndim = 3
Ls = [2.0, 2.0, 2.0]
scale = args.scale
layers = args.layers
num_grad =  args.num_grad


modes = compute_Fourier_modes(ndim, [k_max, k_max, k_max], Ls)
modes = torch.tensor(modes, dtype=torch.float).to(device)
model = PCNO(ndim, modes, nmeasures=1, geo_dims=geo_dims, 
               layer_selection=layer_selection,
               num_grad=num_grad,
               layers=layers,
               fc_dim=128,
               in_dim=x_train.shape[-1], out_dim=y_train.shape[-1],
               inv_L_scale_hyper=[train_inv_L_scale, 0.5, 2.0],
               act='gelu',
               ).to(device)


print(f'kmax = {k_max}')
print(f'Ls = {Ls}')
print(f'n_train = {n_train}, n_test = {n_test}')

print(f'use cube modes, scale = {scale}', modes.shape)
print(f'normal_prod = {normal_prod}')
print(f'geo_dims = {geo_dims}')
print(f'num_grad = {num_grad}')
print(f'layer_selection = {layer_selection}')
print(f'layers = {layers}')

epochs = args.epochs
# base_lr = 5e-4
base_lr = 1e-3
lr_ratio = 10
scheduler = "OneCycleLR"
weight_decay = 1.0e-4
batch_size = args.batch_size
print(f'batch_size = {batch_size}')

normalization_x = False
normalization_y = True
# normalization_y = False

# y_train = torch.atan(y_train)
# y_test = torch.atan(y_test)

print(f'normalization_x = {normalization_x}, normalization_y = {normalization_y}')

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
    x_train, aux_train, y_train, x_test, aux_test, y_test, config, model, save_model_name="./PCNO_mixed_3d_model"
)
