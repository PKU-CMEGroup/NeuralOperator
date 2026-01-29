import os
import torch
import sys

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from timeit import default_timer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from pcno.geo_utility import preprocess_data_mesh, compute_node_weights
from pcno.pcno import compute_Fourier_modes, PCNO, PCNO_train
torch.set_printoptions(precision=16)


torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data(data_file_path):
    data = np.load(data_file_path)
    nodes_list, elems_list, features_list = data["nodes_list"], data["elems_list"], data["features_list"]

    return nodes_list, elems_list, features_list 


try:
    PREPROCESS_DATA = sys.argv[1] == "preprocess_data" if len(sys.argv) > 1 else False
except IndexError:
    PREPROCESS_DATA = False
###################################
# load data
###################################
data_path = "../../data/curve/"
if PREPROCESS_DATA:
    print("Loading data")
    nodes_list, elems_list, features_list  = load_data(data_file_path = data_path + "curve_data_3_3.npz")

    print("Preprocessing data")
    nnodes, node_mask, nodes, node_measures_raw, features, directed_edges, edge_gradient_weights = preprocess_data_mesh(nodes_list, elems_list, features_list, mesh_type = "vertex_centered", adjacent_type="edge")
    node_measures, node_weights = compute_node_weights(nnodes,  node_measures_raw,  equal_measure = False)
    node_equal_measures, node_equal_weights = compute_node_weights(nnodes,  node_measures_raw,  equal_measure = True)
    np.savez_compressed(data_path+"/pcno_curve_data_3_3.npz", \
                        nnodes=nnodes, node_mask=node_mask, nodes=nodes, \
                        node_measures_raw = node_measures_raw, \
                        node_measures=node_measures, node_weights=node_weights, \
                        node_equal_measures=node_equal_measures, node_equal_weights=node_equal_weights, \
                        features=features, \
                        directed_edges=directed_edges, edge_gradient_weights=edge_gradient_weights) 
    exit()
else:
    # load data 
    equal_weights = False
    data_file_path = data_path+"/pcno_curve_data_1_1_5_5_grad_deformed.npz"
    print("Loading data from ", data_file_path, flush = True)
    data = np.load(data_file_path)
    nnodes, node_mask, nodes = data["nnodes"], data["node_mask"], data["nodes"]
    print(nnodes.shape,node_mask.shape,nodes.shape,flush = True)
    ####################
    ##   measure 1
    print('use normalized raw measures')
    node_weights = data["node_measures_raw"]
    node_weights = node_weights/np.amax(np.sum(node_weights, axis = 1))
    ####################
    ##   measure 2
    # print('use L normalized measures')
    # _, node_weights = compute_unnormalized_node_measures(nnodes, data["node_measures_raw"], measure_dims=[1], Ls=[2.5,2.5])
    ####################

    node_measures = data["node_measures"]
    directed_edges, edge_gradient_weights = data["directed_edges"], data["edge_gradient_weights"]
    features = data["features"]

    node_measures_raw = data["node_measures_raw"]
    indices = np.isfinite(node_measures_raw)
    node_rhos = np.copy(node_weights)
    node_rhos[indices] = node_rhos[indices]/node_measures[indices]

print("Casting to tensor",flush = True)
nnodes = torch.from_numpy(nnodes)
node_mask = torch.from_numpy(node_mask)
nodes = torch.from_numpy(nodes.astype(np.float32))
node_weights = torch.from_numpy(node_weights.astype(np.float32))
node_rhos = torch.from_numpy(node_rhos.astype(np.float32))
features = torch.from_numpy(features.astype(np.float32))
directed_edges = torch.from_numpy(directed_edges.astype(np.int64))
edge_gradient_weights = torch.from_numpy(edge_gradient_weights.astype(np.float32))


nodes_input = nodes.clone()

n_train, n_test = 900, 100


x_train = torch.cat((features[:n_train, ...][...,[0,1,2]], features[:n_train, ...][...,[1,2]]*features[:n_train, ...][...,[0]], nodes_input[:n_train, ...], node_rhos[:n_train, ...]), -1)
x_test  = torch.cat((features[-n_test:, ...][...,[0,1,2]], features[-n_test:, ...][...,[1,2]]*features[-n_test:, ...][...,[0]], nodes_input[-n_test:, ...], node_rhos[-n_test:, ...]), -1)

y_train, y_test = (features[:n_train, ...][...,[3]], features[-n_test:, ...][...,[3]])

aux_train       = (node_mask[0:n_train,...], nodes[0:n_train,...], node_weights[0:n_train,...], directed_edges[0:n_train,...], edge_gradient_weights[0:n_train,...])
aux_test        = (node_mask[-n_test:,...],  nodes[-n_test:,...],  node_weights[-n_test:,...],  directed_edges[-n_test:,...],  edge_gradient_weights[-n_test:,...])

print(f'x_train shape {x_train.shape}, y_train shape {y_train.shape}')
print('length of each dim: ',torch.amax(nodes_input, dim = [0,1]) - torch.amin(nodes_input, dim = [0,1]), flush = True)
k_max = 8

print(f'kmax = {k_max}')
ndim = 2
train_inv_L_scale = False
L = 10
print("L = ", L)
ndim = 2


modes = compute_Fourier_modes(ndim, [k_max, k_max], [L, L])
modes = torch.tensor(modes, dtype=torch.float).to(device)


model = PCNO(ndim, modes, nmeasures=1,
               layers=[128,128,128,128,128],
               fc_dim=128,
               in_dim=x_train.shape[-1], out_dim=y_train.shape[-1],
               inv_L_scale_hyper = [train_inv_L_scale, 0.5, 2.0],
               act='gelu').to(device)

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
    x_train, aux_train, y_train, x_test, aux_test, y_test, config, model, save_model_name=None
)
