import os
import glob
import random
import torch
import sys
import numpy as np
import math
from timeit import default_timer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from pcno.geo_utility import preprocess_data_mesh, compute_node_weights
from pcno.pcno import compute_Fourier_modes, PCNO, PCNO_train, euler2d_PCNO

torch.set_printoptions(precision=16)


torch.manual_seed(0)
np.random.seed(0)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_data(data_path):
    ndata = 100
    nodes_list, elems_list, features_list = [], [], []
    for i in range(ndata):    
        nodes_list.append(np.load(data_path+str(i)+"/points.npy"))
        elems_list.append(np.load(data_path+str(i)+"/elems.npy"))
        features_list.append(np.load(data_path+str(i)+"/features.npy"))
    return nodes_list, elems_list, features_list 



try:
    PREPROCESS_DATA = sys.argv[1] == "preprocess_data" if len(sys.argv) > 1 else False
except IndexError:
    PREPROCESS_DATA = False

###################################
# load data
###################################
data_path = "/root/autodl-tmp/data/Euler_eq_2d/npy_forward_300/"

if PREPROCESS_DATA:
    print("Loading data")
    nodes_list, elems_list, features_list  = load_data(data_path = data_path)
    
    print("Preprocessing data")
    nnodes, node_mask, nodes, node_measures_raw, features, directed_edges, edge_gradient_weights = preprocess_data_mesh(nodes_list, elems_list, features_list, mesh_type="vertex_centered", adjacent_type="element")
    node_measures, node_weights = compute_node_weights(nnodes,  node_measures_raw,  equal_measure = False)
    node_equal_measures, node_equal_weights = compute_node_weights(nnodes,  node_measures_raw,  equal_measure = True)
    np.savez_compressed(data_path+"pcno_Euler_forward_data.npz", \
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

    data = np.load(data_path+"pcno_Euler_forward_data.npz")
    nnodes, node_mask, nodes = data["nnodes"], data["node_mask"], data["nodes"]
    node_weights = data["node_equal_weights"] if equal_weights else data["node_weights"]
    node_measures = data["node_measures"]
    directed_edges, edge_gradient_weights = data["directed_edges"], data["edge_gradient_weights"]
    features = data["features"]

    node_measures_raw = data["node_measures_raw"]
    indices = np.isfinite(node_measures_raw)
    node_rhos = np.copy(node_weights)
    node_rhos[indices] = node_rhos[indices]/node_measures[indices]


print("Casting to tensor")
nnodes = torch.from_numpy(nnodes)
node_mask = torch.from_numpy(node_mask)
nodes = torch.from_numpy(nodes.astype(np.float32))
node_weights = torch.from_numpy(node_weights.astype(np.float32))
node_rhos = torch.from_numpy(node_rhos.astype(np.float32))
features = torch.from_numpy(features.astype(np.float32))
directed_edges = torch.from_numpy(directed_edges.astype(np.int64))
edge_gradient_weights = torch.from_numpy(edge_gradient_weights.astype(np.float32))

nodes_input = nodes.clone()

n_train, n_test = 40, 5


n_time = 60

features_x_train = features[:n_train, :n_time, ...].reshape(n_train * n_time, features.shape[2], features.shape[3])
features_x_test = features[-n_test:, :n_time, ...].reshape(n_test * n_time, features.shape[2], features.shape[3])
features_y_train = features[:n_train, 1:n_time + 1, ...].reshape(n_train * n_time, features.shape[2], features.shape[3])
features_y_test = features[-n_test:, 1:n_time + 1, ...].reshape(n_test * n_time, features.shape[2], features.shape[3])
nodes_train = nodes_input[:n_train, ...].repeat_interleave(n_time, dim=0)
nodes_test = nodes_input[-n_test:, ...].repeat_interleave(n_time, dim=0)
node_rhos_train = node_rhos[:n_train, ...].repeat_interleave(n_time, dim=0)
node_rhos_test = node_rhos[-n_test:, ...].repeat_interleave(n_time, dim=0)

x_train = torch.cat((nodes_train, node_rhos_train, features_x_train), -1)
x_test = torch.cat((nodes_test, node_rhos_test, features_x_test), -1)
y_train, y_test = features_y_train, features_y_test

aux_train = (
    node_mask[:n_train, ...].repeat_interleave(n_time, dim=0),
    nodes[:n_train, ...].repeat_interleave(n_time, dim=0),
    node_weights[:n_train, ...].repeat_interleave(n_time, dim=0),
    directed_edges[:n_train, ...].repeat_interleave(n_time, dim=0),
    edge_gradient_weights[:n_train, ...].repeat_interleave(n_time, dim=0),
)
aux_test = (
    node_mask[-n_test:, ...].repeat_interleave(n_time, dim=0),
    nodes[-n_test:, ...].repeat_interleave(n_time, dim=0),
    node_weights[-n_test:, ...].repeat_interleave(n_time, dim=0),
    directed_edges[-n_test:, ...].repeat_interleave(n_time, dim=0),
    edge_gradient_weights[-n_test:, ...].repeat_interleave(n_time, dim=0),
)


k_max = 12
ndim = nodes_input.shape[-1]
nmeasures = node_weights.shape[-1]
modes = compute_Fourier_modes(ndim, [k_max] * (ndim * nmeasures), [6.0,2.0][:ndim] * nmeasures)
modes = torch.tensor(modes, dtype=torch.float).to(device)
model = euler2d_PCNO(ndim, modes, nmeasures=nmeasures,
        #PCNO(ndim, modes, nmeasures=nmeasures,
               layers=[128,128,128,128,128],
               fc_dim=128,
               in_dim=x_train.shape[-1], out_dim=y_train.shape[-1],
               act='gelu').to(device)



epochs = 500
base_lr = 5e-4 #0.001
lr_ratio = 10
scheduler = "OneCycleLR"
weight_decay = 1.0e-4
batch_size=8

normalization_x = False
normalization_y = False
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
    x_train, aux_train, y_train, x_test, aux_test, y_test, config, model, save_model_name="./PCNO_forward_euler_exp_model"
)