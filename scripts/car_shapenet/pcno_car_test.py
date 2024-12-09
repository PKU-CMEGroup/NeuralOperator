import os
import glob
import random
import torch
import sys
import numpy as np
import math
from timeit import default_timer
sys.path.append("../../")

from pcno.geo_utility import preprocess_data, compute_node_weights
from pcno.pcno import compute_Fourier_modes, PCNO, PCNO_train

torch.set_printoptions(precision=16)


torch.manual_seed(0)
np.random.seed(0)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_data(data_path = "../data/car_shapenet"):
    ndata = 611
    nodes_list, elems_list, features_list = [], [], []
    for i in range(ndata):    
        nodes_list.append(np.load(data_path+"/nodes_%05d"%(i)+".npy"))
        elems_list.append(np.load(data_path+"/elems_%05d"%(i)+".npy"))
        features_list.append(np.load(data_path+"/features_%05d"%(i)+".npy"))
    return nodes_list, elems_list, features_list 
###################################
# load data
###################################

CONVERT_DATA = True
if CONVERT_DATA:
    print("Loading data")
    data_path = "../../data/car_shapenet"
    nodes_list, elems_list, features_list  = load_data(data_path = data_path)

    print("Preprocessing data")

    nnodes, node_mask, nodes, node_measures, features, directed_edges, edge_gradient_weights = preprocess_data(nodes_list, elems_list, features_list)
    _, node_weights = compute_node_weights(nnodes,  node_measures,  equal_measure = False)
    node_equal_measures, node_equal_weights = compute_node_weights(nnodes,  node_measures,  equal_measure = True)
    np.savez_compressed("../../data/car_shapenet/pcno_triangle_data.npz", \
                        nnodes=nnodes, node_mask=node_mask, nodes=nodes, \
                        node_measures=node_measures, node_weights=node_weights, \
                        node_equal_measures=node_equal_measures, node_equal_weights=node_equal_weights, \
                        features=features, \
                        directed_edges=directed_edges, edge_gradient_weights=edge_gradient_weights) 
    exit()
else:
    # load data 
    equal_measure = True

    data = np.load("../../data/car_shapenet/pcno_triangle_data.npz")
    nnodes, node_mask, nodes = data["nnodes"], data["node_mask"], data["nodes"]
    if equal_measure:
        node_measures, node_weights = data["node_equal_measures"], data["node_equal_weights"]
    else:
        node_measures, node_weights = data["node_measures"], data["node_weights"]

    directed_edges, edge_gradient_weights = data["directed_edges"], data["edge_gradient_weights"]



print("Casting to tensor")
nnodes = torch.from_numpy(nnodes)
node_mask = torch.from_numpy(node_mask)
nodes = torch.from_numpy(nodes.astype(np.float32))
node_weights = torch.from_numpy(node_weights.astype(np.float32))
features = torch.from_numpy(features.astype(np.float32))
directed_edges = torch.from_numpy(directed_edges)
edge_gradient_weights = torch.from_numpy(edge_gradient_weights.astype(np.float32))

nodes_input = nodes.clone()

n_train, n_test = 500, 100

OUTPUT = "pressure" # "normal"  or "pressure"


x_train, x_test = nodes_input[:n_train,...], nodes_input[-n_test:,...]
aux_train       = (node_mask[0:n_train,...], nodes[0:n_train,...], node_weights[0:n_train,...], directed_edges[0:n_train,...], edge_gradient_weights[0:n_train,...])
aux_test        = (node_mask[-n_test:,...],  nodes[-n_test:,...],  node_weights[-n_test:,...],  directed_edges[-n_test:,...],  edge_gradient_weights[-n_test:,...])

if OUTPUT == "pressure":
    y_train, y_test = features[:n_train, :, 0:1],     features[-n_test:, :, 0:1]
else:  #OUTPUT == "normal":
    y_train, y_test = features[:n_train, :, 1:],     features[-n_test:, :, 1:]

k_max = 16
ndim = 3
modes = compute_Fourier_modes(ndim, [k_max,k_max,k_max], [2.0,2.0,5.0])
modes = torch.tensor(modes, dtype=torch.float).to(device)
model = PCNO(ndim, modes,
               layers=[128,128,128,128,128],
               fc_dim=128,
               in_dim=3, out_dim=y_train.shape[-1],
               act='gelu').to(device)



epochs = 1000
base_lr = 0.001
scheduler = "OneCycleLR"
weight_decay = 1.0e-4
batch_size=8

normalization_x = True
normalization_y = True
normalization_dim = []
non_normalized_dim_x = 0
non_normalized_dim_y = 0


config = {"train" : {"base_lr": base_lr, "weight_decay": weight_decay, "epochs": epochs, "scheduler": scheduler,  "batch_size": batch_size, 
                     "normalization_x": normalization_x,"normalization_y": normalization_y, "normalization_dim": normalization_dim, 
                     "non_normalized_dim_x": non_normalized_dim_x, "non_normalized_dim_y": non_normalized_dim_y}}


train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = PCNO_train(
    x_train, aux_train, y_train, x_test, aux_test, y_test, config, model, save_model_name="./PCNO_car_shapenet_model"
)





