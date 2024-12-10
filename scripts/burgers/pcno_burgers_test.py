import os
import glob
import random
import torch
import sys
import numpy as np
import math
from scipy.io import loadmat
sys.path.append("../../")

from pcno.geo_utility import preprocess_data, compute_node_weights
from pcno.pcno import compute_Fourier_modes, PCNO, PCNO_train

torch.set_printoptions(precision=16)


torch.manual_seed(0)
np.random.seed(0)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


###################################
# load data
###################################

try:
    PREPROCESS_DATA = sys.argv[1] == "preprocess_data" if len(sys.argv) > 1 else False
except IndexError:
    PREPROCESS_DATA = False


data_path = "../../data/burgers/"

if PREPROCESS_DATA:
    print("Loading data")
    data = loadmat(data_path+"burgers_data_R10.mat")
    ndata, nnodes_ref = data["a"].shape
    grid = np.linspace(0, 1, nnodes_ref)

    #downsample
    downsample_ratio = 4
    features = np.stack((data["a"], data["u"]), axis=2)[:,::downsample_ratio,:]
    grid = grid[::downsample_ratio, np.newaxis]
    nnodes = nnodes_ref//downsample_ratio
    elems = np.vstack((np.full(nnodes - 1, 1), np.arange(0, nnodes - 1), np.arange(1, nnodes))).T
            
    nodes_list, elems_list, features_list = [], [], []
    nodes_list = [grid for i in range(ndata)]
    elems_list = [elems for i in range(ndata)]
    features_list = [features[i,...] for i in range(ndata)]

    nnodes, node_mask, nodes, node_measures, features, directed_edges, edge_gradient_weights = preprocess_data(nodes_list, elems_list, features_list)
    _, node_weights = compute_node_weights(nnodes,  node_measures,  equal_measure = False)
    node_equal_measures, node_equal_weights = compute_node_weights(nnodes,  node_measures,  equal_measure = True)
    np.savez_compressed(data_path+"pcno_data.npz", \
                        nnodes=nnodes, node_mask=node_mask, nodes=nodes, \
                        node_measures=node_measures, node_weights=node_weights, \
                        node_equal_measures=node_equal_measures, node_equal_weights=node_equal_weights, \
                        features=features, \
                        directed_edges=directed_edges, edge_gradient_weights=edge_gradient_weights) 
    exit()
else:
    # load data 
    equal_weights = True

    data = np.load(data_path+"pcno_data.npz")
    nnodes, node_mask, nodes = data["nnodes"], data["node_mask"], data["nodes"]
    node_weights = data["node_equal_weights"] if equal_weights else data["node_weights"]
    directed_edges, edge_gradient_weights = data["directed_edges"], data["edge_gradient_weights"]
    features = data["features"]


print("Casting to tensor")
nnodes = torch.from_numpy(nnodes)
node_mask = torch.from_numpy(node_mask)
nodes = torch.from_numpy(nodes.astype(np.float32))
node_weights = torch.from_numpy(node_weights.astype(np.float32))
features = torch.from_numpy(features.astype(np.float32))
directed_edges = torch.from_numpy(directed_edges)
edge_gradient_weights = torch.from_numpy(edge_gradient_weights.astype(np.float32))

n_train, n_test = 1000, 200

nodes_input = nodes.clone()


x_train, x_test = torch.cat((features[:n_train,:,[0]], nodes_input[:n_train,...]), -1), torch.cat((features[-n_test:,:,[0]],nodes_input[-n_test:,...]), -1)
aux_train       = (node_mask[0:n_train,...], nodes[0:n_train,...], node_weights[0:n_train,...], directed_edges[0:n_train,...], edge_gradient_weights[0:n_train,...])
aux_test        = (node_mask[-n_test:,...],  nodes[-n_test:,...],  node_weights[-n_test:,...],  directed_edges[-n_test:,...],  edge_gradient_weights[-n_test:,...])


y_train, y_test = features[:n_train, :, [1]],     features[-n_test:, :, [1]]

k_max = 32
ndim = 1
modes = compute_Fourier_modes(ndim, [k_max], [1.0])
modes = torch.tensor(modes, dtype=torch.float).to(device)
model = PCNO(ndim, modes,
               layers=[128,128,128,128,128],
               fc_dim=128,
               in_dim=2, out_dim=y_train.shape[-1],
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
    x_train, aux_train, y_train, x_test, aux_test, y_test, config, model, save_model_name="./PCNO_burgers_model"
)





