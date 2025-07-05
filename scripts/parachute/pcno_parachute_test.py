import os
import glob
import random
import torch
import sys
import numpy as np
import math
from timeit import default_timer
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


from pcno.geo_utility import preprocess_data, compute_node_weights
from pcno.pcno import compute_Fourier_modes, PCNO, PCNO_train

torch.set_printoptions(precision=16)


torch.manual_seed(0)
np.random.seed(0)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_data(data_path):
    ndata, nt = 2000, 51
    nodes_list, elems_list, features_list = [], [], []
    for i in range(ndata):    
        for mesh_type in ["DGB", "Arc", "Ribbon"]:
            nodes = np.load(data_path+mesh_type+"/nodes_%05d"%(i)+".npy")
            nodes_list.append(nodes)
            elems_list.append(np.load(data_path+mesh_type+"/elems_%05d"%(i)+".npy"))
            features = np.load(data_path+mesh_type+"/features_%05d"%(i)+".npy")
            nnodes = nodes.shape[0]

            displacement = features[:,:-1].reshape(nnodes, nt, 3)
            line_node_ind = features[:,[-1]]
            # keep displacements at t = 0.04, 0.08, 0.12, 0.16
            features_list.append(np.concatenate((line_node_ind, displacement[:, [10, 20, 30, 40] , :].reshape(nnodes, -1)), axis=-1))

    return nodes_list, elems_list, features_list 


try:
    PREPROCESS_DATA = sys.argv[1] == "preprocess_data" if len(sys.argv) > 1 else False
except IndexError:
    PREPROCESS_DATA = False


parser = argparse.ArgumentParser(description='Train model with different types.')
parser.add_argument('--equal_weight', type=str, default='False', help='Specify whether to use equal weight')
parser.add_argument('--train_inv_L_scale', type=str, default='False', choices=['False' , 'together' , 'independently'],
                    help='type of train_inv_L_scale (False, together, independently )')

parser.add_argument('--lr_ratio', type=float, default=10, help='lr ratio for independent training for L')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
if not PREPROCESS_DATA:
    args = parser.parse_args()
    args_dict = vars(args)
    for i, (key, value) in enumerate(args_dict.items()):
        print(f"{key}: {value}")
###################################
# load data
###################################
data_path = "../../data/parachute/parachute/"




if PREPROCESS_DATA:
    print("Loading data")
    nodes_list, elems_list, features_list  = load_data(data_path = data_path)
    
    print("Preprocessing data")
    nnodes, node_mask, nodes, node_measures_raw, features, directed_edges, edge_gradient_weights = preprocess_data(nodes_list, elems_list, features_list)
    node_measures, node_weights = compute_node_weights(nnodes,  node_measures_raw,  equal_measure = False)
    node_equal_measures, node_equal_weights = compute_node_weights(nnodes,  node_measures_raw,  equal_measure = True)
    np.savez_compressed(data_path+"pcno_data.npz", \
                        nnodes=nnodes, node_mask=node_mask, nodes=nodes, \
                        node_measures_raw = node_measures_raw, \
                        node_measures=node_measures, node_weights=node_weights, \
                        node_equal_measures=node_equal_measures, node_equal_weights=node_equal_weights, \
                        features=features, \
                        directed_edges=directed_edges, edge_gradient_weights=edge_gradient_weights) 
    exit()
else:
    # load data 
    
    equal_weights = args.equal_weight.lower() == "true"

    data = np.load(data_path+"pcno_data.npz")
    nnodes, node_mask, nodes = data["nnodes"], data["node_mask"], data["nodes"]
    node_measures = data["node_measures"]
    node_weights = data["node_equal_weights"] if equal_weights else data["node_weights"]
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
#!! compress measures
# node_weights = torch.sum(node_weights, dim=-1).unsqueeze(-1)

features = torch.from_numpy(features.astype(np.float32))
directed_edges = torch.from_numpy(directed_edges.astype(np.int64))
edge_gradient_weights = torch.from_numpy(edge_gradient_weights.astype(np.float32))

L_ref = 10.0
nodes_input = nodes.clone()/L_ref  # scale length input

n_train, n_test = 1000, 200


x_train, x_test = torch.cat((features[:n_train, :, [0]], nodes_input[:n_train, ...], node_rhos[:n_train, ...]), -1), torch.cat((features[-n_test:, :, [0]], nodes_input[-n_test:, ...], node_rhos[-n_test:, ...]),-1)
aux_train       = (node_mask[0:n_train,...], nodes[0:n_train,...], node_weights[0:n_train,...], directed_edges[0:n_train,...], edge_gradient_weights[0:n_train,...])
aux_test        = (node_mask[-n_test:,...],  nodes[-n_test:,...],  node_weights[-n_test:,...],  directed_edges[-n_test:,...],  edge_gradient_weights[-n_test:,...])
y_train, y_test = features[:n_train, :, 1:],     features[-n_test:, :, 1:]


k_max = 12
ndim = 3


#!! compress measures
modes = compute_Fourier_modes(ndim, [k_max,k_max,k_max, k_max,k_max,k_max], [9.0,9.0,12.0, 9.0,9.0,12.0])
# modes = compute_Fourier_modes(ndim, [k_max,k_max,k_max], [9.0,9.0,12.0])

modes = torch.tensor(modes, dtype=torch.float).to(device)

if args.train_inv_L_scale == 'False':
    args.train_inv_L_scale = False
train_inv_L_scale = args.train_inv_L_scale
#!! compress measures
model = PCNO(ndim, modes, nmeasures=2,
# model = PCNO(ndim, modes, nmeasures=1,
               layers=[128,128,128,128,128],
               fc_dim=128,
               in_dim=x_train.shape[-1], out_dim=y_train.shape[-1], 
               inv_L_scale_hyper = [train_inv_L_scale, 0.5, 2.0],
               act='gelu').to(device)



epochs = 500
base_lr = 5e-4 #0.001
lr_ratio = args.lr_ratio
scheduler = "OneCycleLR"
weight_decay = 1.0e-4
batch_size = args.batch_size

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
    x_train, aux_train, y_train, x_test, aux_test, y_test, config, model, save_model_name="./PCNO_parachute_model"
)
