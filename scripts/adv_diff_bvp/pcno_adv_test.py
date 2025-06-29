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
    """
    feature Format

    First column:     Force function, (p(x))
    Second column:    Solution, (u(x))
    Third columns:    Diffusion coefficient, (D)  
    Fourth columns:   Left boundary condition, (u_l)
    """

    ndata = 2500
    nodes_list, elems_list, features_list = [], [], []
    for i in range(ndata):    
        for mesh_type in ["uniform", "exponential", "linear"]:
            data = np.load(data_path+"/"+mesh_type+"/data_%05d"%(i)+".npy")
            nnodes = data.shape[0]
            elems = np.vstack((np.full(nnodes - 1, 1), np.arange(0, nnodes - 1), np.arange(1, nnodes))).T
            
            nodes_list.append(data[:,0:1])
            elems_list.append(elems)
            features_list.append(data[:,1:])
    return nodes_list, elems_list, features_list 
    
###################################
# load data
###################################

try:
    PREPROCESS_DATA = sys.argv[1] == "preprocess_data" if len(sys.argv) > 1 else False
except IndexError:
    PREPROCESS_DATA = False

parser = argparse.ArgumentParser(description='Train model with different configurations and options.')

parser.add_argument('--train_distribution', type=str, default='uniform', choices=['uniform', 'exponential', 'linear', 'mixed'],
                    help='distribution of training dataset (uniform, exponential, linear, mixed)')

parser.add_argument('--n_train', type=int, default=1000, choices=[500, 1000, 1500],
                    help='training datasize (500,1000,1500)')
parser.add_argument('--n_test', type=int, default=200, help='Number of testing samples')
parser.add_argument('--equal_weight', type=str, default='False', help='Specify whether to use equal weight')

parser.add_argument('--L', type=float, default=15.0, help='Initial value for the length scale parameter L')
parser.add_argument('--train_inv_L_scale', type=str, default='False', choices=['False' , 'together' , 'independently'],
                    help='type of train_inv_L_scale (False, together, independently )')

parser.add_argument('--lr_ratio', type=float, default=10, help='Learning rate ratio of main parameters and L parameters when train_inv_L_scale is set to `independently`')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')


data_path = "../../data/adv_diff_bvp/"

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
    args = parser.parse_args()
    equal_weights = args.equal_weight.lower() == "true"

    data = np.load(data_path+"pcno_data.npz")
    nnodes, node_mask, nodes = data["nnodes"], data["node_mask"], data["nodes"]
    node_weights = data["node_equal_weights"] if equal_weights else data["node_weights"]
    node_measures = data["node_measures"]
    directed_edges, edge_gradient_weights = data["directed_edges"], data["edge_gradient_weights"]
    features = data["features"]

    node_measures_raw = data["node_measures_raw"]
    indices = np.isfinite(node_measures_raw)
    node_rhos = np.copy(node_weights)
    node_rhos[indices] = node_rhos[indices]/node_measures[indices]






if args.train_inv_L_scale == 'False':
    args.train_inv_L_scale = False
train_distribution = args.train_distribution
n_train = args.n_train
n_test = args.n_test
train_inv_L_scale = args.train_inv_L_scale
print(f'train_distribution = {train_distribution}, n_train = {n_train}, train_inv_L_scale = {train_inv_L_scale}')


print("Casting to tensor")
indices_dict = {'uniform': np.arange(nodes.shape[0]) % 3 == 0,
            'exponential': np.arange(nodes.shape[0]) % 3 == 1,
              "linear": np.arange(nodes.shape[0]) % 3 == 2,
              'mixed': np.arange(nodes.shape[0])}

# normalize features
features /= np.array([1.0, 1.0, 0.01, 1.0])
nnodes = torch.from_numpy(nnodes[indices_dict[train_distribution]])
node_mask = torch.from_numpy(node_mask[indices_dict[train_distribution]])
nodes = torch.from_numpy(nodes[indices_dict[train_distribution]].astype(np.float32))
node_weights = torch.from_numpy(node_weights[indices_dict[train_distribution]].astype(np.float32))
node_rhos = torch.from_numpy(node_rhos.astype(np.float32))
features = torch.from_numpy(features[indices_dict[train_distribution]].astype(np.float32))
directed_edges = torch.from_numpy(directed_edges[indices_dict[train_distribution]].astype(np.int64))
edge_gradient_weights = torch.from_numpy(edge_gradient_weights[indices_dict[train_distribution]].astype(np.float32))

print(f'nodes.shape: {nodes.shape}, feature.shape: {features.shape}')


nodes_input = nodes.clone()

x_train, x_test = torch.cat((features[:n_train,:,[0,2,3]], nodes_input[:n_train,...], node_rhos[:n_train, ...]), -1), torch.cat((features[-n_test:,:,[0,2,3]],nodes_input[-n_test:,...], node_rhos[-n_test:, ...]), -1)
aux_train       = (node_mask[0:n_train,...], nodes[0:n_train,...], node_weights[0:n_train,...], directed_edges[0:n_train,...], edge_gradient_weights[0:n_train,...])
aux_test        = (node_mask[-n_test:,...],  nodes[-n_test:,...],  node_weights[-n_test:,...],  directed_edges[-n_test:,...],  edge_gradient_weights[-n_test:,...])

y_train, y_test = features[:n_train, :, [1]],     features[-n_test:, :, [1]]


print(f'x_train.shape: {x_train.shape}, y_train.shape: {y_train.shape}')
k_max = 64
ndim = 1


modes = compute_Fourier_modes(ndim, [k_max], [args.L])
modes = torch.tensor(modes, dtype=torch.float).to(device)
model = PCNO(ndim, modes, nmeasures=1,
               layers=[128,128,128,128,128],
               fc_dim=128,
               in_dim=x_train.shape[-1], out_dim=y_train.shape[-1],
               inv_L_scale_hyper = [train_inv_L_scale, 0.5, 2.0],
               act='gelu').to(device)



epochs = 500
base_lr = 0.001
lr_ratio = args.lr_ratio
scheduler = "OneCycleLR"
weight_decay = 1.0e-4
batch_size = args.batch_size

normalization_x = False
normalization_y = False
normalization_dim_x = []
normalization_dim_y = []
non_normalized_dim_x = 2
non_normalized_dim_y = 0


config = {"train" : {"base_lr": base_lr, 'lr_ratio': lr_ratio, "weight_decay": weight_decay, "epochs": epochs, "scheduler": scheduler,  "batch_size": batch_size, 
                     "normalization_x": normalization_x,"normalization_y": normalization_y, 
                     "normalization_dim_x": normalization_dim_x, "normalization_dim_y": normalization_dim_y, 
                     "non_normalized_dim_x": non_normalized_dim_x, "non_normalized_dim_y": non_normalized_dim_y}
                     }
print(f'Start training , train_inv_L_scale = {train_inv_L_scale}, lr_ratio = {lr_ratio}')

train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = PCNO_train(
    x_train, aux_train, y_train, x_test, aux_test, y_test, config, model, save_model_name=f"model/pcno_adv_{n_train}/{train_distribution}_{train_inv_L_scale}_{equal_weights}"
)





