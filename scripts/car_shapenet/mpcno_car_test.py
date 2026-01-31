import os
import glob
import random
import torch
import sys
import numpy as np
import math
import argparse
from timeit import default_timer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from pcno.geo_utility import preprocess_data_mesh, compute_node_weights
from pcno.mpcno import compute_Fourier_modes, MPCNO, MPCNO_train

torch.set_printoptions(precision=16)


torch.manual_seed(0)
np.random.seed(0)

###################################
# load parameters
###################################

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_data(data_path):
    ndata = 611
    nodes_list, elems_list, features_list = [], [], []
    for i in range(ndata):    
        nodes_list.append(np.load(data_path+"/nodes_%05d"%(i)+".npy"))
        elems_list.append(np.load(data_path+"/elems_%05d"%(i)+".npy"))
        features_list.append(np.load(data_path+"/features_%05d"%(i)+".npy"))
    return nodes_list, elems_list, features_list 

parser = argparse.ArgumentParser(description='Train model with different configurations and options.')

parser.add_argument('--grad', type=str, default='True', choices=['True', 'False'])
parser.add_argument('--geo', type=str, default='True', choices=['True', 'False'])
parser.add_argument('--geointegral', type=str, default='True', choices=['True', 'False'])
parser.add_argument('--to_divide_factor', type=float, default=1.0)
parser.add_argument('--k_max', type=int, default=16)
parser.add_argument('--bsz', type=int, default=8)
parser.add_argument('--ep', type=int, default=500)
parser.add_argument('--act', type=str, default="gelu")
parser.add_argument('--geo_act', type=str, default="softsign")
parser.add_argument("--layer_sizes", type=str, default="64,64,64,64,64,64")
args = parser.parse_args()

layer_selection = {'grad': args.grad.lower() == "true", 'geo': args.geo.lower() == "true", 'geointegral': args.geointegral.lower() == "true"}

train_inv_L_scale = False
k_max = args.k_max
ndim = 2
Ls = [4.0,4.0,10.0]
layers = [int(size) for size in args.layer_sizes.split(",")]
act = args.act
geo_act = args.geo_act
to_divide_factor = args.to_divide_factor

try:
    PREPROCESS_DATA = sys.argv[1] == "preprocess_data" if len(sys.argv) > 1 else False
except IndexError:
    PREPROCESS_DATA = False

###################################
# load data
###################################
data_path = "../../data/car_shapenet/"

if PREPROCESS_DATA:
    print("Loading data")
    nodes_list, elems_list, features_list  = load_data(data_path = data_path)
    
    print("Preprocessing data")
    nnodes, node_mask, nodes, node_measures_raw, features, directed_edges, edge_gradient_weights = preprocess_data_mesh(nodes_list, elems_list, features_list, mesh_type="vertex_centered", adjacent_type="element")
    node_measures, node_weights = compute_node_weights(nnodes,  node_measures_raw,  equal_measure = False)
    node_equal_measures, node_equal_weights = compute_node_weights(nnodes,  node_measures_raw,  equal_measure = True)
    np.savez_compressed(data_path+"pcno_triangle_data.npz", \
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

    data = np.load(data_path+"pcno_triangle_data.npz")
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

n_train, n_test = 500, 111

OUTPUT = "pressure" # "normal"  or "pressure"

normals = np.load(data_path+"/car_shapenet_normals.npz")["normals"]
normals = torch.from_numpy(normals.astype(np.float32))


x_train, x_test = torch.cat((nodes_input[:n_train, ...], node_rhos[:n_train, ...]), -1), torch.cat((nodes_input[-n_test:, ...],  node_rhos[-n_test:, ...]),-1)

aux_train       = (node_mask[0:n_train,...], nodes[0:n_train,...], node_weights[0:n_train,...], directed_edges[0:n_train,...], edge_gradient_weights[0:n_train,...], normals[:n_train, ...].permute(0,2,1))
aux_test        = (node_mask[-n_test:,...],  nodes[-n_test:,...],  node_weights[-n_test:,...],  directed_edges[-n_test:,...],  edge_gradient_weights[-n_test:,...], normals[-n_test:, ...].permute(0,2,1))

if OUTPUT == "pressure":
    y_train, y_test = features[:n_train, :, 0:1],     features[-n_test:, :, 0:1]
else:  #OUTPUT == "normal":
    y_train, y_test = features[:n_train, :, 1:],     features[-n_test:, :, 1:]

###################################
# load model and train
###################################
ndim = 3
modes = compute_Fourier_modes(ndim, [k_max,k_max,k_max], Ls)
modes = torch.tensor(modes, dtype=torch.float).to(device)

print('------Parameters------')
print(f'kmax = {k_max}')
print(f'n_train = {n_train}, n_test = {n_test}')
print(f'Ls = {Ls}')
print(f'Shape of Fourier modes: ', modes.shape)
print(f'layer_selection = {layer_selection}')
print(f'layers = {layers}')
print(f'activation = {act}')
print(f'geo_activation = {geo_act}')

model = MPCNO(ndim, modes, nmeasures=1,
               layers=layers,
               layer_selection=layer_selection,
               fc_dim=128,
               in_dim=x_train.shape[-1], out_dim=y_train.shape[-1],
               inv_L_scale_hyper = [train_inv_L_scale, 0.5, 2.0],
               scaling_mode='sqrt_inv',
               act=act, geo_act=geo_act).to(device)



epochs = 500
base_lr = 5e-4 #0.001
lr_ratio = 10
scheduler = "OneCycleLR"
weight_decay = 1.0e-4
batch_size=args.bsz
print('batch_size', batch_size, '\n')

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


train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = MPCNO_train(
    x_train, aux_train, y_train, x_test, aux_test, y_test, config, model,
    save_model_name = None,
)
