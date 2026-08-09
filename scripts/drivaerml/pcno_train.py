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
from pcno.pcno import compute_Fourier_modes, PCNO, PCNO_train, PCNO_train_multidist
torch.set_printoptions(precision=16)


torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###################################
# load parameters
###################################

parser = argparse.ArgumentParser(description='Train model with different configurations and options.')

parser.add_argument('--to_divide_factor', type=float, default=20.0)
parser.add_argument('--k_max', type=int, default=12)
parser.add_argument('--bsz', type=int, default=8)
parser.add_argument('--ep', type=int, default=200)
parser.add_argument('--Ls', type=str, default="")
parser.add_argument('--n_train', type=int, default=400)
parser.add_argument('--n_test', type=int, default=80)
parser.add_argument('--act', type=str, default="gelu")
parser.add_argument("--layer_sizes", type=str, default="64,64,64,64")
parser.add_argument("--model_name", type=str, default="")
args = parser.parse_args()


train_inv_L_scale = False
k_max = args.k_max
ndim = 3
layers = [int(size) for size in args.layer_sizes.split(",")]
act = args.act
to_divide_factor = args.to_divide_factor

###################################
# load data
###################################
def load_data_to_torch(data_file_path, to_divide = None, factor = 1.0):
    '''
    returns:
        torch tensors:
            nnodes : int[ndata]
            node_mask : int[ndata, max_nnodes, 1]         
            nodes : float[ndata, max_nnodes, ndims]     
            node_measures : float[ndata, max_nnodes, nmeasures] 
            features : float[ndata, max_nnodes, nfeatures]  
            directed_edges :  float[ndata, max_nedges, 2]         
            edge_gradient_weights   :  float[ndata, max_nedges, ndims]    

    '''
    data = np.load(data_file_path)
    nnodes, node_mask, nodes = data["nnodes"], data["node_mask"], data["nodes"]
    print(f"Loaded {nodes.shape[0]} samples from {data_file_path}", flush = True)
    
    node_weights = data["node_measures"]
    if to_divide is None:
        to_divide = factor * np.amax(np.sum(node_weights, axis = 1))
    node_weights = node_weights/to_divide
    
    node_measures = data["node_measures"]
    directed_edges, edge_gradient_weights = data["directed_edges"], data["edge_gradient_weights"]
    features = data["features"]

    node_measures_raw = data["node_measures_raw"]
    indices = np.isfinite(node_measures_raw)
    node_rhos = np.copy(node_weights)
    node_rhos[indices] = node_rhos[indices]/node_measures[indices]

    nnodes = torch.from_numpy(nnodes)
    node_mask = torch.from_numpy(node_mask)
    nodes = torch.from_numpy(nodes.astype(np.float32))
    node_weights = torch.from_numpy(node_weights.astype(np.float32))
    node_rhos = torch.from_numpy(node_rhos.astype(np.float32))
    features = torch.from_numpy(features.astype(np.float32))
    directed_edges = torch.from_numpy(directed_edges.astype(np.int64))
    edge_gradient_weights = torch.from_numpy(edge_gradient_weights.astype(np.float32))

    # huang
    edge_gradient_weights /= 10

    return nnodes, node_mask, nodes, node_weights, node_rhos, features, directed_edges, edge_gradient_weights, to_divide


def gen_data_tensors(data_indices, nodes, features, node_mask, node_weights, directed_edges, edge_gradient_weights):
    """
        Generate and format tensors for a specific data batch.
        
        Parameters:
            data_indices : LongTensor[batch_size]
                Indices of the samples to be included in the batch.
            nodes : Tensor[ndata, max_nnodes, ndim]
            features : Tensor[ndata, max_nnodes, nfeatures]
            node_mask : Tensor[ndata, max_nnodes, 1]
            node_weights : Tensor[ndata, max_nnodes, nmeasures]
            directed_edges : Tensor[ndata, max_nedges, 2]
            edge_gradient_weights : Tensor[ndata, max_nedges, ndim]


        Returns:
            x : Tensor[batch_size, max_nnodes, in_dim]
                Input features, typically including raw features and coordinates.
            y : Tensor[batch_size, max_nnodes, out_dim]
                Target labels or ground truth fields.
            aux : tuple
                A collection of geometric and structural tensors:
                (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, outward_normals)
                where outward_normals (nx) is permuted to [batch_size, ndim, max_nnodes].
        """
    nodes_input = nodes.clone()
    # x是法向(3维)+点坐标(3维)
    normals = features[data_indices][...,:3]
    x = torch.cat((normals, nodes_input[data_indices, ...]), -1)
    # y是Cp即压力系数(1维)
    y = features[data_indices][...,-1:]
    aux = (node_mask[data_indices], nodes[data_indices], node_weights[data_indices], directed_edges[data_indices], edge_gradient_weights[data_indices])
    
    return x, y, aux

data_path = "../../data/hifi3d_processed/test"

n_train, n_test = args.n_train, args.n_test
data_file_path = data_path+f"/drivaerml_vertex_centered.npz"
nnodes, node_mask, nodes, node_weights, node_rhos, features, directed_edges, edge_gradient_weights, to_divide = load_data_to_torch(data_file_path, to_divide = None, factor = to_divide_factor)

if args.Ls:
    Ls = [float(L) for L in args.Ls.split(",")]
    if len(Ls) != ndim:
            raise ValueError(f"Expected {ndim} values in --Ls, got {Ls}")
else:
    lengths = torch.amax(nodes[..., 0:3], dim=(0, 1)) - torch.amin(nodes[..., 0:3], dim=(0, 1))
    Ls = [float(length.item())*2+0.2 for length in lengths]

seed = 0
if seed:
    ndata = nnodes.shape[0]
    rng = np.random.default_rng(seed)
    order = rng.permutation(ndata)
    train_idx = order[: n_train]
    test_idx = order[n_train :n_train + n_test]
    x_train, y_train, aux_train = gen_data_tensors(train_idx, nodes, features, node_mask, node_weights, directed_edges, edge_gradient_weights)
    x_test, y_test, aux_test = gen_data_tensors(test_idx, nodes, features, node_mask, node_weights, directed_edges, edge_gradient_weights)

else:
    x_train, y_train, aux_train = gen_data_tensors(np.arange(n_train), nodes, features, node_mask, node_weights, directed_edges, edge_gradient_weights)
    x_test, y_test, aux_test = gen_data_tensors(np.arange(-n_test, 0), nodes, features, node_mask, node_weights, directed_edges, edge_gradient_weights)

x_test_list, y_test_list, aux_test_list = [x_test], [y_test], [aux_test]
label_list = ['Default']

print(f'x_train shape {x_train.shape}, x_test shape {[x.shape for x in x_test_list]}, y_train shape {y_train.shape}, y_test shape {[y.shape for y in y_test_list]}', flush = True)
print('Domain range per dimension: ',torch.amax(nodes, dim = [0,1]) - torch.amin(nodes, dim = [0,1]), flush = True)


###################################
# load model and train
###################################

modes = compute_Fourier_modes(ndim, [k_max,k_max,k_max], Ls)

print('------Parameters------')
print(f'kmax = {k_max}')
print(f'n_train = {n_train}, n_test = {n_test}')
print(f'Ls = {Ls}')
print(f'Shape of Fourier modes: ', modes.shape)
print(f'layers = {layers}')
print(f'activation = {act}')


modes = torch.tensor(modes, dtype=torch.float).to(device)
model = PCNO(ndim, modes, nmeasures=1,
               layers=layers,
               fc_dim=128,
               in_dim=x_train.shape[-1], out_dim=y_train.shape[-1],
               inv_L_scale_hyper = [train_inv_L_scale, 0.5, 2.0],
               act = act,
            ).to(device)



epochs = args.ep
base_lr = 5e-4 #0.001
lr_ratio = 10
scheduler = "OneCycleLR"
weight_decay = 1.0e-4
batch_size = args.bsz
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


train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = PCNO_train(
    x_train, aux_train, y_train, x_test, aux_test, y_test, config, model,
     save_model_name = args.model_name if args.model_name else None,
)
