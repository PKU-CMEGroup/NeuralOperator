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
from pcno.mpcno import compute_Fourier_modes, MPCNO, MPCNO_train_multidist
torch.set_printoptions(precision=16)


torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###################################
# load parameters
###################################

parser = argparse.ArgumentParser(description='Train model with different configurations and options.')

parser.add_argument('--grad', type=str, default='True', choices=['True', 'False'])
parser.add_argument('--geo', type=str, default='True', choices=['True', 'False'])
parser.add_argument('--geointegral', type=str, default='True', choices=['True', 'False'])
parser.add_argument('--to_divide_factor', type=float, default=20.0)
parser.add_argument('--bsz', type=int, default=32)
parser.add_argument('--ep', type=int, default=500)
parser.add_argument('--k_max', type=int, default=8)
parser.add_argument('--n_train', type=int, default=2000)
parser.add_argument('--n_test', type=int, default=1000)
parser.add_argument('--n_two_circles_test', type=int, default=0)
parser.add_argument('--act', type=str, default="gelu")
parser.add_argument('--geo_act', type=str, default="softsign")
parser.add_argument("--layer_sizes", type=str, default="64,64,64,64,64,64")
parser.add_argument('--kernel_type', type=str, default="panel_method")
# 新增读取输入文件路径之参数
parser.add_argument('--data_path', type=str)
parser.add_argument('--two_circles_data_path', type=str)
args = parser.parse_args()

layer_selection = {'grad': args.grad.lower() == "true", 'geo': args.geo.lower() == "true", 'geointegral': args.geointegral.lower() == "true"}
f_in_dim = 0
f_out_dim = 1
train_inv_L_scale = False
ndim = 2
L = 10
layers = [int(size) for size in args.layer_sizes.split(",")]
act = args.act
k_max = args.k_max
geo_act = args.geo_act
to_divide_factor = args.to_divide_factor

###################################
# load data
###################################
def load_data_to_torch(data_file_path, to_divide=None, factor=1.0):
    """
    与原始 mpcno_curve_test.py 的 load_data_to_torch 相同，
    额外读取 betas 字段。

    Returns:
        nnodes, node_mask, nodes, node_weights, node_rhos,
        features, directed_edges, edge_gradient_weights,
        betas  [ndata,],
        to_divide
    """
    data = np.load(data_file_path)
    nnodes            = data['nnodes']
    node_mask         = data['node_mask']
    nodes             = data['nodes']
    node_measures_raw = data['node_measures_raw']

    print(f'Loaded {nodes.shape[0]} samples from {data_file_path}', flush=True)

    if to_divide is None:
        to_divide = factor * np.amax(np.sum(node_measures_raw, axis=1))
    node_weights = node_measures_raw / to_divide

    node_measures         = data['node_measures']
    directed_edges        = data['directed_edges']
    edge_gradient_weights = data['edge_gradient_weights']
    features              = data['features']

    indices   = np.isfinite(node_measures_raw)
    node_rhos = np.copy(node_weights)
    node_rhos[indices] = node_rhos[indices] / node_measures[indices]

    betas = data['betas'].astype(np.float32)   # [ndata,]

    nnodes                = torch.from_numpy(nnodes)
    node_mask             = torch.from_numpy(node_mask)
    nodes                 = torch.from_numpy(nodes.astype(np.float32))
    node_weights          = torch.from_numpy(node_weights.astype(np.float32))
    node_rhos             = torch.from_numpy(node_rhos.astype(np.float32))
    features              = torch.from_numpy(features.astype(np.float32))
    directed_edges        = torch.from_numpy(directed_edges.astype(np.int64))
    edge_gradient_weights = torch.from_numpy(edge_gradient_weights.astype(np.float32))
    betas                 = torch.from_numpy(betas)

    return (nnodes, node_mask, nodes, node_weights, node_rhos,
            features, directed_edges, edge_gradient_weights,
            betas, to_divide)

def gen_data_tensors(data_indices,
                     nodes, features, node_mask, node_weights,
                     directed_edges, edge_gradient_weights, betas):    
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

    

    y   = features[data_indices][..., -f_out_dim:]
    nx  = features[data_indices][..., f_in_dim:f_in_dim + 2]
    aux = (
        node_mask[data_indices],
        nodes[data_indices],
        node_weights[data_indices],
        directed_edges[data_indices],
        edge_gradient_weights[data_indices],
        nx.permute(0, 2, 1),    # [B, 2, N]
    )
    beta_batch = betas[data_indices].unsqueeze(-1)   # [B, 1]

    M = features.shape[1]

    beta_expanded = beta_batch.unsqueeze(1).expand(-1, M, -1)  # shape: [B, M, 1]

    x = torch.cat([
        # features[data_indices][..., :f_in_dim],   # f
        beta_expanded,
        features[data_indices][..., f_in_dim:f_in_dim+2],
        nodes_input[data_indices],                    # coordinates
    ], dim=-1)

    return x, y, aux

data_base_path = "../../data/curve_beta/"

n_train, n_test, n_two_circles_test = args.n_train, args.n_test, args.n_two_circles_test
data_file_path = data_base_path + args.data_path
nnodes, node_mask, nodes, node_weights, node_rhos, features, directed_edges, edge_gradient_weights, betas, to_divide = load_data_to_torch(data_file_path, to_divide = None, factor = to_divide_factor)

x_train, y_train, aux_train = gen_data_tensors(np.arange(n_train), nodes, features, node_mask, node_weights, directed_edges, edge_gradient_weights, betas)
x_test, y_test, aux_test = gen_data_tensors(np.arange(-n_test, 0), nodes, features, node_mask, node_weights, directed_edges, edge_gradient_weights, betas)
x_test_list, y_test_list, aux_test_list = [x_test], [y_test], [aux_test]
label_list = ['Default']

if n_two_circles_test > 0:
    data_file_path = data_base_path + args.two_circles_data_path

    nnodes2, node_mask2, nodes2, node_weights2, node_rhos2, features2, directed_edges2, edge_gradient_weights2, betas, _ = load_data_to_torch(data_file_path, to_divide = to_divide)
    x_two_circles_test, y_two_circles_test, aux_two_circles_test = gen_data_tensors(np.arange(n_two_circles_test),nodes2, features2, node_mask2, node_weights2, directed_edges2, edge_gradient_weights2, betas)
    x_test_list.append(x_two_circles_test)
    y_test_list.append(y_two_circles_test)
    aux_test_list.append(aux_two_circles_test)
    label_list.append('Two Circles')

print(f'x_train shape {x_train.shape}, x_test shape {[x.shape for x in x_test_list]}, y_train shape {y_train.shape}, y_test shape {[y.shape for y in y_test_list]}', flush = True)
print('Domain range per dimension: ',torch.amax(nodes, dim = [0,1]) - torch.amin(nodes, dim = [0,1]), flush = True)



###################################
# load model and train
###################################

modes = compute_Fourier_modes(ndim, [k_max,k_max], [L,L])

print('------Parameters------')
print(f'kmax = {k_max}')
print(f'n_train = {n_train}, n_test = {n_test}')
print(f'L = {L}')
print(f'Shape of Fourier modes: ', modes.shape)
print(f'layer_selection = {layer_selection}')
print(f'layers = {layers}')
print(f'activation = {act}')
print(f'geo_activation = {geo_act}')


modes = torch.tensor(modes, dtype=torch.float).to(device)
model = MPCNO(ndim, modes, nmeasures=1,
               layer_selection = layer_selection,
               layers=layers,
               fc_dim=128,
               in_dim=x_train.shape[-1], out_dim=y_train.shape[-1],
               inv_L_scale_hyper = [train_inv_L_scale, 0.5, 2.0],
               scaling_mode='inv',
               act = act,
               geo_act = geo_act,
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


train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = MPCNO_train_multidist(
    x_train, aux_train, y_train, x_test_list, aux_test_list, y_test_list, config, model, label_test_list=label_list,
     save_model_name = None,
)