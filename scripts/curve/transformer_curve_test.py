import os
import torch
import sys
import argparse

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from timeit import default_timer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from baselines.transformer import Transformer, Transformer_train_multidist
torch.set_printoptions(precision=16)



torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###################################
# load parameters
###################################

parser = argparse.ArgumentParser(description='Train model with different configurations and options.')


parser.add_argument('--bsz', type=int, default=32)
parser.add_argument('--ep', type=int, default=500)
parser.add_argument('--n_train', type=int, default=2000)
parser.add_argument('--n_test', type=int, default=1000)
parser.add_argument("--layer_sizes", type=str, default="64,64,64,64,64,64")
parser.add_argument('--n_two_circles_test', type=int, default=0)
parser.add_argument('--kernel_type', type=str, default='sp_laplace', choices=['sp_laplace', 'dp_laplace', 'adjoint_dp_laplace', 'stokes', 'modified_dp_laplace',
                                                                               'exterior_laplace_neumann','weighted_sp_laplace','weighted_dp_laplace'])
args = parser.parse_args()

f_in_dim = 2 if args.kernel_type in ['stokes'] else 1
f_out_dim = 2 if args.kernel_type in ['modified_dp_laplace','stokes'] else 1
ndim = 2
layers = [int(size) for size in args.layer_sizes.split(",")]


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
    node_weights = data["node_measures_raw"]
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
    x = torch.cat((features[data_indices][...,:f_in_dim+2],
                            nodes_input[data_indices, ...]), -1)
    y = features[data_indices][...,-f_out_dim:]
    aux = (node_mask[data_indices], nodes[data_indices])
    
    return x, y, aux

data_path = "../../data/curve/"

n_train, n_test, n_two_circles_test = args.n_train, args.n_test, args.n_two_circles_test
data_file_path = data_path+f"/pcno_curve_data_1_1_5_2d_{args.kernel_type}_panel.npz"
nnodes, node_mask, nodes, node_weights, node_rhos, features, directed_edges, edge_gradient_weights, to_divide = load_data_to_torch(data_file_path, to_divide = None)

x_train, y_train, aux_train = gen_data_tensors(np.arange(n_train), nodes, features, node_mask, node_weights, directed_edges, edge_gradient_weights)
x_test, y_test, aux_test = gen_data_tensors(np.arange(-n_test, 0), nodes, features, node_mask, node_weights, directed_edges, edge_gradient_weights)
x_test_list, y_test_list, aux_test_list = [x_test], [y_test], [aux_test]
label_list = ['Default']

if n_two_circles_test > 0:
    data_file_path = data_path+f"/pcno_curve_data_1_1_5_2d_{args.kernel_type}_panel_two_circles.npz"

    nnodes2, node_mask2, nodes2, node_weights2, node_rhos2, features2, directed_edges2, edge_gradient_weights2, _ = load_data_to_torch(data_file_path, to_divide = to_divide)
    x_two_circles_test, y_two_circles_test, aux_two_circles_test = gen_data_tensors(np.arange(n_two_circles_test),nodes2, features2, node_mask2, node_weights2, directed_edges2, edge_gradient_weights2)
    x_test_list.append(x_two_circles_test)
    y_test_list.append(y_two_circles_test)
    aux_test_list.append(aux_two_circles_test)
    label_list.append('Two Circles')

print(f'x_train shape {x_train.shape}, x_test shape {[x.shape for x in x_test_list]}, y_train shape {y_train.shape}, y_test shape {[y.shape for y in y_test_list]}', flush = True)
print('Domain range per dimension: ',torch.amax(nodes, dim = [0,1]) - torch.amin(nodes, dim = [0,1]), flush = True)



###################################
# load model and train
###################################


print('------Parameters------')
print(f'n_train = {n_train}, n_test = {n_test}')
print(f'layers = {layers}')


model = Transformer(coord_dim=ndim, 
               in_channels=x_train.shape[-1], out_channels=y_train.shape[-1],
               d_model = layers[0],
               dim_feedforward=layers[0]*4,
                ).to(device)



epochs = args.ep
base_lr = 5e-4 #0.001
scheduler = "OneCycleLR"
weight_decay = 1.0e-4
batch_size = args.bsz
print('base_lr', base_lr, '\n')
print('batch_size', batch_size, '\n')

normalization_x = False
normalization_y = True
normalization_dim_x = []
normalization_dim_y = []
non_normalized_dim_x = 4
non_normalized_dim_y = 0


config = {"train" : {"base_lr": base_lr, "weight_decay": weight_decay, "epochs": epochs, "scheduler": scheduler,  "batch_size": batch_size, 
                     "normalization_x": normalization_x,"normalization_y": normalization_y, 
                     "normalization_dim_x": normalization_dim_x, "normalization_dim_y": normalization_dim_y, 
                     "non_normalized_dim_x": non_normalized_dim_x, "non_normalized_dim_y": non_normalized_dim_y}
                     }


train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = Transformer_train_multidist(
    x_train, aux_train, y_train, x_test_list, aux_test_list, y_test_list, config, model, label_list, save_model_name = None,
)
