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
from pcno.pcno_geo_scaled import compute_Fourier_modes, PCNO, PCNO_train, PCNO_train_multidist
from pcno.modes_discretization import discrete_half_ball_modes
torch.set_printoptions(precision=16)


torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###################################
# load parameters
###################################

parser = argparse.ArgumentParser(description='Train model with different configurations and options.')

parser.add_argument('--grad', type=str, default='True', choices=['True', 'False'])
parser.add_argument('--geo', type=str, default='False', choices=['True', 'False'])
parser.add_argument('--geograd', type=str, default='False', choices=['True', 'False'])
parser.add_argument('--lap', type=str, default='False', choices=['True', 'False'])
parser.add_argument('--geo_dims', type=int, nargs='+', default=None)
parser.add_argument('--num_grad', type=int, default=3)
parser.add_argument('--k_max', type=int, default=16)
parser.add_argument('--bsz', type=int, default=128)
parser.add_argument('--ep', type=int, default=500)
parser.add_argument('--n_train', type=int, default=900)
parser.add_argument('--n_test', type=int, default=100)
parser.add_argument('--act', type=str, default="none")
parser.add_argument('--scale', type=float, default=0.0)
parser.add_argument("--layer_sizes", type=str, default="128,128")
parser.add_argument('--normal_prod', type=str, default='False', choices=['True', 'False'])
parser.add_argument('--n_two_circles_test', type=int, default=0)
parser.add_argument('--kernel_type', type=str, default='sp_laplace', choices=['sp_laplace', 'dp_laplace', 'adjoint_dp_laplace', 'stokes', 'modified_dp_laplace', 'fredholm_laplace', 'exterior_laplace_neumann'])
args = parser.parse_args()

layer_selection = {'grad': args.grad.lower() == "true", 'geograd': args.geograd.lower() == "true", 'geo': args.geo.lower() == "true", 'lap': args.lap.lower() == "true"}
normal_prod = args.normal_prod.lower() == "true"
f_in_dim = 2 if args.kernel_type in ['stokes'] else 1
f_out_dim = 2 if args.kernel_type in ['modified_dp_laplace','stokes'] else 1
train_inv_L_scale = False
k_max = args.k_max
ndim = 2
L = 10
scale = args.scale
layers = [int(size) for size in args.layer_sizes.split(",")]
num_grad =  args.num_grad
geo_dims = args.geo_dims if args.geo_dims is not None else [f_in_dim, f_in_dim+1, 3*f_in_dim+2, 3*f_in_dim+3] if normal_prod else [f_in_dim, f_in_dim+1, f_in_dim+2, f_in_dim+3]
act = args.act


###################################
# load data
###################################
def load_data_to_torch(data_file_path, to_divide = None):
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
    print("Loading data from ", data_file_path, flush = True)
    data = np.load(data_file_path)
    nnodes, node_mask, nodes = data["nnodes"], data["node_mask"], data["nodes"]
    print(nnodes.shape,node_mask.shape,nodes.shape,flush = True)
    node_weights = data["node_measures_raw"]
    if to_divide is None:
        to_divide = np.amax(np.sum(node_weights, axis = 1))
    print('Node weights are devided by ', to_divide.item())
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


def gen_data_tensors(data_indices, nodes, features, node_mask, node_weights, directed_edges, edge_gradient_weights, node_rhos):
    nodes_input = nodes.clone()
    if normal_prod:
        x = torch.cat((features[data_indices][...,:f_in_dim+2],
                            features[data_indices][...,f_in_dim:f_in_dim+2]*features[data_indices][..., :f_in_dim],
                            nodes_input[data_indices, ...],
                            node_rhos[data_indices, ...]), -1)
    else:
        x = torch.cat((features[data_indices][...,:f_in_dim+2],
                            nodes_input[data_indices, ...],
                            node_rhos[data_indices, ...]), -1)
    y = features[data_indices][...,-f_out_dim:]
    aux = (node_mask[data_indices], nodes[data_indices], node_weights[data_indices], directed_edges[data_indices], edge_gradient_weights[data_indices])
    
    return x, y, aux


data_path = "../../data/curve/"

n_train, n_test, n_two_circles_test = args.n_train, args.n_test, args.n_two_circles_test
data_file_path = data_path+f"/pcno_curve_data_1_1_5_2d_{args.kernel_type}_panel.npz"
nnodes, node_mask, nodes, node_weights, node_rhos, features, directed_edges, edge_gradient_weights, to_divide = load_data_to_torch(data_file_path, to_divide = None)

x_train, y_train, aux_train = gen_data_tensors(np.arange(n_train), nodes, features, node_mask, node_weights, directed_edges, edge_gradient_weights, node_rhos,)
x_test, y_test, aux_test = gen_data_tensors(np.arange(-n_test, 0), nodes, features, node_mask, node_weights, directed_edges, edge_gradient_weights, node_rhos,)
x_test_list, y_test_list, aux_test_list = [x_test], [y_test], [aux_test]
label_list = ['Default']

if n_two_circles_test > 0:
    data_file_path = data_path+f"/pcno_curve_data_1_1_5_2d_{args.kernel_type}_panel_two_circles.npz"

    nnodes2, node_mask2, nodes2, node_weights2, node_rhos2, features2, directed_edges2, edge_gradient_weights2, _ = load_data_to_torch(data_file_path, to_divide = to_divide)
    x_two_circles_test, y_two_circles_test, aux_two_circles_test = gen_data_tensors(np.arange(n_two_circles_test),nodes2, features2, node_mask2, node_weights2, directed_edges2, edge_gradient_weights2, node_rhos2,)
    x_test_list.append(x_two_circles_test)
    y_test_list.append(y_two_circles_test)
    aux_test_list.append(aux_two_circles_test)
    label_list.append('Two Circles')

print(f'x_train shape {x_train.shape}, x_test shape {[x.shape for x in x_test_list]}, y_train shape {y_train.shape}, y_test shape {[y.shape for y in y_test_list]}', flush = True)
print('length of each dim: ',torch.amax(nodes, dim = [0,1]) - torch.amin(nodes, dim = [0,1]), flush = True)

modes = compute_Fourier_modes(ndim, [k_max,k_max], [L,L])

def nonlinear_scale(modes, scale=1.0):
    norms = np.linalg.norm(modes, axis=1)
    max_norm = norms.max()
    scaled_norms = (norms / max_norm) ** scale
    modes_scaled = modes * scaled_norms[:, np.newaxis]
    return modes_scaled

modes = nonlinear_scale(modes, scale=scale)

print(f'kmax = {k_max}')
print(f'n_train = {n_train}, n_test = {n_test}')
print(f'L = {L}')
print(f'use cube modes, scale = {scale}', modes.shape)
print(f'normal_prod = {normal_prod}')
print(f'geo_dims = {geo_dims}')
print(f'num_grad = {num_grad}')
print(f'layer_selection = {layer_selection}')
print(f'layers = {layers}')
print(f'activation = {act}')


modes = torch.tensor(modes, dtype=torch.float).to(device)
model = PCNO(ndim, modes, nmeasures=1, geo_dims=geo_dims, 
               layer_selection = layer_selection,
               num_grad = num_grad,
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
print('batch_size', batch_size)

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


train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = PCNO_train_multidist(
    x_train, aux_train, y_train, x_test_list, aux_test_list, y_test_list, config, model, label_test_list=label_list,
     save_model_name = None,#"model/1_1_5_5_grad_deformed/k8_L10_normal_prod_layer2_geo2_softsign_new"
)
