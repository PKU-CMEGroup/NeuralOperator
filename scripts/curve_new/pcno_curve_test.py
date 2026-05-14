import os
import torch
import sys
import argparse

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from pcno.geo_utility import preprocess_data_mesh, compute_node_weights
from pcno.pcno_structured import compute_Fourier_modes, PCNO, PCNO_train_multidist

torch.set_printoptions(precision=16)

torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_raw_data(data_file_path):
    data = np.load(data_file_path)
    nodes_list, elems_list, features_list = data["nodes_list"], data["elems_list"], data["features_list"]
    return nodes_list, elems_list, features_list


def load_data_to_torch(data_file_path, to_divide=None):
    """
    Returns tensors:
        nnodes, node_mask, nodes, node_weights, node_rhos, features,
        directed_edges, edge_gradient_weights, to_divide
    """
    data = np.load(data_file_path)
    nnodes, node_mask, nodes = data["nnodes"], data["node_mask"], data["nodes"]
    print(f"Loaded {nodes.shape[0]} samples from {data_file_path}", flush=True)

    node_weights = data["node_measures_raw"]
    if to_divide is None:
        to_divide = np.amax(np.sum(node_weights, axis=1))
    node_weights = node_weights / to_divide

    node_measures = data["node_measures"]
    directed_edges, edge_gradient_weights = data["directed_edges"], data["edge_gradient_weights"]
    features = data["features"]

    node_measures_raw = data["node_measures_raw"]
    indices = np.isfinite(node_measures_raw)
    node_rhos = np.copy(node_weights)
    node_rhos[indices] = node_rhos[indices] / node_measures[indices]

    nnodes = torch.from_numpy(nnodes)
    node_mask = torch.from_numpy(node_mask)
    nodes = torch.from_numpy(nodes.astype(np.float32))
    node_weights = torch.from_numpy(node_weights.astype(np.float32))
    node_rhos = torch.from_numpy(node_rhos.astype(np.float32))
    features = torch.from_numpy(features.astype(np.float32))
    directed_edges = torch.from_numpy(directed_edges.astype(np.int64))
    edge_gradient_weights = torch.from_numpy(edge_gradient_weights.astype(np.float32))

    return (
        nnodes,
        node_mask,
        nodes,
        node_weights,
        node_rhos,
        features,
        directed_edges,
        edge_gradient_weights,
        to_divide,
    )


def gen_data_tensors(data_indices, nodes, features, node_mask, node_weights, node_rhos, directed_edges, edge_gradient_weights):
    nodes_input = nodes.clone()
    x = torch.cat(
        (
            features[data_indices][..., [0, 1, 2]],
            features[data_indices][..., [1, 2]] * features[data_indices][..., [0]],
            nodes_input[data_indices, ...],
            node_rhos[data_indices, ...],
        ),
        -1,
    )
    y = features[data_indices][..., [3]]
    aux = (
        node_mask[data_indices],
        nodes[data_indices],
        node_weights[data_indices],
        directed_edges[data_indices],
        edge_gradient_weights[data_indices],
    )
    return x, y, aux


parser = argparse.ArgumentParser(description='PCNO Structured training on curve data')
parser.add_argument('--n_train', type=int, default=900)
parser.add_argument('--n_test', type=int, default=100)
parser.add_argument('--n_two_circles_test', type=int, default=0)
parser.add_argument('--to_divide_factor', type=float, default=20.0)
parser.add_argument('--k_max', type=int, default=8)
parser.add_argument('--ep', type=int, default=500)
parser.add_argument('--bsz', type=int, default=8)
parser.add_argument('--act', type=str, default='gelu')
parser.add_argument('--layer_sizes', type=str, default='64,64')
parser.add_argument('--proj_layer_sizes', type=str, default='128,128,128')
parser.add_argument('--proj_act', type=str, default="gelu")
parser.add_argument('--data_file', type=str)
parser.add_argument('--two_circles_data_file', type=str)
args = parser.parse_args()

###################################
# load data
###################################
data_path = "../../data/curve/"
to_divide = args.to_divide_factor

main_data_file = data_path + args.data_file
(
    nnodes,
    node_mask,
    nodes,
    node_weights,
    node_rhos,
    features,
    directed_edges,
    edge_gradient_weights,
    to_divide,
) = load_data_to_torch(main_data_file)

n_train, n_test, n_two_circles_test = args.n_train, args.n_test, args.n_two_circles_test

x_train, y_train, aux_train = gen_data_tensors(
    np.arange(n_train),
    nodes,
    features,
    node_mask,
    node_weights,
    node_rhos,
    directed_edges,
    edge_gradient_weights,
)
x_test, y_test, aux_test = gen_data_tensors(
    np.arange(-n_test, 0),
    nodes,
    features,
    node_mask,
    node_weights,
    node_rhos,
    directed_edges,
    edge_gradient_weights,
)

x_test_list, y_test_list, aux_test_list = [x_test], [y_test], [aux_test]
label_list = ['Default']

if n_two_circles_test > 0:
    two_circles_path = data_path + args.two_circles_data_file
    (
        nnodes2,
        node_mask2,
        nodes2,
        node_weights2,
        node_rhos2,
        features2,
        directed_edges2,
        edge_gradient_weights2,
        _,
    ) = load_data_to_torch(two_circles_path, to_divide=to_divide)

    x_two_circles_test, y_two_circles_test, aux_two_circles_test = gen_data_tensors(
        np.arange(n_two_circles_test),
        nodes2,
        features2,
        node_mask2,
        node_weights2,
        node_rhos2,
        directed_edges2,
        edge_gradient_weights2,
    )
    x_test_list.append(x_two_circles_test)
    y_test_list.append(y_two_circles_test)
    aux_test_list.append(aux_two_circles_test)
    label_list.append('Two Circles')

print(
    f'x_train shape {x_train.shape}, x_test shape {[x.shape for x in x_test_list]}, '
    f'y_train shape {y_train.shape}, y_test shape {[y.shape for y in y_test_list]}',
    flush=True,
)
print(
    'Domain range per dimension: ',
    torch.amax(nodes, dim=[0, 1]) - torch.amin(nodes, dim=[0, 1]),
    flush=True,
)

###################################
# load model and train
###################################
k_max = args.k_max
ndim = 2
train_inv_L_scale = False
L = 10
layers = [int(size) for size in args.layer_sizes.split(',')]
proj_layers = [int(size) for size in args.proj_layer_sizes.split(',') if int(size) > 0]

modes = compute_Fourier_modes(ndim, [k_max, k_max], [L, L])

print('------Parameters------')
print(f'kmax = {k_max}')
print(f'n_train = {n_train}, n_test = {n_test}, n_two_circles_test = {n_two_circles_test}')
print(f'L = {L}')
print(f'Shape of Fourier modes: {modes.shape}')
print(f'layers = {layers}')
print(f'proj_layers = {proj_layers}')
print(f'activation = {args.act}')
print(f'activation_projection = {args.proj_act}')

modes = torch.tensor(modes, dtype=torch.float).to(device)
model = PCNO(
    ndim,
    modes,
    nmeasures=1,
    layers=layers,
    proj_layers=proj_layers,
    fc_dim=0,
    in_dim=x_train.shape[-1],
    out_dim=y_train.shape[-1],
    inv_L_scale_hyper=[train_inv_L_scale, 0.5, 2.0],
    act=args.act,
    proj_act=args.proj_act,
).to(device)

epochs = args.ep
base_lr = 5e-4
lr_ratio = 10
scheduler = "OneCycleLR"
weight_decay = 1.0e-4
batch_size = args.bsz

normalization_x = False
normalization_y = True
normalization_dim_x = []
normalization_dim_y = []
non_normalized_dim_x = 4
non_normalized_dim_y = 0

config = {
    "train": {
        "base_lr": base_lr,
        'lr_ratio': lr_ratio,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "scheduler": scheduler,
        "batch_size": batch_size,
        "normalization_x": normalization_x,
        "normalization_y": normalization_y,
        "normalization_dim_x": normalization_dim_x,
        "normalization_dim_y": normalization_dim_y,
        "non_normalized_dim_x": non_normalized_dim_x,
        "non_normalized_dim_y": non_normalized_dim_y,
    }
}

train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = PCNO_train_multidist(
    x_train,
    aux_train,
    y_train,
    x_test_list,
    aux_test_list,
    y_test_list,
    config,
    model,
    label_test_list=label_list,
    save_model_name=None,
)
