import torch
import sys
import os
import numpy as np
import argparse
from datetime import datetime
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


from pcno.geo_utility import preprocess_data, compute_node_weights
from pcno.pcno import compute_Fourier_modes, PCNO, PCNO_train


torch.set_printoptions(precision=16)

torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_data(data_path):
    """
    Load the airfoil flap data including nodes, elements, and features from the specified path.
    
    Args:
        data_path (str): The base path where data is stored.
        
    Returns:
        tuple: A tuple containing three lists (nodes, elements, features) for all data samples.
    """
    path1 = os.path.join(data_path, "Airfoil_flap_data/fluid_mesh")
    path2 = os.path.join(data_path, "Airfoil_data/fluid_mesh")

    ndata1, ndata2 = 1931, 1932
    elem_dim = 2

    nodes_list = []
    elems_list = []
    features_list = []

    for i in range(ndata1):
        nodes_list.append(np.load(path1 + "/nodes_%05d" % (i) + ".npy"))
        elems = np.load(path1 + "/elems_%05d" % (i) + ".npy")
        elems_list.append(np.concatenate((np.full((elems.shape[0], 1), elem_dim, dtype=int), elems), axis=1))
        features_list.append(np.load(path1 + "/features_%05d" % (i) + ".npy"))

    for i in range(ndata2):
        nodes_list.append(np.load(path2 + "/nodes_%05d" % (i) + ".npy"))

        elems = np.load(path2 + "/elems_%05d" % (i) + ".npy")
        elems_list.append(np.concatenate((np.full((elems.shape[0], 1), elem_dim, dtype=int), elems), axis=1))
        features_list.append(np.load(path2 + "/features_%05d" % (i) + ".npy"))

    return nodes_list, elems_list, features_list

parser = argparse.ArgumentParser(description='Train model with different configurations and options.')

parser.add_argument('--preprocess_data', type=str, default='False', help='Whether to preprocess the data before training (True/False)')

parser.add_argument('--n_train', type=int, default=1000, help='Number of training samples')
parser.add_argument('--n_test', type=int, default=400, help='Number of testing samples')
parser.add_argument('--train_type', type=str, default='mixed', choices=['standard', 'flap', 'mixed'], help='Type of training data')
parser.add_argument('--feature_type', type=str, default='pressure', choices=['pressure', 'mach'], help='Type of feature to use')
parser.add_argument('--equal_weight', type=str, default='True', help='Specify whether to use equal weight')
parser.add_argument('--rho', type=str, default='True', help='Specify whether to include rho in the input')

parser.add_argument('--Lx', type=float, default=1.0, help='Initial value for the length of the x dimension')
parser.add_argument('--Ly', type=float, default=0.5, help='Initial value for the length of the y dimension')
parser.add_argument('--train_sp_L', type=str, default='independently', choices=['False', 'together', 'independently'], help='type of train_sp_L (False, together, independently)')

parser.add_argument('--normalization_x', type=str, default='False', help='Whether to normalize the x dimension (True/False)')
parser.add_argument('--normalization_y', type=str, default='True', help='Whether to normalize the y dimension (True/False)')
parser.add_argument('--non_normalized_dim_x', type=int, default=0, choices=[0, 1], help='Specifies the dimension of x that should not be normalized')
parser.add_argument('--non_normalized_dim_y', type=int, default=0, choices=[0], help='Specifies the dimension of y that should not be normalized')

parser.add_argument('--lr_ratio', type=float, default=10, help='Learning rate ratio of main parameters and L parameters when train_sp_L is set to `independently`')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')


args = parser.parse_args()
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
save_model_name = f"PCNO_airfoil_flap_{args.train_type}_n{args.n_train}_{current_time}"
print(save_model_name)
args_dict = vars(args)
for i, (key, value) in enumerate(args_dict.items()):
    print(f"{key}: {value}")


###################################
# load data
###################################
data_path = "../../data/airfoil_flap/"

if args.preprocess_data.lower() == "true":
    print("Loading data")
    nodes_list, elems_list, feats_list = load_data(data_path)
    nnodes, node_mask, nodes, node_measures_raw, features, directed_edges, edge_gradient_weights = preprocess_data(
        nodes_list, elems_list, feats_list)
    print(f"nodes{nodes.shape}", flush=True)
    node_measures, node_weights = compute_node_weights(nnodes, node_measures_raw, equal_measure = False)
    node_equal_measures, node_equal_weights = compute_node_weights(nnodes, node_measures_raw, equal_measure = True)
    np.savez_compressed(data_path+"pcno_triangle_data.npz",
                        nnodes=nnodes, node_mask=node_mask, nodes=nodes,
                        node_measures_raw = node_measures_raw,
                        node_measures=node_measures, node_weights=node_weights,
                        node_equal_measures=node_equal_measures, node_equal_weights=node_equal_weights,
                        features=features,
                        directed_edges=directed_edges, edge_gradient_weights=edge_gradient_weights) 
    exit()
else:
    print("Loading data")
    equal_weights = args.equal_weight.lower() == "true"

    data = np.load(data_path + "pcno_triangle_data.npz")
    nnodes, node_mask, nodes = data["nnodes"], data["node_mask"], data["nodes"]

    node_weights = data["node_equal_weights"] if equal_weights else data["node_weights"]
    node_measures = data["node_measures"]
    node_measures_raw = data["node_measures_raw"]
    indices = np.isfinite(node_measures_raw)
    node_rhos = np.copy(node_weights)
    node_rhos[indices] = node_rhos[indices]/node_measures[indices]

    directed_edges, edge_gradient_weights = data["directed_edges"], data["edge_gradient_weights"]
    features = data["features"]


###################################
# prepare data
###################################
print("Casting to tensor")
nnodes = torch.from_numpy(nnodes)
node_mask = torch.from_numpy(node_mask)
nodes = torch.from_numpy(nodes.astype(np.float32))
node_weights = torch.from_numpy(node_weights.astype(np.float32))
node_rhos = torch.from_numpy(node_rhos.astype(np.float32))
features = torch.from_numpy(features.astype(np.float32))
directed_edges = torch.from_numpy(directed_edges)
edge_gradient_weights = torch.from_numpy(edge_gradient_weights.astype(np.float32))

# This is important
nodes_input = nodes.clone()

in_dim = 2
if args.rho == "True":
    nodes_input = torch.cat([nodes_input, node_rhos], dim=-1)
    in_dim = 3

ndata = nodes_input.shape[0]
ndata1 = 1931
ndata2 = 1932
n_train, n_test = args.n_train, args.n_test
m_train, m_test = n_train // 2, n_test // 2

train_type = args.train_type
if train_type == "flap":
    train_index = torch.arange(n_train)
elif train_type  == "standard":
    train_index = torch.arange(ndata - n_train, ndata) 
elif train_type == "mixed":
    train_index = torch.cat(
        [torch.arange(m_train), torch.arange(ndata - m_train, ndata)], dim=0
    ) 
test_index = torch.cat(
    [torch.arange(ndata1 - m_test, ndata1), torch.arange(ndata1, ndata1 + m_test)], dim=0
)


x_train, x_test = nodes_input[train_index, ...], nodes_input[test_index, ...]
aux_train = (
    node_mask[train_index, ...],
    nodes[train_index, ...],
    node_weights[train_index, ...],
    directed_edges[train_index, ...],
    edge_gradient_weights[train_index, ...],
)
aux_test = (
    node_mask[test_index, ...],
    nodes[test_index, ...],
    node_weights[test_index, ...],
    directed_edges[test_index, ...],
    edge_gradient_weights[test_index, ...],
)
feature_type = args.feature_type
if feature_type == "mach":
    feature_type_index = 1
elif feature_type == "pressure":
    feature_type_index = 0
y_train, y_test = (
    features[train_index, ...][...,feature_type_index],
    features[test_index, ...][...,feature_type_index],
)
print(
    f"x train:{x_train.shape}, y train:{y_train.shape}, x test:{x_test.shape}, y test:{y_test.shape}",
    flush=True,
)

###################################
# train
###################################
kx_max, ky_max = 16, 16
ndim = 2
if args.train_sp_L == 'False':
    args.train_sp_L = False
train_sp_L = args.train_sp_L
Lx, Ly = args.Lx, args.Ly
print("Lx, Ly = ", Lx, Ly)
modes = compute_Fourier_modes(ndim, [kx_max, ky_max], [Lx, Ly])
modes = torch.tensor(modes, dtype=torch.float).to(device)
model = PCNO(ndim, modes, nmeasures=1,
             layers=[128, 128, 128, 128, 128],
             fc_dim=128,
             in_dim=in_dim, out_dim=1,
             train_sp_L=train_sp_L,
             act='gelu').to(device)

epochs = 500
base_lr = 0.001
scheduler = "OneCycleLR"
weight_decay = 1.0e-4
batch_size = args.batch_size
lr_ratio = args.lr_ratio

normalization_x = args.normalization_x.lower() == "true"
normalization_y = args.normalization_y.lower() == "true"
normalization_dim_x = []
normalization_dim_y = []
non_normalized_dim_x = args.non_normalized_dim_x
non_normalized_dim_y = args.non_normalized_dim_y


config = {"train": {"base_lr": base_lr, 'lr_ratio': lr_ratio, "weight_decay": weight_decay, "epochs": epochs, "scheduler": scheduler, "batch_size": batch_size,
                    "normalization_x": normalization_x, "normalization_y": normalization_y,
                    "normalization_dim_x": normalization_dim_x, "normalization_dim_y": normalization_dim_y,
                    "non_normalized_dim_x": non_normalized_dim_x, "non_normalized_dim_y": non_normalized_dim_y}
          }

train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = PCNO_train(
    x_train, aux_train, y_train, x_test, aux_test, y_test, config, model, save_model_name=save_model_name
)