import torch
import sys
import os
import numpy as np
import argparse
from datetime import datetime
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


from pcno.geo_utility import preprocess_data_mesh, compute_node_weights, compute_unnormalized_node_measures
from pcno.geo_utility import sample_close_node_pairs
from pcno.pcno import compute_Fourier_modes, PCNO, PCNO_train


torch.set_printoptions(precision=16)

torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_data(data_path):

    ndata = 10000
    elem_dim = 1

    vertices_list = []
    elems_list = []
    features_list = []

    for index in range(ndata):
        vertices = np.load(data_path + "./nodes_%07d"%(index) + ".npy")
        
        elems = np.load(data_path + "./elems_%07d"%(index) + ".npy")
        elem_features = np.load(data_path +"./elem_features_%07d"%(index) + ".npy")
        
        middle_points = (vertices[elems[:,1],:] + vertices[elems[:,0],:]) / 2.0
        tagents = vertices[elems[:,1],:] - vertices[elems[:,0],:]
        lengths = np.linalg.norm(tagents, axis=1, keepdims=True)  # shape: (num_edges, 1)
        tagents = tagents / lengths
        normals = np.column_stack((-tagents[:, 1], tagents[:,0]))


        vertices_list.append(vertices)
        elems_list.append(np.concatenate((np.full((elems.shape[0], 1), elem_dim, dtype=int), elems), axis=1))
        features_list.append(np.concatenate((normals, elem_features), axis=1))

    return vertices_list, elems_list, features_list

try:
    PREPROCESS_DATA = sys.argv[1] == "preprocess_data" if len(
        sys.argv) > 1 else False
except IndexError:
    PREPROCESS_DATA = False


###################################
# load data
###################################
data_path = "../../data/laplace_double_layer_kernel/"

if PREPROCESS_DATA:
    print("Loading data")
    vertices_list, elems_list, features_list = load_data(data_path)
    nnodes, node_mask, nodes, node_measures_raw, features, directed_edges, edge_gradient_weights = preprocess_data_mesh(
        vertices_list, elems_list, features_list, mesh_type='cell_centered', adjacent_type='node')
    print(f"nodes{nodes.shape}", flush=True)
    # node_measures, node_weights = compute_node_weights(nnodes, node_measures_raw, equal_measure = False)
    node_measures, node_weights = compute_unnormalized_node_measures(nnodes, node_measures_raw, measure_dims=[1], Ls=[2.5,2.5])
    node_equal_measures, node_equal_weights = compute_node_weights(nnodes, node_measures_raw, equal_measure = True)
    np.savez_compressed(data_path+"pcno_data_basic.npz",
                        nnodes=nnodes, node_mask=node_mask, nodes=nodes,
                        node_measures_raw = node_measures_raw,
                        node_measures=node_measures, node_weights=node_weights,
                        node_equal_measures=node_equal_measures, node_equal_weights=node_equal_weights,
                        features=features) 

    np.savez_compressed(data_path+"pcno_data_gradient.npz",
                        directed_edges=directed_edges, edge_gradient_weights=edge_gradient_weights) 

    normals = features[...,[0,1]][..., np.newaxis] # nmeasures=1
    max_nnodes = max(nnodes)
    close_directed_edges, _, close_directed_edge_node_weights = sample_close_node_pairs(nodes, nnodes, node_weights, dist_threshold=0.1, max_nedges=10*max_nnodes)
    
    np.savez_compressed(data_path+"pcno_data_close_loc.npz",
                        close_directed_edges=close_directed_edges, close_directed_edge_node_weights=close_directed_edge_node_weights) 
    exit()
else:
    print("Loading data")
    equal_weights = False

    ########## basic data
    data = np.load(data_path + "pcno_data_basic.npz")
    nnodes, node_mask, nodes = data["nnodes"], data["node_mask"], data["nodes"]
    node_weights = data["node_equal_weights"] if equal_weights else data["node_weights"]
    node_measures = data["node_measures"]
    node_measures_raw = data["node_measures_raw"]
    indices = np.isfinite(node_measures_raw)
    node_rhos = np.copy(node_weights)
    node_rhos[indices] = node_rhos[indices]/node_measures[indices]
    features = data["features"]

    ########## gradient related data
    data = np.load(data_path + "pcno_data_gradient.npz")
    directed_edges, edge_gradient_weights = data["directed_edges"], data["edge_gradient_weights"]
    
    ########## neighbor related data
    data = np.load(data_path + "pcno_data_close_loc.npz")
    close_directed_edges, close_directed_edge_node_weights = data["close_directed_edges"], data["close_directed_edge_node_weights"]
    
    del data


parser = argparse.ArgumentParser(description='Train model with different configurations and options.')
parser.add_argument('--n_train', type=int, default=1000, help='Number of training samples')
parser.add_argument('--n_test', type=int, default=400, help='Number of testing samples')
parser.add_argument('--train_type', type=str, default='mixed', choices=['standard', 'flap', 'mixed'], help='Type of training data')
parser.add_argument('--train_inv_L_scale', type=str, default='False', choices=['False', 'together', 'independently'], help='Type of train_inv_L_scale')
parser.add_argument('--lr_ratio', type=float, default=10.0, help='Learning rate ratio of L-parameters to main parameters')


args = parser.parse_args()
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
save_model_name = f"PCNO_2dpanel_n{args.n_train}_{current_time}"
print(save_model_name)
args_dict = vars(args)
for i, (key, value) in enumerate(args_dict.items()):
    print(f"{key}: {value}")

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
directed_edges = torch.from_numpy(directed_edges.astype(np.int64))
edge_gradient_weights = torch.from_numpy(edge_gradient_weights.astype(np.float32))
close_directed_edges = torch.from_numpy(close_directed_edges)
close_directed_edge_node_weights = torch.from_numpy(close_directed_edge_node_weights.astype(np.float32))
normals = features[...,[0,1]].unsqueeze(-1)

# This is important
nodes_input = nodes.clone()
ndata = nodes_input.shape[0]
n_train, n_test = args.n_train, args.n_test


# features are normal, source_strength, source potential, tangential velocity, pressure coefficient
# x_train = torch.cat((features[:n_train, :, [0,1]], nodes_input[:n_train, ...], node_rhos[:n_train, ...]), -1)
# x_test = torch.cat((features[-n_test:, :, [0,1]], nodes_input[-n_test:, ...], node_rhos[-n_test:, ...]),-1)
# x_train = torch.cat((features[:n_train, ...][...,[0,1,2]], features[:n_train, ...][...,[0,1]]*features[:n_train, ...][...,[2]], nodes_input[:n_train, ...], node_rhos[:n_train, ...]), -1)
# x_test  = torch.cat((features[-n_test:, ...][...,[0,1,2]], features[-n_test:, ...][...,[0,1]]*features[-n_test:, ...][...,[2]], nodes_input[-n_test:, ...], node_rhos[-n_test:, ...]), -1)
x_train = torch.cat((features[:n_train, ...][...,[0,1,2]], features[:n_train, ...][...,[0,1]]*features[:n_train, ...][...,[2]]), -1)
x_test  = torch.cat((features[-n_test:, ...][...,[0,1,2]], features[-n_test:, ...][...,[0,1]]*features[-n_test:, ...][...,[2]]), -1)

y_train, y_test = (features[:n_train, ...][...,[3]], features[-n_test:, ...][...,[3]])


aux_train = (
    node_mask[:n_train, ...],
    nodes[:n_train, ...],
    node_weights[:n_train, ...],
    directed_edges[:n_train, ...],
    edge_gradient_weights[:n_train, ...],
    close_directed_edges[:n_train, ...], 
    close_directed_edge_node_weights[:n_train, ...],
)
aux_test = (
    node_mask[-n_test:, ...],
    nodes[-n_test:, ...],
    node_weights[-n_test:, ...],
    directed_edges[-n_test:, ...],
    edge_gradient_weights[-n_test:, ...],
    close_directed_edges[-n_test:, ...], 
    close_directed_edge_node_weights[-n_test:, ...],
)


print(
    f"x train:{x_train.shape}, y train:{y_train.shape}, x test:{x_test.shape}, y test:{y_test.shape}",
    flush=True,
)

###################################
# train
###################################
kx_max, ky_max = 64, 64
ndim = 2
if args.train_inv_L_scale == 'False':
    args.train_inv_L_scale = False
train_inv_L_scale = args.train_inv_L_scale
Lx, Ly = 5, 5
print("Lx, Ly = ", Lx, Ly)
modes = compute_Fourier_modes(ndim, [kx_max, ky_max], [Lx, Ly])
modes = torch.tensor(modes, dtype=torch.float).to(device)

loc_modes = compute_Fourier_modes(ndim, [8, 8], [0.3, 0.3])
loc_modes = torch.tensor(loc_modes, dtype=torch.float).to(device)

model = PCNO(ndim, modes, loc_modes, nmeasures=1,
             layers=[128,128],
             fc_dim=-1,               #128,
             in_dim=x_train.shape[-1], out_dim=y_train.shape[-1],
             inv_L_scale_hyper = [train_inv_L_scale, 0.5, 2.0],
             act='none').to(device)
#model = PCNO(ndim, modes, nmeasures=1,
#             layers=[128,128],
#             fc_dim=-1,               #128,
#             in_dim=x_train.shape[-1], out_dim=y_train.shape[-1],
#             inv_L_scale_hyper = [train_inv_L_scale, 0.5, 2.0],
#             act='none').to(device)
epochs = 500
base_lr = 0.0005
scheduler = "OneCycleLR"
weight_decay = 1.0e-4
batch_size = 8
lr_ratio = args.lr_ratio

normalization_x = False
normalization_y = True
normalization_dim_x = []
normalization_dim_y = []
non_normalized_dim_x = 0
non_normalized_dim_y = 0


config = {"train": {"base_lr": base_lr, 'lr_ratio': lr_ratio, "weight_decay": weight_decay, "epochs": epochs, "scheduler": scheduler, "batch_size": batch_size,
                    "normalization_x": normalization_x, "normalization_y": normalization_y,
                    "normalization_dim_x": normalization_dim_x, "normalization_dim_y": normalization_dim_y,
                    "non_normalized_dim_x": non_normalized_dim_x, "non_normalized_dim_y": non_normalized_dim_y}
          }

train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = PCNO_train(
    x_train, aux_train, y_train, x_test, aux_test, y_test, config, model, save_model_name=save_model_name
)
