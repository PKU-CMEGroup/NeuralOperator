import torch
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))


from pcno.geo_utility import preprocess_data, compute_node_weights
from pcno.pcno import compute_Fourier_modes, PCNO, PCNO_train


torch.set_printoptions(precision=16)

torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_data(data_path):
    ndata = 551
    nodes_list, elems_list, features_list = [], [], []
    for i in range(ndata):
        nodes_list.append(np.load(data_path + "/nodes_%05d" % (i) + ".npy"))
        elems_list.append(np.load(data_path + "/elems_%05d" % (i) + ".npy"))
        features_list.append(
            np.load(data_path + "/features_%05d" % (i) + ".npy"))
    return nodes_list, elems_list, features_list


try:
    PREPROCESS_DATA = sys.argv[1] == "preprocess_data" if len(
        sys.argv) > 1 else False
except IndexError:
    PREPROCESS_DATA = False

###################################
# load data
###################################
data_path = "../../data/ahmed_body/"
if PREPROCESS_DATA:
    print("Loading data")
    nodes_list, elems_list, features_list = load_data(data_path=data_path)

    print("Preprocessing data")
    nnodes, node_mask, nodes, node_measures_raw, features, directed_edges, edge_gradient_weights = preprocess_data(
        nodes_list, elems_list, features_list)
    node_measures, node_weights = compute_node_weights(
        nnodes, node_measures_raw, equal_measure=False)
    node_equal_measures, node_equal_weights = compute_node_weights(
        nnodes, node_measures_raw, equal_measure=True)
    np.savez_compressed(data_path + "pcno_triangle_data.npz",
                        nnodes=nnodes, node_mask=node_mask, nodes=nodes,
                        node_measures=node_measures, node_weights=node_weights,
                        node_equal_measures=node_equal_measures, node_equal_weights=node_equal_weights,
                        features=features,
                        directed_edges=directed_edges, edge_gradient_weights=edge_gradient_weights)
    exit()

else:
    # load data
    print("Loading data")
    equal_weights = False
    data = np.load(data_path + "pcno_triangle_data.npz")
    nnodes, node_mask, nodes = data["nnodes"], data["node_mask"], data["nodes"]
    node_weights = data["node_equal_weights"] if equal_weights else data["node_weights"]
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

# Normalize the infos to the range [0,1]
normalization_infos = True
if normalization_infos:
    eps = 1e-06
    with open(data_path + "info_bounds.txt", "r") as fp:
        min_bounds = fp.readline().split(" ")
        max_bounds = fp.readline().split(" ")

        min_bounds = [float(a) - eps for a in min_bounds]
        max_bounds = [float(a) + eps for a in max_bounds]

    for i in range(8):
        features[..., i + 1] = (features[..., i + 1]
                                - min_bounds[i]) / (max_bounds[i] - min_bounds[i])
features = torch.from_numpy(features.astype(np.float32))

directed_edges = torch.from_numpy(directed_edges)
edge_gradient_weights = torch.from_numpy(
    edge_gradient_weights.astype(np.float32))

# This is important
nodes_input = nodes.clone()

data_in, data_out = torch.cat(
    [nodes_input, features[..., 1:]], dim=-1), features[..., :1]
print(f"data in:{data_in.shape}, data out:{data_out.shape}")
n_train, n_test = 250, 50


x_train, x_test = data_in[:n_train, ...], data_in[-n_test:, ...]
aux_train = (node_mask[0:n_train, ...], nodes[0:n_train, ...], node_weights[0:n_train, ...],
             directed_edges[0:n_train, ...], edge_gradient_weights[0:n_train, ...])
aux_test = (node_mask[-n_test:, ...], nodes[-n_test:, ...], node_weights[-n_test:, ...],
            directed_edges[-n_test:, ...], edge_gradient_weights[-n_test:, ...])
y_train, y_test = data_out[:n_train, ...], data_out[-n_test:, ...]

print(f"x train:{x_train.shape}, y train:{y_train.shape}", flush=True)


###################################
# train
###################################
k_max = 8
ndim = 3
train_sp_L = False

Lx = 0.0004795 - (-1.34399998)
Ly = 0.25450477 - 0
Lz = 0.43050185 - 0

ndim = 3

print("Lx, Ly, Lz = ", Lx, Ly, Lz)
modes = compute_Fourier_modes(ndim, [k_max, k_max, k_max], [Lx, Ly, Lz])
modes = torch.tensor(modes, dtype=torch.float).to(device)
model = PCNO(ndim, modes, nmeasures=1,
             layers=[128, 128, 128, 128, 128],
             fc_dim=128,
             in_dim=11, out_dim=1,
             train_sp_L=train_sp_L,
             act='gelu').to(device)

epochs = 500
base_lr = 0.001
scheduler = "OneCycleLR"
weight_decay = 1.0e-4
batch_size = 5
lr_ratio = 10

normalization_x = False
normalization_y = False
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
    x_train, aux_train, y_train, x_test, aux_test, y_test, config, model, save_model_name="./PCNO_ahmedbody_model"
)
