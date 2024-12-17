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

    path1 = os.path.join(data_path, "Airfoil_flap_data/fluid_mesh")
    path2 = os.path.join(data_path, "Airfoil_data/fluid_mesh")

    ndata1 = 1932
    ndata2 = 1931
    elem_dim = 2

    nodes_list = []
    elems_list = []
    features_list = []

    for i in range(ndata1):
        nodes_list.append(np.load(path1 + "/nodes_%05d" % (i) + ".npy"))
        elems = np.load(path1 + "/elems_%05d" % (i) + ".npy")
        elems_list.append(np.concatenate(
            (np.full((elems.shape[0], 1), elem_dim, dtype=int), elems), axis=1))
        features_list.append(np.load(path1 + "/features_%05d" % (i) + ".npy"))

    for i in range(ndata2):
        nodes_list.append(np.load(path2 + "/nodes_%05d" % (i) + ".npy"))

        elems = np.load(path2 + "/elems_%05d" % (i) + ".npy")
        elems_list.append(np.concatenate(
            (np.full((elems.shape[0], 1), elem_dim, dtype=int), elems), axis=1))

        features_list.append(np.load(path2 + "/features_%05d" % (i) + ".npy"))

    return nodes_list, elems_list, features_list


try:
    PREPROCESS_DATA = sys.argv[1] == "preprocess_data" if len(
        sys.argv) > 1 else False
except IndexError:
    PREPROCESS_DATA = False

###################################
# load data
###################################
data_path = "../../data/airfoil_flap/"
if PREPROCESS_DATA:
    print("Loading data")

    nodes_list, elems_list, feats_list = load_data(data_path)
    nnodes, node_mask, nodes, node_measures_raw, features, directed_edges, edge_gradient_weights = preprocess_data(
        nodes_list, elems_list, feats_list)
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
    equal_weights = True

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
features = torch.from_numpy(features.astype(np.float32))
directed_edges = torch.from_numpy(directed_edges)
edge_gradient_weights = torch.from_numpy(
    edge_gradient_weights.astype(np.float32))

# This is important
nodes_input = nodes.clone()

ndata = nodes_input.shape[0]
n_train, n_test = 1000, 500
m_train, m_test = n_train // 2, n_test // 2
train_index = torch.cat([torch.arange(m_train),
                        torch.arange(ndata - m_train, ndata)], dim=0)
test_index = torch.cat([torch.arange(m_train, m_train + m_test),
                       torch.arange(ndata - m_train - m_test, ndata - m_train)], dim=0)


x_train, x_test = nodes_input[train_index, ...], nodes_input[test_index, ...]
aux_train = (node_mask[train_index, ...], nodes[train_index, ...], node_weights[train_index, ...],
             directed_edges[train_index, ...], edge_gradient_weights[train_index, ...])
aux_test = (node_mask[test_index, ...], nodes[test_index, ...], node_weights[test_index, ...],
            directed_edges[test_index, ...], edge_gradient_weights[test_index, ...])
y_train, y_test = features[train_index, :, [1]], features[test_index, :, [1]]


###################################
# train
###################################
kx_max, ky_max = 32, 16
ndim = 2
train_sp_L = 'independently'

Lx, Ly = 1.25, 0.5
print("Lx, Ly = ", Lx, Ly)
modes = compute_Fourier_modes(ndim, [kx_max, ky_max], [Lx, Ly])
modes = torch.tensor(modes, dtype=torch.float).to(device)
model = PCNO(ndim, modes, nmeasures=1,
             layers=[128, 128, 128, 128, 128],
             fc_dim=128,
             in_dim=2, out_dim=1,
             train_sp_L=train_sp_L,
             act='gelu').to(device)

epochs = 500
base_lr = 0.001
scheduler = "OneCycleLR"
weight_decay = 1.0e-4
batch_size = 20
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
    x_train, aux_train, y_train, x_test, aux_test, y_test, config, model, save_model_name="./PCNO_airfoilflap_model"
)
