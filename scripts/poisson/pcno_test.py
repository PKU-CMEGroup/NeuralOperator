import os
import glob
import random
import torch
import sys
import numpy as np
import math
from timeit import default_timer
from itertools import accumulate
from scipy.spatial import cKDTree
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


from pcno.geo_utility import preprocess_data, compute_node_weights
from pcno.pcno import compute_Fourier_modes, PCNO, PCNO_train
from pcno_utility import load_raw_data
torch.set_printoptions(precision=16)


torch.manual_seed(0)
np.random.seed(0)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



##########################################################################################################
# parser
##########################################################################################################
parser = argparse.ArgumentParser(description="Train model with different types.")
parser.add_argument("--problem_type", type=str, default="preprocess_data", choices=["preprocess_data" , "Poisson" , "Laplace"], help="Specify the problem type")
parser.add_argument("--equal_weight", type=str, default="False", help="Specify whether to use equal weight")
parser.add_argument("--train_sp_L", type=str, default="False", choices=["False" , "together" , "independently"],
                    help="type of train_sp_L (False, together, independently )")
parser.add_argument("--feature_SDF", type=str, default="False", choices=["False", "True"])
parser.add_argument("--lr_ratio", type=float, default=10, help="lr ratio for independent training for L")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size")

args = parser.parse_args()

args_dict = vars(args)
for i, (key, value) in enumerate(args_dict.items()):
    print(f"{key}: {value}")
###################################
data_path_pref = "../../data/poisson/"
shapes_list = ["lowfreq", "highfreq", "double", "hole"]
# 前处理只处理了这么多数据， 前处理储存在  pcno_data"+str(ndata_list[0])+".npz"
ndata_list = [256, 256, 256, 256]
ndata_list_prefix_sum = [0] + list(accumulate(ndata_list))
ntrain_list, ntest_list = [200,200,200,200], [50,50,50,50]
# features_list ：0: Laplace solution, 1: Poisson solution, 2: source term, 3: boundary condition, 4: boundary node indicator, 5: SDF, 
# 对于Laplacian方程，我们的输入包含：边界条件, 边界条件indicator （到边界的距离）, 输出包含：解
# 对于Laplacian方程，我们的输入包含：边界条件, 边界条件indicator （到边界的距离）, 输出包含：解
feature_in = [3,4] if args.problem_type == "Laplace" else [2,3,4]
feature_out = [0] if args.problem_type == "Laplace" else [1]
# 是否使用到边界的距离 （SDF），
if args.feature_SDF == "True":
    feature_in += [5] 
###################################
# 前处理
###################################
if args.problem_type == "preprocess_data":
    print(f"{"%" * 40} Preprocess Raw Data {"%" * 40}", flush=True)
    nodes_all_list, elems_all_list, features_all_list = [],[],[]
    for i, (ndata, shape) in enumerate(zip(ndata_list, shapes_list)):
        nodes_list, elems_list, features_list  = load_raw_data(data_path=data_path_pref+shape, ndata=ndata)
        nodes_all_list.extend(nodes_list)
        elems_all_list.extend(elems_list)
        features_all_list.extend(features_list)
    print("Preprocessing data")
    nnodes, node_mask, nodes, node_measures_raw, features, directed_edges, edge_gradient_weights = preprocess_data(nodes_all_list, elems_all_list, features_all_list)
    node_measures, node_weights = compute_node_weights(nnodes,  node_measures_raw,  equal_measure = False)
    node_equal_measures, node_equal_weights = compute_node_weights(nnodes,  node_measures_raw,  equal_measure = True)
    np.savez_compressed(data_path_pref+"pcno_data"+str(ndata_list[0])+".npz", \
                        nnodes=nnodes, node_mask=node_mask, nodes=nodes, \
                        node_measures_raw = node_measures_raw, \
                        node_measures=node_measures, node_weights=node_weights, \
                        node_equal_measures=node_equal_measures, node_equal_weights=node_equal_weights, \
                        features=features, \
                        directed_edges=directed_edges, edge_gradient_weights=edge_gradient_weights) 
    exit()
else:
    # load data 
    
    equal_weights = args.equal_weight.lower() == "true"

    data = np.load(data_path_pref+"pcno_data"+str(ndata_list[0])+".npz")
    nnodes, node_mask, nodes = data["nnodes"], data["node_mask"], data["nodes"]
    node_measures = data["node_measures"]
    node_weights = data["node_equal_weights"] if equal_weights else data["node_weights"]
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
#!! compress measures
# node_weights = torch.sum(node_weights, dim=-1).unsqueeze(-1)

features = torch.from_numpy(features.astype(np.float32))
directed_edges = torch.from_numpy(directed_edges.astype(np.int64))
edge_gradient_weights = torch.from_numpy(edge_gradient_weights.astype(np.float32))

nodes_input = nodes.clone() 


train_indices = [idx for start, ntrain in zip(ndata_list_prefix_sum[0:-1], ntrain_list) for idx in range(start, start + ntrain)]
test_indices  = [idx for end, ntest in zip(ndata_list_prefix_sum[1:], ntest_list) for idx in range(end - ntest, end)] 

x_train, x_test = torch.cat((features[train_indices][:, :, feature_in], nodes_input[train_indices, ...], node_rhos[train_indices, ...]), -1), torch.cat((features[test_indices][:, :, feature_in], nodes_input[test_indices, ...], node_rhos[test_indices, ...]),-1)
aux_train       = (node_mask[train_indices,...], nodes[train_indices,...], node_weights[train_indices,...], directed_edges[train_indices,...], edge_gradient_weights[train_indices,...])
aux_test        = (node_mask[test_indices,...],  nodes[test_indices,...],  node_weights[test_indices,...],  directed_edges[test_indices,...],  edge_gradient_weights[test_indices,...])
y_train, y_test = features[train_indices][:, :, feature_out],     features[test_indices][:, :, feature_out]


k_max = 16
ndim = 2
print("Maximum Lx, Ly = ", torch.max(torch.max(nodes_input, dim=1).values - torch.min(nodes_input, dim=1).values, dim=0))

#!! compress measures
modes = compute_Fourier_modes(ndim, [k_max,k_max, k_max,k_max], [4.0,4.0, 4.0,4.0])

modes = torch.tensor(modes, dtype=torch.float).to(device)


train_sp_L = False if args.train_sp_L == "False" else args.train_sp_L
#!! compress measures
model = PCNO(ndim, modes, nmeasures=2,
               layers=[128,128,128,128,128],
               fc_dim=128,
               in_dim=x_train.shape[-1], out_dim=y_train.shape[-1], 
               train_sp_L = train_sp_L,
               act="gelu").to(device)



epochs = 500
base_lr = 0.001
lr_ratio = args.lr_ratio
scheduler = "OneCycleLR"
weight_decay = 1.0e-4
batch_size = args.batch_size

normalization_x = False 
normalization_y = True 
normalization_dim_x = []
normalization_dim_y = []
non_normalized_dim_x = 0
non_normalized_dim_y = 0


config = {"train" : {"base_lr": base_lr, "lr_ratio": lr_ratio, "weight_decay": weight_decay, "epochs": epochs, "scheduler": scheduler,  "batch_size": batch_size, 
                     "normalization_x": normalization_x,"normalization_y": normalization_y, 
                     "normalization_dim_x": normalization_dim_x, "normalization_dim_y": normalization_dim_y, 
                     "non_normalized_dim_x": non_normalized_dim_x, "non_normalized_dim_y": non_normalized_dim_y}
                     }




train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = PCNO_train(
    x_train, aux_train, y_train, x_test, aux_test, y_test, config, model, save_model_name="./PCNO_" + args.problem_type + "_"+str(ndata_list[0])+"_model"
)
