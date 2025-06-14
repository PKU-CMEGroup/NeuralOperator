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
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utility.normalizer import UnitGaussianNormalizer
from utility.losses import LpLoss
from pcno_utility import plot_solution, get_median_index

from pcno.pcno import compute_Fourier_modes, PCNO

torch.set_printoptions(precision=16)


torch.manual_seed(0)
np.random.seed(0)



    

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
##########################################################################################################
# parser
##########################################################################################################
parser = argparse.ArgumentParser(description="Train model with different types.")
parser.add_argument("--problem_type", type=str, default="Laplace", choices=["Poisson" , "Laplace"], help="Specify the problem type")
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
ndata_list = [256, 256, 256, 256]
ndata_list_prefix_sum = [0] + list(accumulate(ndata_list))
ntrain_list, ntest_list = [200,200,200,200], [50,50,50,50]
# features_list ：Laplace solution, Poisson solution, source term, boundary condition, boundary node indicator, SDF, 
feature_in = [3,4] if args.problem_type == "Laplace" else [2,3,4]
feature_out = [0] if args.problem_type == "Laplace" else [1]
if args.feature_SDF == "True":
    feature_in += [5] 
###################################
# load data
###################################
data_path_pref = "../../data/poisson/"

ndata_list = [256, 256, 256, 256]
ndata_list_prefix_sum = [0] + list(accumulate(ndata_list))
shapes_list = ["lowfreq", "highfreq", "double", "hole"]
ntrain_list, ntest_list = [200,200,200,200], [50,50,50,50]
n_train, n_test = sum(ntrain_list), sum(ntest_list)
ntest_list_prefix_sum = [0] + list(accumulate(ntest_list))
################################## load data 
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

nodes_input = nodes.clone() # scale length input


train_indices = [idx for start, ntrain in zip(ndata_list_prefix_sum[0:-1], ntrain_list) for idx in range(start, start + ntrain)]
test_indices  = [idx for end, ntest in zip(ndata_list_prefix_sum[1:], ntest_list) for idx in range(end - ntest, end)] 

x_train, x_test = torch.cat((features[train_indices][:, :, feature_in], nodes_input[train_indices, ...], node_rhos[train_indices, ...]), -1), torch.cat((features[test_indices][:, :, feature_in], nodes_input[test_indices, ...], node_rhos[test_indices, ...]),-1)
aux_train       = (node_mask[train_indices,...], nodes[train_indices,...], node_weights[train_indices,...], directed_edges[train_indices,...], edge_gradient_weights[train_indices,...])
aux_test        = (node_mask[test_indices,...],  nodes[test_indices,...],  node_weights[test_indices,...],  directed_edges[test_indices,...],  edge_gradient_weights[test_indices,...])
y_train, y_test = features[train_indices][:, :, feature_out],     features[test_indices][:, :, feature_out]


k_max = 16
ndim = 2


#!! compress measures
modes = compute_Fourier_modes(ndim, [k_max,k_max, k_max,k_max], [3.0,3.0, 3.0,3.0])
modes = torch.tensor(modes, dtype=torch.float).to(device)

train_sp_L = False if args.train_sp_L == "False" else True
#!! compress measures
model = PCNO(ndim, modes, nmeasures=2,
               layers=[128,128,128,128,128],
               fc_dim=128,
               in_dim=x_train.shape[-1], out_dim=y_train.shape[-1], 
               train_sp_L = train_sp_L,
               act="gelu").to(device)
model.load_state_dict(torch.load("PCNO_" + args.problem_type + "_"+str(ndata_list[0])+"_model.pth", weights_only=True))
model = model.to(device)


normalization_x = False 
normalization_y = True 
normalization_dim_x = []
normalization_dim_y = []
non_normalized_dim_x = 0
non_normalized_dim_y = 0


if normalization_x:
    x_normalizer = UnitGaussianNormalizer(x_train, non_normalized_dim = non_normalized_dim_x, normalization_dim=normalization_dim_x)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)
    x_normalizer.to(device)
    
if normalization_y:
    y_normalizer = UnitGaussianNormalizer(y_train, non_normalized_dim = non_normalized_dim_y, normalization_dim=normalization_dim_y)
    y_train = y_normalizer.encode(y_train)
    y_test = y_normalizer.encode(y_test)
    y_normalizer.to(device)


test_rel_l2 = np.zeros(n_test)
myloss = LpLoss(d=1, p=2, size_average=False)
for i in range(n_test):
    x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights = x_test[[i],...], y_test[[i],...], aux_test[0][[i],...], aux_test[1][[i],...], aux_test[2][[i],...], aux_test[3][[i],...], aux_test[4][[i],...]
    x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights = x.to(device), y.to(device), node_mask.to(device), nodes.to(device), node_weights.to(device), directed_edges.to(device), edge_gradient_weights.to(device)

    batch_size_ = x.shape[0]
    out = model(x, (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights)) #.reshape(batch_size_,  -1)

    if normalization_y:
        out = y_normalizer.decode(out)
        y = y_normalizer.decode(y)
    out=out*node_mask #mask the padded value with 0,(1 for node, 0 for padding)
    test_rel_l2[i] = myloss(out.view(batch_size_,-1), y.view(batch_size_,-1)).item()

np.save("test_rel_l2.npy", test_rel_l2)

############################################################################
fig, ax = plt.subplots(figsize=(4, 4))
ax.hist([test_rel_l2[ntest_list_prefix_sum[i]:ntest_list_prefix_sum[i+1]] for i in range(len(shapes_list))], bins=10, 
         color=["C0", "C1", "C2", "C3"], label=shapes_list, edgecolor="black")
# 添加标签和标题
ax.set_xlabel("$L_2$ error")
ax.legend()
plt.savefig("pcno_" + args.problem_type + "_error_distribution.pdf")
############################################################################
fig, ax = plt.subplots(8, 5, figsize=(20, 20))
for i, shape in enumerate(shapes_list):
    shape_test_rel_l2 = test_rel_l2[ntest_list_prefix_sum[i]:ntest_list_prefix_sum[i+1]]
    largest_error_ind = np.argmax(shape_test_rel_l2)
    median_error_ind = get_median_index(shape_test_rel_l2)  # Get the index (or indices)
    print("For shape = ", shape, ", largest error is ", shape_test_rel_l2[largest_error_ind], " ; median error is ", shape_test_rel_l2[median_error_ind])
    
    for j, ind in enumerate([median_error_ind, largest_error_ind]):
        itest = ind + ntest_list_prefix_sum[i]
        x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights = x_test[[itest],...], y_test[[itest],...], aux_test[0][[itest],...], aux_test[1][[itest],...], aux_test[2][[itest],...], aux_test[3][[itest],...], aux_test[4][[itest],...]
        x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights = x.to(device), y.to(device), node_mask.to(device), nodes.to(device), node_weights.to(device), directed_edges.to(device), edge_gradient_weights.to(device)

        batch_size_ = x.shape[0]
        out = model(x, (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights)) #.reshape(batch_size_,  -1)
        if normalization_y:
            out = y_normalizer.decode(out)

        out = out.cpu().detach().numpy()[0,:,0]
        
        itest = ind + ndata_list[i] - ntest_list[i]
        plot_solution(args.problem_type, data_path_pref, shape, itest, out, ax[j+2*i,:], fig)

ax[0,0].set_title("Grid");ax[0,1].set_title("Boundary condition");ax[0,2].set_title("Reference");ax[0,3].set_title("Prediction");ax[0,4].set_title("error")
plt.tight_layout()
plt.savefig("pcno_" + args.problem_type + "_results.pdf")







































