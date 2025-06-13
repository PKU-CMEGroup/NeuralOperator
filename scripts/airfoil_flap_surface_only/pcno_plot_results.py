import torch
import sys
import os
import numpy as np
import argparse
from datetime import datetime
import matplotlib.pyplot as plt 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utility.normalizer import UnitGaussianNormalizer
from utility.losses import LpLoss
from pcno.pcno import compute_Fourier_modes, PCNO 

def get_median_index(arr):
    # 确保输入是一个 NumPy 数组
    arr = np.asarray(arr)
    # 获取排序后的索引
    sorted_indices = np.argsort(arr)
    # 计算中位数的索引
    mid_index = len(arr) // 2
    
    if len(arr) % 2 == 1:
        # 如果是奇数长度，返回中间元素的原始索引
        median_index = sorted_indices[mid_index]
    else:
        # 如果是偶数长度，返回中间两个元素的原始索引
        median_index_1 = sorted_indices[mid_index - 1]
        median_index_2 = sorted_indices[mid_index]
        # 通常我们不会为偶数长度的数组返回单个索引，因为中位数是两个值的平均。
        # 但是，如果你需要，你可以选择返回这两个索引或仅其中一个。
        # 这里我们简单地返回一个元组
        median_index = median_index_1
    
    return median_index


    
parser = argparse.ArgumentParser(description='Train model with different configurations and options.')
parser.add_argument('--n_train', type=int, default=1000, help='Number of training samples')
parser.add_argument('--n_test', type=int, default=400, help='Number of testing samples')
parser.add_argument('--train_type', type=str, default='mixed', choices=['standard', 'flap', 'mixed'], help='Type of training data')
parser.add_argument('--feature_type', type=str, default='pressure', choices=['pressure', 'mach'], help='Type of feature to use')
parser.add_argument('--train_inv_L_scale', type=str, default='independently', choices=['False', 'together', 'independently'], help='Type of train_inv_L_scale')
parser.add_argument('--lr_ratio', type=float, default=10.0, help='Learning rate ratio of L-parameters to main parameters')


args = parser.parse_args([])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


###################################
# load data
###################################
data_path = "../../data/airfoil_flap/"

equal_weights = False
data = np.load(data_path + "pcno_airfoil_data.npz")
nnodes, node_mask, nodes = data["nnodes"], data["node_mask"], data["nodes"]
node_weights = data["node_equal_weights"] if equal_weights else data["node_weights"]
node_measures = data["node_measures"]
node_measures_raw = data["node_measures_raw"]
indices = np.isfinite(node_measures_raw)
node_rhos = np.copy(node_weights)
node_rhos[indices] = node_rhos[indices]/node_measures[indices]
directed_edges, edge_gradient_weights = data["directed_edges"], data["edge_gradient_weights"]
features = data["features"]



print("Casting to tensor")
nnodes = torch.from_numpy(nnodes)
node_mask = torch.from_numpy(node_mask)
nodes = torch.from_numpy(nodes.astype(np.float32))
node_weights = torch.from_numpy(node_weights.astype(np.float32))
node_rhos = torch.from_numpy(node_rhos.astype(np.float32))
features = torch.from_numpy(features.astype(np.float32))
directed_edges = torch.from_numpy(directed_edges.astype(np.int64))
edge_gradient_weights = torch.from_numpy(edge_gradient_weights.astype(np.float32))


# This is important
nodes_input = nodes.clone()
nodes_input = torch.cat([nodes_input, node_rhos], dim=-1)

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
if args.train_inv_L_scale == 'False':
    args.train_inv_L_scale = False
train_inv_L_scale = args.train_inv_L_scale
Lx, Ly = 1, 0.5
print("Lx, Ly = ", Lx, Ly)
modes = compute_Fourier_modes(ndim, [kx_max, ky_max], [Lx, Ly])
modes = torch.tensor(modes, dtype=torch.float).to(device)
model = PCNO(ndim, modes, nmeasures=1,
             layers=[128, 128, 128, 128, 128],
             fc_dim=128,
             in_dim=3, out_dim=1,
             inv_L_scale_hyper = [train_inv_L_scale, 0.5, 2.0],
             act='gelu').to(device)

epochs = 500
base_lr = 0.001
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

model_name = "PCNO_airfoil_flap_mixed_n1000_20250327_190444.pth"
model.load_state_dict(torch.load(model_name, weights_only=True, map_location=device))
model = model.to(device)



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




       
myloss = LpLoss(d=1, p=2, size_average=False)
######################################################### TRAIN ERROR ####################################################
train_rel_l2 = np.zeros(n_train)
for i in range(n_train):
    x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights = x_train[[i],...], y_train[[i],...], aux_train[0][[i],...], aux_train[1][[i],...], aux_train[2][[i],...], aux_train[3][[i],...], aux_train[4][[i],...]
    x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights = x.to(device), y.to(device), node_mask.to(device), nodes.to(device), node_weights.to(device), directed_edges.to(device), edge_gradient_weights.to(device)

    batch_size_ = x.shape[0]
    out = model(x, (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights)) #.reshape(batch_size_,  -1)

    if normalization_y:
        out = y_normalizer.decode(out)
        y = y_normalizer.decode(y)
    out=out*node_mask #mask the padded value with 0,(1 for node, 0 for padding)
    train_rel_l2[i] = myloss(out.view(batch_size_,-1), y.view(batch_size_,-1)).item()


######################################################### TEST ERROR ####################################################
test_rel_l2 = np.zeros(n_test)
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

fig, ax = plt.subplots(1, 2, figsize=(12,6))
ax[0].plot(train_rel_l2, color="C0")
ax[1].plot(test_rel_l2, color="C0")
ax[0].set_title("Relative Training Error")
ax[1].set_title("Relative Test Error")
plt.tight_layout()
plt.savefig("Airfoil_Error.pdf")



######################################################### TEST ERROR ####################################################
largest_error_ind = np.argmax(test_rel_l2)
median_error_ind = get_median_index(test_rel_l2)  # Get the index (or indices)
print("largest error is ", test_rel_l2[largest_error_ind], " ; median error is ", test_rel_l2[median_error_ind])
print("largest error index is ", largest_error_ind, " ; median error index is ", median_error_ind)
    
for i in [largest_error_ind, median_error_ind]:

    x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights = x_test[[i],...], y_test[[i],...], aux_test[0][[i],...], aux_test[1][[i],...], aux_test[2][[i],...], aux_test[3][[i],...], aux_test[4][[i],...]
    x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights = x.to(device), y.to(device), node_mask.to(device), nodes.to(device), node_weights.to(device), directed_edges.to(device), edge_gradient_weights.to(device)
    
    batch_size_ = x.shape[0]
    out = model(x, (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights)) #.reshape(batch_size_,  -1)
    
    if normalization_y:
        out = y_normalizer.decode(out)
        y = y_normalizer.decode(y)
    out=out*node_mask #mask the padded value with 0,(1 for node, 0 for padding)
        
        
    y = y.cpu().detach().numpy()[0,node_mask[0,:,0]==1]
    out = out.cpu().detach().numpy()[0,node_mask[0,:,0]==1,0]
    points = nodes.cpu().detach().numpy()[0,node_mask[0,:,0]==1,:]  # Extract (x, y) coordinates
    
    # Nodal values: scalar (e.g., temperature) and vector (e.g., displacement)
    pressure_ref = y  # Scalar values at each node
    pressure_pred = out  # Scalar values at each node
    pressure_error = out - y  # Scalar values at each node
    
    # Convert nodes to meshio-compatible format
    if i < m_test:
        elems = np.load("../../data/airfoil_flap/Airfoil_flap_data/airfoil_mesh/elems_%05d"%(test_index[i])+".npy")
    else:
        elems = np.load("../../data/airfoil_flap/Airfoil_data/airfoil_mesh/elems_%05d"%(test_index[i] - ndata1)+".npy")
        
    fig, axs = plt.subplots(4, 1, figsize=(10,6), sharex=True)
    airfoil = points[elems, :] 
    # segment k is airfoil[k, 0, :]-[k, 1, :]
    axs[0].plot(airfoil[:,:,0].T, airfoil[:,:,1].T, "-o", color="C0", markersize=2)
    # # segment k is airfoil[k, 0, :]-[k, 1, :]
    axs[1].scatter(points[:,0], pressure_ref, color="C0", s=1)
    axs[1].plot(airfoil[:,:,0].T, pressure_ref[elems].T, color="C0")
    axs[1].set_title("Ground Truth")
    axs[2].scatter(points[:,0], pressure_pred, color="C0", s=1)
    axs[2].plot(airfoil[:,:,0].T, pressure_pred[elems].T, color="C0")
    axs[2].set_title("Prediction")
    axs[3].scatter(points[:,0], pressure_error, color="C0", s=1)
    axs[3].plot(airfoil[:,:,0].T, pressure_error[elems].T, color="C0")
    axs[3].set_title("Error")
    plt.tight_layout()
    plt.savefig("Airfoil_" + ("Largest_Error" if i==largest_error_ind else "Median_Error") + ".pdf")
