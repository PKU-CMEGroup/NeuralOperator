import random
import torch
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
from timeit import default_timer
from scipy.io import loadmat

sys.path.append("../../")


from baselines.geofno import  GeoFNO, GeoFNO_train, compute_Fourier_modes



torch.set_printoptions(precision=16)


torch.manual_seed(0)
np.random.seed(0)



downsample_ratio = 2
n_train = 1000
n_test = 200
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


###################################
# load data
###################################
data_path = "../../data/"

data1 = loadmat(data_path + "darcy_square/piececonst_r421_N1024_smooth1")
data2 = loadmat(data_path + "darcy_square/piececonst_r421_N1024_smooth2")
data_in = np.vstack((data1["coeff"], data2["coeff"]))  # shape: 2048,421,421
data_out = np.vstack((data1["sol"], data2["sol"]))     # shape: 2048,421,421

print("data_in.shape:" , data_in.shape)
print("data_out.shape", data_out.shape)

Np_ref = data_in.shape[1]
Np = 1 + (Np_ref -  1)//downsample_ratio
L = 1.0
grid_1d = np.linspace(0, L, Np)
grid_x_ds, grid_y_ds = np.meshgrid(grid_1d, grid_1d)
grid_x_ds, grid_y_ds = grid_x_ds.T, grid_y_ds.T

data_in_ds  = torch.from_numpy(data_in[:,  ::downsample_ratio, 0::downsample_ratio].reshape(data_in.shape[0], -1, 1).astype(np.float32))
data_out_ds = torch.from_numpy(data_out[:, ::downsample_ratio, 0::downsample_ratio].reshape(data_out.shape[0], -1, 1).astype(np.float32))
nodes = torch.from_numpy(np.stack(
        (
            np.tile(grid_x_ds, (data_in.shape[0], 1, 1)),
            np.tile(grid_y_ds, (data_in.shape[0], 1, 1)),
        ),
        axis=-1,
    ).reshape(data_in.shape[0], -1, 2).astype(np.float32))
node_weights = torch.from_numpy((np.ones(nodes[...,0:1].shape) / nodes.shape[1]).astype(np.float32))
node_mask = torch.from_numpy(np.ones(node_weights.shape, dtype=int))

nodes_input = nodes.clone()

x_train, x_test = torch.cat((data_in_ds[:n_train,...],nodes_input[:n_train,...]), -1) ,  torch.cat((data_in_ds[-n_test:,...], nodes_input[-n_test:,...]), -1)
aux_train       = (node_mask[0:n_train,...], nodes[0:n_train,...], node_weights[0:n_train,...])
aux_test        = (node_mask[-n_test:,...],  nodes[-n_test:,...],  node_weights[-n_test:,...])
y_train, y_test = data_out_ds[:n_train,...],    data_out_ds[-n_test:,...]
print("x_train.shape: ",x_train.shape)
print("y_train.shape: ",y_train.shape)



k_max = 16
ndim = 2
modes = compute_Fourier_modes(ndim, [k_max,k_max], [1.0,1.0])
modes = torch.tensor(modes, dtype=torch.float).to(device)
model = GeoFNO(ndim, modes,
               layers=[128,128,128,128,128],
               fc_dim=128,
               in_dim=3, out_dim=1,
               act='gelu').to(device)


epochs = 1000
base_lr = 0.001
scheduler = "OneCycleLR"
weight_decay = 1.0e-4
batch_size=8

normalization_x = True
normalization_y = True
normalization_dim_x = []
normalization_dim_y = []
non_normalized_dim_x = 0
non_normalized_dim_y = 0


config = {"train" : {"base_lr": base_lr, "weight_decay": weight_decay, "epochs": epochs, "scheduler": scheduler,  "batch_size": batch_size, 
                     "normalization_x": normalization_x,"normalization_y": normalization_y, 
                     "normalization_dim_x": normalization_dim_x, "normalization_dim_y": normalization_dim_y, 
                     "non_normalized_dim_x": non_normalized_dim_x, "non_normalized_dim_y": non_normalized_dim_y}
                     }

train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = GeoFNO_train(
    x_train, aux_train, y_train, x_test, aux_test, y_test, config, model, save_model_name="./GeoFNO_darcy_square_model"
)





