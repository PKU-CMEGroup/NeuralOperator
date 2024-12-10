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



downsample_ratio = 1
n_train = 1000
n_test = 200
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


###################################
# load data
###################################
data_path = "../../data/airfoil/"
coordx = np.load(data_path+"NACA_Cylinder_X.npy")
coordy = np.load(data_path+"NACA_Cylinder_Y.npy")
data_in = np.stack((coordx, coordy), axis=3)
data_out = np.load(data_path+"NACA_Cylinder_Q.npy")[:,4,:,:] #density, velocity 2d, pressure, mach number

print("data_in.shape:" , data_in.shape)
print("data_out.shape", data_out.shape)

SURFACE_ONLY = False
if SURFACE_ONLY:
    cny, cnx = 1, 50 # how many layer to compute
    data_in_ds  = data_in[:,  cnx:-cnx:downsample_ratio, 0:cny:downsample_ratio, :]
    data_out_ds = data_out[:, cnx:-cnx:downsample_ratio, 0:cny:downsample_ratio, np.newaxis]
    Lx = Ly = 1.0
    print("SURFACE ONLY Lx, Ly = ", Lx, Ly)
else:
    data_in_ds  = data_in[:,  ::downsample_ratio, 0::downsample_ratio, :]
    data_out_ds = data_out[:, ::downsample_ratio, 0::downsample_ratio, np.newaxis]
    Lx = Ly = 4.0
    print("Lx, Ly = ", Lx, Ly)



data_in_ds  = torch.from_numpy(data_in[:,  ::downsample_ratio, 0::downsample_ratio, :].reshape(data_in.shape[0], -1, data_in.shape[-1]).astype(np.float32))
data_out_ds = torch.from_numpy(data_out[:, ::downsample_ratio, 0::downsample_ratio, np.newaxis].reshape(data_out.shape[0], -1, 1).astype(np.float32))
nodes = torch.from_numpy(np.copy(data_in_ds).astype(np.float32))
node_weights = torch.from_numpy((np.ones(nodes[...,0:1].shape) / (nodes.shape[1]*nodes.shape[2])).astype(np.float32))
node_mask = torch.from_numpy(np.ones(nodes[...,0:1].shape, dtype=int))



x_train, x_test = data_in_ds[:n_train,...],     data_in_ds[-n_test:,...]
aux_train       = (node_mask[0:n_train,...], nodes[0:n_train,...], node_weights[0:n_train,...])
aux_test        = (node_mask[-n_test:,...],  nodes[-n_test:,...],  node_weights[-n_test:,...])
y_train, y_test = data_out_ds[:n_train,...],    data_out_ds[-n_test:,...]
print("x_train.shape: ",x_train.shape)
print("y_train.shape: ",y_train.shape)



kx_max, ky_max = 32, 16
ndim = 2

modes = compute_Fourier_modes(ndim, [kx_max,ky_max], [Lx, Ly])
modes = torch.tensor(modes, dtype=torch.float).to(device)
model = GeoFNO(ndim, modes,
               layers=[128,128,128,128,128],
               fc_dim=128,
               in_dim=2, out_dim=1,
               act='gelu').to(device)


epochs = 500
base_lr = 0.001
scheduler = "OneCycleLR"
weight_decay = 1.0e-4
batch_size=20

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
    x_train, aux_train, y_train, x_test, aux_test, y_test, config, model, save_model_name="./GeoFNO_darcy_model"
)





