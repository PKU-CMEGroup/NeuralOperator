import random
import torch
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
from timeit import default_timer
from scipy.io import loadmat

sys.path.append("../")


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
data_path = "../data/airfoil/"
coordx = np.load(data_path+"NACA_Cylinder_X.npy")
coordy = np.load(data_path+"NACA_Cylinder_Y.npy")
data_in = np.stack((coordx, coordy), axis=3)
data_out = np.load(data_path+"NACA_Cylinder_Q.npy")[:,4,:,:] #density, velocity 2d, pressure, mach number

print("data_in.shape:" , data_in.shape)
print("data_out.shape", data_out.shape)


Nx_ref, Ny_ref = data_in.shape[1], data_in.shape[2]
#L=1.0
#grid_1d = np.linspace(0, L, Np_ref)
#grid_x, grid_y = np.meshgrid(grid_1d, grid_1d)
#grid_x_ds = grid_x[0::downsample_ratio, 0::downsample_ratio]
#grid_y_ds = grid_y[0::downsample_ratio, 0::downsample_ratio]
Nx, Ny = 1+(Nx_ref -  1)//downsample_ratio, 1+(Ny_ref -  1)//downsample_ratio
L = 1.0
# grid_1d = np.linspace(0, L, Np+1)[0:Np]
grid_x_1d, grid_y_1d = np.linspace(0, L, Nx+1)[0:Nx], np.linspace(0, L, Ny+1)[0:Ny]
grid_x_ds, grid_y_ds = np.meshgrid(grid_x_1d, grid_y_1d)
grid_x_ds, grid_y_ds = grid_x_ds.T, grid_y_ds.T
data_in_ds = data_in[:, 0::downsample_ratio, 0::downsample_ratio, :]
data_out_ds = data_out[:, 0::downsample_ratio, 0::downsample_ratio, np.newaxis]

weights = np.ones(grid_x_ds.shape) / (grid_x_ds.shape[0]*grid_x_ds.shape[1])
mask = np.ones(grid_x_ds.shape)

print(data_in_ds.shape, grid_x_ds.shape, grid_y_ds.shape)
# x_train, y_train are [n_data, n_x, n_channel] arrays
x_train = torch.from_numpy(
    np.concatenate(
        (
            data_in_ds[0:n_train,:,:,:],
            np.tile(grid_x_ds, (n_train, 1, 1))[:,:,:, np.newaxis],
            np.tile(grid_y_ds, (n_train, 1, 1))[:,:,:, np.newaxis],
            np.tile(weights, (n_train, 1, 1))[:,:,:, np.newaxis],
            np.tile(mask, (n_train, 1, 1))[:,:,:, np.newaxis],
        ),
        axis=3,
    ).astype(np.float32)
)
y_train = torch.from_numpy(data_out_ds[0:n_train, :, :, :].astype(np.float32))
# x_test, y_test are [n_data, n_x, n_channel] arrays
x_test = torch.from_numpy(
    np.concatenate(
        (
            data_in_ds[-n_test:, :, :,:],
            np.tile(grid_x_ds, (n_test, 1, 1))[:,:,:, np.newaxis],
            np.tile(grid_y_ds, (n_test, 1, 1))[:,:,:, np.newaxis],
            np.tile(weights, (n_test, 1, 1))[:,:,:, np.newaxis],
            np.tile(mask, (n_test, 1, 1))[:,:,:, np.newaxis],
        ),
        axis=3,
    ).astype(np.float32)
)
y_test = torch.from_numpy(
    data_out_ds[-n_test:, :, :, :].astype(
        np.float32
    )
)

x_train = x_train.reshape(x_train.shape[0], -1, x_train.shape[-1])   # shape: 800,11236,3  (11236 = 106*106 , 106-1 = (421-1) /4)
x_test = x_test.reshape(x_test.shape[0], -1, x_test.shape[-1])
y_train = y_train.reshape(y_train.shape[0], -1, y_train.shape[-1])   # shape: 800,11236,1
y_test = y_test.reshape(y_test.shape[0], -1, y_test.shape[-1])
print("x_train.shape: ",x_train.shape)
print("y_train.shape: ",y_train.shape)



kx_max, ky_max = 32, 16
ndim = 2
modes = compute_Fourier_modes(ndim, [kx_max,ky_max], [L, L])
modes = torch.tensor(modes, dtype=torch.float).to(device)
model = GeoFNO(ndim, modes,
               layers=[128,128,128,128,128],
               #layers=[1,1,1,1,1],
               fc_dim=128,
               in_dim=4, out_dim=1,
               act='gelu').to(device)


epochs = 500
base_lr = 0.001
scheduler = "OneCycleLR"
weight_decay = 1.0e-4
batch_size=20

normalization_x = True
normalization_y = True
normalization_dim = []

config = {"train" : {"base_lr": base_lr, "weight_decay": weight_decay, "epochs": epochs, "scheduler": scheduler,  "batch_size": batch_size, 
                     "normalization_x": normalization_x,"normalization_y": normalization_y, "normalization_dim": normalization_dim}}

train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = GeoFNO_train(
    x_train, y_train, x_test, y_test, config, model, save_model_name="./GeoFNO_airfoil_model"
)





