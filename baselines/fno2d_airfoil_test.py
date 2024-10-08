import random
import torch
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
from timeit import default_timer
from scipy.io import loadmat

sys.path.append("../")


from baselines.fno import  FNO2d, FNO_train



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

_, nx, ny, _ = data_in.shape

data_in_ds = data_in[:, 0::downsample_ratio, 0::downsample_ratio, :]
data_out_ds = data_out[:, 0::downsample_ratio, 0::downsample_ratio, np.newaxis]

L=1.0
grid_x, grid_y = np.meshgrid(np.linspace(0, L, nx), np.linspace(0, L, ny))
grid_x, grid_y = grid_x.T, grid_y.T
grid_x_ds = grid_x[0::downsample_ratio, 0::downsample_ratio]
grid_y_ds = grid_y[0::downsample_ratio, 0::downsample_ratio]
# x_train, y_train are [n_data, n_x, n_channel] arrays

print(data_in_ds[0:n_train,:,:,:].shape, np.tile(grid_x_ds, (n_train, 1, 1))[:,:,:, np.newaxis].shape)
# x_train = torch.from_numpy(data_in_ds[0:n_train, :, :, :].astype(np.float32))
x_train = torch.from_numpy(
    np.concatenate(
        (data_in_ds[0:n_train,:,:,:], 
         np.tile(grid_x_ds, (n_train, 1, 1))[:,:,:, np.newaxis],
         np.tile(grid_y_ds, (n_train, 1, 1))[:,:,:, np.newaxis],
        ),
        axis=3,
    ).astype(np.float32)
)

y_train = torch.from_numpy(data_out_ds[0:n_train, :, :, :].astype(np.float32))
# x_test, y_test are [n_data, n_x, n_channel] arrays
# x_test = torch.from_numpy(data_in_ds[-n_test:, :, :, :].astype(np.float32))
x_test = torch.from_numpy(
    np.concatenate(
        (data_in_ds[-n_test:,:,:,:], 
         np.tile(grid_x_ds, (n_test, 1, 1))[:,:,:, np.newaxis],
         np.tile(grid_y_ds, (n_test, 1, 1))[:,:,:, np.newaxis],
        ),
        axis=3,
    ).astype(np.float32)
)




y_test = torch.from_numpy(data_out_ds[-n_test:, :, :, :].astype(np.float32))

print("x_train.shape: ",x_train.shape)
print("y_train.shape: ",y_train.shape)


kx_max, ky_max = 32,16
###################################
#construct model and train
###################################
model = FNO2d(modes1=[kx_max,kx_max,kx_max,kx_max], modes2=[ky_max,ky_max,ky_max,ky_max],
                        fc_dim=128,
                        # 4 fourier layers
                        layers=[128,128,128,128,128],
                        in_dim=2+2, 
                        out_dim=1,
                        act="gelu",
                        pad_ratio=0.0).to(device)

epochs = 1000
base_lr = 0.001
scheduler = "OneCycleLR"
weight_decay = 1.0e-4
batch_size=20

normalization_x = True
normalization_y = True
normalization_dim = []

config = {"train" : {"base_lr": base_lr, "weight_decay": weight_decay, "epochs": epochs, "scheduler": scheduler,  "batch_size": batch_size, 
                     "normalization_x": normalization_x,"normalization_y": normalization_y, "normalization_dim": normalization_dim}}

train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = FNO_train(
    x_train, y_train, x_test, y_test, config, model, save_model_name="./FNO_naca_model"
)





