import torch
import sys
import numpy as np
import os
import gc
from scipy.io import loadmat

script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_dir)
sys.path.append("../")
np.random.seed(0)
torch.manual_seed(0)
torch.set_printoptions(precision=16)

from geo.geofno import LowrankGeoGkNO, GeoFNO_train, compute_Fourier_modes


downsample_ratio = 2
n_train = 1000
n_test = 200
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = 1000
base_lr = 0.001
scheduler = "OneCycleLR"
weight_decay = 1.0e-4
batch_size = 8

normalization_x = True
normalization_y = True
normalization_dim = []

config = {
    "train": {
        "base_lr": base_lr,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "scheduler": scheduler,
        "batch_size": batch_size,
        "normalization_x": normalization_x,
        "normalization_y": normalization_y,
        "normalization_dim": normalization_dim,
    }
}


###################################
# load data
###################################
data_path1 = "../data/darcy_2d/piececonst_r421_N1024_smooth1"
data_path2 = "../data/darcy_2d/piececonst_r421_N1024_smooth2"
data1 = loadmat(data_path1)
coeff1 = data1["coeff"]
sol1 = data1["sol"]
del data1
data2 = loadmat(data_path2)
coeff2 = data2["coeff"][:300, ...]
sol2 = data2["sol"][:300, ...]
del data2
gc.collect()
data_in = np.vstack((coeff1, coeff2))  # shape: 1324,421,421
data_out = np.vstack((sol1, sol2))  # shape: 1324,421,421


print("data_in.shape:", data_in.shape)
print("data_out.shape", data_out.shape)

Np_ref = data_in.shape[1]
Np = 1 + (Np_ref - 1) // downsample_ratio
L = 1.0
grid_1d = np.linspace(0, L, Np)
grid_x_ds, grid_y_ds = np.meshgrid(grid_1d, grid_1d)
grid_x_ds, grid_y_ds = grid_x_ds.T, grid_y_ds.T


data_in_ds = data_in[:, 0::downsample_ratio, 0::downsample_ratio]
data_out_ds = data_out[:, 0::downsample_ratio, 0::downsample_ratio]

weights = np.ones(grid_x_ds.shape) / (grid_x_ds.shape[0] * grid_x_ds.shape[1])
mask = np.ones(grid_x_ds.shape)

# x_train, y_train are [n_data, n_x, n_channel] arrays
x_train = torch.from_numpy(
    np.stack(
        (
            data_in_ds[0:n_train, :, :],
            np.tile(grid_x_ds, (n_train, 1, 1)),
            np.tile(grid_y_ds, (n_train, 1, 1)),
            np.tile(weights, (n_train, 1, 1)),
            np.tile(mask, (n_train, 1, 1)),
        ),
        axis=-1,
    ).astype(np.float32)
)
y_train = torch.from_numpy(data_out_ds[0:n_train, :, :, np.newaxis].astype(np.float32))
# x_test, y_test are [n_data, n_x, n_channel] arrays
x_test = torch.from_numpy(
    np.stack(
        (
            data_in_ds[-n_test:, :, :],
            np.tile(grid_x_ds, (n_test, 1, 1)),
            np.tile(grid_y_ds, (n_test, 1, 1)),
            np.tile(weights, (n_test, 1, 1)),
            np.tile(mask, (n_test, 1, 1)),
        ),
        axis=-1,
    ).astype(np.float32)
)
y_test = torch.from_numpy(data_out_ds[-n_test:, :, :, np.newaxis].astype(np.float32))

x_train = x_train.reshape(x_train.shape[0], -1, x_train.shape[-1])
x_test = x_test.reshape(x_test.shape[0], -1, x_test.shape[-1])
y_train = y_train.reshape(y_train.shape[0], -1, y_train.shape[-1])
y_test = y_test.reshape(y_test.shape[0], -1, y_test.shape[-1])
print("x_train.shape: ", tuple(x_train.shape))
print("y_train.shape: ", tuple(y_train.shape))


k_max = 16
ndim = 2
L = [1.0, 1.0]
k_pairs = compute_Fourier_modes(ndim, [k_max, k_max], L)
k_pairs = torch.tensor(k_pairs, dtype=torch.float).to(device)
model = LowrankGeoGkNO(
    ndim,
    k_pairs,
    L=L,
    should_learn_L=False,
    rank=4,
    layers=[128, 128, 128, 128, 128],
    fc_dim=128,
    in_dim=3,
    out_dim=1,
    act="gelu",
).to(device)


train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = GeoFNO_train(
    x_train, y_train, x_test, y_test, config, model, should_print_L=True
)