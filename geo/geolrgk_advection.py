import torch
import sys
import numpy as np
import os
from scipy.io import loadmat

script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_dir)
sys.path.append("../")
np.random.seed(0)
torch.manual_seed(0)
torch.set_printoptions(precision=16)

from geo.geofno import LowrankGeoGkNO, GeoFNO_train, compute_Fourier_modes


downsample_ratio = 1
n_train = 1000
n_test = 200
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = 100
base_lr = 0.001
scheduler = "OneCycleLR"
weight_decay = 1.0e-4
batch_size = 20
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
data_path = "../data/advection_1d/advection1d_4_512"

data = loadmat(data_path)
data_in = data["u"].T
data_out = data["v"].T
grid = data["grid"].T
print("data_in.shape:", data_in.shape)
print("data_out.shape:", data_out.shape)


data_in_ds = data_in[:, 0::downsample_ratio, np.newaxis]
data_out_ds = data_out[:, 0::downsample_ratio, np.newaxis]
grid_ds = grid[:, 0::downsample_ratio]


Nx = grid_ds.shape[1]
weights = np.zeros_like(grid_ds)
weights[:, 1:-1] = (grid_ds[:, 2:] - grid_ds[:, :-2]) / 2
weights[:, 0] = (grid_ds[:, 0] + grid_ds[:, 1]) / 2
weights[:, -1] = 1 - (grid_ds[:, -1] + grid_ds[:, -2]) / 2
mask = np.ones(Nx)

# x_train, y_train are [n_data, n_x, n_channel] arrays
x_train = torch.from_numpy(
    np.concatenate(
        (
            data_in_ds[0:n_train, :, :],
            grid_ds[0:n_train, :, np.newaxis],
            weights[0:n_train, :, np.newaxis],
            np.tile(mask, (n_train, 1))[:, :, np.newaxis],
        ),
        axis=2,
    ).astype(np.float32)
)
y_train = torch.from_numpy(data_out_ds[0:n_train, :, :].astype(np.float32))
# x_test, y_test are [n_data, n_x, n_channel] arrays
x_test = torch.from_numpy(
    np.concatenate(
        (
            data_in_ds[-n_test:, :, :],
            grid_ds[-n_test:, :, np.newaxis],
            weights[-n_test:, :, np.newaxis],
            np.tile(mask, (n_test, 1))[:, :, np.newaxis],
        ),
        axis=2,
    ).astype(np.float32)
)
y_test = torch.from_numpy(data_out_ds[-n_test:, :, :].astype(np.float32))
print("x_train.shape: ", tuple(x_train.shape))
print("y_train.shape: ", tuple(y_train.shape))

L = 1.0
kx_max = 256  # (-32,...,32)
ndim = 1
k_pairs = compute_Fourier_modes(ndim, kx_max, L)
k_pairs = torch.tensor(k_pairs, dtype=torch.float).to(device)
model = LowrankGeoGkNO(
    ndim,
    k_pairs,
    L=[1.0],
    should_learn_L=False,
    rank=16,
    layers=[128, 128, 128, 128, 128],
    fc_dim=128,
    in_dim=4,
    out_dim=1,
    act="gelu",
).to(device)


train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = GeoFNO_train(
    x_train, y_train, x_test, y_test, config, model, should_print_L=True
)
