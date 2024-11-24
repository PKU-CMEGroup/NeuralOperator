import torch
import sys
import numpy as np
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_dir)
sys.path.append("../")
np.random.seed(0)
torch.manual_seed(0)
torch.set_printoptions(precision=16)

from geo.geofno import GeoFNO, GeoFNO_train, compute_Fourier_modes


downsample_ratio = 1
n_train = 1000
n_test = 200
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

epochs = 500
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
data_path = "../data/airfoil/"
coordx = np.load(data_path + "NACA_Cylinder_X.npy")
coordy = np.load(data_path + "NACA_Cylinder_Y.npy")
data_in = np.stack((coordx, coordy), axis=3)
data_out = np.load(data_path + "NACA_Cylinder_Q.npy")[
    :, 4, :, :
]  # density, velocity 2d, pressure, mach number

print("data_in.shape:", data_in.shape)
print("data_out.shape", data_out.shape)


data_in_ds = data_in[:, ::downsample_ratio, 0::downsample_ratio, :]
data_out_ds = data_out[:, ::downsample_ratio, 0::downsample_ratio, np.newaxis]

# cny, cnx = 40, 10 # how many layer to compute
# data_in_ds  = data_in[:,  cnx:-cnx:downsample_ratio, 0:cny:downsample_ratio, :]
# data_out_ds = data_out[:, cnx:-cnx:downsample_ratio, 0:cny:downsample_ratio, np.newaxis]

grid_ds = data_in_ds  ####

weights = np.ones(grid_ds[..., 0:1].shape) / (data_in_ds.shape[1] * data_in_ds.shape[2])
mask = np.ones(grid_ds[..., 0:1].shape)

print(data_in_ds.shape, grid_ds.shape, weights.shape)
# x_train, y_train are [n_data, n_x, n_channel] arrays
x_train = torch.from_numpy(
    np.concatenate(
        (
            data_in_ds[0:n_train, ...],
            grid_ds[0:n_train, ...],
            weights[0:n_train, ...],
            mask[0:n_train, ...],
        ),
        axis=3,
    ).astype(np.float32)
)
y_train = torch.from_numpy(data_out_ds[0:n_train, :, :, :].astype(np.float32))
# x_test, y_test are [n_data, n_x, n_channel] arrays
x_test = torch.from_numpy(
    np.concatenate(
        (
            data_in_ds[-n_test:, ...],
            grid_ds[-n_test:, ...],
            weights[-n_test:, ...],
            mask[-n_test:, ...],
        ),
        axis=3,
    ).astype(np.float32)
)
y_test = torch.from_numpy(data_out_ds[-n_test:, :, :, :].astype(np.float32))

x_train = x_train.reshape(
    x_train.shape[0], -1, x_train.shape[-1]
)  # shape: 800,11236,3  (11236 = 106*106 , 106-1 = (421-1) /4)
x_test = x_test.reshape(x_test.shape[0], -1, x_test.shape[-1])
y_train = y_train.reshape(y_train.shape[0], -1, y_train.shape[-1])  # shape: 800,11236,1
y_test = y_test.reshape(y_test.shape[0], -1, y_test.shape[-1])
print("x_train.shape: ", tuple(x_train.shape))
print("y_train.shape: ", tuple(y_train.shape))


kx_max, ky_max = 32, 16
ndim = 2
pad_ratio = 0.05
# Lx, Ly = (1.0+pad_ratio)*(grid_ds[0:n_train,...,0].max()-grid_ds[0:n_train,...,0].min()), (1.0+pad_ratio)*(grid_ds[0:n_train,...,1].max()-grid_ds[0:n_train,...,1].min())
Lx = Ly = 4.0
print("Lx, Ly = ", Lx, Ly)
modes = compute_Fourier_modes(ndim, [kx_max, ky_max], [Lx, Ly])
modes = torch.tensor(modes, dtype=torch.float).to(device)
model = GeoFNO(
    ndim,
    modes,
    layers=[128, 128, 128, 128, 128],
    fc_dim=128,
    in_dim=2,
    out_dim=1,
    act="gelu",
).to(device)


train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = GeoFNO_train(
    x_train, y_train, x_test, y_test, config, model
)
