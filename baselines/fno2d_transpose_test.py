import random
import torch
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
from timeit import default_timer
from scipy.io import loadmat

sys.path.append("../")


from baselines.fno import FNO2d, FNO_train


torch.set_printoptions(precision=16)


torch.manual_seed(0)
np.random.seed(0)


downsample_ratio = 10
n_train = 1000
n_test = 200
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


###################################
# load data
###################################
data_src = "out"

data_path = "../data/darcy_2d/piececonst_r421_N1024_smooth1"
data1 = loadmat(data_path)
data_path = "../data/darcy_2d/piececonst_r421_N1024_smooth2"
data2 = loadmat(data_path)
data_in = np.vstack((data1["coeff"], data2["coeff"]))  # shape: 2048,421,421
data_out = np.vstack((data1["sol"], data2["sol"]))     # shape: 2048,421,421
print("data_in.shape:", data_in.shape)
print("data_out.shape", data_out.shape)

Np_ref = data_in.shape[1]
L = 1.0
grid_1d = np.linspace(0, L, Np_ref)
grid_x, grid_y = np.meshgrid(grid_1d, grid_1d)

data_in_ds = data_in[:, 0::downsample_ratio, 0::downsample_ratio]
grid_x_ds = grid_x[0::downsample_ratio, 0::downsample_ratio]
grid_y_ds = grid_y[0::downsample_ratio, 0::downsample_ratio]
data_out_ds = data_out[:, 0::downsample_ratio, 0::downsample_ratio]


if data_src == "in":
    x_train = torch.from_numpy(
        np.stack(
            (
                data_in_ds[0:n_train, :, :],
                np.tile(grid_x_ds, (n_train, 1, 1)),
                np.tile(grid_y_ds, (n_train, 1, 1)),
            ),
            axis=-1,
        ).astype(np.float32)
    )
    x_test = torch.from_numpy(
        np.stack(
            (
                data_in_ds[-n_test:, :, :],
                np.tile(grid_x[0::downsample_ratio,
                               0::downsample_ratio], (n_test, 1, 1)),
                np.tile(grid_y[0::downsample_ratio,
                               0::downsample_ratio], (n_test, 1, 1)),
            ),
            axis=-1,
        ).astype(np.float32)
    )
elif data_src == "out":
    x_train = torch.from_numpy(
        data_out_ds[0:n_train, :, :, np.newaxis].astype(np.float32))
    x_test = torch.from_numpy(
        data_out_ds[-n_test:, :, :, np.newaxis].astype(
            np.float32
        )
    )


y_train = torch.transpose(x_train, 1, 2).flip(2)[..., :1]
y_test = torch.transpose(x_test, 1, 2).flip(2)[..., :1]

print("x_train.shape: ", x_train.shape)
print("y_train.shape: ", y_train.shape)

# fig, ax = plt.subplots(1, 2)
# ax[0].imshow(x_train[0, ..., 0])
# ax[1].imshow(y_train[0, ..., 0])
# plt.show()

k_max = 16
###################################
# construct model and train
###################################
model = FNO2d(modes1=[k_max, k_max, k_max, k_max], modes2=[k_max, k_max, k_max, k_max],
              fc_dim=128,
              # 4 fourier layers
              layers=[128, 128, 128, 128, 128],
              in_dim=3,
              out_dim=1,
              act="gelu",
              pad_ratio=0.05).to(device)

epochs = 500
base_lr = 0.001
scheduler_gamma = 0.5
pad_ratio = 0.05
scheduler = "OneCycleLR"
weight_decay = 1.0e-4
batch_size = 8

normalization_x = True
normalization_y = True
normalization_dim = []

config = {"train": {"base_lr": base_lr, "weight_decay": weight_decay, "epochs": epochs, "scheduler": scheduler, "batch_size": batch_size,
                    "normalization_x": normalization_x, "normalization_y": normalization_y, "normalization_dim": normalization_dim}}

train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = FNO_train(
    x_train, y_train, x_test, y_test, config, model
)