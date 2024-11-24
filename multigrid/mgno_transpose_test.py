import torch
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_dir)
print("dir now:", script_dir)
sys.path.append("../")

from baselines.mgno import MgNO, MgNO_train

torch.set_printoptions(precision=16)
torch.manual_seed(0)
np.random.seed(0)


downsample_ratio = 10
n_train = 1000
n_test = 200
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_choice = "in"
###################################
# load data
###################################
data_path = "../data/darcy_2d/piececonst_r421_N1024_smooth1"
data1 = loadmat(data_path)
data_path = "../data/darcy_2d/piececonst_r421_N1024_smooth2"
data2 = loadmat(data_path)
if data_choice == "in":
    data = np.vstack((data1["coeff"], data2["coeff"]))  # shape: 2048,421,421
elif data_choice == "out":
    data = np.vstack((data1["sol"], data2["sol"]))  # shape: 2048,421,421
print("data.shape:", data.shape)

data_ds = data[:, 0::downsample_ratio, 0::downsample_ratio]
Np_ref = data.shape[1]
grid_1d = np.linspace(0, 1, Np_ref)
gridx, gridy = np.meshgrid(grid_1d, grid_1d)
gridx = gridx[0::downsample_ratio, 0::downsample_ratio]
gridy = gridy[0::downsample_ratio, 0::downsample_ratio]

x_train = torch.from_numpy(data_ds[0:n_train, np.newaxis, :, :].astype(np.float32))
y_train = x_train.transpose(2, 3)
x_test = torch.from_numpy(data_ds[-n_test:, np.newaxis, :, :].astype(np.float32))
y_test = x_test.transpose(2, 3)

print("x_train.shape: ", x_train.shape)
print("y_train.shape: ", y_train.shape)

# fig, ax = plt.subplots(ncols=3, nrows=2)
# for i in range(3):
#     ax[0, i].pcolormesh(gridx, gridy, x_train[5 * i, 0, ...])
#     ax[1, i].pcolormesh(gridx, gridy, y_train[5 * i, 0, ...])
# plt.show()

###################################
# construct model and train
###################################
print("")
model = MgNO(
    num_layer=5,
    num_channel_u=32,
    num_channel_f=1,
    num_iteration=[1, 1, 1, 1, 2],
    activation="gelu",
).to(device)


config = {
    "train": {
        "base_lr": 5e-04,
        "weight_decay": 1.0e-4,
        "epochs": 500,
        "scheduler": "OneCycleLR",
        "batch_size": 8,
        "normalization_x": True,
        "normalization_y": True,
        "normalization_dim": [],
    }
}

train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = MgNO_train(
    x_train, y_train, x_test, y_test, config, model
)

