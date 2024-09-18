import torch
import sys
import numpy as np
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_dir)
print("dir now:", script_dir)
sys.path.append("../")

from baselines.mgno import MgNO, MgNO_train


torch.set_printoptions(precision=16)
torch.manual_seed(0)
np.random.seed(0)


downsample_ratio = 1
n_train = 1000
n_test = 200
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
_, nx, ny, _ = data_in.shape
print("data_in.shape:", data_in.shape)
print("data_out.shape", data_out.shape)

data_in_ds = data_in[:, 0::downsample_ratio, 0::downsample_ratio, :]
data_out_ds = data_out[:, 0::downsample_ratio, 0::downsample_ratio, np.newaxis]

x_train = torch.from_numpy(data_in_ds[0:n_train, :, :, :].astype(np.float32))
y_train = torch.from_numpy(data_out_ds[0:n_train, :, :, :].astype(np.float32))
x_test = torch.from_numpy(data_in_ds[-n_test:, :, :, :].astype(np.float32))
y_test = torch.from_numpy(data_out_ds[-n_test:, :, :, :].astype(np.float32))

x_train = x_train.permute(0, 3, 1, 2)
y_train = y_train.permute(0, 3, 1, 2)
x_test = x_test.permute(0, 3, 1, 2)
y_test = y_test.permute(0, 3, 1, 2)


print("x_train.shape: ", x_train.shape)
print("y_train.shape: ", y_train.shape)


###################################
# construct model and train
###################################
model = MgNO(
    num_layer=5,
    num_channel_u=128,
    num_channel_f=2,
    num_iteration=[2, 2, 2, 2, 4],
    activation="gelu",
).to(device)

config = {
    "train": {
        "base_lr": 5e-04,
        "weight_decay": 1.0e-4,
        "epochs": 500,
        "scheduler": "OneCycleLR",
        "batch_size": 20,
        "normalization_x": True,
        "normalization_y": True,
        "normalization_dim": [],
    }
}

train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = MgNO_train(
    x_train, y_train, x_test, y_test, config, model
)
