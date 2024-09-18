import torch
import sys
import numpy as np
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


downsample_ratio = 14
n_train = 1000
n_test = 200
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


###################################
# load data
###################################
data_path = "../data/darcy_2d/piececonst_r421_N1024_smooth1"
data1 = loadmat(data_path)
data_path = "../data/darcy_2d/piececonst_r421_N1024_smooth2"
data2 = loadmat(data_path)
data_in = np.vstack((data1["coeff"], data2["coeff"]))  # shape: 2048,421,421
data_out = np.vstack((data1["sol"], data2["sol"]))  # shape: 2048,421,421
print("data_in.shape:", data_in.shape)
print("data_out.shape", data_out.shape)


data_in_ds = data_in[:, 0::downsample_ratio, 0::downsample_ratio]
data_out_ds = data_out[:, 0::downsample_ratio, 0::downsample_ratio]


x_train = torch.from_numpy(data_in_ds[0:n_train, np.newaxis, :, :].astype(np.float32))
y_train = torch.from_numpy(data_out_ds[0:n_train, np.newaxis, :, :].astype(np.float32))
x_test = torch.from_numpy(data_in_ds[-n_test:, np.newaxis, :, :].astype(np.float32))
y_test = torch.from_numpy(data_out_ds[-n_test:, np.newaxis, :, :].astype(np.float32))

print("x_train.shape: ", x_train.shape)
print("y_train.shape: ", y_train.shape)


###################################
# construct model and train
###################################
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
