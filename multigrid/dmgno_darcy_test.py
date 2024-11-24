import torch
import sys
import numpy as np
from scipy.io import loadmat
import os


os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.append("../")

torch.set_printoptions(precision=16)
torch.manual_seed(0)
np.random.seed(0)

from multigrid.dmgno import MgNO, MgNO_train
from myutils.basics import compute_fourier2d_bases

downsample_ratio = 2
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
Np_ref = data_in.shape[1]

data_in_ds = data_in[:, 0::downsample_ratio, 0::downsample_ratio]
data_out_ds = data_out[:, 0::downsample_ratio, 0::downsample_ratio]


x_train = torch.from_numpy(
    data_in_ds[0:n_train, np.newaxis, :, :].astype(np.float32))
y_train = torch.from_numpy(
    data_out_ds[0:n_train, np.newaxis, :, :].astype(np.float32))
x_test = torch.from_numpy(
    data_in_ds[-n_test:, np.newaxis, :, :].astype(np.float32))
y_test = torch.from_numpy(
    data_out_ds[-n_test:, np.newaxis, :, :].astype(np.float32))

print("x_train.shape: ", x_train.shape)
print("y_train.shape: ", y_train.shape)

###################################
# compute fourier bases
###################################
k_max = 128
L = 1
Np = (Np_ref + downsample_ratio - 1) // downsample_ratio
bases, wbases = compute_fourier2d_bases(Np, Np, k_max, L, L, False)
bases = torch.from_numpy(bases.astype(np.float32)).to(device)
wbases = torch.from_numpy(wbases.astype(np.float32)).to(device)

###################################
# construct model and train
###################################
model = MgNO(
    num_layer=5,
    num_channel_u=32,
    num_channel_f=1,
    num_iteration=[1, 1, 1, 1, 2],
    activation="gelu",
    modes=k_max,
    bases=bases,
    wbases=wbases,
    p=0.95,
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
        "lambda_reg": 1e-03,
    }
}

train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = MgNO_train(
    x_train, y_train, x_test, y_test, config, model
)
