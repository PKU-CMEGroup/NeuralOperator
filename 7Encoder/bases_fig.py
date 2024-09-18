import torch
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import yaml
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_dir)

print("dir now:", script_dir)
sys.path.append("../")


from models import compute_2dFourier_bases, compute_2dpca_bases

torch.set_printoptions(precision=16)


torch.manual_seed(0)
np.random.seed(0)


###################################
# load configs
###################################
with open("config.yml", "r", encoding="utf-8") as f:
    config = yaml.full_load(f)

config = config["FFT_2D"]
config = dict(config)
config_data, config_model, config_train = (
    config["data"],
    config["model"],
    config["train"],
)
downsample_ratio = config_data["downsample_ratio"]
L = config_data["L"]
n_train = config_data["n_train"]
n_test = config_data["n_test"]
device = torch.device(config["train"]["device"])


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
grid_1d = np.linspace(0, L, Np_ref)
grid_x, grid_y = np.meshgrid(grid_1d, grid_1d)

data_in_ds = data_in[0:n_train, 0::downsample_ratio, 0::downsample_ratio]
grid_x_ds = grid_x[0::downsample_ratio, 0::downsample_ratio]
grid_y_ds = grid_y[0::downsample_ratio, 0::downsample_ratio]
data_out_ds = data_out[0:n_train, 0::downsample_ratio, 0::downsample_ratio]

# x_train, y_train are [n_data, n_x, n_channel] arrays
x_train = torch.from_numpy(
    np.stack(
        (
            data_in_ds,
            np.tile(grid_x_ds, (n_train, 1, 1)),
            np.tile(grid_y_ds, (n_train, 1, 1)),
        ),
        axis=-1,
    ).astype(np.float32)
)
y_train = torch.from_numpy(data_out_ds[:, :, :, np.newaxis].astype(np.float32))
# x_test, y_test are [n_data, n_x, n_channel] arrays
x_test = torch.from_numpy(
    np.stack(
        (
            data_in[-n_test:, 0::downsample_ratio, 0::downsample_ratio],
            np.tile(grid_x[0::downsample_ratio, 0::downsample_ratio], (n_test, 1, 1)),
            np.tile(grid_y[0::downsample_ratio, 0::downsample_ratio], (n_test, 1, 1)),
        ),
        axis=-1,
    ).astype(np.float32)
)
y_test = torch.from_numpy(
    data_out[-n_test:, 0::downsample_ratio, 0::downsample_ratio, np.newaxis].astype(
        np.float32
    )
)

x_train = x_train.reshape(
    x_train.shape[0], -1, x_train.shape[-1]
)  # shape: 800,11236,3  (11236 = 106*106 , 106-1 = (421-1) /4)
x_test = x_test.reshape(x_test.shape[0], -1, x_test.shape[-1])
y_train = y_train.reshape(y_train.shape[0], -1, y_train.shape[-1])  # shape: 800,11236,1
y_test = y_test.reshape(y_test.shape[0], -1, y_test.shape[-1])
print("x_train.shape: ", x_train.shape)
print("y_train.shape: ", y_train.shape)

###################################
# compute fourier bases
###################################
k_max = max(config_model["GkNN_modes"])
Np = (Np_ref + downsample_ratio - 1) // downsample_ratio
gridx, gridy, fbases, weights = compute_2dFourier_bases(Np, Np, k_max, L, L)
fbases = fbases.reshape(-1, k_max)
weights = weights.reshape(-1)
wfbases = fbases * np.tile(weights, (k_max, 1)).T
bases_fourier = torch.from_numpy(fbases.astype(np.float32)).to(device)
wbases_fourier = torch.from_numpy(wfbases.astype(np.float32)).to(device)

weights = torch.from_numpy(weights.astype(np.float32)).to(device)


####################################
# compute pca bases from input
####################################
k_max = 512
Np = (Np_ref + downsample_ratio - 1) // downsample_ratio
pca_data = data_in_ds.reshape((data_in_ds.shape[0], -1))
print("Start SVD with data shape: ", pca_data.shape)
bases_pca, wbases_pca = compute_2dpca_bases(Np, k_max, L, pca_data)
bases_pca_in, wbases_pca_in = bases_pca.to(device), wbases_pca.to(device)

bases_fig = bases_pca_in.reshape((Np, Np, k_max)).cpu()
# bases_fig = bases_fig[:, :, 1:]
fig, ax = plt.subplots(ncols=3, nrows=2)
c0 = ax[0, 0].pcolormesh(gridx, gridy, bases_fig[:, :, 0])
ax[0, 0].set_title("mode 0")
c1 = ax[0, 1].pcolormesh(gridx, gridy, bases_fig[:, :, 1])
ax[0, 1].set_title("mode 1")
c2 = ax[1, 0].pcolormesh(gridx, gridy, bases_fig[:, :, 2])
ax[1, 0].set_title("mode 2")
c3 = ax[1, 1].pcolormesh(gridx, gridy, bases_fig[:, :, 3])
ax[1, 1].set_title("mode 3")

index = 5
co = torch.einsum("x,xk->k", x_train[index, :, 0].to(device), wbases_pca_in)
data_in_proj = torch.einsum("k,xk->x", co, bases_pca_in).reshape((Np, Np)).cpu()

c4 = ax[0, 2].pcolormesh(gridx, gridy, data_in_ds[index, :, :])
c5 = ax[1, 2].pcolormesh(gridx, gridy, data_in_proj)

fig.colorbar(c4, ax=ax, orientation="vertical", fraction=0.05, pad=0.04)

delta = data_in_proj - data_in_ds[index, :, :]
print(data_in_proj[:3, :3])
print(data_in_ds[index, :3, :3])
plt.show()
####################################
# compute pca bases from output
####################################
# k_max = max(config_model["GkNN_modes"])
# Np = (Np_ref + downsample_ratio - 1) // downsample_ratio
# pca_data = data_out_ds.reshape((data_out_ds.shape[0], -1))
# print("Start SVD with data shape: ", pca_data.shape)
# bases_pca, wbases_pca = compute_2dpca_bases(Np, k_max, L, pca_data)
# bases_pca_out, wbases_pca_out = bases_pca.to(device), wbases_pca.to(device)

# bases_list = [
#     bases_fourier,
#     wbases_fourier,
#     bases_pca_in,
#     wbases_pca_in,
#     bases_pca_out,
#     wbases_pca_out,
# ]
