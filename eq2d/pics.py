import torch
import sys
import numpy as np
from scipy.io import loadmat
import yaml
import os
import matplotlib.pyplot as plt


script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_dir)
sys.path.append("../")
from models import compute_2dFourier_bases, compute_2dpca_bases, UnitGaussianNormalizer

torch.set_printoptions(precision=16)


torch.manual_seed(0)
np.random.seed(0)

###################################
# load configs
###################################
with open("config_2D.yml", "r", encoding="utf-8") as f:
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
data_out_ds_test = data_out[-n_test:, 0::downsample_ratio, 0::downsample_ratio]


nx = grid_x_ds.shape[1]
ny = grid_x_ds.shape[0]
n = nx * ny
boundary_indices = (
    list(range(nx))
    + list(range(n - nx, n - 1))
    + list(range(nx, n - nx, nx))
    + list(range(2 * nx - 1, n, nx))
)
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


k_max = max(config_model["GkNN_modes"])
Np = (Np_ref + downsample_ratio - 1) // downsample_ratio
gridx, gridy, fbases, weights = compute_2dFourier_bases(Np, Np, k_max, L, L)
fbases = fbases.reshape(-1, k_max)
weights = weights.reshape(-1)
wfbases = fbases * np.tile(weights, (k_max, 1)).T
bases_fourier = torch.from_numpy(fbases.astype(np.float32)).to(device)
wbases_fourier = torch.from_numpy(wfbases.astype(np.float32)).to(device)


####################################
# compute pca bases
####################################
k_max = max(config_model["GkNN_modes"])
Np = (Np_ref + downsample_ratio - 1) // downsample_ratio
pca_data = data_out_ds.reshape((data_out_ds.shape[0], -1))
bases_pca, wbases_pca = compute_2dpca_bases(Np, k_max, L, pca_data)

pca_data_test = data_out_ds_test.reshape((data_out_ds_test.shape[0], -1))
bases_pca_test, wbases_pca_test = compute_2dpca_bases(Np, k_max, L, pca_data_test)

bases_pca_delta = torch.abs(bases_pca - bases_pca_test)
print(bases_pca.size(), bases_pca_delta.size())
print(bases_pca_delta)
# print(bases_pca_delta.mean(dim=0))

fig, ax = plt.subplots(ncols=10, nrows=3)
for i in range(5):
    ax[0, i].imshow(
        bases_pca_test.reshape(Np, Np, k_max)[..., 5 * i],
        cmap="plasma",
        interpolation="nearest",
    )
    ax[1, i].imshow(
        bases_pca.reshape(Np, Np, k_max)[..., 5 * i],
        cmap="plasma",
        interpolation="nearest",
    )
    ax[2, i].imshow(
        bases_pca_delta.reshape(Np, Np, k_max)[..., 5 * i],
        cmap="plasma",
        interpolation="nearest",
    )


normalization_x, normalization_y, normalization_dim = (
    config["train"]["normalization_x"],
    config["train"]["normalization_y"],
    config["train"]["normalization_dim"],
)

if normalization_y:
    y_normalizer = UnitGaussianNormalizer(y_train, dim=normalization_dim)
    y_train = y_normalizer.encode(y_train)
    y_test = y_normalizer.encode(y_test)

pca_data = y_test.reshape((y_test.shape[0], -1))
bases_pca, wbases_pca = compute_2dpca_bases(Np, k_max, L, pca_data)

for i in range(5):
    im = ax[0, 5 + i].imshow(
        bases_pca.reshape(Np, Np, k_max)[..., 5 * i],
        cmap="plasma",
        interpolation="nearest",
    )

pca_data1 = y_train.reshape((y_train.shape[0], -1))
bases_pca1, wbases_pca1 = compute_2dpca_bases(Np, k_max, L, pca_data1)
bases_pca_delta = torch.abs(bases_pca1 - bases_pca)

for i in range(5):
    im = ax[1, 5 + i].imshow(
        bases_pca1.reshape(Np, Np, k_max)[..., 5 * i],
        cmap="plasma",
        interpolation="nearest",
    )
    ax[2, i + 5].imshow(
        bases_pca_delta.reshape(Np, Np, k_max)[..., 5 * i],
        cmap="plasma",
        interpolation="nearest",
    )


fig.colorbar(im, ax=ax.ravel().tolist())

plt.show()


def f_transformed(x, bases, wbases):
    x_co = torch.einsum("bcx,xk->bck", x, wbases)
    x1 = torch.real(torch.einsum("bck,xk->bcx", x_co, bases))
    delta = torch.abs(x - x1)
    return x1, delta
