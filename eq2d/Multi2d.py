import torch
import sys
import numpy as np
from scipy.io import loadmat
import yaml
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_dir)
sys.path.append("../")


from models import FNN_train, compute_2dFourier_bases, compute_2dpca_bases

# from models.MultiGkNN import MultiGalerkinNN,SimpleMultiGalerkinNN
from models.MultiGkNN1 import MultiGalerkinNN1

torch.set_printoptions(precision=16)


torch.manual_seed(0)
np.random.seed(0)

###################################
# load configs
###################################
with open("Multi2D.yml", "r", encoding="utf-8") as f:
    config = yaml.full_load(f)

config = config["FFT_2D"]
config = dict(config)
config_data, config_train = (
    config["data"],
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
data = loadmat(data_path)

data_in = data["coeff"]  # shape: 1024,421,421
data_out = data["sol"]  # shape: 1024,421,421
print("data_in.shape:", data_in.shape)
print("data_out.shape", data_out.shape)

Np_ref = data_in.shape[1]
grid_1d = np.linspace(0, L, Np_ref)
grid_x, grid_y = np.meshgrid(grid_1d, grid_1d)

data_in_ds = data_in[0:n_train, 0::downsample_ratio, 0::downsample_ratio]
grid_x_ds = grid_x[0::downsample_ratio, 0::downsample_ratio]
grid_y_ds = grid_y[0::downsample_ratio, 0::downsample_ratio]
data_out_ds = data_out[0:n_train, 0::downsample_ratio, 0::downsample_ratio]

nx = grid_x_ds.shape[1]
ny = grid_x_ds.shape[0]
n = nx * ny
boundary_indices = (
    list(range(nx))
    + list(range(n - nx, n - 1))
    + list(range(nx, n - nx, nx))
    + list(range(2 * nx - 1, n, nx))
)
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
# compute bases
###################################

modes_list = [32, 32, 32, 32]
k_max = max(modes_list)
Np = (Np_ref + downsample_ratio - 1) // downsample_ratio
gridx, gridy, fbases, weights = compute_2dFourier_bases(Np, Np, k_max, L, L)
fbases = fbases.reshape(-1, k_max)
weights = weights.reshape(-1)
wfbases = fbases * np.tile(weights, (k_max, 1)).T
bases_fourier = torch.from_numpy(fbases.astype(np.float32)).to(device)
wbases_fourier = torch.from_numpy(wfbases.astype(np.float32)).to(device)

k_max = max(modes_list)
Np = (Np_ref + downsample_ratio - 1) // downsample_ratio
pca_data = data_out_ds.reshape((data_out_ds.shape[0], -1))
print("Start SVD with data shape: ", pca_data.shape)
bases_pca, wbases_pca = compute_2dpca_bases(Np, k_max, L, pca_data)
bases_pca, wbases_pca = bases_pca.to(device), wbases_pca.to(device)


###################################
# construct model and train
###################################
# model = MultiGalerkinNN(
#     bases_fourier,
#     wbases_fourier,
#     modes_list,
#     dim_physic=2,
#     a_channels_list=[16, 16, 16, 16],
#     u_channels_list=[16, 16, 16, 16],
#     f_channels_list=[8, 8, 8, 8],
#     stride=2,
#     kernel_size_R=5,
#     kernel_size_P=5,
#     padding_R=2,
#     padding_P=0,
# ).to(device)

# model = SimpleMultiGalerkinNN(
#     bases_fourier,
#     wbases_fourier,
#     modes_list,
#     dim_physic=2,
#     a_channels_list=[16, 16, 16, 16],
#     u_channels_list=[16, 16, 16, 16],
#     f_channels_list=[8, 8, 8, 8],
#     stride=2,
# ).to(device)

model = MultiGalerkinNN1(
    bases_fourier,
    wbases_fourier,
    modes_list,
    dim_physic=2,
    a_channels_list=[16, 16, 16, 16],
    u_channels_list=[16, 16, 16, 16],
    f_channels_list=[8, 8, 8, 8],
    stride=2,
    kernel_size_R=3,
).to(device)


print("Start training ")
train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = FNN_train(
    x_train,
    y_train,
    x_test,
    y_test,
    config,
    model,
    boundary_indices,
    save_model_name=False,
)
