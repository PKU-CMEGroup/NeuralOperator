import torch
import sys
import numpy as np
from scipy.io import loadmat
import yaml
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_dir)
print("dir now:", script_dir)
sys.path.append("../")

from models.GalerkinOptions import GkNNOptions
from models import FNN_train

torch.set_printoptions(precision=16)
torch.manual_seed(0)
np.random.seed(0)


class DualWriter:
    def __init__(self, file_name):
        self.file = open(file_name, "a")
        self.terminal = sys.stdout

    def write(self, message):
        self.file.write(message)
        self.terminal.write(message)

    def flush(self):
        self.file.flush()
        self.terminal.flush()


dual_writer = DualWriter("test.log")


###################################
# load configs
###################################
with open("configs/config_standard_darcy.yml", "r", encoding="utf-8") as f:
    config = yaml.full_load(f)

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

nx = grid_x_ds.shape[1]
ny = grid_x_ds.shape[0]
n = nx * ny
bundary_indices = (
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

print("x_train.shape: ", x_train.shape)  # 800,106,106,3
print("y_train.shape: ", y_train.shape)  # 800,106,106,1


###################################
# construct model and train
###################################
print("#" * 50)
print(
    "Hidden Bases Test for Darcy2d: Adjust bases by using a modes*modes learnable matrix initilized as an identity"
)
sys.stdout = dual_writer
model = GkNNOptions(bases_list, **config_model).to(device)

for section, settings in config.items():
    print(f"--{section}--")
    for key, value in settings.items():
        print(f"{key}: {value}")

print("--Start Training--")
train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = FNN_train(
    x_train,
    y_train,
    x_test,
    y_test,
    config,
    model,
    bundary_indices,
    save_model_name=False,
)
