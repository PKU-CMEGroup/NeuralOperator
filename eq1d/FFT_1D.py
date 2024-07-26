import random
import torch
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
from timeit import default_timer
from scipy.io import loadmat
import yaml
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_dir)
sys.path.append("../")

from models import FNN_train, compute_1dFourier_bases, compute_1dpca_bases
from models.Galerkin import GkNN

torch.set_printoptions(precision=16)


torch.manual_seed(0)
np.random.seed(0)


###################################
# load configs
###################################
with open("config.yml", "r", encoding="utf-8") as f:
    config = yaml.full_load(f)

config = config["FFT_1D"]
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


###################################
# load data
###################################
data_path = "../data/burgers_1d/burgers_data_R10.mat"
data = loadmat(data_path)

data_in = data["a"]
data_out = data["u"]

Ne_ref = data_in.shape[1]
grid = np.linspace(0, L, Ne_ref + 1)[:-1]

x_train = torch.from_numpy(
    np.stack(
        (
            data_in[0:n_train, 0::downsample_ratio],
            np.tile(grid[0::downsample_ratio], (n_train, 1)),
        ),
        axis=-1,
    ).astype(np.float32)
)
y_train = torch.from_numpy(
    data_out[0:n_train, 0::downsample_ratio, np.newaxis].astype(np.float32)
)

x_test = torch.from_numpy(
    np.stack(
        (
            data_in[-n_test:, 0::downsample_ratio],
            np.tile(grid[0::downsample_ratio], (n_train, 1)),
        ),
        axis=-1,
    ).astype(np.float32)
)
y_test = torch.from_numpy(
    data_out[-n_test:, 0::downsample_ratio, np.newaxis].astype(np.float32)
)


device = torch.device(config["train"]["device"])


###################################
# compute fourier bases
###################################
Ne = Ne_ref // downsample_ratio
k_max = max(config_model["GkNN_modes"])

grid, fbases, weights = compute_1dFourier_bases(Ne, k_max, L)
wfbases = fbases * np.tile(weights, (k_max, 1)).T
bases_fourier = torch.from_numpy(fbases.astype(np.float32)).to(device)
wbases_fourier = torch.from_numpy(wfbases.astype(np.float32)).to(device)


####################################
# compute pca bases
####################################
Ne = Ne_ref // downsample_ratio
k_max = max(config_model["GkNN_modes"])

pca_data = data_out[0:n_train, 0::downsample_ratio]
if config_model["pca_include_input"]:
    pca_data = np.vstack((pca_data, data_in[0:n_train, 0::downsample_ratio]))
if config_model["pca_include_grid"]:
    n_grid = 1
    pca_data = np.vstack((pca_data, np.tile(grid[0::downsample_ratio], (n_grid, 1))))
bases_pca, wbases_pca = compute_1dpca_bases(Ne, k_max, L, pca_data)
bases_pca, wbases_pca = bases_pca.to(device), wbases_pca.to(device)


bases_list = [bases_fourier, wbases_fourier, bases_pca, wbases_pca]


###################################
# construct model and train
###################################
model = GkNN(bases_list, **config_model).to(device)

print("Start training ", config_model["layer_types"])
train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = FNN_train(
    x_train, y_train, x_test, y_test, config, model, save_model_name=False
)
