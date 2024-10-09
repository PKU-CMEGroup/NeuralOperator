import random
import torch
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
from timeit import default_timer
from scipy.io import loadmat
import yaml
import gc

sys.path.append("../")


from models import  PhyHGkNN_train, compute_2dFourier_bases, compute_2dpca_bases, compute_2dFourier_cbases, count_params

from models.MPhyHGkNN import MPhyHGkNN

torch.set_printoptions(precision=16)


torch.manual_seed(0)
np.random.seed(0)


###################################
# load configs
###################################
with open('config.yml', 'r', encoding='utf-8') as f:
    config = yaml.full_load(f)

config = config["Darcy"]
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
# data_path = "../data/darcy_2d/piececonst_r421_N1024_smooth1"
# data1 = loadmat(data_path)
# coeff1 = data1["coeff"]
# sol1 = data1["sol"]
# del data1
# data_path = "../data/darcy_2d/piececonst_r421_N1024_smooth2"
# data2 = loadmat(data_path)
# coeff2 = data2["coeff"][:300,:,:]
# sol2 = data2["sol"][:300,:,:]
# del data2
# gc.collect()

# data_in = np.vstack((coeff1, coeff2))  # shape: 2048,421,421
# data_out = np.vstack((sol1, sol2))     # shape: 2048,421,421

data_path = "../data/darcy_2d/piececonst_r421_N1024_smooth1"
data1 = loadmat(data_path)

data_in = data1["coeff"]
data_out = data1["sol"]

print("data_in.shape:" , data_in.shape)
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

x_train = x_train.reshape(x_train.shape[0], -1, x_train.shape[-1])   # shape: 800,11236,3  (11236 = 106*106 , 106-1 = (421-1) /4)
x_test = x_test.reshape(x_test.shape[0], -1, x_test.shape[-1])
y_train = y_train.reshape(y_train.shape[0], -1, y_train.shape[-1])   # shape: 800,11236,1
y_test = y_test.reshape(y_test.shape[0], -1, y_test.shape[-1])
print("x_train.shape: ",x_train.shape)
print("y_train.shape: ",y_train.shape)




####################################
#compute pca bases
####################################
# k_max = max(config_model['layer']["GkNN_mode_in"],config_model['layer']["GkNN_mode_out"])
k_max = max(config_model['layer']["GkNN_mode_out_list"])
Np = (Np_ref + downsample_ratio - 1) // downsample_ratio
# pca_data_in = data_in_ds.reshape((data_in_ds.shape[0], -1))
pca_data_out = data_out_ds.reshape((data_out_ds.shape[0], -1))


# percentage = 0.1
# mask1 = torch.rand(pca_data_in.shape) > percentage
# mask2 = torch.rand(pca_data_out.shape) > percentage
# pca_data_in = (torch.from_numpy(pca_data_in)*mask1).numpy()
# pca_data_out = (torch.from_numpy(pca_data_out)*mask2).numpy()


print("Start SVD with data shape: ", pca_data_out.shape, flush = True)

# bases_pca_in, wbases_pca_in = compute_2dpca_bases(Np , k_max , L,  pca_data_in)
# bases_pca_in, wbases_pca_in = bases_pca_in.to(device), wbases_pca_in.to(device)

bases_pca_out, wbases_pca_out = compute_2dpca_bases(Np , k_max , L,  pca_data_out)
bases_pca_out, wbases_pca_out = bases_pca_out.to(device), wbases_pca_out.to(device)



# bases_list = [ bases_pca_out, wbases_pca_out, 0,0]
base_out = bases_pca_out
###################################
#construct model and train
###################################
model = MPhyHGkNN(base_out,**config_model).to(device)
print('params:',count_params(model))

print("Start training ", "layer_type: ",config_model,config_train, flush = True)
train_rel_l2_losses, test_rel_l2_losses = PhyHGkNN_train(
    x_train, y_train, x_test, y_test, config, model, save_model_name=False
)