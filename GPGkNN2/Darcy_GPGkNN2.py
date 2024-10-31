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
from pprint import pprint


sys.path.append("../")
from tqdm import tqdm

from models import  GPtrain2, compute_2dFourier_bases, compute_2dpca_bases, compute_2dFourier_cbases, count_params

from models.GPGkNN2 import GPGkNN2

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

print("data_in.shape:" , data_in.shape, flush = True)
print("data_out.shape", data_out.shape, flush = True)

Np_ref = data_in.shape[1]
grid_1d = np.linspace(0, L, Np_ref)
grid_x, grid_y = np.meshgrid(grid_1d, grid_1d)

data_in_train = data_in[0:n_train, 0::downsample_ratio, 0::downsample_ratio]
data_out_train = data_out[0:n_train, 0::downsample_ratio, 0::downsample_ratio]
data_in_test = data_in[-n_test:, 0::downsample_ratio, 0::downsample_ratio]
data_out_test = data_out[-n_test:, 0::downsample_ratio, 0::downsample_ratio]

grid_x_ds = grid_x[0::downsample_ratio, 0::downsample_ratio]
grid_y_ds = grid_y[0::downsample_ratio, 0::downsample_ratio]

Np = data_in_train.shape[1]
grid_weight_train = 1/(Np*Np)*torch.ones((n_train,Np,Np,1))
grid_weight_test = 1/(Np*Np)*torch.ones((n_test,Np,Np,1))
grid_train = torch.from_numpy(
    np.stack(
        (
            np.tile(grid_x_ds, (n_train, 1, 1)),
            np.tile(grid_y_ds, (n_train, 1, 1)),
        ),
        axis=-1,
    ).astype(np.float32)
)
grid_test = torch.from_numpy(
    np.stack(
        (
            np.tile(grid_x_ds, (n_test, 1, 1)),
            np.tile(grid_y_ds, (n_test, 1, 1)),
        ),
        axis=-1,
    ).astype(np.float32)
)
# x_train, y_train are [n_data, n_x, n_channel] arrays
x_train = torch.cat(
    (torch.from_numpy(data_in_train[:, :, :, np.newaxis].astype(np.float32)),  grid_train,  grid_weight_train),
        dim=-1 )

y_train = torch.from_numpy(data_out_train[:, :, :, np.newaxis].astype(np.float32))
# x_test, y_test are [n_data, n_x, n_channel] arrays

x_test = torch.cat(
    (torch.from_numpy(data_in_test[:, :, :, np.newaxis].astype(np.float32)),  grid_test,  grid_weight_test),
        dim=-1 )

y_test = torch.from_numpy(
    data_out_test[:, :, :, np.newaxis].astype(np.float32)
)

x_train = x_train.reshape(x_train.shape[0], -1, x_train.shape[-1])   # shape: 800,11236,4  (11236 = 106*106 , 106-1 = (421-1) /4)
x_test = x_test.reshape(x_test.shape[0], -1, x_test.shape[-1])
y_train = y_train.reshape(y_train.shape[0], -1, y_train.shape[-1])   # shape: 800,11236,1
y_test = y_test.reshape(y_test.shape[0], -1, y_test.shape[-1])
grid_train = grid_train.reshape(grid_train.shape[0],-1,grid_train.shape[-1])   # 800,11236,2
grid_test = grid_test.reshape(grid_test.shape[0],-1,grid_test.shape[-1])
print("x_train.shape: ",x_train.shape)
print("y_train.shape: ",y_train.shape)


Fourier_freq_path = config_model['Fourier_freq_para']
# init_weight_path = config_model['init_weight_para']
# init_freq_path = config_model['init_freq_para']
Fourier_freq = torch.load(Fourier_freq_path)
# init_weight = torch.load(init_weight_path)
# init_freq = torch.load(init_freq_path)
print(f'load Fourier_freq paras from {Fourier_freq_path}', flush = True)
# print(f'load init_weight paras from {init_weight_path}')
# print(f'load init_freq paras from {init_freq_path}')
# para_list = [Fourier_freq, init_weight, init_freq]
para_list = [Fourier_freq, None,None]





###################################
#construct model and train
###################################
model = GPGkNN2(para_list, **config_model).to(device)
print('params:',count_params(model), flush = True)

print('config_model: ')
pprint(config_model)

print('config_train: ')
pprint(config_train)

print("Start training ", flush = True)
train_rel_l2_losses, test_rel_l2_losses = GPtrain2(
    x_train, y_train, grid_train,x_test,y_test, grid_test, config, model, save_model_name=False
)