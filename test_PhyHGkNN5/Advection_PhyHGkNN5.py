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


from models import  newPhyHGkNN_train, compute_2dFourier_bases, compute_2dpca_bases, compute_2dFourier_cbases, count_params

from models.PhyHGkNN5 import PhyHGkNN5

torch.set_printoptions(precision=16)


torch.manual_seed(0)
np.random.seed(0)


###################################
# load configs
###################################
with open('config.yml', 'r', encoding='utf-8') as f:
    config = yaml.full_load(f)

config = config["Advection"]
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

data_path = "../data/advection/advection1d_4_512"
data = loadmat(data_path)

data_in = data["u"].T
data_out = data["v"].T
data_grid = data['grid'].T

print("data_in.shape:" , data_in.shape)  #(50000, 512)
print("data_out.shape", data_out.shape)  #(50000, 512)
print("data_grid.shape", data_grid.shape)  #(50000, 512)

Np_ref = data_in.shape[1]
data_in_ds = data_in[:, 0::downsample_ratio]
data_grid_ds = data_grid[:, 0::downsample_ratio]
data_out_ds = data_out[:, 0::downsample_ratio]

num_samples = data_grid.shape[0]
data_grid_padded = np.hstack((np.zeros((num_samples, 1)), data_grid, np.ones((num_samples, 1))))
grid_weight = (data_grid_padded[:,2:]-data_grid_padded[:,:-2])/2
print("grid_weight.shape", grid_weight.shape)  #(50000, 512)

# x_train, y_train are [n_data, n_x, n_channel] arrays
x_train = torch.from_numpy(
    np.stack(
        (data_in_ds[:n_train,:],data_grid_ds[:n_train,:],grid_weight[:n_train,:]),axis=-1,
    ).astype(np.float32)
)
# x_train = torch.from_numpy(
#     np.stack(
#         (data_in_ds[:n_train,:],data_grid_ds[:n_train,:]),axis=-1,
#     ).astype(np.float32)
# )
y_train = torch.from_numpy(data_out_ds[:n_train, :,np.newaxis].astype(np.float32))
# x_test, y_test are [n_data, n_x, n_channel] arrays
x_test = torch.from_numpy(
    np.stack(
        (data_in_ds[-n_test:,:],data_grid_ds[-n_test:,:],grid_weight[-n_test:,:]),
        axis=-1,
    ).astype(np.float32)
)
# x_test = torch.from_numpy(
#     np.stack(
#         (data_in_ds[-n_test:,:],data_grid_ds[-n_test:,:]),
#         axis=-1,
#     ).astype(np.float32)
# )
y_test = torch.from_numpy(
    data_out_ds[-n_test:, :, np.newaxis].astype(
        np.float32
    )
)

x_train = x_train.reshape(x_train.shape[0], -1, x_train.shape[-1])   # shape:  1000, 512, 2
x_test = x_test.reshape(x_test.shape[0], -1, x_test.shape[-1])
y_train = y_train.reshape(y_train.shape[0], -1, y_train.shape[-1])   # shape: 1000, 512, 1
y_test = y_test.reshape(y_test.shape[0], -1, y_test.shape[-1])
print("x_train.shape: ",x_train.shape)
print("y_train.shape: ",y_train.shape)



Fourier_para_path = config_model['Fourier_para']
Fourier_weight = torch.load(Fourier_para_path)
print(f'load Fourier paras from {Fourier_para_path}')
if config['model']['local_bases_type'] =='Gauss':
    Gauss_para_path = config_model['Gauss_para']
    Gauss_pts, Gauss_weight = torch.load(Gauss_para_path)
    para_list = [Fourier_weight,Gauss_pts,Gauss_weight]
    print(f'load Gauss paras from {Gauss_para_path}')
elif config['model']['local_bases_type'] =='Morlet':
    Morlet_para_path = config_model['Morlet_para']
    Morlet_pts, Morlet_weight,Morlet_freq = torch.load(Morlet_para_path)
    para_list = [Fourier_weight,Morlet_pts, Morlet_weight,Morlet_freq]
    print(f'load Morlet paras from {Morlet_para_path}')


###################################
#construct model and train
###################################
model = PhyHGkNN5(para_list, **config_model).to(device)
print('params:',count_params(model))
print('\n')
print('config_model: ')
pprint(config_model)
print('\n')
print('config_train: ')
pprint(config_train)
print('\n')
print("Start training ", flush = True)
train_rel_l2_losses, test_rel_l2_losses = newPhyHGkNN_train(
    x_train, y_train, x_test, y_test, config, model, save_model_name=False
)