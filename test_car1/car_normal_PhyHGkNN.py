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
with open('config_PhyHGkNN.yml', 'r', encoding='utf-8') as f:
    config = yaml.full_load(f)

config = config["car_normal"]
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



data_path = "..\data\car\car_data.npz"
data = np.load(data_path)

all_points = data["points"]
all_triangles = data["triangles"]
all_normals = data["normals"]
all_pressures = data["pressures"]

data_path2 = '..\data\car\car_grid_weight.npy'
grid_weight = np.load(data_path2)
grid_weight = grid_weight[:,:,np.newaxis]

x_train = torch.from_numpy(
    np.concatenate(
        (all_points[:n_train],
         grid_weight[:n_train]),
        axis=-1,
    ).astype(np.float32)
)
y_train = torch.from_numpy(all_normals[:n_train].astype(np.float32))
# x_test, y_test are [n_data, n_x, n_channel] arrays
x_test = torch.from_numpy(
    np.concatenate(
        (all_points[-n_test:],
         grid_weight[-n_test:]),
        axis=-1,
    ).astype(np.float32)
)
y_test = torch.from_numpy(
    all_normals[-n_test:].astype(
        np.float32
    )
)

print("x_train.shape: ",x_train.shape)  # shape: 500,3586,4 
print("y_train.shape: ",y_train.shape)  # shape: 100,3586,3


Fourier_para_path = config_model['Fourier_para']
Fourier_weight = torch.load(Fourier_para_path)
print(f'load Fourier paras from {Fourier_para_path}')
if config['model']['local_bases_type'] =='Gauss':
    Gauss_para_path = config_model['Gauss_para']
    Gauss_pts, Gauss_weight = torch.load(Gauss_para_path)
    para_list = [Fourier_weight,Gauss_pts,Gauss_weight]
    print(f'load Gauss paras from {Gauss_para_path}')

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