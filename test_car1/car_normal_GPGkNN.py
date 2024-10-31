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

from models import  GPtrain, compute_2dFourier_bases, compute_2dpca_bases, compute_2dFourier_cbases, count_params

from models.GPGkNN import GPGkNN,compute_edge_fixpts, compute_self_edge

torch.set_printoptions(precision=16)

torch.manual_seed(0)
np.random.seed(0)

###################################
# load configs
###################################
with open('config.yml', 'r', encoding='utf-8') as f:
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
if 'GPDconv' in config['model']['layer_types_local']:
    Gauss_para_path = config_model['Gauss_para']
    Gauss_pts, Gauss_weight = torch.load(Gauss_para_path)
    para_list = [Fourier_weight,Gauss_pts,Gauss_weight]
    
    print(f'load Gauss paras from {Gauss_para_path}')

    k = config['model']['k']
    bsz = 100
    print('Start compute edges', flush = True)
    if not config['model']['same_edge']:
        edge_src_list, edge_dst_list = [],[]
        grid = torch.cat((x_train[:,:,0:-1],x_test[:,:,0:-1]),dim=0)
        print(f'grid.shape {grid.shape}')
        assert (n_train + n_test)%bsz == 0
        times = (n_train + n_test)//bsz
        for i in range(times):
            grid_batch = grid[i*bsz:(i+1)*bsz]
            edge_src_batch, edge_dst_batch = compute_edge_fixpts(grid_batch,Gauss_pts,k)
            edge_src_list.append(edge_src_batch)
            edge_dst_list.append(edge_dst_batch)
            del edge_src_batch
            del edge_dst_batch
            gc.collect()
            print(f'{i+1}/{times} finished',flush = True)
        edge_src = torch.cat(edge_src_list,dim=0)
        edge_dst = torch.cat(edge_dst_list,dim=0)
        edge_src_train, edge_dst_train = edge_src[:n_train], edge_dst[:n_train]
        edge_src_test, edge_dst_test = edge_src[-n_test:], edge_dst[-n_test:]
    else:
        edge_src_train, edge_dst_train = compute_edge_fixpts(x_train[0,:,1:-1].unsqueeze(0),Gauss_pts,k)
        edge_src_test, edge_dst_test = edge_src_train, edge_dst_train



elif 'GraphGaussconv' in config['model']['layer_types_local']:
    dim = config['model']['layer_hidden_dim'][0]
    init_weight = config['model']['init_weight_GraphGaussconv']*(torch.arange(dim)/dim+1/2)
    para_list = [Fourier_weight,None,None,None,init_weight]
    
    k = config['model']['k']
    bsz = 100
    print('Start compute edges', flush = True)
    if not config['model']['same_edge']:
        edge_src_list, edge_dst_list = [],[]
        grid = torch.cat((x_train[:,:,0:-1],x_test[:,:,0:-1]),dim=0)
        print(f'grid.shape {grid.shape}')
        assert (n_train + n_test)%bsz == 0
        times = (n_train + n_test)//bsz
        for i in range(times):
            grid_batch = grid[i*bsz:(i+1)*bsz]
            edge_src_batch, edge_dst_batch = compute_self_edge(grid_batch,k)
            edge_src_list.append(edge_src_batch)
            edge_dst_list.append(edge_dst_batch)
            del edge_src_batch
            del edge_dst_batch
            gc.collect()
            print(f'{i+1}/{times} finished',flush = True)
        edge_src = torch.cat(edge_src_list,dim=0)
        edge_dst = torch.cat(edge_dst_list,dim=0)
        edge_src_train, edge_dst_train = edge_src[:n_train], edge_dst[:n_train]
        edge_src_test, edge_dst_test = edge_src[-n_test:], edge_dst[-n_test:]
    else:
        edge_src_train, edge_dst_train = compute_self_edge(x_train[0,:,0:-1].unsqueeze(0),k)
        edge_src_test, edge_dst_test = edge_src_train, edge_dst_train

###################################
#construct model and train
###################################
model = GPGkNN(para_list, **config_model).to(device)
print('params:',count_params(model))
print('\n')
print('config_model: ')
pprint(config_model)
print('\n')
print('config_train: ')
pprint(config_train)
print('\n')
print("Start training ", flush = True)
train_rel_l2_losses, test_rel_l2_losses = GPtrain(
    x_train, y_train,edge_src_train, edge_dst_train, x_test, y_test, edge_src_test, edge_dst_test, config, model, save_model_name=False
)