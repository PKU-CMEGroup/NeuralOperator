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

Np = data_in_ds.shape[1]
grid_weight_train = 1/(Np*Np)*np.ones((n_train,Np,Np))
grid_weight_test = 1/(Np*Np)*np.ones((n_test,Np,Np))
# x_train, y_train are [n_data, n_x, n_channel] arrays
x_train = torch.from_numpy(
    np.stack(
        (
            data_in_ds,
            np.tile(grid_x_ds, (n_train, 1, 1)),
            np.tile(grid_y_ds, (n_train, 1, 1)),
            grid_weight_train
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
            grid_weight_test
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
        grid = torch.cat((x_train[:,:,1:-1],x_test[:,:,1:-1]),dim=0)
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
            print(f'{i}/{times} finished',flush = True)
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
        grid = torch.cat((x_train[:,:,1:-1],x_test[:,:,1:-1]),dim=0)
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
            print(f'{i}/{times} finished',flush = True)
        edge_src = torch.cat(edge_src_list,dim=0)
        edge_dst = torch.cat(edge_dst_list,dim=0)
        edge_src_train, edge_dst_train = edge_src[:n_train], edge_dst[:n_train]
        edge_src_test, edge_dst_test = edge_src[-n_test:], edge_dst[-n_test:]
    else:
        edge_src_train, edge_dst_train = compute_self_edge(x_train[0,:,1:-1].unsqueeze(0),k)
        print(edge_src_train[0,0,:],edge_src_train[0,45,:],edge_src_train[0,90,:],edge_src_train[0,100,:])
        print(edge_dst_train[0,0,:],edge_dst_train[0,45,:],edge_dst_train[0,90,:],edge_dst_train[0,100,:])
        edge_src_test, edge_dst_test = edge_src_train, edge_dst_train

###################################
#construct model and train
###################################
model = GPGkNN(para_list, **config_model).to(device)
print('params:',count_params(model))

print('config_model: ')
pprint(config_model)

print('config_train: ')
pprint(config_train)

print("Start training ", flush = True)
train_rel_l2_losses, test_rel_l2_losses = GPtrain(
    x_train, y_train,edge_src_train, edge_dst_train, x_test, y_test, edge_src_test, edge_dst_test, config, model, save_model_name=False
)