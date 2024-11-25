import open3d as o3d
import os
import glob
import random
import torch
import sys
import numpy as np
import math
from timeit import default_timer
sys.path.append("../")
import yaml
from pprint import pprint
torch.set_printoptions(precision=16)

from models import  newPhyHGkNN_train, compute_2dFourier_bases, compute_2dpca_bases, compute_2dFourier_cbases, count_params,optPhytrain

from models.PhyHGkNN6 import PhyHGkNN6
torch.manual_seed(0)
np.random.seed(0)

with open('config.yml', 'r', encoding='utf-8') as f:
    config = yaml.full_load(f)

config = config["car_pressure"]
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





data = np.load("../data/car_pressure/geokno_triangle_data.npz")
nnodes, node_mask, nodes, node_weights, features, directed_edges, edge_gradient_weights = data["nnodes"], data["node_mask"], data["nodes"], data["node_weights"], data["features"], data["directed_edges"], data["edge_gradient_weights"]



x_train = torch.from_numpy(
    np.concatenate(
        (nodes[:n_train],
         node_weights[:n_train]),
        axis=-1,
    ).astype(np.float32)
)
y_train = torch.from_numpy(features[:n_train].astype(np.float32))
# x_test, y_test are [n_data, n_x, n_channel] arrays
x_test = torch.from_numpy(
    np.concatenate(
        (nodes[-n_test:],
         node_weights[-n_test:]),
        axis=-1,
    ).astype(np.float32)
)
y_test = torch.from_numpy(
    features[-n_test:].astype(
        np.float32
    )
)

print("x_train.shape: ",x_train.shape)  # shape: 500,3586,4 
print("y_train.shape: ",y_train.shape)  # shape: 100,3586,3


Fourier_para_path = config_model['Fourier_para']
Fourier_weight = torch.load(Fourier_para_path)
print(f'load Fourier paras from {Fourier_para_path}')

Gauss_para_path = config_model['Gauss_para']
# Gauss_pts, Gauss_weight = torch.load(Gauss_para_path)
# _, Gauss_weight = torch.load(Gauss_para_path)

random_indices1 = torch.randint(0, nodes.shape[1], (400,))
# random_indices2 = torch.randint(0, all_points.shape[1], (400,))
selected_points1 = torch.from_numpy(nodes[torch.arange(400), random_indices1])
# selected_points2 = torch.from_numpy(all_points[torch.arange(400), random_indices2])
# selected_points = torch.cat((selected_points1,selected_points2),dim=0).to(dtype=torch.float)
selected_points = selected_points1.to(dtype=torch.float)
selected_weight = 20*torch.ones(400,3,dtype=torch.float)
para_list = [Fourier_weight,selected_points,selected_weight]
print('select pts from data',400,20)
# print(f'load Gauss paras from {Gauss_para_path}')

###################################
#construct model and train
###################################
model = PhyHGkNN6(para_list, **config_model).to(device)
print('params:',count_params(model))
print('config_model: ')
pprint(config_model)
print('config_train: ')
pprint(config_train)
print("Start training ", flush = True)
if config["model"]['train_local_pts']==True and config["model"]['with_local'] == True:
    train_rel_l2_losses, test_rel_l2_losses = optPhytrain(
        x_train, y_train, x_test, y_test, config, model, save_model_name=False
    )
else:
    train_rel_l2_losses, test_rel_l2_losses = newPhyHGkNN_train(
        x_train, y_train, x_test, y_test, config, model, save_model_name=False
    )    