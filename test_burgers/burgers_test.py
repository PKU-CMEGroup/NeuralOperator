import random
import torch
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
from timeit import default_timer
from scipy.io import loadmat

sys.path.append("../")

from baselines.geo_utility import preprocess_data
from baselines.geokno import compute_Fourier_modes, GeoKNO, GeoKNO_test

torch.set_printoptions(precision=16)

torch.manual_seed(0)
np.random.seed(10)

downsample_ratio = 1
n_train = 1000
n_test = 200
print('n_test: ',n_test)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


train_k_str = '1e+02' # '0e+00,1e+00,1e+01,1e+02,-1e+00,+1e+01,-1e+02'
train_equal_weight = True
test_k_str = '0e+00' # '0e+00,1e+00,1e+01,1e+02,-1e+00,+1e+01,-1e+02'
test_equal_weight = True
train_N = 512
test_N = 512
print('train_k: '+ train_k_str)
print('train_equal_weight: ', train_equal_weight)
print('train_N: ',train_N)
print('test_k: '+ test_k_str)
print('test_equal_weight: ', test_equal_weight)
print('test_N: ',test_N)


if train_equal_weight:
    train_data_path = f"../data/burgers/burgers_k{train_k_str}_N{train_N}_equal_weight_data.npz"
    model_path = f'model/k{train_k_str}_N{train_N}_equal_weight.pth'
else:
    train_data_path = f"../data/burgers/burgers_k{train_k_str}_N{train_N}_data.npz"    
    model_path = f'model/k{train_k_str}_N{train_N}.pth'

if test_equal_weight:
    test_data_path = f"../data/burgers/burgers_k{test_k_str}_N{test_N}_equal_weight_data.npz"
else:
    test_data_path = f"../data/burgers/burgers_k{test_k_str}_N{test_N}_data.npz"
train_data = np.load(train_data_path)
print('load train data from' + train_data_path)
test_data = np.load(test_data_path)
print('load test data from' + test_data_path)
nodes_train,  features_train = train_data["nodes"],  train_data["features"]
nnodes_test, node_mask_test, nodes_test, node_weights_test, features_test, directed_edges_test, edge_gradient_weights_test = test_data["nnodes"], test_data["node_mask"], test_data["nodes"], test_data["node_weights"], test_data["features"], test_data["directed_edges"], test_data["edge_gradient_weights"]


nodes_train = torch.from_numpy(nodes_train.astype(np.float32))
features_train = torch.from_numpy(features_train.astype(np.float32))

nnodes_test = torch.from_numpy(nnodes_test)
node_mask_test = torch.from_numpy(node_mask_test)
nodes_test = torch.from_numpy(nodes_test.astype(np.float32))
node_weights_test = torch.from_numpy(node_weights_test.astype(np.float32))
features_test = torch.from_numpy(features_test.astype(np.float32))
directed_edges_test = torch.from_numpy(directed_edges_test).to(torch.int64)
edge_gradient_weights_test = torch.from_numpy(edge_gradient_weights_test.astype(np.float32))
# print("all test features.shape: ", features_test.shape)


x_train, x_test = torch.cat((features_train[:n_train, :, [0]],nodes_train[:n_train, ...]),-1), torch.cat((features_test[-n_test:, :, [0]],nodes_test[-n_test:, ...]),-1)
aux_test        = (node_mask_test[-n_test:,...],  nodes_test[-n_test:,...],  node_weights_test[-n_test:,...],  directed_edges_test[-n_test:,...],  edge_gradient_weights_test[-n_test:,...])
y_train, y_test = features_train[:n_train, :, [1]],     features_test[-n_test:, :, [1]]
# print("x_test.shape: ", x_test.shape)
# print("y_test.shape: ", y_test.shape)


model = torch.load(model_path)
print(f'load model from {model_path}')

epochs       = 500
base_lr      = 0.001
scheduler    = "OneCycleLR"
weight_decay = 1.0e-4
batch_size   = 20

normalization_x = True
normalization_y = True
normalization_dim = []
x_aux_dim = 1
y_aux_dim = 0


config = {"train" : {"base_lr": base_lr, "weight_decay": weight_decay, "epochs": epochs, "scheduler": scheduler,  "batch_size": batch_size, 
                     "normalization_x": normalization_x,"normalization_y": normalization_y, "normalization_dim": normalization_dim, 
                     "x_aux_dim": x_aux_dim, "y_aux_dim": y_aux_dim}}

GeoKNO_test(x_train, y_train, x_test, aux_test, y_test, config, model)