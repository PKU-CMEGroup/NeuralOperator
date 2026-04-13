import sys
import os
import math 
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from timeit import default_timer

# 获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 向上两级找到项目根目录
project_root = os.path.dirname(os.path.dirname(current_dir))
# 添加到路径
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from specformer.specformer import Specformer, compute_Fourier_modes, Specformer_train
from generate_schrodinger1d_data import set_default_params




def preprocess_periodic_data(data, n_train, n_test):
    '''
    参数：
    data: (ndata, nT+1, N, 4) 的 numpy 数组，分别表示 nT+1 个时间步的波函数实部、虚部、势能和位置
    n_train: 训练样本数量
    n_test: 测试样本数量
    '''
    
    in_dim, out_dim = 4, 2
    ndata, nT, N, _ = data.shape
    nT = nT - 1
    
    X, Y = [], []
    for i in list(range(math.ceil(n_train / nT))) + list(range(-math.ceil(n_test / nT), 0)):
        for j in range(nT):
            X.append(data[i,j,  :,:in_dim])    #前一步的波函数实部、虚部、势能和位置
            Y.append(data[i,j+1,:,:out_dim])   #后一步的波函数实部和虚部
    X, Y = np.array(X), np.array(Y)
    X, Y = torch.from_numpy(X.astype(np.float32)), torch.from_numpy(Y.astype(np.float32))
    x_train, y_train  = X[:n_train,...], Y[:n_train,...] 
    x_test,  y_test   = X[-n_test:,...], Y[-n_test:,...] 

    # periodic boundary condition treatment
    node_mask = torch.ones((X.shape[0], N, 1), dtype=torch.float32) # (ndata, N, 1)
    nodes = X[..., 3:] # (ndata, N, 1)
    dx = nodes[0,1,0] - nodes[0,0,0]
    node_weights = (torch.ones((X.shape[0], N), dtype=torch.float32)) * dx / (2*np.pi) # scaled by a constant
    node_list = np.arange(N)
    directed_edges = torch.from_numpy(np.tile(np.column_stack([ np.concatenate([node_list, node_list])  , np.concatenate([np.roll(node_list, -1), np.roll(node_list, 1)])]) , (X.shape[0],1,1))) 
    edge_gradient_weights = torch.from_numpy(np.tile(np.concatenate([np.full(N, 1/(2.0*dx)), np.full(N, -1/(2.0*dx))])[...,np.newaxis] , (X.shape[0],1,1))) 
 
    

    aux_train  = (node_mask[0:n_train,...], nodes[0:n_train,...], node_weights[0:n_train,...], directed_edges[0:n_train,...], edge_gradient_weights[0:n_train,...])
    aux_test   = (node_mask[-n_test:,...],  nodes[-n_test:,...],  node_weights[-n_test:,...],  directed_edges[-n_test:,...],  edge_gradient_weights[-n_test:,...])
    print("data.shape = ", data.shape)
    print("x_train.shape = ", x_train.shape, "y_train.shape = ", y_train.shape)
    
    return x_train, aux_train, y_train, x_test, aux_test, y_test

def setup_model(in_dim, out_dim, device, checkpoint_path = None):
    nlayers = 6
    ndim=1
    L_b = 2*np.pi
    modes = compute_Fourier_modes(ndim, [32], [L_b])
    modes = torch.tensor(modes, dtype=torch.float).to(device)
    
    model = Specformer(
        ndims = 1,
        modes = modes,
        in_channels = in_dim,
        out_channels = out_dim,
        coord_dim = 1,
        d_model = 128,
        nhead = 8,
        num_layers = 6,
        dim_feedforward = 512,
        coord_mode = "fourier",   # "linear" or "fourier"
        d_coord = 64,
        num_frequencies = 16,
        dropout = 0.0,)
    
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    model = model.to(device)

    return model

if __name__ == "__main__":

    nT, T, k_max, N, L, V_type = set_default_params()
    in_dim, out_dim = 4, 2 

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = setup_model(in_dim, out_dim, device)

    n_train, n_test = 1000, 500 #10000, 500
    data = np.load("../../data/schrodinger_1d/schrodinger1d_"+V_type+"_data.npz")['u_refs']
    x_train, aux_train, y_train, x_test, aux_test, y_test = preprocess_periodic_data(data, n_train, n_test)

    epochs = 500
    base_lr = 5e-4 #0.001
    lr_ratio = 10
    scheduler = "OneCycleLR"
    weight_decay = 1.0e-4
    batch_size = 8

    print('batch_size', batch_size, '\n')

    normalization_x = True
    normalization_y = True
    normalization_dim_x = [0,1] # channel-wise normalization
    normalization_dim_y = []
    non_normalized_dim_x = 0
    non_normalized_dim_y = 0

    config = {"train" : {"base_lr": base_lr, 'lr_ratio': lr_ratio, "weight_decay": weight_decay, "epochs": epochs, "scheduler": scheduler,  "batch_size": batch_size, 
                        "normalization_x": normalization_x,"normalization_y": normalization_y, 
                        "normalization_dim_x": normalization_dim_x, "normalization_dim_y": normalization_dim_y, 
                        "non_normalized_dim_x": non_normalized_dim_x, "non_normalized_dim_y": non_normalized_dim_y}
                        }


    train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = Specformer_train(x_train, aux_train, y_train, x_test, aux_test, y_test, config, model, save_model_name="./Specformer_model_V"+V_type)
    

    
