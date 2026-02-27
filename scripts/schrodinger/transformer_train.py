import math 
import argparse

from zmq import device
from baselines.transformer import Transformer_train
from scripts.schrodinger.generate_schrodinger1d_data import set_default_params
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from timeit import default_timer

from transformer import LpLoss, Transformer


def preprocess_data(data, n_train, n_test):
    '''
    参数：
    data: (ndata, nT+1, N, 4) 的 numpy 数组，分别表示 nT+1 个时间步的波函数实部、虚部、势能和位置
    n_train: 训练样本数量
    n_test: 测试样本数量
    '''
    in_dim, out_dim = 3, 2
    ndata, nT, N, _ = data.shape
    nT = nT - 1
    
    X, Y = [], []
    for i in list(range(math.ceil(n_train / nT))) + list(range(-math.ceil(n_test / nT), 0)):
        for j in range(nT):
            X.append(data[i,j,  :in_dim])   #前一步的波函数实部、虚部、势能和位置
            Y.append(data[i,j+1,:in_dim])   #后一步的波函数实部和虚部
    X, Y = np.array(X), np.array(Y)
    X, Y = torch.from_numpy(X.astype(np.float32)).to(device), torch.from_numpy(Y.astype(np.float32)).to(device)
    x_train, y_train  = X[:n_train,...], Y[:n_train,...] 
    x_test,  y_test   = X[-n_test:,...], Y[-n_test:,...] 

    
    node_mask = torch.ones((X.shape[0], N, 1), dtype=torch.float32).to(device) # (ndata, N, 1)
    nodes = X[..., 3:4] # (ndata, N, 1)
    aux_train = (node_mask[:n_train,...], nodes[:n_train,...])
    aux_test = (node_mask[-n_test:,...], nodes[-n_test:,...])

    return x_train, aux_train, y_train, x_test, aux_test, y_test

def setup_model(in_dim, out_dim, device, checkpoint_path = None):

    model = Transformer(
        in_channels = in_dim,
        out_channels = out_dim,
        coord_dim = 1,
        d_model = 128,
        nhead = 8,
        num_layers = 6,
        dim_feedforward = 512,
        dropout = 0.0,
        coord_mode = "fourier",   # "linear" or "fourier"
        d_coord = 64,
        num_frequencies = 16,)
    
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    model = model.to(device)

    return model

if __name__ == "__main__":

    nT, T, k_max, N, L, V_type = set_default_params()
    in_dim, out_dim = 3, 2 

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = setup_model(in_dim, out_dim, device)

    n_train, n_test = 10000, 500
    data = np.load("../../data/schrodinger/schrodinger1d_"+V_type+"_data.npz")['u_refs']
    x_train, aux_train, y_train, x_test, aux_test, y_test = preprocess_data(data, n_train, n_test)

    epochs = 500
    base_lr = 5e-4 #0.001
    lr_ratio = 10
    scheduler = "OneCycleLR"
    weight_decay = 1.0e-4
    batch_size = 8

    print('batch_size', batch_size, '\n')

    normalization_x = False
    normalization_y = True
    normalization_dim_x = []
    normalization_dim_y = []
    non_normalized_dim_x = 4
    non_normalized_dim_y = 0

    config = {"train" : {"base_lr": base_lr, 'lr_ratio': lr_ratio, "weight_decay": weight_decay, "epochs": epochs, "scheduler": scheduler,  "batch_size": batch_size, 
                        "normalization_x": normalization_x,"normalization_y": normalization_y, 
                        "normalization_dim_x": normalization_dim_x, "normalization_dim_y": normalization_dim_y, 
                        "non_normalized_dim_x": non_normalized_dim_x, "non_normalized_dim_y": non_normalized_dim_y}
                        }


    train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = Transformer_train(x_train, aux_train, y_train, x_test, aux_test, y_test, config, model, save_model_name="./Transformer_model_V"+V_type)
    

    
