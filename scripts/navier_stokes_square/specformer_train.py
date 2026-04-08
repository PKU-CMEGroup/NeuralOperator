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
from generate_kolmogorovflow2d_data import set_default_params


def preprocess_data(n_train, n_test):
    '''
    参数：
    n_train: 训练样本数量
    n_test: 测试样本数量
    '''
    in_dim, out_dim = 3, 1
    ndata = 1000
    nT, T, k_max, N, L, nu, A, n = set_default_params()
    
    
    x = np.linspace(0.0, L, N, endpoint=False)
    y = np.linspace(0.0, L, N, endpoint=False)
    x_grid, y_grid = np.meshgrid(x, y, indexing="ij")
    dx, dy = x[1] - x[0], y[1] - y[0]
    
    X, Y = [], []
    for i in list(range(math.ceil(n_train / nT))) + [ndata + x for x in range(-math.ceil(n_test / nT), 0)]:
        data = np.load(f"../../data/navier_stokes_square/kolmogorovflow2d_data_{i:04d}.npy")
        for j in range(nT):
            X.append(np.stack([data[j,  :,:], x_grid, y_grid], axis=2))    #前一步的vorticity
            Y.append(data[j+1,:,:,np.newaxis])                             #后一步的vorticity
    X, Y = np.array(X), np.array(Y)
    X, Y = torch.from_numpy(X.astype(np.float32)), torch.from_numpy(Y.astype(np.float32))
    x_train, y_train  = X[:n_train,...].reshape(n_train, -1, in_dim),  Y[:n_train,...].reshape(n_train, -1, out_dim)
    x_test,  y_test   = X[-n_test:,...].reshape(n_test,  -1, in_dim),  Y[-n_test:,...].reshape(n_test,  -1, out_dim)

    
    # periodic boundary condition treatment
    node_mask = torch.ones((X.shape[0], N*N, 1), dtype=torch.float32)                             # (ndata, N, N, 1)
    nodes = X[..., 1:].reshape(-1, N*N, 2)                                                        # (ndata, N, N, 2)
    node_weights = (torch.ones((X.shape[0], N*N), dtype=torch.float32))*dx*dy/(2*np.pi * 2*np.pi) # scaled by a constant
    
    node_list = np.arange(N*N).reshape(N,N)
    horizontal_edges = np.stack((node_list, np.roll(node_list, shift=-1, axis=1)), axis=2)  # N by N-1 by 2
    vertical_edges   = np.stack((node_list, np.roll(node_list, shift=-1, axis=0)), axis=2)  # N by N-1 by 2
    directed_edges = np.vstack((horizontal_edges.reshape(-1,2), vertical_edges.reshape(-1,2), horizontal_edges[:,:,::-1].reshape(-1,2), vertical_edges[:,:,::-1].reshape(-1,2)))
    edge_gradient_weights = np.concatenate([
        np.full((N*(N-1), 2), [1/(2.0*dx),  0]),
        np.full((N*(N-1), 2), [0,  1/(2.0*dy)]), 
        np.full((N*(N-1), 2), [-1/(2.0*dx), 0]),
        np.full((N*(N-1), 2), [0, -1/(2.0*dy)])])
    
    directed_edges = torch.from_numpy(np.tile(directed_edges, (X.shape[0],1,1))) 
    edge_gradient_weights = torch.from_numpy(np.tile(edge_gradient_weights, (X.shape[0],1,1))) 
    
 
    aux_train  = (node_mask[0:n_train,...], nodes[0:n_train,...], node_weights[0:n_train,...], directed_edges[0:n_train,...], edge_gradient_weights[0:n_train,...])
    aux_test   = (node_mask[-n_test:,...],  nodes[-n_test:,...],  node_weights[-n_test:,...],  directed_edges[-n_test:,...],  edge_gradient_weights[-n_test:,...])
    print("data.shape = ", data.shape)
    print("x_train.shape = ", x_train.shape, "y_train.shape = ", y_train.shape)
    
    return x_train, aux_train, y_train, x_test, aux_test, y_test



def setup_model(in_dim, out_dim, device, checkpoint_path = None):
    nlayers = 6
    ndim=2
    L_b = 2*np.pi
    modes = compute_Fourier_modes(ndim, [12,12], [L_b,L_b])
    modes = torch.tensor(modes, dtype=torch.float).to(device)
    
    model = Specformer(
        ndims = 2,
        modes = modes,
        in_channels = in_dim,
        out_channels = out_dim,
        coord_dim = 2,
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

    nT, T, k_max, N, L, nu, A, n = set_default_params()
    in_dim, out_dim = 3, 1

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = setup_model(in_dim, out_dim, device)

    n_train, n_test = 10000, 500
    x_train, aux_train, y_train, x_test, aux_test, y_test = preprocess_data(n_train, n_test)

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


    train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = Specformer_train(x_train, aux_train, y_train, x_test, aux_test, y_test, config, model, save_model_name="./Specformer_model")
    

    
