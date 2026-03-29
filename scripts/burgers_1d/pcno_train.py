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
from pcno.pcno import PCNO, PCNO_train, compute_Fourier_modes 
from pcno.geo_utility import preprocess_data_mesh
from generate_burgers1d_data import set_default_params

np.set_printoptions(precision=10, suppress=True)

def preprocess_data(data, n_train, n_test, preprocess_data:bool = True, pcno_data_file = "../../data/burgers_1d/pcno_data.npz"):
    '''
    参数：
    data: (ndata, nT+1, N, 2) 的 numpy 数组，分别表示 nT+1 个时间步的u和位置
    n_train: 训练样本数量
    n_test: 测试样本数量

    添上最后一个点
    '''

    in_dim, out_dim = 2, 1
    _, nT, N, _ = data.shape
    nT = nT - 1
    x = data[0,0, :,-1]
    dx = x[1] - x[0]
    L = x[-1] + dx
    x = np.concatenate([x, [L]])[:,np.newaxis]

    if preprocess_data:
        features_list = []
        for i in list(range(math.ceil(n_train / nT))) + list(range(-math.ceil(n_test / nT), 0)):
            for j in range(nT):
                #前一步的波函数实部、虚部、势能和位置 和 后一步的波函数实部和虚部
                feature = np.concatenate([data[i,j,  :,:in_dim],data[i,j+1,:,:out_dim]], axis=-1)
                feature = np.concatenate([feature, feature[0:1,:]], axis=0)
                feature[-1,in_dim-1] = L # location should not be periodic
                features_list.append(feature)    
        
        n_data = len(features_list)
        assert(n_data >= n_train+n_test)
        vertices_list = [x] * n_data
        node_list = np.arange(N)
        elem = np.column_stack([np.ones(N, dtype=int),node_list,node_list+1])
        elems_list = [elem] * n_data


        nnodes, node_mask, nodes, node_measures, features, directed_edges, edge_gradient_weights = preprocess_data_mesh(vertices_list, elems_list, features_list, mesh_type='vertex_centered', adjacent_type='edge')

        np.savez_compressed(pcno_data_file, \
                        nnodes=nnodes, node_mask=node_mask, nodes=nodes, \
                        node_measures = node_measures, \
                        features=features, \
                        directed_edges=directed_edges, edge_gradient_weights=edge_gradient_weights) 
        
    else:
        
        data = np.load(pcno_data_file)
        node_mask, nodes, node_measures, features, directed_edges, edge_gradient_weights = data["node_mask"], data["nodes"], data["node_measures"], data["features"], data["directed_edges"], data["edge_gradient_weights"]

    # scaled by a constant
    node_weights = node_measures / L 

    # 转化为 torch tensor
    node_mask, nodes, node_weights, features, directed_edges, edge_gradient_weights = torch.from_numpy(node_mask), torch.from_numpy(nodes.astype(np.float32)), torch.from_numpy(node_weights.astype(np.float32)), torch.from_numpy(features.astype(np.float32)), torch.from_numpy(directed_edges), torch.from_numpy(edge_gradient_weights.astype(np.float32))

    x_train, y_train = features[0:n_train,...,0:in_dim], features[0:n_train,...,in_dim:in_dim+out_dim]
    x_test, y_test = features[-n_test:,...,0:in_dim], features[-n_test:,...,in_dim:in_dim+out_dim]

    aux_train  = (node_mask[0:n_train,...], nodes[0:n_train,...], node_weights[0:n_train,...], directed_edges[0:n_train,...], edge_gradient_weights[0:n_train,...])
    aux_test   = (node_mask[-n_test:,...],  nodes[-n_test:,...],  node_weights[-n_test:,...],  directed_edges[-n_test:,...],  edge_gradient_weights[-n_test:,...])
    print("x_train.shape = ", x_train.shape, "y_train.shape = ", y_train.shape)
    
    return x_train, aux_train, y_train, x_test, aux_test, y_test


# TODO
def setup_model(in_dim, out_dim, device, L = 2*np.pi, checkpoint_path = None):
    nlayers = 6
    ndim=1
    L_b = L * 2.1  #at least 2 times larger
    modes = compute_Fourier_modes(ndim, [32], [L_b])
    modes = torch.tensor(modes, dtype=torch.float).to(device)
    model = PCNO(ndim, modes, nmeasures=1,
               layers=[128]*nlayers,
               fc_dim=128,
               in_dim=in_dim, out_dim=out_dim,
               inv_L_scale_hyper = [False, 0.5, 2.0],
               act='gelu')
    
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    model = model.to(device)
    return model

if __name__ == "__main__":

    nT, T, k_max, N, L = set_default_params()
    in_dim, out_dim = 2, 1 

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = setup_model(in_dim, out_dim, device, L = L)
    

    n_train, n_test = 10000, 500
    data = np.load("../../data/burgers_1d/burgers1d_data.npz")['u_refs']
    x_train, aux_train, y_train, x_test, aux_test, y_test = preprocess_data(data, n_train, n_test, preprocess_data=False, pcno_data_file = "../../data/burgers_1d/pcno_burgers1d_data.npz")

    epochs = 500
    base_lr = 5e-4 #0.001
    lr_ratio = 10
    scheduler = "OneCycleLR"
    weight_decay = 1.0e-4
    batch_size = 8

    print('batch_size', batch_size, '\n')

    normalization_x = True
    normalization_y = True
    normalization_dim_x = [0,1] #channel-wise normalization
    normalization_dim_y = []
    non_normalized_dim_x = 0
    non_normalized_dim_y = 0

    config = {"train" : {"base_lr": base_lr, 'lr_ratio': lr_ratio, "weight_decay": weight_decay, "epochs": epochs, "scheduler": scheduler,  "batch_size": batch_size, 
                        "normalization_x": normalization_x,"normalization_y": normalization_y, 
                        "normalization_dim_x": normalization_dim_x, "normalization_dim_y": normalization_dim_y, 
                        "non_normalized_dim_x": non_normalized_dim_x, "non_normalized_dim_y": non_normalized_dim_y}
                        }


    train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = PCNO_train(x_train, aux_train, y_train, x_test, aux_test, y_test, config, model, save_model_name="./PCNO_model")
    

    



