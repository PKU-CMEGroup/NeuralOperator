import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import operator
from functools import reduce
from .basics import SpectralConv1d
from .utils import _get_act, add_padding, remove_padding
from timeit import default_timer
from torch.nn.utils import clip_grad_norm_
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
from torch_geometric.data import Data
from torch_cluster import radius_graph
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree

from .adam import Adam
from .losses import LpLoss
from .normalizer import UnitGaussianNormalizer
from .Galerkin import GkNN



def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul,
                    list(p.size()+(2,) if p.is_complex() else p.size()))
    return c


def GPtrain2(
    x_train, y_train, grid_train,x_test,y_test, grid_test, config, model, save_model_name="./FNO_model"
):

    n_train, n_test = x_train.shape[0], x_test.shape[0]
    train_rel_l2_losses = []
    test_rel_l2_losses = []

    normalization_x, normalization_y, normalization_dim = (
        config["train"]["normalization_x"],
        config["train"]["normalization_y"],
        config["train"]["normalization_dim"],
    )
    # dim = len(x_train.shape) - 2  # n_train, size, n_channel

    # cost = FNN_cost(x_train.shape[1], config, dim)

    device = torch.device(config["train"]["device"])

    if normalization_x:
        x_normalizer = UnitGaussianNormalizer(x_train, dim=normalization_dim)
        x_train = x_normalizer.encode(x_train)
        x_test = x_normalizer.encode(x_test)
        x_normalizer.to(device)

    if normalization_y:
        y_normalizer = UnitGaussianNormalizer(y_train, dim=normalization_dim)
        y_train = y_normalizer.encode(y_train)
        y_test = y_normalizer.encode(y_test)
        y_normalizer.to(device)
    

    print('Start compute edges', flush = True)
    r = config['model']['r'] + 1e-5
    max_num_neighbors = config['model']['max_num_neighbors']

    data_train_list = []
    data_test_list = []
    for i in tqdm(range(n_train)):
        edge_src, edge_dst = radius_graph(grid_train[i], r, max_num_neighbors = max_num_neighbors, loop=True) #shape: sum(k_n)   k_n is the number of nth neighborhood pts
        x = x_train[i]
        y = y_train[i]
        data = Data(x = x, y = y ,edge_index=torch.stack((edge_src, edge_dst),dim = 0))
        data_train_list.append(data)
        del edge_src, edge_dst, x, y, data
        torch.cuda.empty_cache()
    del grid_train, x_train, y_train
    torch.cuda.empty_cache()
    for i in tqdm(range(n_test)):
        edge_src, edge_dst = radius_graph(grid_test[i], r, max_num_neighbors = max_num_neighbors, loop=True) #shape: sum(k_n)   k_n is the number of nth neighborhood pts
        x = x_test[i]
        y = y_test[i]
        data = Data(x = x, y = y, edge_index=torch.stack((edge_src, edge_dst),dim = 0))
        data_test_list.append(data)
        del edge_src, edge_dst, x, y, data
        torch.cuda.empty_cache()
    del grid_test, x_test, y_test
    torch.cuda.empty_cache()    


    train_loader = DataLoader(data_train_list, batch_size=config["train"]["batch_size"], shuffle=True)
    test_loader = DataLoader(data_test_list, batch_size=config["train"]["batch_size"], shuffle=False)

    
    optimizer = Adam(
        model.parameters(),
        betas=(0.9, 0.999),
        lr=config["train"]["base_lr"],
        weight_decay=config["train"]["weight_decay"],
    )

    if config["train"]["scheduler"] == "MultiStepLR":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config["train"]["milestones"],
            gamma=config["train"]["scheduler_gamma"],
        )
    elif config["train"]["scheduler"] == "CosineAnnealingLR":
        T_max = (config["train"]["epochs"] // 10) * (
            n_train // config["train"]["batch_size"]
        )
        eta_min = 0.0
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min
        )
    elif config["train"]["scheduler"] == "OneCycleLR":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config['train']['base_lr'], 
            div_factor=2, final_div_factor=100,pct_start=0.2,
            steps_per_epoch=1, epochs=config['train']['epochs'])
        
    else:
        print("Scheduler ", config["train"]["scheduler"], " has not implemented.")

    model.train()
    myloss = LpLoss(d=1, p=2, size_average=False)

    epochs = config["train"]["epochs"]

    plot_hidden_layers_num = config.get('plot', {}).get('plot_hidden_layers_num', False)
    indices = [i * ((n_test-1)//(8-1)) for i in range(8)]
    plot_data_index = config.get('plot', {}).get('plot_data_index', indices)
    data_plot_list  = [data_test_list[i] for i in plot_data_index]
    plot_loader = DataLoader(data_plot_list, batch_size=len(data_plot_list), shuffle=False)

    t1 = default_timer()


    for ep in range(epochs):
        
        train_rel_l2 = 0
        model.train()
        for batch in train_loader:

            node_counts = degree(batch.batch, dtype=torch.long)
            x = torch.stack(torch.split(batch.x, node_counts.tolist(), dim=0), dim=0)
            y = torch.stack(torch.split(batch.y, node_counts.tolist(), dim=0), dim=0)
            edge_index = batch.edge_index
            x, y = x.to(device), y.to(device)
            edge_index = edge_index.to(device)
            batch_size_ = x.shape[0]
            optimizer.zero_grad()

            out = model(x, edge_index)  # .reshape(batch_size_,  -1)

            if normalization_y:
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)
            loss = myloss(out.view(batch_size_, -1), y.view(batch_size_, -1))
            train_rel_l2 += loss.item()
            loss.backward()

            optimizer.step()

        test_rel_l2 = 0

        with torch.no_grad():
            if plot_hidden_layers_num:
                save_figure_hidden = config['plot']['save_figure_hidden']
                Nx,Ny = config['plot']['plot_shape'][0],config['plot']['plot_shape'][1]
                if ep%10 ==0:
                    for batch in plot_loader:
                        node_counts = degree(batch.batch, dtype=torch.long)
                        x = torch.stack(torch.split(batch.x, node_counts.tolist(), dim=0), dim=0)
                        y = torch.stack(torch.split(batch.y, node_counts.tolist(), dim=0), dim=0)
                        edge_index = batch.edge_index
                        x, y = x.to(device), y.to(device)
                        edge_index = edge_index.to(device)
                        if config['model']['phy_dim']==3:
                            model.plot_hidden_layer_3d(x, edge_index,y,save_figure_hidden,ep,plot_hidden_layers_num)
                        else:
                            model.plot_hidden_layer(x, edge_index,y,Nx,Ny,save_figure_hidden,ep,plot_hidden_layers_num)

                
            for batch in test_loader:

                node_counts = degree(batch.batch, dtype=torch.long)
                x = torch.stack(torch.split(batch.x, node_counts.tolist(), dim=0), dim=0)
                y = torch.stack(torch.split(batch.y, node_counts.tolist(), dim=0), dim=0)
                edge_index = batch.edge_index
                x, y = x.to(device), y.to(device)
                edge_index = edge_index.to(device)

                batch_size_ = x.shape[0]
                out = model(x, edge_index)
                
                if normalization_y:
                    out = y_normalizer.decode(out)
                    y = y_normalizer.decode(y)

                test_rel_l2 += myloss(
                    out.view(batch_size_, -1), y.view(batch_size_, -1)
                ).item()



        scheduler.step()

        train_rel_l2 /= n_train
        # test_l2 /= n_test
        test_rel_l2 /= n_test

        train_rel_l2_losses.append(train_rel_l2)
        test_rel_l2_losses.append(test_rel_l2)
        # test_l2_losses.append(test_l2)
        # current_lr = optimizer.param_groups[0]['lr']
        

        t2 = default_timer()
        print(
            "Epoch : ",
            ep,
            " Time : ",
            round(t2-t1,3),
            " Rel. Train L2 Loss : ",
            train_rel_l2,
            " Rel. Test L2 Loss : ",
            test_rel_l2,
            # 'lr :',
            # current_lr,
            # 'H non_zero_percent: ',
            # round(non_zero_ratio_out,3),
            # 'H1 norm: ',
            # round(torch.norm(model.H1).item(),3),
            flush=True

        )
        t1 = default_timer()
        if save_model_name:
            torch.save(model, save_model_name)

    return train_rel_l2_losses, test_rel_l2_losses