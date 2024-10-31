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


def GPtrain(
    x_train, y_train,edge_grid_train, edge_Gauss_train, x_test, y_test, edge_grid_test, edge_Gauss_test, config, model, save_model_name="./FNO_model"
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
    same_edge = config['model']['same_edge']
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

    if not same_edge:
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_train, y_train, edge_grid_train, edge_Gauss_train,),
            batch_size=config["train"]["batch_size"],
            shuffle=True,
            pin_memory=True
        )
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_test, y_test, edge_grid_test, edge_Gauss_test,),
            batch_size=config["train"]["batch_size"],
            shuffle=False,
            pin_memory=True
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_train, y_train),
            batch_size=config["train"]["batch_size"],
            shuffle=True,
            pin_memory=True
        )
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_test, y_test),
            batch_size=config["train"]["batch_size"],
            shuffle=False,
            pin_memory=True
        )


    
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
    plot_bases_num = config.get('plot', {}).get('plot_bases_num', 0)
    plot_data_index = config.get('plot', {}).get('plot_data_index', False)
    t1 = default_timer()


    for ep in range(epochs):
        
        train_rel_l2 = 0
        model.train()
        for item in train_loader:
            if not same_edge:
                x, y ,edge_grid, edge_Gauss = item
            else:
                x, y = item
                edge_grid = edge_grid_train.repeat(config["train"]["batch_size"],1,1)
                edge_Gauss = edge_Gauss_train.repeat(config["train"]["batch_size"],1,1)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            edge_grid, edge_Gauss = edge_grid.to(device, non_blocking=True), edge_Gauss.to(device, non_blocking=True)
            batch_size_ = x.shape[0]
            optimizer.zero_grad()
            out = model(x, edge_grid, edge_Gauss)  # .reshape(batch_size_,  -1)

            if normalization_y:
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)
            loss = myloss(out.view(batch_size_, -1), y.view(batch_size_, -1))
            train_rel_l2 += loss.item()
            loss.backward()

            # special_weights = []
            # try:
            #     for i in range(4):
            #         special_weights.append(getattr(model.sp_layers_local[i], 'baseweight'))
            # except Exception:
            #     pass 
            # try:
            #     if model.train_local_weight:
            #         special_weights.append(model.baseweight_Gauss_in)
            # except Exception:
            #     pass
            # for sw in special_weights:
            #     if sw.grad is not None:
            #         # print(sw.grad[:3])
            #         sw.grad.data.mul_(0)
            #         # print(sw.grad[:3])
            optimizer.step()

        # test_l2 = 0
        test_rel_l2 = 0
        with torch.no_grad():
            if plot_hidden_layers_num:
                if plot_data_index:
                    x = x_test[plot_data_index,:,:].to(device)
                    y = y_test[plot_data_index,:,:].to(device)
                    edge_grid = edge_grid_test[plot_data_index,:,:].to(device)
                    edge_Gauss = edge_Gauss_test[plot_data_index,:,:].to(device)
                else:
                    interval = n_test//(8-1)
                    indices = [i * interval for i in range(8)]
                    x = x_test[indices].to(device)
                    y = y_test[indices].to(device)
                    edge_grid = edge_grid_test[indices].to(device)
                    edge_Gauss = edge_Gauss_test[indices].to(device)
                save_figure_hidden = config['plot']['save_figure_hidden']
                Nx,Ny = config['plot']['plot_shape'][0],config['plot']['plot_shape'][1]
                if ep%10 ==0:
                    if config['model']['phy_dim']==3:
                        model.plot_hidden_layer_3d(x,edge_grid, edge_Gauss,y,save_figure_hidden,ep,plot_hidden_layers_num)
                    else:
                        model.plot_hidden_layer(x,edge_grid, edge_Gauss,y,Nx,Ny,save_figure_hidden,ep,plot_hidden_layers_num)
            if plot_bases_num:
                if ep%10==0:
                    Nx,Ny = config['plot']['plot_shape'][0],config['plot']['plot_shape'][1]
                    save_figure_bases = config['plot']['save_figure_bases']
                    grid =  x[:,:,model.in_dim-model.phy_dim:]
                    model.plot_bases(plot_bases_num,grid,Nx,Ny,save_figure_bases,ep)
                
            for item in test_loader:
                if not same_edge:
                    x, y ,edge_grid, edge_Gauss = item
                else:
                    x, y = item
                    edge_grid = edge_grid_test.repeat(config["train"]["batch_size"],1,1)
                    edge_Gauss = edge_Gauss_test.repeat(config["train"]["batch_size"],1,1)
                x, y = x.to(device), y.to(device)
                edge_grid, edge_Gauss = edge_grid.to(device, non_blocking=True), edge_Gauss.to(device, non_blocking=True)

                batch_size_ = x.shape[0]
                out = model(x, edge_grid, edge_Gauss)
                
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
        
        if (ep % 1 == 0) or (ep == epochs - 1):
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