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


def PhyHGkNN_train(
    x_train, y_train, x_test, y_test, config, model, save_model_name="./FNO_model"
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

    # Load from checkpoint
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
    plot_H_num = config.get('plot', {}).get('plot_H_num', 0)
    plot_hidden_layers = config.get('plot', {}).get('plot_hidden_layers', False)

    t1 = default_timer()


    for ep in range(epochs):
        train_rel_l2 = 0

        model.train()

        update_base_ep =config['train']['update_base_ep']
        if ep>update_base_ep:
            model.if_update_bases =False

        for x, y in train_loader:

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            batch_size_ = x.shape[0]
            optimizer.zero_grad()
            out = model(x)  # .reshape(batch_size_,  -1)

            if normalization_y:
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)
            loss = myloss(out.view(batch_size_, -1), y.view(batch_size_, -1))
            train_rel_l2 += loss.item()

            if ep>config['train']['regularization_ep']:
                loss_regularization = PhyHGkNN_regularization(model,config)
                loss += loss_regularization
            loss.backward()

            optimizer.step()

        # test_l2 = 0
        test_rel_l2 = 0
        with torch.no_grad():
            if plot_hidden_layers:
                x = x_test[:8,:,:].to(device)
                y = y_test[:8,:,:].to(device)
                save_figure_hidden = config['plot']['save_figure_hidden']
                Nx,Ny = config['plot']['plot_shape'][0],config['plot']['plot_shape'][1]
                if ep%1 ==0:
                    fig, axs = plt.subplots(1, 8, figsize=(24, 5))
                    length = len(model.ws)



                    x0 = x[0,:,0].cpu().reshape(Nx,Ny)
                    im = axs[0].imshow(x0, cmap='viridis')
                    axs[0].set_title('input')
                    fig.colorbar(im, ax=axs[0])

                    if model.if_update_bases:
                        model.update_bases(x)

                    length = len(model.ws)
                    x = model.fc0(x)
                    x = x.permute(0, 2, 1)
                    x = model.ln0(x.transpose(1,2)).transpose(1,2)

                    x0 = x[0,0,:].cpu().reshape(Nx,Ny)
                    im = axs[1].imshow(x0, cmap='viridis')
                    axs[1].set_title(f'x{0}')
                    fig.colorbar(im, ax=axs[1])

                    for i, (layer , w, dplayer,ln) in enumerate(zip(model.sp_layers, model.ws, model.dropout_layers,model.ln_layers)):
                        x1 = layer(x)
                        x2 = w(x)
                        x = x1 + x2
                        x = dplayer(x)
                        x = ln(x.transpose(1,2)).transpose(1,2)
                        if model.act is not None and i != length - 1:
                            x = model.act(x)
                        x0 = x[0,0,:].cpu().reshape(Nx,Ny)
                        im = axs[i+2].imshow(x0, cmap='viridis')
                        axs[i+2].set_title(f'x{i+1}')
                        fig.colorbar(im, ax=axs[i+2])
                    
                    x = x.permute(0, 2, 1)

                    fc_dim = model.fc_dim 
                    
                    if fc_dim > 0:
                        x = model.fc1(x)
                        if model.act is not None:
                            x = model.act(x)

                    x = model.fc2(x)
                    x0 = x[0,:,:].cpu().reshape(Nx,Ny)
                    im = axs[6].imshow(x0, cmap='viridis')
                    axs[6].set_title('output')
                    fig.colorbar(im, ax=axs[6])

                    y0 = y[0,:,:].cpu().reshape(Nx,Ny)
                    im = axs[7].imshow(y0, cmap='viridis')
                    axs[7].set_title('truth_y')
                    fig.colorbar(im, ax=axs[7])


                    plt.tight_layout()
                    plt.savefig(save_figure_hidden + 'ep'+str(ep).zfill(3)+'.png', format='png')
                    plt.close()  # 关闭图形窗口

                    fig, axs = plt.subplots(1, 8, figsize=(24, 5))     
                    for i in range(8):       
                        y0 = model.wbases2[0,:,i*12].cpu().reshape(Nx,Ny)
                        im = axs[i].imshow(y0, cmap='viridis')
                        axs[i].set_title(f'bases{i*12}')
                        fig.colorbar(im, ax=axs[i])
                    plt.tight_layout()
                    plt.savefig(save_figure_hidden + 'bases_ep'+str(ep).zfill(3)+'.png', format='png')
                    plt.close()  # 关闭图形窗口


            for x, y in test_loader:
                x, y = x.to(device), y.to(device)

                batch_size_ = x.shape[0]
                out = model(x)  # .reshape(batch_size_,  -1)
                

                if normalization_y:
                    out = y_normalizer.decode(out)
                    y = y_normalizer.decode(y)
                # if config['train']['device']=='cuda':
                #     out = out.cpu()
                #     y = y.cpu()

                test_rel_l2 += myloss(
                    out.view(batch_size_, -1), y.view(batch_size_, -1)
                ).item()
                # test_l2 += myloss.abs(
                #     out.view(batch_size_, -1), y.view(batch_size_, -1)
                # ).item()


        scheduler.step()

        train_rel_l2 /= n_train
        # test_l2 /= n_test
        test_rel_l2 /= n_test

        train_rel_l2_losses.append(train_rel_l2)
        test_rel_l2_losses.append(test_rel_l2)
        # test_l2_losses.append(test_l2)
        # current_lr = optimizer.param_groups[0]['lr']

        # non_zero_ratio_out = torch.count_nonzero(model.H_out).item() / model.H_out.numel()
        
        

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

def PhyHGkNN_regularization(model,config):
    loss_regularization = 0
    try:
        l1_lambda1 = config['train']['L1regularization_lambda_H1']
    except:
        l1_lambda1 = 0
    if l1_lambda1 > 0:
        l1_norm = model.H1.abs().sum()
        loss_regularization += l1_lambda1 * l1_norm
    try:
        l1_lambda2 = config['train']['L1regularization_lambda_H2']
    except:
        l1_lambda2 = 0
    if l1_lambda2 > 0:
        l1_norm = model.H2.abs().sum()
        loss_regularization += l1_lambda2 * l1_norm  
    return loss_regularization