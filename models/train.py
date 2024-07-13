import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import operator
from functools import reduce
from .basics import SpectralConv1d
from .utils import _get_act, add_padding, remove_padding

from .adam import Adam
from .losses import LpLoss
from .normalizer import UnitGaussianNormalizer
from .fourier1d import FNN1d
from .fourier2d import FNN2d
from .fourier3d import FNN3d
from .fourier4d import FNN4d
from .Galerkin import GkNN


def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size() + (2,) if p.is_complex() else p.size()))
    return c


def construct_model(config, bases=None, wbases=None):
    dim = config["model"]["dim"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #######################################################################
    # FNO
    #######################################################################
    if config["model"]["model"] == "FNO":
        if dim == 1:
            modes1 = (
                config["model"]["modes1"]
                if "modes1" in config["model"].keys()
                else config["model"]["modes"]
            )

            model = FNN1d(
                modes=modes1,
                fc_dim=config["model"]["fc_dim"],
                layers=config["model"]["layers"],
                in_dim=config["model"]["in_dim"],
                out_dim=config["model"]["out_dim"],
                act=config["model"]["act"],
                pad_ratio=config["model"]["pad_ratio"],
            ).to(device)

        elif dim == 2:
            modes1 = (
                config["model"]["modes1"]
                if "modes1" in config["model"].keys()
                else config["model"]["modes"]
            )
            modes2 = (
                config["model"]["modes2"]
                if "modes2" in config["model"].keys()
                else config["model"]["modes"]
            )

            model = FNN2d(
                modes1=modes1,
                modes2=modes2,
                fc_dim=config["model"]["fc_dim"],
                layers=config["model"]["layers"],
                in_dim=config["model"]["in_dim"],
                out_dim=config["model"]["out_dim"],
                act=config["model"]["act"],
                pad_ratio=config["model"]["pad_ratio"],
            ).to(device)

        elif dim == 3:
            modes1 = (
                config["model"]["modes1"]
                if "modes1" in config["model"].keys()
                else config["model"]["modes"]
            )
            modes2 = (
                config["model"]["modes2"]
                if "modes2" in config["model"].keys()
                else config["model"]["modes"]
            )
            modes3 = (
                config["model"]["modes3"]
                if "modes3" in config["model"].keys()
                else config["model"]["modes"]
            )

            model = FNN3d(
                modes1=modes1,
                modes2=modes2,
                modes3=modes3,
                fc_dim=config["model"]["fc_dim"],
                layers=config["model"]["layers"],
                in_dim=config["model"]["in_dim"],
                out_dim=config["model"]["out_dim"],
                act=config["model"]["act"],
                pad_ratio=config["model"]["pad_ratio"],
            ).to(device)

        elif dim == 4:
            modes1 = (
                config["model"]["modes1"]
                if "modes1" in config["model"].keys()
                else config["model"]["modes"]
            )
            modes2 = (
                config["model"]["modes2"]
                if "modes2" in config["model"].keys()
                else config["model"]["modes"]
            )
            modes3 = (
                config["model"]["modes3"]
                if "modes3" in config["model"].keys()
                else config["model"]["modes"]
            )
            modes4 = (
                config["model"]["modes4"]
                if "modes4" in config["model"].keys()
                else config["model"]["modes"]
            )

            model = FNN4d(
                modes1=modes1,
                modes2=modes2,
                modes3=modes3,
                modes4=modes4,
                fc_dim=config["model"]["fc_dim"],
                layers=config["model"]["layers"],
                in_dim=config["model"]["in_dim"],
                out_dim=config["model"]["out_dim"],
                act=config["model"]["act"],
                pad_ratio=config["model"]["pad_ratio"],
            ).to(device)

        else:
            print("FNO with Dim = ", dim, ", which has not been implemented.")

    #######################################################################
    # GalerkinNO
    #######################################################################

    elif config["model"]["model"] == "GalerkinNO":
        model = GkNN(
            in_dim=config["model"]["in_dim"],
            out_dim=config["model"]["out_dim"],
            pad_ratio=config["model"]["pad_ratio"],
            layers=config["model"]["layers"],
            layer_configs=config["model"]["layer_configs"],
            fc_dim=config["model"]["fc_dim"],
            act=config["model"]["act"],
        ).to(device)
    else:
        print("Model type ", config["model"]["model"], " has not implemented")

    return model


# x_train, y_train, x_test, y_test are [n_data, n_x, n_channel] arrays
def FNN_train(
    x_train, y_train, x_test, y_test, config, model, save_model_name="./FNO_model"
):

    n_train, n_test = x_train.shape[0], x_test.shape[0]
    train_rel_l2_losses = []
    test_rel_l2_losses = []
    test_l2_losses = []
    normalization_x, normalization_y, normalization_dim = (
        config["train"]["normalization_x"],
        config["train"]["normalization_y"],
        config["train"]["normalization_dim"],
    )
    dim = len(x_train.shape) - 2  # n_train, size, n_channel

    # cost = FNN_cost(x_train.shape[1], config, dim)

    device = torch.device(config["train"]["device"])

    # print(model.wbases[1].device.type)

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
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, y_test),
        batch_size=config["train"]["batch_size"],
        shuffle=False,
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
    else:
        print("Scheduler ", config["train"]["scheduler"], " has not implemented.")

    model.train()
    myloss = LpLoss(d=1, p=2, size_average=False)

    epochs = config["train"]["epochs"]

    for ep in range(epochs):
        train_rel_l2 = 0

        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            batch_size_ = x.shape[0]
            optimizer.zero_grad()
            out = model(x)  # .reshape(batch_size_,  -1)
            if normalization_y:
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)
            loss = myloss(out.view(batch_size_, -1), y.view(batch_size_, -1))
            loss.backward()

            optimizer.step()
            train_rel_l2 += loss.item()

        test_l2 = 0
        test_rel_l2 = 0
        with torch.no_grad():
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
                test_l2 += myloss.abs(
                    out.view(batch_size_, -1), y.view(batch_size_, -1)
                ).item()

        scheduler.step()

        train_rel_l2 /= n_train
        test_l2 /= n_test
        test_rel_l2 /= n_test

        train_rel_l2_losses.append(train_rel_l2)
        test_rel_l2_losses.append(test_rel_l2)
        test_l2_losses.append(test_l2)

        if (ep % 10 == 0) or (ep == epochs - 1):
            print(
                "Epoch : ",
                ep,
                " Rel. Train L2 Loss : ",
                train_rel_l2,
                " Rel. Test L2 Loss : ",
                test_rel_l2,
                " Test L2 Loss : ",
                test_l2,
            )
            if save_model_name:
                torch.save(model, save_model_name)

    return train_rel_l2_losses, test_rel_l2_losses, test_l2_losses


# , cost
