import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import operator
from timeit import default_timer
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


def FNN_train(
    x_train,
    y_train,
    x_test,
    y_test,
    config,
    model,
    bundary_indices,
    save_model_name="./FNO_model",
):
    print(count_params(model))
    n_train, n_test = x_train.shape[0], x_test.shape[0]
    train_rel_l2_losses = []
    test_rel_l2_losses = []
    test_l2_losses = []
    normalization_x, normalization_y, normalization_dim = (
        config["train"]["normalization_x"],
        config["train"]["normalization_y"],
        config["train"]["normalization_dim"],
    )

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
    elif config["train"]["scheduler"] == "OneCycleLR":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config["train"]["base_lr"],
            div_factor=2,
            final_div_factor=100,
            pct_start=0.2,
            steps_per_epoch=1,
            epochs=config["train"]["epochs"],
        )

    else:
        print("Scheduler ", config["train"]["scheduler"], " has not implemented.")

    model.train()
    myloss = LpLoss(d=1, p=2, size_average=False)

    epochs = config["train"]["epochs"]

    start_time = default_timer()
    for ep in range(epochs):
        if ep % 10 == 1:
            start_time = default_timer()

        train_rel_l2 = 0

        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            batch_size_ = x.shape[0]
            optimizer.zero_grad()
            out = model(x)
            if normalization_y:
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)
            loss = myloss(out.view(batch_size_, -1), y.view(batch_size_, -1))
            loss.backward()

            optimizer.step()
            train_rel_l2 += loss.item()

        test_l2 = 0
        test_rel_l2 = 0
        test_l2_bundary = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)

                batch_size_ = x.shape[0]
                out = model(x)

                if normalization_y:
                    out = y_normalizer.decode(out)
                    y = y_normalizer.decode(y)

                test_l2_bundary += myloss.abs(
                    out.view(batch_size_, -1)[:, bundary_indices],
                    y.view(batch_size_, -1)[:, bundary_indices],
                ).item()
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
        test_l2_bundary /= n_test

        train_rel_l2_losses.append(train_rel_l2)
        test_rel_l2_losses.append(test_rel_l2)
        test_l2_losses.append(test_l2)

        if (ep % 10 == 0) or (ep == epochs - 1):
            end_time = default_timer()
            print(
                "Epoch:",
                ep,
                " Train rel:",
                train_rel_l2,
                " Test rel:",
                test_rel_l2,
                " Test abs:",
                test_l2,
                " Test abs bundary:",
                test_l2_bundary,
                " Time:",
                end_time - start_time,
            )
            if save_model_name:
                torch.save(model, save_model_name)

    return train_rel_l2_losses, test_rel_l2_losses, test_l2_losses
