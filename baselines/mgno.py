import torch
import torch.nn as nn
import torch.nn.functional as F
import operator
from functools import reduce

from models import Adam, LpLoss, UnitGaussianNormalizer, _get_act


class MgIte(nn.Module):
    def __init__(self, num_channel_u, num_channel_f):
        super().__init__()

        self.A = nn.Conv2d(
            num_channel_u,
            num_channel_f,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.S = nn.Conv2d(
            num_channel_f,
            num_channel_u,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

    def forward(self, pair):
        if isinstance(pair, tuple):
            u, f = pair
            u = u + self.S(f - self.A(u))
        else:
            f = pair
            u = self.S(f)
        return (u, f)


class MgIte_init(nn.Module):
    def __init__(self, num_channel_u, num_channel_f):
        super().__init__()
        self.S = nn.Conv2d(
            num_channel_f,
            num_channel_u,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

    def forward(self, f):
        u = self.S(f)
        return (u, f)


class Restrict(nn.Module):
    def __init__(self, num_channel_u, num_channel_f):
        super().__init__()
        self.R_u = nn.Conv2d(
            num_channel_u, num_channel_u, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.R_f = nn.Conv2d(
            num_channel_f, num_channel_f, kernel_size=3, stride=2, padding=1, bias=False
        )

    def forward(self, pair):
        u, f = pair
        u, f = self.R_u(u), self.R_f(f)
        return (u, f)


class MgConv(nn.Module):
    def __init__(
        self,
        num_iteration,
        num_channel_u,
        num_channel_f,
    ):
        super().__init__()
        self.num_iteration = num_iteration
        self.num_channel_u = num_channel_u

        self.layers_up = nn.ModuleList()
        for _ in range(len(num_iteration) - 1):
            self.layers_up.append(
                nn.ConvTranspose2d(
                    num_channel_u,
                    num_channel_u,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )

        self.layers_down, components = nn.ModuleList(), nn.ModuleList()
        for l, num_iteration_l in enumerate(num_iteration):
            for i in range(num_iteration_l):
                if l == 0 and i == 0:
                    components.append(MgIte_init(num_channel_u, num_channel_f))
                else:
                    components.append(MgIte(num_channel_u, num_channel_f))
            self.layers_down.append(nn.Sequential(*components))

            if l < len(num_iteration) - 1:
                components = [Restrict(num_channel_u, num_channel_f)]

    def forward(self, f):
        pair_list = [(0, 0)] * len(self.num_iteration)
        pair = f

        # downblock
        for l in range(len(self.num_iteration)):
            pair = self.layers_down[l](pair)
            pair_list[l] = pair

        # upblock
        for j in range(len(self.num_iteration) - 2, -1, -1):
            u, f = pair_list[j][0], pair_list[j][1]
            u1 = self.layers_up[j](pair_list[j + 1][0])
            if u1.size(-1) > u.size(-1):
                u1 = u1[..., :-1]
            if u1.size(-2) > u.size(-2):
                u1 = u1[..., :-1, :]
            u = u + u1
            pair_list[j] = (u, f)

        return pair_list[0][0]


class MgNO(nn.Module):
    def __init__(
        self,
        num_layer,
        num_channel_u,
        num_channel_f,
        num_iteration,
        activation,
    ):
        super().__init__()
        self.num_layer = num_layer
        self.num_channel_u = num_channel_u
        self.num_channel_f = num_channel_f
        self.num_iteration = num_iteration

        self.conv_layers = nn.ModuleList([])
        self.ws = nn.ModuleList([])

        self.conv_layers.append(MgConv(num_iteration, num_channel_u, num_channel_f))
        self.ws.append(
            nn.Conv2d(num_channel_f, num_channel_u, kernel_size=1, bias=True)
        )
        for _ in range(num_layer - 1):
            self.conv_layers.append(MgConv(num_iteration, num_channel_u, num_channel_u))
            self.ws.append(
                nn.Conv2d(num_channel_u, num_channel_u, kernel_size=1, bias=True)
            )

        self.fc = nn.Conv2d(num_channel_u, 1, kernel_size=1, bias=False)
        self.act = _get_act(activation)

    def forward(self, u):

        for layer, w in zip(self.conv_layers, self.ws):
            u = self.act(layer(u) + w(u))

        u = self.fc(u)
        return u


def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size() + (2,) if p.is_complex() else p.size()))
    return c


def MgNO_train(x_train, y_train, x_test, y_test, config, model):
    print(f"number of parameters is {count_params(model)}")
    n_train, n_test = x_train.shape[0], x_test.shape[0]
    train_rel_l2_losses = []
    test_rel_l2_losses = []
    test_l2_losses = []
    normalization_x, normalization_y, normalization_dim = (
        config["train"]["normalization_x"],
        config["train"]["normalization_y"],
        config["train"]["normalization_dim"],
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    for ep in range(epochs):
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
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)

                batch_size_ = x.shape[0]
                out = model(x)

                if normalization_y:
                    out = y_normalizer.decode(out)
                    y = y_normalizer.decode(y)

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
                " rel.Train: ",
                train_rel_l2,
                " rel.Test: ",
                test_rel_l2,
                " abs.Test : ",
                test_l2,
            )

    return train_rel_l2_losses, test_rel_l2_losses, test_l2_losses
