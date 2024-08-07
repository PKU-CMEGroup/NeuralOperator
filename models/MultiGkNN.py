import torch
import torch.nn as nn
import torch.nn.functional as F
from .basics import compl_mul1d


class Restriction1d(nn.Module):
    def __init__(self, a_channels, u_channels, f_channels):
        super(Restriction1d, self).__init__()
        self.R_a = nn.Conv1d(
            a_channels, a_channels, kernel_size=3, padding=1, stride=2, bias=False
        )
        self.R_u = nn.Conv1d(
            u_channels, u_channels, kernel_size=3, padding=1, stride=2, bias=False
        )
        self.R_f = nn.Conv1d(
            f_channels, f_channels, kernel_size=3, padding=1, stride=2, bias=False
        )
        self.R_bases = nn.Conv1d(1, 1, kernel_size=3, padding=1, stride=2, bias=False)
        self.R_wbases = nn.Conv1d(1, 1, kernel_size=3, padding=1, stride=2, bias=False)

    def forward(self, a, u, f, bases, wbases):
        a = self.R_a(a)
        u = self.R_u(u)
        f = self.R_f(f)

        modes = bases.size(1)
        gridsize = bases.size(0)

        bases = bases.permute(1, 0)
        wbases = wbases.permute(1, 0)

        bases1 = torch.zeros(((gridsize + 1) // 2, modes))
        wbases1 = torch.zeros(((gridsize + 1) // 2, modes))
        for mode in range(modes):
            bases1[:, mode] = self.R_bases(bases[mode, :].unsqueeze(0)).squeeze(0)
            wbases1[:, mode] = self.R_wbases(wbases[mode, :].unsqueeze(0)).squeeze(0)

        return a, u, f, bases1, wbases1


class Prolongation1d(nn.Module):
    def __init__(self, u_channels):
        super(Prolongation1d, self).__init__()
        self.P_u = nn.ConvTranspose1d(
            u_channels, u_channels, kernel_size=3, stride=2, padding=1, bias=False
        )

    def forward(self, u):
        return self.P_u(u)


class GalerkinConv_test(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(GalerkinConv_test, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, self.modes, dtype=torch.float)
        )

    def forward(self, x, bases, wbases):
        x_hat = torch.einsum("bcx,xk->bck", x, wbases[:, : self.modes])
        x_hat = compl_mul1d(x_hat, self.weights)
        x = torch.real(torch.einsum("bck,xk->bcx", x_hat, bases[:, : self.modes]))

        return x


class GalerkinPositive(nn.Module):
    def __init__(self, a_channels, u_channels, f_channels, modes):
        super(GalerkinPositive, self).__init__()
        self.in_channels = a_channels + u_channels
        self.out_channels = f_channels
        self.modes_list = modes
        self.sp_layer = GalerkinConv_test(self.in_channels, self.in_channels, modes)
        self.w = nn.Conv1d(self.in_channels, self.in_channels, 1, bias=True)
        self.fc = nn.Conv1d(self.in_channels, self.out_channels, 1)

    def forward(self, a, u, bases, wbases):
        x = torch.cat((a, u), dim=-2)
        x1 = self.sp_layer(x, bases, wbases)
        x2 = self.w(x)
        res = x1 + x2
        x = x + F.gelu(res)
        u = self.fc(x)

        return u


class MultiGalerkinLayer(nn.Module):
    def __init__(
        self,
        bases,
        wbases,
        modes,
        a_channels,
        u_channels,
        f_channels,
        num_levels,
    ):
        super(MultiGalerkinLayer, self).__init__()

        self.modes = modes
        self.bases = bases
        self.wbases = wbases
        self.num_levels = num_levels
        self.positive = GalerkinPositive(a_channels, u_channels, f_channels, modes)
        self.restrictions = nn.ModuleList(
            [
                Restriction1d(a_channels, u_channels, f_channels)
                for _ in range(num_levels - 1)
            ]
        )

        self.smoothers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Conv1d(f_channels, u_channels, kernel_size=3, padding=1)
                        for _ in range(2)
                    ]
                )
                for _ in range(self.num_levels)
            ]
        )

        self.prolongations = nn.ModuleList(
            [Prolongation1d(u_channels) for _ in range(num_levels - 1)]
        )

    def forward(self, a, u, f):
        u_list = []
        bases = self.bases
        wbases = self.wbases

        for level, restriction in enumerate(self.restrictions):
            for _, smoother in enumerate(self.smoothers[level]):
                u = u + smoother(f - self.positive(a, u, bases, wbases))
            u_list.append(u)
            a, u, f, bases, wbases = restriction(a, u, f, bases, wbases)

        for _, smoother in enumerate(self.smoothers[self.num_levels - 1]):
            u = u + smoother(f - self.positive(a, u, bases, wbases))

        for rlevel, prolongation in enumerate(self.prolongations):
            level = self.num_levels - 2 - rlevel
            u = prolongation(u)
            u = u_list[level] + u
        return u


class MultiGalerkinNN(nn.Module):
    def __init__(
        self,
        bases,
        wbases,
        modes_list,
        dim_physic,
        a_channels,
        u_channels,
        f_channels,
        num_levels,
    ):
        super(MultiGalerkinNN, self).__init__()

        self.fc0_a = nn.Linear(dim_physic + 1, a_channels)
        self.fc0_f = nn.Linear(dim_physic + 1, f_channels)
        self.fc0_u = nn.Linear(a_channels + f_channels, u_channels)

        self.layers = nn.ModuleList(
            [
                MultiGalerkinLayer(
                    bases,
                    wbases,
                    modes,
                    a_channels,
                    u_channels,
                    f_channels,
                    num_levels,
                )
                for _, modes in enumerate(modes_list)
            ]
        )

        self.fc1_u = nn.Sequential(
            nn.Linear(u_channels, 2 * u_channels),
            nn.GELU(),
            nn.Linear(2 * u_channels, 1),
        )

    def forward(self, a):
        f = torch.ones_like(a)
        f = self.fc0_f(f)
        a = self.fc0_a(a)
        u = self.fc0_u(torch.cat((a, f), dim=-1))

        a = a.permute(0, 2, 1)
        u = u.permute(0, 2, 1)
        f = f.permute(0, 2, 1)
        for _, layer in enumerate(self.layers):
            u = layer(a, u, f)

        u = u.permute(0, 2, 1)
        u = self.fc1_u(u)

        return u
