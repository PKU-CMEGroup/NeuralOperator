import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.basics import compl_mul1d
from torch_geometric.nn import NNConv


class Restriction1d(nn.Module):
    def __init__(
        self,
        u_channels_in,
        u_channels_out,
        f_channels_in,
        f_channels_out,
        stride,
        padding,
        kernel_size,
    ):
        super(Restriction1d, self).__init__()

        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

        self.R_u = nn.Conv1d(
            u_channels_in,
            u_channels_out,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=False,
        )
        self.R_f = nn.Conv1d(
            f_channels_in,
            f_channels_out,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=False,
        )
        self.R_bases = nn.Conv1d(
            1, 1, kernel_size=kernel_size, padding=padding, stride=stride, bias=False
        )
        self.R_wbases = nn.Conv1d(
            1, 1, kernel_size=kernel_size, padding=padding, stride=stride, bias=False
        )

    def forward(self, u, f, bases, wbases):
        u = self.R_u(u)
        f = self.R_f(f)

        modes = bases.size(1)
        gridsize = bases.size(0)
        gridsize_rough = (
            gridsize + 2 * self.padding - self.kernel_size
        ) // self.stride + 1

        bases = bases.permute(1, 0)
        wbases = wbases.permute(1, 0)

        bases_rough = torch.zeros((gridsize_rough, modes)).to("cuda")
        wbases_rough = torch.zeros((gridsize_rough, modes)).to("cuda")
        for mode in range(modes):
            bases_rough[:, mode] = self.R_bases(bases[mode, :].unsqueeze(0)).squeeze(0)
            wbases_rough[:, mode] = self.R_wbases(wbases[mode, :].unsqueeze(0)).squeeze(
                0
            )

        return u, f, bases_rough, wbases_rough


class Prolongation1d(nn.Module):
    def __init__(self, u_channels_in, u_channels_out, stride, kernel_size, padding):
        super(Prolongation1d, self).__init__()
        self.P_u = nn.ConvTranspose1d(
            u_channels_in,
            u_channels_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )

    def forward(self, u, gridsize_fine):
        u = self.P_u(u)
        u = u[..., :gridsize_fine]
        return u


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


class SimpleGalerkinConv_test(nn.Module):
    def __init__(self, in_channels, out_channels, modes, bases, wbases):
        super(SimpleGalerkinConv_test, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, self.modes, dtype=torch.float)
        )
        self.bases = bases[:, : self.modes]
        self.wbases = wbases[:, : self.modes]

    def forward(self, x):
        bases = self.bases
        wbases = self.wbases
        x_hat = torch.einsum("bcx,xk->bck", x, wbases)
        x_hat = compl_mul1d(x_hat, self.weights)
        x = torch.real(torch.einsum("bck,xk->bcx", x_hat, bases))

        return x


class GalerkinSolver(nn.Module):
    def __init__(self, u_channels, f_channels, modes):
        super(GalerkinSolver, self).__init__()
        self.in_channels = u_channels + f_channels
        self.out_channels = u_channels
        self.modes_list = modes
        self.sp_layer = GalerkinConv_test(self.in_channels, self.in_channels, modes)
        self.w = nn.Conv1d(self.in_channels, self.in_channels, 1, bias=True)
        self.fc = nn.Conv1d(self.in_channels, self.out_channels, 1)

    def forward(self, u, f, bases, wbases):
        x = torch.cat((u, f), dim=-2)
        x1 = self.sp_layer(x, bases, wbases)
        x2 = self.w(x)
        res = x1 + x2
        x = x + F.gelu(res)
        u = u + self.fc(x)  #
        return u


class SimpleGalerkinSolver(nn.Module):
    def __init__(self, a_channels, u_channels, f_channels, modes, bases, wbases):
        super(SimpleGalerkinSolver, self).__init__()
        self.in_channels = a_channels + u_channels + f_channels
        self.out_channels = u_channels
        self.modes_list = modes
        self.sp_layer = SimpleGalerkinConv_test(
            self.in_channels, self.in_channels, modes, bases, wbases
        )
        self.w = nn.Conv1d(self.in_channels, self.in_channels, 1, bias=True)
        self.fc = nn.Conv1d(self.in_channels, self.out_channels, 1)

    def forward(self, a, u, f):
        x = torch.cat((a, u, f), dim=-2)
        x1 = self.sp_layer(x)
        x2 = self.w(x)
        res = x1 + x2
        x = x + F.gelu(res)
        u = u + self.fc(x)  #
        return u


class Conv1dPositive(nn.Module):
    def __init__(self, u_channels, f_channels):
        super(Conv1dPositive, self).__init__()
        self.conv = nn.Conv1d(
            u_channels, f_channels, kernel_size=3, padding=1, bias=False
        )

    def forward(self, u):
        out = self.conv(u)
        return out


# class SimpleNN(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super(SimpleNN, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(in_channels, hidden_channels),
#             nn.GELU(),
#             nn.Linear(hidden_channels, out_channels),
#         )

#     def forward(self, x):
#         return self.fc(x)


class MultiGalerkinNN(nn.Module):
    def __init__(
        self,
        bases,
        wbases,
        modes_list,
        dim_physic,
        u_channels_list,
        f_channels_list,
        stride,
        kernel_size_R,
        kernel_size_P,
        padding_R,
        padding_P,
    ):
        super(MultiGalerkinNN, self).__init__()

        self.dim_physic = dim_physic
        self.bases = bases
        self.wbases = wbases
        self.num_levels = len(modes_list)

        self.fc0_a = nn.Linear(dim_physic + 1, 16)
        self.fc0_f = nn.Linear(dim_physic + 1, f_channels_list[0])
        self.fc0_u = nn.Linear(16 + f_channels_list[0], u_channels_list[0])

        self.positives = nn.ModuleList(
            [
                Conv1dPositive(u_channels_list[l], f_channels_list[l])
                for l in range(self.num_levels - 1)
            ]
        )
        self.solvers = nn.ModuleList(
            [
                GalerkinSolver(
                    u_channels_list[l],
                    f_channels_list[l],
                    modes_list[l],
                )
                for l in range(self.num_levels)
            ]
        )
        self.restrictions = nn.ModuleList(
            [
                Restriction1d(
                    u_channels_list[l],
                    u_channels_list[l + 1],
                    f_channels_list[l],
                    f_channels_list[l + 1],
                    stride=stride,
                    kernel_size=kernel_size_R,
                    padding=padding_R,
                )
                for l in range(self.num_levels - 1)
            ]
        )

        self.prolongations = nn.ModuleList(
            [
                Prolongation1d(
                    u_channels_list[l + 1],
                    u_channels_list[l],
                    stride=stride,
                    kernel_size=kernel_size_P,
                    padding=padding_P,
                )
                for l in range(self.num_levels - 1)
            ]
        )

        self.fc1_u = nn.Sequential(
            nn.Linear(u_channels_list[0], 2 * u_channels_list[0]),
            nn.GELU(),
            nn.Linear(2 * u_channels_list[0], 1),
        )
        print(
            "modes_list:",
            modes_list,
            "u_channels_list:",
            u_channels_list,
            "f_channels:",
            f_channels_list,
            "stride:",
            stride,
            "kernel_size_R:",
            kernel_size_R,
            "kernel_size_P:",
            kernel_size_P,
            "padding_R:",
            padding_R,
            "padding_P:",
            padding_P,
        )

    def forward(self, a):

        f = torch.ones_like(a)
        f[..., 1:] = a[..., 1:]

        f = self.fc0_f(f)
        a = self.fc0_a(a)
        u = self.fc0_u(torch.cat((a, f), dim=-1))

        u = u.permute(0, 2, 1)
        f = f.permute(0, 2, 1)

        u_list = []
        gridsize_list = []
        bases = self.bases
        wbases = self.wbases

        for positive, solver, restriction in zip(
            self.positives, self.solvers, self.restrictions
        ):
            gridsize_list.append(u.size(-1))
            df = f - positive(u)
            u = solver(u, df, bases, wbases)
            u_list.append(u)
            f = f - positive(u)
            u, f, bases, wbases = restriction(u, f, bases, wbases)

        df = f - positive(u)
        u = self.solvers[self.num_levels - 1](u, f, bases, wbases)

        for level in range(self.num_levels - 2, -1, -1):
            u = self.prolongations[level](u, gridsize_list[level])
            u = u_list[level] + u

        u = u.permute(0, 2, 1)
        u = self.fc1_u(u)

        return u


class SimpleMultiGalerkinNN(nn.Module):
    def __init__(
        self,
        bases,
        wbases,
        modes_list,
        dim_physic,
        a_channels_list,
        u_channels_list,
        f_channels_list,
        stride,
    ):
        super(SimpleMultiGalerkinNN, self).__init__()

        self.dim_physic = dim_physic
        self.num_levels = len(modes_list)
        self.stride = stride

        self.fc0_a = nn.Linear(dim_physic + 1, a_channels_list[0])
        self.fc0_f = nn.Linear(dim_physic + 1, f_channels_list[0])
        self.fc0_u = nn.Linear(
            a_channels_list[0] + f_channels_list[0], u_channels_list[0]
        )

        self.positives = nn.ModuleList(
            [
                Conv1dPositive(
                    a_channels_list[l], u_channels_list[l], f_channels_list[l]
                )
                for l in range(self.num_levels - 1)
            ]
        )
        self.solvers = nn.ModuleList()
        for l in range(self.num_levels):
            self.solvers.append(
                SimpleGalerkinSolver(
                    a_channels_list[l],
                    u_channels_list[l],
                    f_channels_list[l],
                    modes_list[l],
                    bases,
                    wbases,
                )
            )
            bases, wbases = (
                bases[stride - 1 :: stride, :],
                wbases[stride - 1 :: stride, :],
            )

        self.fc1_u = nn.Sequential(
            nn.Linear(u_channels_list[0], 128),
            nn.Linear(128, 1),
        )
        print(
            "modes_list:",
            modes_list,
            "a_channels_list:",
            a_channels_list,
            "u_channels_list:",
            u_channels_list,
            "f_channels:",
            f_channels_list,
        )

    def forward(self, a):

        f = torch.ones_like(a)
        f[..., 1 : self.dim_physic] = a[..., 1 : self.dim_physic]

        f = self.fc0_f(f)
        a = self.fc0_a(a)
        u = self.fc0_u(torch.cat((a, f), dim=-1))

        a = a.permute(0, 2, 1)
        u = u.permute(0, 2, 1)
        f = f.permute(0, 2, 1)

        u_list = []
        gridsize_list = []

        for positive, solver in zip(self.positives, self.solvers):

            gridsize_list.append(u.size(-1))
            df = f - positive(a, u)
            u = solver(a, u, df)
            u_list.append(u)
            f = f - positive(a, u)
            a, u, f = (
                self._meanRestriction(a),
                self._meanRestriction(u),
                self._meanRestriction(f),
            )

        df = f - positive(a, u)
        u = self.solvers[self.num_levels - 1](a, u, f)

        for level in range(self.num_levels - 2, -1, -1):
            u = self._repeatProlongation(u, gridsize_list[level])
            u = u_list[level] + u

        u = u.permute(0, 2, 1)
        u = self.fc1_u(u)

        return u

    def _meanRestriction(self, x):
        stride = self.stride
        batchsize, channels, num_points = x.shape
        r = num_points % stride
        if r != 0:
            indicex_remove = [(i + 1) * (num_points // stride) - 1 for i in range(r)]
            mask = torch.ones(num_points, dtype=bool)
            mask[indicex_remove] = False
            x = x[..., mask]
        x_reshape = x.view(batchsize, channels, num_points // stride, stride)
        x_mean = x_reshape.mean(dim=-1)
        return x_mean

    def _repeatProlongation(self, x, target_num_points):
        stride = self.stride
        batchsize, channels, num_points = x.shape
        x_repeat = torch.zeros(batchsize, channels, target_num_points, device=x.device)
        x_repeat[..., : num_points * stride] = torch.repeat_interleave(
            x, stride, dim=-1
        )
        return x_repeat