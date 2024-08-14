import torch
import torch.nn as nn
import torch.nn.functional as F
from .basics import compl_mul1d
from torch_geometric.nn import NNConv


class GraphConvPositive(nn.Module):
    def __init__(self, a_channels, u_channels, f_channels, dim_physic):
        super(GraphConvPositive, self).__init__()
        self.in_channels = a_channels + u_channels
        self.in_channels1 = self.in_channels // 4
        self.out_channels = f_channels
        self.out_channels1 = self.out_channels // 2
        self.edge_channels = 2 * (2 + dim_physic)
        self.fc0_u = nn.Linear(u_channels, 1)
        self.fc0_a = nn.Linear(a_channels, 1)
        self.fc1 = nn.Linear(self.in_channels, self.in_channels1)
        self.fc2 = nn.Linear(self.out_channels1, self.out_channels)
        self.kernel = nn.Sequential(
            nn.Linear(self.edge_channels, 8),
            nn.GELU(),
            nn.Linear(8, self.in_channels1 * self.out_channels1),
        )
        self.conv = NNConv(
            self.in_channels1,
            self.out_channels1,
            self.kernel,
            aggr="mean",
            root_weight=True,
            bias=False,
        )

    def forward(self, a, u, grid, edge_index_one):
        batch_size = a.size(0)

        a = a.permute(0, 2, 1)
        u = u.permute(0, 2, 1)

        edge_attr = self._generate_edge_attr(a, u, grid, edge_index_one)

        x = torch.cat((a, u), dim=-1)
        x = self.fc1(x)

        edge_index = edge_index_one.repeat(batch_size, 1)
        edge_attr = edge_attr.view(-1, edge_attr.size(-1))
        x = x.view(-1, x.size(-1))

        x = self.conv(x, edge_attr, edge_index).view(batch_size, -1, self.out_channels1)

        x = x.permute(0, 2, 1)

        out = self.fc2(x)

        return out

    def _generate_edge_attr(self, a, u, grid, edge_index):
        u0 = self.fc0_u(u)
        a0 = self.fc0_a(a)
        points_feature = torch.cat((a0, u0, grid), dim=-1)
        start_points_index = edge_index[0, :]
        end_points_index = edge_index[1, :]
        attr_start = points_feature[:, start_points_index, :]
        attr_end = points_feature[:, end_points_index, :]
        attr = torch.cat((attr_start, attr_end), dim=-1)

        return attr


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
    def __init__(self, a_channels, u_channels, f_channels, modes):
        super(GalerkinSolver, self).__init__()
        self.in_channels = a_channels + u_channels + f_channels
        self.out_channels = u_channels
        self.modes_list = modes
        self.sp_layer = GalerkinConv_test(self.in_channels, self.in_channels, modes)
        self.w = nn.Conv1d(self.in_channels, self.in_channels, 1, bias=True)
        self.fc = nn.Conv1d(self.in_channels, self.out_channels, 1)

    def forward(self, a, u, f, bases, wbases):
        x = torch.cat((a, u, f), dim=-2)
        x1 = self.sp_layer(x, bases, wbases)
        x2 = self.w(x)
        res = x1 + x2
        x = x + F.gelu(res)
        u = u + self.fc(x)  #
        return u


class MultiGraphGalerkinNN(nn.Module):
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
        kernel_size_R,
        kernel_size_P,
        padding_R,
        padding_P,
    ):
        super(MultiGraphGalerkinNN, self).__init__()

        self.dim_physic = dim_physic
        self.bases = bases
        self.wbases = wbases
        self.num_levels = len(modes_list)

        self.fc0_a = nn.Linear(dim_physic + 1, a_channels_list[0])
        self.fc0_f = nn.Linear(dim_physic + 1, f_channels_list[0])
        self.fc0_u = nn.Linear(
            a_channels_list[0] + f_channels_list[0], u_channels_list[0]
        )

        self.positives = nn.ModuleList(
            [
                GraphConvPositive(
                    a_channels_list[l],
                    u_channels_list[l],
                    f_channels_list[l],
                    dim_physic,
                )
                for l in range(self.num_levels - 1)
            ]
        )
        self.solvers = nn.ModuleList(
            [
                GalerkinSolver(
                    a_channels_list[l],
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
                    a_channels_list[l],
                    a_channels_list[l + 1],
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
            "a_channels_list:",
            a_channels_list,
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

        grid = a[..., 1 : self.dim_physic]
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
        bases = self.bases
        wbases = self.wbases

        for positive, solver, restriction in zip(
            self.positives, self.solvers, self.restrictions
        ):
            gridsize_list.append(u.size(-1))
            # df = f - positive(a, u)
            df = f - positive(a, u, grid, edge_index)
            u = solver(a, u, df, bases, wbases)
            u_list.append(u)
            # f = f - positive(a, u)
            df = f - positive(a, u, grid, edge_index)
            a, u, f, bases, wbases = restriction(a, u, f, bases, wbases)

        df = f - positive(a, u)
        u = self.solvers[self.num_levels - 1](a, u, f, bases, wbases)

        for level in range(self.num_levels - 2, -1, -1):
            u = self.prolongations[level](u, gridsize_list[level])
            u = u_list[level] + u

        u = u.permute(0, 2, 1)
        u = self.fc1_u(u)

        return u

    def _gen_edges(self, grid, radius):
        num
