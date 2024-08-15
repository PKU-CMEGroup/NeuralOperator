import torch
import torch.nn as nn
import torch.nn.functional as F
from .basics import compl_mul1d
from torch_geometric.nn import NNConv


class GraphConvPositive(nn.Module):
    def __init__(self, a_channels, u_channels, f_channels, edge_index_one, dim_physic):
        super(GraphConvPositive, self).__init__()

        self.a_channels = a_channels
        self.u_channels = u_channels
        self.in_channels = a_channels + u_channels
        self.out_channels = f_channels

        self.edge_channels = 2 * (self.in_channels + dim_physic)
        self.edge_index_one = edge_index_one

        self.kernel = nn.Sequential(
            nn.Linear(self.edge_channels, 8),
            nn.GELU(),
            nn.Linear(8, self.in_channels * self.out_channels),
        )
        self.conv = NNConv(
            self.in_channels,
            self.out_channels,
            self.kernel,
            aggr="mean",
            root_weight=True,
            bias=False,
        )

    def forward(self, a, u, grid):
        batch_size = a.size(0)

        a = a.permute(0, 2, 1)
        u = u.permute(0, 2, 1)

        x = torch.cat((a, u), dim=-1)

        edge_index = self.edge_index_one.repeat(1, 1, batch_size)
        edge_index = edge_index.view(2, -1)
        edge_attr = self._get_edge_attr(a, u, grid, edge_index)
        x = x.view(-1, x.size(-1))

        out = self.conv(x, edge_index, edge_attr).view(
            batch_size, -1, self.out_channels
        )

        out = out.permute(0, 2, 1)

        return out

    def _get_edge_attr(self, a, u, grid, edge_index):

        u0 = u.reshape(-1, self.a_channels)
        a0 = a.reshape(-1, self.u_channels)

        pointwise_feature = torch.cat((a0, u0, grid.reshape(-1, 2)), dim=-1)
        attr1 = torch.index_select(pointwise_feature, 0, edge_index[0, :])
        attr2 = torch.index_select(pointwise_feature, 0, edge_index[1, :])
        attr = torch.cat((attr1, attr2), dim=-1)

        return attr


class GraphConvProlongation(nn.Module):
    def __init__(self, in_channels, out_channels, edge_index_one, dim_physic):
        super(GraphConvProlongation, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.edge_channels = 2 * (self.in_channels + dim_physic)
        self.edge_index_one = edge_index_one

        self.fc0_u = nn.Linear(in_channels, 1)

        self.kernel = nn.Sequential(
            nn.Linear(self.edge_channels, 8),
            nn.GELU(),
            nn.Linear(8, self.in_channels * self.out_channels),
        )

        self.conv = NNConv(
            self.in_channels,
            self.out_channels,
            self.kernel,
            aggr="mean",
            root_weight=True,
            bias=False,
        )

    def forward(self, u, grid):
        batch_size = u.size(0)

        u = u.permute(0, 2, 1)

        edge_index = self.edge_index_one.repeat(1, 1, batch_size)
        edge_index = edge_index.view(2, -1)
        edge_attr = self._get_edge_attr(u, grid, edge_index)
        edge_attr = edge_attr.view(-1, edge_attr.size(-1))
        u = u.reshape(-1, u.size(-1))

        out = self.conv(u, edge_index, edge_attr).view(
            batch_size, -1, self.out_channels
        )

        out = out.permute(0, 2, 1)

        return out

    def _get_edge_attr(self, u, grid, edge_index):

        u0 = u.reshape(-1, self.in_channels)

        pointwise_feature = torch.cat((u0, grid.reshape(-1, 2)), dim=-1)
        attr1 = torch.index_select(pointwise_feature, 0, edge_index[0, :])
        attr2 = torch.index_select(pointwise_feature, 0, edge_index[1, :])
        attr = torch.cat((attr1, attr2), dim=-1)

        return attr


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
    def __init__(self, a_channels, u_channels, f_channels, modes, bases, wbases):
        super(GalerkinSolver, self).__init__()
        self.in_channels = a_channels + u_channels + f_channels
        self.out_channels = u_channels
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


class MultiGraphGalerkinNN(nn.Module):
    def __init__(
        self,
        bases,
        wbases,
        modes_list,
        n_list,
        edge_index_positive_list,
        edge_index_re_list,
        edge_index_pro_list,
        dim_physic,
        a_channels_list,
        u_channels_list,
        f_channels_list,
        stride,
    ):
        super(MultiGraphGalerkinNN, self).__init__()

        self.dim_physic = dim_physic
        self.bases = bases
        self.wbases = wbases
        self.num_levels = len(modes_list)
        self.n_list = n_list

        self.fc0_a = nn.Linear(dim_physic + 1, a_channels_list[0])
        self.fc0_f = nn.Linear(dim_physic + 1, f_channels_list[0])
        self.fc0_u = nn.Linear(
            a_channels_list[0] + f_channels_list[0], u_channels_list[0]
        )
        self.u_channels_list = u_channels_list

        self.positives = nn.ModuleList(
            [
                GraphConvPositive(
                    a_channels_list[l],
                    u_channels_list[l],
                    f_channels_list[l],
                    edge_index_positive_list[l],
                    dim_physic,
                )
                for l in range(self.num_levels - 1)
            ]
        )
        self.solvers = nn.ModuleList()
        for l in range(self.num_levels):
            bases = bases[: n_list[l], :]
            wbases = wbases[: n_list[l], :]
            self.solvers.append(
                GalerkinSolver(
                    a_channels_list[l],
                    u_channels_list[l],
                    f_channels_list[l],
                    modes_list[l],
                    bases,
                    wbases,
                )
            )

        # self.restrictions = nn.ModuleList(
        #     [
        #         GraphRestriction(
        #             a_channels_list[l],
        #             a_channels_list[l + 1],
        #             u_channels_list[l],
        #             u_channels_list[l + 1],
        #             f_channels_list[l],
        #             f_channels_list[l + 1],
        #             edge_index_re_list[l],
        #             stride=stride,
        #         )
        #         for l in range(self.num_levels - 1)
        #     ]
        # )

        self.prolongations = nn.ModuleList(
            [
                GraphConvProlongation(
                    u_channels_list[l + 1],
                    u_channels_list[l],
                    edge_index_pro_list[l],
                    dim_physic=2,
                )
                for l in range(self.num_levels - 1)
            ]
        )

        self.fc1_u = nn.Sequential(
            nn.Linear(u_channels_list[0], 2 * u_channels_list[0]),
            nn.GELU(),
            nn.Linear(2 * u_channels_list[0], 1),
        )

    def forward(self, a):
        batch_size = a.size(0)
        grid = a[..., 1:]
        f = torch.ones_like(a)
        f[..., 1:] = a[..., 1:]

        f = self.fc0_f(f)
        a = self.fc0_a(a)
        u = self.fc0_u(torch.cat((a, f), dim=-1))

        u = u.permute(0, 2, 1)
        f = f.permute(0, 2, 1)

        u_list = []

        for l in range(self.num_levels - 1):
            a = a[:, :, : self.n_list[l]]
            u = u[:, :, : self.n_list[l]]
            f = f[:, :, : self.n_list[l]]
            grid_l = grid[:, : self.n_list[l], :]
            df = f - self.positives[l](a, u, grid_l)
            u = self.solvers[l](a, u, df)
            u_list.append(u)
            df = f - self.positives[l](a, u, grid_l)

        a = a[:, :, : self.n_list[l + 1]]
        u = u[:, :, : self.n_list[l + 1]]
        f = f[:, :, : self.n_list[l + 1]]
        u = self.solvers[l + 1](a, u, f)

        for l in range(self.num_levels - 2, -1, -1):
            grid_l = grid[:, : self.n_list[l], :]
            u_l = torch.zeros(
                batch_size, self.u_channels_list[l], self.n_list[l], device=u.device
            )
            u = self.prolongations[l](u_l, grid_l)
            u = u_list[l] + u

        u = u.permute(0, 2, 1)
        u = self.fc1_u(u)

        return u
