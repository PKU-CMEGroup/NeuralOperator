import torch
import torch.nn as nn
from collections import defaultdict
from myutils.basics import compl_mul1d, compl_mul1d_matrix, _get_act


@torch.jit.script
def mul_co(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    res = torch.einsum("bixy,xyk->bik", a, b)
    return res


def mul_back(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    res = torch.einsum("bok,xyk->boxy", a, b)
    return res


class MgIte(nn.Module):
    def __init__(self, u_channels, f_channels):
        super().__init__()

        self.A = nn.Conv2d(
            u_channels,
            f_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.S = nn.Conv2d(
            f_channels,
            u_channels,
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
    def __init__(self, u_channels, f_channels):
        super().__init__()
        self.S = nn.Conv2d(
            f_channels,
            u_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

    def forward(self, f):
        u = self.S(f)
        return (u, f)


class Restrict(nn.Module):
    def __init__(self, u_channels, f_channels):
        super().__init__()
        self.R_u = nn.Conv2d(
            u_channels, u_channels, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.R_f = nn.Conv2d(
            f_channels, f_channels, kernel_size=3, stride=2, padding=1, bias=False
        )

    def forward(self, pair):
        u, f = pair
        u, f = self.R_u(u), self.R_f(f)
        return (u, f)


class MgConv(nn.Module):
    def __init__(self, num_iteration, f_channels, u_channels):
        super().__init__()
        self.num_iteration = num_iteration
        self.u_channels = u_channels

        self.layers_up = nn.ModuleList()
        for _ in range(len(num_iteration) - 1):
            self.layers_up.append(
                nn.ConvTranspose2d(
                    u_channels,
                    u_channels,
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
                    components.append(MgIte_init(u_channels, f_channels))
                else:
                    components.append(MgIte(u_channels, f_channels))
            self.layers_down.append(nn.Sequential(*components))

            if l < len(num_iteration) - 1:
                components = [Restrict(u_channels, f_channels)]

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


class AdjGalerkinConv(nn.Module):
    def __init__(self, in_channels, out_channels, modes, bases, wbases, diag_width):
        super(AdjGalerkinConv, self).__init__()

        """
        1D Spectral layer. It avoids FFT, but utilizes low rank approximation.
        Options:
            1. Adjust bases by learning a linear transformation W.
            2. Set the width of the weights to enable interaction between coefficients of different modes.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.bases = bases[:, : self.modes]
        self.wbases = wbases[:, : self.modes]

        self.diag_width = diag_width
        self.scale = 1 / (in_channels * out_channels * (2 * self.diag_width - 1))
        if self.diag_width > 0:
            # The main diagonal
            self.weights_list = nn.ParameterList(
                [
                    nn.Parameter(
                        self.scale
                        * torch.rand(
                            in_channels, out_channels, self.modes, dtype=torch.float
                        )
                    )
                ]
            )
            for i in range(self.diag_width - 1):
                # The i-th and the -i-th diagonal
                for _ in range(2):
                    self.weights_list.append(
                        nn.Parameter(
                            self.scale
                            * torch.rand(
                                in_channels,
                                out_channels,
                                self.modes - i - 1,
                                dtype=torch.float,
                            )
                        )
                    )

        elif self.diag_width == -1:
            # Using the full rank matrix
            self.weights_matrix = nn.Parameter(
                self.scale
                * torch.rand(
                    in_channels, out_channels, self.modes, self.modes, dtype=torch.float
                )
            )
        else:
            raise ValueError("Diagonal Width Is Invalid!")

    def forward(self, x):
        bases, wbases = self.bases, self.wbases

        # Compute coeffcients
        x_co = mul_co(x, wbases)

        # Multiply relevant modes
        if self.diag_width > 0:
            x_hat = compl_mul1d(x_co, self.weights_list[0])
            for i in range(self.diag_width - 1):
                x1 = compl_mul1d(x_co[..., i + 1 :], self.weights_list[2 * i + 1])
                x2 = compl_mul1d(x_co[..., : -(i + 1)], self.weights_list[2 * i + 2])
                x_hat_clone = x_hat.clone()
                x_hat_clone[..., : -(i + 1)] += x1
                x_hat_clone[..., (i + 1) :] += x2
                x_hat = x_hat_clone
        elif self.diag_width == -1:
            x_hat = compl_mul1d_matrix(x_co, self.weights_matrix)

        # Return to physical space
        x = mul_back(x_hat, bases)

        return x


class DoubleFreqNO(nn.Module):
    def __init__(self, bases, **config):
        super(DoubleFreqNO, self).__init__()
        """
        The overall network. It contains several layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        Options:
            1. Type of spectral convolution layer.
            2. Whether to include the f==1 in the input.
            3. Nomalizations
            4. Dropouts
        """
        self.bases = bases

        self.config = defaultdict(lambda: None, **config)
        self.config = dict(self.config)
        all_attr = list(self.config.keys())
        for key in all_attr:
            setattr(self, key, self.config[key])

        # input channel is 2: (a(x), x) or 3: (F(x), a(x), x)
        self.fc0 = nn.Linear(self.in_dim, self.layer_channels[0])

        # self.sp_layers = nn.ModuleList(
        #     [
        #         self._choose_layer(index, in_channels, out_channels, layer_type)
        #         for index, (in_channels, out_channels, layer_type) in enumerate(
        #             zip(self.layer_channels, self.layer_channels[1:], self.layer_types)
        #         )
        #     ]
        # )
        self.conv_layers = nn.ModuleList(
            [
                MgConv(self.iters, in_channels, out_channels)
                for in_channels, out_channels in zip(
                    self.layer_channels, self.layer_channels[1:]
                )
            ]
        )
        self.ws = nn.ModuleList(
            [
                nn.Conv2d(in_channels, out_channels, 1)
                for in_channels, out_channels in zip(
                    self.layer_channels, self.layer_channels[1:]
                )
            ]
        )
        # self.dropouts = nn.ModuleList(
        #     [nn.Dropout(p=dropout) for dropout in self.dropouts]
        # )
        self.length = len(self.ws)

        # if fc_channels = 0, we do not have nonlinear layer
        if self.fc_channels > 0:
            self.fc1 = nn.Linear(self.layer_channels[-1], self.fc_channels)
            self.fc2 = nn.Linear(self.fc_channels, self.out_dim)
        else:
            self.fc2 = nn.Linear(self.layer_channels[-1], self.out_dim)

        self.act = _get_act(self.act)

    def forward(self, x):

        x = self.fc0(x)

        x = x.permute(0, 3, 1, 2)

        for i, (layer2, w) in enumerate(zip(self.conv_layers, self.ws)):

            # x1 = layer1(x)
            x2 = layer2(x)
            x3 = w(x)
            # res = self.p1 * x1 + self.p2 * x2 + x3
            res = x2 + x3

            # activation function
            if self.act is not None and i != self.length - 1:
                res = self.act(res)

            # skipping connection
            if self.should_residual[i] == True:
                x = x + res
            else:
                x = res

        x = x.permute(0, 2, 3, 1)

        if self.fc_channels > 0:
            x = self.fc1(x)
            if self.act is not None:
                x = self.act(x)
        x = self.fc2(x)

        return x

    def _choose_layer(self, index, in_channels, out_channels, layer_type):
        if layer_type == "GalerkinConv":
            modes = self.modes_gk[index]
            bases = self.bases[self.bases_types[index]][0]
            wbases = self.bases[self.wbases_types[index]][1]
            return AdjGalerkinConv(
                in_channels,
                out_channels,
                modes,
                bases,
                wbases,
                self.diag_widths[index],
            )

        else:
            raise KeyError("Layer Type Undefined.")


# class MgIte(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.S = nn.Conv2d(
#             channels, channels, kernel_size=3, stride=1, padding=1, bias=False
#         )

#     def forward(self, x):
#         x = x + self.S(x)
#         return x


# class MgConv(nn.Module):
#     def __init__(self, num_iteration, channels):
#         super().__init__()
#         self.num_iteration = num_iteration
#         self.channels = channels
#         self.levels = len(num_iteration)

#         self.layers_up = nn.ModuleList()
#         for _ in range(self.levels - 1):
#             self.layers_up.append(
#                 nn.ConvTranspose2d(
#                     channels, channels, kernel_size=4, stride=2, padding=1, bias=False
#                 )
#             )

#         self.layers_down, components = nn.ModuleList(), nn.ModuleList()
#         for l, num_iteration_l in enumerate(num_iteration):

#             for _ in range(num_iteration_l):
#                 components.append(MgIte(channels))
#             self.layers_down.append(nn.Sequential(*components))

#             if l < self.levels - 1:
#                 components = [
#                     nn.Conv2d(
#                         channels,
#                         channels,
#                         kernel_size=3,
#                         stride=2,
#                         padding=1,
#                         bias=False,
#                     )
#                 ]

#     def forward(self, x):
#         res_list = []

#         # downblock
#         for l in range(self.levels):
#             x = self.layers_down[l](x)
#             res_list.append(x)

#         # upblock
#         for j in range(self.levels - 2, -1, -1):
#             res = res_list[j]
#             x = self.layers_up[j](x)
#             if x.size(-1) > res.size(-1):
#                 x = x[..., :-1]
#             if x.size(-2) > res.size(-2):
#                 x = x[..., :-1, :]
#             x = x + res

#         return x
