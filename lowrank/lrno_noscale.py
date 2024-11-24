import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from models import _get_act


@torch.jit.script
def mul_diag_lr(U1: torch.Tensor, U2: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    x = torch.einsum("irkh,bik->brkh", U2, x)
    x = torch.einsum("orkh,brkh->bok", U1, x)
    return x


@torch.jit.script
def mul_diag_lr_M(
    U1: torch.Tensor, M: torch.Tensor, U2: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    x = torch.einsum("irkh,bik->brkh", U2, x)
    x = torch.einsum("rskh,brkh->bskh", M, x)
    x = torch.einsum("oskh,bskh->bok", U1, x)
    return x


@torch.jit.script
def mul_hier_lr(U1: torch.Tensor, U2: torch.Tensor, x: torch.Tensor) -> torch.Tensor:

    # (in_channel, blocks, 2, parts, rank), (batch, in_channel, blocks, 2, parts)
    # -> (batch, blocks, 2, parts, rank)
    x = torch.einsum("indpr,bindp->bndpr", U2, x)

    # (out_channel, blocks, 2, parts, rank), (batch, blocks, 2, parts, rank)
    # -> (batch, blocks, 2, parts, out_channels)
    x = torch.einsum("ondpr,bndpr->bondp", U1, x)

    return x


@torch.jit.script
def mul_hier_lr_M(
    U1: torch.Tensor, M: torch.Tensor, U2: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:

    # (in_channel, blocks, 2, parts, rank), (batch, in_channel, blocks, 2, parts)
    # -> (batch, blocks, 2, parts, rank)
    x = torch.einsum("indpr,bindp->bndpr", U2, x)

    # (blocks, 2, rank, rank, parts, parts), (batch, blocks, 2, parts, rank)
    # -> (batch, blocks, 2, parts, rank)
    x = torch.einsum("ndrspq,bndpr->bndqs", M, x)

    # (out_channel, blocks, 2, parts, rank), (batch, blocks, 2, parts, rank)
    # -> (batch, out_channels, blocks, 2, parts)
    x = torch.einsum("ondpr,bndpr->bondp", U1, x)

    return x


class DiagLrGreen(nn.Module):
    def __init__(self, modes, in_channels, out_channels, rank, head):
        super(DiagLrGreen, self).__init__()
        self.scale = 1 / math.sqrt(in_channels * out_channels)

        self.U1 = nn.Parameter(
            self.scale * torch.rand(in_channels, rank,
                                    modes, head, dtype=torch.float)
        )
        self.U2 = nn.Parameter(
            self.scale * torch.rand(out_channels, rank,
                                    modes, head, dtype=torch.float)
        )

    def forward(self, x):
        x = mul_diag_lr(self.U1, self.U2, x)
        return x


class DiagLrMGreen(nn.Module):
    def __init__(self, modes, in_channels, out_channels, rank, head):
        super(DiagLrMGreen, self).__init__()
        self.scale = 1 / math.sqrt((in_channels * out_channels))

        self.U1 = nn.Parameter(
            self.scale * torch.rand(in_channels, rank,
                                    modes, head, dtype=torch.float)
        )
        self.M = nn.Parameter(
            self.scale * torch.rand(rank, rank, modes, head, dtype=torch.float))
        self.U2 = nn.Parameter(
            self.scale * torch.rand(out_channels, rank,
                                    modes, head, dtype=torch.float)
        )

    def forward(self, x):
        x = mul_diag_lr_M(self.U1, self.M, self.U2, x)
        return x


class HierLrGreen(nn.Module):
    def __init__(self, modes, in_channels, out_channels, rank, level):
        super(HierLrGreen, self).__init__()

        self.modes = modes
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.level = level
        self.diag = DiagLrGreen(modes, in_channels, out_channels, rank, 1)

        self.U1 = nn.ParameterList()
        self.U2 = nn.ParameterList()
        self.scale = math.sqrt(1 / (in_channels * out_channels))
        for l in range(1, level):
            num_blocks = self.modes // 2**l
            size_part = 2 ** (l - 1)
            self.U1.append(
                nn.Parameter(
                    self.scale
                    * torch.rand(
                        in_channels, num_blocks, 2, size_part, rank, dtype=torch.float
                    )
                )
            )
            self.U2.append(
                nn.Parameter(
                    self.scale
                    * torch.rand(
                        out_channels, num_blocks, 2, size_part, rank, dtype=torch.float
                    )
                )
            )

    def forward(self, x):
        batch_size = x.shape[0]

        x_add = torch.zeros(batch_size, self.out_channels,
                            self.modes, device=x.device)
        for l in range(1, self.level):
            num_blocks = self.modes // 2**l
            size_part = 2 ** (l - 1)
            x_blocks = x.reshape(batch_size, self.in_channels,
                                 num_blocks, 2, size_part)
            x_blocks = x_blocks[..., [1, 0], :]
            x_add += mul_hier_lr(self.U1[l - 1], self.U2[l - 1], x_blocks).reshape(
                batch_size, self.out_channels, self.modes
            )
        x_diag = self.diag(x)

        return x_diag + x_add


class HierLrMGreen(nn.Module):
    def __init__(self, modes, in_channels, out_channels, rank, level):
        super(HierLrMGreen, self).__init__()

        self.modes = modes
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.level = level
        self.diag = DiagLrGreen(modes, in_channels, out_channels, rank, 1)

        self.M = nn.ParameterList()
        self.U1 = nn.ParameterList()
        self.U2 = nn.ParameterList()
        self.scale = math.sqrt(1 / (in_channels * out_channels))
        for l in range(1, level):
            num_blocks = self.modes // 2**l
            size_part = 2 ** (l - 1)
            self.U1.append(
                self.scale
                * nn.Parameter(
                    torch.rand(
                        in_channels, num_blocks, 2, size_part, rank, dtype=torch.float
                    )
                )
            )
            self.M.append(
                nn.Parameter(
                    (
                        self.scale
                        * torch.rand(
                            num_blocks,
                            2,
                            rank,
                            rank,
                            size_part,
                            size_part,
                            dtype=torch.float,
                        )
                    )
                )
            )
            self.U2.append(
                self.scale
                * nn.Parameter(
                    torch.rand(
                        out_channels, num_blocks, 2, size_part, rank, dtype=torch.float
                    )
                )
            )

    def forward(self, x):
        batch_size = x.shape[0]

        x_add = torch.zeros(batch_size, self.out_channels,
                            self.modes, device=x.device)
        for l in range(1, self.level):
            num_blocks = self.modes // 2**l
            size_part = 2 ** (l - 1)
            x_blocks = x.reshape(batch_size, self.in_channels,
                                 num_blocks, 2, size_part)
            x_blocks = x_blocks[..., [1, 0], :]
            x_add += mul_hier_lr_M(
                self.U1[l - 1], self.M[l - 1], self.U2[l - 1], x_blocks
            ).reshape(batch_size, self.out_channels, self.modes)
        x_diag = self.diag(x)

        return x_diag + x_add


class LrGalerkinConv(nn.Module):
    def __init__(self, in_channels, out_channels, modes, bases, wbases, green):
        super(LrGalerkinConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.modes = modes
        self.bases = bases[:, :modes]
        self.wbases = wbases[:, :modes]

        self.green = green

    def forward(self, x):
        bases, wbases = self.bases, self.wbases

        # Compute coeffcients
        x_hat = torch.einsum("bcx,xk->bck", x, wbases)

        # Multiply relevant Fourier modes
        x_hat = self.green(x_hat)

        # Return to physical space
        x = torch.real(torch.einsum("bck,xk->bcx", x_hat, bases))

        return x


class LrGkNO(nn.Module):
    def __init__(self, bases, **config):
        super(LrGkNO, self).__init__()

        self.bases = bases

        self.config = defaultdict(lambda: None, **config)
        self.config = dict(self.config)
        all_attr = list(self.config.keys())
        for key in all_attr:
            setattr(self, key, self.config[key])

        self.fc0 = nn.Linear(self.in_dim, self.layer_channels[0])

        self.length = len(self.layer_channels) - 1
        self.sp_layers = nn.ModuleList(
            [self._choose_layer(idx) for idx in range(self.length)]
        )
        self.ws = nn.ModuleList(
            [
                nn.Conv1d(in_channels, out_channels, 1)
                for (in_channels, out_channels) in zip(
                    self.layer_channels, self.layer_channels[1:]
                )
            ]
        )

        if self.fc_channels > 0:
            self.fc1 = nn.Linear(self.layer_channels[-1], self.fc_channels)
            self.fc2 = nn.Linear(self.fc_channels, self.out_dim)
        else:
            self.fc2 = nn.Linear(self.layer_channels[-1], self.out_dim)

        self.act = _get_act(self.act)

    def forward(self, x):

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        for i, (layer, w) in enumerate(zip(self.sp_layers, self.ws)):
            x1 = layer(x)
            x2 = w(x)
            res = x1 + x2

            # activation function
            if self.act is not None and i != self.length - 1:
                res = self.act(res)

            # skipping connection
            if self.should_residual[i] == True:
                x = x + res
            else:
                x = res

        x = x.permute(0, 2, 1)

        if self.fc_channels > 0:
            x = self.fc1(x)
            if self.act is not None:
                x = self.act(x)
        x = self.fc2(x)

        return x

    def _choose_green(self, idx):
        in_channels = self.layer_channels[idx]
        out_channels = self.layer_channels[idx + 1]

        green_type = self.green_types[idx]

        if green_type == "HierLrGreen":
            modes = self.modes_gk[idx]
            level = self.levels[idx]
            rank = self.ranks_h[idx]
            return HierLrGreen(modes, in_channels, out_channels, rank, level)
        elif green_type == "HierLrMGreen":
            modes = self.modes_gk[idx]
            level = self.levels[idx]
            rank = self.ranks_h[idx]
            return HierLrMGreen(modes, in_channels, out_channels, rank, level)

        elif green_type == "DiagLrGreen":
            modes = self.modes_gk[idx]
            rank = self.ranks_d[idx]
            head = self.heads[idx]
            return DiagLrGreen(modes, in_channels, out_channels, rank, head)

        elif green_type == "DiagLrMGreen":
            modes = self.modes_gk[idx]
            rank = self.ranks_d[idx]
            head = self.heads[idx]
            return DiagLrMGreen(modes, in_channels, out_channels, rank, head)

        else:
            return KeyError("Green Function Undefined.")

    def _choose_layer(self, idx):
        in_channels = self.layer_channels[idx]
        out_channels = self.layer_channels[idx + 1]

        layer_type = self.layer_types[idx]
        if layer_type == "GalerkinConv":

            modes = self.modes_gk[idx]
            bases = self.bases[self.bases_types[idx]][0]
            wbases = self.bases[self.wbases_types[idx]][1]
            green = self._choose_green(idx)

            return LrGalerkinConv(
                in_channels, out_channels, modes, bases, wbases, green
            )

        else:
            raise KeyError("Layer Type Undefined.")
