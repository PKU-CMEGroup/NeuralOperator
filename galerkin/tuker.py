import torch
import torch.nn as nn
from collections import defaultdict
from myutils.basics import _get_act


@torch.jit.script
def mul_tucker_all(
    x: torch.Tensor,
    G: torch.Tensor,
    I: torch.Tensor,
    O: torch.Tensor,
    K: torch.Tensor,
    L: torch.Tensor,
) -> torch.Tensor:
    x = torch.einsum("bik,ir,ks->brs", x, I, K)
    x = torch.einsum("brs,rstq->btq", x, G)
    x = torch.einsum("btq,ot,lq->bol", x, O, L)
    return x


class TuckerGalerkinConv(nn.Module):
    def __init__(self, in_channels, out_channels, modes, bases, wbases, rank):
        super(TuckerGalerkinConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.modes = modes
        self.bases = bases
        self.wbases = wbases

        self.w_I = nn.Parameter(
            torch.rand(in_channels, rank[0], dtype=torch.float) / in_channels
        )
        self.w_O = nn.Parameter(
            torch.rand(out_channels, rank[1], dtype=torch.float) / out_channels
        )
        self.w_K = nn.Parameter(
            torch.rand(self.modes, rank[2], dtype=torch.float) / self.modes
        )
        self.w_L = nn.Parameter(
            torch.rand(self.modes, rank[3], dtype=torch.float) / self.modes
        )
        self.w_G = nn.Parameter(
            torch.rand(rank[0], rank[1], rank[2], rank[3], dtype=torch.float)
        )

    def forward(self, x):
        bases, wbases = self.bases, self.wbases

        # Compute coeffcients
        x_hat = torch.einsum("bcx,xk->bck", x, wbases)

        # Multiply relevant Fourier modes
        x_hat = mul_tucker_all(x_hat, self.w_G, self.w_I, self.w_O, self.w_K, self.w_L)

        # Return to physical space
        x = torch.einsum("bck,xk->bcx", x_hat, bases)

        return x


class TuckerGkNO(nn.Module):
    def __init__(self, bases, **config):
        super(TuckerGkNO, self).__init__()

        self.bases = bases

        self.config = defaultdict(lambda: None, **config)
        self.config = dict(self.config)
        all_attr = list(self.config.keys())
        for key in all_attr:
            setattr(self, key, self.config[key])

        # input channel is 2: (a(x), x) or 3: (F(x), a(x), x)
        self.fc0 = nn.Linear(self.in_dim, self.layer_channels[0])

        self.sp_layers = nn.ModuleList(
            [
                self._choose_layer(index, in_channels, out_channels, layer_type)
                for index, (in_channels, out_channels, layer_type) in enumerate(
                    zip(self.layer_channels, self.layer_channels[1:], self.layer_types)
                )
            ]
        )
        self.ws = nn.ModuleList(
            [
                nn.Conv1d(in_channels, out_channels, 1)
                for in_channels, out_channels in zip(
                    self.layer_channels, self.layer_channels[1:]
                )
            ]
        )
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

    def _choose_layer(self, index, in_channels, out_channels, layer_type):
        if layer_type == "GalerkinConv":
            modes = self.modes_gk[index]
            bases = self.bases[self.bases_types[index]][0]
            wbases = self.bases[self.wbases_types[index]][1]
            return TuckerGalerkinConv(
                in_channels,
                out_channels,
                modes,
                bases,
                wbases,
                self.rank,
            )

        else:
            raise KeyError("Layer Type Undefined.")
