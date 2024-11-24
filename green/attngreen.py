import torch
import torch.nn as nn
from collections import defaultdict
from myutils.basics import _get_act
import copy
import math



class SoftmaxAttention(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SoftmaxAttention, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.linears = nn.ModuleList(
            [
                nn.Linear(
                    in_channels,
                    hidden_channels,
                )
                for _ in range(2)
            ]
        )
        self.linears.append(
            nn.Linear(
                in_channels,
                out_channels,
            )
        )

    def forward(self, x):

        query, key, value = [W(x) for W, x in zip(self.linears, (x, x, x))]
        x = self._attention(query, key, value)

        return x

    def _attention(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = torch.softmax(scores / math.sqrt(self.hidden_channels), dim=-1)
        return torch.matmul(scores, value)


class SoftmaxAttention1(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, modes):
        super(SoftmaxAttention1, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.modes = modes

        self.linears = nn.ModuleList(
            [
                nn.Linear(
                    in_channels,
                    hidden_channels,
                )
                for _ in range(2)
            ]
        )

        self.scale = 1 / (self.in_channels * self.out_channels)
        self.weights = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, self.modes, dtype=torch.float)
        )

    def forward(self, x):

        query, key = [W(x) for W, x in zip(self.linears, (x, x))]
        value = self._mul(x, self.weights)
        x = self._attention(query, key, value)

        return x

    def _attention(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = torch.softmax(scores / math.sqrt(self.hidden_channels), dim=-1)
        return torch.matmul(scores, value)

    @torch.jit.script
    def _mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # (batch, modes, in_channel ), (in_channel, out_channel, modes) -> (batch, out_channel, modes)
        res = torch.einsum("bki,iok->bko", a, b)
        return res


class GalerkinConv(nn.Module):
    def __init__(self, in_channels, out_channels, modes, bases, wbases, green):
        super(GalerkinConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.modes = modes
        self.bases = bases[:, :modes]
        self.wbases = wbases[:, :modes]

        self.green = green

    def forward(self, x):
        bases, wbases = self.bases, self.wbases

        # Compute coeffcients
        x_hat = torch.einsum("bcx,xk->bkc", x, wbases)

        # Multiply relevant Fourier modes
        x_hat = self.green(x_hat)

        # Return to physical space
        x = torch.real(torch.einsum("bkc,xk->bcx", x_hat, bases))

        return x


class AttnGkNO(nn.Module):
    def __init__(self, bases, **config):
        super(AttnGkNO, self).__init__()

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
        hidden_channels = self.hidden_channels[idx]
        modes = self.modes_gk[idx]

        green_type = self.green_types[idx]

        if green_type == "AttnGreen":
            return SoftmaxAttention(in_channels, hidden_channels, out_channels)
        if green_type == "AttnGreen1":
            return SoftmaxAttention1(in_channels, hidden_channels, out_channels, modes)

        else:
            raise KeyError("Green Function Undefined.")

    def _choose_layer(self, idx):
        in_channels = self.layer_channels[idx]
        out_channels = self.layer_channels[idx + 1]

        layer_type = self.layer_types[idx]
        if layer_type == "GalerkinConv":

            modes = self.modes_gk[idx]
            bases = self.bases[self.bases_types[idx]][0]
            wbases = self.bases[self.wbases_types[idx]][1]
            green = self._choose_green(idx)

            return GalerkinConv(in_channels, out_channels, modes, bases, wbases, green)

        else:
            raise KeyError("Layer Type Undefined.")
