import torch.nn as nn
from models import _get_act


class MgIte(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.S = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, bias=False
        )

    def forward(self, x):
        x = x + self.S(x)
        return x


class MgConv(nn.Module):
    def __init__(self, num_iteration, channels):
        super().__init__()
        self.num_iteration = num_iteration
        self.channels = channels

        self.layers_up = nn.ModuleList()
        for _ in range(len(num_iteration) - 1):
            self.layers_up.append(
                nn.ConvTranspose2d(
                    channels, channels, kernel_size=4, stride=2, padding=1, bias=False
                )
            )

        self.layers_down, components = nn.ModuleList(), nn.ModuleList()
        for l, num_iteration_l in enumerate(num_iteration):

            for _ in range(num_iteration_l):
                components.append(MgIte(channels))
            self.layers_down.append(nn.Sequential(*components))

            if l < len(num_iteration) - 1:
                components = [
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                    )
                ]

    def forward(self, x):
        res_list = []

        # downblock
        for l in range(len(self.num_iteration)):
            x = self.layers_down[l](x)
            res_list[l].append(x)

        # upblock
        for j in range(len(self.num_iteration) - 2, -1, -1):
            res = res_list[j]
            x = self.layers_up[j](x)
            if x.size(-1) > res.size(-1):
                x = x[..., :-1]
            if x.size(-2) > res.size(-2):
                x = x[..., :-1, :]
            x = x + res

        return x
