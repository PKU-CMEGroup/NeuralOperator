import torch.nn as nn
import torch
from models import _get_act
import torch.nn.init as init

from gradient.geokno import compute_Fourier_bases, SpectralConv2d
from gradient.mlp import MLP


class GaussGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, ndims=2):
        super(GaussGraphConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mlp = MLP(in_channels, out_channels)

        self.iSigma = nn.Linear(ndims, ndims)
        init.eye_(self.iSigma.weight)
        self.C = nn.Parameter(torch.tensor([1], dtype=torch.float))

    def forward(self, f, nodes, edges_index):
        """
        normalization in each node?
        """

        batch_size, _, max_nnodes = f.shape
        f = self.mlp(f)
        f = f.permute(0, 2, 1)

        src, target = edges_index[..., 0], edges_index[..., 1]
        f_src = f[torch.arange(batch_size).unsqueeze(1), src]

        edges_feature = nodes[torch.arange(batch_size).unsqueeze(
            1), src] - nodes[torch.arange(batch_size).unsqueeze(1), target]
        edges_feature = self.C * \
            torch.exp(torch.einsum("bed,bed->be", edges_feature,
                      self.iSigma(edges_feature)))

        message = torch.einsum("bec,be->bec", f_src, edges_feature)

        out = torch.zeros(
            (batch_size, max_nnodes, self.out_channels), dtype=f.dtype, device=f.device)
        out.scatter_add_(dim=1, src=message, index=target.unsqueeze(
            2).repeat(1, 1, self.out_channels).to(torch.int64))

        return out.permute(0, 2, 1)


class GeoGaussNO(nn.Module):
    def __init__(
        self,
        ndims,
        modes,
        layers,
        fc_dim=128,
        in_dim=3,
        out_dim=1,
        act="gelu",
    ):
        super(GeoGaussNO, self).__init__()

        self.modes = modes

        self.layers = layers
        self.fc_dim = fc_dim

        self.ndims = ndims
        self.in_dim = in_dim

        self.fc0 = nn.Linear(
            in_dim, layers[0])

        self.sp_convs = nn.ModuleList(
            [
                SpectralConv2d(
                    in_size, out_size, modes)
                for in_size, out_size in zip(
                    self.layers, self.layers[1:]
                )
            ]
        )

        self.gsp_convs = nn.ModuleList(
            [
                GaussGraphConv(
                    in_size, out_size, ndims=self.ndims)
                for in_size, out_size in zip(
                    self.layers, self.layers[1:]
                )
            ]
        )

        self.ws = nn.ModuleList(
            [
                nn.Conv1d(
                    in_size, out_size, 1)
                for in_size, out_size in zip(self.layers, self.layers[1:])
            ]
        )

        if fc_dim > 0:
            self.fc1 = nn.Linear(
                layers[-1], fc_dim)
            self.fc2 = nn.Linear(
                fc_dim, out_dim)
        else:
            self.fc2 = nn.Linear(
                layers[-1], out_dim)

        self.act = _get_act(act)

    def forward(self, x, aux):
        """
        Args:
            - x : (batch nnodes, x_grid, y_grid, 2)
        Returns:
            - x : (batch nnodes, x_grid, y_grid, 1)
        """
        length = len(self.ws)

        # batch_size, nnodes, ndims
        node_mask, nodes, node_weights, directed_edges, _ = aux

        bases_c, bases_s, bases_0 = compute_Fourier_bases(
            nodes, self.modes, node_mask)
        wbases_c, wbases_s, wbases_0 = bases_c * \
            node_weights, bases_s * \
            node_weights, bases_0 * node_weights

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        for i, (speconv, gspeconv, w) in enumerate(zip(self.sp_convs, self.gsp_convs, self.ws)):
            x1 = speconv(
                x, bases_c, bases_s, bases_0, wbases_c, wbases_s, wbases_0)
            x2 = w(x)
            x3 = gspeconv(x, nodes, directed_edges)
            x = x1 + x2  # + x3
            if self.act is not None and i != length - 1:
                x = self.act(
                    x) + self.act(x3)

        x = x.permute(0, 2, 1)

        if self.fc_dim > 0:
            x = self.fc1(x)
            if self.act is not None:
                x = self.act(x)

        x = self.fc2(x)
        return x
