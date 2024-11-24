import torch.nn as nn
from models import _get_act
from gradient.geokno import compute_Fourier_bases, compute_gradient, SpectralConv2d


class MLP(nn.Module):
    def __init__(self, in_size, out_size, hidden_size=64, act="gelu"):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, out_size)
        self.act = _get_act(act)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        x = x.permute(0, 2, 1)
        return x


class GeoKNO(nn.Module):
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
        super(GeoKNO, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batch_size, x=s, y=s, c=3)
        output: the solution 
        output shape: (batch_size, x=s, y=s, c=1)
        """
        self.modes = modes

        self.layers = layers
        self.fc_dim = fc_dim

        self.ndims = ndims
        self.in_dim = in_dim

        self.fc0 = nn.Linear(in_dim, layers[0])

        self.sp_convs = nn.ModuleList(
            [
                SpectralConv2d(in_size, out_size, modes)
                for in_size, out_size in zip(
                    self.layers, self.layers[1:]
                )
            ]
        )

        self.ws = nn.ModuleList(
            [
                nn.Conv1d(in_size, out_size, 1)
                for in_size, out_size in zip(self.layers, self.layers[1:])
            ]
        )

        # self.gws = nn.ModuleList(
        #     [
        #         nn.Conv1d(ndims*in_size, out_size, 1)
        #         for in_size, out_size in zip(self.layers, self.layers[1:])
        #     ]
        # )
        self.gws = nn.ModuleList(
            [
                MLP(ndims * in_size, out_size)
                for in_size, out_size in zip(self.layers, self.layers[1:])
            ]
        )
        if fc_dim > 0:
            self.fc1 = nn.Linear(layers[-1], fc_dim)
            self.fc2 = nn.Linear(fc_dim, out_dim)
        else:
            self.fc2 = nn.Linear(layers[-1], out_dim)

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
        node_mask, nodes, node_weights, directed_edges, edge_gradient_weights = aux

        bases_c, bases_s, bases_0 = compute_Fourier_bases(
            nodes, self.modes, node_mask)
        wbases_c, wbases_s, wbases_0 = bases_c * \
            node_weights, bases_s * node_weights, bases_0 * node_weights

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        for i, (speconv, w, gw) in enumerate(zip(self.sp_convs, self.ws, self.gws)):
            x1 = speconv(x, bases_c, bases_s, bases_0,
                         wbases_c, wbases_s, wbases_0)
            x2 = w(x)
            x3 = gw(compute_gradient(x, directed_edges, edge_gradient_weights))
            x = x1 + x2  # + x3
            if self.act is not None and i != length - 1:
                x = self.act(x) + self.act(x3)

        x = x.permute(0, 2, 1)

        if self.fc_dim > 0:
            x = self.fc1(x)
            if self.act is not None:
                x = self.act(x)

        x = self.fc2(x)
        return x
