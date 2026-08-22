"""Point-cloud Fourier neural operator with stochastic point sampling.

PCFNO keeps the pointwise and Fourier-integral branches of PCNO and removes
the gradient branch entirely.  The accompanying dataset samples a fixed
number of valid points from each (possibly padded) mesh, so training does not
have to materialize the network activations for every point in a large mesh.
"""

from timeit import default_timer
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

def _get_act(act):
    if act == "tanh":
        func = F.tanh
    elif act == "gelu":
        func = F.gelu
    elif act == "relu":
        func = F.relu_
    elif act == "elu":
        func = F.elu_
    elif act == "leaky_relu":
        func = F.leaky_relu_
    elif act == "none":
        func = None
    else:
        raise ValueError(f"{act} is not supported")
    return func


def compute_Fourier_bases(
    nodes: torch.Tensor, modes: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute only the non-constant Fourier bases needed by PCFNO.

    The constant basis is identically one, so PCFNO handles its coefficient by
    summing the weighted features directly instead of allocating ``bases_0``.

    modes is 2pik/l
    """
    phase = torch.einsum("bxd,kdw->bxkw", nodes, modes)
    return torch.cos(phase), torch.sin(phase)


class SpectralConv(nn.Module):
    """Memory-efficient PCFNO spectral convolution.

    Quadrature weights are applied to the changing feature tensor ``x`` in
    each layer.  This avoids materializing ``wbases_c``, ``wbases_s``, and
    ``wbases_0`` over all points and Fourier modes.
    """
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        nmodes, ndims, nmeasures = modes.shape
        self.modes = modes
        self.nmeasures = nmeasures
        self.scale = 1 / (in_channels * out_channels)

        self.weights_c = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, nmodes, nmeasures, dtype=torch.float
            )
        )
        self.weights_s = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, nmodes, nmeasures, dtype=torch.float
            )
        )
        self.weights_0 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, 1, nmeasures, dtype=torch.float
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        bases_c: torch.Tensor,
        bases_s: torch.Tensor,
        node_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the Fourier integral using weighted input features."""

        # x=[bix]  node_weights=[bxw] ->  weighted_x=[bixw]
        weighted_x =  torch.einsum("bix,bxw->bixw", x, node_weights)   
        x_c_hat    =  torch.einsum("bixw,bxkw->bikw", weighted_x, bases_c)
        x_s_hat    = -torch.einsum("bixw,bxkw->bikw", weighted_x, bases_s)
        # The constant Fourier basis is one, so no bases_0 tensor is needed.
        x_0_hat = weighted_x.sum(dim=2, keepdim=True)

        
        weights_c, weights_s, weights_0 = self.weights_c, self.weights_s, self.weights_0
        
        f_c_hat = torch.einsum("bikw,iokw->bokw", x_c_hat, weights_c) - torch.einsum("bikw,iokw->bokw", x_s_hat, weights_s)
        f_s_hat = torch.einsum("bikw,iokw->bokw", x_s_hat, weights_c) + torch.einsum("bikw,iokw->bokw", x_c_hat, weights_s)
        f_0_hat = torch.einsum("bikw,iokw->bokw", x_0_hat, weights_0) 

        x = f_0_hat.sum(dim=(2, 3)).unsqueeze(-1) + 2 * torch.einsum("bokw,bxkw->box", f_c_hat, bases_c) - 2 * torch.einsum("bokw,bxkw->box", f_s_hat, bases_s)
        return x
                




class PCFNO(nn.Module):
    """PCNO with only pointwise and Fourier-integral operator branches.

    Unlike :class:`pcno.pcno.PCNO`, this model has no gradient parameters and
    does not need mesh edges or edge-gradient weights.  ``aux`` therefore only
    contains ``(node_mask, nodes, node_weights)``.  For convenience, a longer
    PCNO-style auxiliary tuple is also accepted; entries after the first three
    are ignored.

    Parameters are the same as ``PCNO``.  ``modes`` has shape
    ``[nmodes, ndims, nmeasures]`` and ``layers`` contains the channel width at
    every Fourier layer boundary.  Set
    ``layer_selection={"geointegral": True}`` to enable MPCNO's source- and
    target-normal Fourier integral corrections without enabling its explicit
    gradient or local geometry branches.
    """

    def __init__(
        self,
        ndims: int,
        modes: torch.Tensor,
        nmeasures: int,
        layers: Sequence[int],
        fc_dim: int = 128,
        in_dim: int = 3,
        out_dim: int = 1,
        act: str = "gelu",
        layer_selection: Optional[dict] = None,
    ) -> None:
        super().__init__()

        self.register_buffer("modes", modes)
        self.ndims = ndims
        self.nmeasures = nmeasures
        self.layers = list(layers)
        self.fc_dim = fc_dim
        self.in_dim = in_dim
        self.layer_selection = layer_selection
        
        self.fc0 = nn.Linear(in_dim, self.layers[0])
        self.sp_convs = nn.ModuleList(
            SpectralConv(in_size, out_size, modes)
            for in_size, out_size in zip(self.layers, self.layers[1:])
        )
        self.ws = nn.ModuleList(
            nn.Conv1d(in_size, out_size, 1, bias = False)
            for in_size, out_size in zip(self.layers, self.layers[1:])
        )

        if layer_selection['geointegral']:
            # Source-side normal augmentation: [x, x*n_1, ..., x*n_d].
            self.sp_convs_nws = nn.ModuleList(
                nn.Conv1d(in_size * (ndims + 1), in_size, 1, bias=False)
                for in_size, _ in zip(self.layers, self.layers[1:])
            )
            # MPCNO-style spectral projection and target-side normal term.
            self.sp_ws = nn.ModuleList(
                nn.Conv1d(out_size, out_size, 1, bias=False)
                for out_size in self.layers[1:]
            )
            self.sp_convs_adj_nws = nn.ModuleList(
                nn.Conv1d(out_size * ndims, out_size, 1, bias=False)
                for out_size in self.layers[1:]
            )
        else:
            self.sp_convs_nws = [None]*len(layers[1:])
            self.sp_ws = [None]*len(layers[1:])
            self.sp_convs_adj_nws = [None]*len(layers[1:])


        if fc_dim > 0:
            self.fc1 = nn.Linear(self.layers[-1], fc_dim)
            self.fc2 = nn.Linear(fc_dim, out_dim)
        else:
            self.fc2 = nn.Linear(self.layers[-1], out_dim)

        self.act = _get_act(act)



    def forward(
        self, x: torch.Tensor, aux: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        """Evaluate PCFNO on the points supplied in ``x`` and ``aux``.

        Args:
            x: Input features with shape ``[batch, npoints, in_dim]``.
            aux: ``(node_mask, nodes, node_weights, outward_normals)`` where
                the shapes are
                ``[batch, npoints, 1]``, ``[batch, npoints, ndims]``, and
                ``[batch, npoints, nmeasures]``, ``[batch, npoints, ndims]``
                respectively.  Normals are converted to channel-major layout
                internally when ``geointegral=True``.

        Returns:
            Output values with shape ``[batch, npoints, out_dim]``.  Padded
            points are exactly zero.
        """
        # node_mask=[bx1], nodes=[bxd], node_weights=[bxw], normals=[bxd]
        node_mask, nodes, node_weights, outward_normals = aux


        bases_c, bases_s = compute_Fourier_bases(nodes, self.modes)

        x = self.fc0(x)  
        x = x.permute(0, 2, 1)

        if self.layer_selection['geointegral']:
            outward_normals = outward_normals.permute(0, 2, 1)

        
        last_layer = len(self.ws) - 1
        for i, (speconv, spw, spconvnw, spconvadjnw, w) in enumerate(zip(self.sp_convs, self.sp_ws, self.sp_convs_nws, self.sp_convs_adj_nws, self.ws)):
            if self.layer_selection['geointegral']:
                x1 = speconv( spconvnw(  torch.cat([x] + [x * outward_normals[:, i:i+1, :] for i in range(outward_normals.size(1))], dim=1)  ), bases_c, bases_s, node_weights)
                x1 = spw(x1) + spconvadjnw(torch.cat([x1 * outward_normals[:, i:i+1, :] for i in range(outward_normals.size(1))], dim=1))
            else:
                x1 = speconv(x, bases_c, bases_s, node_weights)
                x1 = spw(x1)
                
            x2 = w(x)

            if self.act is not None and i != last_layer:
                x = x + self.act(x1 + x2)

        x = x.permute(0, 2, 1)
        if self.fc_dim > 0:
            x = self.fc1(x)
            if self.act is not None:
                x = self.act(x)

        x = self.fc2(x)
        return x * node_mask.to(dtype=x.dtype)
