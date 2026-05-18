import torch
import torch.nn as nn

from .mpcno import (
    _get_act,
    scaled_sigmoid,
    scaled_logit,
    compute_Fourier_modes_helper,
    compute_Fourier_modes,
    compute_Fourier_bases,
    SpectralConv,
    CombinedOptimizer,
    Combinedscheduler_OneCycleLR,
    MPCNO_train,
    MPCNO_train_multidist,
    MPCNO_train_parallel,
)


class StructuredNN(nn.Module):
    def __init__(self, in_dim, mid_layers, out_dim, act=None):
        super().__init__()
        self.act = act
        self.layers = nn.ModuleList()
        prev_dim = in_dim
        for hidden_dim in mid_layers:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        self.layers.append(nn.Linear(prev_dim, out_dim))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1 and self.act is not None:
                x = self.act(x)
        return x


class MPCNO(nn.Module):
    """
    MPCNO variant aligned with mpcno_structured, but with derivative/geo
    branches removed:
      - keep long-range spectral branch K and linear branch W
      - keep residual update style and branch scaling logic
      - keep StructuredNN projection head with configurable proj_act
      - remove grad/geo-related modules and paths
      - optionally keep geointegral branch (controlled by layer_selection['geointegral'])
    """

    def __init__(
        self,
        ndims,
        modes,
        nmeasures,
        layers,
        layer_selection=None,
        fc_dim=128,
        proj_layers=None,
        in_dim=3,
        out_dim=1,
        inv_L_scale_hyper=["independently", 0.5, 2.0],
        scaling_mode="inv",
        act="gelu",
        geo_act="softsign",
        proj_act="gelu",
    ):
        super().__init__()
        if layer_selection is None:
            layer_selection = {"grad": False, "geo": False, "geointegral": False}
        else:
            layer_selection = {
                "grad": bool(layer_selection.get("grad", False)),
                "geo": bool(layer_selection.get("geo", False)),
                "geointegral": bool(layer_selection.get("geointegral", False)),
            }

        self.register_buffer("modes", modes)
        self.nmeasures = nmeasures
        self.layers = layers
        self.fc_dim = fc_dim
        self.proj_layers = proj_layers
        self.layer_selection = layer_selection
        self.ndims = ndims
        self.in_dim = in_dim

        self.fc0 = nn.Linear(in_dim, layers[0])
        self.sp_convs = nn.ModuleList(
            [SpectralConv(in_size, out_size, modes) for in_size, out_size in zip(self.layers, self.layers[1:])]
        )
        self.sp_ws = nn.ModuleList(
            [nn.Conv1d(out_size, out_size, 1, bias=False) for _, out_size in zip(self.layers, self.layers[1:])]
        )
        self.sp_convs_nws = nn.ModuleList(
            [
                nn.Conv1d(in_size * (ndims + 1), in_size, 1, bias=False)
                for in_size, out_size in zip(self.layers[1:], self.layers[1:])
            ]
        ) if self.layer_selection["geointegral"] else [None]*len(layers[1:])

        self.sp_convs_adj_nws = nn.ModuleList(
            [
                nn.Conv1d(out_size * ndims, out_size, 1, bias = False)
                for in_size, out_size in zip(self.layers, self.layers[1:])
            ]
        ) if self.layer_selection["geointegral"] else [None]*len(layers[1:])
        
        self.ws = nn.ModuleList(
            [nn.Conv1d(in_size, out_size, 1) for in_size, out_size in zip(self.layers, self.layers[1:])]
        )

        self.train_inv_L_scale, self.inv_L_scale_min, self.inv_L_scale_max = (
            inv_L_scale_hyper[0],
            inv_L_scale_hyper[1],
            inv_L_scale_hyper[2],
        )
        self.inv_L_scale_latent = nn.Parameter(
            torch.full(
                (ndims, nmeasures),
                scaled_logit(torch.tensor(1.0), self.inv_L_scale_min, self.inv_L_scale_max),
            ),
            requires_grad=bool(self.train_inv_L_scale),
        )

        num_branches = 2  # K + W
        if scaling_mode == "inv":
            self.scale_factor = 1.0 / num_branches
        elif scaling_mode == "sqrt_inv":
            self.scale_factor = 1.0 / (num_branches ** 0.5)
        elif scaling_mode == "none":
            self.scale_factor = 1.0
        else:
            raise ValueError(f"{scaling_mode} is not supported")

        self.act = _get_act(act)
        self.proj_act = _get_act(proj_act)
        if proj_layers is None:
            proj_layers = [fc_dim] if fc_dim > 0 else []
        elif isinstance(proj_layers, int):
            proj_layers = [proj_layers] if proj_layers > 0 else []
        else:
            proj_layers = [int(width) for width in proj_layers if int(width) > 0]
        self.proj = StructuredNN(
            in_dim=layers[-1],
            mid_layers=proj_layers,
            out_dim=out_dim,
            act=self.proj_act,
        )

        self.normal_params = []
        self.inv_L_scale_params = []
        for _, param in self.named_parameters():
            if param is not self.inv_L_scale_latent:
                self.normal_params.append(param)
            else:
                if self.train_inv_L_scale == "together":
                    self.normal_params.append(param)
                elif self.train_inv_L_scale == "independently":
                    self.inv_L_scale_params.append(param)
                elif self.train_inv_L_scale is False:
                    continue
                else:
                    raise ValueError(f"{self.train_inv_L_scale} is not supported")

    def forward(self, x, aux):
        length = len(self.sp_convs)

        node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, outward_normals = aux


        bases_c,  bases_s,  bases_0  = compute_Fourier_bases(nodes, self.modes * (scaled_sigmoid(self.inv_L_scale_latent, self.inv_L_scale_min , self.inv_L_scale_max))) 

        wbases_c = torch.einsum("bxkw,bxw->bxkw", bases_c, node_weights)
        wbases_s = torch.einsum("bxkw,bxw->bxkw", bases_s, node_weights)
        wbases_0 = torch.einsum("bxkw,bxw->bxkw", bases_0, node_weights)

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        for i, (speconv, spw, spconvnw, spconvadjnw, w) in enumerate(zip(self.sp_convs, self.sp_ws, self.sp_convs_nws, self.sp_convs_adj_nws, self.ws)):
            
            if self.layer_selection['geointegral']:
                x1 = speconv( spconvnw(  torch.cat([x] + [x * outward_normals[:, i:i+1, :] for i in range(outward_normals.size(1))], dim=1)  ), bases_c, bases_s, bases_0, wbases_c, wbases_s, wbases_0)
                x1 = spw(x1) + spconvadjnw(torch.cat([x1 * outward_normals[:, i:i+1, :] for i in range(outward_normals.size(1))], dim=1))
            else:
                x1 = speconv(x, bases_c, bases_s, bases_0, wbases_c, wbases_s, wbases_0)
                x1 = spw(x1)
                
            x2 = w(x)

            
            if self.act is not None and i != length - 1:
                x = x + self.act(self.scale_factor*(x1 + x2))
            else:
                x = self.scale_factor*(x1 + x2)

        x = x.permute(0, 2, 1)
        x = self.proj(x)
        return x


# Backward-compatible alias name
MPCNO_NoGrad = MPCNO
