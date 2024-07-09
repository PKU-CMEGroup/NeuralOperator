import torch
from torch import nn
import torch.fft as fft
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_

import math
import copy
from functools import partial
from collections import defaultdict


class SimpleTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model=96,
        pos_dim=1,
        n_head=2,
        dim_feedforward=512,
        attention_type="fourier",
        # xavier_init: float = 1e-2,
        # diagonal_weight: float = 1e-2,
    ):
        super(SimpleTransformerEncoderLayer, self).__init__()

        self.attn = SimpleAttention(
            n_head=n_head,
            d_model=d_model,
            attention_type=attention_type,
            pos_dim=pos_dim,
            eps=1e-5,
            # diagonal_weight=diagonal_weight,
            # xavier_init=xavier_init,
        )
        self.d_model = d_model
        self.n_head = n_head
        self.pos_dim = pos_dim

        dim_feedforward = dim_feedforward
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
        )

        self.__name__ = attention_type.capitalize() + "TransformerEncoderLayer"

    def forward(self, x, pos=None):
        if pos is not None and self.pos_dim > 0:
            att_output = self.attn(x, x, x, pos=pos)

        x = x + att_output

        x1 = self.ff(x)
        x = x + x1

        return x


class SpectralRegressor(nn.Module):
    def __init__(
        self,
        in_dim,
        freq_dim,
        out_dim,
        modes: int,
        num_spectral_layers: int = 2,
        dim_feedforward=None,
    ):
        super(SpectralRegressor, self).__init__()
        self.activation = nn.SiLU()

        self.spectral_conv = nn.ModuleList(
            [
                SpectralConv1d(
                    in_dim=in_dim,
                    out_dim=freq_dim,
                    modes=modes,
                )
            ]
        )
        for _ in range(num_spectral_layers - 1):
            self.spectral_conv.append(
                SpectralConv1d(
                    in_dim=freq_dim,
                    out_dim=freq_dim,
                    modes=modes,
                )
            )

        self.dim_feedforward = dim_feedforward
        self.regressor = nn.Sequential(
            nn.Linear(freq_dim, self.dim_feedforward),
            self.activation,
            nn.Linear(self.dim_feedforward, out_dim),
        )

    def forward(self, x):
        """
        1D:
            Input: (-1, n, in_features)
            Output: (-1, n, n_targets)
        """

        for layer in self.spectral_conv:
            x = layer(x)

        x = self.regressor(x)

        return x


class SimpleTransformer(nn.Module):
    def __init__(self, **kwargs):
        super(SimpleTransformer, self).__init__()
        self.config = defaultdict(lambda: None, **kwargs)
        self._get_setting()
        self._initialize()
        self.__name__ = self.attention_type.capitalize() + "Transformer"

    def forward(self, node, edge, pos, grid=None, weight=None):
        x = self.feat_extract(node)

        for encoder in self.encoder_layers:
            x = encoder(x, pos)

        x = self.regressor(x)

        return dict(preds=x, preds_freq=None, preds_latent=[], attn_weights=[])

    def _initialize(self):
        self.feat_extract = nn.Linear(self.node_feats, self.n_hidden)
        self._get_encoder()
        self._get_regressor()

        self.config = dict(self.config)

    def _get_setting(self):
        all_attr = list(self.config.keys())
        for key in all_attr:
            setattr(self, key, self.config[key])

    def _get_encoder(self):
        encoder_layer = SimpleTransformerEncoderLayer(
            d_model=self.n_hidden,
            n_head=self.n_head,
            attention_type=self.attention_type,
            dim_feedforward=self.dim_feedforward,
            pos_dim=self.pos_dim,
            xavier_init=self.xavier_init,
            diagonal_weight=self.diagonal_weight,
        )
        self.encoder_layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(self.num_encoder_layers)]
        )

    def _get_regressor(self):
        self.regressor = SpectralRegressor(
            in_dim=self.n_hidden,
            freq_dim=self.freq_dim,
            out_dim=self.n_targets,
            modes=self.fourier_modes,
            num_spectral_layers=self.num_regressor_layers,
            dim_feedforward=self.freq_dim,
        )


# def attention(query, key, value):
#     d_k = query.size(-1)
#     scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
#     seq_len = scores.size(-1)
#     p_attn = scores / seq_len
#     p_attn = F.dropout(p_attn)
#     return torch.matmul(p_attn, value)


# class SimpleAttention(nn.Module):
#     def __init__(
#         self,
#         n_head,
#         d_model,
#         pos_dim: int = 1,
#         attention_type="fourier",
#         eps=1e-5,
#         xavier_init=1e-2,
#         diagonal_weight=1e-2,
#     ):
#         super(SimpleAttention, self).__init__()
#         self.attention_type = attention_type

#         assert d_model % n_head == 0
#         self.d_k = d_model // n_head
#         self.n_head = n_head
#         self.pos_dim = pos_dim

#         self.linears = nn.ModuleList(
#             [copy.deepcopy(nn.Linear(d_model, d_model)) for _ in range(3)]
#         )
#         self.xavier_init = xavier_init
#         self.diagonal_weight = diagonal_weight
#         self._reset_parameters()

#         self._get_norm(eps=eps)

#         if pos_dim > 0:
#             self.fc = nn.Linear(d_model + n_head * pos_dim, d_model)

#     def forward(self, query, key, value, pos=None):
#         bsz = query.size(0)

#         query, key, value = [
#             layer(x).view(bsz, -1, self.n_head, self.d_k).transpose(1, 2)
#             for layer, x in zip(self.linears, (query, key, value))
#         ]

#         if self.attention_type == "galerkin":
#             key = torch.stack(
#                 [
#                     norm(x)
#                     for norm, x in zip(
#                         self.norm_K, (key[:, i, ...] for i in range(self.n_head))
#                     )
#                 ],
#                 dim=1,
#             )
#             value = torch.stack(
#                 [
#                     norm(x)
#                     for norm, x in zip(
#                         self.norm_V, (value[:, i, ...] for i in range(self.n_head))
#                     )
#                 ],
#                 dim=1,
#             )

#         elif self.attention_type == "fourier":
#             key = torch.stack(
#                 [
#                     norm(x)
#                     for norm, x in zip(
#                         self.norm_K, (key[:, i, ...] for i in range(self.n_head))
#                     )
#                 ],
#                 dim=1,
#             )
#             query = torch.stack(
#                 [
#                     norm(x)
#                     for norm, x in zip(
#                         self.norm_Q, (query[:, i, ...] for i in range(self.n_head))
#                     )
#                 ],
#                 dim=1,
#             )

#         if pos is not None and self.pos_dim > 0:
#             assert pos.size(-1) == self.pos_dim
#             pos = pos.unsqueeze(1)
#             pos = pos.repeat([1, self.n_head, 1, 1])
#             query, key, value = [
#                 torch.cat([pos, x], dim=-1) for x in (query, key, value)
#             ]

#         x = attention(query, key, value)

#         out_dim = (
#             self.n_head * self.d_k
#             if pos is None
#             else self.n_head * (self.d_k + self.pos_dim)
#         )
#         att_output = x.transpose(1, 2).contiguous().view(bsz, -1, out_dim)

#         if pos is not None and self.pos_dim > 0:
#             att_output = self.fc(att_output)

#         return att_output

#     def _get_norm(self, eps):
#         if self.attention_type == "galerkin":
#             self.norm_K = self._get_layernorm(self.d_k, self.n_head, eps=eps)
#             self.norm_V = self._get_layernorm(self.d_k, self.n_head, eps=eps)
#         elif self.attention_type == "fourier":
#             self.norm_K = self._get_layernorm(self.d_k, self.n_head, eps=eps)
#             self.norm_Q = self._get_layernorm(self.d_k, self.n_head, eps=eps)

#     @staticmethod
#     def _get_layernorm(normalized_dim, n_head, **kwargs):
#         return nn.ModuleList(
#             [
#                 copy.deepcopy(nn.LayerNorm(normalized_dim, **kwargs))
#                 for _ in range(n_head)
#             ]
#         )

#     def _reset_parameters(self):
#         for param in self.linears.parameters():
#             if param.ndim > 1:
#                 xavier_uniform_(param, gain=self.xavier_init)
#                 param.data += self.diagonal_weight * torch.diag(
#                     torch.ones(param.size(-1), dtype=torch.float)
#                 )
#             else:
#                 constant_(param, 0)


class SpectralConv1d(nn.Module):
    def __init__(self, in_dim, out_dim, modes: int):
        super(SpectralConv1d, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.modes = modes

        self.fourier_weight = Parameter(torch.FloatTensor(in_dim, out_dim, modes, 2))
        xavier_normal_(self.fourier_weight, gain=1 / (in_dim * out_dim))

        self.activation = nn.SiLU()

    def forward(self, x):
        seq_len = x.size(1)
        res = self.linear(x)

        x = x.permute(0, 2, 1)
        x_ft = fft.rfft(x, n=seq_len, norm="ortho")
        x_ft = torch.stack([x_ft.real, x_ft.imag], dim=-1)

        out_ft = self.complex_matmul_1d(x_ft[:, :, : self.modes], self.fourier_weight)

        pad_size = seq_len // 2 + 1 - self.modes
        out_ft = F.pad(out_ft, (0, 0, 0, pad_size), "constant", 0)

        out_ft = torch.complex(out_ft[..., 0], out_ft[..., 1])

        x = fft.irfft(out_ft, n=seq_len, norm="ortho")

        x = x.permute(0, 2, 1)
        x = self.activation(x + res)

        return x

    @staticmethod
    def complex_matmul_1d(a, b):
        op = partial(torch.einsum, "bix,iox->box")
        return torch.stack(
            [
                op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
                op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1]),
            ],
            dim=-1,
        )
