import math
import sys
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_
from collections import defaultdict
from myutils.basics import _get_act, mul_index1, mul_index2
import math


class SimpleAttention(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_heads,
        attention_type,
        eps=1e-5,
    ):
        super(SimpleAttention, self).__init__()

        assert in_channels % num_heads == 0
        self.in_channels = in_channels
        self.d_k = in_channels // num_heads
        self.num_heads = num_heads
        self.attention_type = attention_type

        self.linears = nn.ModuleList(
            [copy.deepcopy(nn.Linear(in_channels, in_channels))
             for _ in range(3)]
        )
        self.xavier_init = 1e-2
        self.diagonal_weight = 1e-2
        self._reset_parameters()

        self.fc = nn.Linear(in_channels, out_channels)

        self._get_norm(eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        batchsize = x.size(0)

        query, key, value = [
            layer(x).view(batchsize, -1, self.num_heads, self.d_k).transpose(1, 2)
            for layer, x in zip(self.linears, (x, x, x))
        ]

        if self.attention_type == "galerkin":
            key = torch.stack(
                [
                    norm(x)
                    for norm, x in zip(
                        self.norm_K, (key[:, i, ...]
                                      for i in range(self.num_heads))
                    )
                ],
                dim=1,
            )
            value = torch.stack(
                [
                    norm(x)
                    for norm, x in zip(
                        self.norm_V, (value[:, i, ...]
                                      for i in range(self.num_heads))
                    )
                ],
                dim=1,
            )

        elif self.attention_type == "fourier":
            key = torch.stack(
                [
                    norm(x)
                    for norm, x in zip(
                        self.norm_K, (key[:, i, ...]
                                      for i in range(self.num_heads))
                    )
                ],
                dim=1,
            )
            query = torch.stack(
                [
                    norm(x)
                    for norm, x in zip(
                        self.norm_Q, (query[:, i, ...]
                                      for i in range(self.num_heads))
                    )
                ],
                dim=1,
            )

        x = self._attention(query, key, value)
        x = x.transpose(1, 2).contiguous().view(batchsize, -1, self.in_dim)

        x = self.fc(x)
        x = x.permute(0, 2, 1)

        return x

    @staticmethod
    def _attention(query, key, value):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        seq_len = scores.size(-1)
        p_attn = scores / seq_len
        p_attn = F.dropout(p_attn)
        return torch.matmul(p_attn, value)

    def _get_norm(self, eps):
        if self.attention_type == "galerkin":
            self.norm_K = self._get_layernorm(self.d_k, self.num_heads, eps=eps)
            self.norm_V = self._get_layernorm(self.d_k, self.num_heads, eps=eps)
        elif self.attention_type == "fourier":
            self.norm_K = self._get_layernorm(self.d_k, self.num_heads, eps=eps)
            self.norm_Q = self._get_layernorm(self.d_k, self.num_heads, eps=eps)

    @staticmethod
    def _get_layernorm(normalized_dim, num_heads, **kwargs):
        return nn.ModuleList(
            [
                copy.deepcopy(nn.LayerNorm(normalized_dim, **kwargs))
                for _ in range(num_heads)
            ]
        )

    def _reset_parameters(self):
        for param in self.linears.parameters():
            if param.ndim > 1:
                xavier_uniform_(param, gain=self.xavier_init)
                param.data += self.diagonal_weight * torch.diag(
                    torch.ones(param.size(-1), dtype=torch.float)
                )
            else:
                constant_(param, 0)


class IterLocalSoftmaxAttention(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, neighbor_index):
        super(IterLocalSoftmaxAttention, self).__init__()

        self.neighbor_index = neighbor_index

        self.linears = nn.ModuleList(
            [
                copy.deepcopy(
                    nn.Linear(
                        in_channels,
                        hidden_channels,
                    )
                )
                for _ in range(3)
            ]
        )

        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        query, key, value = [layer(x)
                             for layer, x in zip(self.linears, (x, x, x))]
        x = self._attention(query, key, value, self.neighbor_index)

        x = self.fc(x)
        x = x.permute(0, 2, 1)

        return x

    @staticmethod
    def _attention(query, key, value, index):

        scores = [
            torch.einsum(
                "bc,bic->bi",
                query[:, j, :],
                key[:, idx, :],
            )
            for j, idx in enumerate(index)
        ]
        scores = [F.softmax(s / math.sqrt(query.size(-1)), dim=1)
                  for s in scores]

        attn = [
            torch.einsum(
                "bi,bic->bc",
                scores[j],
                value[:, idx, :],
            )
            for j, idx in enumerate(index)
        ]
        attn = torch.stack(attn, dim=1)

        return attn


class LocalSoftmaxAttention(nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, heads, neighbor_index
    ):
        super(LocalSoftmaxAttention, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        assert (
            hidden_channels % heads == 0
        ), f"hidden channels({hidden_channels}) % heads({heads}) != 0"
        self.heads = heads
        self.d = hidden_channels // heads

        self.neighbor_index = neighbor_index

        self.linears = nn.ModuleList(
            [
                copy.deepcopy(
                    nn.Linear(
                        in_channels,
                        hidden_channels,
                    )
                )
                for _ in range(3)
            ]
        )

        # self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x):

        batch_size = x.shape[0]

        query, key, value = [
            layer(x).reshape(batch_size, -1, self.heads, self.d)
            for layer, x in zip(self.linears, (x, x, x))
        ]
        key = key[:, self.neighbor_index, ...]
        value = value[:, self.neighbor_index, ...]
        x = self._attention(query, key, value).reshape(
            batch_size, -1, self.hidden_channels
        )

        # x = self.fc(x)

        return x

    @staticmethod
    def _attention(query, key, value):
        scores = mul_index1(query, key)
        scores = F.softmax(scores / math.sqrt(query.size(-1)), dim=1)
        attn = mul_index2(scores, value)
        return attn


class SoftmaxAttention(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SoftmaxAttention, self).__init__()

        self.linears = nn.ModuleList(
            [nn.Linear(in_channels, hidden_channels) for _ in range(2)]
        )
        self.linears.append(nn.Linear(in_channels, out_channels))

        self.fc = nn.Linear(hidden_channels, out_channels)

    def forward(self, x):

        query, key, value = [layer(x)
                             for layer, x in zip(self.linears, (x, x, x))]

        x = self._attention(query, key, value)

        return x

    @staticmethod
    def _attention(query, key, value):
        channels, len = query.size(-1), query.size(-2)
        scores = torch.matmul(query, key.transpose(-2, -1)
                              ) / math.sqrt(channels)
        scores = torch.softmax(scores, dim=-1) / len
        return torch.matmul(scores, value)


class AttnNO(nn.Module):
    def __init__(self, neighbor_index=None, **config):
        super(AttnNO, self).__init__()

        self.neighbor_index = neighbor_index

        self.config = defaultdict(lambda: None, **config)
        self.config = dict(self.config)
        all_attr = list(self.config.keys())
        for key in all_attr:
            setattr(self, key, self.config[key])

        self.fc0 = nn.Linear(self.in_dim, self.layer_channels[0])

        self.length = len(self.layer_channels) - 1
        self.attn_layers = nn.ModuleList(
            [
                self._choose_layer(index, in_channels, out_channels, layer_type)
                for index, (in_channels, out_channels, layer_type) in enumerate(
                    zip(self.layer_channels,
                        self.layer_channels[1:], self.layer_types)
                )
            ]
        )
        self.ws = nn.ModuleList(
            [
                nn.Linear(in_channels, out_channels)
                for (in_channels, out_channels) in zip(
                    self.layer_channels, self.layer_channels[1:]
                )
            ]
        )
        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(channels, self.fc_channels),
                    nn.GELU(),
                    nn.Linear(self.fc_channels, channels),
                )
                for channels in self.layer_channels[1:]
            ]
        )
        self.norms1 = nn.ModuleList(
            [nn.LayerNorm(channels) for channels in self.layer_channels[1:]]
        )
        self.norms2 = nn.ModuleList(
            [nn.LayerNorm(channels) for channels in self.layer_channels[1:]]
        )

        if self.fc_channels > 0:
            self.fc1 = nn.Linear(self.layer_channels[-1], self.fc_channels)
            self.fc2 = nn.Linear(self.fc_channels, self.out_dim)
        else:
            self.fc2 = nn.Linear(self.layer_channels[-1], self.out_dim)

        self.act = _get_act(self.act)

    def forward(self, x):

        x = self.fc0(x)

        # for layer, ffn, norm1, norm2 in zip(
        #     self.attn_layers, self.ffns, self.norms1, self.norms2
        # ):
        #     res = x + layer(x)
        #     res = norm1(res)

        #     x = res + ffn(res)
        #     x = norm2(x)

        length = len(self.ws)
        for i, (layer, w) in enumerate(zip(self.attn_layers, self.ws)):
            x1 = layer(x)
            x2 = w(x)
            x = x1 + x2
            if self.act is not None and i != length - 1:
                x = self.act(x)

        if self.fc_channels > 0:
            x = self.fc1(x)
            if self.act is not None:
                x = self.act(x)
        x = self.fc2(x)

        return x

    def _choose_layer(self, index, in_channels, out_channels, layer_type):
        if layer_type == "SimpleAttention":
            modes = self.modes_gk[index]
            bases = self.bases_dict[self.bases_types[index]][0]
            wbases = self.bases_dict[self.wbases_types[index]][1]
            return SimpleAttention(
                in_channels,
                out_channels,
                modes,
                bases,
                wbases,
                self.diag_widths[index],
            )
        elif layer_type == "LocalSoftmaxAttention":
            return LocalSoftmaxAttention(
                in_channels, in_channels, out_channels, self.heads, self.neighbor_index
            )
        elif layer_type == "SoftmaxAttention":
            return SoftmaxAttention(in_channels, in_channels, out_channels)

        else:
            raise KeyError("Layer Type Undefined.")


# class LocalSoftmaxAttention(nn.Module):
#     def __init__(
#         self, in_channels, hidden_channels, out_channels, heads, neighbor_index
#     ):
#         super(LocalSoftmaxAttention, self).__init__()

#         self.in_channels = in_channels
#         self.hidden_channels = hidden_channels
#         self.out_channels = out_channels

#         assert (
#             hidden_channels % heads == 0
#         ), f"hidden channels({hidden_channels}) % heads({heads}) != 0"
#         self.heads = heads
#         self.d = hidden_channels // heads

#         self.neighbor_index = neighbor_index

#         self.linears = nn.ModuleList(
#             [
#                 copy.deepcopy(
#                     nn.Linear(
#                         in_channels,
#                         hidden_channels,
#                     )
#                 )
#                 for _ in range(3)
#             ]
#         )
#         self.print_memory_usage("1")
#         # self.fc = nn.Linear(in_channels, out_channels)

#     def forward(self, x):

#         self.print_memory_usage("1")
#         batch_size = x.shape[0]

#         query, key, value = [
#             layer(x).reshape(batch_size, -1, self.heads, self.d)
#             for layer, x in zip(self.linears, (x, x, x))
#         ]
#         self.print_memory_usage("2")
#         key = key[:, self.neighbor_index, :, :]
#         print(key.shape)
#         self.print_memory_usage("3")

#         value = value[:, self.neighbor_index, :, :]
#         self.print_memory_usage("4")

#         start_time = time.time()
#         x = mul_index1(query, key)
#         end_time = time.time()
#         print(f"score{end_time-start_time}")
#         self.print_memory_usage("5")

#         start_time = time.time()
#         x = F.softmax(x / math.sqrt(query.size(-1)), dim=1)
#         end_time = time.time()
#         print(f"softmax{end_time-start_time}")
#         self.print_memory_usage("6")

#         start_time = time.time()
#         x = mul_index2(x, value)
#         end_time = time.time()
#         print(f"back{end_time-start_time}")
#         self.print_memory_usage("7")

#         x = x.reshape(batch_size, -1, self.hidden_channels)

#         # x = self.fc(x)

#         return x

#     def print_memory_usage(self, message):
#         allocated = torch.cuda.memory_allocated()
#         reserved = torch.cuda.memory_reserved()
#         print(
#             f"{message}: Allocated = {allocated / 1024**2:.2f} MB, Reserved = {reserved / 1024**2:.2f} MB"
#         )

#     @staticmethod
#     def _attention(query, key, value):

#         start_time = time.time()
#         scores = mul_index1(query, key)
#         end_time = time.time()
#         print(f"score{end_time-start_time}")

#         start_time = time.time()
#         scores = F.softmax(scores / math.sqrt(query.size(-1)), dim=1)
#         end_time = time.time()
#         print(f"softmax{end_time-start_time}")

#         start_time = time.time()
#         attn = mul_index2(scores, value)
#         end_time = time.time()
#         print(f"back{end_time-start_time}")

#         # sys.exit()
#         return attn
