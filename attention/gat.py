import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from myutils.basics import _get_act
import numpy as np
from dgl import function as fn
from dgl.ops import edge_softmax


class GATLayer(nn.Module):
    """A simple local softmax attention module for DGL graphs."""

    def __init__(
        self, in_channels: int, hidden_channels: int, out_channels: int, n_heads=2
    ):
        """Local softmax attention layer.

        Args:
            n_heads: number of attention heads
            in_channels: input dimension of features
            out_channels: output dimension of features
        """
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        assert hidden_channels % n_heads == 0
        assert out_channels % n_heads == 0
        self.n_heads = n_heads
        self.dk = self.hidden_channels // self.n_heads
        self.dv = self.out_channels // self.n_heads

        # Linear transformations for keys, queries, and values
        self.Wk = nn.Linear(in_channels, hidden_channels)
        self.Wq = nn.Linear(in_channels, hidden_channels)
        self.Wv = nn.Linear(in_channels, out_channels)

    def forward(self, g, x):
        """Forward pass of the attention layer.

        Args:
            g: DGL graph
            node_features: Node feature tensor of shape [N, in_channels]

        Returns:
            Output tensor of shape [N, n_heads, out_channels]
        """

        batch_size = x.shape[0]
        with g.local_scope():
            # Transform node features into keys, queries, and values
            key = self.Wk(x).view(-1, self.n_heads, self.dk)
            query = self.Wq(x).view(-1, self.n_heads, self.dk)
            value = self.Wv(x).view(-1, self.n_heads, self.dv)

            g.ndata["k"] = key
            g.ndata["q"] = query
            g.ndata["v"] = value

            # Compute attention scores using dot product
            g.apply_edges(fn.u_dot_v("k", "q", "e"))
            g.edata["e"] = edge_softmax(g, g.edata["e"])

            # Perform attention-weighted message passing
            g.update_all(fn.u_mul_e("v", "e", "m"), fn.sum("m", "res"))

            # Gather outputs and reshape
            return g.ndata["res"].view(batch_size, -1, self.out_channels)


class GAT(nn.Module):
    def __init__(self, **config):
        super(GAT, self).__init__()

        self.config = defaultdict(lambda: None, **config)
        self.config = dict(self.config)
        all_attr = list(self.config.keys())
        for key in all_attr:
            setattr(self, key, self.config[key])

        # input channel is 2: (a(x), x) or 3: (F(x), a(x), x)
        self.fc0 = nn.Linear(self.in_dim, self.layer_channels[0])

        self.attn_layers = nn.ModuleList(
            [
                GATLayer(in_channels, in_channels, out_channels)
                for in_channels, out_channels in zip(
                    self.layer_channels, self.layer_channels[1:]
                )
            ]
        )
        self.ws = nn.ModuleList(
            [
                nn.Linear(in_channels, out_channels)
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

    def forward(self, g, x):

        x = self.fc0(x)

        for i, (layer, w) in enumerate(zip(self.attn_layers, self.ws)):
            x = w(x) + layer(g, x)

            # activation function
            if self.act is not None and i != self.length - 1:
                x = self.act(x)

        if self.fc_channels > 0:
            x = self.fc1(x)
            if self.act is not None:
                x = self.act(x)
        x = self.fc2(x)

        return x





# class GATLayer(nn.Module):
#     """A simple local softmax attention module for DGL graphs."""

#     def __init__(
#         self, in_channels: int, hidden_channels: int, out_channels: int, n_heads=2
#     ):
#         """Local softmax attention layer.

#         Args:
#             n_heads: number of attention heads
#             in_channels: input dimension of features
#             out_channels: output dimension of features
#         """
#         super().__init__()

#         self.in_channels = in_channels
#         self.hidden_channels = hidden_channels
#         self.out_channels = out_channels

#         assert hidden_channels % n_heads == 0
#         assert out_channels % n_heads == 0
#         self.n_heads = n_heads
#         self.dk = self.hidden_channels // self.n_heads
#         self.dv = self.out_channels // self.n_heads

#         self.Wk = nn.Linear(in_channels, hidden_channels)
#         self.Wq = nn.Linear(in_channels, hidden_channels)
#         self.Wv = nn.Linear(in_channels, out_channels)

#     def forward(self, g, x):
#         """Forward pass of the attention layer.

#         Args:
#             g: DGL graph
#             node_features: Node feature tensor of shape [N, in_channels]

#         Returns:
#             Output tensor of shape [N, n_heads, out_channels]
#         """
#         start_total = time.time()  # Total start time

#         batch_size = x.shape[0]

#         with g.local_scope():
#             # 1. Transform node features into keys, queries, and values
#             start_kqv = time.time()
#             key = self.Wk(x).view(-1, self.n_heads, self.dk)
#             query = self.Wq(x).view(-1, self.n_heads, self.dk)
#             value = self.Wv(x).view(-1, self.n_heads, self.dv)
#             end_kqv = time.time()
#             print(f"Key, Query, Value calculation time: {end_kqv - start_kqv:.6f}s")

#             # 2. Assign key, query, and value to graph nodes
#             start_assign = time.time()
#             g.ndata["k"] = key
#             g.ndata["q"] = query
#             g.ndata["v"] = value
#             end_assign = time.time()
#             print(
#                 f"Assign key, query, value to nodes: {end_assign - start_assign:.6f}s"
#             )

#             # 3. Compute attention scores using dot product
#             start_attn = time.time()
#             g.apply_edges(fn.u_dot_v("k", "q", "e"))
#             end_attn = time.time()
#             print(f"Attention score calculation time: {end_attn - start_attn:.6f}s")

#             # 4. Softmax
#             start_attn = time.time()
#             g.edata["e"] = edge_softmax(g, g.edata["e"])
#             end_attn = time.time()
#             print(f"Softmax time: {end_attn - start_attn:.6f}s")

#             # 5. Perform attention-weighted message passing
#             start_msg_passing = time.time()
#             g.update_all(fn.u_mul_e("v", "e", "m"), fn.sum("m", "res"))
#             end_msg_passing = time.time()
#             print(f"Message passing time: {end_msg_passing - start_msg_passing:.6f}s")

#             # 6. Gather outputs and reshape
#             start_output = time.time()
#             output = g.ndata["res"].view(batch_size, -1, self.out_channels)
#             end_output = time.time()
#             print(
#                 f"Output gathering and reshaping time: {end_output - start_output:.6f}s"
#             )

#         end_total = time.time()
#         print(f"Total forward pass time: {end_total - start_total:.6f}s")
#         sys.exit()
#         return output



