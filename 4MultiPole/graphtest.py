import torch
import torch.nn as nn
from torch_geometric.nn import NNConv
from torch_geometric.data import Data

# Initialize NNConv layer
conv = NNConv(
    in_channels=3,
    out_channels=5,
    nn=nn.Linear(7, 3 * 5),
    aggr="mean",
    root_weight=True,
    bias=True,
)

# Example data
x = torch.randn((1, 10, 3))  # Node features: (batch_size, num_nodes, in_channels)
edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])  # Edge indices: (2, num_edges)
edge_attr = torch.randn((4, 7))  # Edge attributes: (num_edges, edge_attr_channels)

# Create a Data object for a single graph
data = Data(x=x.squeeze(0), edge_index=edge_index, edge_attr=edge_attr)

# Convert the single Data object into a batch
from torch_geometric.data import Batch

data_batch = Batch.from_data_list([data] * 4)  # Create a batch with 4 identical graphs

# Run the NNConv layer
out = conv(data_batch.x, data_batch.edge_index, data_batch.edge_attr)
print(out.size())  # Should be (batch_size * num_graphs, num_nodes, out_channels)
