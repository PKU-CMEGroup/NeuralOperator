import torch
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from geo_utility import convert_structured_data, preprocess_data

torch.set_printoptions(precision=16)

torch.manual_seed(0)
np.random.seed(10)

downsample_ratio = 8

###################################
# load data
###################################
dataloader = loadmat("datasets\\burgers\\burgers_data_R10.mat")
data_in = np.array(dataloader.get('a'))
data_out = np.array(dataloader.get('u'))

print("data_in.shape " , data_in.shape)
print("data_out.shape", data_out.shape)

Np_ref = data_in.shape[1]
Np = 1 + (Np_ref - 1)//downsample_ratio
L = 1.0
grid_1d = np.linspace(0, L, Np)

data_in_ds = data_in[:, 0::downsample_ratio]
data_out_ds = data_out[:, 0::downsample_ratio]
ndata = data_in_ds.shape[0]

# generate uniform mask
mask = np.random.choice([0, 1], data_in_ds.shape, p=[0.5, 0.5])
sizes = np.zeros(ndata, dtype=int)
for n in range(ndata):
    sizes[n] = np.count_nonzero(mask[n])
plt.plot(sizes)
plt.xlabel("Data id")
plt.ylabel("Data size")
# plt.show()

'''
# mixed mask
mask1 = np.random.choice([0, 1], size=(200, data_in_ds.shape[1]), p=[0.0, 1.0])
mask2 = np.random.choice([0, 1], size=(200, data_in_ds.shape[1]), p=[0.2, 0.8])
mask3 = np.random.choice([0, 1], size=(200, data_in_ds.shape[1]), p=[0.4, 0.6])
mask4 = np.random.choice([0, 1], size=(200, data_in_ds.shape[1]), p=[0.6, 0.4])
mask5 = np.random.choice([0, 1], size=(200, data_in_ds.shape[1]), p=[0.8, 0.2])
mask6 = np.ones((ndata-1000, data_in_ds.shape[1]))
mask = np.concatenate((mask1, mask2, mask3, mask4, mask5, mask6), axis=0)
sizes = np.zeros(ndata, dtype=int)
for n in range(ndata):
    sizes[n] = len(np.nonzero(mask[n])[0])
plt.plot(sizes)
plt.xlabel("Data id")
plt.ylabel("Data size")
# plt.show()
'''

max_size = int(np.max(sizes))

elems_list = -np.ones((ndata, max_size, 3), dtype=int)
for n in range(ndata):
    for i in range(sizes[n]-1):
        points = np.nonzero(mask[n])[0]
        elems_list[n, i, :] = 2, points[i], points[i+1]

grid = [np.tile(grid_1d, (ndata, 1))]
grid[0] = grid[0] * mask
data_in_ds = data_in_ds * mask
data_out_ds = data_out_ds * mask

features = np.stack((data_in_ds, data_out_ds), axis = 2)

nodes_list, _, features_list = convert_structured_data(grid, features, nnodes_per_elem = 2, feature_include_coords = True)

# equal_weights data
nnodes, node_mask, nodes, node_weights, features, directed_edges, edge_gradient_weights = preprocess_data(nodes_list, elems_list, features_list, node_weight_type=None)
np.savez_compressed("datasets/burgers/geokno_uniform_coord_equal_weight.npz", nnodes=nodes, node_mask=node_mask, nodes=nodes, node_weights=node_weights, features=features, directed_edges=directed_edges, edge_gradient_weights=edge_gradient_weights)

# unequal_weights data
nnodes, node_mask, nodes, node_weights, features, directed_edges, edge_gradient_weights = preprocess_data(nodes_list, elems_list, features_list, node_weight_type="length")
np.savez_compressed("datasets/burgers/geokno_uniform_coord.npz", nnodes=nodes, node_mask=node_mask, nodes=nodes, node_weights=node_weights, features=features, directed_edges=directed_edges, edge_gradient_weights=edge_gradient_weights)
