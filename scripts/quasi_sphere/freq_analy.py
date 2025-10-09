import os
import torch
import sys

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from timeit import default_timer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from pcno.geo_utility import preprocess_data_mesh, compute_node_weights
from pcno.pcno import compute_Fourier_modes, PCNO, PCNO_train

device = 'cpu'

data_path = "../../data/quasi_sphere/NPYSmax_l4"

equal_weights = False
data = np.load(data_path+"/pcno_quasisphere_data.npz")
nnodes, node_mask, nodes = data["nnodes"], data["node_mask"], data["nodes"]
print(nnodes.shape,node_mask.shape,nodes.shape,flush = True)
node_weights = data["node_equal_weights"] if equal_weights else data["node_weights"]
node_measures = data["node_measures"]
directed_edges, edge_gradient_weights = data["directed_edges"], data["edge_gradient_weights"]
features = data["features"]

node_measures_raw = data["node_measures_raw"]
indices = np.isfinite(node_measures_raw)
node_rhos = np.copy(node_weights)
node_rhos[indices] = node_rhos[indices]/node_measures[indices]

print("Casting to tensor",flush = True)
nnodes = torch.from_numpy(nnodes)
node_mask = torch.from_numpy(node_mask)
nodes = torch.from_numpy(nodes.astype(np.float32))
node_weights = torch.from_numpy(node_weights.astype(np.float32))
node_rhos = torch.from_numpy(node_rhos.astype(np.float32))
features = torch.from_numpy(features.astype(np.float32))
directed_edges = torch.from_numpy(directed_edges.astype(np.int64))
edge_gradient_weights = torch.from_numpy(edge_gradient_weights.astype(np.float32))


nodes_input = nodes.clone()

n_train, n_test = 900,100


x_train, x_test = torch.cat((nodes_input[:n_train, ...], node_rhos[:n_train, ...]), -1), torch.cat((nodes_input[-n_test:, ...], node_rhos[-n_test:, ...]),-1)

aux_train       = (node_mask[0:n_train,...], nodes[0:n_train,...], node_weights[0:n_train,...], directed_edges[0:n_train,...], edge_gradient_weights[0:n_train,...])
aux_test        = (node_mask[-n_test:,...],  nodes[-n_test:,...],  node_weights[-n_test:,...],  directed_edges[-n_test:,...],  edge_gradient_weights[-n_test:,...])

y_train, y_test = features[:n_train, :, :1],     features[-n_test:, :, :1]


train_inv_L_scale = False
k_max = 8
ndim = 3
modes = compute_Fourier_modes(ndim, [k_max,k_max,k_max], [3,3,3])
modes = torch.tensor(modes, dtype=torch.float).to(device)
model = PCNO(ndim, modes, nmeasures=1,
               layers=[128,128,128,128,128],
               fc_dim=128,
               in_dim=x_train.shape[-1], out_dim=y_train.shape[-1],
               inv_L_scale_hyper = [train_inv_L_scale, 0.5, 2.0],
               act='gelu').to(device)

model_path = 'PCNO_quasisphere_model_gathered.pth'
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint)



from mpl_toolkits.mplot3d import Axes3D

weights_c_list = [layer.weights_c for layer in model.sp_convs]  # weight_c: [in_channels, out_channels, nmodes, nmeasures]
weights_s_list = [layer.weights_s for layer in model.sp_convs]  # weight_s: [in_channels, out_channels, nmodes, nmeasures]

freq_c_norm_list = [torch.norm(w, dim=(0,1)).reshape(-1).detach() for w in weights_c_list]
freq_s_norm_list = [torch.norm(w, dim=(0,1)).reshape(-1).detach() for w in weights_s_list]

np.savez(f"npzs/freq_norm_layers_k{k_max}_gathered.npz",
         freq_c_norm_list = [f.cpu().numpy() for f in freq_c_norm_list],
         freq_s_norm_list = [f.cpu().numpy() for f in freq_s_norm_list],
         modes = modes.cpu().numpy())

import matplotlib.pyplot as plt

n_layers = len(freq_c_norm_list)
fig = plt.figure(figsize=(6 * n_layers, 12))

for i in range(n_layers):
    # freq_c_norm_list[i] and freq_s_norm_list[i] shape: [nmodes]
    nmodes = freq_c_norm_list[i].shape[0]
    mode_xyz = modes[:nmodes].cpu().numpy()  # modes shape: [nmodes, 3]
    norm_c = freq_c_norm_list[i].cpu().numpy()
    norm_s = freq_s_norm_list[i].cpu().numpy()

    ax_c = fig.add_subplot(2, n_layers, i + 1, projection='3d')
    ax_c.scatter(mode_xyz[:, 0], mode_xyz[:, 1], mode_xyz[:, 2], c=norm_c, cmap='viridis')
    ax_c.set_title(f'Layer {i+1} freq_c_norm')
    ax_c.set_xlabel('kx')
    ax_c.set_ylabel('ky')
    ax_c.set_zlabel('kz')

    ax_s = fig.add_subplot(2, n_layers, n_layers + i + 1, projection='3d')
    ax_s.scatter(mode_xyz[:, 0], mode_xyz[:, 1], mode_xyz[:, 2], c=norm_s, cmap='plasma')
    ax_s.set_title(f'Layer {i+1} freq_s_norm')
    ax_s.set_xlabel('kx')
    ax_s.set_ylabel('ky')
    ax_s.set_zlabel('kz')

plt.tight_layout()
plt.savefig(f"figures/freq_norm_layers_k{k_max}_gathered.png")