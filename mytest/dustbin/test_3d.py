import os
import glob
import random
import torch
import sys
import numpy as np
import math
from timeit import default_timer
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from pcno.geo_utility import preprocess_data, compute_node_weights
from pcno.dustbin.pcno_normfix import truncate, compute_Fourier_modes, compute_Fourier_bases, direct

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_path = "../data/car_shapenet/"
equal_weights = False
data = np.load(data_path+"pcno_triangle_data.npz")
nnodes, node_mask, nodes = data["nnodes"], data["node_mask"], data["nodes"]
node_weights = data["node_equal_weights"] if equal_weights else data["node_weights"]
node_measures = data["node_measures"]
directed_edges, edge_gradient_weights = data["directed_edges"], data["edge_gradient_weights"]
features = data["features"]

i = 10
x = torch.from_numpy(nodes[i:i+1]).to(device)
nx = torch.from_numpy(features[i:i+1, :, 1:]).to(device)
y = torch.from_numpy(features[i:i+1, :, 0:1]).to(device)
node_weights = torch.from_numpy(node_weights[i:i+1]).to(device)



k_max = 10
L = [2.0,2.0,5.0]
# L = [4.0,4.0,10.0]
sigma = 0.1


print(f'sigma = {sigma}, k_max = {k_max}, L = {L}')

modes = torch.tensor(compute_Fourier_modes(3, [k_max,k_max,k_max], L)).to(device)  # nmodes, 2, 1
bases_c,  bases_s,  bases_0  = compute_Fourier_bases(x, modes)  # (1, n, nmodes, 1) , (1, n, nmodes, 1), (1, n, 1, 1)

wbases_c = torch.einsum("bxkw,bxw->bxkw", bases_c, node_weights)
wbases_s = torch.einsum("bxkw,bxw->bxkw", bases_s, node_weights)
wbases_0 = torch.einsum("bxkw,bxw->bxkw", bases_0, node_weights)

y_out1 = truncate(nx, y.transpose(2,1), wbases_c, wbases_s, wbases_0, bases_c, bases_s, bases_0, modes, sigma).transpose(2,1)
y_out2 = direct(y.transpose(2,1), wbases_c, wbases_s, wbases_0, bases_c, bases_s, bases_0).transpose(2,1)


# argmin_c ||y - c*y_out|| = <y, y_out>/||y_out||^2
# print(y.shape,y_out.shape)
coffe1 = torch.dot(y[0,:,0], y_out1[0,:,0])/torch.norm(y_out1[0,:,0])**2
y_out1 = y_out1 * coffe1

coffe2 = torch.dot(y[0,:,0], y_out2[0,:,0])/torch.norm(y_out2[0,:,0])**2
y_out2 = y_out2 * coffe2

print('loss(normfix): ', (torch.norm(y - y_out1)/torch.norm(y)).item())
print('loss(direct): ', (torch.norm(y - y_out2)/torch.norm(y)).item())



x, y, y_out1, y_out2 = x.to('cpu'), y.to('cpu'), y_out1.to('cpu'), y_out2.to('cpu') 
fig, ax = plt.subplots(2, 3,figsize=(18,14), subplot_kw={'projection': '3d'})

surf1 = ax[0,0].scatter(x[0,:,0], x[0,:,1], x[0,:,2], c=y[0,:,0], s=10)
fig.colorbar(surf1 , ax=ax[0,0], shrink=0.5, aspect=10, pad=0.1)
ax[0,0].view_init(elev=-50, azim=50)
ax[0,0].set_title('y_ref', pad=30)

surf2 = ax[0,1].scatter(x[0,:,0], x[0,:,1], x[0,:,2], c=y_out1[0,:,0], s=10)
fig.colorbar(surf2 , ax=ax[0,1], shrink=0.5, aspect=10, pad=0.1)
ax[0,1].view_init(elev=-50, azim=50)
ax[0,1].set_title('y_normfix', pad=30)

surf3 = ax[0,2].scatter(x[0,:,0], x[0,:,1], x[0,:,2], c=y_out2[0,:,0], s=10)
fig.colorbar(surf3 , ax=ax[0,2], shrink=0.5, aspect=10, pad=0.1)
ax[0,2].view_init(elev=-50, azim=50)
ax[0,2].set_title('y_direct', pad=30)



surf21 = ax[1,1].scatter(x[0,:,0], x[0,:,1], x[0,:,2], c = torch.abs(y_out1[0,:,0] - y[0,:,0]), s=10)
fig.colorbar(surf21 , ax=ax[1,1], shrink=0.5, aspect=10, pad=0.1)
ax[1,1].view_init(elev=-50, azim=50)
ax[1,1].set_title('error_normfix', pad=30)

surf31 = ax[1,2].scatter(x[0,:,0], x[0,:,1], x[0,:,2], c = torch.abs(y_out2[0,:,0] - y[0,:,0]), s=10)
fig.colorbar(surf31 , ax=ax[1,2], shrink=0.5, aspect=10, pad=0.1)
ax[1,2].view_init(elev=-50, azim=50)
ax[1,2].set_title('error_direct', pad=30)

plt.show()