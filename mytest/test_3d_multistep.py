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
from pcno.pcno_normfix_res import truncate, compute_Fourier_modes, compute_Fourier_bases, direct


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



k_max = 4
L = [2.0,2.0,5.0]
# L = [4.0,4.0,10.0]
sigma = 0.2


print(f'sigma = {sigma}, k_max = {k_max}, L = {L}')



modes = torch.tensor(compute_Fourier_modes(3, [k_max,k_max,k_max], L)).to(device)  # nmodes, 2, 1
bases_c,  bases_s,  bases_0  = compute_Fourier_bases(x, modes)  # (1, n, nmodes, 1) , (1, n, nmodes, 1), (1, n, 1, 1)
wbases_c = torch.einsum("bxkw,bxw->bxkw", bases_c, node_weights)
wbases_s = torch.einsum("bxkw,bxw->bxkw", bases_s, node_weights)
wbases_0 = torch.einsum("bxkw,bxw->bxkw", bases_0, node_weights)


def normfix_onestep(x, y, nx, sigma, k, L):

    num_modes = ((2*k+1)**3- 1 )//2
    # t = default_timer()
    y_normfix = truncate(nx, y.transpose(2,1), wbases_c[:,:,:num_modes,:], wbases_s[:,:,:num_modes,:], wbases_0, 
                         bases_c[:,:,:num_modes,:], bases_s[:,:,:num_modes,:], bases_0, modes[:num_modes], sigma).transpose(2,1)
    # s = default_timer()
    # print(f'time for k_max = {k_max}: ',s-t)

    # coffe = torch.dot(y[0,:,0], y_normfix[0,:,0])/torch.norm(y_normfix[0,:,0])**2
    coffe = torch.norm(y[0,:,0])/torch.norm(y_normfix[0,:,0])
    y_normfix = y_normfix * coffe
    
    return y_normfix

def normfix_multistep(x, y, nx, sigma_list, k_max_list, L):
    for i,(sigma,k) in enumerate(zip(sigma_list, k_max_list)):
        if i==0:
            y_normfix = normfix_onestep(x, y, nx, sigma, k, L)
        else:
            y_normfix_new = normfix_onestep(x, y - y_normfix, nx, sigma, k, L)
            y_normfix = y_normfix + y_normfix_new
    return y_normfix

sigma_list = [0.8,0.6,0.4,0.2]
k_max_list = [4,4,4,4]

y_normfix_multi = normfix_multistep(x, y, nx, sigma_list, k_max_list, L)
# t1 = default_timer()
y_normfix_single = normfix_onestep(x, y, nx, sigma, k_max, L)
# t2 = default_timer()

y_normfix_multi = normfix_multistep(x, y, nx, sigma_list, k_max_list, L)
# t3 = default_timer()

y_direct = direct(y.transpose(2,1), wbases_c, wbases_s, wbases_0, bases_c, bases_s, bases_0).transpose(2,1)

# t4 = default_timer()


# argmin_c ||y - c*y_out|| = <y, y_out>/||y_out||^2
# print(y.shape,y_out.shape)
# coffe1 = torch.dot(y[0,:,0], y_out1[0,:,0])/torch.norm(y_out1[0,:,0])**2
# y_out1 = y_out1 * coffe1

coffe = torch.dot(y[0,:,0], y_direct[0,:,0])/torch.norm(y_direct[0,:,0])**2
y_direct = y_direct * coffe

print('loss(normfix multi): ', (torch.norm(y - y_normfix_multi)/torch.norm(y)).item())
print('loss(normfix single): ', (torch.norm(y - y_normfix_single)/torch.norm(y)).item())
print('loss(direct): ', (torch.norm(y - y_direct)/torch.norm(y)).item())
# print('\n')
# print('time(normfix multi): ',t3-t2)
# print('time(direct): ',t4-t3)
# print('time(normfix single): ',t2-t1)

x, y, y_normfix_single, y_normfix_multi, y_direct = x.to('cpu'), y.to('cpu'), y_normfix_single.to('cpu'), y_normfix_multi.to('cpu') ,y_direct.to('cpu')
fig, ax = plt.subplots(2, 4,figsize=(20,14), subplot_kw={'projection': '3d'})

surf1 = ax[0,0].scatter(x[0,:,0], x[0,:,1], x[0,:,2], c=y[0,:,0], s=10)
fig.colorbar(surf1 , ax=ax[0,0], shrink=0.5, aspect=10, pad=0.1)
ax[0,0].view_init(elev=-50, azim=50)
ax[0,0].set_title('y_ref', pad=30)

surf2 = ax[0,1].scatter(x[0,:,0], x[0,:,1], x[0,:,2], c=y_normfix_single[0,:,0], s=10)
fig.colorbar(surf2 , ax=ax[0,1], shrink=0.5, aspect=10, pad=0.1)
ax[0,1].view_init(elev=-50, azim=50)
ax[0,1].set_title('y_normfix_single', pad=30)

surf3 = ax[0,2].scatter(x[0,:,0], x[0,:,1], x[0,:,2], c=y_normfix_multi[0,:,0], s=10)
fig.colorbar(surf3 , ax=ax[0,2], shrink=0.5, aspect=10, pad=0.1)
ax[0,2].view_init(elev=-50, azim=50)
ax[0,2].set_title('y_normfix_multi', pad=30)

surf4 = ax[0,3].scatter(x[0,:,0], x[0,:,1], x[0,:,2], c=y_direct[0,:,0], s=10)
fig.colorbar(surf4 , ax=ax[0,3], shrink=0.5, aspect=10, pad=0.1)
ax[0,3].view_init(elev=-50, azim=50)
ax[0,3].set_title('y_direct', pad=30)



surf21 = ax[1,1].scatter(x[0,:,0], x[0,:,1], x[0,:,2], c = torch.abs(y_normfix_single[0,:,0] - y[0,:,0]), s=10)
fig.colorbar(surf21 , ax=ax[1,1], shrink=0.5, aspect=10, pad=0.1)
ax[1,1].view_init(elev=-50, azim=50)
ax[1,1].set_title('error_normfix_single', pad=30)

surf31 = ax[1,2].scatter(x[0,:,0], x[0,:,1], x[0,:,2], c = torch.abs(y_normfix_multi[0,:,0] - y[0,:,0]), s=10)
fig.colorbar(surf31 , ax=ax[1,2], shrink=0.5, aspect=10, pad=0.1)
ax[1,2].view_init(elev=-50, azim=50)
ax[1,2].set_title('error_normfix_multi', pad=30)

surf41 = ax[1,3].scatter(x[0,:,0], x[0,:,1], x[0,:,2], c = torch.abs(y_direct[0,:,0] - y[0,:,0]), s=10)
fig.colorbar(surf41 , ax=ax[1,3], shrink=0.5, aspect=10, pad=0.1)
ax[1,3].view_init(elev=-50, azim=50)
ax[1,3].set_title('error_direct', pad=30)
plt.show()