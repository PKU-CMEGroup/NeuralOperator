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

N = 100
t = np.linspace(0,2*np.pi,N, endpoint=False)
x = torch.from_numpy(np.array([2*np.cos(t), np.sin(t)]).T).unsqueeze(0)  #1,n,2
w = torch.from_numpy(np.linalg.norm(np.array([-np.sin(t), 2*np.cos(t)]).T, axis = -1, keepdims = True)).unsqueeze(0).unsqueeze(-1)   #1,n,1,1

def f(x):
    b,n,c = x.shape
    x, y = x[:,:,0], x[:,:,1]
    fx = torch.exp(torch.sin(x**2 + y**2) + torch.abs(x) + torch.abs(y))
    return fx.reshape(b,n,1)

k_max = 40
L = 8.0
sigma = 0.4

# nx = torch.from_numpy(np.array([np.cos(np.linspace(0,2*np.pi,N, endpoint=False)), np.sin(np.linspace(0,2*np.pi,N, endpoint=False))]).T ).unsqueeze(0) #1,N,2
norm0 = np.array([2*np.cos(t), np.sin(t)]).T  # n,2
nx = torch.from_numpy(norm0/(np.linalg.norm(norm0, axis = -1, keepdims = True)) ).unsqueeze(0) #1,N,2
print(f'sigma = {sigma}, k_max = {k_max}, L = {L}')
y = f(x)
modes = torch.tensor(compute_Fourier_modes(2, [k_max,k_max], [L,L]))  # nmodes, 2, 1
bases_c,  bases_s,  bases_0  = compute_Fourier_bases(x, modes)  # (1, n, nmodes, 1) , (1, n, nmodes, 1), (1, n, 1, 1)
wbases_c,  wbases_s,  wbases_0 = bases_c*w,  bases_s*w,  bases_0*w


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
fig, ax = plt.subplots(2, 3,figsize=(18,14))

surf1 = ax[0,0].scatter(x[0,:,0], x[0,:,1], c=y[0,:,0], s=10)
fig.colorbar(surf1 , ax=ax[0,0], shrink=0.5, aspect=10, pad=0.1)
ax[0,0].set_title('y_ref', pad=30)

surf2 = ax[0,1].scatter(x[0,:,0], x[0,:,1], c=y_out1[0,:,0], s=10)
fig.colorbar(surf2 , ax=ax[0,1], shrink=0.5, aspect=10, pad=0.1)
ax[0,1].set_title('y_normfix', pad=30)

surf3 = ax[0,2].scatter(x[0,:,0], x[0,:,1], c=y_out2[0,:,0], s=10)
fig.colorbar(surf3 , ax=ax[0,2], shrink=0.5, aspect=10, pad=0.1)
ax[0,2].set_title('y_direct', pad=30)



surf21 = ax[1,1].scatter(x[0,:,0], x[0,:,1], c = torch.abs(y_out1[0,:,0] - y[0,:,0]), s=10)
fig.colorbar(surf21 , ax=ax[1,1], shrink=0.5, aspect=10, pad=0.1)
ax[1,1].set_title('error_normfix', pad=30)

surf31 = ax[1,2].scatter(x[0,:,0], x[0,:,1], c = torch.abs(y_out2[0,:,0] - y[0,:,0]), s=10)
fig.colorbar(surf31 , ax=ax[1,2], shrink=0.5, aspect=10, pad=0.1)
ax[1,2].set_title('error_direct', pad=30)

plt.show()