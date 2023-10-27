#!/usr/bin/env python
# coding: utf-8

# In[54]:


import random
import torch
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.io as sio
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

from timeit import default_timer

sys.path.append('../')
from models import FNN_train, UnitGaussianNormalizer, LpLoss


# # Load fracture problem dataset
# 
# A total of 181 samples are divided into 3 mat files. In this problem, we will be mapping the initial condition to the growth of the crack. 
# The initial condition is phi(:,0) for all the samples. The space is defined using 34 x 34 x 34 points in space. The coordinates of these points are in "coordinates.mat".
# The evolution is mapped using 16 steps which can be considered equally spaced between (0,1]. In each of the dataset.mat, you could take phi(:,0) as the initial condition and then map it to phi(:,1:end). The parameter phi denotes the crack evolution. In case we also want to do the elastic fields, we could map phi(:,0) \rightarraow [u(:,1:end), v(:,1:end), w(:,1:end), phi(:,1:end)].
# 

# In[21]:

FNO_dim = int(sys.argv[1]) #5000


pref = "/bicmr/home/huangdz/data/crack/"
dataset1 = sio.loadmat(pref + "dataset_uniform1.mat")
dataset2 = sio.loadmat(pref + "dataset_uniform2.mat")
dataset3 = sio.loadmat(pref + "dataset_uniform3.mat")

nq = 41
xx = np.linspace(-0.1, 0.1, nq)
yy = np.linspace(-0.1, 0.1, nq)
zz = np.linspace(-0.1, 0.1, nq)
[yq,xq,zq] = np.meshgrid(xx,yy,zz)


# In[4]:


phi = np.vstack((dataset1["phi_uniform"], dataset2["phi_uniform"], dataset3["phi_uniform"]))
ndata = phi.shape[0]
T = 1.0
nt = phi.shape[2]
phi_data = phi.reshape((ndata, nq, nq, nq, nt), order="F")

coord_data  = np.zeros((nq, nq, nq, nt, 4),  dtype="float64")
for i in range(nt):
    coord_data[:,:,:,i,0] = xq
    coord_data[:,:,:,i,1] = yq
    coord_data[:,:,:,i,2] = zq
    coord_data[:,:,:,i,3] = i/nt * T
    

if FNO_dim == 3:
    input_data = np.concatenate((phi_data[:, :, :, :, 0:1], np.tile(coord_data[:, :, :, 0, 0:3], (ndata,1,1,1,1))), axis=-1)
    output_data = phi_data[:, :, :, :, nt-1:nt]
elif FNO_dim == 4:
    input_data = np.concatenate((np.tile(phi_data[:, :, :, :, 0:1], (1,1,1,1,nt-1))[..., np.newaxis], np.tile(coord_data[:, :, :, 1:nt, :], (ndata,1,1,1,1,1))), axis=-1)
    output_data = phi_data[:, :, :, :, 1:nt, np.newaxis]
  


# # FNO training

# In[5]:


torch.manual_seed(0)
np.random.seed(0)


n_train = 160
n_test = 21
x_train = torch.from_numpy(input_data[0:n_train, ...].astype(np.float32))
y_train = torch.from_numpy(output_data[0:n_train, ...].astype(np.float32))
x_test = torch.from_numpy(input_data[-n_test:, ...].astype(np.float32))
y_test = torch.from_numpy(output_data[-n_test:, ...].astype(np.float32))



# In[6]:


n_fno_layers = 3
k_max = 12
d_f = 32
# fourier k_max
modes = [k_max] * n_fno_layers
modes4 = [6] * n_fno_layers
# channel d_f
layers = [d_f] * (n_fno_layers + 1)
fc_dim = d_f
in_dim = 1 + FNO_dim
out_dim = 1
act = "gelu"

base_lr = 0.001
epochs = 500
# scheduler = "CosineAnnealingLR"
weight_decay = 1e-4

scheduler = "MultiStepLR"

pad_ratio = 0.05

milestones = [200, 300, 400, 500, 800, 900]
scheduler_gamma = 0.5
batch_size=2
normalization_x = True
normalization_y = True
normalization_dim = []


config = {"model" : {"modes": modes, "modes4": modes4, "fc_dim": fc_dim, "layers": layers, "in_dim": in_dim, "out_dim":out_dim, "act": act, "pad_ratio":pad_ratio},
          "train" : {"base_lr": base_lr, "weight_decay": weight_decay, "epochs": epochs, "scheduler": scheduler, "milestones": milestones, "scheduler_gamma": scheduler_gamma, "batch_size": batch_size, 
                    "normalization_x": normalization_x,"normalization_y": normalization_y, "normalization_dim": normalization_dim}}




# In[10]:


start = default_timer()
train_rel_l2_losses, test_rel_l2_losses, test_l2_losses, cost = FNN_train(x_train, y_train, x_test, y_test, config, save_model_name="models/FNO_crack_dim"+str(FNO_dim))
end = default_timer()
print("epochs = ", epochs, "elapsed time = ", end - start)


