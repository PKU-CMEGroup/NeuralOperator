import random
import torch
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
from timeit import default_timer
from scipy.io import loadmat

sys.path.append("../")


from baselines.pit import  PiT, PiT_train



torch.set_printoptions(precision=16)


torch.manual_seed(0)
np.random.seed(0)



downsample_ratio = 2
n_train = 1000
n_test = 200
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


###################################
# load data
###################################
data_path = "../data/darcy_2d/piececonst_r421_N1024_smooth1"
data1 = loadmat(data_path)
data_path = "../data/darcy_2d/piececonst_r421_N1024_smooth2"
data2 = loadmat(data_path)
data_in = np.vstack((data1["coeff"], data2["coeff"]))  # shape: 2048,421,421
data_out = np.vstack((data1["sol"], data2["sol"]))     # shape: 2048,421,421
print("data_in.shape:" , data_in.shape)
print("data_out.shape", data_out.shape)

Np_ref = data_in.shape[1]
L=1.0
grid_1d = np.linspace(0, L, Np_ref)
grid_x, grid_y = np.meshgrid(grid_1d, grid_1d)

data_in_ds = data_in[:, 0::downsample_ratio, 0::downsample_ratio]
grid_x_ds = grid_x[0::downsample_ratio, 0::downsample_ratio]
grid_y_ds = grid_y[0::downsample_ratio, 0::downsample_ratio]
data_out_ds = data_out[:, 0::downsample_ratio, 0::downsample_ratio]

# x_train, y_train are [n_data, n_x, n_channel] arrays
x_train = torch.from_numpy(
    np.stack(
        (
            data_in_ds[0:n_train,:,:],
            np.tile(grid_x_ds, (n_train, 1, 1)),
            np.tile(grid_y_ds, (n_train, 1, 1)),
        ),
        axis=-1,
    ).astype(np.float32)
)
y_train = torch.from_numpy(data_out_ds[0:n_train, :, :, np.newaxis].astype(np.float32))
# x_test, y_test are [n_data, n_x, n_channel] arrays
x_test = torch.from_numpy(
    np.stack(
        (
            data_in_ds[-n_test:, :, :],
            np.tile(grid_x[0::downsample_ratio, 0::downsample_ratio], (n_test, 1, 1)),
            np.tile(grid_y[0::downsample_ratio, 0::downsample_ratio], (n_test, 1, 1)),
        ),
        axis=-1,
    ).astype(np.float32)
)
y_test = torch.from_numpy(
    data_out_ds[-n_test:, :, :, np.newaxis].astype(
        np.float32
    )
)

x_train = x_train.reshape(x_train.shape[0], -1, x_train.shape[-1])   # shape: 800,11236,3  (11236 = 106*106 , 106-1 = (421-1) /4)
x_test = x_test.reshape(x_test.shape[0], -1, x_test.shape[-1])
y_train = y_train.reshape(y_train.shape[0], -1, y_train.shape[-1])   # shape: 800,11236,1
y_test = y_test.reshape(y_test.shape[0], -1, y_test.shape[-1])
print("x_train.shape: ",x_train.shape)
print("y_train.shape: ",y_train.shape)


n_epochs     = 500
lr           = 0.001
batch_size   = 8
hid_channels   = 128
in_channels    = 3
out_channels   = 1
n_head       = 2
qry_res      = int((421-1)/downsample_ratio+1)
ltt_res      = 32
localities = [2, 200,200,200,200, 5]
# localities = [200, 200,200,200,200, 200]
### define a model

def pairwise_dist(res1x, res1y, res2x, res2y):
    gridx1 = torch.linspace(0, 1, res1x+1)[:-1].view(1, -1, 1).repeat(res1y, 1, 1)
    gridy1 = torch.linspace(0, 1, res1y+1)[:-1].view(-1, 1, 1).repeat(1, res1x, 1)
    grid1 = torch.cat([gridx1, gridy1], dim=-1).view(res1x*res1y, 2)
    
    gridx2 = torch.linspace(0, 1, res2x+1)[:-1].view(1, -1, 1).repeat(res2y, 1, 1)
    gridy2 = torch.linspace(0, 1, res2y+1)[:-1].view(-1, 1, 1).repeat(1, res2x, 1)
    grid2 = torch.cat([gridx2, gridy2], dim=-1).view(res2x*res2y, 2)
    
    grid1 = grid1.unsqueeze(1).repeat(1, grid2.shape[0], 1)
    grid2 = grid2.unsqueeze(0).repeat(grid1.shape[0], 1, 1)
    
    dist = torch.norm(grid1 - grid2, dim=-1)
    return (dist**2 / 2.0).float()


m_cross   = pairwise_dist(qry_res, qry_res, ltt_res, ltt_res).to(device) # pairwise distance matrix for encoder and decoder
m_latent  = pairwise_dist(ltt_res, ltt_res, ltt_res, ltt_res).to(device) # pairwise distance matrix for processor
m_dists = [m_cross.T, m_latent, m_latent, m_latent, m_latent, m_cross]
model = PiT(in_channels, out_channels, hid_channels, n_head, localities, m_dists).to(device)


epochs = 500
base_lr = 0.001
scheduler = "OneCycleLR"
weight_decay = 1.0e-4
batch_size=8

normalization_x = True
normalization_y = True
normalization_dim = []

config = {"train" : {"base_lr": base_lr, "weight_decay": weight_decay, "epochs": epochs, "scheduler": scheduler,  "batch_size": batch_size, 
                     "normalization_x": normalization_x,"normalization_y": normalization_y, "normalization_dim": normalization_dim}}

train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = PiT_train(
    x_train, y_train, x_test, y_test, config, model, save_model_name="./Pit_darcy_model"
)





import random
import torch
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
from timeit import default_timer
from scipy.io import loadmat
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_dir)
print("dir now:", script_dir)
sys.path.append("../")


from baselines.pit import  PiT, PiT_train



torch.set_printoptions(precision=16)


torch.manual_seed(0)
np.random.seed(0)



downsample_ratio = 2
n_train = 1000
n_test = 200
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


###################################
# load data
###################################
data_path = "../data/darcy_2d/piececonst_r421_N1024_smooth1"
data1 = loadmat(data_path)
data_path = "../data/darcy_2d/piececonst_r421_N1024_smooth2"
data2 = loadmat(data_path)
data_in = np.vstack((data1["coeff"], data2["coeff"]))  # shape: 2048,421,421
data_out = np.vstack((data1["sol"], data2["sol"]))     # shape: 2048,421,421
print("data_in.shape:" , data_in.shape)
print("data_out.shape", data_out.shape)

Np_ref = data_in.shape[1]
L=1.0
grid_1d = np.linspace(0, L, Np_ref)
grid_x, grid_y = np.meshgrid(grid_1d, grid_1d)

data_in_ds = data_in[:, 0::downsample_ratio, 0::downsample_ratio]
grid_x_ds = grid_x[0::downsample_ratio, 0::downsample_ratio]
grid_y_ds = grid_y[0::downsample_ratio, 0::downsample_ratio]
data_out_ds = data_out[:, 0::downsample_ratio, 0::downsample_ratio]

# x_train, y_train are [n_data, n_x, n_channel] arrays
x_train = torch.from_numpy(
    np.stack(
        (
            data_in_ds[0:n_train,:,:],
            np.tile(grid_x_ds, (n_train, 1, 1)),
            np.tile(grid_y_ds, (n_train, 1, 1)),
        ),
        axis=-1,
    ).astype(np.float32)
)
y_train = torch.from_numpy(data_out_ds[0:n_train, :, :, np.newaxis].astype(np.float32))
# x_test, y_test are [n_data, n_x, n_channel] arrays
x_test = torch.from_numpy(
    np.stack(
        (
            data_in_ds[-n_test:, :, :],
            np.tile(grid_x[0::downsample_ratio, 0::downsample_ratio], (n_test, 1, 1)),
            np.tile(grid_y[0::downsample_ratio, 0::downsample_ratio], (n_test, 1, 1)),
        ),
        axis=-1,
    ).astype(np.float32)
)
y_test = torch.from_numpy(
    data_out_ds[-n_test:, :, :, np.newaxis].astype(
        np.float32
    )
)

x_train = x_train.reshape(x_train.shape[0], -1, x_train.shape[-1])   # shape: 800,11236,3  (11236 = 106*106 , 106-1 = (421-1) /4)
x_test = x_test.reshape(x_test.shape[0], -1, x_test.shape[-1])
y_train = y_train.reshape(y_train.shape[0], -1, y_train.shape[-1])   # shape: 800,11236,1
y_test = y_test.reshape(y_test.shape[0], -1, y_test.shape[-1])
print("x_train.shape: ",x_train.shape)
print("y_train.shape: ",y_train.shape)


n_epochs     = 500
lr           = 0.001
batch_size   = 8
hid_channels   = 128
in_channels    = 3
out_channels   = 1
n_head       = 2
qry_res      = int((421-1)/downsample_ratio+1)
ltt_res      = 32
localities = [2, 200,200,200,200, 5]
# localities = [200, 200,200,200,200, 200]
### define a model

def pairwise_dist(res1x, res1y, res2x, res2y):
    gridx1 = torch.linspace(0, 1, res1x+1)[:-1].view(1, -1, 1).repeat(res1y, 1, 1)
    gridy1 = torch.linspace(0, 1, res1y+1)[:-1].view(-1, 1, 1).repeat(1, res1x, 1)
    grid1 = torch.cat([gridx1, gridy1], dim=-1).view(res1x*res1y, 2)
    
    gridx2 = torch.linspace(0, 1, res2x+1)[:-1].view(1, -1, 1).repeat(res2y, 1, 1)
    gridy2 = torch.linspace(0, 1, res2y+1)[:-1].view(-1, 1, 1).repeat(1, res2x, 1)
    grid2 = torch.cat([gridx2, gridy2], dim=-1).view(res2x*res2y, 2)
    
    grid1 = grid1.unsqueeze(1).repeat(1, grid2.shape[0], 1)
    grid2 = grid2.unsqueeze(0).repeat(grid1.shape[0], 1, 1)
    
    dist = torch.norm(grid1 - grid2, dim=-1)
    return (dist**2 / 2.0).float()


m_cross   = pairwise_dist(qry_res, qry_res, ltt_res, ltt_res).to(device) # pairwise distance matrix for encoder and decoder
m_latent  = pairwise_dist(ltt_res, ltt_res, ltt_res, ltt_res).to(device) # pairwise distance matrix for processor
m_dists = [m_cross.T, m_latent, m_latent, m_latent, m_latent, m_cross]
model = PiT(in_channels, out_channels, hid_channels, n_head, localities, m_dists).to(device)


epochs = 500
base_lr = 0.001
scheduler = "OneCycleLR"
weight_decay = 1.0e-4
batch_size=8

normalization_x = True
normalization_y = True
normalization_dim = []

config = {"train" : {"base_lr": base_lr, "weight_decay": weight_decay, "epochs": epochs, "scheduler": scheduler,  "batch_size": batch_size, 
                     "normalization_x": normalization_x,"normalization_y": normalization_y, "normalization_dim": normalization_dim}}

train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = PiT_train(
    x_train, y_train, x_test, y_test, config, model, save_model_name="./Pit_darcy_model"
)





