import random
import torch
import sys
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from timeit import default_timer
from scipy.io import loadmat
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from baselines.fno import  FNO1d, FNO_train

torch.set_printoptions(precision=16)


torch.manual_seed(0)
np.random.seed(0)



downsample_ratio = 4
n_train = 1000
n_test = 200
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')




data_path = "../../data/burgers/"
data = loadmat(data_path+"burgers_data_R10.mat")
ndata, nnodes_ref = data["a"].shape
grid = np.linspace(0, 1, nnodes_ref)
#downsample
downsample_ratio = 4
features = np.stack((data["a"], data["u"]), axis=2)[:,::downsample_ratio,:]
grid = grid[::downsample_ratio]
nnodes = nnodes_ref//downsample_ratio
    
# x_train, y_train are [n_data, n_x, n_channel] arrays
x_train = torch.from_numpy(
    np.stack((features[0:n_train,:,0], np.tile(grid, (n_train, 1)) ), axis=-1).astype(np.float32)
)
y_train = torch.from_numpy(features[0:n_train, :, [1]].astype(np.float32))

# x_test, y_test are [n_data, n_x, n_channel] arrays
x_test = torch.from_numpy(
    np.stack((features[-n_test:,:,0], np.tile(grid, (n_test, 1))), axis=-1).astype(np.float32)
)
y_test = torch.from_numpy(features[-n_test:, :, [1]].astype(np.float32))

print("x_train.shape: ",x_train.shape)
print("y_train.shape: ",y_train.shape)

###################################
#construct model and train
###################################
k_max = 32
cnn_kernel_size=1
model = FNO1d(modes=[k_max,k_max,k_max,k_max],
               layers=[128,128,128,128,128],
               fc_dim=128,
               in_dim=2, 
               out_dim=1,
               act='gelu',
               pad_ratio=0.05,
               cnn_kernel_size=cnn_kernel_size).to(device)



epochs = 5000
base_lr = 0.001
scheduler = "OneCycleLR"
weight_decay = 1.0e-4
batch_size=8

normalization_x = True
normalization_y = True
normalization_dim_x = []
normalization_dim_y = []
non_normalized_dim_x = 0
non_normalized_dim_y = 0


config = {"train" : {"base_lr": base_lr, "weight_decay": weight_decay, "epochs": epochs, "scheduler": scheduler,  "batch_size": batch_size, 
                     "normalization_x": normalization_x,"normalization_y": normalization_y, 
                     "normalization_dim_x": normalization_dim_x, "normalization_dim_y": normalization_dim_y, 
                     "non_normalized_dim_x": non_normalized_dim_x, "non_normalized_dim_y": non_normalized_dim_y}
                     }


train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = FNO_train(
    x_train, y_train, x_test, y_test, config, model, save_model_name="./FNO_burgers_model"
)





