import random
import torch
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
from timeit import default_timer
from scipy.io import loadmat
import yaml

sys.path.append("../")


from models import  PhyHGkNN_train, compute_2dFourier_bases, compute_2dpca_bases, compute_2dFourier_cbases, count_params

from models.PhyHGkNN2 import PhyHGkNN2

torch.set_printoptions(precision=16)


torch.manual_seed(0)
np.random.seed(0)


###################################
# load configs
###################################
with open('config.yml', 'r', encoding='utf-8') as f:
    config = yaml.full_load(f)

config = config["airfoil"]
config = dict(config)
config_data, config_model, config_train = (
    config["data"],
    config["model"],
    config["train"],
)
downsample_ratio = config_data["downsample_ratio"]
L = config_data["L"]
n_train = config_data["n_train"]
n_test = config_data["n_test"]
device = torch.device(config["train"]["device"])


###################################
# load data
###################################
data_path = "../data/airfoil/"
coordx = np.load(data_path+"NACA_Cylinder_X.npy")
coordy = np.load(data_path+"NACA_Cylinder_Y.npy")
data_in = np.stack((coordx, coordy), axis=3)
data_out = np.load(data_path+"NACA_Cylinder_Q.npy")[:,4,:,:] #density, velocity 2d, pressure, mach number

_, nx, ny, _ = data_in.shape

data_in_ds = data_in[:, 0::downsample_ratio, 0::downsample_ratio, :]
data_out_ds = data_out[:, 0::downsample_ratio, 0::downsample_ratio, np.newaxis]

L=1.0
grid_x, grid_y = np.meshgrid(np.linspace(0, L, nx), np.linspace(0, L, ny))
grid_x, grid_y = grid_x.T, grid_y.T
grid_x_ds = grid_x[0::downsample_ratio, 0::downsample_ratio]
grid_y_ds = grid_y[0::downsample_ratio, 0::downsample_ratio]
# x_train, y_train are [n_data, n_x, n_channel] arrays

print(data_in_ds[0:n_train,:,:,:].shape, np.tile(grid_x_ds, (n_train, 1, 1))[:,:,:, np.newaxis].shape)
# x_train = torch.from_numpy(data_in_ds[0:n_train, :, :, :].astype(np.float32))
x_train = torch.from_numpy(
    np.concatenate(
        (data_in_ds[0:n_train,:,:,:], 
         np.tile(grid_x_ds, (n_train, 1, 1))[:,:,:, np.newaxis],
         np.tile(grid_y_ds, (n_train, 1, 1))[:,:,:, np.newaxis],
        ),
        axis=3,
    ).astype(np.float32)
)

y_train = torch.from_numpy(data_out_ds[0:n_train, :, :, :].astype(np.float32))
# x_test, y_test are [n_data, n_x, n_channel] arrays
# x_test = torch.from_numpy(data_in_ds[-n_test:, :, :, :].astype(np.float32))
x_test = torch.from_numpy(
    np.concatenate(
        (data_in_ds[-n_test:,:,:,:], 
         np.tile(grid_x_ds, (n_test, 1, 1))[:,:,:, np.newaxis],
         np.tile(grid_y_ds, (n_test, 1, 1))[:,:,:, np.newaxis],
        ),
        axis=3,
    ).astype(np.float32)
)

y_test = torch.from_numpy(data_out_ds[-n_test:, :, :, :].astype(np.float32))



x_train = x_train.reshape(x_train.shape[0], -1, x_train.shape[-1])   
x_test = x_test.reshape(x_test.shape[0], -1, x_test.shape[-1])
y_train = y_train.reshape(y_train.shape[0], -1, y_train.shape[-1])   
y_test = y_test.reshape(y_test.shape[0], -1, y_test.shape[-1])
new_order = [2,3,0,1]
x_train = x_train[:, :, new_order]
x_test = x_test[:, :, new_order]
print("x_train.shape: ",x_train.shape)   #torch.Size([1000, 11271, 4])
print("y_train.shape: ",y_train.shape)   #torch.Size([1000, 11271, 1])


####################################
#compute pca bases
####################################
k_max = max(config_model['layer']["GkNN_mode_in"],config_model['layer']["GkNN_mode_out"],config_model['layer']["GkNN_mode_out_phybases"])
Nx = (nx + downsample_ratio - 1) // downsample_ratio
Ny = (ny + downsample_ratio - 1) // downsample_ratio

# pca_data_in_1 = data_in_ds[:,:,:,0].reshape((data_in_ds.shape[0], -1 ))
pca_data_in_2 = data_in_ds[:,:,:,1].reshape((data_in_ds.shape[0], -1 ))
pca_data_out = data_out_ds[0:n_train, :, :, :].reshape((n_train, -1))

from models import UnitGaussianNormalizer
# x_normalizer = UnitGaussianNormalizer(x_train, dim=[])
y_normalizer = UnitGaussianNormalizer(y_train, dim=[])
# pca_data_in_1 = x_normalizer.encode(torch.from_numpy(pca_data_in_1)).numpy()
# pca_data_in_2 = x_normalizer.encode(torch.from_numpy(pca_data_in_2)).numpy()
pca_data_out = y_normalizer.encode(torch.from_numpy(pca_data_out)).numpy()

print("Start SVD with data shape: ", pca_data_out.shape,flush=True)
# U, S, VT = np.linalg.svd(pca_data_in_1.T, full_matrices=False)
# fbases = U[:, 0:k_max] / np.sqrt(L * L / (Nx*Ny))
# wfbases = L * L / (Nx*Ny) * fbases
# bases_pca_in_1 = torch.from_numpy(fbases.astype(np.float32))
# wbases_pca_in_1 = torch.from_numpy(wfbases.astype(np.float32))
# bases_pca_in_1, wbases_pca_in_1 = bases_pca_in_1.to(device), wbases_pca_in_1.to(device)

U, S, VT = np.linalg.svd(pca_data_in_2.T, full_matrices=False)
fbases = U[:, 0:k_max] / np.sqrt(L * L / (Nx*Ny))
wfbases = L * L / (Nx*Ny) * fbases
bases_pca_in_2 = torch.from_numpy(fbases.astype(np.float32))
wbases_pca_in_2 = torch.from_numpy(wfbases.astype(np.float32))
bases_pca_in_2, wbases_pca_in_2 = bases_pca_in_2.to(device), wbases_pca_in_2.to(device)



U, S, VT = np.linalg.svd(pca_data_out.T, full_matrices=False)
fbases = U[:, 0:k_max] / np.sqrt(L * L / (Nx*Ny))
wfbases = L * L / (Nx*Ny) * fbases
bases_pca = torch.from_numpy(fbases.astype(np.float32))
wbases_pca = torch.from_numpy(wfbases.astype(np.float32))
bases_pca_out, wbases_pca_out = bases_pca.to(device), wbases_pca.to(device)




bases_list = [ bases_pca_out, wbases_pca_in_2,bases_pca_out,0]
print('bases_list = [ bases_pca_out, wbases_pca_in_2,bases_pca_out,0]',flush=True)
# bases_list = [ bases_pca_out, 0,0,0]
# print('bases:','bases_pca_out','0',flush=True)
###################################
#construct model and train
###################################
model = PhyHGkNN2(bases_list, **config_model).to(device)
print(count_params(model),flush=True)


print("Start training ", "layer_type: ",config_model,config_train,flush=True)
train_rel_l2_losses, test_rel_l2_losses = PhyHGkNN_train(
    x_train, y_train, x_test, y_test, config, model, save_model_name=False
)