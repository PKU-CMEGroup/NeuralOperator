import random
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
from timeit import default_timer
from scipy.io import loadmat
import yaml
import gc
from pprint import pprint

sys.path.append("../")

from models.base_approximator2 import base_approximator2

torch.set_printoptions(precision=16)


torch.manual_seed(0)
np.random.seed(0)


def uniform_points(range_pts,num_pts,dim):
    a = round(math.pow(num_pts, 1 / dim))
    index_tensors = []
    for k in range(dim):
        xmin,xmax = range_pts[k][0],range_pts[k][1]
        idx = xmin + (xmax-xmin)*torch.arange(a).float().add(0.5).div(a)
        idx = idx.view((1,) * k+ (-1,) + (1,) * (dim - k - 1))
        index_tensors.append(idx.expand(a, *([a] * (dim - 1))))
    num_pts1 = int(torch.pow(torch.tensor(a),dim))
    x = torch.stack(index_tensors, dim=dim).reshape(num_pts1,dim)
    return x

def compute_bases_Gauss(basepts_Gauss ,baseweight_Gauss, grid):
    grid = grid.unsqueeze(2) #bsz,n,1,phy_in_channel

    basepts = basepts_Gauss.unsqueeze(0).unsqueeze(0) # 1,1,phy_out_channel,phy_in_channel
    baseweight = torch.abs(baseweight_Gauss).unsqueeze(0).unsqueeze(0)  #1,1,phy_out_channel,phy_in_channel
    sum = torch.sum(baseweight*(grid-basepts)**2,dim=3)
    bases = torch.sqrt(torch.prod(baseweight, dim=3))*torch.exp(-sum)  #bsz,n,phy_out_channel,phy_in_channel-->bsz,n,phy_out_channel
    bases = bases*math.sqrt(bases.shape[1])/torch.norm(bases, p=2, dim=1, keepdim=True)
    return bases

def compute_bases_Fourier(baseweight_Fourier,grid):
    #grid.shape:  bsz,n,phy_in_channel
    grid = grid.unsqueeze(2) #bsz,n,1,phy_in_channel
    baseweight = baseweight_Fourier.unsqueeze(0).unsqueeze(0)  #1,1,phy_out_channel,phy_in_channel
    bases_complex = torch.exp(torch.sum(1j*baseweight*grid,dim=3))  #bsz,n,phy_out_channel,phy_in_channel-->bsz,n,phy_out_channel
    bases = torch.cat((torch.real(bases_complex),torch.imag(bases_complex)),dim=-1)
    return bases        


def compute_mixed_bases(para_if_train,grid):
    baseweight_Fourier, basepts_Gauss ,baseweight_Gauss = para_if_train
    bases_Fourier = compute_bases_Fourier(baseweight_Fourier , grid)
    is_zero_dim = torch.all(bases_Fourier == 0, dim=tuple(range(bases_Fourier.dim() - 1)))
    non_zero_dims = torch.where(~is_zero_dim)[0] 
    bases_Fourier = bases_Fourier.index_select(-1, non_zero_dims)
    bases_Gauss = compute_bases_Gauss(basepts_Gauss, baseweight_Gauss, grid)

    return bases_Fourier,bases_Gauss

with open('config.yml', 'r', encoding='utf-8') as f:
    config = yaml.full_load(f)

config = config["Darcy"]
config = dict(config)

downsample_ratio = config['downsample_ratio']
L = config['L']
n_train = config['n_train']
n_test = config['n_test']
device = config['device']


data_path = "../data/darcy_2d/piececonst_r421_N1024_smooth1"
data1 = loadmat(data_path)

data_out = data1["sol"]

print("data_out.shape", data_out.shape)
x_train = torch.from_numpy(data_out[0:n_train, 0::downsample_ratio, 0::downsample_ratio])
x_test = torch.from_numpy(data_out[-n_test:, 0::downsample_ratio, 0::downsample_ratio])

Np_ref = data_out.shape[1]
grid_1d = np.linspace(0, L, Np_ref)
grid_x, grid_y = np.meshgrid(grid_1d, grid_1d)

grid_x_ds = grid_x[0::downsample_ratio, 0::downsample_ratio]
grid_y_ds = grid_y[0::downsample_ratio, 0::downsample_ratio]
x_train = torch.from_numpy(
    np.stack(
        (
            x_train,
            np.tile(grid_x_ds, (n_train, 1, 1)),
            np.tile(grid_y_ds, (n_train, 1, 1)),
        ),
        axis=-1,
    ).astype(np.float32)
)
x_test = torch.from_numpy(
    np.stack(
        (
            x_test,
            np.tile(grid_x_ds, (n_test, 1, 1)),
            np.tile(grid_y_ds, (n_test, 1, 1)),
        ),
        axis=-1,
    ).astype(np.float32)
)
x_train = x_train.reshape(x_train.shape[0], -1, x_train.shape[-1])
x_test = x_test.reshape(x_test.shape[0], -1, x_test.shape[-1])


print("x_train.shape: ",x_train.shape)
print("x_test.shape: ",x_test.shape)



learning_rate = config['learning_rate']
batch_size = config['batch_size']
num_epochs = config['num_epochs']

Nx = 420//downsample_ratio+1
Ny = 420//downsample_ratio+1
plot_Gaussbases = config['plot_Gaussbases']
plot_x = config['plot_x']
plotGaussbasepts= config['plotGaussbasepts']
save_figure_bases = config['save_figure_bases']
save_figure_x = save_figure_bases
num_plot_bases = config['num_plot_bases']
save_tensor = config['save_tensor']



if config['basepts_init'] == 'uniform':
    basepts_Gauss = uniform_points([[0,1],[0,1]],config['num_Gauss'],config['phy_dim'])
baseweight_Gauss = 100*torch.ones(config['num_Gauss'],config['phy_dim'], dtype = torch.float)

x = torch.arange(-7, 8) 
y = torch.arange(-7, 8) 
X, Y = torch.meshgrid(x, y)
baseweight_Fourier = 2*np.pi*torch.stack((X.flatten(), Y.flatten()), dim=1)

para_list = [baseweight_Fourier, basepts_Gauss , baseweight_Gauss]
para_if_train = config['para_if_train']
model = base_approximator2(compute_mixed_bases , para_list, para_if_train).to(device)



model.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=learning_rate, 
    div_factor=2, final_div_factor=100,pct_start=0.2,
    steps_per_epoch=1, epochs=num_epochs)
train_loader = torch.utils.data.DataLoader(x_train, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(x_test, batch_size=batch_size, shuffle=False)


pprint(config)
if config['train']:
    print('start training')
    for epoch in range(num_epochs):
        model.train() 
        t1 = default_timer()
        for batch_idx, data in enumerate(train_loader):

            data = data.to('cuda')
            x = data[:,:,0].unsqueeze(-1)
            grid = data[:,:,1:]
            optimizer.zero_grad()
            output = model(x,grid)
            loss = torch.norm(output)/torch.norm(x)
            loss.backward()
            optimizer.step()
            
        t2 = default_timer()
        print(f'Epoch [{epoch}/{num_epochs-1}], time: {t2-t1}, Loss: {loss.item():.4f}',flush=True)
        scheduler.step()
        if epoch % 10 ==0:
            data = x_test[0].unsqueeze(0).to(device)
            x = data[:,:,0].unsqueeze(-1)
            grid = data[:,:,1:]
            if plot_Gaussbases:
                model.plotGaussbases(num_plot_bases,grid,Nx,Ny,save_figure_bases,epoch)
            if plot_x:
                model.plotx(x,grid,Nx,Ny,save_figure_x,epoch)
    if config['save']:
        torch.save([model.base_para1.detach().cpu(), model.base_para2.detach().cpu()], save_tensor)

if config['eval']:
    tensor1,tensor2 = torch.load(config['eval_tensor']) 
    modeltest = base_approximator2(compute_bases_Gauss , tensor1, tensor2,False,False).to(device)
    with torch.no_grad():
        test_loss = 0
        n=0
        for data in test_loader:
            data = data.to(device)
            x = data[:,:,0].unsqueeze(-1)
            grid = data[:,:,1:]
            output = modeltest(x,grid)
            test_loss += torch.norm(output)/torch.norm(x)
            n = n+1
        print(f'Test Loss: {test_loss/n:.4f}')


