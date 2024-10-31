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
import pprint

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

config = config["airfoil"]
config = dict(config)

downsample_ratio = config['downsample_ratio']
L = config['L']
n_train = config['n_train']
n_test = config['n_test']
device = config['device']





data_path = "../data/airfoil/"
coordx = np.load(data_path+"NACA_Cylinder_X.npy")
coordy = np.load(data_path+"NACA_Cylinder_Y.npy")
data_in = np.stack((coordx, coordy), axis=3)
data_out = np.load(data_path+"NACA_Cylinder_Q.npy")[:,4,:,:] #density, velocity 2d, pressure, mach number

_, nx, ny, _ = data_in.shape

data_in_ds = data_in[:, 0::downsample_ratio, 0::downsample_ratio, :]
data_out_ds = data_out[:, 0::downsample_ratio, 0::downsample_ratio, np.newaxis]

L=1.0
Nx = 221
Ny = 51
N = Nx*Ny

grid_train = data_in_ds[0:n_train,:,:,:].reshape(n_train,-1,data_in_ds.shape[-1])
grid_test = data_in_ds[-n_test:,:,:,:].reshape(n_test,-1,data_in_ds.shape[-1])
out_train = data_out_ds[0:n_train,:,:,:].reshape(n_train,-1,data_out_ds.shape[-1])
out_test = data_out_ds[-n_test:,:,:,:].reshape(n_test,-1,data_out_ds.shape[-1])
x_train = torch.from_numpy(
    np.concatenate(
        (out_train, 
         grid_train
        ),
        axis=2,
    ).astype(np.float32)
)

x_test = torch.from_numpy(
    np.concatenate(
        (out_test, 
         grid_test
        ),
        axis=2,
    ).astype(np.float32)
)


print("x_train.shape: ",x_train.shape) 
print("x_test.shape: ",x_test.shape)


learning_rate = config['learning_rate']
batch_size = config['batch_size']
num_epochs = config['num_epochs']
L2_lambda = config['L2_lambda']
K = config['K']


plot_Gaussbases = config['plot_Gaussbases']
plot_x = config['plot_x']
plotGaussbasepts= config['plotGaussbasepts']
save_figure_bases = config['save_figure_bases']
save_figure_x = save_figure_bases
save_figure_basespts = save_figure_bases
num_plot_bases = config['num_plot_bases']
save_tensor = config['save_tensor']

range_pts = [[-1,1],[-1,1]]


if config['basepts_init'] == 'uniform':
    basepts_Gauss = uniform_points(range_pts,225,2)
    baseweight_Gauss = 100*torch.ones(225, 2, dtype = torch.float)
else:
    basepts_Gauss = torch.load(config['basepts_init'])
    distances = torch.cdist(basepts_Gauss, basepts_Gauss)
    mask = ~torch.eye(distances.size(0), dtype=torch.bool)
    min_distances = torch.where(mask, distances, torch.inf)
    min_distance_per_point = torch.min(min_distances, dim=1).values
    D_squared = min_distance_per_point.pow(2)
    baseweight_Gauss = config['phy_dim'] / ( min_distance_per_point.pow(2)).unsqueeze(1).repeat(1,config['phy_dim'])

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
            loss_item = loss.item()

            if L2_lambda:
                para_list = model.get_para_list()
                below_K_mask = para_list[2] < K
                loss_reg = L2_lambda * torch.sum((para_list[2][below_K_mask] - K) ** 2)
                loss = loss + loss_reg

        
            loss.backward()
            optimizer.step()
        t2 = default_timer()
        print(f'Epoch [{epoch}/{num_epochs-1}], time: {t2-t1}, Loss: {loss_item:.4f}')
        scheduler.step()
        if epoch % 10 ==0:
            data = x_test[0].unsqueeze(0).to(device)
            x = data[:,:,0].unsqueeze(-1)
            grid = data[:,:,1:]
            if plot_Gaussbases:
                model.plotGaussbases(num_plot_bases,grid,Nx,Ny,save_figure_bases,epoch)
            if plot_x:
                model.plotx(x,grid,Nx,Ny,save_figure_x,epoch)
            if plotGaussbasepts and epoch<5:
                model.plotGaussbasepts(range_pts,save_figure_basespts,epoch)
if config['save']:
    torch.save([model.base_para1.detach().cpu(), model.base_para2.detach().cpu()],save_tensor)
if config['eval']:
    model.eval()  
    with torch.no_grad():
        test_loss = 0
        n=0
        for data in test_loader:
            data = data.to(device)
            x = data[:,:,0].unsqueeze(-1)
            grid = data[:,:,1:]
            output = model(x,grid)
            test_loss += torch.norm(output)/torch.norm(x)
            n = n+1
        print(f'Test Loss: {test_loss/n:.4f}')


