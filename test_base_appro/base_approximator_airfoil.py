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


sys.path.append("../")

from models.base_approximator import base_approximator

torch.set_printoptions(precision=16)


torch.manual_seed(0)
np.random.seed(0)

def compute_bases_Gauss(basepts_Gauss ,baseweight_Gauss, grid):
    grid = grid.unsqueeze(2) #bsz,n,1,phy_in_channel

    # num_pts = self.num_basepts_Gauss
    # dim = self.phy_dim

    basepts = basepts_Gauss.unsqueeze(0).unsqueeze(0) # 1,1,phy_out_channel,phy_in_channel
    baseweight = torch.abs(baseweight_Gauss).unsqueeze(0).unsqueeze(0)  #1,1,phy_out_channel,phy_in_channel
    sum = torch.sum(baseweight*(grid-basepts)**2,dim=3)
    bases = torch.sqrt(torch.prod(baseweight, dim=3))*torch.exp(-sum)  #bsz,n,phy_out_channel,phy_in_channel-->bsz,n,phy_out_channel
    bases = bases*math.sqrt(bases.shape[1])/torch.norm(bases, p=2, dim=1, keepdim=True)
    ones1 = torch.ones_like(bases)[:,:,0].unsqueeze(-1)  #bsz,n,1
    
    ones2_bool = (grid >= -5) & (grid <= 5)  # (bsz, n, 1, phy_in_channel)
    ones2 = ones2_bool.all(dim=3)  #  (bsz, n, 1)
    ones2 = ones2.float()
    bases = torch.cat((bases,ones1,ones2),dim=-1)
    # bases = bases*math.sqrt(bases.shape[1])/torch.norm(bases, p=2, dim=1, keepdim=True)
    return bases

def compute_bases_Fourier(baseweight_Fourier,a,grid):
    #grid.shape:  bsz,n,phy_in_channel
    grid = grid.unsqueeze(2) #bsz,n,1,phy_in_channel
    baseweight = baseweight_Fourier.unsqueeze(0).unsqueeze(0)  #1,1,phy_out_channel,phy_in_channel
    bases_complex = torch.exp(torch.sum(1j*baseweight*grid,dim=3))  #bsz,n,phy_out_channel,phy_in_channel-->bsz,n,phy_out_channel
    bases = torch.cat((torch.real(bases_complex),torch.imag(bases_complex)),dim=-1)

    is_zero_dim = torch.all(bases == 0, dim=tuple(range(bases.dim() - 1)))
    non_zero_dims = torch.where(~is_zero_dim)[0] 
    filtered_bases = bases.index_select(-1, non_zero_dims)
    return filtered_bases        

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




downsample_ratio = 1
L = 1.0
n_train = 1000
n_test = 200
phy_dim = 2
device = 'cuda'
base_type = 'Gauss'

base_init = 'load'
print(f'base_type = {base_type}')
print(f'base_init = {base_init}')


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


if base_type == 'Gauss':
    if base_init == 'uniform':
        basepts_Gauss = uniform_points([[-1,1],[-1,1]],225,2)
        # basepts_Gauss2 = uniform_points([[-40,40],[-40,40]],4,2)
        baseweight_Gauss = 100*torch.ones(225, 2, dtype = torch.float)
        # baseweight_Gauss2 = 50*torch.ones(4, 2, dtype = torch.float)
        # basepts_Gauss = torch.cat((basepts_Gauss1,basepts_Gauss2),dim=0)
        # baseweight_Gauss = torch.cat((baseweight_Gauss1,baseweight_Gauss2),dim=0)
    elif base_init == 'load':
        basepts_Gauss= torch.load('para/airfoil/basepts_Gauss225.pt')
        distances = torch.cdist(basepts_Gauss, basepts_Gauss)
        mask = ~torch.eye(distances.size(0), dtype=torch.bool)
        min_distances = torch.where(mask, distances, torch.inf)
        min_distance_per_point = torch.min(min_distances, dim=1).values
        D_squared = min_distance_per_point.pow(2)
        baseweight_Gauss = phy_dim / ( min_distance_per_point.pow(2)).unsqueeze(1).repeat(1,phy_dim)
    weight_pos = 2
    model = base_approximator(compute_bases_Gauss , basepts_Gauss, baseweight_Gauss,False,True).to(device)
    print('base train:',False,True)
elif base_type == 'Fourier':
    x = torch.arange(-7, 8) 
    y = torch.arange(-7, 8) 
    X, Y = torch.meshgrid(x, y)
    baseweight_Fourier = 2*np.pi*torch.stack((X.flatten(), Y.flatten()), dim=1)
    a = torch.tensor(0,dtype = torch.float)
    weight_pos = 1
    model = base_approximator(compute_bases_Fourier , baseweight_Fourier, a,True,False).to(device)
    print(True,False)
model.to(device)





# 设置训练参数
learning_rate = 0.1
batch_size = 1
num_epochs = 100
L2_lambda = True
K = 0

print(f'learning_rate = {learning_rate}')
print(f'batch_size = {batch_size}')
print(f'L2_lambda = {L2_lambda} , K = {K}')

plotbases=True
plotx=True
plotGaussbasepts=False
save_figure_bases = 'figure/airfoil/test_Gauss3/'
save_figure_x = save_figure_bases 
save_figure_basespts = save_figure_bases
num_plot_bases = 32
range_pts = [[-1,1],[-1,1]]

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=learning_rate, 
    div_factor=2, final_div_factor=100,pct_start=0.2,
    steps_per_epoch=1, epochs=num_epochs)
train_loader = torch.utils.data.DataLoader(x_train, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(x_test, batch_size=batch_size, shuffle=False)
# 训练循环
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
            below_K_mask = model.base_para2 < K
            loss_reg = L2_lambda * torch.sum((model.base_para2[below_K_mask] - K) ** 2)
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
        if plotbases:
            model.plotbases(num_plot_bases,weight_pos,grid,Nx,Ny,save_figure_bases,epoch,additional_plotbases=227)
        if plotx:
            model.plotx(x,grid,Nx,Ny,save_figure_x,epoch)
        if plotGaussbasepts and epoch<5:
            model.plotGaussbasepts(range_pts,save_figure_basespts,epoch)

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


torch.save([model.base_para1.detach().cpu(), model.base_para2.detach().cpu()], 'para/airfoil/base_Gauss225_bsz1.pt')