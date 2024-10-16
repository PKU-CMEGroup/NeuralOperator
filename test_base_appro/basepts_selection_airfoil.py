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

class base_selection(nn.Module):
    def __init__(self, basepts , range_pts):
        super(base_selection, self).__init__()  
        self.basepts = nn.Parameter(basepts)
        self.range_pts = torch.tensor(range_pts, dtype=torch.float32, device = device)

    def forward(self,grid):
        
        assert grid.shape[0]==1
        grid = grid.squeeze(0)
        #grid (1, N, phy_dim)
        mins = self.range_pts[:, 0]  # shape will be (phy_dim,)
        maxs = self.range_pts[:, 1]  # shape will be (phy_dim,)

        # 创建 mask 来筛选符合条件的点
        mask = ((grid >= mins) & (grid <= maxs)).all(dim=1)

        # 应用 mask 获取符合条件的点
        grid = grid[mask]  # M,phy_dim
        
        basepts = self.basepts  #(m,phy_dim)
        basepts_expanded = basepts.unsqueeze(1)  # shape will be (m,1, phy_dim)
        grid = grid.unsqueeze(0)  #(1, M, phy_dim)
        # 使用广播来计算所有点对之间的距离
        diff = grid - basepts_expanded  # ( m, M, phy_dim)
        squared_distances = torch.sum(diff ** 2, dim=-1)  # shape will be (m, M)
        min_distances, _ = torch.min(squared_distances, dim=0)  # 取每个样本到 basepts 的最小距离，shape will be (M)

        min_distances_sqrt = torch.sqrt(min_distances)  #(M)

        total_min_distance = torch.sum(min_distances_sqrt,dim=0) #(1)
        return total_min_distance
    
    def plotGaussbasepts(self,range_pts,save_figure_basespts,epoch):
        basepts = self.basepts.detach().cpu().numpy()

        xmin, xmax = range_pts[0][0],range_pts[0][1]
        ymin, ymax = range_pts[1][0],range_pts[1][1]

        in_range_points = basepts[(basepts[:, 0] >= xmin) & (basepts[:, 0] <= xmax) &
                                (basepts[:, 1] >= ymin) & (basepts[:, 1] <= ymax)]
        
        plt.scatter(in_range_points[:, 0], in_range_points[:, 1])
        plt.title(f'Gaussbasepts in [{xmin},{xmax}]*[{ymin},{ymax}]')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_figure_basespts + 'Gaussbasepts_ep' + str(epoch).zfill(3)+ '.png', format='png')
        plt.close()


downsample_ratio = 1
L = 1.0
n_train = 1000
n_test = 200
phy_dim = 2
device = 'cuda'
base_type = 'Gauss'



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

basepts = uniform_points([[-1,1],[-1,1]],225,2)
range_pts = [[-1,1],[-1,1]]
model = base_selection(basepts,range_pts)
model.to(device)



# 设置训练参数
learning_rate = 0.001
batch_size = 1
num_epochs = 50

print(f'learning_rate = {learning_rate}')


# plotbases=True
# plotx=True
plotGaussbasepts=True
# save_figure_bases = 'figure/airfoil/test_Gauss2/'
# save_figure_x = 'figure/airfoil/test_Gauss2/'
save_figure_basespts = 'figure/airfoil_baseselection/test/'
# num_plot_bases = 32
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
    if epoch % 10 ==0:
        if plotGaussbasepts:
            model.plotGaussbasepts(range_pts,save_figure_basespts,epoch)
    model.train() 
    for batch_idx, data in enumerate(train_loader):
        t1 = default_timer()
        data = data.to('cuda')
        grid = data[:,:,1:]
        optimizer.zero_grad()
        output = model(grid)
        loss = torch.norm(output)
        loss_item = loss.item()
        loss.backward()
        optimizer.step()
        t2 = default_timer()
    print(f'Epoch [{epoch}/{num_epochs-1}], time: {t2-t1}, Loss: {loss_item:.4f}')
    scheduler.step()

torch.save(model.basepts.detach().cpu(), 'para/airfoil/basepts_Gauss225.pt')