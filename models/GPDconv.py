from timeit import default_timer
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch_scatter import scatter

def mycompl_mul1d_D(weights, D , x_hat):
    x_hat_expanded = x_hat.unsqueeze(2)  # shape: (bsz, input channel, 1, modes)
    D_expanded = D.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, kernel_modes, modes)
    x_hat1 = x_hat_expanded * D_expanded  # shape: (bsz, input channel, kernel_modes, modes)
    y = torch.einsum('bijk,ioj -> bok', x_hat1 , weights )
    return y


class GPDconv(nn.Module):
    def __init__(self, in_channels, out_channels, modes, kernel_modes, D, basepts, base_weight):
        super(GPDconv, self).__init__()


        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes = modes

        self.D = D
        self.dtype = D.dtype
        self.basepts = basepts
        self.base_weight = base_weight

        # hidden_channel = 8
        # self.scale = 1 / (hidden_channel * hidden_channel)
        # self.kernel_modes = kernel_modes

        # self.weights = nn.Parameter(
        #     self.scale
        #     * torch.randn(hidden_channel, hidden_channel, self.kernel_modes, dtype=self.dtype)
        # )
        # self.fc1 = nn.Linear(in_channels,hidden_channel)
        # self.fc2 = nn.Linear(hidden_channel,out_channels)

        self.scale = 1 / (in_channels * out_channels)
        self.kernel_modes = kernel_modes

        self.weights = nn.Parameter(
            self.scale
            * torch.randn(in_channels, out_channels, self.kernel_modes, dtype=self.dtype)
        )

                
    def forward(self, x, grid , grid_weight , edge_grid , edge_Gauss):
        '''
        x:                          (bsz, in_channel, N)

        grid:                       (bsz, N, phy_dim)

        grid_weight:                (bsz, N)

        edge_grid, edge_Gauss:      (bsz, num_pts, k)

        out:                        (bsz, in_channel, num_pts)
        '''
        # Compute coeffcients
        # print(0,torch.norm(x).item())
        x = x.permute(0,2,1)
        # x = self.fc1(x)
        x_hat = value2features(x,grid,grid_weight,self.basepts,self.base_weight,edge_grid,edge_Gauss)

        x_hat = x_hat.permute(0,2,1)
        # Multiply relevant Fourier modes
        # print(1,torch.norm(x_hat).item())
        x_hat = mycompl_mul1d_D(self.weights, self.D , x_hat)
        # print(2,torch.norm(x_hat).item())
        x_hat = x_hat.permute(0,2,1)
        # Return to physical space
        x = features2value(x_hat,grid,self.basepts,self.base_weight,edge_grid,edge_Gauss)

        # x = self.fc2(x)
        x = x.permute(0,2,1)
        # print(3,torch.norm(x).item())
        return x
    

def value2features(x,grid,grid_weight,basepts,baseweight,edge_grid,edge_Gauss):
    '''
    x:                          (bsz, N, in_channel)

    grid:                       (bsz, N, phy_dim)

    grid_weight:                (bsz, N)

    basepts:                    (num_pts, phy_dim)

    baseweight:                 (num_pts, phy_dim)

    edge_grid, edge_Gauss:          (bsz, num_pts, k)

    out:                        (bsz, num_pts, in_channel)
    '''

    bsz,N,phy_dim = grid.shape
    num_pts = basepts.shape[0]
    in_channels = x.shape[-1]
    k = edge_grid.shape[-1]


    grid = grid.reshape(-1,phy_dim)   # bsz*N, phy_dim
    grid_weight = grid_weight.reshape(bsz*N)  # bsz*N

    #### shift value of edge_grid from {0,1,...,N-1} to {0,1,...,bsz*N-1}
    edge_grid = edge_grid.permute(0,2,1) + torch.arange(bsz,device = x.device).unsqueeze(-1).unsqueeze(-1)*N  # bsz, k, num_pts
    edge_grid = edge_grid.reshape(bsz*k*num_pts)  # bsz*k*num_pts

    ### compute Gauss on each dege
    edge_Gauss = edge_Gauss.permute(0,2,1).reshape(-1)
    dist_square = (grid[edge_grid] - basepts[edge_Gauss])**2   # bsz*k*num_pts, phy_dim

    dist_square = dist_square.reshape(bsz*k,num_pts,phy_dim)  # bsz*k, num_pts, phy_dim
    dist_weighted = torch.sum(baseweight.unsqueeze(0)*dist_square,dim = -1)     # bsz*k, num_pts
    dist_weighted = dist_weighted.reshape(bsz*k*num_pts).unsqueeze(-1).repeat(1,in_channels)  # bsz*k*num_pts, in_channels
    baseweight = baseweight.unsqueeze(0).unsqueeze(0).repeat(bsz,k,1,1).reshape(bsz*k*num_pts,phy_dim)   # bsz*k*num_pts, phy_dim
    # Gauss_edge = torch.pow(torch.prod(baseweight/torch.pi, dim=-1, keepdim=True),1/2)*torch.exp(-dist_weighted)  # bsz*k*num_pts, in_channels
    Gauss_edge = torch.exp(-dist_weighted)  # bsz*k*num_pts, in_channels
    ### compute value on each edge
    gridweight_edge = grid_weight[edge_grid]  # bsz*k*num_pts
    x_edge = x.reshape(-1, in_channels)[edge_grid]  # bsz*k*num_pts,in_channels
    weighted_Gauss_edge = Gauss_edge * gridweight_edge.unsqueeze(-1)  

    weighted_Gauss_edge = weighted_Gauss_edge.reshape(bsz,k,num_pts,in_channels)
    norms = torch.norm(weighted_Gauss_edge, p=2, dim=1, keepdim=True)+1e-5
    weighted_Gauss_edge = (weighted_Gauss_edge/norms).reshape(bsz*k*num_pts,in_channels)

    value_edge = weighted_Gauss_edge * x_edge   # bsz*k*num_pts, in_channels

    #### shift value of edge_Gauss from {0,1,...,M-1} to {0,1,...,bsz*M-1}
    edge_Gauss = edge_Gauss.reshape(bsz,k,num_pts) + torch.arange(bsz,device = x.device).unsqueeze(-1).unsqueeze(-1)*num_pts  # bsz, k, num_pts
    edge_Gauss = edge_Gauss.reshape(-1)  # bsz*k*num_pts,in_channels

    ### sum
    # edge_Gauss = edge_Gauss.unsqueeze(-1).repeat(1,in_channels)
    # x_hat = torch.zeros(bsz * num_pts, in_channels, device=edge_grid.device).scatter_add_(0, edge_Gauss, value_edge)
    x_hat = scatter(value_edge, edge_Gauss, dim=0)
    x_hat = x_hat.reshape(bsz,num_pts,in_channels)
    return x_hat


def features2value(x_hat,grid,basepts,baseweight,edge_grid,edge_Gauss):
    '''
    x_hat:                      (bsz, num_pts, in_channel)

    grid:                       (bsz, N, phy_dim)

    grid_weight:                (bsz, N)

    basepts:                    (num_pts, phy_dim)

    baseweight:                 (num_pts, phy_dim)

    edge_grid, edge_Gauss:          (bsz, num_pts, k)

    out:                        (bsz, num_pts, in_channel)
    '''

    bsz,N,phy_dim = grid.shape
    num_pts = basepts.shape[0]
    in_channels = x_hat.shape[-1]
    k = edge_grid.shape[-1]


    grid = grid.reshape(-1,phy_dim)   # bsz*N, phy_dim

    #### shift value of edge_grid from {0,1,...,N-1} to {0,1,...,bsz*N-1}
    edge_grid = edge_grid.permute(0,2,1) + torch.arange(bsz,device = x_hat.device).unsqueeze(-1).unsqueeze(-1)*N  # bsz, k, num_pts
    edge_grid = edge_grid.reshape(-1)  # bsz*k*num_pts

    ### compute Gauss on each dege
    edge_Gauss = edge_Gauss.permute(0,2,1).reshape(-1)
    dist_square = (grid[edge_grid] - basepts[edge_Gauss])**2   # bsz*k*num_pts, phy_dim

    dist_square = dist_square.reshape(-1,num_pts,phy_dim)  # bsz*k, num_pts, phy_dim
    dist_weighted = torch.sum(baseweight.unsqueeze(0)*dist_square,dim = -1)     # bsz*k, num_pts
    dist_weighted = dist_weighted.reshape(-1).unsqueeze(-1).repeat(1,in_channels)  # bsz*k*num_pts, in_channels
    baseweight = baseweight.unsqueeze(0).unsqueeze(0).repeat(bsz,k,1,1).reshape(-1,phy_dim)   # bsz*k*num_pts, phy_dim
    # Gauss_edge = torch.pow(torch.prod(baseweight/torch.pi, dim=-1, keepdim=True),1/2)*torch.exp(-dist_weighted)  # bsz*k*num_pts, in_channels
    Gauss_edge = torch.exp(-dist_weighted)  # bsz*k*num_pts, in_channels

    #### shift value of edge_Gauss from {0,1,...,M-1} to {0,1,...,bsz*M-1}
    edge_Gauss = edge_Gauss.reshape(bsz,k,num_pts) + torch.arange(bsz,device = x_hat.device).unsqueeze(-1).unsqueeze(-1)*num_pts  # bsz, k, num_pts
    edge_Gauss = edge_Gauss.reshape(-1)


    ### compute value on each edge
    x_hat_edge = x_hat.reshape(-1, in_channels)[edge_Gauss]  # bsz*k*num_pts,in_channels
    value_edge = Gauss_edge * x_hat_edge   # bsz*k*num_pts,in_channels

    ### sum
    # edge_grid = edge_grid.unsqueeze(-1).repeat(1,in_channels)
    # x = torch.zeros(bsz * N, in_channels, device=edge_grid.device).scatter_add_(0, edge_grid, value_edge)
    x = scatter(value_edge, edge_grid, dim=0)
    x = x.reshape(bsz,N,in_channels)
    return x