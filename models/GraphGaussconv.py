import torch
import torch.nn as nn
from torch_scatter import scatter


class GraphGaussconv(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, init_weight, device='cuda'):
        super(GraphGaussconv, self).__init__()

        self.fc1 = nn.Linear(in_channels,hidden_channels)
        self.fc2 = nn.Linear(hidden_channels,out_channels)
        self.baseweight = nn.Parameter(init_weight)  #out_channels
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.device = device

    def forward(self, x, grid, grid_weight, edge_src, edge_dst):
        '''
        x:                          (bsz, in_channel, N)

        grid:                       (bsz, N, phy_dim)

        grid_weight:                (bsz, N)

        edge_src,edge_dst:          (bsz, N, k)
        '''
        x = x.permute(0,2,1)
        x = self.fc1(x)
        bsz,N,phy_dim = grid.shape
        baseweight = self.baseweight

        edge_src = edge_src + (torch.arange(bsz, device = x.device) * N).reshape(bsz, 1, 1)  # Shape: (bsz, N, k)
        edge_dst = edge_dst + (torch.arange(bsz, device = x.device) * N).reshape(bsz, 1, 1)  # Shape: (bsz, N, k)

        batch = torch.arange(bsz, device = self.device).reshape(bsz,1).repeat(1,N) 
        batch = batch.reshape(-1)  
        grid = grid.reshape(-1,phy_dim)   # bsz*N,phy_dim
        grid_weight = grid_weight.reshape(-1)  # bsz*N
        edge_src = edge_src.reshape(-1)  #bsz*N*K
        edge_dst = edge_dst.reshape(-1)  #bsz*N*K

        dist = torch.sqrt(torch.sum((grid[edge_src] - grid[edge_dst])**2,dim=1)).unsqueeze(1).repeat(1,self.hidden_channels)  # bsz*N*K, hidden_channels


        weight_expand = grid_weight[edge_src] # bsz*N*K
        x_expand = x.reshape(-1, self.hidden_channels)[edge_src]  #bsz*N*K,hidden_channels

        # ##### k(x,y) = exp(-w*(x-y)^2)
        Gauss_edge = torch.sqrt((baseweight/torch.pi)**phy_dim)*torch.exp(-baseweight*dist**2)  # bsz*N*K, hidden_channels

        src = Gauss_edge * x_expand * weight_expand.unsqueeze(-1) # bsz*N*K, hidden_channels

        edge_dst = edge_dst.unsqueeze(-1).repeat(1,self.hidden_channels)
        out = torch.zeros(bsz*N,self.hidden_channels,device = x.device).scatter_add_(0,edge_dst,src) 
        out = out.reshape(bsz,N,self.hidden_channels)

        out = self.fc2(out)

        out = out.permute(0,2,1)

        return out
    


