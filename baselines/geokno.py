import math
import numpy as np
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
sys.path.append("../")
from models.adam import Adam
from models.losses import LpLoss
## KNO 1D and 2D



class UnitGaussianNormalizer(object):
    def __init__(self, x, aux_dim = 0, eps=1.0e-5):
        super(UnitGaussianNormalizer, self).__init__()
        # x: ndata, nx, nchannels
        # when dim = [], mean and std are both scalars
        self.aux_dim = aux_dim
        self.mean = torch.mean(x[...,0:x.shape[-1]-aux_dim])
        self.std = torch.std(x[...,0:x.shape[-1]-aux_dim])
        self.eps = eps

    def encode(self, x):
        x[...,0:x.shape[-1]-self.aux_dim] = (x[...,0:x.shape[-1]-self.aux_dim] - self.mean) / (self.std + self.eps)
        return x
    

    def decode(self, x):
        std = self.std + self.eps # n
        mean = self.mean
        x[...,0:x.shape[-1]-self.aux_dim] = (x[...,0:x.shape[-1]-self.aux_dim] * std) + mean
        return x
    
    
    def to(self, device):
        if device == torch.device('cuda:0'):
            self.mean = self.mean.cuda()
            self.std = self.std.cuda()
        else:
            self.mean = self.mean.cpu()
            self.std = self.std.cpu()
        



def _get_act(act):
    if act == "tanh":
        func = F.tanh
    elif act == "gelu":
        func = F.gelu
    elif act == "relu":
        func = F.relu_
    elif act == "elu":
        func = F.elu_
    elif act == "leaky_relu":
        func = F.leaky_relu_
    elif act == "none":
        func = None
    else:
        raise ValueError(f"{act} is not supported")
    return func



def compute_edge_weights(nodes, elems):
    nnodes, ndims = nodes.shape
    nelems, _ = elems.shape
    # Initialize adjacency list as a list of sets
    adj_list = [set() for _ in range(nnodes)]
    # Use a set to store unique directed edges
    directed_edge_set = set()

    # Loop through each triangle and create directed edges
    for a, b, c in elems:
        # Add each node's neighbors to its set
        adj_list[a].update([b, c])
        adj_list[b].update([a, c])
        adj_list[c].update([a, b])
    
    directed_edges = []
    edge_weights = [] 
    for a in range(nnodes):
        dx = np.zeros((len(adj_list[a]), ndims))
        for i, b in enumerate(adj_list[a]):
            dx[i, :] = nodes[b,:] - nodes[a,:]
            directed_edges.append([a,b])
        edge_weights.append(np.linalg.pinv(dx).T)
        
    directed_edges = np.array(directed_edges)
    edge_weights = np.concatenate(edge_weights, axis=0)
    return directed_edges, edge_weights



class GradientConv(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        nmode, ndim = modes.shape
        self.modes = modes

        self.scale = 1 / (in_channels * out_channels)

        self.weights_c = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, nmode, dtype=torch.float
            )
        )
        self.weights_s = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, nmode, dtype=torch.float
            )
        )
        self.weights_0 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, 1, dtype=torch.float
            )
        )


    def forward(self, x, directed_edges, edge_weights):
        # compute gradient x, apply a linear layer to [x, gradient x]
        # inputs :
        # x : float Tensor[batch_size, in_channels, max_nnodes]
        # directed_edges : int Tensor[batch_size, max_nedges, 2] 
        # edge_weights : float Tensor[batch_size, max_nedges, ndims]
        
        batch_size, in_channels, max_nnodes = x.shape
        _, max_nedges, ndims = edge_weights.shape
        # Message passing : compute message = edge_weights * (f_source - f_target) for each edge
        # target\source : int Tensor[batch_size, max_nedges]
        # message : float Tensor[batch_size, max_nedges, in_channels*ndims]

        target, source = directed_edges[...,0], directed_edges[...,1]  # source and target nodes of edges
        message = torch.einsum('bed,bec->becd', edge_weights, x[torch.arange(batch_size).unsqueeze(1), source] - x[torch.arange(batch_size).unsqueeze(1), target]).reshape(batch_size, max_nedges, in_channels*ndims)
        
        # x_gradients : float Tensor[batch_size, max_nedges, in_channels*ndims]
        x_gradients = torch.zeros(batch_size, max_nnodes, in_channels*ndims, dtype=message.dtype)
        x_gradients.scatter_add_(dim=1,  src=message, index=target.unsqueeze(2).repeat(1,1,in_channels*ndims))
        
        return x_gradients
    



class GeoKNO(nn.Module):
    def __init__(
        self,
        ndim,
        modes,
        layers,
        fc_dim=128,
        in_dim=3,
        out_dim=1,
        act="gelu",
    ):
        super(GeoKNO, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """
        self.modes = modes
        
        self.layers = layers
        self.fc_dim = fc_dim

        self.ndim = ndim
        self.in_dim = in_dim

        self.fc0 = nn.Linear(in_dim, layers[0])

        self.sp_convs = nn.ModuleList(
            [
                SpectralConv2d(in_size, out_size, modes)
                for in_size, out_size in zip(
                    self.layers, self.layers[1:]
                )
            ]
        )

        self.ws = nn.ModuleList(
            [
                nn.Conv1d(in_size, out_size, 1)
                for in_size, out_size in zip(self.layers, self.layers[1:])
            ]
        )

        if fc_dim > 0:
            self.fc1 = nn.Linear(layers[-1], fc_dim)
            self.fc2 = nn.Linear(fc_dim, out_dim)
        else:
            self.fc2 = nn.Linear(layers[-1], out_dim)

        self.act = _get_act(act)

    def forward(self, x):
        """
        Args:
            - x : (batch size, x_grid, y_grid, 2)
        Returns:
            - x: (batch size, x_grid, y_grid, 1)
        """
        length = len(self.ws)

        aux = x[...,-2-self.ndim:].permute(0, 2, 1)    # coord, weights, mask

        grid, weights, mask = aux[:, 0:self.ndim, :], aux[:, -2:-1, :], aux[:, -1:, :]

        size = grid.shape[-1]
        bases_c, bases_s, bases_0 = compute_Fourier_bases(grid, self.modes, mask)
        wbases_c, wbases_s, wbases_0 = bases_c*(weights*size), bases_s*(weights*size), bases_0*(weights*size)
        
        
        
        x = self.fc0(x[...,0:self.in_dim])
        x = x.permute(0, 2, 1)

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x, wbases_c, wbases_s, wbases_0, bases_c, bases_s, bases_0)
            x2 = w(x)
            x = x1 + x2
            if self.act is not None and i != length - 1:
                x = self.act(x)

        x = x.permute(0, 2, 1)

        if self.fc_dim > 0:
            x = self.fc1(x)
            if self.act is not None:
                x = self.act(x)

        x = self.fc2(x)
        return x



def gradient_test(ndims = 2):
    ################################
    # Preprocess
    ################################
    #nnodes by ndims
    if ndims == 2:
        nodes = np.array([[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0],[0.5,0.5]])
    else: 
        nodes = np.array([[0.0,0.0,1.0],[1.0,0.0,1.0],[1.0,1.0,1.0],[0.0,1.0,1.0],[0.5,0.5,1.0]])
    nnodes, ndims = nodes.shape
    elems = np.array([[0,1,4],[2,4,1],[2,3,4],[0,4,3]], dtype=np.int64)
    # (nedges, 2), (nedges, ndims)
    directed_edges, edge_weights = compute_edge_weights(nodes, elems)
    nedges = directed_edges.shape[0]
    directed_edges = torch.from_numpy(directed_edges)
    edge_weights = torch.from_numpy(edge_weights)

    ################################
    # Construct features
    ################################
    nchannels = 4
    # features is a nchannels by nnodes array, for each channel, the gradient 
    # is gradients[i, :], and the gradient is constant for all nodes
    gradients = np.random.rand(nchannels, ndims)
    features =  gradients @ nodes.T
    # nnodes by (nchannels * ndims) f1_x f1_y f2_x f2_y,.....
    features_gradients_ref = np.repeat(gradients.reshape(1,-1), nnodes, axis=0)
    features = torch.from_numpy(features).permute(1,0)  #nx by nchannels
    
    ################################
    # Online computation
    ################################
    # Message passing: compute f_source - f_target for each edge
    target, source = directed_edges.T  # source and target nodes of edges
    message = torch.einsum('ed,ec->ecd', edge_weights, features[source] - features[target]).reshape(nedges, nchannels*ndims)
    features_gradients = torch.zeros(nnodes, nchannels*ndims, dtype=message.dtype)
    features_gradients.scatter_add_(dim=0,  src=message, index=target.unsqueeze(1).repeat(1,nchannels*ndims))
    
    print("gradient error is ", np.linalg.norm(features_gradients-features_gradients_ref))


def batch_gradient_test(ndims = 2):
    ################################
    # Preprocess
    ################################
    batch_size = 2
    if ndims == 2:
        # batch by nnodes by ndims
        nodes_list = [np.array([[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0],[0.5,0.5]]), \
                    np.array([[1.0,0.0],[1.0,1.0],[0.0,1.0],[0.0,0.0]])]
        
        elems_list = [np.array([[0,1,4],[2,4,1],[2,3,4],[0,4,3]], dtype=np.int64), \
                    np.array([[0,1,2],[0,2,3]], dtype=np.int64)]
    else:
        # batch by nnodes by ndims
        nodes_list = [np.array([[0.0,0.0, 0.0],[1.0,0.0, 1.0],[1.0,1.0, 1.0],[0.0,1.0, 0.0],[0.5,0.5, 0.3]]), \
                      np.array([[1.0,0.0, 1.0],[1.0,1.0, 1.0],[0.0,1.0, 0.0],[0.0,0.0, 0.0]])]
        
        elems_list = [np.array([[0,1,4],[2,4,1],[2,3,4],[0,4,3]], dtype=np.int64), \
                    np.array([[0,1,2],[0,2,3]], dtype=np.int64)]
    max_nnodes = max([nodes.shape[0] for nodes in nodes_list])

    # batch by ndims by nnodes
    grids = np.zeros((batch_size,ndims,max_nnodes))
    for b in range(batch_size):
        grids[b,:,:nodes_list[b].shape[0]] = nodes_list[b].T


    directed_edges_list, edge_weights_list = [], []
    for b in range(batch_size):
        directed_edges, edge_weights = compute_edge_weights(nodes_list[b], elems_list[b])
        directed_edges_list.append(directed_edges)
        edge_weights_list.append(edge_weights) 
    max_nedges = max([directed_edges.shape[0] for directed_edges in directed_edges_list])
    
    #padding with zero
    directed_edges = np.zeros((batch_size, max_nedges, 2), dtype=np.int64)
    edge_weights = np.zeros((batch_size, max_nedges, ndims))
    for b in range(batch_size):
        directed_edges[b, :directed_edges_list[b].shape[0], :] = directed_edges_list[b]
        edge_weights[b, :edge_weights_list[b].shape[0], :] = edge_weights_list[b]
    # batch_size by ndims by max_nnodes 
    grids = torch.from_numpy(grids)
    # batch_size by max_edges by 2
    directed_edges = torch.from_numpy(directed_edges)
    # batch_size by max_edges by ndims
    edge_weights = torch.from_numpy(edge_weights)

    ################################
    # Construct features
    ################################
    nchannels = 5
    # features is a batch_size by nchannels by max_nnodes array, 
    # for each channel, the gradient is gradients[i, :], 
    # and the gradient is constant for all nodes
    gradients = np.random.rand(batch_size, nchannels, ndims)
    # grids = batch_size, ndims, nnodes
    features =  np.einsum('bcd,bdn->bnc', gradients, grids)
    # batch_size by nnodes by (nchannels * ndims) f1_x f1_y f2_x f2_y,.....
    features_gradients_ref = np.zeros((batch_size, max_nnodes, nchannels * ndims))
    for b in range(batch_size):
        features_gradients_ref[b,:nodes_list[b].shape[0], :] = np.repeat(gradients[b,:,:].reshape((1,-1)), nodes_list[b].shape[0], axis=0)
    np.repeat(gradients.reshape((batch_size,1,-1)), max_nnodes, axis=1)
    # batch_size, nnodes, nchannels
    features = torch.from_numpy(features)  
    ################################
    # Online computation
    ################################
    # Message passing: compute f_source - f_target for each edge
    target, source = directed_edges[...,0], directed_edges[...,1]  # source and target nodes of edges
    
    message = torch.einsum('bed,bec->becd', edge_weights, features[torch.arange(batch_size).unsqueeze(1), source] - features[torch.arange(batch_size).unsqueeze(1), target]).reshape(batch_size, max_nedges, nchannels*ndims)
    features_gradients = torch.zeros(batch_size, max_nnodes, nchannels*ndims, dtype=message.dtype)
    # message: batch_size, nedges, nchannels*ndims
    # target: batch_size, nedges
    features_gradients.scatter_add_(dim=1,  src=message, index=target.unsqueeze(2).repeat(1,1,nchannels*ndims))
    
    print("batch gradient error is ", np.linalg.norm(features_gradients-features_gradients_ref))
    for b in range(batch_size):
        print("batch gradient[%d] error is "%b, np.linalg.norm(features_gradients[b,...]-features_gradients_ref[b,...]))

    print("When the point and its neighbors are on the a degenerated plane, the gradient in the normal direction is not known")

if __name__ == "__main__":
    print("2d gradient test")
    gradient_test(ndims = 2)
    batch_gradient_test(ndims=2)
    print("3d gradient test")
    gradient_test(ndims = 3)
    batch_gradient_test(ndims=3)


