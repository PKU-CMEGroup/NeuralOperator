import math
import numpy as np
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.adam import Adam
from models.losses import LpLoss
from geo_utility import compute_edge_gradient_weights
from timeit import default_timer
## KNO 1D and 2D


class UnitGaussianNormalizer(object):
    def __init__(self, x, aux_dim = 0, normalization_dim = [], eps=1.0e-5):
        super(UnitGaussianNormalizer, self).__init__()
        '''
        Normalize the input

            Parameters:  
                x : float[..., nchannels]
                normalization_dim  : list, which dimension to normalize
                                  when normalization_dim = [], global normalization 
                                  when normalization_dim = [0,1,...,len(x.shape)-2], and channel-by-channel normalization 
                aux_dim  : last aux_dim channels are note normalized

            Return :
                UnitGaussianNormalizer : class 
        '''
        self.aux_dim = aux_dim
        self.mean = torch.mean(x[...,0:x.shape[-1]-aux_dim], dim=normalization_dim)
        self.std  = torch.std(x[...,0:x.shape[-1]-aux_dim],  dim=normalization_dim)
        self.eps  = eps

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


def compute_Fourier_modes(ndims, nks, Ls):
    '''
    Compute Fourier modes number k
    Fourier bases are cos(kx), sin(kx), 1
    * We cannot have both k and -k

        Parameters:  
            ndims : int
            nks   : int[ndims]
            Ls    : float[ndims]

        Return :
            k_pairs : float[:, ndims]
    '''
        
    if ndims == 1:
        nx = nks
        Lx = Ls
        nk = nx
        k_pairs    = np.zeros((nk, ndims))
        k_pair_mag = np.zeros(nk)
        i = 0
        for kx in range(1, nx + 1):
            k_pairs[i, :] = 2*np.pi/Lx*kx
            k_pair_mag[i] = np.linalg.norm(k_pairs[i, :])
            i += 1

    elif ndims == 2:
        nx, ny = nks
        Lx, Ly = Ls
        nk = 2*nx*ny + nx + ny
        k_pairs    = np.zeros((nk, ndims))
        k_pair_mag = np.zeros(nk)
        i = 0
        for kx in range(-nx, nx + 1):
            for ky in range(0, ny + 1):
                if (ky==0 and kx<=0): 
                    continue

                k_pairs[i, :] = 2*np.pi/Lx*kx, 2*np.pi/Ly*ky
                k_pair_mag[i] = np.linalg.norm(k_pairs[i, :])
                i += 1

    elif ndims == 3:
        nx, ny, nz = nks
        Lx, Ly, Lz = Ls
        nk = 4*nx*ny*nz + 2*(nx*ny + nx*nz + ny*nz) + nx + ny + nz
        k_pairs    = np.zeros((nk, ndims))
        k_pair_mag = np.zeros(nk)
        i = 0
        for kx in range(-nx, nx + 1):
            for ky in range(-ny, ny + 1):
                for kz in range(0, nz + 1):
                    if (kz==0 and (ky<0  or (ky==0 and kx<=0))): 
                        continue

                    k_pairs[i, :] = 2*np.pi/Lx*kx, 2*np.pi/Ly*ky, 2*np.pi/Lz*kz
                    k_pair_mag[i] = np.linalg.norm(k_pairs[i, :])
                    i += 1
    else:
        raise ValueError(f"{ndims} in compute_Fourier_modes is not supported")
    
    k_pairs = k_pairs[np.argsort(k_pair_mag), :]
    return k_pairs

def compute_Fourier_bases(nodes, modes, node_mask):
    '''
    Compute Fourier bases
    Fourier bases are cos(kx), sin(kx), 1

        Parameters:  
            nodes        : float[batch_size, nnodes, ndims]
            modes        : float[nmodes, ndims]
            node_mask    : float[batch_size, nnodes, 1]

        Return :
            bases_c, bases_s : float[batch_size, nnodes, nmodes]
            bases_0 : float[batch_size, nnodes, 1]
    '''
    # temp : float[batch_size, nnodes, nmodes]
    temp  = torch.einsum("bxd,kd->bxk", nodes, modes) 
    
    bases_c = torch.cos(temp) * node_mask
    bases_s = torch.sin(temp) * node_mask
    bases_0 = node_mask
    return bases_c, bases_s, bases_0

################################################################
# 2d fourier layer
################################################################


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        nmodes, ndims = modes.shape
        self.modes = modes

        self.scale = 1 / (in_channels * out_channels)

        self.weights_c = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, nmodes, dtype=torch.float
            )
        )
        self.weights_s = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, nmodes, dtype=torch.float
            )
        )
        self.weights_0 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, 1, dtype=torch.float
            )
        )


    def forward(self, x, bases_c, bases_s, bases_0, wbases_c, wbases_s, wbases_0):
        '''
        Compute Fourier neural layer
            Parameters:  
                x                   : float[batch_size, in_channels, nnodes]
                bases_c, bases_s    : float[batch_size, nnodes, nmodes]
                bases_0             : float[batch_size, nnodes, 1]
                wbases_c, wbases_s  : float[batch_size, nnodes, nmodes]
                wbases_0            : float[batch_size, nnodes, 1]

            Return :
                x                   : float[batch_size, out_channels, nnodes]
        '''    
        x_c_hat =  torch.einsum("bix,bxk->bik", x, wbases_c)
        x_s_hat = -torch.einsum("bix,bxk->bik", x, wbases_s)
        x_0_hat =  torch.einsum("bix,bxk->bik", x, wbases_0)

        weights_c, weights_s, weights_0 = self.weights_c, self.weights_s, self.weights_0
        
        f_c_hat = torch.einsum("bik,iok->bok", x_c_hat, weights_c) - torch.einsum("bik,iok->bok", x_s_hat, weights_s)
        f_s_hat = torch.einsum("bik,iok->bok", x_s_hat, weights_c) + torch.einsum("bik,iok->bok", x_c_hat, weights_s)
        f_0_hat = torch.einsum("bik,iok->bok", x_0_hat, weights_0) 

        x = torch.einsum("bok,bxk->box", f_0_hat, bases_0)  + 2*torch.einsum("bok,bxk->box", f_c_hat, bases_c) -  2*torch.einsum("bok,bxk->box", f_s_hat, bases_s) 
        
        return x
    
def compute_gradient(f, directed_edges, edge_gradient_weights):
    '''
    Compute gradient of field f at each node
    The gradient is computed by least square.
    Node x has neighbors x1, x2, ..., xj

    x1 - x                        f(x1) - f(x)
    x2 - x                        f(x2) - f(x)
       :      gradient f(x)   =         :
       :                                :
    xj - x                        f(xj) - f(x)
    
    in matrix form   dx  nable f(x)   = df.
    
    The pseudo-inverse of dx is pinvdx.
    Then gradient f(x) for any function f, is pinvdx * df
    directed_edges stores directed edges (x, x1), (x, x2), ..., (x, xj)
    edge_gradient_weights stores its associated weight pinvdx[:,1], pinvdx[:,2], ..., pinvdx[:,j]

    Then the gradient can be computed 
    gradient f(x) = sum_i pinvdx[:,i] * [f(xi) - f(x)] 
    with scatter_add for each edge
    
    
        Parameters: 
            f : float[batch_size, in_channels, nnodes]
            directed_edges : int[batch_size, max_nedges, 2] 
            edge_gradient_weights : float[batch_size, max_nedges, ndims]
            
        Returns:
            x_gradients : float Tensor[batch_size, in_channels*ndims, max_nnodes]
            * in_channels*ndims dimension is gradient[x_1], gradient[x_2], gradient[x_3]......
    '''

    f = f.permute(0,2,1)
    batch_size, max_nnodes, in_channels = f.shape
    _, max_nedges, ndims = edge_gradient_weights.shape
    # Message passing : compute message = edge_gradient_weights * (f_source - f_target) for each edge
    # target\source : int Tensor[batch_size, max_nedges]
    # message : float Tensor[batch_size, max_nedges, in_channels*ndims]

    target, source = directed_edges[...,0], directed_edges[...,1]  # source and target nodes of edges
    message = torch.einsum('bed,bec->becd', edge_gradient_weights, f[torch.arange(batch_size).unsqueeze(1), source] - f[torch.arange(batch_size).unsqueeze(1), target]).reshape(batch_size, max_nedges, in_channels*ndims)
    
    # f_gradients : float Tensor[batch_size, max_nnodes, in_channels*ndims]
    f_gradients = torch.zeros(batch_size, max_nnodes, in_channels*ndims, dtype=message.dtype, device=message.device)
    index1 = target.unsqueeze(2).repeat(1,1,in_channels*ndims).to(torch.int64)
    f_gradients.scatter_add_(dim=1,  src=message, index=index1)
    
    return f_gradients.permute(0,2,1)
    



class GeoKNO(nn.Module):
    def __init__(
        self,
        ndims,
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
        input shape: (batch_size, x=s, y=s, c=3)
        output: the solution 
        output shape: (batch_size, x=s, y=s, c=1)
        """
        self.modes = modes
        
        self.layers = layers
        self.fc_dim = fc_dim

        self.ndims = ndims
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

        self.gws = nn.ModuleList(
            [
                nn.Conv1d(ndims*in_size, out_size, 1)
                for in_size, out_size in zip(self.layers, self.layers[1:])
            ]
        )

        if fc_dim > 0:
            self.fc1 = nn.Linear(layers[-1], fc_dim)
            self.fc2 = nn.Linear(fc_dim, out_dim)
        else:
            self.fc2 = nn.Linear(layers[-1], out_dim)

        self.act = _get_act(act)
        self.softsign = F.softsign

    def forward(self, x, aux):
        """
        Args:
            - x : (batch nnodes, x_grid, y_grid, 2)
        Returns:
            - x : (batch nnodes, x_grid, y_grid, 1)
        """
        length = len(self.ws)

        # batch_size, nnodes, ndims
        node_mask, nodes, node_weights, directed_edges, edge_gradient_weights = aux

        bases_c,  bases_s,  bases_0  = compute_Fourier_bases(nodes, self.modes, node_mask)
        wbases_c, wbases_s, wbases_0 = bases_c*node_weights, bases_s*node_weights, bases_0*node_weights
        
        
        
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        for i, (speconv, w, gw) in enumerate(zip(self.sp_convs, self.ws, self.gws)):
            x1 = speconv(x, bases_c, bases_s, bases_0, wbases_c, wbases_s, wbases_0)
            x2 = w(x)
            x3 = gw(self.softsign(compute_gradient(x, directed_edges, edge_gradient_weights)))
            x = x1 + x2 #+ x3
            if self.act is not None and i != length - 1:
                x = self.act(x)

        x = x.permute(0, 2, 1)

        if self.fc_dim > 0:
            x = self.fc1(x)
            if self.act is not None:
                x = self.act(x)

        x = self.fc2(x)
        return x
    




# x_train, y_train, x_test, y_test are [n_data, n_x, n_channel] arrays
def GeoKNO_train(x_train, aux_train, y_train, x_test, aux_test, y_test, config, model, save_model_name="./GeoKNO_model"):
    n_train, n_test = x_train.shape[0], x_test.shape[0]
    train_rel_l2_losses = []
    test_rel_l2_losses = []
    test_l2_losses = []
    normalization_x, normalization_y, normalization_dim = config["train"]["normalization_x"], config["train"]["normalization_y"], config["train"]["normalization_dim"]
    x_aux_dim, y_aux_dim = config["train"]["x_aux_dim"], config["train"]["y_aux_dim"]
    
    ndims = model.ndims # n_train, size, n_channel
    print("In GeoKNO_train, ndims = ", ndims)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    if normalization_x:
        x_normalizer = UnitGaussianNormalizer(x_train, aux_dim = x_aux_dim, normalization_dim=normalization_dim)
        x_train = x_normalizer.encode(x_train)
        x_test = x_normalizer.encode(x_test)
        x_normalizer.to(device)
        
    if normalization_y:
        y_normalizer = UnitGaussianNormalizer(y_train, aux_dim = y_aux_dim, normalization_dim=normalization_dim)
        y_train = y_normalizer.encode(y_train)
        y_test = y_normalizer.encode(y_test)
        y_normalizer.to(device)


    node_mask_train, nodes_train, node_weights_train, directed_edges_train, edge_gradient_weights_train = aux_train
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train, node_mask_train, nodes_train, node_weights_train, directed_edges_train, edge_gradient_weights_train), 
                                               batch_size=config['train']['batch_size'], shuffle=True)
    
    node_mask_test, nodes_test, node_weights_test, directed_edges_test, edge_gradient_weights_test = aux_test
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test, node_mask_test, nodes_test, node_weights_test, directed_edges_test, edge_gradient_weights_test), 
                                               batch_size=config['train']['batch_size'], shuffle=False)
    
    
    # Load from checkpoint
    optimizer = Adam(model.parameters(), betas=(0.9, 0.999),
                     lr=config['train']['base_lr'], weight_decay=config['train']['weight_decay'])
    
    if config['train']['scheduler'] == "MultiStepLR":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=config['train']['milestones'],
                                                     gamma=config['train']['scheduler_gamma'])
    elif config['train']['scheduler'] == "CosineAnnealingLR":
        T_max = (config['train']['epochs']//10)*(n_train//config['train']['batch_size'])
        eta_min  = 0.0
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min = eta_min)
    elif config["train"]["scheduler"] == "OneCycleLR":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config['train']['base_lr'], 
            div_factor=2, final_div_factor=100,pct_start=0.2,
            steps_per_epoch=1, epochs=config['train']['epochs'])
    else:
        print("Scheduler ", config['train']['scheduler'], " has not implemented.")

    model.train()
    myloss = LpLoss(d=1, p=2, size_average=False)

    epochs = config['train']['epochs']


    for ep in range(epochs):
        t = 0
        t1 = default_timer()
        train_rel_l2 = 0

        model.train()
        for x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights in train_loader:
            x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights = x.to(device), y.to(device), node_mask.to(device), nodes.to(device), node_weights.to(device), directed_edges.to(device), edge_gradient_weights.to(device)

            batch_size_ = x.shape[0]
            optimizer.zero_grad()
            out = model(x, (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights)) #.reshape(batch_size_,  -1)
            if normalization_y:
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)

            loss = myloss(out.view(batch_size_,-1), y.view(batch_size_,-1))
            loss.backward()

            optimizer.step()
            train_rel_l2 += loss.item()

        test_l2 = 0
        test_rel_l2 = 0
        with torch.no_grad():
            for x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights in test_loader:
                x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights = x.to(device), y.to(device), node_mask.to(device), nodes.to(device), node_weights.to(device), directed_edges.to(device), edge_gradient_weights.to(device)

                batch_size_ = x.shape[0]
                out = model(x, (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights)) #.reshape(batch_size_,  -1)

                if normalization_y:
                    out = y_normalizer.decode(out)
                    y = y_normalizer.decode(y)

                test_rel_l2 += myloss(out.view(batch_size_,-1), y.view(batch_size_,-1)).item()
                test_l2 += myloss.abs(out.view(batch_size_,-1), y.view(batch_size_,-1)).item()




        scheduler.step()

        train_rel_l2/= n_train
        test_l2 /= n_test
        test_rel_l2/= n_test
        
        train_rel_l2_losses.append(train_rel_l2)
        test_rel_l2_losses.append(test_rel_l2)
        test_l2_losses.append(test_l2)
    
        t2 = default_timer()
        t += t2 - t1
        if (ep %10 == 0) or (ep == epochs -1):
            t_avg = t / 10
            t = 0
            print("Epoch : ", ep, "Train time : ", t_avg, " Rel. Train L2 Loss : ", train_rel_l2, " Rel. Test L2 Loss : ", test_rel_l2, " Test L2 Loss : ", test_l2, flush=True)
            torch.save(model, save_model_name)
    
    
    return train_rel_l2_losses, test_rel_l2_losses, test_l2_losses









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
    elem_dim = 2
    elems = np.array([[elem_dim,0,1,4],[elem_dim,2,4,1],[elem_dim,2,3,4],[elem_dim,0,4,3]], dtype=np.int64)
    # (nedges, 2), (nedges, ndims)
    directed_edges, edge_gradient_weights, _ = compute_edge_gradient_weights(nodes, elems, rcond=1e-3)
    nedges = directed_edges.shape[0]
    directed_edges = torch.from_numpy(directed_edges)
    edge_gradient_weights = torch.from_numpy(edge_gradient_weights)

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
    if ndims == 3:
        # remove the gradient in the normal direction
        features_gradients_ref[:,2::ndims] = 0.0

    features = torch.from_numpy(features).permute(1,0)  #nx by nchannels
    
    ################################
    # Online computation
    ################################
    # Message passing: compute f_source - f_target for each edge
    target, source = directed_edges.T  # source and target nodes of edges
    message = torch.einsum('ed,ec->ecd', edge_gradient_weights, features[source] - features[target]).reshape(nedges, nchannels*ndims)
    features_gradients = torch.zeros(nnodes, nchannels*ndims, dtype=message.dtype)
    features_gradients.scatter_add_(dim=0,  src=message, index=target.unsqueeze(1).repeat(1,nchannels*ndims))
    
    print("gradient error is ", np.linalg.norm(features_gradients-features_gradients_ref))
    assert(np.allclose(features_gradients-features_gradients_ref, 0.0, rtol=1e-15))


def batch_gradient_test(ndims = 2):
    ################################
    # Preprocess
    ################################
    batch_size = 2
    if ndims == 2:
        elem_dims = [2,2]
        # batch by nnodes by ndims
        nodes_list = [np.array([[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0],[0.5,0.5]]), \
                    np.array([[1.0,0.0],[1.0,1.0],[0.0,1.0],[0.0,0.0]])]
        
        elems_list = [np.array([[elem_dims[0],0,1,4],[elem_dims[0],2,4,1],[elem_dims[0],2,3,4],[elem_dims[0],0,4,3]], dtype=np.int64), \
                    np.array([[elem_dims[1],0,1,2],[elem_dims[1],0,2,3]], dtype=np.int64)]
    else:
        # batch by nnodes by ndims
        nodes_list = [np.array([[0.0,0.0, 0.0],[1.0,0.0, 0.0],[1.0,1.0, 0.0],[0.0,1.0, 0.0],[0.5,0.5, 0.0]]), \
                      np.array([[1.0,0.0, 0.0],[1.0,1.0, 0.0],[0.0,1.0, 0.0],[1.0,0.0, 1.0]])]
        elem_dims = [2,3]
        elems_list = [np.array([[elem_dims[0],0,1,4],[elem_dims[0],2,4,1],[elem_dims[0],2,3,4],[elem_dims[0],0,4,3]], dtype=np.int64), \
                    np.array([[elem_dims[1],0,1,2, 3]], dtype=np.int64)]
    max_nnodes = max([nodes.shape[0] for nodes in nodes_list])

    # batch by ndims by nnodes
    grids = np.zeros((batch_size,ndims,max_nnodes))
    for b in range(batch_size):
        grids[b,:,:nodes_list[b].shape[0]] = nodes_list[b].T


    directed_edges_list, edge_weights_list = [], []
    for b in range(batch_size):
        directed_edges, edge_gradient_weights, _ = compute_edge_gradient_weights(nodes_list[b], elems_list[b], rcond=1e-3)
        directed_edges_list.append(directed_edges)
        edge_weights_list.append(edge_gradient_weights) 
    max_nedges = max([directed_edges.shape[0] for directed_edges in directed_edges_list])
    
    #padding with zero
    directed_edges = np.zeros((batch_size, max_nedges, 2), dtype=np.int64)
    edge_gradient_weights = np.zeros((batch_size, max_nedges, ndims))
    for b in range(batch_size):
        directed_edges[b, :directed_edges_list[b].shape[0], :] = directed_edges_list[b]
        edge_gradient_weights[b, :edge_weights_list[b].shape[0], :] = edge_weights_list[b]
    # batch_size by ndims by max_nnodes 
    grids = torch.from_numpy(grids)
    # batch_size by max_edges by 2
    directed_edges = torch.from_numpy(directed_edges)
    # batch_size by max_edges by ndims
    edge_gradient_weights = torch.from_numpy(edge_gradient_weights)

    ################################
    # Construct features
    ################################
    nchannels = 5
    # features is a batch_size by nchannels by max_nnodes array, 
    # for each channel, the gradient is gradients[i, :], 
    # and the gradient is constant for all nodes
    gradients = np.random.rand(batch_size, nchannels, ndims)
    # grids = batch_size, ndims, nnodes
    features =  np.einsum('bcd,bdn->bcn', gradients, grids)
    # batch_size by nnodes by (nchannels * ndims) f1_x f1_y f2_x f2_y,.....
    features_gradients_ref = np.zeros((batch_size, nchannels * ndims, max_nnodes))
    for b in range(batch_size):
        # print(np.tile(gradients[b,:,:].flatten(), (nodes_list[b].shape[0],1)).shape)
        features_gradients_ref[b,:,:nodes_list[b].shape[0]] = np.tile(gradients[b,:,:].flatten(), (nodes_list[b].shape[0],1)).T
    for i, elem_dim in enumerate(elem_dims):
        if ndims == 3 and elem_dim == 2:
            # remove the gradient in the normal direction
            features_gradients_ref[i,2::ndims,:] = 0.0
    # batch_size, nnodes, nchannels
    features = torch.from_numpy(features)  
    ##############################
    # Online computation
    ##############################
    features_gradients = compute_gradient(features, directed_edges, edge_gradient_weights)

    
    for b in range(batch_size):
        print("batch gradient[%d] error is "%b, np.linalg.norm(features_gradients[b,...]-features_gradients_ref[b,...]))

    assert(np.allclose(features_gradients-features_gradients_ref, 0.0, rtol=1e-15))
    
    print("When the point and its neighbors are on the a degenerated plane, the gradient in the normal direction is not known")

if __name__ == "__main__":
    print("2d gradient test")
    gradient_test(ndims = 2)
    batch_gradient_test(ndims=2)
    print("3d gradient test")
    gradient_test(ndims = 3)
    batch_gradient_test(ndims=3)


