import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utility.adam import Adam
from utility.losses import LpLoss
from utility.normalizer import UnitGaussianNormalizer
## FNO 1D and 2D



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
        nk, Lx = nks[0], Ls[0]
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
# Fourier layer
################################################################
class SpectralConv(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv, self).__init__()
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
    



class GeoFNO(nn.Module):
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
        super(GeoFNO, self).__init__()
        """
        The overall network. 
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. len(layers)-1 layers of the Fourier neural layers u' = (W + K)(u).
           linear functions  W: parameterized by self.ws; 
           integral operator K: parameterized by self.sp_convs
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
            
            Parameters: 
                ndims : int 
                    Dimensionality of the problem
                modes : float[nmodes, ndims]
                    It contains nmodes modes k, and Fourier bases include : cos(k x), sin(k x), 1  
                    * We cannot have both k and -k
                layers : list of int
                    number of channels of each layer
                    The lifting layer first lifts to layers[0]
                    The first Fourier layer maps between layers[0] and layers[1]
                    ...
                    The nlayers Fourier layer maps between layers[nlayers-1] and layers[nlayers]
                    The number of Fourier layers is len(layers) - 1
                fc_dim : int 
                    hidden layers for the projection layer, when fc_dim > 0, otherwise there is no hidden layer
                in_dim : int 
                    The number of channels for the input function
                    For example, when the coefficient function and locations (a(x, y), x, y) are inputs, in_dim = 3
                out_dim : int 
                    The number of channels for the output function
                act : string (default gelu)
                    The activation function

            
            Returns:
                Geometry aware FNO

        """
        self.modes = modes
        
        self.layers = layers
        self.fc_dim = fc_dim

        self.ndims = ndims
        self.in_dim = in_dim

        self.fc0 = nn.Linear(in_dim, layers[0])

        self.sp_convs = nn.ModuleList(
            [
                SpectralConv(in_size, out_size, modes)
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

    def forward(self, x, aux):
        """
        Forward evaluation. 
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. len(layers)-1 layers of the Fourier neural layers u' = (W + K)(u).
           linear functions  W: parameterized by self.ws; 
           integral operator K: parameterized by self.sp_convs
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
            
            Parameters: 
                x : Tensor float[batch_size, max_nnomdes, in_dim] 
                    Input data
                aux : list of Tensor, containing
                    node_mask : Tensor int[batch_size, max_nnomdes, 1]  
                                1: node; otherwise 0

                    nodes : Tensor float[batch_size, max_nnomdes, ndim]  
                            nodal coordinate; padding with 0

                    node_weights  : Tensor float[batch_size, max_nnomdes, 1]  
                                    rho(x)dx used for integration; padding with 0
            
            Returns:
                G(x) : Tensor float[batch_size, max_nnomdes, out_dim] 

        """

        length = len(self.ws)

        node_mask, nodes, node_weights = aux
        # bases  : float[batch_size, max_nnodes, nmodes]
        # node_weights : float[batch_size, max_nnomdes, 1]  
        bases_c,  bases_s,  bases_0  = compute_Fourier_bases(nodes, self.modes, node_mask)
        wbases_c, wbases_s, wbases_0 = bases_c*node_weights, bases_s*node_weights, bases_0*node_weights
        
        # x: batch_size, nnodes, ndims
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x, bases_c, bases_s, bases_0, wbases_c, wbases_s, wbases_0)
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

        return x * node_mask






# x_train, y_train, x_test, y_test are [n_data, n_x, n_channel] arrays
def GeoFNO_train(x_train, aux_train, y_train, x_test, aux_test, y_test, config, model, save_model_name="./GeoFNO_model"):
    n_train, n_test = x_train.shape[0], x_test.shape[0]
    train_rel_l2_losses = []
    test_rel_l2_losses = []
    test_l2_losses = []
    normalization_x, normalization_y = config["train"]["normalization_x"], config["train"]["normalization_y"]
    normalization_dim_x, normalization_dim_y = config["train"]["normalization_dim_x"], config["train"]["normalization_dim_y"]
    non_normalized_dim_x, non_normalized_dim_y = config["train"]["non_normalized_dim_x"], config["train"]["non_normalized_dim_y"]
    
    ndims = model.ndims # n_train, size, n_channel
    print("In GeoFNO_train, ndims = ", ndims)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    if normalization_x:
        x_normalizer = UnitGaussianNormalizer(x_train, non_normalized_dim = non_normalized_dim_x, normalization_dim=normalization_dim_x)
        x_train = x_normalizer.encode(x_train)
        x_test = x_normalizer.encode(x_test)
        x_normalizer.to(device)
        
    if normalization_y:
        y_normalizer = UnitGaussianNormalizer(y_train, non_normalized_dim = non_normalized_dim_y, normalization_dim=normalization_dim_y)
        y_train = y_normalizer.encode(y_train)
        y_test = y_normalizer.encode(y_test)
        y_normalizer.to(device)


    node_mask_train, nodes_train, node_weights_train = aux_train
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train, node_mask_train, nodes_train, node_weights_train), 
                                               batch_size=config['train']['batch_size'], shuffle=True)
    
    node_mask_test, nodes_test, node_weights_test = aux_test
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test, node_mask_test, nodes_test, node_weights_test), 
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
        train_rel_l2 = 0

        model.train()
        for x, y, node_mask, nodes, node_weights in train_loader:
            x, y, node_mask, nodes, node_weights = x.to(device), y.to(device), node_mask.to(device), nodes.to(device), node_weights.to(device)

            batch_size_ = x.shape[0]
            optimizer.zero_grad()
            out = model(x, (node_mask, nodes, node_weights)) #.reshape(batch_size_,  -1)
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
            for x, y, node_mask, nodes, node_weights in test_loader:
                x, y, node_mask, nodes, node_weights = x.to(device), y.to(device), node_mask.to(device), nodes.to(device), node_weights.to(device)

                batch_size_ = x.shape[0]
                out = model(x, (node_mask, nodes, node_weights)) #.reshape(batch_size_,  -1)

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
    

        if (ep %10 == 0) or (ep == epochs -1):
            print("Epoch : ", ep, " Rel. Train L2 Loss : ", train_rel_l2, " Rel. Test L2 Loss : ", test_rel_l2, " Test L2 Loss : ", test_l2, flush=True)
            torch.save(model, save_model_name)
    
    
    return train_rel_l2_losses, test_rel_l2_losses, test_l2_losses


