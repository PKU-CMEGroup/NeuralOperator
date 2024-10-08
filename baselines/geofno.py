import math
import numpy as np
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
sys.path.append("../")
from models.adam import Adam
from models.losses import LpLoss
## FNO 1D and 2D



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



@torch.jit.script
def compl_mul1d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
    res = torch.einsum("bix,iox->box", a, b)
    return res


@torch.jit.script
def compl_mul2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
    res = torch.einsum("bixy,ioxy->boxy", a, b)
    return res



# def compute_Fourier_modes(ndim, nks, Ls):
#     # 2d 
#     if ndim == 2:
#         modes1, modes2 = nks
#         Lx, Ly = Ls
#         k_pairs = np.zeros((2*modes1, modes2, ndim))
#         for kx in list(range(0, modes1)) + list(range(-modes1, 0)): #range(-modes1, modes1):
#             for ky in range(modes2):
#                 k_pairs[kx, ky, :] = 2*np.pi/Lx*kx, 2*np.pi/Ly*ky
#     return k_pairs

def compute_Fourier_modes(ndim, nks, Ls):
    # 2d 
    if ndim == 2:
        nx, ny = nks
        Lx, Ly = Ls
        nk = 2*nx*ny + nx + ny
        k_pairs    = np.zeros((nk, ndim))
        k_pair_mag = np.zeros(nk)
        i = 0
        for kx in range(-nx, nx + 1):
            for ky in range(0, ny + 1):
                if (ky==0 and kx<=0): 
                    continue

                k_pairs[i, :] = 2*np.pi/Lx*kx, 2*np.pi/Ly*ky
                k_pair_mag[i] = np.linalg.norm(k_pairs[i, :])
                i += 1
        

    k_pairs = k_pairs[np.argsort(k_pair_mag), :]
    return k_pairs

def compute_Fourier_bases(grid, modes, mask):
    #grid : batchsize, ndim, nx
    #modes: nk, ndim
    #mask : batchsize, 1, nx
    temp  = torch.einsum("bdx,kd->bkx", grid, modes) 
    #temp: batchsize, nx, nk
    bases_c = torch.cos(temp) * mask
    bases_s = torch.sin(temp) * mask
    bases_0 = mask
    return bases_c, bases_s, bases_0

################################################################
# 2d fourier layer
################################################################


class SpectralConv2d(nn.Module):
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


    def forward(self, x, wbases_c, wbases_s, wbases_0, bases_c, bases_s, bases_0):
        #batchsize = x.shape[0]
        size = x.shape[-1]
        # # Compute Fourier coeffcients up to factor of e^(- something constant)
        # x_ft = torch.fft.rfftn(x, dim=[2, 3])
        # # Multiply relevant Fourier modes
        # out_ft0 = torch.zeros(
        #     batchsize,
        #     self.out_channels,
        #     x.size(-2),
        #     x.size(-1) // 2 + 1,
        #     device=x.device,
        #     dtype=torch.cfloat,
        # )
        # out_ft0[:, :, : self.modes1, : self.modes2] = compl_mul2d(
        #     x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        # )
        # out_ft0[:, :, -self.modes1 :, : self.modes2] = compl_mul2d(
        #     x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        # )
        # x0 = torch.fft.irfftn(out_ft0, s=(x.size(-2), x.size(-1)), dim=[2, 3])


        x_c_hat =  torch.einsum("bix,bkx->bik", x, wbases_c)
        x_s_hat = -torch.einsum("bix,bkx->bik", x, wbases_s)
        x_0_hat =  torch.einsum("bix,bkx->bik", x, wbases_0)

        # print("fft+: ", torch.norm(x_ft[:, :,  : self.modes1, : self.modes2] - torch.complex(x_c_hat[:, :, : self.modes1  , :], x_s_hat[:, :, : self.modes1  , :]))/torch.norm(x_ft[:, :,  : self.modes1, : self.modes2]))
        # print("fft-: ", torch.norm(x_ft[:, :,  -self.modes1:, : self.modes2] - torch.complex(x_c_hat[:, :, -self.modes1:  , :], x_s_hat[:, :, -self.modes1:  , :]))/torch.norm(x_ft[:, :,  -self.modes1:, : self.modes2]))
        
        # weights12 = torch.cat((self.weights1, self.weights2), axis=2)/(size1*size2)
        weights_c, weights_s, weights_0 = self.weights_c/(size), self.weights_s/(size), self.weights_0/(size)
        f_c_hat = torch.einsum("bik,iok->bok", x_c_hat, weights_c) - torch.einsum("bik,iok->bok", x_s_hat, weights_s)
        f_s_hat = torch.einsum("bik,iok->bok", x_s_hat, weights_c) + torch.einsum("bik,iok->bok", x_c_hat, weights_s)
        f_0_hat = torch.einsum("bik,iok->bok", x_0_hat, weights_0) 

        # print(torch.norm(torch.einsum("bok,bkx->box", f_0_hat, bases_0)), torch.norm(torch.einsum("bok,bkx->box", f_c_hat, bases_c)), torch.norm(torch.einsum("bok,bkx->box", f_s_hat, bases_s)))
        x = torch.einsum("bok,bkx->box", f_0_hat, bases_0)  + 2*torch.einsum("bok,bkx->box", f_c_hat, bases_c) -  2*torch.einsum("bok,bkx->box", f_s_hat, bases_s) 
        
        return x
    

# Epoch :  490  Rel. Train L2 Loss :  0.002117458561435342  Rel. Test L2 Loss :  0.0061201491020619865  Test L2 Loss :  4.155228569288738e-05
# Epoch :  499  Rel. Train L2 Loss :  0.002108049723319709  Rel. Test L2 Loss :  0.006121462248265743  Test L2 Loss :  4.155958908086177e-05
# Epoch :  490  Rel. Train L2 Loss :  0.0022899034479632973  Rel. Test L2 Loss :  0.006440818700939417  Test L2 Loss :  4.3853282695636155e-05
# Epoch :  499  Rel. Train L2 Loss :  0.002278903964906931  Rel. Test L2 Loss :  0.006441509649157524  Test L2 Loss :  4.385238949907943e-05




class GeoFNO(nn.Module):
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
        super(GeoFNO, self).__init__()

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

        # size_x, size_y = x.shape[-2], x.shape[-1]

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x, wbases_c, wbases_s, wbases_0, bases_c, bases_s, bases_0)
            # x2 = w(x.view(batchsize, self.layers[i], -1)).view(
            #     batchsize, self.layers[i + 1], size_x, size_y
            # )
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






# x_train, y_train, x_test, y_test are [n_data, n_x, n_channel] arrays
def GeoFNO_train(x_train, y_train, x_test, y_test, config, model, save_model_name="./GeoFNO_model"):
    n_train, n_test = x_train.shape[0], x_test.shape[0]
    train_rel_l2_losses = []
    test_rel_l2_losses = []
    test_l2_losses = []
    normalization_x, normalization_y, normalization_dim = config["train"]["normalization_x"], config["train"]["normalization_y"], config["train"]["normalization_dim"]
    ndim = model.ndim # n_train, size, n_channel
    print("In GeoFNO_train, ndim = ", ndim)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    if normalization_x:
        x_normalizer = UnitGaussianNormalizer(x_train, aux_dim = ndim+2)
        x_train = x_normalizer.encode(x_train)
        x_test = x_normalizer.encode(x_test)
        x_normalizer.to(device)
        
    if normalization_y:
        y_normalizer = UnitGaussianNormalizer(y_train, aux_dim = 0)
        y_train = y_normalizer.encode(y_train)
        y_test = y_normalizer.encode(y_test)
        y_normalizer.to(device)


    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), 
                                               batch_size=config['train']['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), 
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
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            batch_size_ = x.shape[0]
            optimizer.zero_grad()
            out = model(x) #.reshape(batch_size_,  -1)
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
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                batch_size_ = x.shape[0]
                out = model(x) #.reshape(batch_size_,  -1)

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


# class UnitGaussianNormalizer(object):
#     def __init__(self, x, aux_dim = 0, eps=1.0e-5):
#         super(UnitGaussianNormalizer, self).__init__()
#         # x: ndata, nx, nchannels
#         # when dim = [], mean and std are both scalars
#         self.aux_dim = aux_dim
#         self.mean = torch.mean(x[:,:,0:x.shape[-1]-aux_dim])
#         self.std = torch.std(x[:,:,0:x.shape[-1]-aux_dim])
#         self.eps = eps

#     def encode(self, x):
#         x[:,:,0:x.shape[-1]-self.aux_dim] = (x[:,:,0:x.shape[-1]-self.aux_dim] - self.mean) / (self.std + self.eps)
#         return x
    

#     def decode(self, x):
#         std = self.std + self.eps # n
#         mean = self.mean
#         x[:,:,0:x.shape[-1]-self.aux_dim] = (x[:,:,0:x.shape[-1]-self.aux_dim] * std) + mean
#         return x
    
        
    
#     def to(self, device):
#         if device == torch.device('cuda:0'):
#             self.mean = self.mean.cuda()
#             self.std = self.std.cuda()
#         else:
#             self.mean = self.mean.cpu()
#             self.std = self.std.cpu()
        

# def _get_act(act):
#     if act == "tanh":
#         func = F.tanh
#     elif act == "gelu":
#         func = F.gelu
#     elif act == "relu":
#         func = F.relu_
#     elif act == "elu":
#         func = F.elu_
#     elif act == "leaky_relu":
#         func = F.leaky_relu_
#     elif act == "none":
#         func = None
#     else:
#         raise ValueError(f"{act} is not supported")
#     return func





# ################################################################
# # fourier modes and basis
# ################################################################

# def compute_Fourier_modes(ndim, nks, Ls):
#     # 2d 
#     if ndim == 2:
#         nx, ny = nks
#         Lx, Ly = Ls
#         nk = 2*nx*ny + nx + ny
#         k_pairs    = np.zeros((nk, ndim))
#         k_pair_mag = np.zeros(nk)
#         i = 0
#         for kx in range(-nx, nx + 1):
#             for ky in range(-ny, ny + 1):
#                 if kx < 0 or (kx==0 and ky<=0): 
#                     continue

#                 k_pairs[i, :] = 2*np.pi/Lx*kx, 2*np.pi/Ly*ky
#                 k_pair_mag[i] = np.linalg.norm(k_pairs[i, :])
#                 i += 1
        
#     elif ndim == 3:
#         nx, ny, nz = nks
#         Lx, Ly, Lz = Ls
#         nk = 2*nx*ny*nz + 2*nx*ny + 2*ny*nz+ 2*nx*nz + nx + ny + nz
#         k_pairs    = np.zeros((nk, ndim))
#         k_pair_mag = np.zeros(nk)

#         i = 0
#         for kx in range(-nx, nx + 1):
#             for ky in range(-ny, ny + 1):
#                 for kz in range(-nz, nz + 1):
#                     if kx < 0 or (kx==0 and ky<0) or (kx==0 and ky==0 and kz<=0): 
#                         continue
                
#                     k_pairs[i, :] = 2*np.pi/Lx * kx, 2*np.pi/Ly * ky, 2*np.pi/Lz * kz
#                     k_pair_mag[i] = np.linalg.norm(k_pairs[i, :])
#                     i += 1
#     else:
#         exit("In compute_Fourier_modes, dim = ", ndim)

#     k_pairs = k_pairs[np.argsort(k_pair_mag), :]
#     return k_pairs


# def compute_Fourier_bases(grid, modes, mask):
#     #grid : batchsize, ndim, nx
#     #modes: nk, ndim
#     #mask : batchsize, 1, nx
    
#     temp  = torch.einsum("bdx,kd->bkx", grid, modes) 
    
#     #temp: batchsize, nx, nk
#     bases_c = torch.cos(temp) * mask
#     bases_s = torch.sin(temp) * mask
#     bases_0 = mask
#     return bases_c, bases_s, bases_0


# class GeoSpectralConv(nn.Module):
#     def __init__(self, in_channels, out_channels, modes):
#         super(GeoSpectralConv, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         # Array[nm,dim], modes[i, :] = kx, ky, kz, 
#         # kx> 0 or (kx=0, ky>0) or (kx=0, ky=0, kz >= 0)
#         nk, dim = modes.shape  
#         self.modes = modes

#         self.scale = 1 / (in_channels * out_channels)
#         self.weights_c = nn.Parameter(
#             self.scale
#             * torch.rand(
#                 in_channels, out_channels, nk, dtype=torch.float
#             )
#         )
#         self.weights_s = nn.Parameter(
#             self.scale
#             * torch.rand(
#                 in_channels, out_channels, nk, dtype=torch.float
#             )
#         )
#         self.weights_0 = nn.Parameter(
#             self.scale
#             * torch.rand(
#                 in_channels, out_channels, 1, dtype=torch.float
#             )
#         )

#     def forward(self, x, wbases_c, wbases_s, wbases_0, bases_c, bases_s, bases_0):
#         # x: batchsize, in_channels, nx 
#         nx = x.shape[2]

#         # Compute Fourier coeffcients up to factor of e^(- something constant)
#         x_c_hat =  torch.einsum("bix,bkx->bik", x, wbases_c)
#         x_s_hat = -torch.einsum("bix,bkx->bik", x, wbases_s)
#         x_0_hat =  torch.einsum("bix,bkx->bik", x, wbases_0)
        

#         # Multiply relevant Fourier modes
#         # W_k a_k e^ikx = (W_k + W_k'i) (a_k + a_k'i) [cos(kx) + sin(kx)i]
#         f_c_hat =  torch.einsum("bik,iok->bok", x_c_hat, self.weights_c) - torch.einsum("bik,iok->bok", x_s_hat, self.weights_s)
#         f_s_hat =  torch.einsum("bik,iok->bok", x_s_hat, self.weights_c) + torch.einsum("bik,iok->bok", x_c_hat, self.weights_s)
#         f_0_hat =  torch.einsum("bik,iok->bok", x_0_hat, self.weights_0) 


#         # Return to physical space
#         x = torch.einsum("bok,bkx->box", f_0_hat, bases_0) + 2*torch.einsum("bok,bkx->box", f_c_hat, bases_c) -  2*torch.einsum("bok,bkx->box", f_s_hat, bases_s)
#         # print(x.shape, torch.norm(x))
#         # x += 2*torch.einsum("bok,bkx->box", f_c_hat, bases_c) 
#         # print("fc", torch.norm(x))
#         # x -= 2*torch.einsum("bok,bkx->box", f_s_hat, bases_s)
#         # print("fs", torch.norm(x))       
#         # print(torch.norm(x/nx))
#         return x/nx
    




# class GeoFNO(nn.Module):
#     def __init__(self, ndim,
#                  modes, layers=[128,128,128,128,128],
#                  fc_dim=128,
#                  in_dim=2, out_dim=1,
#                  act='gelu'):
#         super(GeoFNO, self).__init__()

#         """
#         The overall network. It contains several layers of the Fourier layer.
#         1. Lift the input to the desire channel dimension by self.fc0 .
#         2. 4 layers of the integral operators u' = (W + K)(u).
#             W defined by self.w; K defined by self.conv .
#         3. Project from the channel space to the output space by self.fc1 and self.fc2 .

#         input: the solution of the initial condition and location (a(x), x)
#         input shape: (batchsize, x=s, c=2)
#         output: the solution of a later timestep
#         output shape: (batchsize, x=s, c=1)
#         """
#         self.ndim = ndim
#         self.modes = modes
#         self.layers = layers

#         self.fc_dim = fc_dim
        
#         self.fc0 = nn.Linear(in_dim, layers[0])  # input channel is 2: (a(x), x)

#         self.sp_convs = nn.ModuleList([GeoSpectralConv(
#             in_size, out_size, self.modes) for in_size, out_size in zip(layers, layers[1:])])

#         self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
#                                  for in_size, out_size in zip(layers, layers[1:])])
        
#         self.fc1 = nn.Linear(layers[-1], fc_dim)
#         self.fc2 = nn.Linear(fc_dim, out_dim) 
        
#         self.act = _get_act(act)

#     def forward(self, x):
#         """
#         Input shape (of x):     (batch, nx_in,  channels_in)
#         Output shape:           (batch, nx_out, channels_out)
        
#         The input resolution is determined by x.shape[-1], the last dim represent locations
#         The output resolution is determined by self.s_outputspace
#         """
        
#         length = len(self.ws)
#         nx = x.shape[2]

#         aux = x[:,:,-2-self.ndim:].permute(0, 2, 1)    # coord, weights, mask
#         grid, weights, mask = aux[:, 0:self.ndim, :], aux[:, -2, :].unsqueeze(1), aux[:, -1, :].unsqueeze(1)
#         bases_c, bases_s, bases_0 = compute_Fourier_bases(grid, self.modes, mask)
#         wbases_c, wbases_s, wbases_0 = bases_c*(weights*nx), bases_c*(weights*nx), bases_c*(weights*nx)
        
#         x = self.fc0(x[:,:,0:-2])
#         x = x.permute(0, 2, 1)


#         for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            
#             x1 = speconv(x, wbases_c, wbases_s, wbases_0, bases_c, bases_s, bases_0)
  
#             x2 = w(x)

#             x = x1 + x2
#             if self.act is not None and i != length - 1:
#                 x = self.act(x)
    
#         x = x.permute(0, 2, 1)
        
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.fc2(x)
        
#         return x







# # x_train, y_train, x_test, y_test are [n_data, n_x, n_channel] arrays
# def GeoFNO_train(x_train, y_train, x_test, y_test, config, model, save_model_name="./GeoFNO_model"):
#     n_train, n_test = x_train.shape[0], x_test.shape[0]
#     train_rel_l2_losses = []
#     test_rel_l2_losses = []
#     test_l2_losses = []
#     normalization_x, normalization_y = config["train"]["normalization_x"], config["train"]["normalization_y"]
    
    
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
#     if normalization_x:
#         x_normalizer = UnitGaussianNormalizer(x_train, aux_dim = 4) #ndim+2
#         x_train = x_normalizer.encode(x_train)
#         x_test = x_normalizer.encode(x_test)
#         x_normalizer.to(device)
        
#     if normalization_y:
#         y_normalizer = UnitGaussianNormalizer(y_train, aux_dim = 0)
#         y_train = y_normalizer.encode(y_train)
#         y_test = y_normalizer.encode(y_test)
#         y_normalizer.to(device)


#     train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), 
#                                                batch_size=config['train']['batch_size'], shuffle=True)
#     test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), 
#                                                batch_size=config['train']['batch_size'], shuffle=False)
    
    
#     # Load from checkpoint
#     optimizer = Adam(model.parameters(), betas=(0.9, 0.999),
#                      lr=config['train']['base_lr'], weight_decay=config['train']['weight_decay'])
    
#     if config['train']['scheduler'] == "MultiStepLR":
#         scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
#                                                      milestones=config['train']['milestones'],
#                                                      gamma=config['train']['scheduler_gamma'])
#     elif config['train']['scheduler'] == "CosineAnnealingLR":
#         T_max = (config['train']['epochs']//10)*(n_train//config['train']['batch_size'])
#         eta_min  = 0.0
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min = eta_min)
#     elif config["train"]["scheduler"] == "OneCycleLR":
#         scheduler = torch.optim.lr_scheduler.OneCycleLR(
#             optimizer, max_lr=config['train']['base_lr'], 
#             div_factor=2, final_div_factor=100,pct_start=0.2,
#             steps_per_epoch=1, epochs=config['train']['epochs'])
#     else:
#         print("Scheduler ", config['train']['scheduler'], " has not implemented.")

#     model.train()
#     myloss = LpLoss(d=1, p=2, size_average=False)

#     epochs = config['train']['epochs']


#     for ep in range(epochs):
#         train_rel_l2 = 0

#         model.train()
#         for x, y in train_loader:
#             x, y = x.to(device), y.to(device)

#             batch_size_ = x.shape[0]
#             optimizer.zero_grad()
#             out = model(x) #.reshape(batch_size_,  -1)
#             if normalization_y:
#                 out = y_normalizer.decode(out)
#                 y = y_normalizer.decode(y)

#             loss = myloss(out.view(batch_size_,-1), y.view(batch_size_,-1))
#             loss.backward()

#             optimizer.step()
#             train_rel_l2 += loss.item()

#         test_l2 = 0
#         test_rel_l2 = 0
#         with torch.no_grad():
#             for x, y in test_loader:
#                 x, y = x.to(device), y.to(device)
#                 batch_size_ = x.shape[0]
#                 out = model(x) #.reshape(batch_size_,  -1)

#                 if normalization_y:
#                     out = y_normalizer.decode(out)
#                     y = y_normalizer.decode(y)

#                 test_rel_l2 += myloss(out.view(batch_size_,-1), y.view(batch_size_,-1)).item()
#                 test_l2 += myloss.abs(out.view(batch_size_,-1), y.view(batch_size_,-1)).item()




#         scheduler.step()

#         train_rel_l2/= n_train
#         test_l2 /= n_test
#         test_rel_l2/= n_test
        
#         train_rel_l2_losses.append(train_rel_l2)
#         test_rel_l2_losses.append(test_rel_l2)
#         test_l2_losses.append(test_l2)
    

#         if (ep %10 == 0) or (ep == epochs -1):
#             print("Epoch : ", ep, " Rel. Train L2 Loss : ", train_rel_l2, " Rel. Test L2 Loss : ", test_rel_l2, " Test L2 Loss : ", test_l2, flush=True)
#             torch.save(model, save_model_name)
    
    
#     return train_rel_l2_losses, test_rel_l2_losses, test_l2_losses



# if __name__ == "__main__":
    
#     torch.autograd.set_detect_anomaly(True)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     ndim = 2
#     nks = [2,3]
#     Ls = [1.0,1.0]
#     modes = compute_Fourier_modes(ndim, nks, Ls)
#     modes = torch.tensor(modes, dtype=torch.float).to(device)

#     model = GeoFNO(ndim,
#                  modes, layers=[128,128,128,128,128],
#                  fc_dim=128,
#                  in_dim=4-2, out_dim=1,
#                  act='gelu').to(device)
    
#     # print(model)
#     inp = torch.randn(10, 221*51, 4).to(device)
#     out = model(inp)
#     # print(out.shape)
#     # backward check
#     out.sum().backward()
#     print('success!')
    
