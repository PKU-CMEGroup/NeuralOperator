import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .basics import SpectralConv1d
from .utils import _get_act, add_padding, remove_padding

from .adam import Adam
from .losses import LpLoss
from .normalizer import UnitGaussianNormalizer


class FNN1d(nn.Module):
    def __init__(self,
                 modes, width=32,
                 layers=None,
                 fc_dim=128,
                 in_dim=2, out_dim=1,
                 act='relu',
                 pad_ratio=0):
        super(FNN1d, self).__init__()

        """
        The overall network. It contains several layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        if layers is None:
            layers = [width] * 4
        self.pad_ratio = pad_ratio
        self.fc_dim = fc_dim
        
        self.fc0 = nn.Linear(in_dim, layers[0])  # input channel is 2: (a(x), x)

        self.sp_convs = nn.ModuleList([SpectralConv1d(
            in_size, out_size, num_modes) for in_size, out_size, num_modes in zip(layers, layers[1:], self.modes1)])

        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(layers, layers[1:])])
        
        # if fc_dim = 0, we do not have nonlinear layer
        if fc_dim > 0:
            self.fc1 = nn.Linear(layers[-1], fc_dim)
            self.fc2 = nn.Linear(fc_dim, out_dim) 
        else:
            self.fc2 = nn.Linear(layers[-1], out_dim)
            
        self.act = _get_act(act)

    def forward(self, x):
        """
        Input shape (of x):     (batch, nx_in,  channels_in)
        Output shape:           (batch, nx_out, channels_out)
        
        The input resolution is determined by x.shape[-1]
        The output resolution is determined by self.s_outputspace
        """
        
        length = len(self.ws)
        
        
        
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        pad_nums = [math.floor(self.pad_ratio * x.shape[-1])]
        
        # add padding
        x = add_padding(x, pad_nums=pad_nums)

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x)
            x = x1 + x2
            if self.act is not None and i != length - 1:
                x = self.act(x)
                
        # remove padding
        x = remove_padding(x, pad_nums=pad_nums)
        
        x = x.permute(0, 2, 1)
        
        # if fc_dim = 0, we do not have nonlinear layer
        fc_dim = self.fc_dim if hasattr(self, 'fc_dim') else 1
        
        if fc_dim > 0:
            x = self.fc1(x)
            if self.act is not None:
                x = self.act(x)
            
        x = self.fc2(x)
        
        
        return x


    
    
def FNN1d_cost(Nx, config):
    pad_ratio = config['model']['pad_ratio']
    modes = config['model']['modes']
    layers = config['model']['layers']
    fc_dim = config['model']['fc_dim']
    in_dim = config['model']['in_dim']
    out_dim = config['model']['out_dim']
    Np = Nx + math.floor(pad_ratio * Nx)
    
    cost_act = 1
    # lifting operator
    cost = 2*Np*in_dim*layers[0]
    for (i, mode) in enumerate(modes):
        df_in, df_out = layers[i], layers[i+1]
        # fourier series transform, inverse fourier series transform, linear
        cost += df_in*5*Np*math.log(Np)+df_out*5*Np*math.log(Np)+mode*df_out*(2*df_in-1) 
        # activation function
        if i != len(modes)-1:
            cost += df_out*Np*cost_act
            
    # project operator
    # if fc_dim = 0, we do not have nonlinear layer
    if fc_dim > 0:
        cost += Np*fc_dim*2*layers[-1] + fc_dim*Np*cost_act + Np*out_dim*2*fc_dim 
    else:
        cost += Np*out_dim*2*layers[-1]
        
    return cost
    

# # x_train, y_train, x_test, y_test are [n_data, n_x, n_channel] arrays
# def FNN1d_train(x_train, y_train, x_test, y_test, config, save_model_name="./FNO_model"):
#     n_train, n_test = x_train.shape[0], x_test.shape[0]
#     train_rel_l2_losses = []
#     test_rel_l2_losses = []
#     test_l2_losses =[]
#     cost = FNN1d_cost(x_train.shape[1], config)
    
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     normalization, normalization_dim = config["train"]["normalization"], config["train"]["normalization_dim"]
#     if normalization:
#         x_normalizer = UnitGaussianNormalizer(x_train, dim=normalization_dim)
#         x_train = x_normalizer.encode(x_train)
#         x_test = x_normalizer.encode(x_test)
#         x_normalizer.to(device)

#         y_normalizer = UnitGaussianNormalizer(y_train, dim=normalization_dim)
#         y_train = y_normalizer.encode(y_train)
#         y_test = y_normalizer.encode(y_test)
#         y_normalizer.to(device)


#     train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), 
#                                                batch_size=config['train']['batch_size'], shuffle=True)
#     test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), 
#                                                batch_size=config['train']['batch_size'], shuffle=False)

#     model = FNN1d(modes=config['model']['modes'],
#                   fc_dim=config['model']['fc_dim'],
#                   layers=config['model']['layers'],
#                   in_dim=config['model']['in_dim'], 
#                   out_dim=config['model']['out_dim'],
#                   act=config['model']['act'],
#                   pad_ratio=config['model']['pad_ratio']).to(device)
#     # Load from checkpoint
#     optimizer = Adam(model.parameters(), betas=(0.9, 0.999),
#                      lr=config['train']['base_lr'])
#     scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
#                                                      milestones=config['train']['milestones'],
#                                                      gamma=config['train']['scheduler_gamma'])

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
#             if normalization:
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

#                 if normalization:
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
#             print("Epoch : ", ep, " Rel. Train L2 Loss : ", train_rel_l2, " Rel. Test L2 Loss : ", test_rel_l2, " Test L2 Loss : ", test_l2)
    

#     torch.save(model, save_model_name)
    
    
#     return train_rel_l2_losses, test_rel_l2_losses, test_l2_losses, cost