import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim


######
#LA-net
######

def FourierOperator(W, b, R_real, R_imag, x):
        #LA-net
        #x [batch_size, nnodes]
        #W,b [1,1]
        #R_real, R_imag [rfft_size]
        W_x = W*x + b
        x_freq = torch.fft.rfft(x, dim=-1)
        
        real_part = x_freq.real  # [batch_size, rfft_size]
        imag_part = x_freq.imag  # [batch_size, rfft_size]
        

        new_real = real_part*R_real  - imag_part*R_imag
        new_imag = imag_part*R_real  + real_part*R_imag  

        new_freq = torch.complex(new_real, new_imag)

        Rx = torch.fft.irfft(new_freq, n=x.shape[-1], dim=-1)
        return Rx + W_x


class LinearModule(nn.Module):
    '''Linear symplectic module.
    '''
    def __init__(self, nnodes, layers):
        super(LinearModule, self).__init__()
        self.nnodes = nnodes
        self.layers = layers
        self.ps = self.__init_params()
        
    
    
    def forward(self, x, t):
        n = self.nnodes
        p , q = x[:,:n], x[:,n:]
        for i in range(self.layers):
            R_real, R_imag = self.ps['R_real{}'.format(i + 1)], self.ps['R_imag{}'.format(i + 1)]
            W, b = self.ps['W{}'.format(i + 1)], self.ps['b{}'.format(i + 1)]

            if i % 2 == 0:
                Rq = FourierOperator(W, b, R_real, R_imag, q)
                p = p + Rq * t.unsqueeze(-1)
            else:
                Rp = FourierOperator(W, b, R_real, R_imag, p)
                q = Rp * t.unsqueeze(-1) + q 
   
   
        return torch.cat([p + t.unsqueeze(1)*self.ps['bp'].unsqueeze(0), q + t.unsqueeze(1)*self.ps['bq'].unsqueeze(0)], dim=-1)
    
    def __init_params(self):
        '''Si is distributed N(0, 0.01), and b is set to zero.
        '''
        rfft_size = self.nnodes//2 + 1 
        params = nn.ParameterDict()
        for i in range(self.layers):
            params['W{}'.format(i + 1)] = nn.Parameter(torch.randn(1, 1))
            params['b{}'.format(i + 1)] = nn.Parameter(torch.randn(1, 1))
            params['R_real{}'.format(i + 1)] = nn.Parameter(torch.randn(rfft_size)*0.1)
            params['R_imag{}'.format(i + 1)] = nn.Parameter(torch.randn(rfft_size)*0.1)
  
        params['bp'] = nn.Parameter(torch.zeros([self.nnodes]).requires_grad_(True))
        params['bq'] = nn.Parameter(torch.zeros([self.nnodes]).requires_grad_(True))

        return params
    
class ActivationModule(nn.Module):
    '''Activation symplectic module.
    '''
    def __init__(self, nnodes, activation, mode):
        super(ActivationModule, self).__init__()
        self.nnodes = nnodes
        self.activation = activation
        self.mode = mode
        self.ps = self.__init_params()
        
    def forward(self, x , t):
        n = self.nnodes
        p , q = x[:,:n], x[:,n:]

        if self.mode == 'up':
            return torch.cat([p + self.act(q) * (t.unsqueeze(1)*self.ps['a'].unsqueeze(0)), q], dim = -1)
        elif self.mode == 'low':
            return torch.cat([p, self.act(p) * (t.unsqueeze(1)*self.ps['a'].unsqueeze(0)) + q], dim = -1)
        else:
            raise ValueError
            
    def __init_params(self):
        
        params = nn.ParameterDict()
        params['a'] = nn.Parameter((torch.randn([self.nnodes])).requires_grad_(True))
        return params
    
    @property
    def act(self):
        if callable(self.activation):
            return self.activation
        elif self.activation == 'sigmoid':
            return torch.sigmoid
        elif self.activation == 'gelu':
            return torch.gelu
        elif self.activation == 'tanh':
            return torch.tanh
        elif self.activation == 'elu':
            return torch.elu
        else:
            raise NotImplementedError


class LASympNet(nn.Module):
    '''LA-SympNet.
    Input: [b_s, nnodes] 
    Output: [b_s, nnodes] 
    '''
    def __init__(self, nnodes, layers=3, sublayers=2, activation='sigmoid'):
        super(LASympNet, self).__init__()
        self.nnodes = nnodes
        self.layers = layers
        self.sublayers = sublayers
        self.activation = activation
        self.ms = self.__init_modules()
        
    def forward(self, x , t):
        n = self.nnodes
        for i in range(self.layers - 1):
            LinM = self.ms['LinM{}'.format(i + 1)]
            ActM = self.ms['ActM{}'.format(i + 1)]
            #y = LinM(x, t)
            x = ActM(LinM(x, t),t)

        return self.ms['LinMout'](x,t)
    
    def __init_modules(self):
        modules = nn.ModuleDict()
        for i in range(self.layers - 1):
            modules['LinM{}'.format(i + 1)] = LinearModule(self.nnodes, self.sublayers)
            mode = 'up' if i % 2 == 0 else 'low'
            modules['ActM{}'.format(i + 1)] = ActivationModule(self.nnodes, self.activation, mode)
        modules['LinMout'] = LinearModule(self.nnodes, self.sublayers)
        return modules
    



####
# G-channels-net
####


def act_F_op(self, R_real, R_imag, W, b, a, x):
    # x: [batch_size, nnodes]
    # R: [1, rfft_size, channels]
    # b: [1, 1, channels] , W: [1, 1, channels]
    # a: [nnodes, diag(channels)]
    x = x.unsqueeze(-1)
    channels = W.shape[-1]
    act = self.act
    W_x = W*x + b
    x_freq = torch.fft.rfft(x, dim=-2)
    real_part = x_freq.real  # [batch_size, rfft_size, 1]
    imag_part = x_freq.imag  # [batch_size, rfft_size, 1]

   
    new_real = real_part*R_real  - imag_part*R_imag #[batch_size, rfft_size, channels]
    new_imag = real_part*R_imag  + imag_part*R_real #[batch_size, rfft_size, channels]

    new_freq = torch.complex(new_real, new_imag)

    Rx = torch.fft.irfft(new_freq, n=x.shape[-1], dim=-2)+ W_x #[batch_size, nnodes, channels]

    Rx = act(Rx) #Rx: [batch_size, nnodes, c]
    
    x1 = torch.einsum('bnc,nc->bnc', Rx, a)

    x1_freq = torch.fft.rfft(x1, dim=-2)

    W_x =  torch.matmul(x1, W.view(1,channels,1)).squeeze(-1) #[batch_size, nnodes]

    real_part1 = x1_freq.real  # [batch_size, rfft_size, c]
    imag_part1 = x1_freq.imag  # [batch_size, rfft_size, c]


    new_real1 = torch.einsum('bsc,sc->bs', real_part1, R_real.squeeze(0))  +  torch.einsum('bsc,sc->bs', imag_part1, R_imag.squeeze(0)) # [batch_size, rfft_size]
    new_imag1 = torch.einsum('bsc,sc->bs', real_part1, R_imag.squeeze(0))  -  torch.einsum('bsc,sc->bs', imag_part1, R_real.squeeze(0)) # [batch_size, rfft_size]


    new_freq1 = torch.complex(new_real1, new_imag1)

    Rx1 = torch.fft.irfft(new_freq1, n=x.shape[-1], dim=-1) + W_x


    return Rx1



class GradientModule(nn.Module):
    '''Gradient symplectic module.
    '''
    def __init__(self, nnodes, activation, mode, channels):
        super(GradientModule, self).__init__()
        self.nnodes = nnodes
        self.activation = activation
        self.mode = mode
        self.channels = channels
        self.ps = self.__init_params()
        
    def forward(self, x, t):
        nnodes = self.nnodes
        p , q = x[:,:nnodes], x[:,nnodes:]
        R_real, R_imag = self.ps['R_real'], self.ps['R_imag']
        W, b, a = self.ps['W'], self.ps['b'], self.ps['a']
        if self.mode == 'up':
            gradH = act_F_op(self, R_real, R_imag, W, b, a, q)
            return torch.cat([p + gradH * t.unsqueeze(-1), q],dim = -1)
        elif self.mode == 'low':
            gradH = act_F_op(self, R_real, R_imag, W, b, a, p)
            return torch.cat([p, gradH * t.unsqueeze(-1) + q], dim = -1)
        else:
            raise ValueError
            
    def __init_params(self):

        params = nn.ParameterDict()
        rfft_size = self.nnodes//2+1
        c = self.channels
        params['R_real'] = nn.Parameter((torch.randn([1, rfft_size, c]) * 0.01).requires_grad_(True))
        params['R_imag'] = nn.Parameter((torch.randn([1, rfft_size, c]) * 0.01).requires_grad_(True))
        params['W'] = nn.Parameter(torch.randn(1, 1, c)* 0.01)
        params['b'] = nn.Parameter(torch.randn(1, 1, c)* 0.01)
        params['a'] = nn.Parameter((torch.randn([self.nnodes, c]) * 0.01).requires_grad_(True))


        return params
    
    @property
    def act(self):
        if callable(self.activation):
            return self.activation
        elif self.activation == 'sigmoid':
            return torch.sigmoid
        elif self.activation == 'gelu':
            return torch.gelu
        elif self.activation == 'tanh':
            return torch.tanh
        elif self.activation == 'elu':
            return torch.elu
        else:
            raise NotImplementedError
    

class GSympNet(nn.Module):
    '''G-SympNet.
    Input: [b_s, nnodes] 
    Output: [b_s, nnodes] 
    '''
    def __init__(self, nnodes, layers=3,  activation='sigmoid', channels = 12):
        super(GSympNet, self).__init__()
        self.nnodes = nnodes
        self.layers = layers
        self.channels = channels
        self.activation = activation
        
        self.ms = self.__init_modules()
        
    def forward(self, x, t):

        for i in range(self.layers):
            GradM = self.ms['GradM{}'.format(i + 1)]
            x = GradM(x,t)
        return x
    
    def __init_modules(self):
        modules = nn.ModuleDict()
        for i in range(self.layers):
            mode = 'up' if i % 2 == 0 else 'low'
            modules['GradM{}'.format(i + 1)] = GradientModule(self.nnodes, self.activation, mode, self.channels)
        return modules
