import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
sys.path.append("../")
from models.adam import Adam
from models.losses import LpLoss
from models.normalizer import UnitGaussianNormalizer



import torch
import torch.nn as nn

def conv(in_planes, output_channels, kernel_size, stride, dropout_rate):
    return nn.Sequential(
        nn.Conv2d(in_planes, output_channels, kernel_size=kernel_size,
                  stride=stride, padding=(kernel_size - 1) // 2, bias = False),
    )

def deconv(input_channels, output_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(input_channels, output_channels, kernel_size=4,
                           stride=2, padding=1),
    )

def output_layer(input_channels, output_channels, kernel_size, stride, dropout_rate):
    return nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size,
                     stride=stride, padding=(kernel_size - 1) // 2)

class UNet(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, dropout_rate):
        super(UNet, self).__init__()
        self.input_channels = input_channels
        self.conv1 = conv(input_channels, 64, kernel_size=kernel_size, stride=2, dropout_rate = dropout_rate)
        self.conv2 = conv(64, 128, kernel_size=kernel_size, stride=2, dropout_rate = dropout_rate)
        self.conv3 = conv(128, 256, kernel_size=kernel_size, stride=2, dropout_rate = dropout_rate)
        self.conv3_1 = conv(256, 256, kernel_size=kernel_size, stride=1, dropout_rate = dropout_rate)
        self.conv4 = conv(256, 512, kernel_size=kernel_size, stride=2, dropout_rate = dropout_rate)
        self.conv4_1 = conv(512, 512, kernel_size=kernel_size, stride=1, dropout_rate = dropout_rate)
        self.conv5 = conv(512, 1024, kernel_size=kernel_size, stride=2, dropout_rate = dropout_rate)
        self.conv5_1 = conv(1024, 1024, kernel_size=kernel_size, stride=1, dropout_rate = dropout_rate)

        self.deconv4 = deconv(1024, 256)
        self.deconv3 = deconv(768, 128)
        self.deconv2 = deconv(384, 64)
        self.deconv1 = deconv(192, 32)
        self.deconv0 = deconv(96, 16)
    
        self.output_layer = output_layer(16 + input_channels, output_channels, 
                                         kernel_size=kernel_size, stride=1, dropout_rate = dropout_rate)


    def forward(self, x):

        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))

        out_deconv4 = self.deconv4(out_conv5)
        concat4 = torch.cat((out_conv4, out_deconv4), 1)
        out_deconv3 = self.deconv3(concat4)
        concat3 = torch.cat((out_conv3, out_deconv3), 1)
        out_deconv2 = self.deconv2(concat3)
        concat2 = torch.cat((out_conv2, out_deconv2), 1)
        out_deconv1 = self.deconv1(concat2)
        concat1 = torch.cat((out_conv1, out_deconv1), 1)
        out_deconv0 = self.deconv0(concat1)
        concat0 = torch.cat((x, out_deconv0), 1)
        out = self.output_layer(concat0)

        return out

  

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    model = UNet(input_channels=1, output_channels=1, kernel_size=3, dropout_rate=0.5).cuda()
    inp = torch.randn(10, 1, 63, 63).cuda()
    out = model(inp)
    print(out.shape)
    summary(model, input_size=(10, 1, 63, 63))
    # backward check
    out.sum().backward()
    print('success!')
    
    # print(model)
    




class MgIte(nn.Module):
    def __init__(self, A, S):
        super().__init__()
 
        self.A = A
        self.S = S

    def forward(self, out):
        
        if isinstance(out, tuple):
            u, f = out
            u = u + (self.S(f-self.A(u)))  
        else:
            f = out
            u = self.S(f)

        out = (u, f)
        return out

class MgIte_init(nn.Module):
    def __init__(self, S):
        super().__init__()
        
        self.S = S

    def forward(self, f):
        u = self.S(f)
        return (u, f)

class Restrict(nn.Module):
    def __init__(self, Pi=None, R=None, A=None):
        super().__init__()
        self.Pi = Pi
        self.R = R
        self.A = A
    def forward(self, out):
        u, f = out
        if self.A is not None:
            f = self.R(f-self.A(u))
        else:
            f = self.R(f)
        u = self.Pi(u)                              
        out = (u,f)
        return out



class MgNO(nn.Module):
    def __init__(self, input_shape, num_layer, num_channel_u, num_channel_f, num_classes, num_iteration, 
    in_chans=1,  normalizer=None, output_dim=1, activation='gelu', padding_mode='zeros', ):
        super().__init__()
        self.num_layer = num_layer
        self.num_channel_u = num_channel_u
        self.num_channel_f = num_channel_f
        self.num_classes = num_classes
        self.num_iteration = num_iteration

        self.conv_list = nn.ModuleList([])
        self.linear_list = nn.ModuleList([])
        self.linear_list.append(nn.Conv2d(num_channel_f, num_channel_u, kernel_size=1, stride=1, padding=0, bias=True))   
        self.conv_list.append(MgConv(input_shape, num_iteration, num_channel_u, num_channel_f, padding_mode=padding_mode))   
        for _ in range(num_layer-1):
            self.conv_list.append(MgConv(input_shape, num_iteration, num_channel_u, num_channel_u, padding_mode=padding_mode)) 
            self.linear_list.append(nn.Conv2d(num_channel_u, num_channel_u, kernel_size=1, stride=1, padding=0, bias=True))
   
        self.linear = nn.Conv2d(num_channel_u, 1, kernel_size=1, bias=False)
        self.normalizer = normalizer

        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'silu':
            self.act = nn.SiLU()
        else: raise NameError('invalid activation') 
        
    def forward(self, u):
     
        for i in range(self.num_layer):
            u = self.act(self.conv_list[i](u) + self.linear_list[i](u))
        u = self.normalizer.decode(self.linear(u)) if self.normalizer else self.linear(u)
        return u 




class MgConv(nn.Module):
    def __init__(self, input_shape, num_iteration, num_channel_u, num_channel_f, padding_mode='zeros', bias=False, use_res=False):
        super().__init__()
        self.num_iteration = num_iteration
        self.num_channel_u = num_channel_u
        self.padding_mode = padding_mode
        self.RTlayers = nn.ModuleList()   
        for j in range(len(num_iteration)-1):
            kernel_size = [4-input_shape[0]%2, 4-input_shape[1]%2]  # odd=>3 even=>4
            self.RTlayers.append(nn.ConvTranspose2d(num_channel_u, num_channel_u, kernel_size=kernel_size, stride=2, padding=1, bias=False))
            input_shape = [(input_shape[0] + 2 - 1)//2, (input_shape[1] + 2  - 1) //2]

        layers = []
        for l, num_iteration_l in enumerate(num_iteration): #l: l-th layer.   num_iteration_l: the number of iterations of l-th layer
            for i in range(num_iteration_l[0]):
                S = nn.Conv2d(num_channel_f, num_channel_u, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode=padding_mode)
                if l==0 and i==0:
                    layers.append(MgIte_init(S))
                else:
                    A = nn.Conv2d(num_channel_u, num_channel_f, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode=padding_mode)
                    layers.append(MgIte(A, S))
            setattr(self, 'layer'+str(l), nn.Sequential(*layers))

            if l < len(num_iteration)-1:
                A = nn.Conv2d(num_channel_u, num_channel_f, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode=padding_mode)
                Pi= nn.Conv2d(num_channel_u, num_channel_u, kernel_size=3, stride=2, padding=1, bias=False, padding_mode=padding_mode)
                R = nn.Conv2d(num_channel_f, num_channel_f, kernel_size=3, stride=2, padding=1, bias=False, padding_mode=padding_mode)
                if use_res:
                    layers= [Restrict(Pi, R, A)]
                else:
                    layers= [Restrict(Pi=Pi, R=R)]
     
    def forward(self, f):
        out_list = [0] * len(self.num_iteration)
        out = f 

        for l in range(len(self.num_iteration)):
            
            out = getattr(self, 'layer'+str(l))(out) 

            out_list[l] = out
        # upblock                                 
        for j in range(len(self.num_iteration)-2,-1,-1):
            # f is not used
            u, f = out_list[j][0], out_list[j][1]
            u_post = u + self.RTlayers[j](out_list[j+1][0])
            out = (u_post, f)
            out_list[j] = out
            
        return out_list[0][0]



# x_train, y_train, x_test, y_test are [n_data, n_x, n_channel] arrays
def MgNO_train(x_train, y_train, x_test, y_test, config, model, save_model_name="./MgNO_model"):
    n_train, n_test = x_train.shape[0], x_test.shape[0]
    train_rel_l2_losses = []
    test_rel_l2_losses = []
    test_l2_losses = []
    normalization_x, normalization_y, normalization_dim = config["train"]["normalization_x"], config["train"]["normalization_y"], config["train"]["normalization_dim"]
    dim = len(x_train.shape) - 2 # n_train, size, n_channel
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    if normalization_x:
        x_normalizer = UnitGaussianNormalizer(x_train, dim=normalization_dim)
        x_train = x_normalizer.encode(x_train)
        x_test = x_normalizer.encode(x_test)
        x_normalizer.to(device)
        
    if normalization_y:
        y_normalizer = UnitGaussianNormalizer(y_train, dim=normalization_dim)
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






if __name__ == "__main__":
    
    torch.autograd.set_detect_anomaly(True)
    model = MgNO(num_layer=5, num_channel_u=24, 
             num_channel_f=4, num_classes=1, 
             num_iteration=[[10,0], [10,0], [10,0], [10,0], [20,0]]).cuda()
    
    print(model)
    inp = torch.randn(10, 4, 221, 51).cuda()
    out = model(inp)
    # print(out.shape)
    # backward check
    out.sum().backward()
    print('success!')
    

