import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.adam import Adam
from models.losses import LpLoss
from models.normalizer import UnitGaussianNormalizer



def pairwise_dist(res1x, res1y, res2x, res2y):
    gridx1 = torch.linspace(0, 1, res1x+1)[:-1].view(1, -1, 1).repeat(res1y, 1, 1)
    gridy1 = torch.linspace(0, 1, res1y+1)[:-1].view(-1, 1, 1).repeat(1, res1x, 1)
    grid1 = torch.cat([gridx1, gridy1], dim=-1).view(res1x*res1y, 2)
    
    gridx2 = torch.linspace(0, 1, res2x+1)[:-1].view(1, -1, 1).repeat(res2y, 1, 1)
    gridy2 = torch.linspace(0, 1, res2y+1)[:-1].view(-1, 1, 1).repeat(1, res2x, 1)
    grid2 = torch.cat([gridx2, gridy2], dim=-1).view(res2x*res2y, 2)
    
    grid1 = grid1.unsqueeze(1).repeat(1, grid2.shape[0], 1)
    grid2 = grid2.unsqueeze(0).repeat(grid1.shape[0], 1, 1)
    
    dist = torch.norm(grid1 - grid2, dim=-1)
    return (dist**2 / 2.0).float()



class MLP(nn.Module):
    '''
    A two-layer MLP with GELU activation.
    '''
    def __init__(self, in_channels, hid_channels, out_channels):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_channels, hid_channels)
        self.fc2 = nn.Linear(hid_channels, out_channels)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        return self.fc2(x)

class MultiHeadPosAtt(nn.Module):
    '''
    Global, local and cross variants of the multi-head position-attention mechanism.
    '''
    def __init__(self, n_head, hid_channels, locality):
        super(MultiHeadPosAtt, self).__init__()
        self.locality = locality
        self.hid_channels = hid_channels
        self.n_head = n_head
        self.v_dim = hid_channels // n_head
        self.r = nn.Parameter(torch.randn(n_head, 1, 1))
        self.weight = nn.Parameter(torch.randn(n_head, hid_channels, self.v_dim))

    def forward(self, m_dist, x):
        scaled_dist = m_dist * torch.tan(0.25 * np.pi * (1 - 1e-7) * (1 + torch.sin(self.r)))
        if self.locality <= 100:
            mask = torch.quantile(scaled_dist, self.locality / 100.0, dim=-1, keepdim=True)
            scaled_dist = torch.where(scaled_dist <= mask, scaled_dist, torch.tensor(np.inf))

        att = F.softmax(-scaled_dist, dim=-1)

        value = torch.einsum('bnj,hjk->bhnk', x, self.weight)
        # h : head; 
        # j : number of points
        # k : number of hidden channels 
        # combine k and h
        concat = torch.einsum('hjn,bhnk->bhjk', att, value).permute(0, 2, 1, 3)
        concat = concat.reshape(concat.shape[0], concat.shape[1], -1)
        return F.gelu(concat)




class PiT(nn.Module):
    '''
    Position-induced Transformer, built upon the multi-head position-attention mechanism.
    '''
    def __init__(self, in_channels, out_channels, hid_channels, n_head, localities, m_dists):
        super(PiT, self).__init__()
        self.out_channels = out_channels
        self.hid_channels = hid_channels
        self.n_head = n_head
        self.localities = localities[1:-1]
        en_locality, de_locality = localities[0], localities[-1]
        self.n_blocks = len(localities) - 2
        self.m_dists = m_dists

        # Encoder
        self.en_fc1 = nn.Linear(in_channels, hid_channels)
        self.down     = MultiHeadPosAtt(n_head, hid_channels, locality=en_locality)
        
        # Processor
        self.PA = nn.ModuleList([MultiHeadPosAtt(n_head, hid_channels, locality) for locality in localities])
        self.MLP = nn.ModuleList([MLP(hid_channels, hid_channels, hid_channels) for _ in range(self.n_blocks)])
        self.W = nn.ModuleList([nn.Linear(hid_channels, hid_channels) for _ in range(self.n_blocks)])

        # Decoder
        self.up     = MultiHeadPosAtt(n_head, hid_channels, locality=de_locality)
        self.de_fc1 = nn.Linear(hid_channels, hid_channels)
        self.de_fc2 = nn.Linear(hid_channels, out_channels)

    def forward(self, x):
        x    = self.en_fc1(x)  # (batch_size, n_x, n_channels)
        x = F.gelu(x)
        x = self.down(self.m_dists[0], x)

        # Processor
        for i in range(self.n_blocks):
            x = self.MLP[i](self.PA[i](self.m_dists[i+1], x)) + self.W[i](x)
            x = F.gelu(x)

        x = self.up(self.m_dists[-1], x)
        x = self.de_fc1(x)
        x = F.gelu(x)
        x = self.de_fc2(x)
        return x





# x_train, y_train, x_test, y_test are [n_data, n_x, n_channel] arrays
def PiT_train(x_train, y_train, x_test, y_test, config, model, save_model_name="./Pit_model"):
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
            print("Epoch : ", ep, " Rel. Train L2 Loss : ", train_rel_l2, " Rel. Test L2 Loss : ", test_rel_l2, " Test L2 Loss : ", test_l2)
            torch.save(model, save_model_name)
    
    
    return train_rel_l2_losses, test_rel_l2_losses, test_l2_losses
