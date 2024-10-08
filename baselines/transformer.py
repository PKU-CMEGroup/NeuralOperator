import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math, sys

sys.path.append("../")
from models.adam import Adam
from models.losses import LpLoss
from models.normalizer import UnitGaussianNormalizer





def mask_(matrices, maskval=0.0, mask_diagonal=True, offset = 0):
    """
    Masks out all values in the given batch of matrices where i <= j holds,
    i < j if mask_diagonal is false

    In place operation

    :param tns:
    :return:
    """
    # mask_diagonal = True
    # t = 7
    # matrices = torch.ones(2,t,t)
    # offset = 0
    h, w = matrices.size(-2), matrices.size(-1)

    indices = torch.triu_indices(h, w, offset=offset if mask_diagonal else offset+1)
    matrices[..., indices[0], indices[1]] = maskval

    # bigmatrices = torch.ones(2,t,t)
    # blocksize = 3
    # for i in range(t//blocksize + 1):
    #     matrices = bigmatrices[:,i*blocksize:min((i+1)*blocksize,t),:]
    #     offset = i*blocksize
    #     h, w = matrices.size(-2), matrices.size(-1)

    #     indices = torch.triu_indices(h, w, offset=offset if mask_diagonal else offset+1)
    #     print(indices)
    #     matrices[..., indices[0], indices[1]] = maskval


class SelfAttention(nn.Module):
    """
    Canonical implementation of multi-head self attention.
    """

    def __init__(self, emb, blocksize=128, heads=8, mask=False, kqnorm=False, scalefactor=None):
        """

        :param emb: The dimension of the input and output vectors.
        :param heads: The number of heads (parallel executions of the self-attention)
        :param mask: Whether to apply an autoregressive mask.
        :param kqnorm: Whether to apply layer normalization to the keys and queries.
        :param scalefactor: Multiplier for the attention weights. If none, the default `1/sqrt(emb/heads)` is used,
        """

        super().__init__()

        assert emb % heads == 0, f'Embedding dimension ({emb}) should be divisible by nr. of heads ({heads})'

        self.emb = emb
        self.blocksize = blocksize
        self.heads = heads
        self.mask = mask

        s = emb // heads
        # - We will break the embedding into `heads` chunks and feed each to a different attention head

        self.tokeys    = nn.Linear(emb, emb, bias=False)
        self.toqueries = nn.Linear(emb, emb, bias=False)
        self.tovalues  = nn.Linear(emb, emb, bias=False)

        self.unifyheads = nn.Linear(emb, emb)

        self.kqnorm = kqnorm
        if kqnorm:
            self.kln = nn.LayerNorm([s])
            self.qln = nn.LayerNorm([s])

        self.scalefactor = 1/math.sqrt(emb // heads) if scalefactor is None else scalefactor

    def forward(self, x):

        b, t, e = x.size()
        h = self.heads
        assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'

        s = e // h

        keys    = self.tokeys(x)
        queries = self.toqueries(x)
        values  = self.tovalues(x)

        keys    = keys.view(b, t, h, s)
        queries = queries.view(b, t, h, s)
        values  = values.view(b, t, h, s)

        if self.kqnorm:
            keys = self.kln(keys)
            queries = self.qln(queries)

        # -- We first compute the k/q/v's on the whole embedding vectors, and then split into the different heads.
        #    See the following video for an explanation: https://youtu.be/KmAISyVvE1Y

        # Compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
        values = values.transpose(1, 2).contiguous().view(b * h, t, s)

        queries = queries
        keys    = keys



        out = torch.zeros(b,h,t,s).to(x.get_device())
        # - get dot product of queries and keys, and scale
        for i in range(t//self.blocksize + 1):
            dot = torch.bmm(queries[:,i*self.blocksize:min((i+1)*self.blocksize,t), :], keys.transpose(1, 2))
            dot = dot * self.scalefactor
        
            if self.mask: # mask out the upper half of the dot matrix, excluding the diagonal
                mask_(dot, maskval=float('-inf'), mask_diagonal=False, offset=i*self.blocksize)
            dot = F.softmax(dot, dim=2)
            # -- dot now has row-wise self-attention probabilities

            # apply the self attention to the values
            out[:,:,i*self.blocksize:min((i+1)*self.blocksize,t),:] = torch.bmm(dot, values).view(b, h, -1, s)


        # # - get dot product of queries and keys, and scale
        # dot = torch.bmm(queries, keys.transpose(1, 2))
        # dot = dot * self.scalefactor

        # assert dot.size() == (b*h, t, t)

        # if self.mask: # mask out the upper half of the dot matrix, excluding the diagonal
        #     mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        # dot = F.softmax(dot, dim=2)
        # # -- dot now has row-wise self-attention probabilities

        # # apply the self attention to the values
        # out = torch.bmm(dot, values).view(b, h, t, s)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, s * h)

        return self.unifyheads(out)



class TransformerBlock(nn.Module):
    """
    A straightforward transformer block.
    """

    def __init__(self, emb, blocksize, heads, mask, ff_hidden_mult=4, dropout=0.0,
                 pos_embedding=None, sa_kwargs={}):
        super().__init__()

        
        self.attention = SelfAttention(emb, blocksize=blocksize, heads=heads, mask=mask, **sa_kwargs)

        self.mask = mask
        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(

            nn.Linear(emb, ff_hidden_mult * emb),
            #nn.ReLU(),
            nn.GELU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )

        self.do = nn.Dropout(dropout)

    def forward(self, x):
        attended = self.attention(x)

        x = self.norm1(attended + x)

        x = self.do(x)

        fedforward = self.ff(x)

        x = self.norm2(fedforward + x)

        x = self.do(x)

        return x
    

class Transformer(nn.Module):
    """
    Transformer for generating text (character by character).
    """

    def __init__(self, in_channels, out_channels, hid_channels, blocksize, heads, depth):

        super().__init__()
        self.en_fc1 = nn.Linear(in_channels, hid_channels)
        

        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(emb=hid_channels, blocksize=blocksize, heads=heads, mask=True))

        self.tblocks = nn.Sequential(*tblocks)
        # self.tblocks = tblocks
        self.de_fc1 = nn.Linear(hid_channels, hid_channels)
        self.de_fc2 = nn.Linear(hid_channels, out_channels)


    def forward(self, x):
        """
        :param x: A (batch, sequence length) integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        x = self.en_fc1(x)  # (batch_size, n_x, n_channels)
        x = F.gelu(x)

        x = self.tblocks(x)
        # for i in range(len(self.tblocks)):
        #     x = self.tblocks[i](x)

        x = self.de_fc1(x)
        x = F.gelu(x)
        x = self.de_fc2(x)

        return x


# x_train, y_train, x_test, y_test are [n_data, n_x, n_channel] arrays
def Transformer_train(x_train, y_train, x_test, y_test, config, model, save_model_name="./Transformer_model"):
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



if __name__=="__main__":
    in_channels, out_channels, hid_channels, blocksize, heads, depth = 3, 1, 64, 128, 8, 4
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    x = torch.randn(8, 1023, 3).to(device) # (batch_size, n_x, n_channels)
    
    model = Transformer(in_channels, out_channels, hid_channels, blocksize, heads, depth).to(device)
    
    y0 = model(x) 

