import math 
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from timeit import default_timer

from fno import LpLoss, FNO1d


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


n_train, n_test = 10000, 500


V_type = "two_mode"
data = np.load("data/schrodinger_"+V_type+"_data.npz")["u_refs"]
in_dim, out_dim = 2, 2
ndata, nT, N = data.shape
N = N//2
nT = nT - 1
data = np.stack([data[..., :N], data[..., N:]], axis=-1) #(ndata, nT+1, N, 2)
X, Y = [], []
for i in list(range(math.ceil(n_train / nT))) + list(range(-math.ceil(n_test / nT), 0)):
    for j in range(nT):
        X.append(data[i,j,...])
        Y.append(data[i,j+1,...])
X, Y = np.array(X), np.array(Y)

    

print("data X shape is ", X.shape)    
    
model = FNO1d([32,32,32,32], width=32,
              layers=[128,128,128,128],
                # [32,32,32,32,32,32], width=32,
                # layers=[64,64,64,64,64,64],
                fc_dim=128,
                in_dim=in_dim, out_dim=out_dim,
                act='gelu',
                pad_ratio=0, 
                cnn_kernel_size=1,
                increment = True).to(device)

X, Y = torch.from_numpy(X.astype(np.float32)).to(device), torch.from_numpy(Y.astype(np.float32)).to(device)


X_train, Y_train  = X[:n_train,...], Y[:n_train,...] 
X_test,  Y_test   = X[-n_test:,...], Y[-n_test:,...] 

batch_size = 8  # 8
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test, Y_test), batch_size=batch_size, shuffle=True)




myloss = LpLoss(p=2, size_average=False)
l2loss = LpLoss(p=2, size_average=False)


epochs = 500
max_lr = 0.001
weight_decay = 5.0e-5
optimizer = optim.Adam(model.parameters(), lr=max_lr, weight_decay=weight_decay)

total_steps = epochs * len(train_loader)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=max_lr,                                                   # peak learning rate (can be tuned)
    steps_per_epoch=len(train_loader),
    epochs = epochs,
    div_factor=2, final_div_factor=100, pct_start=0.2,               # percentage of cycle spent increasing LR
    anneal_strategy='cos'                                            # cosine annealing (default)
)
model.train()
for ep in range(epochs):
    t1 = default_timer()
    train_rel_l2 = 0
    test_rel_l2 = 0
    model.train()
    for x, y in train_loader:
        batch_size_ = x.shape[0]
        optimizer.zero_grad()
        out = model(x)
        loss = myloss(out.view(batch_size_,-1), y.view(batch_size_,-1))
        loss.backward()

        optimizer.step()
        scheduler.step()
        train_rel_l2 += loss.item()
    
    for x, y in test_loader:
        batch_size_ = x.shape[0]
        out = model(x)
        loss = l2loss(out.view(batch_size_,-1), y.view(batch_size_,-1))
        test_rel_l2 += loss.item()

    train_rel_l2 /= n_train 
    test_rel_l2  /= n_test
    t2 = default_timer()
    print("Epoch : ", ep, " Time: ", round(t2-t1,3), " Train L2 Loss : ", train_rel_l2, " Test L2 Loss : ", test_rel_l2 ,flush=True)
    if (ep %100 == 99) or ep == epochs - 1 : 
            torch.save(model.state_dict(), "pth/FNO_"+V_type+".pth")
