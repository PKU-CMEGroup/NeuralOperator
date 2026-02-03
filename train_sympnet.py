import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from timeit import default_timer

from model import GSympNet, LASympNet



data = np.load('data/0.2_1.npz')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


model = LASympNet(nnodes=100, layers = 5, activation='sigmoid', sublayers=5).to(device)

X, T, Y = data['X'], data['T'], data['Y']
X, T, Y = torch.from_numpy(X.astype(np.float32)).to(device), torch.from_numpy(T.astype(np.float32)).to(device), \
torch.from_numpy(Y.astype(np.float32)).to(device)

n_train, n_test = 2000,200
X_train, T_train, Y_train = X[:n_train,:], T[:n_train ], Y[:n_train,: ]
X_test, T_test, Y_test = X[-n_test:,: ], T[-n_test: ], Y[-n_test:,: ]

batch_size = 8
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, T_train, Y_train), batch_size=8, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test, T_test, Y_test), batch_size=8, shuffle=True)


model.train()

myloss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5*1.0e-5)

for ep in range(1000):
    t1 = default_timer()
    train_rel_l2 = 0
    test_rel_l2 = 0
    model.train()
    for x, t, y in train_loader:
        batch_size_ = x.shape[0]
        optimizer.zero_grad()
        out = model(x,t)
        loss = myloss(out.view(batch_size_,-1), y.view(batch_size_,-1))
        loss.backward()

        optimizer.step()
        train_rel_l2 += loss.item()
    
    for x, t, y in test_loader:
        batch_size_ = x.shape[0]
        
        out = model(x,t)
        loss = myloss(out.view(batch_size_,-1), y.view(batch_size_,-1))
        test_rel_l2 += loss.item()

    train_rel_l2/= 2000
    test_rel_l2/= 200
    t2 = default_timer()
    #print("Epoch : ", ep, " Time: ", round(t2-t1,3), " Rel. Train L2 Loss : ", train_rel_l2, flush=True)

    print("Epoch : ", ep, " Time: ", round(t2-t1,3), " Train L2 Loss : ", train_rel_l2, " Test L2 Loss : ", test_rel_l2 ,flush=True)
    if (ep %100 == 99) : 

            torch.save(model.state_dict(), "pth/LA_net.pth")
