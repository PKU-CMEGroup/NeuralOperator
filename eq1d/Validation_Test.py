import random
import torch
import sys
import numpy as np
import math
torch.manual_seed(0)
np.random.seed(0)


sys.path.append('../')
from models import FNN1d, FNN1d_train, FNN1d_cost, UnitGaussianNormalizer, LpLoss


def test(x_train, y_train, x_test, y_test, model_prefix, config, downsample_ratio, n_fno_layers, k_max, d_f):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    normalization, dim = config["train"]["normalization"], config["train"]["dim"]
    
    n_data = x_train.shape[0]
    
    # load model
    # error mean/covariance on validataion set
    # error mean/covariance on test set 
    setup_info="n_data_"+str(n_data)+"_k_max_"+str(k_max)+"_downsample_ratio_"+str(downsample_ratio)+"_n_fno_layers_"+str(n_fno_layers)+"_d_f_"+str(d_f)
    model = torch.load(model_prefix+setup_info, map_location=device)


    n_test = x_test.shape[0]

    test_rel_l2_losses = []
    test_l2_losses =[]
    
    
    if normalization:
        x_normalizer = UnitGaussianNormalizer(x_train, dim=dim)
        x_train = x_normalizer.encode(x_train)
        x_test = x_normalizer.encode(x_test)
        x_normalizer.to(device)

        y_normalizer = UnitGaussianNormalizer(y_train, dim=dim)
        y_train = y_normalizer.encode(y_train)
        y_test = y_normalizer.encode(y_test)
        y_normalizer.to(device)



    myloss = LpLoss(d=1, p=2, size_average=False)
    
    test_l2 = np.zeros(n_test)
    test_rel_l2 = np.zeros(n_test)
    
    for i in range(n_test):
        x, y = x_test[i:i+1,:, :], y_test[i:i+1,:, :]
        x, y = x.to(device), y.to(device)
        
        out = model(x) #.reshape(1,  -1)

        if normalization:
            out = y_normalizer.decode(out)
            y = y_normalizer.decode(y)

        test_rel_l2[i] = myloss(out.view(1,-1), y.view(1,-1)).item()
        test_l2[i] = myloss.abs(out.view(1,-1), y.view(1,-1)).item()

    test_l2_mean, test_l2_cov = np.mean(test_l2), np.cov(test_l2)
    test_rel_l2_mean, test_rel_l2_cov = np.mean(test_rel_l2), np.cov(test_rel_l2)

    print(setup_info, test_l2_mean, test_l2_cov, test_rel_l2_mean, test_rel_l2_cov)
    return test_l2_mean, test_l2_cov, test_rel_l2_mean, test_rel_l2_cov
             



prefix = "/central/groups/esm/dzhuang/cost-accuracy-data/"
x_data = np.load(prefix+"darcy_a.npy")
y_data = np.load(prefix+"darcy_u.npy")
model_prefix = "/central/groups/esm/dzhuang/cost-accuracy-data/models/darcy_FNO_0_"

n_data_array = [256, 512, 1024, 2048, 4096, 8192, 16384]
k_max_array = [16, 32, 64, 128]
d_f_array = [16, 32, 64, 128]
n_fno_layers_array = [3, 4, 5]
downsample_ratio_array = [1, 2, 4, 8]


#optimization
epochs = 1001
base_lr = 0.001
milestones = [200, 300, 400, 500, 800, 900]
scheduler_gamma = 0.5
batch_size=32
normalization = True
dim = []

M = 2**15
L, Ne_ref = 1.0, 2**12
n_test = 8192


n_test_sets = 2
data_analysis = np.zeros((len(n_data_array)*len(downsample_ratio_array)*len(n_fno_layers_array)*len(k_max_array)*len(d_f_array), 4*n_test_sets + 6)) 

i_data_analysis = 0

for n_train in n_data_array:
    for downsample_ratio in downsample_ratio_array:
        for n_fno_layers in n_fno_layers_array:
            for k_max in k_max_array:
                for d_f in d_f_array: 
                    for i_test_set in range(n_test_sets):
                        
                        
                        Ne = Ne_ref//downsample_ratio
                        grid = np.linspace(0, L, Ne+1)
                        
                        x_train = torch.from_numpy(np.stack((x_data[0:n_train, 0::downsample_ratio], np.tile(grid, (n_train,1))), axis=-1).astype(np.float32))
                        y_train = torch.from_numpy(y_data[0:n_train, 0::downsample_ratio, np.newaxis].astype(np.float32))
                        # x_train, y_train are [n_data, n_x, n_channel] arrays
                        x_test = torch.from_numpy(np.stack((x_data[M//2+(i_test_set)*n_test:M//2+(i_test_set+1)*n_test, 0::downsample_ratio], np.tile(grid, (n_test,1))), axis=-1).astype(np.float32))
                        y_test = torch.from_numpy(y_data[M//2+(i_test_set)*n_test:M//2+(i_test_set+1)*n_test, 0::downsample_ratio, np.newaxis].astype(np.float32))
                        
                        
                        modes = [k_max] * n_fno_layers
                        # channel d_f
                        layers = [d_f] * (n_fno_layers + 1)
                        fc_dim = d_f
                        in_dim = 2
                        out_dim = 1
                        act = "gelu"
                        pad_ratio = 0.05
                        config = {"model" : {"modes": modes, "fc_dim": fc_dim, "layers": layers, "in_dim": in_dim, "out_dim":out_dim, "act": act, "pad_ratio":pad_ratio},
                                  "train" : {"base_lr": base_lr, "epochs": epochs, "milestones": milestones, "scheduler_gamma": scheduler_gamma, "batch_size": batch_size, 
                                            "normalization": normalization, "dim": dim}}

                        
                        data_analysis[i_data_analysis, 4*i_test_set:4*(i_test_set+1)] = test(x_train, y_train, x_test, y_test, model_prefix, config, downsample_ratio, n_fno_layers, k_max, d_f)
                        cost = FNN1d_cost(x_test.shape[1], config)
    
                    data_analysis[i_data_analysis, 4*n_test_sets:4*n_test_sets+6] =  n_train, downsample_ratio, n_fno_layers, k_max, d_f, cost
                    i_data_analysis += 1
            


np.save(prefix+"data/darcy_analysis_validation_test.npy", data_analysis)

 





prefix = "/central/groups/esm/dzhuang/cost-accuracy-data/"
heat_u0s    = np.load(prefix+"heat_u0.npy")
heat_fs     = np.load(prefix+"heat_f.npy")
heat_us_ref = np.load(prefix+"heat_u.npy")
model_prefix = "/central/groups/esm/dzhuang/cost-accuracy-data/models/heat_FNO_0_"

n_data_array = [256, 512, 1024, 2048, 4096, 8192, 16384]
k_max_array = [16, 32, 64, 128]
d_f_array = [16, 32, 64, 128]
n_fno_layers_array = [3, 4, 5]
downsample_ratio_array = [1, 2, 4, 8]


#optimization
epochs = 1001
base_lr = 0.001
milestones = [200, 300, 400, 500, 800, 900]
scheduler_gamma = 0.5
batch_size=32
normalization = True
dim = []

M = 2**15
L, Ne_ref = 1.0, 2**12
n_test = 8192


n_test_sets = 2
data_analysis = np.zeros((len(n_data_array)*len(downsample_ratio_array)*len(n_fno_layers_array)*len(k_max_array)*len(d_f_array), 4*n_test_sets + 6)) 

i_data_analysis = 0

for n_train in n_data_array:
    for downsample_ratio in downsample_ratio_array:
        for n_fno_layers in n_fno_layers_array:
            for k_max in k_max_array:
                for d_f in d_f_array: 
                    for i_test_set in range(n_test_sets):
                        
                        
                        Ne = Ne_ref//downsample_ratio
                        grid = np.linspace(0, L, Ne+1)
                        
                        
                        x_train = torch.from_numpy(np.stack((heat_u0s[0:n_train, 0::downsample_ratio], heat_fs[0:n_train, 0::downsample_ratio], np.tile(grid, (n_train,1))), axis=-1).astype(np.float32))
                        y_train = torch.from_numpy(heat_us_ref[0:n_train, 0::downsample_ratio, np.newaxis].astype(np.float32))
                        x_test = torch.from_numpy(np.stack((heat_u0s[M//2+(i_test_set)*n_test:M//2+(i_test_set+1)*n_test, 0::downsample_ratio], heat_fs[M//2+(i_test_set)*n_test:M//2+(i_test_set+1)*n_test, 0::downsample_ratio], np.tile(grid, (n_test,1))), axis=-1).astype(np.float32))
                        y_test = torch.from_numpy(heat_us_ref[M//2+(i_test_set)*n_test:M//2+(i_test_set+1)*n_test, 0::downsample_ratio, np.newaxis].astype(np.float32))
                        
                        
                        
                        modes = [k_max] * n_fno_layers
                        # channel d_f
                        layers = [d_f] * (n_fno_layers + 1)
                        fc_dim = d_f
                        in_dim = 3
                        out_dim = 1
                        act = "gelu"
                        pad_ratio = 0.05
                        config = {"model" : {"modes": modes, "fc_dim": fc_dim, "layers": layers, "in_dim": in_dim, "out_dim":out_dim, "act": act, "pad_ratio":pad_ratio},
                                  "train" : {"base_lr": base_lr, "epochs": epochs, "milestones": milestones, "scheduler_gamma": scheduler_gamma, "batch_size": batch_size, 
                                            "normalization": normalization, "dim": dim}}

                        
                        data_analysis[i_data_analysis, 4*i_test_set:4*(i_test_set+1)] = test(x_train, y_train, x_test, y_test, model_prefix, config, downsample_ratio, n_fno_layers, k_max, d_f)
                        cost = FNN1d_cost(x_test.shape[1], config)
    
                    data_analysis[i_data_analysis, 4*n_test_sets:4*n_test_sets+6] =  n_train, downsample_ratio, n_fno_layers, k_max, d_f, cost
                    i_data_analysis += 1
            


np.save(prefix+"data/heat_analysis_validation_test.npy", data_analysis)

 


