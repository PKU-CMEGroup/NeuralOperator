import random
import torch
import sys
import numpy as np
import math
torch.manual_seed(0)
np.random.seed(0)


sys.path.append('../')
from models import FNN1d, FNN_train

prefix = "/central/groups/esm/dzhuang/cost-accuracy-data/"

N_chunk = 16

chunk_size, Np = np.load(prefix+"burgers_us_1.npy").shape
print("chunk_size, Np = ", chunk_size, Np)
Ne = Np - 1
burgers_input = np.zeros((chunk_size*N_chunk, Ne+1))
burgers_output = np.zeros((chunk_size*N_chunk, Ne+1))

for i in range(N_chunk):
    burgers_input[chunk_size*i:chunk_size*(i+1), :] = np.load(prefix+"burgers_u0s_"+str(i+1)+".npy")
    burgers_output[chunk_size*i:chunk_size*(i+1), :] = np.load(prefix+"burgers_us_"+str(i+1)+".npy")


print(burgers_output.shape)






n_data = int(sys.argv[1]) #5000
k_max = int(sys.argv[2])
d_f=int(sys.argv[3]) 
n_fno_layers=int(sys.argv[4])
downsample_ratio=int(sys.argv[5])



n_data_array = [n_data]
k_max_array = [k_max]
d_f_array = [d_f]
n_fno_layers_array = [n_fno_layers]
downsample_ratio_array = [downsample_ratio]

n_train_repeat = 1

#optimization
epochs = 1001
scheduler = "MultiStepLR"
weight_decay = 1e-4
base_lr = 0.001
milestones = [200, 300, 400, 500, 800,900]
scheduler_gamma = 0.5
batch_size=64
normalization_x = True
normalization_y = True
normalization_dim = []

data_analysis = np.zeros((len(n_data_array)*len(downsample_ratio_array)*len(n_fno_layers_array)*len(k_max_array)*len(d_f_array), 5+(3*epochs+1)*n_train_repeat)) 


L, Ne_ref = 1.0, 2**12
n_test = 2048
#training_rel_l2, test_rel_l2, test_l2, cost
i_data_analysis = 0
for n_data in n_data_array:
    for downsample_ratio in downsample_ratio_array:
        for n_fno_layers in n_fno_layers_array:
            for k_max in k_max_array:
                for d_f in d_f_array:  
                    setup_info="n_data_"+str(n_data)+"_k_max_"+str(k_max)+"_downsample_ratio_"+str(downsample_ratio)+"_n_fno_layers_"+str(n_fno_layers)+"_d_f_"+str(d_f)
                    
                    for i_train_repeat in range(n_train_repeat):

                        
                        Ne = Ne_ref//downsample_ratio

                        grid = np.linspace(0, L, Ne+1)
         
                        n_train = n_data
                         
                        # x_train = torch.from_numpy(np.stack((burgers_input[0:n_train, 0::downsample_ratio], np.tile(grid, (n_train,1))), axis=-1).astype(np.float32))
                        x_train = torch.from_numpy(burgers_input[0:n_train, 0::downsample_ratio, np.newaxis].astype(np.float32))



                        y_train = torch.from_numpy(burgers_output[0:n_train, 0::downsample_ratio, np.newaxis].astype(np.float32))
                        # x_train, y_train are [n_data, n_x, n_channel] arrays
                        # x_test = torch.from_numpy(np.stack((burgers_input[-n_test:, 0::downsample_ratio], np.tile(grid, (n_test,1))), axis=-1).astype(np.float32))
                        x_test = torch.from_numpy(burgers_input[-n_test:, 0::downsample_ratio, np.newaxis].astype(np.float32))
                        y_test = torch.from_numpy(burgers_output[-n_test:, 0::downsample_ratio, np.newaxis].astype(np.float32))
                        # x_test, y_test are [n_data, n_x, n_channel] arrays



                        # fourier k_max
                        modes = [k_max] * n_fno_layers
                        # channel d_f
                        layers = [d_f] * (n_fno_layers + 1)
                        fc_dim = d_f
                        in_dim = 1
                        out_dim = 1
                        act = "gelu"
                        pad_ratio = 0.0

                        

                        config = {"model" : {"modes": modes, "fc_dim": fc_dim, "layers": layers, "in_dim": in_dim,
                                             "out_dim":out_dim, "act": act, "pad_ratio":pad_ratio},
                                  "train" : {"base_lr": base_lr, "weight_decay": weight_decay, "epochs": epochs,
                                             "scheduler": scheduler, "milestones": milestones, 
                                             "scheduler_gamma":scheduler_gamma, "batch_size": batch_size,
                                             "normalization_x": normalization_x,"normalization_y": normalization_y,
                                             "normalization_dim": normalization_dim}}

                        train_rel_l2_losses, test_rel_l2_losses, test_l2_losses, cost = FNN_train(x_train, y_train, x_test, y_test, config, save_model_name=prefix+"models/burgers_FNO_"+str(i_train_repeat)+"_"+setup_info)
                        
                        data_analysis[i_data_analysis,5+i_train_repeat*(3*epochs+1):5+(i_train_repeat+1)*(3*epochs+1)] = np.hstack((train_rel_l2_losses, test_rel_l2_losses, test_l2_losses, cost))
                    
                    data_analysis[i_data_analysis,0:5] = n_data, downsample_ratio, n_fno_layers, k_max, d_f
                    i_data_analysis += 1
                    

np.save(prefix+"data/burgers_analysis_"+setup_info+".npy", data_analysis)

