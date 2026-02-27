import sys
import os
import math 
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from timeit import default_timer

# 获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 向上两级找到项目根目录
project_root = os.path.dirname(os.path.dirname(current_dir))
# 添加到路径
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from scripts.schrodinger.generate_schrodinger1d_data import set_default_params, fixed_periodic_potential, generate_initial_conditions, solve_schrodinger1d_equation
from scripts.schrodinger.transformer_train import setup_model, preprocess_data
from utility.normalizer import UnitGaussianNormalizer


if __name__ == "__main__":
    nT, T, k_max, N, L, V_type = set_default_params()
    
    in_dim, out_dim = 3, 2 
 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 


    u_0, v_0 = generate_initial_conditions(M = 1, N = N, k_max = k_max, L=L, seed=101)
    V = fixed_periodic_potential(N, L=L, V_type=V_type)
    u_ref = np.zeros((nT+1, N, 2))
    u_ref[0,:,0], u_ref[0,:,1] = u_0, v_0
    for i in range(nT):
        u_ref[i+1,:,0], u_ref[i+1,:,1] = solve_schrodinger1d_equation(f=u_ref[i,:,0], g=u_ref[i,:,1], V=V, L=L, T=T)
            

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = setup_model(in_dim, out_dim, device, checkpoint_path = "Transformer_model_V"+V_type+".pth")

    
    # normalizer
    normalization_x = False
    normalization_y = True
    normalization_dim_x = []
    normalization_dim_y = []
    non_normalized_dim_x = 4
    non_normalized_dim_y = 0
    n_train, n_test = 10000, 500
    data = np.load("../../data/schrodinger/schrodinger1d_"+V_type+"_data.npz")['u_refs']
    x_train, _, y_train, _, _, _ = preprocess_data(data, n_train, n_test, device)
    if normalization_x:
        x_normalizer = UnitGaussianNormalizer(x_train, non_normalized_dim = non_normalized_dim_x, normalization_dim=normalization_dim_x)
        x_normalizer.to(device)
    if normalization_y:
        y_normalizer = UnitGaussianNormalizer(y_train, non_normalized_dim = non_normalized_dim_y, normalization_dim=normalization_dim_y)
        y_normalizer.to(device)


    x = np.linspace(0, L, N, endpoint=False)
    batch_size = 1
    u = torch.zeros((batch_size, nT+1, N, out_dim+2))
    u[0,:,:,2], u[0,:,:,3] = torch.tensor(V), torch.tensor(x)
    # initialization
    u[0,0,:,0], u[0,0,:,1] = torch.tensor(u_0), torch.tensor(v_0)
    u = u.to(device)  
    for i in range(nT):
        
        u[:,i+1,:, 0:out_dim] =  model( u = (x_normalizer.encode(u[:, i, :, 0:in_dim]) if normalization_x else u[:, i, :, 0:in_dim]), coords=u[:,i,:,3:4]) 
        if normalization_y:
            u[:,i+1,:, 0:out_dim] = y_normalizer.decode(u[:,i+1,:, 0:out_dim])
    u = u.detach().cpu().numpy()[0,...,0:out_dim]



    ################### Postprocessing ######################
    error = np.zeros(nT+1)
    rel_error = np.zeros(nT+1)
    for i in range(nT):
        error[i+1] = np.linalg.norm(u_ref[i+1,...] - u[i+1,...]) * L/N
        rel_error[i+1] = np.linalg.norm(u_ref[i+1,...] - u[i+1,...])/(np.linalg.norm(u_ref[i+1,:]))




    ############ Visualization ########################

    fig, axs = plt.subplots( out_dim+1, figsize=(8, 8))
    for i in range(out_dim):
        axs[i].plot(x, u_ref[0,:,i], color="C0", label="initial")
        axs[i].plot(x, u_ref[1,:,i], color="C1", label="ref. step 1")
        axs[i].plot(x, u[1, :, i], "--", color="C1", markerfacecolor="none")
        axs[i].plot(x, u_ref[2,:,i], color="C2", label="ref. step 2")
        axs[i].plot(x, u[2, :, i], "--", color="C2", markerfacecolor="none")
        
    axs[-1].plot(error[:3], "-o", markerfacecolor="none", label = "abs. error")
    axs[-1].plot(rel_error[:3], "-o", markerfacecolor="none", label = "rel. error")
    axs[-1].legend()
    axs[0].legend()
    plt.savefig("Transformer_"+V_type+"_error.pdf")




    ############ Visualization ########################

    fig, axs = plt.subplots(2, max(out_dim+1, 2*out_dim), figsize=(16, 8))
    for i in range(out_dim):
        axs[0,i].plot(x, u_ref[0,:,i], color="C0", label="initial")
        axs[0,i].plot(x, u_ref[1,:,i], color="C1", label="ref. step 1")
        axs[0,i].plot(x, u[1, :, i], "--", color="C1", markerfacecolor="none")
        axs[0,i].plot(x, u_ref[2,:,i], color="C2", label="ref. step 2")
        axs[0,i].plot(x, u[2, :, i], "--", color="C2", markerfacecolor="none")
        axs[0,i].plot(x, u_ref[nT,:,i], color="C3", label="ref. step %d" %nT)
        axs[0,i].plot(x, u[nT, :, i], "--", color="C3", markerfacecolor="none")

    axs[0,-1].plot(error, "-o", markerfacecolor="none", label = "abs. error")
    axs[0,-1].plot(rel_error, "-o", markerfacecolor="none", label = "rel. error")
    axs[0,-1].legend()
    axs[0,0].legend()



    for i in range(out_dim):
        # 找到全局最小和最大值用于统一的 color bar
        vmin = min(u_ref[...,i].min(), u[...,i].min())
        vmax = max(u_ref[...,i].max(), u[...,i].max())
        # 绘制第一个图
        axs[1,2*i].imshow(u_ref[:,:,i], cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
        axs[1,2*i].set_title('u_ref')
        axs[1,2*i].set_xlabel('x')
        axs[1,2*i].set_ylabel('time steps')

        axs[1,2*i+1].imshow(u[:,:,i], cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
        axs[1,2*i+1].set_title('u')
        axs[1,2*i+1].set_xlabel('x')

    print("error is ", error)
    plt.show()
    plt.savefig("Transformer_"+V_type+str(nT)+"steps.pdf")
