from scripts.schrodinger.generate_schrodinger1d_data import fixed_periodic_potential
from scripts.schrodinger.transformer_train import setup_model
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from timeit import default_timer
from transformer import Transformer
from gen_schr_eq_data import generate_initial_conditions, solve_schrodinger_equation, set_params



if __name__ == "__main__":
    nT, T, k_max, N, L, V_type = set_params()
    k_max = 20
    
    in_dim, out_dim = 3, 2 
 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 


    u_0, v_0 = generate_initial_conditions(M = 1, N = N, k_max = k_max, L=L, seed=101)
    V = fixed_periodic_potential(N, L=L, V_type=V_type)
    u_ref = np.zeros((nT+1, N, 2))
    u_ref[0,:,0], u_ref[0,:,1] = u_0, v_0
    for i in range(nT):
        u_ref[i+1,:,0], u_ref[i+1,:,1] = solve_schrodinger_equation(f=u_ref[i,:,0], g=u_ref[i,:,1], V=V, L=L, T=T)
            

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = setup_model(in_dim, out_dim, device, checkpoint_path = "pth/TRANSFORMER_"+V_type+".pth")

  


    x = np.linspace(0, L, N, endpoint=False)
    batch_size = 1
    u = torch.zeros((batch_size, nT+1, N, out_dim+2))
    u[:,0,:,2], u[:,0,:,3] = torch.tensor(V), torch.tensor(x)
    u[0,0,:,0], u[0,0,:,1] = torch.tensor(u_0), torch.tensor(v_0)
    u = u.to(device)  
    for i in range(nT):
        u[:,i+1,:, 0:out_dim] = model(u[:, i,...], coords=u[:,i,:,3:4])
    u = u.detach().cpu().numpy()[0,...]



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
