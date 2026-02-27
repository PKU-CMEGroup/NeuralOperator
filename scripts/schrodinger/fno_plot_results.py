import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from timeit import default_timer
from fno import FNO1d
from gen_schr_eq_data import generate_initial_conditions, solve_schrodinger_equation, set_params


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


nT, T, k_max, N, L = set_params()
k_max = 30
V_type = "two_mode"
in_dim, out_dim = 2, 2
u_0, v_0 = generate_initial_conditions(N, k_max = k_max, L=L, seed=101)

u_ref = np.zeros((nT+1, N, 2))
u_ref[0,:,0], u_ref[0,:,1] = u_0, v_0
for i in range(nT):
    u_ref[i+1,:,0], u_ref[i+1,:,1] = solve_schrodinger_equation(f=u_ref[i,:,0], g=u_ref[i,:,1], V_type=V_type, L=L, T=T)
        


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
model.load_state_dict(torch.load("pth/FNO_"+V_type+".pth", map_location="cpu"))
model = model.to(device)


# for name, param in model.named_parameters():
#     print(f"参数名: {name}")
#     print(f"形状: {param.shape}")
#     print(f"数据类型: {param.dtype}")
#     print(f"设备: {param.device}")
#     print(f"数值: {param.data}")  # 打印参数值
#     print(f"最大绝对值: {torch.max(torch.abs(param.data)).item():.6f}")
#     print("-" * 50)

# with torch.no_grad():
#     for i in range(3):
#         weights = model.sp_convs[i].get_parameter('weights1')
#         weights[:,:,20:] *= 0.0
#         print(model.sp_convs[i].get_parameter('weights1'))


xx = np.linspace(0, 2*np.pi, N, endpoint=False)

batch_size = 1
u = torch.zeros((batch_size, nT+1, N, out_dim)).to(device)
u[0,0,:,0], u[0,0,:,1] = torch.tensor(u_0), torch.tensor(v_0)

u = u.to(device)  
for i in range(nT):
    u[:,i+1,:, 0:out_dim] = model(u[:, i,...])
u = u.detach().cpu().numpy()[0,...]

error = np.zeros(nT+1)
rel_error = np.zeros(nT+1)
for i in range(nT):
    error[i+1] = np.linalg.norm(u_ref[i+1,...] - u[i+1,...]) * L/N
    rel_error[i+1] = np.linalg.norm(u_ref[i+1,...] - u[i+1,...])/(np.linalg.norm(u_ref[i+1,:]))




############ Visualization ########################

fig, axs = plt.subplots( out_dim+1, figsize=(8, 8))
for i in range(out_dim):
    axs[i].plot(xx, u_ref[0,:,i], color="C0", label="initial")
    axs[i].plot(xx, u_ref[1,:,i], color="C1", label="ref. step 1")
    axs[i].plot(xx, u[1, :, i], "--", color="C1", markerfacecolor="none")
    axs[i].plot(xx, u_ref[2,:,i], color="C2", label="ref. step 2")
    axs[i].plot(xx, u[2, :, i], "--", color="C2", markerfacecolor="none")
    
axs[-1].plot(error[:3], "-o", markerfacecolor="none", label = "abs. error")
axs[-1].plot(rel_error[:3], "-o", markerfacecolor="none", label = "rel. error")
axs[-1].legend()
axs[0].legend()
plt.savefig("FNO_"+V_type+".pdf")




############ Visualization ########################

fig, axs = plt.subplots(2, max(out_dim+1, 2*out_dim), figsize=(16, 8))
for i in range(out_dim):
    axs[0,i].plot(xx, u_ref[0,:,i], color="C0", label="initial")
    axs[0,i].plot(xx, u_ref[1,:,i], color="C1", label="ref. step 1")
    axs[0,i].plot(xx, u[1, :, i], "--", color="C1", markerfacecolor="none")
    axs[0,i].plot(xx, u_ref[2,:,i], color="C2", label="ref. step 2")
    axs[0,i].plot(xx, u[2, :, i], "--", color="C2", markerfacecolor="none")
    axs[0,i].plot(xx, u_ref[nT,:,i], color="C3", label="ref. step %d" %nT)
    axs[0,i].plot(xx, u[nT, :, i], "--", color="C3", markerfacecolor="none")

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
plt.savefig("FNO_"+V_type+str(nT)+"steps.pdf")
