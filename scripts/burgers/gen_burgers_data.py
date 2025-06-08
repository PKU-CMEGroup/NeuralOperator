import numpy as np
import random
import matplotlib.pyplot as plt

nu = 0.001

T = 1
dt = 0.00001

def initial(nbase, nodes):
    N, = nodes.shape
    u = np.zeros((N,))
    for i in range(nbase):
       u = u + (2*np.random.random()-1)*np.sin((i+1)*np.pi * nodes)/(i+1)/2
       u = u + (2*np.random.random()-1)*np.cos((i+1)*np.pi * nodes)/(i+1)/2

    u = u + (2*np.random.random()-1)/2

    return u

for index in range(1000):
    nnodes = random.randint(4000, 8000)
    #nnodes = 2048
    nodes = np.linspace(-1, 1, nnodes, endpoint=False)
    dx = 2/nnodes
    Nt = int(T / dt) + 1
    nbase = random.randint(10, 30)
    u = initial(nbase, nodes)

    data = []
    data.append(u)

    for iters in range(Nt):

        
        u_prev = u.copy()
        u_left = np.roll(u_prev, 1)   # 左邻点
        u_right = np.roll(u_prev, -1) # 右邻点
    
        # 对流项（迎风格式）
        conv = np.where(u_prev >= 0, 
                    u_prev * (u_prev - u_left)/dx,
                    u_prev * (u_right - u_prev)/dx)
    
        # 扩散项（中心差分）
        diff = nu * (u_right - 2*u_prev + u_left) / dx**2
    
        # 更新解
        u = u_prev - dt * conv + dt * diff
        if (iters+1) % 1000 == 0:
            data.append(u)
    
    data = np.array(data)
    np.save("data_uniform/data_"+str(index).zfill(5),data)









    







