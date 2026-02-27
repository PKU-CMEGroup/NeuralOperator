import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def solve_schrodinger1d_equation(f, g, V, L=2*np.pi, T=1.0, dt_max=0.01, check_conservation=False):
    """
    求解一维周期的薛定谔方程的数值解
    
    参数:
    f: float array
       初始状态，波函数实部，长度为N
    g: float array
       初始状态，波函数虚部，长度为N
    V: float array
       势能函数，长度为N
    L: float
       周期长度
    T: float
       时间
    dt_max: float
       最大时间步长(目前采用这个步长)
    check_conservation: bool
       是否检查概率守恒（诊断输出）
    
    
    返回:
    最终的波函数状态，实部\虚部
    """
        # --- basic checks ---
    if f.shape != g.shape:
        raise ValueError("f, g, V must have the same shape (N,)")

    N = f.size
    
    if T < 0:
        raise ValueError("T must be >= 0")
    
    
    # --- grid + Fourier modes ---
    dx = L / N
    V_max = np.max(np.abs(V))
    k = 2*np.pi * np.fft.fftfreq(N, d=dx)   # angular wave numbers
    k_max = np.max(np.abs(k))
    
    
    # --- initial wavefunction ---
    psi = f + 1j * g

    # kinetic phase: exp(-i dt * k^2/2)
    def kinetic_phase(dt_):
        return np.exp(-1j * 0.5 * (k**2) * dt_)

    # --- diagnostics ---
    def prob(psi_):
        return float(np.sum(np.abs(psi_)**2) * dx)

    p0 = prob(psi)

    # --- time stepping (Strang splitting) ---
    t = 0.0
    while t < T:  
            
        # 估算动能项的特征时间尺度
        # dt_kinetic = CFL / (0.5 * k_max**2)  
        # 估算势能项的特征时间尺度
        # dt_potential = CFL / V_max  
        # 取最小值并确保合理范围
        # dt = min(dt_kinetic, dt_potential, T - t, dt_max) 
        dt = min(T - t, dt_max)
        
        
        
        phaseK = kinetic_phase(dt)
        phaseV_half = np.exp(-1j * V * dt / 2)
    
        # half potential
        psi = phaseV_half * psi

        # kinetic in Fourier
        psi_k = np.fft.fft(psi)
        psi_k *= phaseK
        psi = np.fft.ifft(psi_k)

        # half potential
        psi = phaseV_half * psi
        
        t += dt


    if check_conservation:
        pT = prob(psi)
        rel_err = abs(pT - p0) / (abs(p0) + 1e-15)
        # 你也可以改成 print 或返回该数值
        print(f"[diagnostic] Prob(0)={p0:.16e}, Prob(T)={pT:.16e}, rel_err={rel_err:.3e}")

 
    return psi.real, psi.imag


def generate_initial_conditions(M, N, k_max = 10, L=2*np.pi, normalize=True, seed=None):
    """
    生成一维薛定谔方程的随机初始状态
    
    参数:
    M: int
       生成初始状态的数量
    N: int
       空间网格点数
    k_max: int
       生成初始状态时包含的最大傅里叶模态数量（越大越复杂）
    L: float
       周期长度 
    normalize: bool
       是否归一化波函数
    seed: None 或者 int 
        随机种子，用于复现结果
    
    返回:
        2个N by M 矩阵，分别表示实部和虚部 
    """
    np.random.seed(seed)
    
    x = np.linspace(0, L, N, endpoint=False)
    dx = L/N
    psi_real = np.zeros((M, N))
    psi_imag = np.zeros((M, N))
    
    for k in range(0, k_max):
        kk = np.sqrt(k)
        psi_real += np.outer(np.random.randn(M)/(kk+1), np.sin(2*np.pi*x*k/L))
        psi_real += np.outer(np.random.randn(M)/(kk+1), np.cos(2*np.pi*x*k/L))
        psi_imag += np.outer(np.random.randn(M)/(kk+1), np.sin(2*np.pi*x*k/L))
        psi_imag += np.outer(np.random.randn(M)/(kk+1), np.cos(2*np.pi*x*k/L))
    

    if normalize:
        psi = psi_real + 1j*psi_imag
        norm = np.linalg.norm(psi, axis=1, keepdims=True) * np.sqrt(dx)
        if np.all(norm > 1e-12):
            psi_real /= norm
            psi_imag /= norm
        else:
            raise ValueError("Wavefunction nearly zero; increase k_max or adjust scaling.")

    return psi_real, psi_imag


def fixed_periodic_potential(N, L=2*np.pi, V_type = "two_mode"):
    """
    返回长度 N 的实周期势能 V(x)，x ∈ [0,L).
    参数：
    N: int
       空间网格点数
    L: float
       周期长度 
    V_type: string
      - "constant": 常数势
      - "cosine": 单余弦势
      - "two_mode": 两个模态叠加（更丰富，仍很干净）
      - "lattice": 平滑的周期高斯势垒阵列
    """
    x = np.linspace(0, L, N, endpoint=False)
    if V_type == "constant":
        V = np.ones(N)
        
    elif V_type == "cosine":
        V0, m, phi = 1.0, 2, 0.3
        V = V0 * np.cos(2*np.pi*m*x/L + phi)

    elif V_type == "two_mode":
        a1, m1, phi1 = 1.0, 2, 0.3
        a2, m2, phi2 = 0.6, 5, 1.2
        V = a1*np.cos(2*np.pi*m1*x/L + phi1) + a2*np.cos(2*np.pi*m2*x/L + phi2)

    elif V_type == "lattice":
        n_bumps, V0, sigma = 8, 2.0, 0.12 * L/(2*np.pi)  # sigma随L缩放
        centers = (np.arange(n_bumps)/n_bumps) * L
        V = np.zeros_like(x)
        for c in centers:
            d = np.minimum(np.abs(x-c), L-np.abs(x-c))  # 周期距离
            V += V0 * np.exp(-0.5*(d/sigma)**2)
        V -= V.mean()  # 去均值（可选）

    else:
        raise ValueError("V_type must be one of: 'constant', 'cosine', 'two_mode', 'lattice'")

    return V

def set_default_params():
    nT = 100
    T = 0.2
    k_max = 20
    N = 512
    L = 2*np.pi
    V_type = "two_mode"
    return nT, T, k_max, N, L, V_type



def visualization():
    nT, T, k_max, N, L, V_type = set_default_params()
    
    f, g = generate_initial_conditions(M = 1, N = N, k_max = k_max, seed=42)
    fig, axs = plt.subplots(4, 5, figsize=(16, 8))
    axs[0,0].set_title("V")
    axs[0,1].set_title("real(psi)")
    axs[0,2].set_title("imag(psi)")
    axs[0,3].set_title("real(psi) samples")
    axs[0,4].set_title("imag(psi) samples")

    V_types =  ["constant",  "cosine", "two_mode", "lattice"]
    x = np.linspace(0, L, N, endpoint=False)
    
    for i in range(len(V_types)):
        V_type = V_types[i]
        
        V = fixed_periodic_potential(N, L=L, V_type=V_type)
        axs[i,0].plot(x, V)
        axs[i,0].set_xlabel("x")

        u_ref = np.zeros((nT+1, N, 4))
        u_ref[:,:,2], u_ref[:,:,3] = V, x
        u_ref[0,:,0], u_ref[0,:,1] = f[0,:], g[0,:]

        for j in range(nT):
            u_ref[j+1,:,0], u_ref[j+1,:,1] = solve_schrodinger1d_equation(f=u_ref[j,:,0], g=u_ref[j,:,1], V=V, L=L, T=T, check_conservation=False)   
        
        
    
        axs[i,1].imshow(u_ref[:,:,0], cmap='viridis',  aspect='auto')
        axs[i,1].set_xlabel("x")
        axs[i,1].set_ylabel("time steps")
        axs[i,2].imshow(u_ref[:,:,1],  cmap='viridis',  aspect='auto')
        axs[i,2].set_xlabel("x")
        axs[i,2].set_ylabel("time steps")
        
        axs[i,3].plot(x, u_ref[0, :, 0],  color="C0", label="initial")
        axs[i,3].plot(x, u_ref[1, :, 0],  color="C1", label="step 1")
        axs[i,3].plot(x, u_ref[2, :, 0],  color="C2", label="step 2")
        axs[i,3].plot(x, u_ref[nT,:, 0],  color="C3", label="step %d" %nT)
        axs[i,3].set_xlabel("x")
        
        axs[i,4].plot(x, u_ref[0, :, 1],  color="C0", label="initial")
        axs[i,4].plot(x, u_ref[1, :, 1],  color="C1", label="step 1")
        axs[i,4].plot(x, u_ref[2, :, 1],  color="C2", label="step 2")
        axs[i,4].plot(x, u_ref[nT,:, 1],  color="C3", label="step %d" %nT)
        axs[i,4].set_xlabel("x")
        axs[i,4].legend()
    
    fig.tight_layout() 
    plt.show()
    plt.savefig("schrodinger1d.pdf")
    
    
def generate_data():
    """
    生成一维薛定谔方程的训练数据，保存在 data/schrodinger/schrodinger1d_{V_type}_data.npz 中
    数据格式为一个 N by (nT+1) by N by 4 的数组，分别表示 nT+1 个时间步的波函数实部、虚部、势能和位置
    """
    nT, T, k_max, N, L, V_type = set_default_params()
    x = np.linspace(0, L, N, endpoint=False)
    ndata = 1000
    u_refs = []
    fs,gs = generate_initial_conditions(M = ndata, N = N, k_max = k_max, L=L, normalize=True, seed=None)
    V = fixed_periodic_potential(N, L=L, V_type=V_type)
    for i in range(ndata):
        u_ref = np.zeros((nT+1, N, 4))
        u_ref[:,:,2], u_ref[:,:,3] = V, x
        u_ref[0,:,0], u_ref[0,:,1] = fs[i,:], gs[i,:]
        for j in range(nT):
            u_ref[j+1,:,0], u_ref[j+1,:,1] = solve_schrodinger1d_equation(f=u_ref[j,:,0], g=u_ref[j,:,1], V=V, L=L, T=T, check_conservation=False)
            
        u_refs.append(u_ref)
        
    u_refs = np.array(u_refs)

    Path('../../data/schrodinger').mkdir(parents=True, exist_ok=True)
    np.savez_compressed("../../data/schrodinger/schrodinger1d_"+V_type+"_data.npz", u_refs = u_refs)




    
    
if __name__ == "__main__":
    visualization()
    generate_data()
    

