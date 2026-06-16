import math
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
# 获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 向上两级找到项目根目录
project_root = os.path.dirname(os.path.dirname(current_dir))
# 添加到路径
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from utility.gaussian_random_fields import gaussian_random_field_1d


def solve_burgers1d_equation(f, L=2*np.pi, T=0.0, c=1.0,
                          CFL=0.4, nu=0.0, limiter="vanleer", return_ut=False):
    """
    二阶 shock-capturing 求解 1D Burgers（周期边界）:
        u_t + (c * 0.5*u^2)_x = nu * u_xx

    参数
    ----
    f : float array
        初值 u(x,0)=f(x)
    L : float
        周期长度
    T : float
        终止时间
    c : float
        对流系数（通量为 c*u^2/2）
    N : int
        空间网格点数
    CFL : float
        CFL 数
    nu : float
        粘性系数（nu=0 为无粘）
    limiter : str
        "minmod" / "vanleer" / "mc"
    return_ut : bool
        是否返回 u_t(x,T)

    返回
    ----
    u : ndarray, shape (N,)
        u(x,T)
    (可选) ut : ndarray, shape (N,)
        u_t(x,T)（用离散方程在最后一步估计）
    """
    N = len(f)
    dx = L / N
    u = f

    # ---------- slope limiters ----------
    def minmod(a, b):
        return 0.5*(np.sign(a)+np.sign(b))*np.minimum(np.abs(a), np.abs(b))

    def vanleer(a, b):
        # 2ab/(a+b) if same sign else 0
        s = a*b
        out = np.zeros_like(a)
        mask = s > 0
        out[mask] = 2*s[mask]/(a[mask]+b[mask])
        return out

    def mc(a, b):
        # monotonized central
        return minmod(0.5*(a+b), minmod(2*a, 2*b))

    if limiter.lower() == "minmod":
        slope = minmod
    elif limiter.lower() == "vanleer":
        slope = vanleer
    elif limiter.lower() == "mc":
        slope = mc
    else:
        raise ValueError("limiter must be 'minmod', 'vanleer', or 'mc'")

    # ---------- flux ----------
    def flux(u):
        return 0.5 * c * u*u

    # ---------- periodic shift helpers ----------
    def roll(u, k):
        return np.roll(u, k)

    # ---------- MUSCL reconstruction at interfaces i+1/2 ----------
    # returns left state uL_{i+1/2} and right state uR_{i+1/2} arrays of length N
    def reconstruct(u):
        duL = u - roll(u, 1)      # backward diff
        duR = roll(u, -1) - u     # forward diff
        sigma = slope(duL, duR)   # limited slope at cell centers

        # interface i+1/2: left from cell i, right from cell i+1
        uL = u + 0.5*sigma
        uR = roll(u, -1) - 0.5*roll(sigma, -1)
        return uL, uR

    # ---------- Rusanov flux at interfaces ----------
    def numerical_flux(u):
        uL, uR = reconstruct(u)
        fL, fR = flux(uL), flux(uR)
        a = np.maximum(np.abs(c*uL), np.abs(c*uR))  # local wave speed for Burgers
        F = 0.5*(fL + fR) - 0.5*a*(uR - uL)
        return F  # F[i] is flux at interface i+1/2

    # ---------- diffusion term (2nd order central) ----------
    def diffusion(u):
        return nu * (roll(u, -1) - 2*u + roll(u, 1)) / (dx*dx)

    # ---------- RHS operator ----------
    def rhs(u):
        F = numerical_flux(u)
        # conservative update: -(F_{i+1/2} - F_{i-1/2})/dx
        conv = -(F - roll(F, 1)) / dx
        if nu != 0.0:
            return conv + diffusion(u)
        return conv

    # ---------- time stepping: TVD RK2 ----------
    t = 0.0
    while t < T:
        # dt from CFL: dt <= CFL*dx/max|c*u|
        umax = np.max(np.abs(u))
        # for Burgers: wave speed = |c*u|
        amax = max(1e-14, np.abs(c)*umax)
        dt_adv = CFL * dx / amax
        dt_diff = np.inf if nu == 0 else 0.5 * dx*dx / nu  # stability rough bound
        dt = min(dt_adv, dt_diff, T - t)
        

        # RK2
        k1 = rhs(u)
        u1 = u + dt*k1
        k2 = rhs(u1)
        u = 0.5*u + 0.5*(u1 + dt*k2)

        t += dt

    if return_ut:
        ut = rhs(u)
        return u, ut
    return u


def generate_initial_conditions(M, N, k_max = 10, L=2*np.pi, seed=None):
    """
    生成一维Burgers方程的随机初始状态
    
    参数:
    M: int
       生成初始状态的数量
    N: int
       空间网格点数
    k_max: int
       生成初始状态时包含的最大傅里叶模态数量（越大越复杂）
    L: float
       周期长度 
    seed: None 或者 int 
        随机种子，用于复现结果
    
    返回:
        M by N 矩阵 表示初值
    """
    x = np.linspace(0, L, N, endpoint=False)
    dx = L/N
    f = gaussian_random_field_1d(M, N, L, sigma=10.0, tau = 1.0, alpha = 1.5, bc_name = 'periodic', seed = seed, k_max = k_max)
    return f
    

    
def visualization():
    nT, T, k_max, N, L = set_default_params()
    
    u0 = generate_initial_conditions(M = 4, N = N, L= L, k_max = k_max, seed=42)
    fig, axs = plt.subplots(4, 3, figsize=(16, 8))
    axs[0,0].set_title("u0")
    axs[0,1].set_title("u")
    axs[0,2].set_title("u0 samples")
    
    x = np.linspace(0, L, N, endpoint=False)
    for i in range(4):
        
        axs[i,0].plot(x, u0[i,...])
        axs[i,0].set_xlabel("x")

        u_ref = np.zeros((nT+1, N, 2))
        u_ref[:,:,1] = x
        u_ref[0,:,0] = u0[i,...]

        for j in range(nT):
            u_ref[j+1,:,0] = solve_burgers1d_equation(u_ref[j,:,0], L, T)
            
    
        axs[i,1].imshow(u_ref[:,:,0], cmap='viridis',  aspect='auto')
        axs[i,1].set_xlabel("x")
        axs[i,1].set_ylabel("time steps")
        
        axs[i,2].plot(x, u_ref[0, :, 0],  color="C0", label="initial")
        axs[i,2].plot(x, u_ref[1, :, 0],  color="C1", label="step 1")
        axs[i,2].plot(x, u_ref[2, :, 0],  color="C2", label="step 2")
        axs[i,2].plot(x, u_ref[nT,:, 0],  color="C3", label="step %d" %nT)
        axs[i,2].set_xlabel("x")
        axs[i,2].legend()
        
    
    fig.tight_layout() 
    plt.show()
    plt.savefig("burgers1d.pdf")



def set_default_params():
    nT = 100
    T = 0.2
    
    N = 2048
    k_max = N//2

    L = 2*np.pi

    return nT, T, k_max, N, L


def generate_data():
    """
    生成一维薛定谔方程的训练数据，保存在 data/burgers_1d/burgers1d_data.npz 中
    数据格式为一个 N by (nT+1) by N by 2 的数组，分别表示 nT+1 个时间步的函数值和位置
    """
    nT, T, k_max, N, L = set_default_params()
    x = np.linspace(0, L, N, endpoint=False)
    ndata = 1000
    fs = generate_initial_conditions(M = ndata, N = N, k_max = k_max, L=L, seed=None)

    
    u_refs = []
    for i in range(ndata):
        u_ref = np.zeros((nT+1, N, 2))
        u_ref[:,:,1] = x
        u_ref[0,:,0] = fs[i,:]
        for j in range(nT):
            u_ref[j+1,:,0] = solve_burgers1d_equation(f=u_ref[j,:,0], L=L, T=T)
            
        u_refs.append(u_ref)
        
    u_refs = np.array(u_refs)

    Path('../../data/burgers_1d').mkdir(parents=True, exist_ok=True)
    np.savez_compressed("../../data/burgers_1d/burgers1d_data.npz", u_refs = u_refs)



def extract_evolution_matrix():
    """
    生成一维热方程的训练数据，保存在 data/burgers_1d/burgers1d_data.npz 中
    数据格式为一个 N by (nT+1) by N by 2 的数组，分别表示 nT+1 个时间步的u0, k, f, x
    """
    nT, T, k_max, N, L = set_default_params()
    x = np.linspace(0, L, N, endpoint=False)
    n_train, n_test = 10000, 500
    dim = 1
    fig, axs = plt.subplots(1, figsize=(6, 6))
    
    
    data = np.load("../../data/burgers_1d/burgers1d_data.npz")['u_refs']
    
    X, Y = [], []
    for i in list(range(math.ceil(n_train / nT))) + list(range(-math.ceil(n_test / nT), 0)):
        for j in range(nT):
            X.append(data[i,j,  :,:dim])    #前一步的波函数实部和虚部 
            Y.append(data[i,j+1,:,:dim])    #后一步的波函数实部和虚部
    X, Y = np.array(X), np.array(Y)
    X, Y = X.squeeze(-1), Y.squeeze(-1)
    evolution_matrix = np.linalg.lstsq(X, Y, rcond=None)[0]
    
    print(evolution_matrix)
    im = axs.imshow(evolution_matrix,  cmap='viridis',  aspect='auto', vmin=-1.0, vmax=1.0)
    fig.colorbar(im, ax=axs)

    plt.savefig("burgers1d_evolution_matrix.pdf")
        

    
    
if __name__ == "__main__":
    visualization()
    generate_data()
    extract_evolution_matrix() 
    

