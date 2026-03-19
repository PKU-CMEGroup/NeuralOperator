import math
import os, sys
import numpy as np
from numpy.fft import fft2, ifft2
import matplotlib.pyplot as plt
from pathlib import Path
# 获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 向上两级找到项目根目录
project_root = os.path.dirname(os.path.dirname(current_dir))
# 添加到路径
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from utility.gaussian_random_fields import gaussian_random_field_2d

# ----------------------------
# helper functions
# ----------------------------

def compute_wavenumbers(L, N):
    # FFT conventions: numpy uses exp(+2π i k n/N) in inverse.
    # For domain length L, physical derivative corresponds to (2π/L) * k.
    
    k1 = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)  # [N]
    kx, ky = np.meshgrid(k1, k1, indexing="ij")    # [N,N]
    ikx = 1j * kx
    iky = 1j * ky
    k2 = kx**2 + ky**2
    k2[0, 0] = 1.0  # avoid division by zero for Poisson solve
    return kx, ky, ikx, iky, k2


def grad_from_hat(phi_hat, ikx, iky):
    """Return (phi_x, phi_y) in physical space from phi_hat."""
    px = np.real(ifft2(ikx * phi_hat))
    py = np.real(ifft2(iky * phi_hat))
    return px, py

def lap_hat(phi_hat, k2):
    return -(k2) * phi_hat
    
    
def vorticity_hat_from_uv(u, v, ikx, iky):
    """ω̂ = i kx v̂ - i ky û."""
    uh = fft2(u)
    vh = fft2(v)
    return ikx * vh - iky * uh

def streamfunction_hat(omega_hat, k2):
    """Solve Δψ = -ω  =>  ψ̂ = ω̂ / |k|^2 (with ψ̂(0,0)=0)."""
    psi_hat = omega_hat / k2
    psi_hat[0, 0] = 0.0
    return psi_hat

def velocity_from_omega_hat(omega_hat, ikx, iky, k2):
    """u = ψ_y, v = -ψ_x."""
    psi_hat = streamfunction_hat(omega_hat, k2)
    psi_x, psi_y = grad_from_hat(psi_hat, ikx, iky)
    u = psi_y
    v = -psi_x
    return u, v
    
    
def vorticity_from_uv(u, v, ikx, iky):
    """
    Compute vorticity ω = ∂x v - ∂y u on a periodic grid via FFT.

    Args:
        u, v: [N,N] velocity components.
        L: domain length.

    Returns:
        omega: [N,N] real array.
    """
    omega_hat = vorticity_hat_from_uv(u, v, ikx, iky)
    return np.real(np.fft.ifft2(omega_hat))

def kinetic_energy(u, v):
    return 0.5 * np.mean(u*u + v*v)

def enstrophy(omega):
    return 0.5 * np.mean(omega*omega)

def vorticity_forcing(fx, fy, ikx, iky):
    """
    Compute vorticity forcing f_ω from momentum forcing F=(fx,fy) on a 2D periodic grid.

    Args:
        fx: x-component of body force, shape [N, N].
        fy: y-component of body force, shape [N, N].
        L: Domain length (periodic in both x and y).

    Returns:
        f_omega: Vorticity forcing, shape [N, N].

    Math:
        f_ω = (∇×F)_z = ∂_x Fy - ∂_y Fx

        Using FFT on a periodic grid:
            ∂_x ↔ i kx,   ∂_y ↔ i ky
            f̂_ω = i kx F̂_y - i ky F̂_x

    Require:
        - fx, fy have identical shape [N, N]
        - periodic boundary conditions on [0,L)^2
    """
    fx = np.asarray(fx, dtype=np.float64)
    fy = np.asarray(fy, dtype=np.float64)
    assert fx.shape == fy.shape and fx.ndim == 2 and fx.shape[0] == fx.shape[1], "fx,fy must be [N,N]"

    fxh = np.fft.fft2(fx)
    fyh = np.fft.fft2(fy)

    fomega_hat = ikx * fyh - iky * fxh
    fomega = np.real(np.fft.ifft2(fomega_hat))
    return fomega


def pressure_from_uv(u, v, kx, ky, k2, fx=None, fy=None):
    """
    从速度场 (u,v) 回算压力 p（周期边界，零均值规范）。

    Args:
        u, v: [N,N] velocity components
        kx, ky: [N,N] wave numbers
        k2: [N,N] kx^2 + ky^2 with k2[0,0]=1 to avoid divide-by-zero
        fx, fy: optional body force in momentum equation, shape [N,N]

    Returns:
        p: [N,N] pressure with mean(p)=0

    Math:
        Δp = -[(ux)^2 + (vy)^2 + 2(vx*uy)] + div(F)
        In Fourier:
            p_hat = - rhs_hat / k2, with p_hat[0,0]=0
    """
    ikx = 1j * kx
    iky = 1j * ky

    uh = np.fft.fft2(u)
    vh = np.fft.fft2(v)

    ux = np.real(np.fft.ifft2(ikx * uh))
    uy = np.real(np.fft.ifft2(iky * uh))
    vx = np.real(np.fft.ifft2(ikx * vh))
    vy = np.real(np.fft.ifft2(iky * vh))

    rhs = -(ux*ux + vy*vy + 2.0*vx*uy)

    if fx is not None and fy is not None:
        fxh = np.fft.fft2(fx)
        fyh = np.fft.fft2(fy)
        divf = np.real(np.fft.ifft2(ikx * fxh + iky * fyh))
        rhs = rhs + divf

    rhs_hat = np.fft.fft2(rhs)
    p_hat = -rhs_hat / k2
    p_hat[0,0] = 0.0  # fix gauge: zero-mean pressure
    p = np.real(np.fft.ifft2(p_hat))
    return p



def solve_navierstokes2d_vorticity(omega0, f_omega, nu = 1.0e-3, L=2*np.pi, T=1.0, dt_max=0.01, CFL = 0.5, check_conservation=False):
    """
    求解二维周期边界不可压 Navier–Stokes（采用伪谱 / vorticity–streamfunction 形式）。

    我们使用涡度-流函数形式避免显式求解压强：
        ω = ∂_x v - ∂_y u
        Δψ = -ω
        u =  ∂_y ψ,   v = -∂_x ψ
        ω_t + u ∂_x ω + v ∂_y ω = ν Δω + f_omega

    其中 f_omega 被解释为“涡度方程”的外力项（标量，形状 [N,N]）。
    若你希望输入的是动量方程中的向量力 F=(F_x,F_y)，请告诉我，我可以改成投影法（velocity-pressure form）。

    Args:
        u0: np.ndarray, shape (N, N) 或者 (3, N, N) 
            初始状态 (ω)
            或者 初始状态 (u, v, p)。此实现仅使用 u0[0], u0[1] 作为初始速度；u0[2] (压强) 将被忽略。
        f_omega: np.ndarray, shape (N, N)
            涡度方程外力项。
        L: float
            周期长度（x,y ∈ [0,L)）。
        T: float
            终止时间。
        dt_max: float
            最大时间步长（这里采用固定 dt = min(dt_max, T/ceil(T/dt_max))）。
        check_conservation: bool
            若为 True，输出能量/涡量等诊断（仅作参考，存在粘性/外力时不守恒）。
        compute_pressure
            
    Returns:
        u: np.ndarray, shape (3, N, N)
            最终状态 (u, v, p)。其中 p 在此实现中返回 0（因为涡度形式不显式求压强）。

    Require:
        - omega0.shape == (N, N), f_omega.shape == (N, N)
        - 周期边界
    """

    omega0 = np.asarray(omega0, dtype=np.float64)
    f_omega = np.asarray(f_omega, dtype=np.float64)
    assert omega0.ndim == 2 and omega0.shape[0] == omega0.shape[1], "omega0 must be (N,N)"
    N = omega0.shape[0]
    assert f_omega.shape == (N, N), "f_omega must be (N,N)"


    dx = dy = L / N
    # Wavenumbers and helpers (assumes you already have these utilities)
    kx, ky, ikx, iky, k2 = compute_wavenumbers(L, N)
    # 2/3 de-aliasing mask
    k_cut = (N // 3) * (2.0 * np.pi / L)
    dealias = (np.abs(kx) <= k_cut) & (np.abs(ky) <= k_cut)
    # Initial spectral vorticity
    omega_hat = fft2(omega0)
    omega_hat *= dealias
    omega_hat[0, 0] = 0.0
    # Forcing in spectral space
    f_omega_hat = fft2(f_omega) * dealias

    # ----------------------------
    # RHS of vorticity equation
    # ----------------------------
    def rhs_omega_hat(omega_hat_local):
        """
        ω_t = - u·∇ω + νΔω + f_omega
        computed pseudo-spectrally with 2/3 de-aliasing on nonlinear term.
        """
        # velocity from ω
        u, v = velocity_from_omega_hat(omega_hat_local, ikx, iky, k2)

        # gradients of ω in physical space
        omega_x, omega_y = grad_from_hat(omega_hat_local, ikx, iky)

        # nonlinear term in physical space: u ω_x + v ω_y
        adv = u * omega_x + v * omega_y

        # FFT and dealias
        adv_hat = fft2(adv)
        adv_hat *= dealias

        # diffusion and forcing (in spectral space)
        diff_hat = -nu * k2 * omega_hat_local
        return -(adv_hat) + diff_hat + f_omega_hat

    # ----------------------------
    # time integration: RK4
    # ----------------------------
    
    t = 0.0
    step = 0
    dt_diff = CFL / (nu * (k_cut**2) + 1e-30)
    while t < T:

        # Compute stable time step
        u, v = velocity_from_omega_hat(omega_hat, ikx, iky, k2)
        dt_adv = min(np.inf, CFL*dx/(np.max(np.abs(u))+ 1e-30), CFL*dy/(np.max(np.abs(v))+ 1e-30) )
        dt = min(T - t, dt_max, dt_diff,  dt_adv)

        k1h = rhs_omega_hat(omega_hat)
        k2h = rhs_omega_hat(omega_hat + 0.5 * dt * k1h)
        k3h = rhs_omega_hat(omega_hat + 0.5 * dt * k2h)
        k4h = rhs_omega_hat(omega_hat + dt * k3h)
        omega_hat = omega_hat + (dt / 6.0) * (k1h + 2*k2h + 2*k3h + k4h)

        # dealias (keep ω̂ stable)
        omega_hat *= dealias
        omega_hat[0, 0] = 0.0  # keep mean vorticity 0 (optional; common for periodic)

        t += dt
        step += 1
        if check_conservation and (step % 100 == 0 or t >= T - 1e-15):
            u, v = velocity_from_omega_hat(omega_hat, ikx, iky, k2)
            omega = np.real(ifft2(omega_hat))
            print(
                f"[t {t:.2f}/{T:.2f}] "
                f"E={kinetic_energy(u,v):.6e}  "
                f"Z={enstrophy(omega):.6e}"
            )

    omegaT = np.real(ifft2(omega_hat))

    return omegaT




def solve_navierstokes2d_equation(u0, f, nu = 1.0e-3, L=2*np.pi, T=1.0, dt_max=0.01, CFL = 0.5, check_conservation=False):
    """
    求解二维周期边界不可压 Navier–Stokes（采用伪谱 / vorticity–streamfunction 形式）。

    我们使用涡度-流函数形式避免显式求解压强：
        ω = ∂_x v - ∂_y u
        Δψ = -ω
        u =  ∂_y ψ,   v = -∂_x ψ
        ω_t + u ∂_x ω + v ∂_y ω = ν Δω + f

    其中 f 被解释为“涡度方程”的外力项（标量，形状 [N,N]）。
    若你希望输入的是动量方程中的向量力 F=(F_x,F_y)，请告诉我，我可以改成投影法（velocity-pressure form）。

    Args:
        u0: np.ndarray, (3, N, N) 
            初始状态 (u, v, p), 此实现仅使用 u0[0], u0[1] 作为初始速度；u0[2] (压强) 将被忽略。
        f: np.ndarray, shape (2, N, N)
            方程外力项 (fx, fy)。
        L: float
            周期长度（x,y ∈ [0,L)）。
        T: float
            终止时间。
        dt_max: float
            最大时间步长（这里采用固定 dt = min(dt_max, T/ceil(T/dt_max))）。
        check_conservation: bool
            若为 True，输出能量/涡量等诊断（仅作参考，存在粘性/外力时不守恒）。
        compute_pressure
            
    Returns:
        u: np.ndarray, shape (3, N, N)
            最终状态 (u, v, p)。其中 p 在此实现中返回 0（因为涡度形式不显式求压强）。

    Require:
        - u0.shape == (3, N, N), f.shape == (N, N)
        - 周期边界
    """

    N = u0.shape[1]
    # ----------------------------
    # spectral wave numbers
    # ----------------------------
    dx = dy = L / N
    kx, ky, ikx, iky, k2 = compute_wavenumbers(L, N)
    # 2/3 de-aliasing mask (Orszag rule)
    k_cut = (N // 3) * (2.0 * np.pi / L) * 1.0
    dealias = (np.abs(kx) <= k_cut) & (np.abs(ky) <= k_cut)

    # ----------------------------
    # equation initialization
    # initial vorticity in spectral space
    # ----------------------------
    assert u0.shape[0] == 3 and u0.shape[1] == u0.shape[2], "u0 must be (3,N,N)"
    # initialize with velocity
    u_init = np.asarray(u0[0], dtype=np.float64)
    v_init = np.asarray(u0[1], dtype=np.float64)
    omega0_hat = vorticity_hat_from_uv(u_init, v_init, ikx, iky)
    omega0 = np.real(ifft2(omega0_hat))

    f = np.asarray(f, dtype=np.float64)
    assert f.shape == (2, N, N), "f must be (2, N,N)"
    fx = np.asarray(f[0], dtype=np.float64)
    fy = np.asarray(f[1], dtype=np.float64)
    # forcing in spectral space (assumed vorticity forcing)
    f_omega = vorticity_forcing(fx, fy, ikx, iky)


    # ----------------------------
    # time integration: RK4
    # ----------------------------
    
    omegaT = solve_navierstokes2d_vorticity(omega0, f_omega, nu, L, T, dt_max, CFL, check_conservation)
    omegaT_hat = fft2(omegaT)
    # final velocity
    u, v = velocity_from_omega_hat(omegaT_hat, ikx, iky, k2)

    # pressure is not tracked in vorticity formulation; return zeros
    p = pressure_from_uv(u, v, kx, ky, k2, fx, fy)

    return np.stack([u, v, p], axis=0), omegaT








#########################################################################################################################################
# Test Taylor-Green vortex: an exact solution of 2D Navier-Stokes with zero forcing, useful for verifying numerical solvers.
#########################################################################################################################################

def taylor_green_exact(N, nu, t, L=2*np.pi):
    """
    Args:
        N: grid size
        nu: viscosity
        t: time
        L: domain length

    Returns:
        u, v, p: arrays [N,N]
    """
    x = np.linspace(0.0, L, N, endpoint=False)
    y = np.linspace(0.0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")

    fac_u = np.exp(-2.0 * nu * t)
    fac_p = np.exp(-4.0 * nu * t)

    u = np.sin(X) * np.cos(Y) * fac_u
    v = -np.cos(X) * np.sin(Y) * fac_u
    p = 0.25 * (np.cos(2*X) + np.cos(2*Y)) * fac_p
    return u, v, p

def rel_l2(a, b):
    """Relative L2 error."""
    num = np.sqrt(np.mean((a - b)**2))
    den = np.sqrt(np.mean(b**2)) + 1e-16
    return num / den

def test_taylor_green(solve_fn, N=128, nu=1e-3, T=1.0, dt_max=1e-3, L=2*np.pi):
    # initial condition from exact t=0
    u0, v0, p0 = taylor_green_exact(N, nu, t=0.0, L=L)
    u0_state = np.stack([u0, v0, p0], axis=0)

    # forcing = 0
    f = np.zeros((2, N, N), dtype=np.float64)

    # numerical solve
    uT_state, _ = solve_fn(u0_state, f, nu=nu, L=L, T=T, dt_max=dt_max, check_conservation=False)
    u_num, v_num, p_num = uT_state[0], uT_state[1], uT_state[2]

    # exact at time T
    u_ex, v_ex, p_ex = taylor_green_exact(N, nu, t=T, L=L)

    # errors (pressure from vorticity solver is usually not computed; compare u,v)
    err_u = rel_l2(u_num, u_ex)
    err_v = rel_l2(v_num, v_ex)
    err_p = rel_l2(p_num, p_ex)
    err_uv = np.sqrt(np.mean((u_num-u_ex)**2 + (v_num-v_ex)**2)) / (np.sqrt(np.mean(u_ex**2 + v_ex**2)) + 1e-16)

    print(f"[Taylor-Green] N={N}, nu={nu}, T={T}, dt_max={dt_max}")
    print(f"  relL2(u)  = {err_u:.3e}")
    print(f"  relL2(v)  = {err_v:.3e}")
    print(f"  relL2(p)  = {err_p:.3e}")
    print(f"  relL2(u,v)= {err_uv:.3e}")
    

    fig, axes = plt.subplots(3, 3, figsize=(12, 12), constrained_layout=True)
    fields = [(u0, "u0"), (v0, "v0"), (p0, "p0"),
              (u_ex, "uT"), (v_ex, "vT"), (p_ex, "pT"),
              (u_num, "uT_num"), (v_num, "vT_num"), (p_num, "pT_num")]
    
    for ax, (fld, title) in zip(axes.flat, fields):
        im = ax.imshow(fld, origin="lower", aspect="equal")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.show()


#########################################################################################################################################
# Test Kolmogorov flow: a 2D Navier-Stokes flow driven by a sinusoidal body force, which has a known laminar 
# solution and can transition to turbulence at higher Reynolds numbers. 
# Useful for testing solvers on forced flows and bifurcations.
########################################################################################################################################
def kolmogorov_momentum_forcing(N, n=4, A=1.0, L=2*np.pi):
    """
    Kolmogorov forcing in the momentum equation on Ω=[0,L)^2 (periodic).

    Momentum forcing (velocity-pressure form):
        ∂_t u + (u·∇)u = -∇p + νΔu + F,   ∇·u = 0,
    with
        F(x,y) = (A sin(ny), 0).

    Args:
        N: Grid resolution (N x N).
        n: Forcing wavenumber (in y-direction).
        A: Forcing amplitude.
        L: Domain length.

    Returns:
        Fx, Fy: arrays of shape [N, N].

    Require:
        Periodic boundary conditions on [0,L)^2.
    """
    y = np.linspace(0.0, L, N, endpoint=False)
    Y = np.tile(y.reshape(1, N), (N, 1))  # [N,N], constant in x, varies in y
    fx = A * np.sin(n * Y)
    fy = np.zeros_like(fx)
    return fx, fy


def kolmogorov_vorticity_forcing(N, n=4, A=1.0, L=2*np.pi):
    """
    Kolmogorov forcing in the vorticity equation on Ω=[0,L)^2 (periodic).

    Momentum forcing (in velocity-pressure form):
        F(x,y) = (A sin(ny), 0).

    Vorticity forcing (curl of F):
        f_ω = ∂_x F_y - ∂_y F_x = -A n cos(ny).

    Args:
        N: Grid resolution (N x N).
        n: Forcing wavenumber (in y-direction).
        A: Forcing amplitude.
        L: Domain length.

    Returns:
        f_omega: [N, N] array.
    """
    y = np.linspace(0.0, L, N, endpoint=False)
    Y = np.tile(y.reshape(1, N), (N, 1))  # [N,N], constant in x
    return -A * n * np.cos(n * Y)




def radial_energy_spectrum_uv(u, v, L=2*np.pi, dealias_mask=None):
    """
    Args:
        u, v: velocity components, arrays [N, N] on periodic grid [0,L)^2
        L: domain length
        dealias_mask: optional boolean mask [N,N] applied to Fourier modes

    Returns:
        k_shell: 1D array of shell wavenumbers (physical |k|)
        E_shell: 1D array, radial kinetic energy spectrum E(|k|)
    """
    N = u.shape[0]
    assert u.shape == (N, N) and v.shape == (N, N)

    uh = np.fft.fft2(u)
    vh = np.fft.fft2(v)

    if dealias_mask is not None:
        uh = uh * dealias_mask
        vh = vh * dealias_mask

    # mode energy (up to your FFT normalization convention)
    Ek = 0.5 * (np.abs(uh)**2 + np.abs(vh)**2) / N**4 # [N,N]

    # physical wavenumbers
    k1 = 2*np.pi * np.fft.fftfreq(N, d=L/N)
    kx, ky = np.meshgrid(k1, k1, indexing="ij")
    kmag = np.sqrt(kx**2 + ky**2)

    # bin shells in multiples of fundamental k0=2π/L
    k0 = 2*np.pi / L
    shell = np.floor(kmag / k0 + 1e-12).astype(int)
    smax = shell.max()

    E_shell = np.zeros(smax + 1, dtype=np.float64)
    for s in range(smax + 1):
        E_shell[s] = Ek[shell == s].sum()

    k_shell = k0 * np.arange(smax + 1)
    return k_shell, E_shell


def spectral_dissipation_from_E(k_shell, E_shell, nu):
    """
    Args:
        k_shell: 1D array of |k| for each shell
        E_shell: 1D array of radial kinetic energy spectrum
        nu: viscosity

    Returns:
        eps_shell: 1D array of spectral dissipation density ε(|k|) = 2 ν k^2 E(|k|)
    """
    return 2.0 * nu * (k_shell**2) * E_shell




def test_kolmogorov(
    solve_fn,
    N=128,
    nu=1e-3,
    n=4,
    A=1.0,
    T=5.0,
    dt_max=2e-3,
    L=2*np.pi,
):
    """
    Kolmogorov flow test (vorticity forcing) + visualization.

    We plot 5 panels:
      1) forcing f_ω
      2) vorticity ω at t=0
      3) vorticity ω at t=T/2
      4) vorticity ω at t=T
      5) radial spectrum of ω at t=T

    NOTE:
      This test calls solve_fn twice (for T/2 and T) because solve_fn returns only the final state.
      If you add snapshot output to your solver, we can avoid rerunning.
    """
    # --- initial condition ---
    print(f"[Kolmogorov] N={N}, nu={nu}, T={T}, dt_max={dt_max}, n={n}, A={A}")
    kx, ky, ikx, iky, k2 = compute_wavenumbers(L, N)


    eps = 1e-3
    u0 = eps * np.random.randn(N, N)
    v0 = eps * np.random.randn(N, N)
    # ω at t=0
    omega0 = vorticity_from_uv(u0, v0, ikx, iky)
    # --- forcing in vorticity equation ---
    f_omega = kolmogorov_vorticity_forcing(N, n=n, A=A, L=L)
    

    # run to T/2
    omega_half = solve_fn(omega0, f_omega, nu=nu, L=L, T=0.5*T, dt_max=dt_max, check_conservation=False)
    # continue to run to T
    omegaT = solve_fn(omega_half, f_omega, nu=nu, L=L, T=0.5*T, dt_max=dt_max, check_conservation=True)
    omegaT_hat = fft2(omegaT)
    # final velocity
    uT = velocity_from_omega_hat(omegaT_hat, ikx, iky, k2)


    # spectrum at T
    k_centers, E = radial_energy_spectrum_uv(uT[0], uT[1], L=2*np.pi, dealias_mask=None)
    # --- plotting: 2 rows x 3 cols (last subplot hidden) ---
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    # 1) forcing
    im0 = axes[0, 0].imshow(f_omega, origin="lower", aspect="equal")
    axes[0, 0].set_title(fr"Forcing $f_\omega(x,y)=-{A:.2f} \times {n}\cos({n}y)$")
    axes[0, 0].set_xticks([]); axes[0, 0].set_yticks([])
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # 2) omega at 0
    im1 = axes[0, 1].imshow(omega0, origin="lower", aspect="equal")
    axes[0, 1].set_title(r"Vorticity $\omega$ at $t=0$")
    axes[0, 1].set_xticks([]); axes[0, 1].set_yticks([])
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # 3) omega at T/2
    im2 = axes[0, 2].imshow(omega_half, origin="lower", aspect="equal")
    axes[0, 2].set_title(fr"Vorticity $\omega$ at $t={T:.2f}/2$")
    axes[0, 2].set_xticks([]); axes[0, 2].set_yticks([])
    fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # 4) omega at T
    im3 = axes[1, 0].imshow(omegaT, origin="lower", aspect="equal")
    axes[1, 0].set_title(r"Vorticity $\omega$ at $t={T:.2f}$")
    axes[1, 0].set_xticks([]); axes[1, 0].set_yticks([])
    fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # 5) spectrum at T (radial)
    axes[1, 1].loglog(k_centers, E, lw=2)
    axes[1, 1].loglog(k_centers, k_centers**(-3.0), "--", lw=2, label=f"k^{-3}")
    axes[1, 1].set_title(r"Radial spectrum of energy at $t=T$")
    axes[1, 1].set_xlabel(r"$|k|$")
    axes[1, 1].set_ylabel(r"$E(|k|)$")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    

    # hide unused panel
    axes[1, 2].axis("off")

    plt.show()

    return 


def generate_initial_conditions(M, N, k_max = 10, L=2*np.pi, normalize=True, seed=None):
    """
    生成二维 Navier Stokes 方程的随机初始 Vorticity
    
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
        N by M by M 矩阵
    """
    x = np.linspace(0, L, N, endpoint=False)
    dx = L/N
    grf = gaussian_random_field_2d(M, [N,N], [L,L], sigma=200.0, tau = 1.0, alpha = 2.0, bc_name = 'periodic', seed = seed, k_max = [k_max,k_max])
    return grf



def set_default_params():
    nT = 100
    T = 0.5
    N = 256
    k_max = 256
    L = 2*np.pi

    # viscosity and forcing parameters
    nu = 1e-3
    A = 1.0/4
    n = 4
    return nT, T, k_max, N, L, nu, A, n



def visualization():
    nT, T, k_max, N, L, nu, A, n = set_default_params()
    M = 3
    omega0 = generate_initial_conditions(M = M, N = N, k_max = k_max, seed=42)
    frames = [0, 1, nT//4, nT//4+1, nT//2, nT//2+1, nT-1, nT]
    fig, axs = plt.subplots(M, len(frames)+1, figsize=(20, 12))

    axs[0,0].set_title("f_omega")
    for j in range(len(frames)):
        axs[0,j+1].set_title(f"t = {T*frames[j]:.1f}")
    
    f_omega = kolmogorov_vorticity_forcing(N, n=n, A=A, L=L)
    for i in range(M):
        axs[i,0].imshow(f_omega, origin="lower", aspect="equal")

        omega_ref = np.zeros((nT+1, N, N))
        omega_ref[0,...] = omega0[i,...]

        for j in range(nT):
            omega_ref[j+1,:,:] = solve_navierstokes2d_vorticity(omega_ref[j,:,:], f_omega, nu, L=L, T=T, check_conservation=False)   
        
        
        vmin = min(np.min(omega_ref[t, ...]) for t in frames)
        vmax = max(np.max(omega_ref[t, ...]) for t in frames)
        for j in range(len(frames)):
            im = axs[i,j+1].imshow(omega_ref[frames[j],...], cmap='viridis',  aspect='equal', vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=axs[i,-1], fraction=0.046, pad=0.04)

    fig.tight_layout() 
    plt.show()
    plt.savefig("kolmogorovflow2d.pdf")
    
    
def generate_data():
    """
    生成二维Navier-Stokes方程的训练数据，保存在 data/navier_stokes_square/kolmogorovflow2d_data_XXXX.npz 中
    每个数据格式为 (nT+1) by N by N 的数组，分别表示 nT+1 个时间步的vorticity
    """
    nT, T, k_max, N, L, nu, A, n = set_default_params()
    ndata = 1000
    omega0 = generate_initial_conditions(M = ndata, N = N, k_max = k_max, seed=42)
    f_omega = kolmogorov_vorticity_forcing(N, n=n, A=A, L=L)
    Path('../../data/navier_stokes_square').mkdir(parents=True, exist_ok=True)

    for i in range(ndata):
        omega_ref = np.zeros((nT+1, N, N))
        omega_ref[0,...] = omega0[i,...]
        for j in range(nT):
            omega_ref[j+1,:,:] = solve_navierstokes2d_vorticity(omega_ref[j,:,:], f_omega, nu, L=L, T=T, check_conservation=False)   
        
        np.save(f"../../data/navier_stokes_square/kolmogorovflow2d_data_{i:04d}.npy", omega_ref)



    
    
if __name__ == "__main__":
    
    # test_taylor_green(solve_navierstokes2d_equation, N=128, nu=1e-1, T=5.0, dt_max=1e-3)
    # test_kolmogorov(solve_navierstokes2d_vorticity, N=128, nu=1e-3, n=4, A=1.0, T=10.0, dt_max=2e-3, L=2*np.pi)

    # visualization()
    generate_data()
    # extract_evolution_matrix() 