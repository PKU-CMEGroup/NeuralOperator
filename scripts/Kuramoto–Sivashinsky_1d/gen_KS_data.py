import numpy as np

def generate_random_IC(N, L, seed=None, amplitude=0.1, num_modes=8):
    """
    Generate a random initial condition for 1D KS equation.

    Parameters
    ----------
    N : int
        Number of spatial points
    L : float
        Domain length
    seed : int or None
        Random seed for reproducibility
    amplitude : float
        Scale of initial condition
    num_modes : int
        Number of random Fourier modes to combine

    Returns
    -------
    u0 : ndarray, shape (N,)
        Random initial condition
    """
    if seed is not None:
        np.random.seed(seed)
    x = np.linspace(0, L, N, endpoint=False)
    u0 = np.zeros_like(x)
    # Random combination of low Fourier modes without added noise
    for k in range(1, num_modes + 1):
        a_k = np.random.randn()/np.sqrt(k)
        b_k = np.random.randn()/np.sqrt(k)
        u0 += a_k*np.cos(2*np.pi*k*x/L) + b_k*np.sin(2*np.pi*k*x/L)
    u0 += np.random.randn()
    u0 *= amplitude / np.sqrt(num_modes)
    return u0



def solve_KS(u0, L, T, dt, beta):
    """
    Solve 1D Kuramoto–Sivashinsky equation using ETDRK4.
    
    Parameters
    ----------
    u0 : np.ndarray
        Initial condition, shape (N,)
    L : float
        Spatial domain length
    T : float
        Final time
    dt : float
        Time step
    beta : float
        Coefficient for u_xxxx term
    
    Returns
    -------
    u : np.ndarray
        Solution at t = T, shape (N,)
    """
    N = len(u0)
    x = np.linspace(0, L, N, endpoint=False)
    k = 2*np.pi*np.fft.fftfreq(N, L/N)  # wave numbers
    k2 = k**2
    k4 = k**4

    # Linear operator
    L_hat = k2/beta**2 - k4/beta**4
    E = np.exp(dt*L_hat)
    E2 = np.exp(dt*L_hat/2.0)

    # ETDRK4 coefficients
    M = 32  # number of points for complex means
    r = np.exp(1j*np.pi*(np.arange(1,M+1)-0.5)/M)
    LR = dt*L_hat[:,None] + r[None,:]
    Q = dt*np.mean( (np.exp(LR/2)-1)/LR , axis=1 ).real
    f1 = dt*np.mean( (-4 - LR + np.exp(LR)*(4 - 3*LR + LR**2)) / LR**3 , axis=1 ).real
    f2 = dt*np.mean( ( 2 + LR + np.exp(LR)*(-2 + LR) ) / LR**3 , axis=1 ).real
    f3 = f3 = dt*np.mean((-4 - 3*LR - LR**2 + np.exp(LR)*(4 - LR)) / LR**3,axis=1).real

    u_hat = np.fft.fft(u0)

    nsteps = int(T/dt)
    for n in range(nsteps):
        u = np.fft.ifft(u_hat).real
        Nv = -0.5j * k * np.fft.fft(u**2/(2*beta**2))
        a = E2*u_hat + Q*Nv
        ua = np.fft.ifft(a).real
        Na = -0.5j * k * np.fft.fft(ua**2/(2*beta**2))
        b = E2*u_hat + Q*Na
        ub = np.fft.ifft(b).real
        Nb = -0.5j * k * np.fft.fft(ub**2/(2*beta**2))
        c = E2*a + Q*(2*Nb - Nv)
        uc = np.fft.ifft(c).real
        Nc = -0.5j * k * np.fft.fft(uc**2/(2*beta**2))
        u_hat = E*u_hat + f1*Nv + 2*f2*(Na + Nb) + f3*Nc

    u = np.fft.ifft(u_hat).real
    return u



# Generate KS dataset: 100 samples, each sample has 150 time steps on 200 spatial points
N_SAMPLES = 100
N_POINTS = 200
L = 20.0
NUM_MODES = 5
N_STEPS = 150
DT = 0.2
dt = 0.0001
BETA = 1

np.random.seed(0)
data = np.zeros((N_SAMPLES, N_STEPS, N_POINTS), dtype=np.float64)
x = None

for i in range(N_SAMPLES):
    index = i
    u0 = generate_random_IC(N_POINTS, L, seed=240 + i, amplitude=1, num_modes=NUM_MODES)
    traj = np.zeros((N_STEPS, N_POINTS))
    traj[0,:]=u0
    u_curr = u0.copy()
    for n in range(1, N_STEPS):
        u_curr = solve_KS(u_curr, beta = BETA, L = L , T = DT, dt = dt)
        u_copy = u_curr.copy()
        traj[n,:] = u_copy

    data[i] = traj
    #np.save("data_KS_200/data_"+str(index).zfill(5),traj)
np.save("data_KS_200/KS_200_data_modes5",data)
