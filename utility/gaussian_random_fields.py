# Import default modules/packages
import numpy as np
import matplotlib.pyplot as plt
# Custom imports to this file
from scipy.fft import idct, idst
from scipy.interpolate import interp1d


def gaussian_random_field_1d(m, n, L, sigma, tau, alpha, bc_name, seed = None, k_max = None):
    '''
    Return m samples of a Gaussian random field on [0,L] with: 
        -- mean function m = 0
        -- covariance operator C = sigma^2 (-Delta + tau^2)^(-alpha),
    where Delta is the Laplacian with periodic, zero Dirichlet, or zero Neumann boundary conditions.
    
    Arguments:
        m :         (int),   number of samples
        n :         (int),   number of mesh points
        
        L :         (float), domain length
        
        sigma:      (float), standard deviation
        
        tau:        (float), inverse lengthscale for Gaussian measure covariance operator
        
        alpha:      (alpha), regularity of covariance operator
        
        bc_name:    (str), ``neumann'' for Neumann BCs or ``dirichlet'' for Dirichlet BCs or ``periodic'' for periodic BCs
        
        seed:       (int), random seed

        k_max:      (int), maximum frequency number

    Require:
        sigma, tau > 0, alpha > d/2 = 1/2

    Output:
        grf: (m, n) numpy array, a GRF on the grid 
        when bc_name is ``neumann'' or ``dirichlet'', x = np.linspace(0, L, n, endpoint=True)
        when bc_name is ``periodic'',                 x = np.linspace(0, L, n, endpoint=False)
    
    Math:
        we generate random Gaussian {theta_k}, {theta'_k} ~ N(0,1)

        when bc_name is ``neumann'', the KL expansion is 
            sum_k theta_k \sqrt{\lambda_k} cos(pi k x/L) 
            where \lambda_k = sigma^2 / (tau^2 + (pi k/L)^2)^{-alpha}

        when bc_name is ``dirichlet'', the KL expansion is 
            sum_k theta'_k \sqrt{\lambda_k} sin(pi k x/L)
            where \lambda_k = sigma^2 / (tau^2 + (pi k/L)^2)^{-alpha}

        when bc_name is  ``periodic'', the KL expansion is 
            sum_k theta_k \sqrt{\lambda_k} cos(2pi k x/L) + theta'_k \sqrt{\lambda_k} sin(2pi k x/L)
            where \lambda_k = sigma^2 / (tau^2 + (2pi k/L)^2)^{-alpha}
        
    '''
    
    rng = np.random.default_rng(seed)
    if k_max is None:
        k_max = n

    # Choose BCs
    if bc_name == 'neumann':  
        # grid 
        x = np.linspace(0, L, n, endpoint=True)
        dx = L / (n - 1)

        k = np.pi * np.arange(n) / L
        filt = sigma * (k**2 + tau**2) ** (-0.5 * alpha)
        filt[k_max+1:] = 0.0

        u_hat = rng.normal(size=(m, n)) * filt
        
        # zero_mean:
        u_hat[:, 0] = 0.0

        # idct automatically normalized with sqrt(n/2)
        grf = idct(u_hat, type=2, norm="ortho", axis=-1)
        # since grf uses cell center grid xc
        # interpolate to physical grid x containing the boundary of the domain [0,1]
        xc = np.arange(1/(2*n), (2*n+1)/(2*n), 1/n) * L # IDCT grid
        func_interp = interp1d(xc, grf, kind='cubic', fill_value='extrapolate')
        grf = func_interp(x)

        
    elif bc_name == 'dirichlet':  
        
        x = np.linspace(0, L, n, endpoint=True)
        dx = L / (n - 1)
        # Dirichlet eigen-wavenumbers
        k = np.pi * np.arange(1, n - 1) / L
        filt = sigma * (k**2 + tau**2) ** (-0.5 * alpha)
        filt[k_max:] = 0.0

        # Sample sine coefficients directly:
        # For orthonormal DST-I basis, just take N(0,1) and scale by filt.
        u_hat = rng.normal(size=(m, n-2)) * filt

        # Transform back to interior values
        grf = np.zeros((m, n), dtype=float)

        # idct automatically normalized with sqrt(n-1/2)
        grf[:, 1:-1] = idst(u_hat, type=1, norm="ortho", axis=-1)
   
    
    
        
    elif bc_name == 'periodic': 
        assert(n % 2 == 0)
        # grid 
        x = np.linspace(0, L, n, endpoint=False)
        dx = L / n
        
        k = 2.0 * np.pi * np.fft.rfftfreq(n, d=dx)  # 2pi*0/L, 2pi*1/L, ... 2pi*(n/2)/L
        filt = sigma * (k**2 + tau**2) ** (-0.5 * alpha) 
        filt[k_max+1:] = 0.0

        # We'll sample u_hat so that (unitary) irfft gives the right real-space field
        u_hat = np.empty((m, k.size), dtype=np.complex128)

        # zero-mean
        u_hat[:, 0] = 0.0 
        
        idx = np.arange(1, k.size - 1)
        u_hat[:, -1] = rng.normal(size=m) * filt[-1] + 0j
        re = rng.normal(size=(m, idx.size)) * (filt[idx] * np.sqrt(n/2.0))
        im = rng.normal(size=(m, idx.size)) * (filt[idx] * np.sqrt(n/2.0))
        u_hat[:, idx] = re + 1j * im

        # Transform back to real-space samples
        grf = np.fft.irfft(u_hat, axis=-1)
    
    else:
        raise ValueError(f"bc_name '{bc_name}' is not in ['neumann', 'dirichlet', 'periodic']")

    return grf
    
    
    
def gaussian_random_field_1d_test():        
    # parameters
    m = 10000
    n = 512          # total grid points 
    L = 2*np.pi
    tau = 1.0
    alpha = 1.0
    sigma = 1.0
    k_max = n//3
    
    bc_name = "periodic"
    x = np.linspace(0, L, n, endpoint=False)
    dx = L/n
    # generate samples (interior only)
    u = gaussian_random_field_1d(m=m, n=n, L=L, sigma=sigma, tau=tau, alpha=alpha, bc_name = bc_name, seed=0, k_max = k_max)
    # sample covariance in physical space
    U, svals_hat, V = np.linalg.svd(u, compute_uv=True, full_matrices=False)
    svals_hat = svals_hat/np.sqrt(m)
    
    k = 2 * np.pi * np.arange(1, n//2+1) / L  # 2pi*0/L, 2pi*1/L, ... 2pi*(n/2)/L
    filt = sigma * (k**2 + tau**2) ** (-0.5 * alpha) 

    print("sample eigvals: ", svals_hat[0:5])
    print("reference eigvals: ",filt[0:5])
    fig, axes = plt.subplots(1,3, figsize = (18, 6))
    axes[0].plot(x, u[:3,:].T)
    axes[0].set_xlabel("x")
    axes[0].set_title(f"Gaussian random field with {bc_name}")
    
    axes[1].semilogy(svals_hat[0::2], marker='o', label='sample eigvals (sin)')
    axes[1].semilogy(svals_hat[1::2], marker='o', label='sample eigvals (cos)')
    axes[1].semilogy(filt, marker='x',      label='theory eigvals')
    axes[1].set_xlabel("rank")
    axes[1].set_title("Eigenvalue comparison")
    axes[1].legend()

    k = 3
    axes[2].plot(x, V[2*(k-1):2*(k),:].T, label=f"sample eigvec")
    sinx = np.sin(2*np.pi*k*x/L)
    sinx /= np.linalg.norm(sinx)
    cosx = np.cos(2*np.pi*k*x/L)
    cosx /= np.linalg.norm(cosx)
    axes[2].plot(x, sinx, '--', label=f"theory eigvec (sin{k}x)")
    axes[2].plot(x, cosx, '--', label=f"theory eigvec (cos{k}x)")
    axes[2].set_xlabel("x")
    axes[2].set_title(f"Eigenvector comparison")
    axes[2].legend()
    
    fig.tight_layout()
    plt.show()


    bc_name = "dirichlet"
    x = np.linspace(0, L, n, endpoint=True)
    dx = L/(n - 1)
    # generate samples (interior only)
    u = gaussian_random_field_1d(m=m, n=n, L=L, sigma=sigma, tau=tau, alpha=alpha, bc_name = bc_name, seed=0, k_max = k_max)
    # sample covariance in physical space
    U, svals_hat, V = np.linalg.svd(u, compute_uv=True, full_matrices=False)
    svals_hat = svals_hat/np.sqrt(m)
    
    k = np.pi * np.arange(1, n - 1) / L  # 2pi*0/L, 2pi*1/L, ... 2pi*(n-2)/L
    filt = sigma * (k**2 + tau**2) ** (-0.5 * alpha) 

    print("sample eigvals: ", svals_hat[0:5])
    print("reference eigvals: ",filt[0:5])

    fig, axes = plt.subplots(1,3, figsize = (18, 6))
    axes[0].plot(x, u[:3,:].T)
    axes[0].set_xlabel("x")
    axes[0].set_title(f"Gaussian random field with {bc_name}")
    
    axes[1].semilogy(svals_hat, marker='o', label='sample eigvals')
    axes[1].semilogy(filt, marker='x',      label='theory eigvals')
    axes[1].set_xlabel("rank")
    axes[1].set_title("Eigenvalue comparison")
    axes[1].legend()

    k = 3
    axes[2].plot(x, V[k-1,:], label=f"sample eigvec")
    sinx = np.sin(np.pi*(k)*x/L)
    sinx /= np.linalg.norm(sinx)
    axes[2].plot(x, sinx, '--', label=f"theory eigvec (sin{k}x)")
    axes[2].set_xlabel("x")
    axes[2].set_title(f"Eigenvector comparison")
    axes[2].legend()
    fig.tight_layout()
    plt.show()

    
    


    bc_name = "neumann"
    x = np.linspace(0, L, n, endpoint=True)
    dx = L/(n-1)
    # generate samples (interior only)
    u = gaussian_random_field_1d(m=m, n=n, L=L, sigma=sigma, tau=tau, alpha=alpha, bc_name = bc_name, seed=0, k_max = k_max)

    # sample covariance in physical space
    U, svals_hat, V = np.linalg.svd(u, compute_uv=True, full_matrices=False)
    svals_hat = svals_hat/np.sqrt(m)
    
    k = np.pi * np.arange(1, n) / L  # 2pi*1/L, ... 2pi*(n - 1)/L
    filt = sigma * (k**2 + tau**2) ** (-0.5 * alpha) 

    print("sample eigvals: ", svals_hat[0:5])
    print("reference eigvals: ",filt[0:5])

    fig, axes = plt.subplots(1,3, figsize = (18, 6))
    axes[0].plot(x, u[:3,:].T)
    axes[0].set_xlabel("x")
    axes[0].set_title(f"Gaussian random field with {bc_name}")
    
    axes[1].semilogy(svals_hat, marker='o', label='sample eigvals')
    axes[1].semilogy(filt, marker='x',      label='theory eigvals')
    axes[1].set_xlabel("rank")
    axes[1].set_title("Eigenvalue comparison")
    axes[1].legend()

    k = 3
    axes[2].plot(x, V[k-1,:], label=f"sample eigvec")
    cosx = np.cos(np.pi*(k)*x/L)
    cosx /= np.linalg.norm(cosx)
    axes[2].plot(x, cosx, '--', label=f"theory eigvec (cos{k}x)")
    axes[2].set_xlabel("x")
    axes[2].set_title(f"Eigenvector comparison")
    axes[2].legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    gaussian_random_field_1d_test()