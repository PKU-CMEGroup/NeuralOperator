# Import default modules/packages
import numpy as np
import matplotlib.pyplot as plt
# Custom imports to this file
from scipy.fft import idct, idst, idctn, idstn
from scipy.interpolate import interp1d, RectBivariateSpline


def gaussian_random_field_1d(m, n, L, sigma, tau, alpha, bc_name, seed = None, k_max = None):
    '''
    Return m samples of a Gaussian random field on [0,L] with: 
        -- mean function m = 0
        -- covariance operator C = sigma^2 (-Delta + tau^2)^(-alpha),
    where Delta is the Laplacian with periodic, zero Dirichlet, or zero Neumann boundary conditions.
    We require sigma, tau > 0, alpha > d/2 = 1/2, such that C is in the trace class.
    Generally, when alpha is larger or tau is smaller, the eigenvalues decay faster.

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


    Output:
        grf: (m, n) numpy array, a GRF on the grid 
        when bc_name is ``neumann'' or ``dirichlet'', x = np.linspace(0, L, n, endpoint=True)
        when bc_name is ``periodic'',                 x = np.linspace(0, L, n, endpoint=False)
    
    Math:
        we generate random Gaussian {theta_k}, {theta'_k} ~ N(0,1)

        when bc_name is ``neumann'', the KL expansion is 
            sum_k theta_k sqrt{lambda_k} cos(pi k x/L) 
            where lambda_k = sigma^2 / (tau^2 + (pi k/L)^2)^{-alpha}

        when bc_name is ``dirichlet'', the KL expansion is 
            sum_k theta'_k sqrt{lambda_k} sin(pi k x/L)
            where lambda_k = sigma^2 / (tau^2 + (pi k/L)^2)^{-alpha}

        when bc_name is  ``periodic'', the KL expansion is 
            sum_k theta_k sqrt{lambda_k} cos(2pi k x/L) + theta'_k sqrt{lambda_k} sin(2pi k x/L)
            where lambda_k = sigma^2 / (tau^2 + (2pi k/L)^2)^{-alpha}
        
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
        eigs = sigma * (k**2 + tau**2) ** (-0.5 * alpha)
        eigs[k_max+1:] = 0.0

        u_hat = rng.normal(size=(m, n)) * eigs
        
        # zero_mean:
        u_hat[:, 0] = 0.0

        # idct normalized with sqrt(n/L)
        grf = idct(u_hat, type=2, norm="ortho", axis=-1)
        grf *= np.sqrt(n/L)
        # since grf uses cell center grid xc
        # interpolate to physical grid x containing the boundary of the domain [0, L]
        xc = np.arange(1/(2*n), (2*n+1)/(2*n), 1/n) * L # IDCT grid
        func_interp = interp1d(xc, grf, kind='cubic', fill_value='extrapolate')
        grf = func_interp(x)
        
        

        
    elif bc_name == 'dirichlet':  
        
        x = np.linspace(0, L, n, endpoint=True)
        dx = L / (n - 1)
        # Dirichlet eigen-wavenumbers
        k = np.pi * np.arange(1, n - 1) / L
        eigs = sigma * (k**2 + tau**2) ** (-0.5 * alpha)
        eigs[k_max:] = 0.0

        # Sample sine coefficients directly:
        # For orthonormal DST-I basis, just take N(0,1) and scale by eigs.
        u_hat = rng.normal(size=(m, n-2)) * eigs

        # Transform back to interior values
        grf = np.zeros((m, n), dtype=float)

        # idst normalized with sqrt((n-1)/L)
        grf[:, 1:-1] = idst(u_hat, type=1, norm="ortho", axis=-1)

        grf[:, 1:-1] /= np.sqrt(dx)
    
    
        
    elif bc_name == 'periodic': 
        assert(n % 2 == 0)
        # grid 
        x = np.linspace(0, L, n, endpoint=False)
        dx = L / n
        
        k = 2.0 * np.pi * np.fft.rfftfreq(n, d=dx)  # 2pi*0/L, 2pi*1/L, ... 2pi*(n/2)/L
        eigs = sigma * (k**2 + tau**2) ** (-0.5 * alpha) 
        eigs[k_max+1:] = 0.0

        # We'll sample u_hat so that (unitary) irfft gives the right real-space field
        u_hat = np.empty((m, k.size), dtype=np.complex128)

        # zero-mean
        u_hat[:, 0] = 0.0 
        
        idx = np.arange(1, k.size - 1)
        # Exclude Nyquist mode
        u_hat[:, -1] = 0.0
        re = rng.normal(size=(m, idx.size)) * (eigs[idx] * np.sqrt(n/2.0))
        im = rng.normal(size=(m, idx.size)) * (eigs[idx] * np.sqrt(n/2.0))
        u_hat[:, idx] = re + 1j * im

        # Transform back to real-space samples
        grf = np.fft.irfft(u_hat, axis=-1)
        grf /= np.sqrt(dx)
    else:
        raise ValueError(f"bc_name '{bc_name}' is not in ['neumann', 'dirichlet', 'periodic']")

    return grf
    
    
    
def gaussian_random_field_1d_test():        
    # parameters
    m = 10000
    n = 1024          # total grid points 
    L = 1.5
    tau = 2.0
    alpha = 2.0
    sigma = 1.0
    k_max = n
    
    bc_name = "periodic"
    x = np.linspace(0, L, n, endpoint=False)
    dx = L / n
    # generate samples (interior only)
    u = gaussian_random_field_1d(m=m, n=n, L=L, sigma=sigma, tau=tau, alpha=alpha, bc_name = bc_name, seed=0, k_max = k_max)
    # sample covariance in physical space
    U, svals_hat, V = np.linalg.svd(u, compute_uv=True, full_matrices=False)
    svals_hat = svals_hat/np.sqrt(m)*np.sqrt(dx)
    
    k = 2 * np.pi * np.arange(1, n//2+1) / L  # 2pi*0/L, 2pi*1/L, ... 2pi*(n/2)/L
    eigs = sigma * (k**2 + tau**2) ** (-0.5 * alpha)

    print("sample eigvals: ", svals_hat[0:5])
    print("reference eigvals: ",eigs[0:5])
    fig, axes = plt.subplots(1,3, figsize = (18, 6))
    axes[0].plot(x, u[:3,:].T)
    axes[0].set_xlabel("x")
    axes[0].set_title(f"Gaussian random field with {bc_name}")
    
    axes[1].semilogy(svals_hat[0::2], marker='o', label='sample eigvals (sin)')
    axes[1].semilogy(svals_hat[1::2], marker='o', label='sample eigvals (cos)')
    axes[1].semilogy(eigs, marker='x',      label='theory eigvals')
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
    


    bc_name = "dirichlet"
    x = np.linspace(0, L, n, endpoint=True)
    dx = L / (n-1)
    # generate samples (interior only)
    u = gaussian_random_field_1d(m=m, n=n, L=L, sigma=sigma, tau=tau, alpha=alpha, bc_name = bc_name, seed=0, k_max = k_max)
    # sample covariance in physical space
    U, svals_hat, V = np.linalg.svd(u, compute_uv=True, full_matrices=False)
    svals_hat = svals_hat/np.sqrt(m)*np.sqrt(dx)
    
    k = np.pi * np.arange(1, n - 1) / L  # 2pi*0/L, 2pi*1/L, ... 2pi*(n-2)/L
    eigs = sigma * (k**2 + tau**2) ** (-0.5 * alpha) 

    print("sample eigvals: ", svals_hat[0:5])
    print("reference eigvals: ",eigs[0:5])

    fig, axes = plt.subplots(1,3, figsize = (18, 6))
    axes[0].plot(x, u[:3,:].T)
    axes[0].set_xlabel("x")
    axes[0].set_title(f"Gaussian random field with {bc_name}")
    
    axes[1].semilogy(svals_hat, marker='o', label='sample eigvals')
    axes[1].semilogy(eigs, marker='x',      label='theory eigvals')
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
    

    
    


    bc_name = "neumann"
    x = np.linspace(0, L, n, endpoint=True)
    # generate samples (interior only)
    u = gaussian_random_field_1d(m=m, n=n, L=L, sigma=sigma, tau=tau, alpha=alpha, bc_name = bc_name, seed=0, k_max = k_max)

    # sample covariance in physical space
    U, svals_hat, V = np.linalg.svd(u, compute_uv=True, full_matrices=False)
    svals_hat = svals_hat/np.sqrt(m)*np.sqrt(L/n)
    
    k = np.pi * np.arange(1, n) / L  # 2pi*1/L, ... 2pi*(n - 1)/L
    eigs = sigma * (k**2 + tau**2) ** (-0.5 * alpha) 

    print("sample eigvals: ", svals_hat[0:5])
    print("reference eigvals: ",eigs[0:5])

    fig, axes = plt.subplots(1,3, figsize = (18, 6))
    axes[0].plot(x, u[:3,:].T)
    axes[0].set_xlabel("x")
    axes[0].set_title(f"Gaussian random field with {bc_name}")
    
    axes[1].semilogy(svals_hat, marker='o', label='sample eigvals')
    axes[1].semilogy(eigs, marker='x',      label='theory eigvals')
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



def gaussian_random_field_2d(m, n, L, sigma, tau, alpha, bc_name, seed = None, k_max = None):
    '''
    Return m samples of a Gaussian random field on [0,L]^2 with: 
        -- mean function m = 0
        -- covariance operator C = sigma^2 (-Delta + tau^2)^(-alpha),
    where Delta is the Laplacian with periodic, zero Dirichlet, or zero Neumann boundary conditions.
    We require sigma, tau > 0, alpha > d/2 = 1, such that C is in the trace class.
    Generally, when alpha is larger or tau is smaller, the eigenvalues decay faster.

    Arguments:
        m :         (int),   number of samples
        n :         (int, int),   number of mesh points
        
        L :         (float, float), domain length
        
        sigma:      (float), standard deviation
        
        tau:        (float), inverse lengthscale for Gaussian measure covariance operator
        
        alpha:      (alpha), regularity of covariance operator
        
        bc_name:    (str), ``neumann'' for Neumann BCs or ``dirichlet'' for Dirichlet BCs or ``periodic'' for periodic BCs
        
        seed:       (int), random seed

        k_max:      (int,int), maximum frequency number in each direction


    Output:
        grf: (m, n1, n2) numpy array, a GRF on the grid 
        when bc_name is ``neumann'' or ``dirichlet'', x = np.linspace(0, L, n, endpoint=True)
        when bc_name is ``periodic'',                 x = np.linspace(0, L, n, endpoint=False)
    
    Math:
        we generate random Gaussian {theta_k}, {theta'_k} ~ N(0,1)

        when bc_name is ``neumann'', the KL expansion is 
            sum_k theta_k sqrt{lambda_k} cos(pi k1 x1/L1)cos(pi k2 x2/L2) 
            where lambda_k = sigma^2 / (tau^2 + (pi k/L)^2)^{-alpha}

        when bc_name is ``dirichlet'', the KL expansion is 
            sum_k theta'_k sqrt{lambda_k} sin(pi k1 x1/L1)sin(pi k2 x2/L2)
            where lambda_k = sigma^2 / (tau^2 + (pi k/L)^2)^{-alpha}

        when bc_name is  ``periodic'', the KL expansion is 
            sum_k theta_k sqrt{lambda_k} cos(2pi k x/L) + theta'_k sqrt{lambda_k} sin(2pi k x/L)
            where lambda_k = sigma^2 / (tau^2 + (2pi k/L)^2)^{-alpha}
        
    '''
    
    rng = np.random.default_rng(seed)
    if k_max is None:
        k_max = n

    # Choose BCs
    if bc_name == 'neumann':  
        # grid 
        n1, n2 = n 
        L1, L2 = L
        x1, x2 = np.linspace(0, L1, n1, endpoint=True), np.linspace(0, L2, n2, endpoint=True)

        k1 = np.pi * np.arange(n1) / L1   #0, pi*1/L1, pi*1/L1, ... pi*(n1-1)/L1
        k2 = np.pi * np.arange(n2) / L2   #0, pi*1/L2, pi*1/L2, ... pi*(n2-1)/L2
        # Make 2D frequency grids of shape (n1, n2)
        K1, K2 = np.meshgrid(k1, k2, indexing="ij")    # both shape (n1, n2)
        eigs = sigma * (K1**2 + K2**2 + tau**2) ** (-0.5 * alpha)
        eigs[k_max[0]+1:, :] = 0.0
        eigs[:,k_max[1]+1:] = 0.0

        u_hat = rng.normal(size=(m, n1, n2)) * eigs
        
        # zero_mean:
        u_hat[:, 0, 0] = 0.0

        # idct normalized with sqrt(n/L)
        grf = idctn(u_hat, type=2, norm="ortho", axes=(-2,-1))
        grf *= np.sqrt(n1/L1 * n2/L2)
        
        # since grf uses cell center grid xc
        # interpolate to physical grid x containing the boundary of the domain [0,L]
        x1c, x2c = np.arange(1/(2*n1), (2*n1+1)/(2*n1), 1/n1) * L1, np.arange(1/(2*n2), (2*n2+1)/(2*n2), 1/n2) * L2  # IDCT grid

        # TODO: more accurate interpolation using RectBivariateSpline, but it is very slow for large m.
        # for i in range(m):
        #     func_interp = RectBivariateSpline(x1c, x2c, grf[i])
        #     grf[i] = func_interp(x1, x2)

        grf_tmp = interp1d(x2c, grf, kind="linear", axis=-1, fill_value="extrapolate")(x2)
        grf = interp1d(x1c, grf_tmp, kind="linear", axis=-2, fill_value="extrapolate")(x1)


        
    elif bc_name == 'dirichlet':
        # grid
        n1, n2 = n 
        L1, L2 = L
        x1, x2 = np.linspace(0, L1, n1, endpoint=True), np.linspace(0, L2, n2, endpoint=True)
        dx1, dx2 = L1 / (n1 - 1), L2 / (n2 - 1)
        
        k1 = np.pi * np.arange(1, n1 - 1) / L1   #pi*1/L1, ... pi*(n1-2)/L1
        k2 = np.pi * np.arange(1, n2 - 1) / L2   #pi*1/L2, ... pi*(n2-2)/L2
        # Make 2D frequency grids of shape (n1-2, n2-2)
        K1, K2 = np.meshgrid(k1, k2, indexing="ij")    # both shape (n2, n1//2+1)
        eigs = sigma * (K1**2 + K2**2 + tau**2) ** (-0.5 * alpha)
        
        eigs[k_max[0]:, :] = 0.0
        eigs[:, k_max[1]:] = 0.0

        u_hat = rng.normal(size=(m, n1-2, n2-2)) * eigs
        # Transform back to interior values
        grf = np.zeros((m, n1, n2), dtype=float)

        # idstn normalized with sqrt(n-1/L)
        grf[:, 1:-1, 1:-1] = idstn(u_hat, type=1, norm="ortho", axes=(-2,-1))
        grf /= np.sqrt(dx1 * dx2)
 
        
    elif bc_name == 'periodic': 

        # grid
        n1, n2 = n 
        assert(n1 % 2 == 0 and n2 % 2 == 0)
        L1, L2 = L
        dx1, dx2 = L1 / n1, L2 / n2
        x1, x2 = np.linspace(0, L1, n1, endpoint=False), np.linspace(0, L2, n2, endpoint=False)

        k1 = 2.0 * np.pi * np.fft.fftfreq(n1, d=dx1)    # 2pi*0/L1, 2pi*1/L1, ... -2pi*(n1/2)/L1, -2pi*(n1/2-1)/L1, ..., -2pi*1/L1
        k2 = 2.0 * np.pi * np.fft.rfftfreq(n2, d=dx2)   # 2pi*0/L2, 2pi*1/L2, ... 2pi*(n2/2)/L2
        # Make 2D frequency grids of shape (n1, n2//2+1)
        K1, K2 = np.meshgrid(k1, k2, indexing="ij")    # both shape (n2, n1//2+1)
        eigs = sigma * (K1**2 + K2**2 + tau**2) ** (-0.5 * alpha)
        
        eigs[k_max[0]+1:-k_max[0], :] = 0.0
        eigs[:, k_max[1]+1:] = 0.0

        # We'll sample u_hat so that (unitary) irfft gives the right real-space field
        u_hat = np.zeros((m, k1.size, k2.size), dtype=np.complex128)

        idx1, idx2 = np.arange(1, n1//2), np.arange(1, n2//2)
        
        re = rng.normal(size=(m, k1.size, idx2.size)) * (eigs[:,idx2] * np.sqrt(n1*n2/2))
        im = rng.normal(size=(m, k1.size, idx2.size)) * (eigs[:,idx2] * np.sqrt(n1*n2/2))
        u_hat[:, :,idx2] = re + 1j * im

        # 0 frequency in x2 direction is special since it's real-valued
        re = rng.normal(size=(m, idx1.size)) * (eigs[idx1,0] * np.sqrt(n1*n2/2))
        im = rng.normal(size=(m, idx1.size)) * (eigs[idx1,0] * np.sqrt(n1*n2/2))
        u_hat[:, idx1, 0]  = re + 1j * im
        u_hat[:, -idx1, 0] = re - 1j * im
        u_hat[:, n1//2, 0]  = rng.normal(size=(m)) * (eigs[n1//2,0] * np.sqrt(n1*n2))

        # n2//2 Nyquist frequency is also special since it's real-valued
        re = rng.normal(size=(m, idx1.size)) * (eigs[idx1,-1] * np.sqrt(n1*n2/2))
        im = rng.normal(size=(m, idx1.size)) * (eigs[idx1,-1] * np.sqrt(n1*n2/2))
        u_hat[:, idx1, -1]  = re + 1j * im
        u_hat[:, -idx1, -1] = re - 1j * im
        u_hat[:, n1//2, -1]  = rng.normal(size=(m)) * (eigs[n1//2,-1] * np.sqrt(n1*n2))
        
        # zero-mean
        u_hat[:, 0, 0] = 0.0 

        # Transform back to real-space samples
        grf = np.fft.irfft2(u_hat, axes=(-2, -1))
        grf /= np.sqrt(dx1 * dx2)
    else:
        raise ValueError(f"bc_name '{bc_name}' is not in ['neumann', 'dirichlet', 'periodic']")

    return grf

def gaussian_random_field_2d_test():   
    print("gaussian_random_field_2d_test")     
    # parameters
    m = 10000
    n1, n2 = 128, 64
    L1, L2 = 2*np.pi, 2*np.pi
    k1, k2 = 64, 32
    n = [n1, n2]          # total grid points 
    L = [L1, L2]
    tau = 2.0
    alpha = 2.0
    sigma = 1.0
    k_max = [k1, k2]
    
    bc_name = "periodic"
    x1, x2 = np.linspace(0, L1, n1, endpoint=False), np.linspace(0, L2, n2, endpoint=False)
    dx1, dx2 = L1/n1, L2/n2
    # generate samples (interior only)
    u = gaussian_random_field_2d(m=m, n=n, L=L, sigma=sigma, tau=tau, alpha=alpha, bc_name = bc_name, seed=0, k_max = k_max)
    # sample covariance in physical space
    U, svals_hat, V = np.linalg.svd(u.reshape(-1,n1*n2), compute_uv=True, full_matrices=False)
    svals_hat = svals_hat/np.sqrt(m)*np.sqrt(dx1*dx2)
    
    # Make 2D frequency grids of shape (n2, n1)
    k1 = 2.0 * np.pi * np.fft.fftfreq(n1, d=dx1)   # 2pi*0/L1, 2pi*1/L1, ... -2pi*(n1/2)/L1, -2pi*(n1/2-1)/L1, ..., -2pi*1/L1
    k2 = 2.0 * np.pi * np.fft.fftfreq(n2, d=dx2)   # 2pi*0/L2, 2pi*1/L2, ... -2pi*(n2/2)/L2, -2pi*(n2/2-1)/L2, ..., -2pi*1/L2
    # Make 2D frequency grids of shape (n2, n1)
    K1, K2 = np.meshgrid(k1, k2, indexing="ij")    # both shape (n2, n1)
    eigs = sigma * (K1**2 + K2**2 + tau**2) ** (-0.5 * alpha)  
    eigs[k_max[0]+1:-k_max[0], :] = 0.0
    eigs[:, k_max[1]+1:-k_max[1]] = 0.0
    eigs_seq = np.sort(eigs.reshape(-1))[::-1][1:]
    print("sample eigvals: ", svals_hat[0:5])
    print("reference eigvals: ",eigs_seq[0:5])
    
    fig, axes = plt.subplots(1,2, figsize = (12, 6))
    im = axes[0].pcolormesh(x1, x2, u[0,:,:].T)
    plt.colorbar(im, ax=axes[0])
    axes[0].set_xlabel("x_1")
    axes[0].set_ylabel("x_2")
    axes[0].set_title(f"Gaussian random field with {bc_name}")
    
    axes[1].semilogy(svals_hat, marker='o', label='sample eigvals')
    axes[1].semilogy(eigs_seq, marker='x',      label='theory eigvals')
    axes[1].set_xlabel("rank")
    axes[1].set_title("Eigenvalue comparison")
    axes[1].legend()

    fig.tight_layout()
    


    bc_name = "dirichlet"
    x1, x2 = np.linspace(0, L1, n1, endpoint=True), np.linspace(0, L2, n2, endpoint=True)
    dx1, dx2 = L1/(n1 - 1), L2/(n2 - 1)
    # generate samples (interior only)
    u = gaussian_random_field_2d(m=m, n=n, L=L, sigma=sigma, tau=tau, alpha=alpha, bc_name = bc_name, seed=0, k_max = k_max)
    # sample covariance in physical space
    U, svals_hat, V = np.linalg.svd(u.reshape(-1,n1*n2), compute_uv=True, full_matrices=False)
    svals_hat = svals_hat/np.sqrt(m)*np.sqrt(dx1*dx2)
    
    k1 = np.pi * np.arange(1, n1 - 1) / L1   #pi*1/L1, ... pi*(n1-2)/L1
    k2 = np.pi * np.arange(1, n2 - 1) / L2   #pi*1/L2, ... pi*(n2-2)/L2
    # Make 2D frequency grids of shape (n1, n2//2+1)
    K1, K2 = np.meshgrid(k1, k2, indexing="ij")    # both shape (n2, n1//2+1)
    eigs = sigma * (K1**2 + K2**2 + tau**2) ** (-0.5 * alpha)
    
    eigs[k_max[0]:, :] = 0.0
    eigs[:, k_max[1]:] = 0.0

    eigs_seq = np.sort(eigs.reshape(-1))[::-1]
    print("sample eigvals: ", svals_hat[0:5])
    print("reference eigvals: ",eigs_seq[0:5])

    fig, axes = plt.subplots(1,2, figsize = (12, 6))
    im = axes[0].pcolormesh(x1, x2, u[0,:,:].T)
    plt.colorbar(im, ax=axes[0])
    axes[0].set_xlabel("x_1")
    axes[0].set_ylabel("x_2")
    axes[0].set_title(f"Gaussian random field with {bc_name}")
    
    axes[1].semilogy(svals_hat, marker='o', label='sample eigvals')
    axes[1].semilogy(eigs_seq, marker='x',      label='theory eigvals')
    axes[1].set_xlabel("rank")
    axes[1].set_title("Eigenvalue comparison")
    axes[1].legend()


    
    


    bc_name = "neumann"
    x1, x2 = np.linspace(0, L1, n1, endpoint=True), np.linspace(0, L2, n2, endpoint=True)
    dx1, dx2 = L1/(n1 - 1), L2/(n2 - 1)
    # generate samples (interior only)
    u = gaussian_random_field_2d(m=m, n=n, L=L, sigma=sigma, tau=tau, alpha=alpha, bc_name = bc_name, seed=0, k_max = k_max)
    # sample covariance in physical space
    U, svals_hat, V = np.linalg.svd(u.reshape(-1,n1*n2), compute_uv=True, full_matrices=False)
    svals_hat = svals_hat/np.sqrt(m)*np.sqrt(L1*L2/n1/n2)
    
    k1 = np.pi * np.arange(n1) / L1   #0, pi*1/L1, pi*1/L1, ... pi*(n1-1)/L1
    k2 = np.pi * np.arange(n2) / L2   #0, pi*1/L2, pi*1/L2, ... pi*(n2-1)/L2
    # Make 2D frequency grids of shape (n1, n2)
    K1, K2 = np.meshgrid(k1, k2, indexing="ij")    # both shape (n1, n2)
    eigs = sigma * (K1**2 + K2**2 + tau**2) ** (-0.5 * alpha)
    
    eigs[k_max[0]:, :] = 0.0
    eigs[:, k_max[1]:] = 0.0

    eigs_seq = np.sort(eigs.reshape(-1))[::-1][1:]
    print("sample eigvals: ", svals_hat[0:5])
    print("reference eigvals: ",eigs_seq[0:5])

    fig, axes = plt.subplots(1,2, figsize = (12, 6))
    im = axes[0].pcolormesh(x1, x2, u[0,:,:].T)
    plt.colorbar(im, ax=axes[0])
    axes[0].set_xlabel("x_1")
    axes[0].set_ylabel("x_2")
    axes[0].set_title(f"Gaussian random field with {bc_name}")
    
    axes[1].semilogy(svals_hat, marker='o', label='sample eigvals')
    axes[1].semilogy(eigs_seq, marker='x',      label='theory eigvals')
    axes[1].set_xlabel("rank")
    axes[1].set_title("Eigenvalue comparison")
    axes[1].legend()

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    gaussian_random_field_1d_test()
    gaussian_random_field_2d_test()