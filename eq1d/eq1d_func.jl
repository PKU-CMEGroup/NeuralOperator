using Distributions
using Random
using LinearAlgebra
using SparseArrays
using PyPlot


function generate_1d_rf(θ::Array{FT,2}, Nx::IT;  L::FT=1.0, d::FT=2.0, τ::FT=3.0, boundary_condition="Dirichlet") where {FT<:AbstractFloat, IT<:Int}
    Nθ, Nk = size(θ)
    
    fs = zeros(FT, Nθ, Nx)
    
    xx = LinRange(0, L, Nx)
    
    if boundary_condition=="Dirichlet"
        for k = 1:Nk
            fs += θ[:, k]*sqrt((π^2*k^2/L^2  + τ^2)^(-d)) * sin.(π * k * xx/L)'
        end
    elseif boundary_condition=="periodic"
        for k = 1:div(Nk,2)
            fs += (θ[:,2k-1]*sqrt(2)*cos.(2π*k*xx/L)' + θ[:,2k]*sqrt(2)*sin.(2π*k*xx/L)')*sqrt((4*π^2*k^2/L^2 + τ^2)^(-d)) 
        end 
    end

    return fs
end
    


function generate_1d_poisson_ref(θ::Array{FT,2}, Nx::IT;  d::FT=1.0, τ::FT=3.0) where {FT<:AbstractFloat, IT<:Int}
    Nθ, Nk = size(θ)
    
    us_ref = zeros(FT, Nθ, Nx)
    
    xx = LinRange(0, 1, Nx)
    
    for k = 1:Nk
        us_ref += θ[:, k]*sqrt((π^2*k^2  + τ^2)^(-d))/k^2/π^2 * sin.(π * k * xx)'
    end

    return us_ref
end


function solve_Poisson_1D(f::Array{FT,1}) where {FT<:AbstractFloat}
    Nx = length(f)
    Δx = 1/(Nx - 1)

    # This is a tridiagonal matrix
    d  = 2*ones(FT, Nx-2)
    dl =  -ones(FT, Nx-3)
    dr =  -ones(FT, Nx-3)

    df = Tridiagonal(dl, d, dr)  
    fₕ = Δx^2/4*(2f[2:Nx-1] + f[1:Nx-2] + f[3:Nx])
    
    u = df\fₕ
    
    return [0.; u ; 0.0]
end


    
function solve_Darcy_1D(a::Array{FT,1}, f::Array{FT,1}) where {FT<:AbstractFloat}
    Nx = length(a)
    Δx = 1/(Nx - 1)
    
    # This is a tridiagonal matrix
    d  = (2a[2:Nx-1] + a[1:Nx-2] + a[3:Nx])/2
    dl =  -(a[3:Nx-1] + a[2:Nx-2])/2
    dr =  -(a[2:Nx-2] + a[3:Nx-1])/2

    da = Tridiagonal(dl, d, dr)  
    fₕ = Δx^2/4*(2f[2:Nx-1] + f[1:Nx-2] + f[3:Nx])
    
    u = da\fₕ
    
    return [0.; u ; 0.0]
end



function generate_1d_heat_ref(θ::Array{FT,2}, θᵖ::Array{FT,2},  Nx::IT, t::FT;  d::FT=1.0, τ::FT=3.0) where {FT<:AbstractFloat, IT<:Int}
    Nθ, Nk = size(θ)
    
    us_ref = zeros(FT, Nθ, Nx)
    
    xx = LinRange(0, 1, Nx)
    
    for k = 1:Nk
        us_ref += (exp(-t*k^2*π^2)*θ[:, k] + (1-exp(-t*k^2*π^2))/(k^2*π^2)*θᵖ[:, k])*sqrt((π^2*k^2  + τ^2)^(-d)) * sin.(π * k * xx)'
    end

    return us_ref
end


function solve_heat_1D(u0::Array{FT,1}, f::Array{FT,1}, Δt::FT, Nt::IT; method::String="RK2") where {FT<:AbstractFloat, IT<:Int}
    Nx = length(u0)
    Δx = 1/(Nx - 1)
    
    
    # This for mass matrix
    d  = 2/3*ones(FT, Nx-2)
    dl = 1/6*ones(FT, Nx-3)
    dr = 1/6*ones(FT, Nx-3)
    M = Tridiagonal(dl, d, dr)

    
    # This is a tridiagonal matrix
    d  =   2/Δx^2*ones(FT, Nx-2)
    dl =  -1/Δx^2*ones(FT, Nx-3)
    dr =  -1/Δx^2*ones(FT, Nx-3)
    A = Tridiagonal(dl, d, dr)
    
    # This is for forcing
    fₕ = (2f[2:Nx-1] + f[1:Nx-2] + f[3:Nx])/4
    # fₕ = Δt*f[2:Nx-1]
    
    # The discretization becomes Mu̇ = fₕ - Au
    
    
    
    u = copy(u0)
    
    if method=="Crank-Nicolson"
        for i = 1:Nt
            u[2:Nx-1] += (M + 0.5*Δt*A)\(Δt*(fₕ - A*u[2:Nx-1]))
        end
    elseif method=="RK2"
    
        # Runge Kutta
        x = 1 + sqrt(2)/2
        a = [x 0; 1-2*x x]
        b = [1/2 ;1/2]
        c = [x ;1-x]

        for i = 1:Nt
            k1 = (M + a[1,1]*Δt*A)\(fₕ - A*u[2:Nx-1])
            k2 = (M + a[2,2]*Δt*A)\(fₕ - A*u[2:Nx-1] - Δt*a[2,1]*A*k1 )
            u[2:Nx-1] += Δt*(b[1]*k1 + b[2]*k2)
        
        end
    end
    
    return  u
end








function limiter_func(r::FT) where {FT<:AbstractFloat}
    return max(0, min(1, r))
end

function reconstruct_1D(u::Array{FT,1}; eps=1e-15) where {FT<:AbstractFloat}
    Nx = length(u)
    u_rec₋, u_rec₊ = zeros(Nx), zeros(Nx)
    for i = 1:Nx-1
        u₋₋, u₋, u₊, u₊₊ = (i == 1 ? u[Nx-1] : u[i-1]), u[i], u[i+1], (i == Nx-1 ? u[2] : u[i+2])
        ϕ = limiter_func((u₊-u₋)/(u₋-u₋₋+eps))
        u_rec₋[i] = u₋ + ϕ*(u₋ - u₋₋)/2.0
        ϕ = limiter_func((u₊-u₋)/(u₊₊-u₊+eps))
        u_rec₊[i] = u₊ - ϕ*(u₊₊ - u₊)/2.0
    end
    
    return u_rec₋, u_rec₊
end

# Periodic Burgers equation
# u[1], u[2], ... , u[Nx],  (u[1] = u[Nx])
function solve_Burgers_1D_tendency(u::Array{FT,1}, f::Array{FT,1}, ν::FT) where {FT<:AbstractFloat, IT<:Int}
    Nx = length(u)
    Δx = 1/(Nx - 1)
    
    # forcing 
    tendency = copy(f)
    # viscous term
    tendency[1] += ν*(u[Nx-1] - 2u[1] + u[2])/Δx^2
    for i = 2:Nx-1
        tendency[i] += ν*(u[i-1] - 2u[i] + u[i+1])/Δx^2
    end
    # inviscid term
    u₋, u₊ = reconstruct_1D(u)
    flux = 0.5*(u₊.^2/2.0 + u₋.^2/2.0 - max.(abs.(u₊), abs.(u₋)).*(u₊ .- u₋))
    
       
    tendency[1] -= (flux[1] - flux[Nx-1])/Δx 
    for i = 2:Nx-1
        tendency[i] -= (flux[i] - flux[i-1])/Δx 
    end
    
    
    
    tendency[Nx] = tendency[1]

    return  tendency
end

# Periodic Burgers equation
function solve_Burgers_1D(u0::Array{FT,1}, f::Array{FT,1}, ν::FT, Δt::FT, Nt::IT, save_every::IT=1) where {FT<:AbstractFloat, IT<:Int}
    Nx = length(u0)
    Δx = 1/(Nx - 1)
    
    u_data = zeros(div(Nt,save_every)+1, Nx)
    u_data[1, :] .= u0
    
    u = copy(u0)
    for i = 1:Nt

        k1 = solve_Burgers_1D_tendency(u, f, ν)
        k2 = solve_Burgers_1D_tendency(u + Δt*k1, f, ν)
        u += Δt*(k1 + k2)/2.0
        # save data
        if i%save_every == 0 
            u_data[div(i,save_every)+1, :] .= u
        end

    end

    return  u_data
end






### Kuramoto Sivashinksy Equation

using LinearAlgebra
using Random
using Distributions
using FFTW
using PyPlot

mutable struct Spectral_Mesh
    Nx::Int64
    
    Lx::Float64
    Δx::Float64
    xx::Array{Float64, 1}
  
    kxx::Array{Float64, 1}
    alias_filter::Array{Float64, 1}
    d_x::Array{ComplexF64, 1}
    laplacian_eigs::Array{Float64, 1}
    
    # container
    u::Array{Float64, 1}
    u_x::Array{Float64, 1}
    u_hat::Array{ComplexF64, 1}
    u_x_hat::Array{ComplexF64, 1}
end




function Spectral_Mesh(Nx::Int64,  Lx::Float64)
    @assert(Nx%2 == 0)
    
    Δx = Lx/Nx
    xx = LinRange(0, Lx-Δx, Nx)
    

    kxx, alias_filter = Spectral_Init(Nx)
    d_x = Apply_Gradient_Init(Nx, Lx, kxx) 
    laplacian_eigs = Apply_Laplacian_Init(Nx, Lx, kxx) 
    
    
    # container
    u= zeros(Float64, Nx)
    u_x= zeros(Float64, Nx)
    u_hat= zeros(ComplexF64, Nx)
    u_x_hat= zeros(ComplexF64, Nx)
    
    Spectral_Mesh(Nx, Lx, Δx, xx, kxx, alias_filter, 
                  d_x, laplacian_eigs,
                  u, u_x, u_hat,  u_x_hat)
end

"""
Compute mode numbers kxx and kyy
kxx = [0,1,...,[Nx/2]-1, -[Nx/2], -[Nx/2]+1, ... -1]   
and alias filter based O
"""
function Spectral_Init(Nx::Int64)
    kxx = mod.((1:Nx) .- ceil(Int64, Nx/2+1),Nx) .- floor(Int64, Nx/2)
    alias_filter = zeros(Float64, Nx)
    for i = 1:Nx
        if (abs(kxx[i]) < Nx/3) 
            alias_filter[i] = 1.0
        end
    end
    return kxx, alias_filter
end




function Trans_Spectral_To_Grid!(mesh::Spectral_Mesh, u_hat::Array{ComplexF64,1},  u::Array{Float64,1})
    """
    All indices start with 0
    
    K = {(kx, ky) | kx ∈ 0,1,...,[Nx/2]-1, -[Nx/2], -[Nx/2]+1, ... -1   ,   ky ∈ 0,1,...,[N_y/2]-1, -[N_y/2], -[N_y/2]+1, ... -1
    x ∈ 0, Δx, 2Δx, ..., Lx - Δx   ,   y ∈ 0, Δy, 2Δy, ..., Ly - Δy  here Δx, Δy = Lx/Nx, Ly/N_y
    
    P(x, y) =  1/(nxny)∑_{kx, ky}  F[kx,ky]  e^{i (2π/Lx kx x + 2π/Ly ky y)}
    
    P[jx, jy] = P(jxΔx, jyΔy) =  1/(nxny)∑_{kx, ky}  F[kx,ky]  e^{i (2π kx jx/Nx + 2π ky jy/N_y)}
    
    
    @test
    u_hat = [1 2 3 4; 1.1 2.2 1.3 2.4; 2.1 3.2 4.1 1.2]
    Nx, N_y = size(u_hat)
    u = zeros(ComplexF64, Nx, N_y)
    for jx = 0:Nx-1
        for jy = 0:N_y-1
            for kx = 0:Nx-1
                for ky = 0:N_y-1
                    u[jx+1, jy+1] += u_hat[kx+1,ky+1] *  exp((2π*kx*jx/Nx + 2.0*π*ky*jy/N_y)*im) /(Nx*N_y)
                end
            end
        end
    end
    ufft = ifft(u_hat)
    @info u - ufft
    """
    
    u .= real(ifft(u_hat)) #fourier for the first dimension
    
end



function Trans_Grid_To_Spectral!(mesh::Spectral_Mesh, u::Array{Float64,1}, u_hat::Array{ComplexF64,1})
    
    """
    K = {(kx, ky) | kx ∈ 0,1,...,Nx/2-1, -Nx/2, -Nx/2+1, ... -1   ,   ky ∈ 0,1,...,N_y/2-1, -N_y/2, -N_y/2+1, ... -1}
    
    P(x, y) = 1/(nx⋅ny) ∑_{kx, ky}  F[kx,ky]  e^{i (2π/Lx kx x + 2π/Ly ky y)}
    
    F[kx, ky] = (nx⋅ny)/(Lx⋅Ly) ∫ P(x,y)  e^{-i (2π/Lx kx x + 2π/Ly ky y)}
              = (nx⋅ny)/(Lx⋅Ly) ∑ P[jx,jy]  e^{-i (2π kx jx/Nx + 2π ky jy/N_y)} ΔxΔy
              = ∑ P[jx,jy]  e^{-i (2π kx jx/Nx + 2π ky jy/N_y)}
    
    @test
    u = [1 2 3 4; 1.1 2.2 1.3 2.4; 2.1 3.2 4.1 1.2]
    Nx, N_y = size(u)
    u_hat = zeros(ComplexF64, Nx, N_y)
    for jx = 0:Nx-1
        for jy = 0:N_y-1
            for kx = 0:Nx-1
                for ky = 0:N_y-1
                    u_hat[jx+1, jy+1] += u[kx+1,ky+1] *  exp(-(2π*kx*jx/Nx + 2.0*π*ky*jy/N_y)*im)
                end
            end
        end
    end
    u_hat2 = fft(u)
    @info u_hat - u_hat2
    """
    
    
    u_hat .= mesh.alias_filter.* fft(u)
    
    
end

function Apply_Laplacian_Init(Nx::Int64, Lx::Float64, kxx::Array{Int64, 1}) 
    """
    See Apply_Laplacian!
    """
    laplacian_eig = zeros(Float64, Nx)
    for i = 1:Nx
        kx = kxx[i]
        laplacian_eig[i] = -(2*pi*kx/Lx)^2 
    end
    return laplacian_eig
end

function Apply_Laplacian!(mesh::Spectral_Mesh, u_hat::Array{ComplexF64,1}, Δu_hat::Array{ComplexF64,1}; order=1) 
    """
    Δ (ω_hat[kx,ky]  e^{i 2π/Lx kx x}) = -(2πkx/Lx)² ω_hat
    """
    eig = mesh.laplacian_eigs.^order
    Δu_hat .= eig .* u_hat
end

function Apply_Gradient_Init(Nx::Int64,  Lx::Float64,  kxx::Array{Int64, 1}) 
    """
    ∂f/∂x_hat = alpha_x f_hat
    """
    d_x = zeros(ComplexF64, Nx)
    
    for i = 1:Nx
        kx = kxx[i]
        d_x[i] = (2*pi/Lx * kx)*im
    end

    return d_x
end


"""
Compute gradients u_x_hat, ω_y, from spectral input u_hat
u_x_hat = d_x u_hat
"""
function Apply_Gradient!(mesh::Spectral_Mesh, u_hat::Array{ComplexF64,1}, u_x_hat::Array{ComplexF64,1})
    d_x = mesh.d_x
    
    u_x_hat .= d_x .* u_hat
end

"""
δu_hat = hat { u ∂u } = hat { 0.5 ∂(uu) } = hat {  ∂(uu/0.5) } = d_x hat{uu/0.5}
"""
function Compute_Horizontal_Advection!(mesh::Spectral_Mesh, u_hat::Array{ComplexF64,1}, δu_hat::Array{ComplexF64, 1})
    u = mesh.u
    # contanier
    u2, u2_hat = mesh.u_x, mesh.u_x_hat

    Trans_Spectral_To_Grid!(mesh, u_hat, u)
    # Apply_Gradient!(mesh, u_hat, u_x)
    # Trans_Grid_To_Spectral!(mesh, u.*u_x,  δu_hat)
    u2 .= u.*u/2.0
    Trans_Grid_To_Spectral!(mesh, u2,  u2_hat)
    Apply_Gradient!(mesh, u2_hat, δu_hat)
    
end



function Visual(mesh::Spectral_Mesh, u::Array{Float64,1}, var_name::String,
    save_file_name::String="None", vmin=nothing, vmax=nothing)
    
    Nx, N_y = mesh.Nx, mesh.N_y
    xx, yy = mesh.xx, mesh.yy
    X,Y = repeat(xx, 1, N_y), repeat(yy, 1, Nx)'
    
    figure()
    pcolormesh(X, Y, u, shading= "gouraud", cmap="jet", vmin=vmin, vmax =vmax)
    xlabel("X")
    ylabel("Y")
    colorbar()
    
    if save_file_name != "None"
        tight_layout()
        savefig(save_file_name)
    end
end




"""
Generalized Kuramoto-Sivashinksy equation with periodic boundary condition
 
∂u/∂t + u∂u + ν₂∂²u + ν₄∂⁴u = 0
u = 1/(Nx)∑_{kx}  u_hat[kx,ky]  e^{i 2π kx x/Lx }
∂u_hat/∂t + FFT[(u⋅∇)u] + [ν₂(i2π kx/Lx)² + ν₄(i2π ky/Ly)⁴]u_hat  = 0
The domain is [0,Lx]
"""
mutable struct Spectral_KS_Solver
    mesh::Spectral_Mesh

    ν₂::Float64
    ν₄::Float64

    u::Array{Float64, 1}
    u_hat::Array{ComplexF64, 1}
    u_hat_old::Array{ComplexF64, 1}
    f::Array{Float64, 1}
    f_hat::Array{ComplexF64, 1}

    Δu_hat::Array{ComplexF64, 1}
    δu_hat::Array{ComplexF64, 1}

    k1::Array{ComplexF64, 1}
    k2::Array{ComplexF64, 1}
    k3::Array{ComplexF64, 1}
    k4::Array{ComplexF64, 1}
end


# constructor of the Spectral_KS_Solver
#
# There are two forcing intialization approaches
# * initialize fx and fy components, user shoul make sure they are zero-mean and periodic
# * initialize ∇×f, curl_f
#
# There are velocity(vorticity) intialization approaches
# * initialize u0 and v0 components, user should make sure they are incompressible div ⋅ (u0 , v0) = 0
# * initialize ω0 and mean backgroud velocity ub and vb
#
function Spectral_KS_Solver(mesh::Spectral_Mesh, u0::Array{Float64, 1}, f::Array{Float64, 1}, ν₂::Float64 = 1.0, ν₄::Float64 = 1.0)    
    Nx = mesh.Nx
    
    u = zeros(Float64, Nx)
    u_hat = zeros(ComplexF64, Nx)
    u_hat_old = zeros(ComplexF64, Nx)
    f_hat = zeros(ComplexF64, Nx)
    
    # initialization
    u .= u0
    Trans_Grid_To_Spectral!(mesh, u, u_hat)
    u_hat_old .= u_hat
    
    Trans_Grid_To_Spectral!(mesh, f, f_hat)

    δu_hat = zeros(ComplexF64, Nx)
    Δu_hat = zeros(ComplexF64, Nx)
    
    k1 = zeros(ComplexF64, Nx)
    k2 = zeros(ComplexF64, Nx)
    k3 = zeros(ComplexF64, Nx)
    k4 = zeros(ComplexF64, Nx)
    
    Spectral_KS_Solver(mesh, ν₂, ν₄, u, u_hat, u_hat_old, f, f_hat, Δu_hat, δu_hat, k1, k2, k3, k4)
end




# ∂u/∂t + u∂u + ν₂∂²u + ν₄∂⁴u = f
# ∂u_hat/∂t =  - FFT[(u⋅∇)u] - [ν₂(i2π kx/Lx)² + ν₄(i2π ky/Ly)⁴]u_hat  + f_hat
# Compute the right hand-side
function Explicit_Residual!(self::Spectral_KS_Solver, u_hat::Array{ComplexF64, 1}, δu_hat::Array{ComplexF64, 1})
    mesh = self.mesh
    f_hat = self.f_hat
    Compute_Horizontal_Advection!(mesh, u_hat, δu_hat)

    δu_hat .*= -1.0 
    δu_hat .+= f_hat
    
    Δu_hat = self.Δu_hat
    
    Apply_Laplacian!(mesh, u_hat, Δu_hat; order = 1)
    δu_hat .-= self.ν₂ * Δu_hat
    
    Apply_Laplacian!(mesh, u_hat, Δu_hat; order = 2)
    δu_hat .-= self.ν₄ * Δu_hat
end

# Also the Crank-Nicolson
# ∂u/∂t + u∂u + ν₂∂²u + ν₄∂⁴u = f
# ∂u_hat/∂t =  - FFT[(u⋅∇)u] - [ν₂(i2π kx/Lx)² + ν₄(i2π ky/Ly)⁴]u_hat  + f_hat
# [1 - [ν₂(i2π kx/Lx)² + ν₄(i2π ky/Ly)⁴]/2] (u_hat(n+1)-u_hat(n))/Δt = -FFT[(u⋅∇)u] + f_hat - [ν₂(i2π kx/Lx)² + ν₄(i2π ky/Ly)⁴]u_hat(n)/2 
# compute (ω_hat(n+1)-ω_hat(n))/Δt
function Semi_Implicit_Residual!(self::Spectral_KS_Solver, u_hat::Array{ComplexF64, 1}, u_hat_old::Array{ComplexF64, 1}, 
    Δt::Float64, δu_hat::Array{ComplexF64, 1})

    mesh  = self.mesh
    f_hat = self.f_hat
    
    Compute_Horizontal_Advection!(mesh, u_hat, δu_hat)
    Δu_hat = self.Δu_hat
    Compute_Horizontal_Advection!(mesh, u_hat_old, Δu_hat)

    δu_hat .= -(3.0/2.0*δu_hat - 0.5*Δu_hat) + f_hat

    Δu_hat = self.Δu_hat
    Apply_Laplacian!(mesh, u_hat, Δu_hat; order = 1)
    δu_hat .-= self.ν₂ * Δu_hat

    Apply_Laplacian!(mesh, u_hat, Δu_hat; order = 2)
    δu_hat .-= self.ν₄ * Δu_hat
    
    δu_hat ./= ( 1.0 .+ 0.5*Δt*(self.ν₂ * mesh.laplacian_eigs + self.ν₄ * mesh.laplacian_eigs.^2) )

end




function Solve!(self::Spectral_KS_Solver, Δt::Float64, method::String)
    u, u_hat, u_hat_old, δu_hat, f_hat = self.u, self.u_hat, self.u_hat_old, self.δu_hat, self.f_hat

    if method == "Crank-Nicolson-Adam-Bashforth"
        Semi_Implicit_Residual!(self, u_hat, u_hat_old, Δt, δu_hat)

    elseif method == "RK4"
        k1, k2, k3, k4 = self.k1, self.k2, self.k3, self.k4 
        Explicit_Residual!(self, u_hat,  k1)
        Explicit_Residual!(self, u_hat + Δt/2.0*k1, k2)
        Explicit_Residual!(self, u_hat + Δt/2.0*k2, k3)
        Explicit_Residual!(self, u_hat + Δt*k3, k4)
        δu_hat = (k1 + 2*k2 + 2*k3 + k4)/6.0

    end
    u_hat_old .= u_hat
    u_hat .+= Δt*δu_hat
    
    # clean
    mesh = self.mesh
    Trans_Spectral_To_Grid!(mesh, u_hat, u)
    Trans_Grid_To_Spectral!(mesh, u, u_hat)
end


function solve_KS_1D(u0, f, T=1000.0, Nt=20000, Lx=100.0, ν₂=1.0, ν₄=1.0, save_every=1)
    ####################################
    ν₂ = 1.0              # viscosity
    ν₄ = 1.0              # viscosity
    Ne = length(u0)       # resolution in x
    Δt = T/Nt             # time step 

    method = "Crank-Nicolson-Adam-Bashforth"  # RK4 or Crank-Nicolson
    mesh   = Spectral_Mesh(Ne, Lx)
    Δx, xx = mesh.Δx, mesh.xx


    solver = Spectral_KS_Solver(mesh, u0, f, ν₂, ν₄)  
    Ns = div(Nt, save_every)
    u_all = zeros(Float64, Ns, Ne)
    u_hat_all = zeros(ComplexF64, Ns, Ne)
    
    # compute auto-correlation
    Cx = zeros(Float64, Ns, Ne)
    # compute energy spectral
    Ex = zeros(Float64, Ns, Ne)
    
    for i = 1:Nt
        Solve!(solver, Δt, method)
        Cx[div(i + save_every - 1, save_every), : ] += real(ifft(solver.u_hat .* conj(solver.u_hat)) ) / Ne
        Ex[div(i + save_every - 1, save_every), : ] += (solver.u_hat .* conj(solver.u_hat)) / (Ne^2)
        
        if i%save_every == 0
            u_all[div(i,save_every), :] .= solver.u
            u_hat_all[div(i,save_every), :] .= solver.u_hat
            Cx[div(i,save_every), : ] /= save_every
            Ex[div(i,save_every), : ] /= save_every
        end
    end
    
    
    
    N_burn = div(Ns, 4)
    auto_correlation = Lx/(Ns - N_burn) * (sum(Cx[N_burn+1:end, :], dims=1)[:])
    energy_spectral = 1/(Ns - N_burn) *(sum(Ex[N_burn+1:end, :], dims=1)[1:div(Ne, 2)])

    
    return u_all, auto_correlation, Cx, energy_spectral, Ex
end



function solve_KS_1D_helper(u0, f, T, Nt, Lx, ν₂, ν₄, save_every)
    us, auto_correlation, _, energy_spectral, _ = solve_KS_1D(u0, f, T, Nt, Lx, ν₂, ν₄, save_every)
    return auto_correlation
end