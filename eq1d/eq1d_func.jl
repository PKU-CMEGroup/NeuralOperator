using Distributions
using Random
using LinearAlgebra
using SparseArrays
using PyPlot


function generate_1d_rf(θ::Array{FT,2}, Nx::IT;  d::FT=2.0, τ::FT=3.0) where {FT<:AbstractFloat, IT<:Int}
    Nθ, Nk = size(θ)
    
    fs = zeros(FT, Nθ, Nx)
    
    xx = LinRange(0, 1, Nx)
    for k = 1:Nk
        fs += θ[:, k]*sqrt((π^2*k^2  + τ^2)^(-d)) * sin.(π * k * xx)'
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