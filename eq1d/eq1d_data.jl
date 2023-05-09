include("eq1d_func.jl")
        
        
        
# generate 3,0000 data, 20000 for training 10000 for test
using NPZ
prefix = "/central/groups/esm/dzhuang/cost-accuracy-data/"
Random.seed!(42);
Nθ, Nk, Ne = 2^15, 2^14, 2^12
# Nθ, Nk, Ne = 24, 128, 2^10
θs = rand(Normal(0, 1), Nθ, Nk)
θᵖs = rand(Normal(0, 1), Nθ, Nk)

d, τ = 1.0, 3.0

rf_θs = generate_1d_rf(θs, Ne+1;  d=d, τ=τ)
rf_θᵖs = generate_1d_rf(θᵖs, Ne+1;  d=d, τ=τ)

GENERATE_DARCY = false
if GENERATE_DARCY
    # Darcy flow porblem
    a₊, a₋ = 10.0, 1.0
    as  = copy(rf_θs)
    as[as .>= 0] .= a₊
    as[as .< 0]  .= a₋
    f = 100*ones(Ne + 1)
    us = zeros(Nθ, Ne+1)
    for j = 1:Nθ
        us[j, :] = solve_Darcy_1D(as[j, :], f)
    end
    npzwrite(prefix*"darcy_a.npy", as)
    npzwrite(prefix*"darcy_u.npy", us)
end

GENERATE_HEAT = false
if GENERATE_HEAT
    # heat equation problem
    T = 1.0/8
    u0s = copy(rf_θs)
    fs  = copy(rf_θᵖs)
    us_ref = generate_1d_heat_ref(θs, θᵖs, Ne+1, T;  d=d, τ=τ)
    npzwrite(prefix*"heat_u0.npy", u0s)
    npzwrite(prefix*"heat_f.npy", fs)
    npzwrite(prefix*"heat_u.npy", us_ref)
end

GENERATE_BURGERS = true
if GENERATE_BURGERS
    # Burgers' equation problem
    Nt = 4Ne
    u0s = generate_1d_rf(θs, Ne+1;  d=1.0, τ=3.0, boundary_condition="periodic")
    us = zeros(Nθ, Ne+1)
    f = zeros(Ne+1)
    ν = 1.0e-4
    Δt = 1.0/Nt
    for j = 1:Nθ
        us[j, :] = solve_Burgers_1D(u0s[j, :], f, ν, Δt, Nt)[end, :]
    end
    
    npzwrite(prefix*"Burgers_u0.npy", u0s)
    npzwrite(prefix*"Burgers_u.npy", us)
end



GENERATE_KS = false
if GENERATE_KS
    
    Lx = 100.0
    T=4000.0
    Nt=40*Ne
    ν₂=1.0
    ν₄=1.0

    Δx = Lx/Ne
    xx = Array(LinRange(0, Lx - Δx, Ne))
    u0 = cos.((2 * pi * xx) / Lx) + 0.1*cos.((4 * pi * xx) / Lx)
    f = 0.05*generate_1d_rf(θs, Ne+1; L=Lx,  d=2.0, τ=3.0, boundary_condition="periodic")[:, end-1]

    us_data = zeros(Nθ, Ne)
    auto_correlation_data = zeros(Nθ, Ne)
    energy_spectral_data = zeros(Nθ, div(Ne, 2))

    Threads.@threads for j = 1:Nθ
        us, auto_correlation, energy_spectral = solve_KS_1D(u0, f[j,:], T, Nt, Lx, ν₂, ν₄)
        us_data[j, :], auto_correlation_data[j, :], energy_spectral_data[j, :] = us[end, :], auto_correlation, energy_spectral
    end
    npzwrite(prefix*"KS_u.npy", us_data)
    npzwrite(prefix*"KS_auto_correlation.npy", auto_correlation_data)
    npzwrite(prefix*"KS_energy_spectral.npy", energy_spectral_data)
    


end






GENERATE_KS_TEST = false
if GENERATE_KS_TEST

    Lx = 100.0
    
    T_scale = 500
    T  = T_scale*Lx
    Nt = div(T_scale*Ne, 1)
    save_every = T_scale
    
    ν₂=1.0
    ν₄=1.0

    Δx = Lx/Ne
    xx = Array(LinRange(0, Lx - Δx, Ne))
    u0 = cos.((2 * pi * xx) / Lx) + 0.1*cos.((4 * pi * xx) / Lx)
    f = 0.05*generate_1d_rf(θs, Ne+1; L=Lx,  d=2.0, τ=3.0, boundary_condition="periodic")[:, end-1]

    us_data = zeros(Nθ, Ne)
    auto_correlation_data = zeros(Nθ, Ne)
    energy_spectral_data = zeros(Nθ, div(Ne, 2))

    Threads.@threads for j = 1:Nθ
        us, auto_correlation, _, energy_spectral, _ = solve_KS_1D(u0, f[j,:], T, Nt, Lx, ν₂, ν₄, save_every)
        us_data[j, :], auto_correlation_data[j, :], energy_spectral_data[j, :] = us[end, :], auto_correlation, energy_spectral
    end
    npzwrite(prefix*"KS_test_u.npy", us_data)
    npzwrite(prefix*"KS_test_auto_correlation.npy", auto_correlation_data)
    npzwrite(prefix*"KS_test_energy_spectral.npy", energy_spectral_data)


end