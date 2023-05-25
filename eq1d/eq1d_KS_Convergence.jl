using NPZ
include("eq1d_func.jl")


Random.seed!(42);

Ne_ref = 2^11
Lx = 100.0
Δx = Lx/Ne_ref

xx = Array(LinRange(0, Lx - Δx, Ne_ref))
T_scale_ref = 2^11
T  = T_scale_ref*Lx
Nt = T_scale_ref*Ne_ref
ν₂ = 1.0
ν₄ = 1.0
Δt = T/Nt
save_every = T_scale_ref


Nθ, Nk = 10, 20
θs = rand(Normal(0, 1), Nθ, Nk)
# f = [1.0, 1.0] * 0.05*sin.((8 * pi * xx) / Lx)' + 0.05*generate_1d_rf(θs, Ne+1; L=Lx,  d=2.0, τ=3.0, boundary_condition="periodic")[:, 1:end-1]
f_ref = 0.05*generate_1d_rf(θs, Ne_ref+1; L=Lx,  d=2.0, τ=3.0, boundary_condition="periodic")[:, 1:end-1]
u0_ref = ones(Nθ)*(0.0*xx)'

us_data_ref = zeros(Nθ, Ne_ref)
auto_correlation_data_ref = zeros(Nθ, Ne_ref)
energy_spectral_data_ref = zeros(Nθ, div(Ne_ref, 2))

u_all, Cx = nothing, nothing
for j = 1:Nθ
    @time u_all, auto_correlation, Cx, energy_spectral, Ex = solve_KS_1D(u0_ref[j,:], f_ref[j,:], T, Nt, Lx, ν₂, ν₄, save_every)
    us_data_ref[j, :], auto_correlation_data_ref[j, :], energy_spectral_data_ref[j, :] = u_all[end, :], auto_correlation, energy_spectral
end





Nes = [2^9; 2^10; 2^11]
T_scales = [2^0; 2^1; 2^2; 2^3; 2^4; 2^5; 2^6; 2^7; 2^8; 2^9; 2^10; 2^11]
Δxs = 1.0 ./ Nes
# compute error for space and time solution
errors = zeros(Nθ, 3, length(Nes), length(T_scales))
costs = zeros(length(Nes), length(T_scales))



for (i, Ne) in enumerate(Nes)
    for (j, T_scale) in enumerate(T_scales)
        @info Ne, T_scale
        
        if Ne == Ne_ref && T_scale == T_scale_ref
            continue
        end
        T  = T_scale*Lx
        
        Nt = div(T_scale*Ne, 1)
        Δt = T/Nt
        Δx = Lx/Ne
        xx = Array(LinRange(0, Lx - Δx, Ne))
        save_every = T_scale
    
        u0 = u0_ref[:, 1:Int64(Ne_ref/Ne):end]
        f = f_ref[:, 1:Int64(Ne_ref/Ne):end]
    
    
        us_data = zeros(Nθ, Ne)
        auto_correlation_data = zeros(Nθ, Ne)
        energy_spectral_data = zeros(Nθ, div(Ne, 2))
        
        costs[i, j] = Nt * (4*(5*log(Ne)*Ne) + 9*Ne)
        
        for k = 1:Nθ
            u_all, auto_correlation, Cx, energy_spectral, Ex = solve_KS_1D(u0[k,:], f[k,:], T, Nt, Lx, ν₂, ν₄, save_every)
            us_data[k, :], auto_correlation_data[k, :], energy_spectral_data[k, :] = u_all[end, :], auto_correlation, energy_spectral

            errors[k, 1, i, j] = norm(us_data[k, :] - us_data_ref[k, 1:Int64(Ne_ref/Ne):end])/norm(us_data_ref[k, 1:Int64(Ne_ref/Ne):end])
            errors[k, 2, i, j] = norm(auto_correlation_data[k, :] - auto_correlation_data_ref[k, 1:Int64(Ne_ref/Ne):end])/norm(auto_correlation_data_ref[k, 1:Int64(Ne_ref/Ne):end])
            errors[k, 3, i, j] = norm(energy_spectral_data[k, :] - energy_spectral_data_ref[k, 1:Int64(Ne_ref/Ne):end])/norm(energy_spectral_data_ref[k, 1:Int64(Ne_ref/Ne):end])
        end
    end
end

npzwrite("KS_errors.npy", errors)
npzwrite("KS_costs.npy", costs)