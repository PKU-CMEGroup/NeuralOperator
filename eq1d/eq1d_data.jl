include("eq1d_func.jl")
        
        
        
# generate 3,0000 data, 20000 for training 10000 for test
using NPZ
prefix = "/central/groups/esm/dzhuang/cost-accuracy-data/"
Random.seed!(42);
Nθ, Nk, Ne = 2^15, 2^14, 2^12
θs = rand(Normal(0, 1), Nθ, Ne+1)
θᵖs = rand(Normal(0, 1), Nθ, Ne+1)

d, τ = 1.0, 3.0

rf_θs = generate_1d_rf(θs, Ne+1;  d=d, τ=τ)
rf_θᵖs = generate_1d_rf(θᵖs, Ne+1;  d=d, τ=τ)


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

# heat equation problem
T = 1.0/8
u0s = copy(rf_θs)
fs  = copy(rf_θᵖs)
us_ref = generate_1d_heat_ref(θs, θᵖs, Ne+1, T;  d=d, τ=τ)
npzwrite(prefix*"heat_u0.npy", u0s)
npzwrite(prefix*"heat_f.npy", fs)
npzwrite(prefix*"heat_u.npy", us_ref)
     