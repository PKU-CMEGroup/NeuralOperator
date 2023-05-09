@everywhere using NPZ
@everywhere using ArgParse


@everywhere include("eq1d_func.jl")
         
prefix = "/central/groups/esm/dzhuang/cost-accuracy-data/"


function Data_Generate(θs, chunk_id)
    Nθ, Nk = size(θs)
    Ne = 2^11
    
    Lx = 100.0
    T_scale = 2^11
    T  = T_scale*Lx
    Nt = div(T_scale*Ne, 1)
    
    
    save_every = T_scale
    ν₂=1.0
    ν₄=1.0
    Δx = Lx/Ne
    xx = Array(LinRange(0, Lx - Δx, Ne))
    u0 = 0.0*xx # cos.((2 * pi * xx) / Lx) + 0.1*cos.((4 * pi * xx) / Lx)
    fs = 0.05*generate_1d_rf(θs, Ne+1; L=Lx,  d=2.0, τ=3.0, boundary_condition="periodic")[:, 1:end-1]

    
    # Define caller function
    @everywhere solve_KS_1D_(f::Vector{FT}) where FT<:Real = 
        solve_KS_1D_helper($u0, f, $T, $Nt, $Lx, $ν₂, $ν₄, $save_every)

    
    params = [fs[i, :] for i in 1:Nθ]

    @everywhere params = $params
    auto_correlation_tuple = pmap(solve_KS_1D_, params)     # Outer dim is params iterator

    auto_correlation_data = zeros(Nθ, Ne)
    for i = 1:Nθ
        auto_correlation_data[i,:] = auto_correlation_tuple[i]
    end
    
    
    npzwrite(prefix*"KS_fs_$(chunk_id).npy", fs)
    npzwrite(prefix*"KS_auto_correlation_$(chunk_id).npy", auto_correlation_data)
    

end


Random.seed!(42)
Nθ, Nk = 2048*16, 20
θs_all = rand(Normal(0,1), Nθ, Nk)
npzwrite(prefix * "KS_theta_$(Nθ).npy",   θs_all)


# Read iteration number of ensemble to be recovered
s = ArgParseSettings()
@add_arg_table s begin
    "--chunk_id"
    help = "Chunk ID"
    arg_type = Int
    default = 1
end
parsed_args = parse_args(ARGS, s)
chunk_id_ = parsed_args["chunk_id"]

chunk_size_= 2048
Data_Generate(θs_all[chunk_size_*(chunk_id_-1)+1 : chunk_size_*chunk_id_, :], chunk_id_)