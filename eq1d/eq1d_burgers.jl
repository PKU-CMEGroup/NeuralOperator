@everywhere using NPZ
@everywhere using ArgParse


@everywhere include("eq1d_func.jl")
         
prefix = "/central/groups/esm/dzhuang/cost-accuracy-data/"


function Data_Generate(θs, chunk_id)
    Nθ, Nk = size(θs)
    Ne = 2^12
    Nt = 4Ne
    Δt = 1.0/Nt
    
    f = zeros(Ne+1)
    ν = 1.0e-4
    

    u0s = generate_1d_rf(θs, Ne+1;  d=1.0, τ=3.0, boundary_condition="periodic")

    
    # Define caller function
    @everywhere solve_Burgers_1D_(u0::Vector{FT}) where FT<:Real = 
        solve_Burgers_1D(u0, $f, $ν, $Δt, $Nt, $Nt)

    
    params = [u0s[i, :] for i in 1:Nθ]

    @everywhere params = $params
    us_tuple = pmap(solve_Burgers_1D_, params)     # Outer dim is params iterator

    us = zeros(Nθ, Ne+1)
    for i = 1:Nθ
        us[i,:] = us_tuple[i][end, :]
    end
    
    npzwrite(prefix*"burgers_u0s_$(chunk_id).npy", u0s)
    npzwrite(prefix*"burgers_us_$(chunk_id).npy", us)
    

end


Random.seed!(42)
chunk_size_= 2048
Nθ, Nk = chunk_size_*16, 2^14
θs_all = rand(Normal(0,1), Nθ, Nk)
npzwrite(prefix * "burger_theta_$(Nθ).npy",   θs_all)


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


Data_Generate(θs_all[chunk_size_*(chunk_id_-1)+1 : chunk_size_*chunk_id_, :], chunk_id_)