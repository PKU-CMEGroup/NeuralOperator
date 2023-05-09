#!/bin/bash

#SBATCH --time=168:00:00   # walltime
#SBATCH --nodes=1         # number of nodes
#SBATCH --ntasks=1       # number of processor cores (i.e. tasks)
#SBATCH --mem-per-cpu=4G
#SBATCH -J "KS_Convergence"    # job name
#SBATCH -o "output/KS_Convergence"


# number of tasks
module load julia/1.7.1
julia eq1d_KS_Convergence.jl
