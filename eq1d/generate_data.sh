#!/bin/sh

#SBATCH --time=168:00:00   # walltime
#SBATCH --nodes=1
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH -J "eq1d"    # job name
#SBATCH --mem-per-cpu=64G
#SBATCH --no-requeue
#SBATCH --output=eq1d_generate_data.out

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE



module load julia/1.7.1

export JULIA_NUM_THREADS=12

julia eq1d_data.jl


