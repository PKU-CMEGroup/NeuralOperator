#!/bin/bash

#Submit this script with: sbatch calibrate_script

#SBATCH --time=168:00:00        # walltime
#SBATCH --nodes=1             # number of nodes
#SBATCH --ntasks=16           # number of processor cores (i.e. tasks)
#SBATCH --mem-per-cpu=16G
#SBATCH -J "burgers"               # job name
#SBATCH -o output/burgers-%A_%a.out

module purge
module load julia/1.8.1 


chunk_id=${SLURM_ARRAY_TASK_ID}
echo chunk_id
julia -p 16 eq1d_burgers.jl --chunk_id $chunk_id 
