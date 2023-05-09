#!/bin/bash

#SBATCH --time=168:00:00   # walltime
#SBATCH --nodes=1         # number of nodes
#SBATCH --ntasks=1       # number of processor cores (i.e. tasks)
#SBATCH --mem-per-cpu=4G
#SBATCH -J "generate_KS_data"    # job name
#SBATCH -o "output/generate_KS_data"


# number of tasks
n=16
sbatch --array=1-$n generate_KS_helper.sh
