#!/bin/bash

#SBATCH --time=168:00:00   # walltime
#SBATCH --nodes=1         # number of nodes
#SBATCH --ntasks=1       # number of processor cores (i.e. tasks)
#SBATCH --mem-per-cpu=4G
#SBATCH -J "generate_burgers_data"    # job name
#SBATCH -o "output/generate_burgers_data"


# number of tasks
n=16
sbatch --array=1-$n generate_burgers_helper.sh
