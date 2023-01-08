#!/bin/sh


#SBATCH --time=168:00:00   # walltime
#SBATCH --ntasks=1         # number of processor cores (i.e. tasks)
#SBATCH --nodes=1          # number of nodes
#SBATCH --gres gpu:1
#SBATCH --mem-per-cpu=64G



python -u Validation_Test.py 


