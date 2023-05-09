#!/bin/sh


#SBATCH --time=168:00:00   # walltime
#SBATCH --ntasks=1         # number of processor cores (i.e. tasks)
#SBATCH --nodes=1          # number of nodes
#SBATCH --gres gpu:1
#SBATCH --mem-per-cpu=64G


echo KS_n_data_${1}_k_max_${2}_downsample_ratio_${3}_n_fno_layers_${4}_d_f_${5}

python -u eq1d_fno_KS.py $1 $2 $3 $4 $5


