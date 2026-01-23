#!/bin/bash
#SBATCH -o logs/MPCNO_reduce_data_vertex_centered.out
#SBATCH --qos=low
#SBATCH -p C064M0256G
#SBATCH -J MPCNO_reduce_data_vertex_centered
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=100:00:00

module load conda
source activate pytorch
python mpcno_geo_mixed_3d_reduce_data.py --n_train 1000 \
                                        --n_test 500 \
                                        --mesh_type "vertex_centered"  # "cell_centered" , "vertex_centered"
