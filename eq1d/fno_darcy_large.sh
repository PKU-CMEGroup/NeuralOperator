#!/bin/sh

n_data=16384;
for k_max in 4 8; do
    for d_f in 4 8 16 32 64 128; do
        for n_fno_layers in 3 4 5; do
            for downsample_ratio in 1 2 4 8; do

                scommand="sbatch --job-name=darcy_n_data_${n_data}_k_max_${k_max}_d_f_${d_f}_n_fno_layers_${n_fno_layers}_downsample_ratio_${downsample_ratio}
--output=output/darcy_n_data_${n_data}_k_max_${k_max}_d_f_${d_f}_n_fno_layers_${n_fno_layers}_downsample_ratio_${downsample_ratio}.out 
fno_darcy.sbatch $n_data $k_max $d_f $n_fno_layers $downsample_ratio"

                echo "submit command: $scommand"

                $scommand
            done
        done
    done
done 

