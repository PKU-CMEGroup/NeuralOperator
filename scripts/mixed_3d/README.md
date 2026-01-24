# Mixed_3d Test

## Datasets

link: https://disk.pku.edu.cn/link/AA581EA0843A6441E8BC6CDBE18DE5EA15
file name：mixed_3d.zip

# Warning there are redundant nodes in original mesh files
# there are redundant nodes in ./Plane/J20/0418.npz
# node 4674 : [ 0.7249122  -0.17730147 -0.03439951]
# node 4673 : [ 0.7249122  -0.17730147 -0.03439951]
# there are redundant nodes in ./Plane/P180/1401.npz
# node 2129 : [ 0.36551496  0.08711994 -0.14670402]
# node 2121 : [ 0.36551496  0.08711994 -0.14670402]
# replace ./Plane/J20/0418.npz by ./Plane/J20/0418.npz
# replace ./Plane/P180/1401.npz by ./Plane/P180/1401.npz

Step 1: sbatch mpcno_preprocess_data.sh  
Preprocess all data

Setp 2: sbatch mpcno_reduce_data.sh
Reduce data, random shuffle and save the first n_train data and the last n_test data to reduce data size for training 


Setp 3: sbatch mpcno_mixed_3d_train.sh
Training with a single GPU


Setp 3: sbatch mpcno_mixed_3d_parallel_train.sh
Training with multiple GPUs
