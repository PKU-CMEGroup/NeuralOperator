import os
import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from scripts.mixed_3d.mpcno_mixed_3d_helper import (
    Plane_datasets,
    DrivAerNet_datasets,
    random_shuffle
)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model with different configurations and options.')

    parser.add_argument('--n_train', type=int, default=2000)
    parser.add_argument('--n_test', type=int, default=500)
    
    # Specifies how the computational mesh is represented. 
    # “cell_centered” stores features at cell centers (control-volume based), 
    # while “vertex_centered” stores features at mesh vertices (node-based).
    parser.add_argument('--mesh_type', type=str, default='cell_centered', choices=['cell_centered', 'vertex_centered'])
    args = parser.parse_args()


    n_train = args.n_train
    n_test  = args.n_test
    mesh_type = args.mesh_type


    ###################################
    # load data
    ###################################
    data_path = "../../data/mixed_3d_add_elem_features"
 
    # load data n_train + n_test
    equal_weights = False
    data = np.load(data_path+"/pcno_mixed_3d_"+mesh_type+".npz")
    names_array = np.load(data_path+"/pcno_mixed_3d_names_list.npy", allow_pickle=True)
    
    # random shuffle, and keep only n_train + n_test data
    data, names_list = random_shuffle(data, names_array, n_train, n_test, seed=42)
    
    
    np.savez(
            data_path+"/pcno_mixed_3d_"+mesh_type+"_n_train"+str(n_train)+"_n_test"+str(n_test)+".npz",
            **data
        )
    
    np.save(os.path.join(data_path, "pcno_mixed_3d_names_list"+"_n_train"+str(n_train)+"_n_test"+str(n_test)+".npy"), names_list)



