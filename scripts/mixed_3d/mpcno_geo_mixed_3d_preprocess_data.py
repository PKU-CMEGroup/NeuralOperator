import os
import torch
import argparse
import sys
import numpy as np
from pathlib import Path

from scripts.mixed_3d.mpcno_geo_mixed_3d_helper import (
    Plane_datasets,
    DrivAerNet_datasets,
    load_data,
)

sys.path.append(str(Path(__file__).parent.parent))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from pcno.geo_utility import preprocess_data_mesh, compute_node_weights, compute_outward_normals, element_features_to_vertices



torch.set_printoptions(precision=16)


torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model with different configurations and options.')
    # Preprocess data n_each from each subcategories
    parser.add_argument('--n_each', type=int, default=100)
    
    # Specifies how the computational mesh is represented. 
    # “cell_centered” stores features at cell centers (control-volume based), 
    # while “vertex_centered” stores features at mesh vertices (node-based).
    parser.add_argument('--mesh_type', type=str, default='cell_centered', choices=['cell_centered', 'vertex_centered'])
    args = parser.parse_args()
    mesh_type = args.mesh_type
    n_each  = args.n_each 
    
    ###################################
    # load data
    ###################################
    data_path = "../../data/mixed_3d_add_elem_features"
    
                                                                                                

    print("Loading data: ", n_each, " from each datasets")
    print("Plane datasets: ", Plane_datasets)
    print("DrivAerNet datasets: ", DrivAerNet_datasets)
    
    nodes_list, elems_list, elem_features_list, names_list =  load_data(data_path, 
                                                        Plane_datasets,
                                                        DrivAerNet_datasets,
                                                        n_each)
    ndata = len(nodes_list)
    elem_normals_list = compute_outward_normals(nodes_list, elems_list)
    # concatenate features behind normals 
    elem_features_list = [np.concatenate((elem_normals_list[i], elem_features_list[i]),axis=1) for i in range(ndata)]
    
    features_list =  elem_features_list if mesh_type == "cell_centered" else element_features_to_vertices(nodes_list, elems_list, elem_features_list, reduction = "area")
    
    print("Preprocessing data")
    nnodes, node_mask, nodes, node_measures_raw, features, directed_edges, edge_gradient_weights = preprocess_data_mesh(nodes_list, elems_list, features_list, mesh_type = mesh_type, adjacent_type="edge")
    node_measures, _ = compute_node_weights(nnodes,  node_measures_raw,  equal_measure = False)
    np.savez_compressed(data_path+"/pcno_mixed_3d_"+mesh_type+".npz", \
                        nnodes=nnodes, node_mask=node_mask, nodes=nodes, \
                        node_measures_raw=node_measures_raw, \
                        node_measures=node_measures, \
                        features=features, \
                        directed_edges=directed_edges, edge_gradient_weights=edge_gradient_weights) 
    
    np.save(os.path.join(data_path, "pcno_mixed_3d_names_list.npy"), np.array(names_list, dtype=object))

    