import os
import glob
import random
import torch
import sys
import numpy as np
import math
from timeit import default_timer
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


from pcno.geo_utility import preprocess_data, compute_node_weights
from pcno.pcno_eff import compute_Fourier_modes, PCNO_EFF, PCNO_train

torch.set_printoptions(precision=16)
torch.manual_seed(0)
np.random.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data(data_path):
    ndata, elem_dim = 1211, 2
    nodes_list, elems_list, features_list = [], [], []
    all_simulation_parameters = np.load(data_path+"/all_simulation_parameters.npy") #id, pressure, c0, c1, c2
    NNODES = 11917
    for i in range(ndata):      
        
        deformed_nodes = np.load(data_path+"/coordinates_%05d"%(i+1)+".npy")  # This is the deformed shape
        nnodes = deformed_nodes.shape[0]
        if nnodes != NNODES:  # TODO something wrong with some meshes
            continue
        displacement = np.load(data_path+"/displacements_%05d"%(i+1)+".npy")
        nodes_list.append(deformed_nodes - displacement)

        elems = np.load(data_path+"/quad_connectivity_%05d"%(i+1)+".npy")
        elems_list.append(np.concatenate((np.full((elems.shape[0], 1), elem_dim, dtype=int), elems), axis=1))
        

        displacement = np.load(data_path+"/displacements_%05d"%(i+1)+".npy")
        strain = np.load(data_path+"/lagrange_strain_%05d"%(i+1)+".npy")[:,[0,1,2,4,5,8]]  # xx xy xz ; yx yy yz ; zx zy zz
        stress = np.load(data_path+"/cauchy_stress_%05d"%(i+1)+".npy")[:,[0,1,2,4,5,8]]
        
        simulation_parameters = np.tile(all_simulation_parameters[i,1:], (nnodes, 1))
        features_list.append(np.hstack((simulation_parameters, displacement, stress, strain)))

    return nodes_list, elems_list, features_list 


try:
    PREPROCESS_DATA = sys.argv[1] == "preprocess_data" if len(sys.argv) > 1 else False
except IndexError:
    PREPROCESS_DATA = False


parser = argparse.ArgumentParser(description='Train model with different types.')
parser.add_argument('--equal_weight', type=str, default='False', help='Specify whether to use equal weight')
parser.add_argument('--train_sp_L', type=str, default='False', choices=['False' , 'together' , 'independently'],
                    help='type of train_sp_L (False, together, independently )')

parser.add_argument('--lr_ratio', type=float, default=10, help='lr ratio for independent training for L')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')

if not PREPROCESS_DATA:
    args = parser.parse_args()
    args_dict = vars(args)
    for i, (key, value) in enumerate(args_dict.items()):
        print(f"{key}: {value}")

###################################
# load data
###################################
data_path = "../../data/mitral_valve/single_geometry/"




if PREPROCESS_DATA:
    print("Loading data")
    nodes_list, elems_list, features_list  = load_data(data_path = data_path)
    
    print("Preprocessing data")
    nnodes, node_mask, nodes, node_measures_raw, features, directed_edges, edge_gradient_weights = preprocess_data(nodes_list, elems_list, features_list)
    node_measures, node_weights = compute_node_weights(nnodes,  node_measures_raw,  equal_measure = False)
    node_equal_measures, node_equal_weights = compute_node_weights(nnodes,  node_measures_raw,  equal_measure = True)
    np.savez_compressed(data_path+"mitral_valve_single_geometry_data.npz", \
                        nnodes=nnodes, node_mask=node_mask, nodes=nodes, \
                        node_measures_raw = node_measures_raw, \
                        node_measures=node_measures, node_weights=node_weights, \
                        node_equal_measures=node_equal_measures, node_equal_weights=node_equal_weights, \
                        features=features, \
                        directed_edges=directed_edges, edge_gradient_weights=edge_gradient_weights) 
    exit()
else:
    # load data 
    
    equal_weights = args.equal_weight.lower() == "true"

    data = np.load(data_path+"mitral_valve_single_geometry_data.npz")
    nnodes, node_mask, nodes = data["nnodes"], data["node_mask"], data["nodes"]
    node_measures = data["node_measures"]
    node_weights = data["node_equal_weights"] if equal_weights else data["node_weights"]
    directed_edges, edge_gradient_weights = data["directed_edges"], data["edge_gradient_weights"]
    features = data["features"]

    node_measures_raw = data["node_measures_raw"]
    indices = np.isfinite(node_measures_raw)
    node_rhos = np.copy(node_weights)
    node_rhos[indices] = node_rhos[indices]/node_measures[indices]


print("Casting to tensor")
nnodes = torch.from_numpy(nnodes)
node_mask = torch.from_numpy(node_mask)
nodes = torch.from_numpy(nodes.astype(np.float32))
node_weights = torch.from_numpy(node_weights.astype(np.float32))
node_rhos = torch.from_numpy(node_rhos.astype(np.float32))

#stress and pressure have the same unit, range O(1)
#strain does not have an unit, range O(0.01)
#displacement and nodes have the same unit, range O(1)
L_ref, p_ref, c_ref, strain_ref = 1.0, 1.0, 100, 0.01
nodes_input = nodes.clone()/L_ref  # scale length input
#pressre, c, displacements, stress, strain
features = features / np.array([p_ref] * 1 + [c_ref] * 3 + [L_ref] * 3 + [p_ref] * 6 + [strain_ref] * 6 )
features = torch.from_numpy(features.astype(np.float32))

directed_edges = torch.from_numpy(directed_edges.astype(np.int64))
edge_gradient_weights = torch.from_numpy(edge_gradient_weights.astype(np.float32))



n_train, n_test = 1000, 200


x_train, x_test = torch.cat((features[:n_train, :, 0:4], nodes_input[:n_train, ...], node_rhos[:n_train, ...]), -1), torch.cat((features[-n_test:, :, 0:4], nodes_input[-n_test:, ...], node_rhos[-n_test:, ...]),-1)
aux_train       = (node_mask[0:n_train,...], nodes[0:n_train,...], node_weights[0:n_train,...], directed_edges[0:n_train,...], edge_gradient_weights[0:n_train,...])
aux_test        = (node_mask[-n_test:,...],  nodes[-n_test:,...],  node_weights[-n_test:,...],  directed_edges[-n_test:,...],  edge_gradient_weights[-n_test:,...])
#predict displacement, stress
y_train, y_test = features[:n_train, :, 4:4+3+6],     features[-n_test:, :, 4:4+3+6]


k_max = 12
ndim = 3

print("Nodes range: ", [torch.max(nodes[:,i]) - torch.min(nodes[:,i]) for i in range(3)])

modes = compute_Fourier_modes(ndim, [k_max,k_max,k_max], [40.0,40.0,25.0])

modes = torch.tensor(modes, dtype=torch.float).to(device)

if args.train_sp_L == 'False':
    args.train_sp_L = False
train_sp_L = args.train_sp_L

model = PCNO_EFF(ndim, modes, nmeasures=1,
               layers=[128,128,128,128,128],
               g_dim = 32,
               fc_dim = 128,
               in_dim = x_train.shape[-1], out_dim=y_train.shape[-1], 
               train_sp_L = train_sp_L,
               act='gelu').to(device)



epochs = 500
base_lr = 5e-4 #0.001
lr_ratio = args.lr_ratio
scheduler = "OneCycleLR"
weight_decay = 1.0e-4
batch_size = args.batch_size

normalization_x = False 
normalization_y = True 
normalization_dim_x = []
normalization_dim_y = []
non_normalized_dim_x = 0
non_normalized_dim_y = 0


print('normalizer: ',f'{normalization_x},{normalization_y},{normalization_dim_x},{normalization_dim_y},{non_normalized_dim_x},{non_normalized_dim_y}')


config = {"train" : {"base_lr": base_lr, 'lr_ratio': lr_ratio, "weight_decay": weight_decay, "epochs": epochs, "scheduler": scheduler,  "batch_size": batch_size, 
                     "normalization_x": normalization_x,"normalization_y": normalization_y, 
                     "normalization_dim_x": normalization_dim_x, "normalization_dim_y": normalization_dim_y, 
                     "non_normalized_dim_x": non_normalized_dim_x, "non_normalized_dim_y": non_normalized_dim_y}
                     }


train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = PCNO_train(
    x_train, aux_train, y_train, x_test, aux_test, y_test, config, model, save_model_name="./PCNO_mitral_valve_single_geometry_model"
)
