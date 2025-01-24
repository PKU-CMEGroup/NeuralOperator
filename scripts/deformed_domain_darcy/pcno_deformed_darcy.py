import random
import torch
import sys
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from timeit import default_timer
from scipy.io import loadmat
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


from pcno.geo_utility import preprocess_data, convert_structured_data, compute_node_weights
from pcno.pcno import compute_Fourier_modes, PCNO, PCNO_train



torch.set_printoptions(precision=16)

torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

try:
    PREPROCESS_DATA = sys.argv[1] == "preprocess_data" if len(sys.argv) > 1 else False
except IndexError:
    PREPROCESS_DATA = False


parser = argparse.ArgumentParser(description='Train model with different configurations and options.')


parser.add_argument('--n_train', type=int, default=1000, help='Number of training samples')
parser.add_argument('--n_test', type=int, default=200, help='Number of testing samples')
parser.add_argument('--train_type', type=str, default='mixed', choices=['fine', 'coarse', 'mixed'], help='Type of training data')
parser.add_argument('--equal_weight', type=str, default='False', help='Specify whether to use equal weight')


parser.add_argument('--Lx', type=float, default=2.0, help='Initial value for the length of the x dimension')
parser.add_argument('--Ly', type=float, default=2.0, help='Initial value for the length of the y dimension')
parser.add_argument('--train_sp_L', type=str, default='independently', choices=['False', 'together', 'independently'], help='type of train_sp_L (False, together, independently)')

parser.add_argument('--normalization_x', type=str, default='False', help='Whether to normalize the x dimension (True/False)')
parser.add_argument('--normalization_y', type=str, default='False', help='Whether to normalize the y dimension (True/False)')


parser.add_argument('--lr_ratio', type=float, default=10, help='Learning rate ratio of main parameters and L parameters when train_sp_L is set to `independently`')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')


args = parser.parse_args()

save_model_name = f"PCNO_darcy_{args.train_type}_n{args.n_train}"
print(save_model_name)
args_dict = vars(args)
for i, (key, value) in enumerate(args_dict.items()):
    print(f"{key}: {value}")

data_path = "/lustre/home/2401110057/PCNO/data/"

if PREPROCESS_DATA:
    ###################################
    # load data
    ###################################
    nodes_list=[]
    elems_list=[]
    features_list=[]
    for i in range(2000):
        ##
        nodes=np.load(data_path+'darcy_deformed_domain/smooth_small_scale/'+'nodes_'+str(i).zfill(5)+'.npy')
        elems=np.load(data_path+'darcy_deformed_domain/smooth_small_scale/'+'elements_'+str(i).zfill(5)+'.npy')
        features=np.load(data_path+'darcy_deformed_domain/smooth_small_scale/'+'features_'+str(i).zfill(5)+'.npy')

        nodes_list.append(nodes)
        elems_list.append(elems)
        features_list.append(features)
        ##smooth数据
    for i in range(2000):
        ##
        nodes=np.load(data_path+'darcy_deformed_domain/smooth_large_scale/'+'nodes_'+str(i).zfill(5)+'.npy')
        elems=np.load(data_path+'darcy_deformed_domain/smooth_large_scale/'+'elements_'+str(i).zfill(5)+'.npy')
        features=np.load(data_path+'darcy_deformed_domain/smooth_large_scale/'+'features_'+str(i).zfill(5)+'.npy')

        nodes_list.append(nodes)
        elems_list.append(elems)
        features_list.append(features)
        ##large smooth数据


        
    

    print(nodes_list[401].shape)
    
        

    #uniform weights
    nnodes, node_mask, nodes, node_measures_raw, features, directed_edges, edge_gradient_weights = preprocess_data(nodes_list, elems_list, features_list)
    node_measures, node_weights = compute_node_weights(nnodes,  node_measures_raw,  equal_measure = False)
    node_equal_measures, node_equal_weights = compute_node_weights(nnodes,  node_measures_raw,  equal_measure = True)
    np.savez_compressed("pcno_darcy_data.npz", \
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

    data = np.load("pcno_darcy_data.npz")
    nnodes, node_mask, nodes = data["nnodes"], data["node_mask"], data["nodes"]
    node_weights = data["node_equal_weights"] if equal_weights else data["node_weights"]
    node_measures = data["node_measures"]
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
#!! compress measures
# node_weights = torch.sum(node_weights, dim=-1).unsqueeze(-1)

features = torch.from_numpy(features.astype(np.float32))
directed_edges = torch.from_numpy(directed_edges)
edge_gradient_weights = torch.from_numpy(edge_gradient_weights.astype(np.float32))


nodes_input = nodes.clone()
n_train = args.n_train
n_test = args.n_test


#rows_train=np.concatenate((np.arange(0,n_train),np.arange(2000,2000+n_train)))#mixed data
#rows_train = np.arange(2000,2000+n_train) #coarse data
#rows_train = np.arange(0,n_train) #fine data
train_type = args.train_type
if train_type == "fine":
    rows_train = np.arange(0,n_train) #fine data
elif train_type == "coarse":
    rows_train = np.arange(2000,2000+n_train) #coarse data
elif train_type == "mixed":
    rows_train=np.concatenate((np.arange(0,n_train/2),np.arange(2000,2000+n_train/2)))#mixed data

rows_test_fine   = np.arange(2000-n_test,2000) # fine test
rows_test_coarse = np.arange(4000-n_test,4000) # coarse test



x_train, x_test = torch.cat((features[rows_train, :, 0:1], nodes_input[rows_train, ...], node_rhos[rows_train, ...]),-1), torch.cat((features[rows_test_fine, :, 0:1], nodes_input[rows_test_fine, ...], node_rhos[rows_test_fine, ...]),-1)
aux_train       = (node_mask[rows_train,...], nodes[rows_train,...], node_weights[rows_train,...], directed_edges[rows_train,...], edge_gradient_weights[rows_train,...])
aux_test        = (node_mask[rows_test_fine,...],  nodes[rows_test_fine,...],  node_weights[rows_test_fine,...],  directed_edges[rows_test_fine,...],  edge_gradient_weights[rows_test_fine,...])
y_train, y_test = features[rows_train, :, 1:2],       features[rows_test_fine, :, 1:2]

#x_test_coarse   = torch.cat((features[rows_test_coarse, :, 0:1], nodes_input[rows_test_coarse, ...], node_rhos[rows_test_coarse, ...]),-1)
#y_test_coarse   = features[rows_test_coarse, :, 1:2]
#aux_test_coarse = (node_mask[rows_test_coarse,...], nodes[rows_test_coarse,...], node_weights[rows_test_coarse,...], directed_edges[rows_test_coarse,...], edge_gradient_weights[rows_test_coarse,...])

#x_train, x_test = torch.cat((features[:n_train, :, [0]], nodes_input[:n_train, ...]),-1), torch.cat((features[-n_test:, :, [0]], nodes_input[-n_test:, ...]),-1)
#aux_train       = (node_mask[0:n_train,...], nodes[0:n_train,...], node_weights[0:n_train,...], directed_edges[0:n_train,...], edge_gradient_weights[0:n_train,...])
#aux_test        = (node_mask[-n_test:,...],  nodes[-n_test:,...],  node_weights[-n_test:,...],  directed_edges[-n_test:,...],  edge_gradient_weights[-n_test:,...])
#y_train, y_test = features[:n_train, :, [1]],       features[-n_test:, :, [1]]


k_max = 16
ndim = 2
if args.train_sp_L == 'False':
    args.train_sp_L = False
train_sp_L = args.train_sp_L
Lx, Ly = args.Lx, args.Ly

modes = compute_Fourier_modes(ndim, [k_max,k_max], [Lx,Ly])
modes = torch.tensor(modes, dtype=torch.float).to(device)
model = PCNO(ndim, modes, nmeasures=1,
               layers=[128,128,128,128,128],
               fc_dim=128,
               train_sp_L=train_sp_L,
               in_dim=4, out_dim=1,
               act='gelu').to(device)



epochs = 500
base_lr = 0.001
scheduler = "OneCycleLR"
weight_decay = 1.0e-4
batch_size = args.batch_size
lr_ratio = args.lr_ratio

normalization_x = args.normalization_x.lower() == "true"
normalization_y = args.normalization_y.lower() == "true"
normalization_dim_x = []
normalization_dim_y = []
non_normalized_dim_x = 2
non_normalized_dim_y = 0


config = {"train" : {"base_lr": base_lr, "weight_decay": weight_decay, "epochs": epochs, "scheduler": scheduler,  "batch_size": batch_size, "lr_ratio" : lr_ratio,
                     "normalization_x": normalization_x,"normalization_y": normalization_y, 
                     "normalization_dim_x": normalization_dim_x, "normalization_dim_y": normalization_dim_y, 
                     "non_normalized_dim_x": non_normalized_dim_x, "non_normalized_dim_y": non_normalized_dim_y}
                     }

train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = PCNO_train(
    x_train, aux_train, y_train, x_test, aux_test, y_test, config, model, save_model_name="save_model_name"
)





