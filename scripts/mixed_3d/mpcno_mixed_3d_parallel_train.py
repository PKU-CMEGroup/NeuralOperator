import os
import torch
import sys
import argparse
from pathlib import Path
import gc

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
from timeit import default_timer


sys.path.append(str(Path(__file__).parent.parent))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from scripts.mixed_3d.mpcno_mixed_3d_helper import gen_data_tensors
from pcno.mpcno import compute_Fourier_modes, MPCNO, MPCNO_train_parallel

torch.set_printoptions(precision=16)




def setup_ddp(rank, local_rank, world_size):
    """Initialize distributed environment."""
    torch.cuda.set_device(local_rank)
    

    dist.init_process_group(
        backend="nccl",   
        init_method="env://",  
        rank=rank,
        world_size=world_size
    )
    
    # 设置随机种子以确保可重复性
    torch.manual_seed(0 + rank)  # 每个进程有不同的偏移
    np.random.seed(0 + rank)
    
    if rank == 0:
        print(f"Rank {rank}: CUDA device {torch.cuda.current_device()}")    



def cleanup_ddp():
    """清理分布式训练环境"""
    dist.destroy_process_group()



def train_ddp(rank, local_rank, world_size, args):
    # Initialize distributed environment
    setup_ddp(rank, local_rank, world_size)
    
    # Parse configuration from parameters
    layer_selection = {'grad': args.grad.lower() == "true", 'geo': args.geo.lower() == "true", 'geointegral': args.geointegral.lower() == "true"}
    f_in_dim = 0
    f_out_dim = 1
    train_inv_L_scale = False
    k_max = args.k_max
    ndim = 3
    layers = [int(size) for size in args.layer_sizes.split(",")]
    act = args.act
    geo_act = args.geo_act
    to_divide_factor = args.to_divide_factor
    mesh_type = args.mesh_type
    n_train = args.n_train
    n_test  = args.n_test
    
    

    if rank == 0:
        print("Loading and preprocessing data...")
        
    ###################################
    # load all data (CPU only)
    ###################################
    data_path = "../../data/mixed_3d_add_elem_features"
    # load data n_train + n_test
    # Note: All ranks need to load data, but we'll use DistributedSampler to distribute the data
    
    data = np.load(data_path+"/pcno_mixed_3d_"+mesh_type+"_n_train"+str(n_train)+"_n_test"+str(n_test)+".npz")
    names_array = np.load(data_path+"/pcno_mixed_3d_names_list"+"_n_train"+str(n_train)+"_n_test"+str(n_test)+".npy", allow_pickle=True)
    
    
    nnodes, node_mask, nodes = data["nnodes"], data["node_mask"], data["nodes"]
    
    node_weights = data["node_measures"]
    to_divide = to_divide_factor * np.amax(np.sum(node_weights, axis=1))
    node_weights = node_weights / to_divide

    directed_edges, edge_gradient_weights = data["directed_edges"], data["edge_gradient_weights"]
    features = data["features"]

    
    ndata = nodes.shape[0]
    assert(ndata == n_train + n_test)
    
    # delete data and release its memory
    del data
    gc.collect()

    if rank == 0:
        print(nnodes.shape,node_mask.shape,nodes.shape,flush = True)
        print(args)
        print('Node weights are devided by factor ', to_divide.item())
        print(f"ndata: {ndata},  n_train: {n_train}, n_test: {n_test}", flush=True)
        print("Casting to tensor", flush=True)
        
    nnodes = torch.from_numpy(nnodes)
    node_mask = torch.from_numpy(node_mask)
    nodes = torch.from_numpy(nodes.astype(np.float32))
    node_weights = torch.from_numpy(node_weights.astype(np.float32))
    features = torch.from_numpy(features.astype(np.float32))
    directed_edges = torch.from_numpy(directed_edges.astype(np.int64))
    edge_gradient_weights = torch.from_numpy(edge_gradient_weights.astype(np.float32))


    x_train, y_train, aux_train = gen_data_tensors(np.arange(n_train), nodes, features, node_mask, node_weights, directed_edges, edge_gradient_weights, f_in_dim, f_out_dim)
    x_test, y_test, aux_test = gen_data_tensors(np.arange(-n_test, 0), nodes, features, node_mask, node_weights, directed_edges, edge_gradient_weights, f_in_dim, f_out_dim)


    #！！！！！
    Ls = [4.1, 4.1, 1.5]
    
    if rank == 0:
        print(f'x_train shape {x_train.shape}, x_test shape {x_test.shape}, y_train shape {y_train.shape}, y_test shape {y_train.shape}', flush = True)
        print('length of each dim: ',torch.amax(nodes, dim = [0,1]) - torch.amin(nodes, dim = [0,1]), flush = True)
        print(f'kmax = {k_max}')
        print(f'n_train = {n_train}, n_test = {n_test}')
        print(f'Ls = {Ls}')
        print(f'layer_selection = {layer_selection}')
        print(f'layers = {layers}')
        print(f'activation = {act}')


    modes = compute_Fourier_modes(ndim, [k_max, k_max, k_max], Ls)
    modes = torch.tensor(modes, dtype=torch.float, device=f'cuda:{local_rank}')
    model = MPCNO(ndim, modes, nmeasures=1,
                layer_selection = layer_selection,
                layers=layers,
                fc_dim=128,
                in_dim=x_train.shape[-1], out_dim=y_train.shape[-1],
                inv_L_scale_hyper = [train_inv_L_scale, 0.5, 2.0],
                act = act,
                geo_act = geo_act,
                scaling_mode='sqrt_inv',
                ).to(local_rank)

    # Wrap the model with DDP
    ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    epochs = args.epochs
    base_lr = 5e-4
    lr_ratio = 10
    scheduler = "OneCycleLR"
    weight_decay = 1.0e-4
    batch_size = args.batch_size
    if rank == 0:
        print(f'batch_size = {batch_size}')

    normalization_x = False
    normalization_y = True
    normalization_dim_x = []
    normalization_dim_y = []
    non_normalized_dim_x = 4
    non_normalized_dim_y = 0


    config = {"train" : {"base_lr": base_lr, 'lr_ratio': lr_ratio, "weight_decay": weight_decay, "epochs": epochs, "scheduler": scheduler,  "batch_size": batch_size, 
                        "normalization_x": normalization_x,"normalization_y": normalization_y, 
                        "normalization_dim_x": normalization_dim_x, "normalization_dim_y": normalization_dim_y, 
                        "non_normalized_dim_x": non_normalized_dim_x, "non_normalized_dim_y": non_normalized_dim_y}
                        }


    train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = MPCNO_train_parallel(
        x_train, aux_train, y_train, x_test, aux_test, y_test, config, ddp_model, rank=rank, local_rank = local_rank, world_size=world_size, save_model_name="./PCNO_parallel_mixed_3d_model"
    )
    
    cleanup_ddp()




    
    
def main():
    parser = argparse.ArgumentParser(description='Train model with different configurations and options.')

    parser.add_argument('--grad', type=str, default='True', choices=['True', 'False'])
    parser.add_argument('--geo', type=str, default='True', choices=['True', 'False'])
    parser.add_argument('--geointegral', type=str, default='True', choices=['True', 'False'])
    parser.add_argument('--to_divide_factor', type=float, default=1.0)
    parser.add_argument('--k_max', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=500)
    # Preprocess data n_each from each subcategories
    parser.add_argument('--n_train', type=int, default=900)
    parser.add_argument('--n_test', type=int, default=100)
    parser.add_argument('--act', type=str, default="gelu")
    parser.add_argument('--geo_act', type=str, default="softsign")
    parser.add_argument('--scale', type=float, default=0.0)
    parser.add_argument("--layer_sizes", type=str, default="64,64,64,64,64,64")

    # Specifies how the computational mesh is represented. 
    # “cell_centered” stores features at cell centers (control-volume based), 
    # while “vertex_centered” stores features at mesh vertices (node-based).
    parser.add_argument('--mesh_type', type=str, default='cell_centered', choices=['cell_centered', 'vertex_centered'])
    

    args = parser.parse_args()
    

    # 获取当前进程的rank和world_size（torchrun自动设置）
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    train_ddp(rank, local_rank, world_size, args)
    
if __name__ == "__main__":
    main()
