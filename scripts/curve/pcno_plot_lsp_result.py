import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utility.normalizer import UnitGaussianNormalizer
from utility.losses import LpLoss
from pcno.pcno_geo import compute_Fourier_modes,PCNO

FONTSIZE = 17

def gen_data_tensors(data_indices, nodes, features, node_mask, node_weights, directed_edges, edge_gradient_weights):
    f_in_dim = 1
    f_out_dim = 1

    nodes_input = nodes.clone()
    x = torch.cat((features[data_indices][...,:f_in_dim+2],
                            nodes_input[data_indices, ...]), -1)
    y = features[data_indices][...,-f_out_dim:]
    nx = features[data_indices][...,f_in_dim:f_in_dim+2]
    aux = (node_mask[data_indices], nodes[data_indices], node_weights[data_indices], directed_edges[data_indices], edge_gradient_weights[data_indices], nx.permute(0,2,1))
    
    return x, y, aux

def get_median_index(arr):
    # 确保输入是一个 NumPy 数组
    arr = np.asarray(arr)
    # 获取排序后的索引
    sorted_indices = np.argsort(arr)
    # 计算中位数的索引
    mid_index = len(arr) // 2
    
    if len(arr) % 2 == 1:
        # 如果是奇数长度，返回中间元素的原始索引
        median_index = sorted_indices[mid_index]
    else:
        # 如果是偶数长度，返回中间两个元素的原始索引
        median_index_1 = sorted_indices[mid_index - 1]
        median_index_2 = sorted_indices[mid_index]
        # 通常我们不会为偶数长度的数组返回单个索引，因为中位数是两个值的平均。
        # 但是，如果你需要，你可以选择返回这两个索引或仅其中一个。
        # 这里我们简单地返回一个元组
        median_index = median_index_1
    
    return median_index

def visualize_curve(nodes, features, two_circles = False, figurename = ''):
    print(nodes.shape,features.shape)

    N = len(nodes) if not two_circles else len(nodes) // 2
    if two_circles:
        elems1 = np.stack([np.full(N, 1, dtype=int), np.arange(N), (np.arange(N) + 1) % N], axis=1)
        elems2 = np.stack([np.full(N, 1, dtype=int), np.arange(N, 2*N), (np.arange(N, 2*N) + 1 - N) % N + N], axis=1)
        elems = np.concatenate([elems1, elems2], axis=0)  # 2N, 3
    else:
        elems = np.stack([np.full(N, 1, dtype=int), np.arange(N), (np.arange(N)+1)%N], axis = 1)
    
    plt.figure()
    scatter_g = plt.scatter(nodes[:, 0], nodes[:, 1], c=features[:, 3], cmap='viridis', s=40)
    # plt.plot(nodes[:, 0], nodes[:, 1], color='gray', alpha=0.5)
    plt.colorbar(scatter_g, label='g1(x)')
    plt.title('Feature g1(x) on Random Polar Curve')
    plt.axis('equal')
    for elem in elems:
        node_indices = elem[1:]
        valid_indices = node_indices[node_indices != -1]
        if len(valid_indices) > 1:
            elem_nodes = nodes[valid_indices]
            plt.plot(elem_nodes[:, 0], elem_nodes[:, 1], color='red', linewidth=1, alpha=0.7)
        plt.tight_layout()
    if figurename:
        plt.savefig(figurename)
    else:
        plt.show()

def predict_error(data_ids = None, data_ids_two = None):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ##############################################
    # load data
    ##############################################
    data = np.load("../../data/curve/pcno_curve_data_1_1_5_2d_sp_laplace_panel.npz")
    
    n_data = 10000
    n_train = 8000
    n_test = 1000

    to_divide_factor = 20.0
    f_in_dim, f_out_dim = 0, 1
    nnodes, node_mask, nodes = data["nnodes"], data["node_mask"], data["nodes"]
    print(nnodes.shape,node_mask.shape,nodes.shape,flush = True)
    node_weights = data["node_measures"]
    
    to_divide = to_divide_factor * np.amax(np.sum(node_weights, axis = 1))
    print('Node weights are devided by factor ', to_divide.item())
    node_weights = node_weights / to_divide
    node_measures = data["node_measures"]
    directed_edges, edge_gradient_weights = data["directed_edges"], data["edge_gradient_weights"]
    features = data["features"]

    ndata = nodes.shape[0]
    print(f"ndata: {ndata},  n_train: {n_train}, n_test: {n_test}", flush=True)
        
    print("Casting to tensor", flush=True)
    nnodes = torch.from_numpy(nnodes)
    node_mask = torch.from_numpy(node_mask)
    nodes = torch.from_numpy(nodes.astype(np.float32))
    node_weights = torch.from_numpy(node_weights.astype(np.float32))
    features = torch.from_numpy(features.astype(np.float32))
    directed_edges = torch.from_numpy(directed_edges.astype(np.int64))
    edge_gradient_weights = torch.from_numpy(edge_gradient_weights.astype(np.float32))


    x_train, y_train, aux_train = gen_data_tensors(np.arange(n_train), nodes, features, node_mask, node_weights, directed_edges, edge_gradient_weights)
    x_test, y_test, aux_test = gen_data_tensors(np.arange(-n_test, 0), nodes, features, node_mask, node_weights, directed_edges, edge_gradient_weights)


    print(f'x_train shape {x_train.shape}, x_test shape {x_train.shape}, y_train shape {y_train.shape}, y_test shape {y_train.shape}', flush = True)
    print('length of each dim: ',torch.amax(nodes, dim = [0,1]) - torch.amin(nodes, dim = [0,1]), flush = True)
    
    normalization_x = False
    normalization_y = True
    normalization_dim_x = []
    normalization_dim_y = []
    non_normalized_dim_x = 4
    non_normalized_dim_y = 0

    if normalization_x:
        x_normalizer = UnitGaussianNormalizer(x_train, non_normalized_dim = non_normalized_dim_x, normalization_dim=normalization_dim_x)
        x_test = x_normalizer.encode(x_test)
        x_normalizer.to(device)
        
    if normalization_y:
        y_normalizer = UnitGaussianNormalizer(y_train, non_normalized_dim = non_normalized_dim_y, normalization_dim=normalization_dim_y)
        y_test = y_normalizer.encode(y_test)
        y_normalizer.to(device)

    ndim = 2
    k_max = 32
    L = 5
    modes = compute_Fourier_modes(ndim, [k_max,k_max], [L,L])
    modes = torch.tensor(modes, dtype=torch.float).to(device)
    model = PCNO(ndim, modes, nmeasures=1, 
                layer_selection = {'grad': "true", 'geo': "true", 'geointegral': "true"},
                layers=[64,64,64,64,64,64],
                fc_dim=128,
                in_dim=x_train.shape[-1], out_dim=y_train.shape[-1],
                inv_L_scale_hyper = [False, 0.5, 2.0],
                act = 'gelu',
                ).to(device)
    checkpoint = torch.load('checkpoint.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    if data_ids is None:
        test_rel_l2 = np.zeros(n_test)
        myloss = LpLoss(d=1, p=2, size_average=False)
        for i in tqdm(range(n_test)):
            x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo = x_test[[i],...], y_test[[i],...], aux_test[0][[i],...], aux_test[1][[i],...], aux_test[2][[i],...], aux_test[3][[i],...], aux_test[4][[i],...], aux_test[5][[i],...]
            x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo = x.to(device), y.to(device), node_mask.to(device), nodes.to(device), node_weights.to(device), directed_edges.to(device), edge_gradient_weights.to(device), geo.to(device)

            batch_size_ = x.shape[0]
            out = model(x, (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo)) #.reshape(batch_size_,  -1)

            if normalization_y:
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)
            out=out*node_mask # mask the padded value with 0,(1 for node, 0 for padding)
            test_rel_l2[i] = myloss(out.view(batch_size_,-1), y.view(batch_size_,-1)).item()
        
        np.save('test_rel_l2.npy', test_rel_l2)
    
        largest_error_ind = np.argmax(test_rel_l2)
        median_error_ind = get_median_index(test_rel_l2)  # Get the index (or indices)
        print("largest error is ", test_rel_l2[largest_error_ind], " ; median error is ", test_rel_l2[median_error_ind])
        print("largest error index is ", largest_error_ind, " ; median error index is ", median_error_ind)
        # they are only test data id
        data_ids = [largest_error_ind, median_error_ind]
    
    for data_id in data_ids:
        
        x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo = (x_test[[data_id],...], y_test[[data_id],...], aux_test[0][[data_id],...], aux_test[1][[data_id],...], aux_test[2][[data_id],...], aux_test[3][[data_id],...], aux_test[4][[data_id],...], aux_test[5][[data_id],...])
        
        x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo = x.to(device), y.to(device), node_mask.to(device), nodes.to(device), node_weights.to(device), directed_edges.to(device), edge_gradient_weights.to(device), geo.to(device)

        batch_size_ = x.shape[0]
        out = model(x, (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo)) #.reshape(batch_size_,  -1)
        if normalization_y:
            out = y_normalizer.decode(out)
            y = y_normalizer.decode(y)

        node_mask_bool = node_mask[0,:,0].bool().cpu().numpy()
        y = y.cpu().detach().numpy()[0, node_mask_bool ,0]
        out = out.cpu().detach().numpy()[0, node_mask_bool ,0]
        
        np.savez("single_"+str(data_id)+".npz",accu=y,pred=out)
        
    ##############################################
    # load data of two
    ##############################################
    data_two = np.load("../../data/curve/pcno_curve_data_1_1_5_2d_sp_laplace_panel_two_circles.npz")   

    nnodes_two, node_mask_two, nodes_two = data_two["nnodes"], data_two["node_mask"], data_two["nodes"]
    print(nnodes_two.shape,node_mask_two.shape,nodes_two.shape,flush = True)
    node_weights_two = data_two["node_measures"]
    
    to_divide_two = to_divide_factor * np.amax(np.sum(node_weights_two, axis = 1))
    print('Node weights are devided by factor ', to_divide_two.item())
    node_weights_two = node_weights_two / to_divide
    node_measures_two = data_two["node_measures"]
    directed_edges_two, edge_gradient_weights_two = data_two["directed_edges"], data_two["edge_gradient_weights"]
    features_two = data_two["features"]

    ndata_two = nodes_two.shape[0]

    print("Casting to tensor", flush=True)
    nnodes_two = torch.from_numpy(nnodes_two)

    node_mask_two = torch.from_numpy(node_mask_two)
    nodes_two = torch.from_numpy(nodes_two.astype(np.float32))
    node_weights_two = torch.from_numpy(node_weights_two.astype(np.float32))
    features_two = torch.from_numpy(features_two.astype(np.float32))
    directed_edges_two = torch.from_numpy(directed_edges_two.astype(np.int64))
    edge_gradient_weights_two = torch.from_numpy(edge_gradient_weights_two.astype(np.float32))

    x_test_two, y_test_two, aux_test_two = gen_data_tensors(np.arange(-n_test, 0), nodes_two, features_two, node_mask_two, node_weights_two, directed_edges_two, edge_gradient_weights_two)

    print('length of each dim: ',torch.amax(nodes_two, dim = [0,1]) - torch.amin(nodes_two, dim = [0,1]), flush = True)

    if normalization_x:
        x_test_two = x_normalizer.encode(x_test_two)
        
    if normalization_y:
        y_test_two = y_normalizer.encode(y_test_two)

    if data_ids_two is None:
        test_rel_l2_two = np.zeros(n_test)
        myloss = LpLoss(d=1, p=2, size_average=False)
        for i in tqdm(range(n_test)):
            x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo = x_test_two[[i],...], y_test_two[[i],...], aux_test_two[0][[i],...], aux_test_two[1][[i],...], aux_test_two[2][[i],...], aux_test_two[3][[i],...], aux_test_two[4][[i],...], aux_test_two[5][[i],...]
            x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo = x.to(device), y.to(device), node_mask.to(device), nodes.to(device), node_weights.to(device), directed_edges.to(device), edge_gradient_weights.to(device), geo.to(device)

            batch_size_ = x.shape[0]
            out = model(x, (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo)) #.reshape(batch_size_,  -1)

            if normalization_y:
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)
            out=out*node_mask #mask the padded value with 0,(1 for node, 0 for padding)
            test_rel_l2_two[i] = myloss(out.view(batch_size_,-1), y.view(batch_size_,-1)).item()
        
        np.save('test_rel_l2_two.npy', test_rel_l2_two)
    
        largest_error_ind_two = np.argmax(test_rel_l2_two)
        median_error_ind_two = get_median_index(test_rel_l2_two)  # Get the index (or indices)
        print("largest error is ", test_rel_l2_two[largest_error_ind_two], " ; median error is ", test_rel_l2_two[median_error_ind_two])
        print("largest error index is ", largest_error_ind_two, " ; median error index is ", median_error_ind_two)
        # they are only test data id
        data_ids_two = [largest_error_ind_two, median_error_ind_two]
         
        
    for data_id in data_ids_two:
        
        x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo = (x_test_two[[data_id],...], y_test_two[[data_id],...], aux_test_two[0][[data_id],...], aux_test_two[1][[data_id],...], aux_test_two[2][[data_id],...], aux_test_two[3][[data_id],...], aux_test_two[4][[data_id],...], aux_test_two[5][[data_id],...])
        x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo = x.to(device), y.to(device), node_mask.to(device), nodes.to(device), node_weights.to(device), directed_edges.to(device), edge_gradient_weights.to(device), geo.to(device)

        batch_size_ = x.shape[0]
        out = model(x, (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo)) #.reshape(batch_size_,  -1)
        if normalization_y:
            out = y_normalizer.decode(out)
            y = y_normalizer.decode(y)

        node_mask_bool = node_mask[0,:,0].bool().cpu().numpy()
        y = y.cpu().detach().numpy()[0, node_mask_bool ,0]
        out = out.cpu().detach().numpy()[0, node_mask_bool ,0]
        np.savez("double_"+str(data_id)+".npz",accu=y,pred=out)

def plot_the_curve(list_of_names, figurename=''):
    import numpy as np
    # 4个名字顺序:m1,w1,m2,w1(m:med,w:worst,1:single,2:double)
    
    fig = plt.figure(figsize=(20, 8))
    width_ratios = []
    for i in range(4):
        width_ratios.extend([1, 0.05])  # 图形列 + 颜色条列
    
    gs = fig.add_gridspec(2, 8, width_ratios=width_ratios, 
                         hspace=0.25, wspace=0.6)
    
    all_accu = []
    all_pred = []
    nodes_list = []
    elems_list = []
    two_circles_list = []
    
    for j in range(4):
        two_circles = (j >= 2)
        name = list_of_names[j] + ".npz"
        data = np.load(name)
        nodes = data["nodes"]
        pred = data["pred"]
        accu = data["accu"]

        print(f"Dataset {j}: nodes shape={nodes.shape}, pred shape={pred.shape}, accu shape={accu.shape}")

        N = len(nodes) if not two_circles else len(nodes) // 2
        if two_circles:
            elems1 = np.stack([np.full(N, 1, dtype=int), np.arange(N), 
                              (np.arange(N) + 1) % N], axis=1)
            elems2 = np.stack([np.full(N, 1, dtype=int), np.arange(N, 2*N), 
                              (np.arange(N, 2*N) + 1 - N) % N + N], axis=1)
            elems = np.concatenate([elems1, elems2], axis=0)  # 2N, 3
        else:
            elems = np.stack([np.full(N, 1, dtype=int), np.arange(N), 
                             (np.arange(N) + 1) % N], axis=1)
        
        # 存储数据
        all_accu.append(accu)
        all_pred.append(pred)
        nodes_list.append(nodes)
        elems_list.append(elems)
        two_circles_list.append(two_circles)
    
    # 创建共享的颜色条
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    
    
    figurenames1 = ["Single circle", "Single circle", 
                   "Double circle", "Double circle"]
    figurenames2 = ["(median error)", "(largest error)",
                    "(median error)", "(largest error)"]
    
    for col in range(4):

        accu = all_accu[col]
        pred = all_pred[col]
        nodes = nodes_list[col]
        elems = elems_list[col]
        
        vmin_col = min(accu.min(), pred.min())
        vmax_col = max(accu.max(), pred.max())
        
        # 计算列位置：第col列对应图形列在 2*col，颜色条列在 2*col+1
        graph_col = 2 * col
        cbar_col = 2 * col + 1
        
        # 第一行：真实值 accu
        ax_top = fig.add_subplot(gs[0, graph_col])
        
        scatter_top = ax_top.scatter(nodes[:, 0], nodes[:, 1], c=accu, 
                                    cmap='viridis', s=40, 
                                    vmin=vmin_col, vmax=vmax_col)
        ax_top.set_title("Reference\n" + figurenames2[col], fontsize=18, pad=10)
        
        ax_top.set_aspect('equal', adjustable='box')
        ax_top.set_xlim(-2.5, 2.5)
        ax_top.set_ylim(-2.5, 2.5)
        ax_top.set_xlabel("")
        ax_top.set_xticks([])
        ax_top.xaxis.set_ticks([])
        ax_top.set_ylabel("")
        ax_top.set_yticks([])
        ax_top.yaxis.set_ticks([])

        for elem in elems:
            node_indices = elem[1:]
            valid_indices = node_indices[node_indices != -1]
            if len(valid_indices) > 1:
                elem_nodes = nodes[valid_indices]
                ax_top.plot(elem_nodes[:, 0], elem_nodes[:, 1], color='red', 
                           linewidth=1, alpha=0.7)
        
        # 第二行：预测值 pred
        ax_bottom = fig.add_subplot(gs[1, graph_col])
        
        scatter_bottom = ax_bottom.scatter(nodes[:, 0], nodes[:, 1], c=pred, 
                                          cmap='viridis', s=40, 
                                          vmin=vmin_col, vmax=vmax_col)
        
        ax_bottom.set_title("Prediction\n" + figurenames2[col], fontsize=18, pad=10)
        
        ax_bottom.set_aspect('equal', adjustable='box')
        ax_bottom.set_xlim(-2.5, 2.5)
        ax_bottom.set_ylim(-2.5, 2.5)
        ax_bottom.set_xlabel("")
        ax_bottom.set_ylabel("")
        ax_bottom.set_xticks([])
        ax_bottom.set_yticks([])
        ax_bottom.xaxis.set_ticks([])
        ax_bottom.yaxis.set_ticks([])

        for elem in elems:
            node_indices = elem[1:]
            valid_indices = node_indices[node_indices != -1]
            if len(valid_indices) > 1:
                elem_nodes = nodes[valid_indices]
                ax_bottom.plot(elem_nodes[:, 0], elem_nodes[:, 1], color='red', 
                              linewidth=1, alpha=0.7)
        
        cbar_ax = fig.add_subplot(gs[:, cbar_col])
        
        sm = ScalarMappable(norm=Normalize(vmin=vmin_col, vmax=vmax_col), cmap='viridis')
        sm.set_array([])
        
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
        # 将刻度显示在左侧
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left')
        cbar.ax.tick_params(labelsize=18)

    if figurename:
        plt.savefig(figurename, dpi=300, bbox_inches='tight')
    else:
        plt.show()


if __name__ == "__main__":
    plot_the_curve(["single_140", "single_917", "double_797", "double_493"], "single_layer_potential_comparision.png")