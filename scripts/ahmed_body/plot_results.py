import meshio
import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
plt.rcParams['font.family'] = 'Times New Roman'


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utility.normalizer import UnitGaussianNormalizer
from utility.losses import LpLoss
from pcno.pcno import compute_Fourier_modes, PCNO
FONTSIZE = 17
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







def predict_error():

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    ###################################
    # load data
    ###################################
    data_path = "../../data/ahmed_body/"

    equal_weights = False
    data = np.load(data_path+"pcno_triangle_data.npz")
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
    features /= np.array([1.0, 100.0, 100.0, 100.0, 100.0, 180.0/np.pi, 100.0, 100.0, 1E6])
    # keep only the pressure and Reynolds number
    features = torch.from_numpy(features.astype(np.float32))

    directed_edges = torch.from_numpy(directed_edges)
    edge_gradient_weights = torch.from_numpy(
        edge_gradient_weights.astype(np.float32))

    # This is important
    nodes_input = nodes.clone()

    data_in, data_out = torch.cat(
        [features[..., 1:], nodes_input, node_rhos], dim=-1), features[..., :1]
    print(f"data in:{data_in.shape}, data out:{data_out.shape}")
    n_train, n_test = 500, 51


    x_train, x_test = data_in[:n_train, ...], data_in[-n_test:, ...]
    aux_test = (node_mask[-n_test:, ...], nodes[-n_test:, ...], node_weights[-n_test:, ...],
                directed_edges[-n_test:, ...], edge_gradient_weights[-n_test:, ...])
    y_train, y_test = data_out[:n_train, ...], data_out[-n_test:, ...]

    print(f"x train:{x_train.shape}, y train:{y_train.shape}", flush=True)



    k_max = 8
    ndim = 3
    Lx = 0.0004795 - (-1.34399998)
    Ly = 0.25450477 - 0
    Lz = 0.43050185 - 0
    modes = compute_Fourier_modes(ndim, [k_max,k_max,k_max], [Lx, Ly, Lz])
    modes = torch.tensor(modes, dtype=torch.float).to(device)
    model = PCNO(ndim, modes, nmeasures=1,
                layers=[128,128,128,128,128],
                fc_dim=128,
                in_dim=x_train.shape[-1], out_dim=y_train.shape[-1],
                train_sp_L="together",
                act='gelu')
    model.load_state_dict(torch.load("PCNO_ahmedbody_model.pth", weights_only=True))
    model = model.to(device)



    normalization_x = True
    normalization_y = True
    normalization_dim_x = []
    normalization_dim_y = []
    non_normalized_dim_x = 4
    non_normalized_dim_y = 0

    if normalization_x:
        x_normalizer = UnitGaussianNormalizer(x_train, non_normalized_dim = non_normalized_dim_x, normalization_dim=normalization_dim_x)
        x_train = x_normalizer.encode(x_train)
        x_test = x_normalizer.encode(x_test)
        x_normalizer.to(device)
        
    if normalization_y:
        y_normalizer = UnitGaussianNormalizer(y_train, non_normalized_dim = non_normalized_dim_y, normalization_dim=normalization_dim_y)
        y_train = y_normalizer.encode(y_train)
        y_test = y_normalizer.encode(y_test)
        y_normalizer.to(device)


    test_rel_l2 = np.zeros(n_test)
    myloss = LpLoss(d=1, p=2, size_average=False)
    for i in range(n_test):
        x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights = x_test[[i],...], y_test[[i],...], aux_test[0][[i],...], aux_test[1][[i],...], aux_test[2][[i],...], aux_test[3][[i],...], aux_test[4][[i],...]
        x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights = x.to(device), y.to(device), node_mask.to(device), nodes.to(device), node_weights.to(device), directed_edges.to(device), edge_gradient_weights.to(device)

        batch_size_ = x.shape[0]
        out = model(x, (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights)) #.reshape(batch_size_,  -1)

        if normalization_y:
            out = y_normalizer.decode(out)
            y = y_normalizer.decode(y)
        out=out*node_mask #mask the padded value with 0,(1 for node, 0 for padding)
        test_rel_l2[i] = myloss(out.view(batch_size_,-1), y.view(batch_size_,-1)).item()
    
    np.save('test_rel_l2.npy', test_rel_l2)
    ############################################################################
    

    largest_error_ind = np.argmax(test_rel_l2)
    median_error_ind = get_median_index(test_rel_l2)  # Get the index (or indices)
    print("largest error is ", test_rel_l2[largest_error_ind], " ; median error is ", test_rel_l2[median_error_ind])
    print("largest error index is ", largest_error_ind, " ; median error index is ", median_error_ind)
    
    for i in [largest_error_ind, median_error_ind]:
        
        elems = np.load(data_path+"/elems_%05d"%(i + n_train)+".npy")
        x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights = x_test[[i],...], y_test[[i],...], aux_test[0][[i],...], aux_test[1][[i],...], aux_test[2][[i],...], aux_test[3][[i],...], aux_test[4][[i],...]
        x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights = x.to(device), y.to(device), node_mask.to(device), nodes.to(device), node_weights.to(device), directed_edges.to(device), edge_gradient_weights.to(device)

        batch_size_ = x.shape[0]
        out = model(x, (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights)) #.reshape(batch_size_,  -1)
        if normalization_y:
            out = y_normalizer.decode(out)
            y = y_normalizer.decode(y)

       
        

        y = y.cpu().detach().numpy()[0,:,0]
        out = out.cpu().detach().numpy()[0,:,0]
        points = nodes.cpu().detach().numpy()[0,...]  # Extract (x, y, z) coordinates

        # Nodal values: scalar (e.g., temperature) and vector (e.g., displacement)
        pressure_ref = y  # Scalar values at each node
        pressure_pred = out  # Scalar values at each node
        pressure_error = out - y  # Scalar values at each node
        # Convert nodes to meshio-compatible format
        

        # Convert elements to meshio-compatible format
        cells = []
        cells.append(("triangle", elems[:,1:]))

        # Create the mesh
        mesh = meshio.Mesh(
            points=points,
            cells=cells,
            point_data={
                "pressureref": pressure_ref,
                "pressurepred": pressure_pred,
                "pressureerror": pressure_error,
            }
        )

        # Save to an Exodus II file
        meshio.write("ahmedbody_"+str(i)+".vtk", mesh)





def error_plot():
    data = np.genfromtxt("PCNO_ahmedbody.log", skip_header = 6)
    iterations, training_error, test_error = data[:,2] , data[:, 10], data[:, 16]
    # 创建图形和轴对象
    fig, ax = plt.subplots(figsize=(6,6))

    # 绘制曲线
    ax.plot(iterations, training_error, label='Train')
    ax.plot(iterations, test_error, label='Test')
    ax.grid("on")
    # 设置标题和坐标轴标签
    # ax.set_title('Sine and Cosine Curves', fontsize=16)  # 标题字体大小
    ax.set_xlabel('Epochs', fontsize=FONTSIZE)          # x 轴标签字体大小
    ax.set_ylabel('Relative $L_2$ Error', fontsize=FONTSIZE)          # y 轴标签字体大小

    # 定义 0 到 0.1 之间的额外刻度位置
    extra_ticks = np.arange(0, 0.1, 0.025)  # 从 0 到 0.1，步长为 0.01

    # 获取默认的 y 轴刻度位置，并将新的刻度位置添加到它们之中
    default_ticks = ax.get_yticks()
    all_ticks = np.sort(np.unique(np.concatenate((default_ticks, extra_ticks))))

    # 设置 y 轴刻度位置
    ax.yaxis.set_major_locator(FixedLocator(all_ticks))

    # 可选：设置 y 轴刻度标签（如果不设置，将会自动使用刻度位置作为标签）
    ax.yaxis.set_major_formatter(FixedFormatter([f'{tick:.2f}' for tick in all_ticks]))


    # 调整刻度标签字体大小
    for tick in ax.get_xticklabels():
        tick.set_fontsize(FONTSIZE)  # x 轴刻度标签字体大小
    for tick in ax.get_yticklabels():
        tick.set_fontsize(FONTSIZE)  # y 轴刻度标签字体大小
        
    # 添加图例并调整其字体大小
    legend = ax.legend(prop={'size': FONTSIZE})  # 图例字体大小

    # 显示图形
    plt.tight_layout()
    plt.savefig("ahmedbody_loss.pdf")



    ############################################################################
    test_rel_l2 = np.load('test_rel_l2.npy')
    fig, ax = plt.subplots(figsize=(6,6))
    ax.hist(test_rel_l2, bins=len(test_rel_l2)//2, density=True, alpha=0.7, color='C0', edgecolor='black')
    ax.set_xlabel('Relative $L_2$ Error', fontsize=FONTSIZE)    
    # ax.set_ylabel('Density', fontsize=FONTSIZE)          # y 轴标签字体大小
    # 调整刻度标签字体大小
    for tick in ax.get_xticklabels():
        tick.set_fontsize(FONTSIZE)  # x 轴刻度标签字体大小
    for tick in ax.get_yticklabels():
        tick.set_fontsize(FONTSIZE)  # y 轴刻度标签字体大小

    # 添加图例并调整其字体大小
    # legend = ax.legend(fontsize=FONTSIZE)  # 图例字体大小
    # 显示图形
    plt.tight_layout()
    plt.savefig("ahmedbody_loss_distribution.pdf")




# predict_error()
error_plot()
