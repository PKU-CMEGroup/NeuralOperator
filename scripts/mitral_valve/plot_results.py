import torch
import os
import sys
import numpy as np
import vtk
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



def add_array(name, data, num_components=3):
    
    array = vtk.vtkFloatArray()
    array.SetName(name)
    array.SetNumberOfComponents(num_components)
    for row in data:
        array.InsertNextTuple(row)
    return array 
        


def write_polydata_vtk(points, elems):
    # Create VTK points
    vtk_points = vtk.vtkPoints()
    for pt in points:
        vtk_points.InsertNextPoint(pt)

    # Create vtkPolyData
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)

    elems_num = elems.shape[0]

    lines = vtk.vtkCellArray()
    polygons = vtk.vtkCellArray()
    for j in range(elems_num):
        cell_dim = elems[j,0]
        cell_index = elems[j,1:]
        cell_nodes_sum = len(cell_index)
        if cell_dim == 1:
            #line
            line = vtk.vtkLine()
            line.GetPointIds().SetNumberOfIds(2)
            line.GetPointIds().SetId(0, cell_index[1])
            line.GetPointIds().SetId(1, cell_index[2])
            lines.InsertNextCell(line)

        if cell_dim == 2:
            #triangle
            polygon = vtk.vtkPolygon()
            polygon.GetPointIds().SetNumberOfIds(cell_nodes_sum)
            for i in range(cell_nodes_sum):
                polygon.GetPointIds().SetId(i, cell_index[i])
            polygons.InsertNextCell(polygon)

        polydata.SetPolys(polygons)

    return polydata

# visualize the 
def write_vtk(data_path, i, pred=None, filename_pref=""): 

    myloss = LpLoss(d=1, p=2, size_average=False)
    ##load displacement data 
    displacement = np.load(data_path+"/displacements_%05d"%(1211-199+i)+".npy")
    strain = np.load(data_path+"/lagrange_strain_%05d"%(1211-199+i)+".npy")[:,[0,1,2,4,5,8]]  # xx xy xz ; yx yy yz ; zx zy zz
    stress = np.load(data_path+"/cauchy_stress_%05d"%(1211-199+i)+".npy")[:,[0,1,2,4,5,8]]

    deformed_points = np.load(data_path+"/coordinates_%05d"%(1211-199+i)+".npy") 
    points = deformed_points - displacement
    points_num = points.shape[0]

    ##vtk needs element data
    elem_dim = 2
    elems = np.load(data_path+"/quad_connectivity_%05d"%(1211-199+i)+".npy")
    elems = np.concatenate((np.full((elems.shape[0], 1), elem_dim, dtype=int), elems), axis=1)     
    elems_num = elems.shape[0]


    polydata = write_polydata_vtk(points, elems)
    displacement_array = add_array("Displacement", displacement, num_components=3)
    strain_array = add_array("Strain", strain, num_components=6)
    stress_array = add_array("Stress", stress, num_components=6)


    polydata.GetPointData().AddArray(displacement_array)
    polydata.GetPointData().AddArray(strain_array)
    polydata.GetPointData().AddArray(stress_array)

    if pred is not None:
        pred_displacement_array = add_array("Displacement(Pred)", pred[:,0:3], num_components=3)
        pred_stress_array = add_array("Stress(Pred)", pred[:,3:9], num_components=6)

        rel_l2_displacement = myloss(torch.from_numpy(pred[:,0:3]).reshape(1,-1), torch.from_numpy(displacement).reshape(1,-1)).item()
        rel_l2_Stress = myloss(torch.from_numpy(pred[:,3:9]).reshape(1,-1), torch.from_numpy(stress).reshape(1,-1)).item()

        print("relative test error of displacement ", i, " = ", rel_l2_displacement)
        print("relative test error of stress ", i, " = ", rel_l2_Stress)

        polydata.GetPointData().AddArray(pred_displacement_array)
        polydata.GetPointData().AddArray(pred_stress_array)



    ##Generate the initial state.
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(filename_pref+"_%05d"%(i+1)+".vtk")
    writer.SetInputData(polydata)
    writer.SetFileTypeToBinary()
    writer.Write()




def predict_error(data_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ###################################
    # load data
    ###################################
    
    equal_weights = False
    data = np.load(data_path+"mitral_valve_single_geometry_data.npz")
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
    
    #stress and pressure have the same unit, range O(1)
    #strain does not have an unit, range O(0.01)
    #displacement and nodes have the same unit, range O(1)
    L_ref, p_ref, c_ref, strain_ref = 1.0, 1.0, 100, 0.01
    nodes_input = nodes.clone()/L_ref  # scale length input
    #pressre, c, displacements, stress, strain
    features = features / np.array([p_ref] * 1 + [c_ref] * 3 + [L_ref] * 3 + [p_ref] * 6 + [strain_ref] * 6 )
    features = torch.from_numpy(features.astype(np.float32))

    directed_edges = torch.from_numpy(directed_edges)
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

    train_inv_L_scale = 'together'

    model = PCNO(ndim, modes, nmeasures=1,
                layers=[128,128,128,128,128],
                fc_dim=128,
                in_dim=x_train.shape[-1], out_dim=y_train.shape[-1], 
                train_inv_L_scale = train_inv_L_scale,
                act='gelu').to(device)
    model.load_state_dict(torch.load("PCNO_mitral_valve_single_geometry_model.pth", weights_only=True, map_location=torch.device(device)))
    model = model.to(device)



    normalization_x = False
    normalization_y = True
    normalization_dim_x = []
    normalization_dim_y = []
    non_normalized_dim_x = 0
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
        x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights = x_test[[i],...].clone(), y_test[[i],...].clone(), aux_test[0][[i],...].clone(),\
        aux_test[1][[i],...].clone(), aux_test[2][[i],...].clone(), aux_test[3][[i],...].clone(), aux_test[4][[i],...].clone()
        x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights = x.to(device), y.to(device), node_mask.to(device), nodes.to(device), node_weights.to(device), directed_edges.to(device), edge_gradient_weights.to(device)

        batch_size_ = x.shape[0]
        out = model(x, (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights)) #.reshape(batch_size_,  -1)

        if normalization_y:
            out = y_normalizer.decode(out)
            y = y_normalizer.decode(y)
        out=out*node_mask #mask the padded value with 0,(1 for node, 0 for padding)
        test_rel_l2[i] = myloss(out.view(batch_size_,-1), y.view(batch_size_,-1)).item()
        print("relative test error ", i, " = ", test_rel_l2[i])
    
    np.save('test_rel_l2.npy', test_rel_l2)
    
    ############################################################################
    

    largest_error_ind = np.argmax(test_rel_l2)
    median_error_ind = get_median_index(test_rel_l2)  # Get the index (or indices)
    print("largest error is ", test_rel_l2[largest_error_ind], " ; median error is ", test_rel_l2[median_error_ind])
    print("largest error index is ", largest_error_ind, " ; median error index is ", median_error_ind)
    
    for i in [largest_error_ind, median_error_ind]:
        
        x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights = x_test[[i],...].clone(), y_test[[i],...].clone(), aux_test[0][[i],...].clone(),\
        aux_test[1][[i],...].clone(), aux_test[2][[i],...].clone(), aux_test[3][[i],...].clone(), aux_test[4][[i],...].clone()
        x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights = x.to(device), y.to(device), node_mask.to(device), nodes.to(device), node_weights.to(device), directed_edges.to(device), edge_gradient_weights.to(device)

        batch_size_ = x.shape[0]
        out = model(x, (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights)) #.reshape(batch_size_,  -1)
        print(out.shape)
        if normalization_y:
            out = y_normalizer.decode(out)
            y = y_normalizer.decode(y)

        out=out*node_mask #mask the padded value with 0,(1 for node, 0 for padding)
        rel_l2 = myloss(out.view(batch_size_,-1), y.view(batch_size_,-1)).item()
        print("relative test error ", i, " = ", rel_l2)

        out = out.cpu().detach().numpy()[0,:,:]

        write_vtk(data_path, i, out, filename_pref="mitral_valve")






def error_plot():
    data = np.genfromtxt("PCNO_mitral_valve_test.log", skip_header = 7)
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
    plt.savefig("loss.pdf")






    ############################################################################
    test_rel_l2 = np.load('test_rel_l2.npy')
    fig, ax = plt.subplots(figsize=(6,6))
    ax.hist(test_rel_l2, bins=len(test_rel_l2)//4, density=True, alpha=0.7, color='C0', edgecolor='black')
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
    plt.savefig("loss_distribution.pdf")


data_path = "../../data/mitral_valve/single_geometry/"
predict_error(data_path)
error_plot()







