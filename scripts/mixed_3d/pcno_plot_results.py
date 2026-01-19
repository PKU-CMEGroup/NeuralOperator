import meshio
import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
plt.rcParams['font.family'] = 'Times New Roman'

from pcno_geo_mixed_3d_helper import gen_data_tensors


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utility.normalizer import UnitGaussianNormalizer
from utility.losses import LpLoss
from pcno.pcno_geo_head import compute_Fourier_modes, PCNO

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




def plot_results(vertices, elems, vertex_data, elem_data, file_name):
    
    """
    Save mesh data to a VTK file using meshio.
    
    Parameters:
    -----------
    vertices : numpy array (n_vertices, 3)
        Vertex coordinates
    elems : numpy array (n_elements, 4)
        Element connectivity (assuming triangle elements with 3 nodes + 1 element dimension)
    vertex_data : dict
        Data defined at vertices (e.g., {"displacement": vertex_displacements})
    elem_data : dict
        Data defined at elements (e.g., {"stress": element_stresses})
    file_name : str
        Output file name without extension
    """
 
    cells = []
    cells.append(("triangle", elems[:,1:]))

    # Create the mesh
    mesh = meshio.Mesh(
        points=vertices,
        cells=cells,
        point_data=vertex_data,
        cell_data=elem_data,
    )

    # Save to an Exodus II file
    meshio.write(file_name+ ".vtk", mesh)

def plot_raw_data(folder = "../../data/mixed_3d_add_elem_features", category = "Plane", subcategory = "boeing737", data_id = 10):
# visualize raw data
    data_file = folder + "/" + category + "/" + subcategory + "/" + str(data_id).zfill(4) + ".npz"
    data = np.load(data_file)
    vertices = data["nodes_list"]
    elems = data["elems_list"]

    print("number of vertices : ", vertices.shape[0])
    print("number of elems : ", elems.shape[0])

    vertex_data = {"vertex_Cp":  data["features_list"][:,0]}
    elem_data   = {"element_Cp": [data["elem_features_list"][:,0]]}
    file_name = category + "_" + subcategory + "_" + str(data_id).zfill(4)
    plot_results(vertices, elems, vertex_data, elem_data, file_name)


def plot_reduced_data(folder = "../../data/mixed_3d_add_elem_features", mesh_type = "vertex_centered", n_train = 1000, n_test = 100, data_id = 0):
    # # visualize postprocessed data
    data = np.load(folder+"/pcno_mixed_3d_"+mesh_type+"_n_train"+str(n_train)+"_n_test"+str(n_test)+".npz")
    names_array = np.load(folder+"/pcno_mixed_3d_names_list"+"_n_train"+str(n_train)+"_n_test"+str(n_test)+".npy", allow_pickle=True)
    
    
    nnodes, node_mask, nodes, features = data["nnodes"][data_id], data["node_mask"][data_id], data["nodes"][data_id], data["features"][data_id]
    print("nnodes = ", nnodes, " total nnodes = ", node_mask.shape[0], flush = True)

    raw_data_category, raw_data_subcategory, raw_data_id = names_array[data_id].split('-')
    raw_data_id = int(raw_data_id)
    print("Visualize ", raw_data_category, " ", raw_data_subcategory, " ", raw_data_id)
    raw_data_file = folder + "/" + raw_data_category + "/" + raw_data_subcategory + "/" + str(raw_data_id).zfill(4) + ".npz"
    raw_data = np.load(raw_data_file)
    elems = raw_data["elems_list"]
    vertices = raw_data["nodes_list"]

    print("number of vertices : ", vertices.shape[0])
    print("number of elems : ", elems.shape[0])

    if mesh_type == "cell_centered":
        vertex_data = {"vertex_Cp":  raw_data["features_list"][:,0]}
        elem_data   = {"element_Cp": [raw_data["elem_features_list"][:,0]], "post_element_Cp": [features[0:nnodes,3]], "post_element_normal": [features[0:nnodes,0:3]]}
        
    elif mesh_type == "vertex_centered":
        vertex_data = {"vertex_Cp":  raw_data["features_list"][:,0], "post_vertex_Cp": features[0:nnodes,3], "post_vertex_normal": features[0:nnodes,0:3]}
        elem_data   = {"element_Cp": [raw_data["elem_features_list"][:,0]]}
        
    else:
        raise NotImplementedError(
                        f"Unsupported mesh_type={mesh_type}"
                    ) 
        
    file_name = "post_" + mesh_type + "_" + raw_data_category + "_" +  raw_data_subcategory + "_" + str(raw_data_id).zfill(4)
    plot_results(vertices, elems, vertex_data, elem_data, file_name)
    
    


def predict_error(folder = "../../data/mixed_3d_add_elem_features", mesh_type = "vertex_centered", n_train = 1000, n_test = 100, data_ids = None):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    ##############################################
    # load data
    ##############################################
    data = np.load(folder+"/pcno_mixed_3d_"+mesh_type+"_n_train"+str(n_train)+"_n_test"+str(n_test)+".npz")
    names_array = np.load(folder+"/pcno_mixed_3d_names_list"+"_n_train"+str(n_train)+"_n_test"+str(n_test)+".npy", allow_pickle=True)
    
    to_divide_factor = 1.0
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
    assert(ndata == n_train + n_test)
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


    print(f'x_train shape {x_train.shape}, x_test shape {x_train.shape}, y_train shape {y_train.shape}, y_test shape {y_train.shape}', flush = True)
    print('length of each dim: ',torch.amax(nodes, dim = [0,1]) - torch.amin(nodes, dim = [0,1]), flush = True)

    #！！！！！
    Ls = [4.1, 4.1, 1.5]
    k_max = 16
    ndim = 3
    
    modes = compute_Fourier_modes(ndim, [k_max, k_max, k_max], Ls)
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
    
    
    
    normalization_x = False
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

    if data_ids is None:
        test_rel_l2 = np.zeros(n_test)
        myloss = LpLoss(d=1, p=2, size_average=False)
        for i in range(n_test):
            x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo = x_test[[i],...], y_test[[i],...], aux_test[0][[i],...], aux_test[1][[i],...], aux_test[2][[i],...], aux_test[3][[i],...], aux_test[4][[i],...], aux_test[5][[i],...]
            x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo = x.to(device), y.to(device), node_mask.to(device), nodes.to(device), node_weights.to(device), directed_edges.to(device), edge_gradient_weights.to(device), geo.to(device)

            batch_size_ = x.shape[0]
            out = model(x, (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo)) #.reshape(batch_size_,  -1)

            if normalization_y:
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)
            out=out*node_mask #mask the padded value with 0,(1 for node, 0 for padding)
            test_rel_l2[i] = myloss(out.view(batch_size_,-1), y.view(batch_size_,-1)).item()
        
        np.save('test_rel_l2.npy', test_rel_l2)
    
        largest_error_ind = np.argmax(test_rel_l2)
        median_error_ind = get_median_index(test_rel_l2)  # Get the index (or indices)
        print("largest error is ", test_rel_l2[largest_error_ind], " ; median error is ", test_rel_l2[median_error_ind])
        print("largest error index is ", largest_error_ind, " ; median error index is ", median_error_ind)
        # they are only test data id
        data_ids = [largest_error_ind + n_train, median_error_ind + n_train]
        
        
        
        
    for data_id in data_ids:
        
        ################################################
        # load raw data to get elems, vertices
        ################################################
        raw_data_category, raw_data_subcategory, raw_data_id = names_array[data_id].split('-')
        raw_data_id = int(raw_data_id)
        print("Visualize ", raw_data_category, " ", raw_data_subcategory, " ", raw_data_id)
        raw_data_file = folder + "/" + raw_data_category + "/" + raw_data_subcategory + "/" + str(raw_data_id).zfill(4) + ".npz"
        raw_data = np.load(raw_data_file)
        elems = raw_data["elems_list"]
        vertices = raw_data["nodes_list"]
        
        
        if data_id < n_train:
            x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo = (x_train[[data_id],...], y_train[[data_id],...], aux_train[0][[data_id],...], aux_train[1][[data_id],...], aux_train[2][[data_id],...], aux_train[3][[data_id],...], aux_train[4][[data_id],...], aux_train[5][[data_id],...])
        else:
            x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo = (x_test[[data_id - n_train],...], y_test[[data_id - n_train],...], aux_test[0][[data_id - n_train],...], aux_test[1][[data_id - n_train],...], aux_test[2][[data_id - n_train],...], aux_test[3][[data_id - n_train],...], aux_test[4][[data_id - n_train],...], aux_test[5][[data_id - n_train],...])
        
        x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo = x.to(device), y.to(device), node_mask.to(device), nodes.to(device), node_weights.to(device), directed_edges.to(device), edge_gradient_weights.to(device), geo.to(device)

        batch_size_ = x.shape[0]
        out = model(x, (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights, geo)) #.reshape(batch_size_,  -1)
        if normalization_y:
            out = y_normalizer.decode(out)
            y = y_normalizer.decode(y)

       
        
        node_mask_bool = node_mask[0,:,0].bool().numpy()
        y = y.cpu().detach().numpy()[0, node_mask_bool ,0]
        out = out.cpu().detach().numpy()[0, node_mask_bool ,0]
        
        # Nodal values: scalar (e.g., temperature) and vector (e.g., displacement)
        Cp_ref = y  # Scalar values at each node
        Cp_pred = out  # Scalar values at each node
        Cp_error = out - y  # Scalar values at each node
        # Convert nodes to meshio-compatible format
        

        # Convert elements to meshio-compatible format
        cells = []
        cells.append(("triangle", elems[:,1:]))

        # Create the mesh
        mesh = meshio.Mesh(
            points=vertices,
            cells=cells,
            point_data={
                "Cp_ref": Cp_ref,
                "Cp_pred": Cp_pred,
                "Cp_error": Cp_error,
            }
        )
        
        file_name = "predict_" + mesh_type + "_" + raw_data_category + "_" + raw_data_subcategory + "_" + str(raw_data_id).zfill(4)
        meshio.write(file_name+ ".vtk", mesh)
        
        
# predict_error(folder = "../../data/mixed_3d_add_elem_features", mesh_type = "vertex_centered", n_train = 1000, n_test = 100, data_ids = [0,1])
predict_error(folder = "../../data/mixed_3d_add_elem_features", mesh_type = "vertex_centered", n_train = 1000, n_test = 100, data_ids = None)
