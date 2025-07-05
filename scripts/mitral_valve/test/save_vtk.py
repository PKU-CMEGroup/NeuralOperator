import numpy as np
import pyvista as pv

# 加载NPZ文件
def load_data(data_path):
    ndata, elem_dim = 1211, 2
    nodes_list, elems_list, features_list, deformed_nodes_list = [], [], [],[]
    all_simulation_parameters = np.load(data_path+"/all_simulation_parameters.npy") #id, pressure, c0, c1, c2
    for i in range(ndata):      
        
        deformed_nodes = np.load(data_path+"/coordinates_%05d"%(i+1)+".npy")  # This is the deformed shape
        nnodes = deformed_nodes.shape[0]
        displacement = np.load(data_path+"/displacements_%05d"%(i+1)+".npy")
        nodes_list.append(deformed_nodes - displacement)
        deformed_nodes_list.append(deformed_nodes)

        elems = np.load(data_path+"/quad_connectivity_%05d"%(i+1)+".npy")
        elems_list.append(np.concatenate((np.full((elems.shape[0], 1), elem_dim, dtype=int), elems), axis=1))
        

        displacement = np.load(data_path+"/displacements_%05d"%(i+1)+".npy")
        strain = np.load(data_path+"/lagrange_strain_%05d"%(i+1)+".npy")[:,[0,1,2,4,5,8]]  # xx xy xz ; yx yy yz ; zx zy zz
        stress = np.load(data_path+"/cauchy_stress_%05d"%(i+1)+".npy")[:,[0,1,2,4,5,8]]
        
        simulation_parameters = np.tile(all_simulation_parameters[i,1:], (nnodes, 1))
        features_list.append(np.hstack((simulation_parameters, displacement, stress, strain)))

    return nodes_list, elems_list, features_list, deformed_nodes_list
data_path = "../../../data/mitral_valve/single_geometry/"

i = 187
nodes_list, elems_list, features_list , deformed_nodes_list  = load_data(data_path = data_path)
points = nodes_list[i]  # 顶点坐标 (N,3)
faces = elems_list[i][:,1:]    # 面片索引 (M,K)，例如三角形为(M,3)
features = features_list[i]

print(points.shape)
print(faces.shape)
# 确保面片格式符合PyVista要求（每个面片前加顶点数）
if faces.shape[1] == 3:  # 三角形
    faces_pv = np.insert(faces, 0, 3, axis=1).ravel()
elif faces.shape[1] == 4:  # 四边形
    faces_pv = np.insert(faces, 0, 4, axis=1).ravel()
else:
    raise ValueError("Unsupported face type")

# 创建PyVista网格对象
mesh = pv.PolyData(points, faces_pv)


sim_params = features[:, 0:4]  # pressure, c0, c1, c2
displacement = features[:, 4:7]  # x,y,z位移
stress = features[:, 7:13]     # 应力分量
strain = features[:, 13:19]    # 应变分量

# 将数据附加到网格点属性（自动保存在VTK中）
mesh.point_data["displacement"] = displacement  # 作为矢量场
mesh.point_data["stress"] = stress               # 作为6分量张量
mesh.point_data["strain"] = strain               # 作为6分量张量
mesh.point_data["pressure"] = sim_params[:, 0]   # 标量场
mesh.point_data["c0"] = sim_params[:, 1]         # 材料参数
mesh.point_data["c1"] = sim_params[:, 2]
mesh.point_data["c2"] = sim_params[:, 3]

# 保存为ParaView支持的格式（推荐VTK或PLY）
mesh.save(f"data{i}.vtk")  