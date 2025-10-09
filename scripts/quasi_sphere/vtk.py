import os
import numpy as np
import pyvista as pv

def save_sourcedata_as_vtk(l, base_path, index, output_dir):
    idx_str = f"{index:05d}"
    elems_path = os.path.join(base_path, "elems", f"elems_{idx_str}.npy")
    nodes_path = os.path.join(base_path, "nodes", f"nodes_{idx_str}.npy")

    elems = np.load(elems_path)      # (num_elems, elem_size)
    nodes = np.load(nodes_path)      # (num_nodes, 3)

    elem_size = elems.shape[1]
    if elem_size == 3:
        cell_type = pv.CellType.TRIANGLE
    elif elem_size == 4:
        cell_type = pv.CellType.QUAD
    else:
        raise ValueError("Unsupported element size")

    cells = np.hstack([
        np.column_stack([np.full(len(elems), elem_size), elems]).flatten()
    ])


    mesh = pv.UnstructuredGrid(cells, np.full(len(elems), cell_type), nodes)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"instance_l{l}_{idx_str}.vtk")
    mesh.save(out_path)
    print(f"Saved VTK file to {out_path}")

def save_freqnorm_npz_as_vtk(npz_path, output_dir, output_name):
    data = np.load(npz_path, allow_pickle=True)
    freq_c_norm_list = data['freq_c_norm_list']
    freq_s_norm_list = data['freq_s_norm_list']
    modes = data['modes'].squeeze()  # shape: (num_points, n_dim)
    if modes.shape[1] == 2:
        z = np.zeros((modes.shape[0], 1))
        modes = np.hstack([modes, z])  # shape: (num_points, 3)


    features = np.concatenate([np.array(freq_c_norm_list), np.array(freq_s_norm_list)], axis=0)  # shape: (num_points, num_features)
    print(features.shape)

    point_cloud = pv.PolyData(modes)
    for i in range(features.shape[0]):
        freq_type = 'c' if i < len(freq_c_norm_list) else 's'
        point_cloud.point_data[f'layer_{i%len(freq_c_norm_list)}{freq_type}'] = features[i, :]

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, output_name)
    point_cloud.save(out_path)
    print(f"Saved VTK file to {out_path}")


if __name__ == "__main__":
    # l = 3
    # base_path = f'../../data/quasi_sphere/npys/NPYSmax_l{l}'
    # index = 1
    # output_dir = os.path.join(os.path.dirname(__file__), "vtks")
    # save_sourcedata_as_vtk(l, base_path, index, output_dir)


    npz_name = 'curve_k16_gathered'
    output_dir = os.path.join(os.path.dirname(__file__), "vtks")
    save_freqnorm_npz_as_vtk('npzs/' + npz_name + '.npz', output_dir, npz_name + '.vtk')
    