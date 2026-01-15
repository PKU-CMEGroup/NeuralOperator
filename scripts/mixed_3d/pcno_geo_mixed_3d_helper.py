import os
import numpy as np


Plane_datasets = [
    "boeing737",
    "erj",
    "J20",
    "P180"
]
DrivAerNet_datasets = [
    "E_S_WW_WM",
    "E_S_WWC_WM",
    "F_D_WM_WW",
    "F_S_WWC_WM",
    "F_S_WWS_WM",
    "N_S_WW_WM",
    "N_S_WWC_WM",
    "N_S_WWS_WM"
]


def _load_data(file_path, nodes_list, elems_list, elem_features_list):
    data = np.load(file_path)
    nodes_list.append(data["nodes_list"])
    elems = data["elems_list"]
    elems[:, 0] = 2 # element dim = 2
    elems_list.append(elems)
    elem_features_list.append(data["elem_features_list"])


def load_data(data_path, Plane_datasets, DrivAerNet_datasets, n_each):
    names_list, nodes_list, elems_list, elem_features_list = [], [], [], []

    Plane_dir = os.path.join(data_path, "Plane")
    DrivAerNet_dir = os.path.join(data_path, "DrivAerNet")

    # load data
    
    for subdir in Plane_datasets:
        for i in range(n_each):
            file_path = os.path.join(Plane_dir, subdir, "%04d"%(i+1)+".npz")
            if os.path.exists(file_path):
                _load_data(file_path, nodes_list, elems_list, elem_features_list)
                names_list.append("Plane-" + subdir + "-%04d"%(i+1))
            else:
                print("Warning: ignore ", file_path)
                
    for subdir in DrivAerNet_datasets:
        for i in range(n_each):
            file_path = os.path.join(DrivAerNet_dir, subdir, "%04d"%(i+1)+".npz")
            if os.path.exists(file_path):
                _load_data(file_path, nodes_list, elems_list, elem_features_list)
                names_list.append("DrivAerNet-" + subdir + "-%04d"%(i+1))
            else:
                print("Warning: ignore ", file_path)


    return nodes_list, elems_list, elem_features_list, names_list

def random_shuffle(data, names_array, n_train, n_test, seed=42):
    np.random.seed(seed)  # 可选的：为了可重复性设置随机种子
    
    ndata = data["nodes"].shape[0]
    assert(ndata >= n_train + n_test)
    print("Total data number =", ndata, " n_train = ", n_train, " n_test = ", n_test)
    random_indices = np.arange(ndata)
    np.random.shuffle(random_indices)
    
    # 取前n_train 和后n_test 个分别作为训练和测试集
    train_indices = random_indices[:n_train]
    test_indices = random_indices[-n_test:]
    indices = np.concatenate([train_indices, test_indices])
    
    
    data = {key: value[indices] for key, value in data.items()}
    names_array = names_array[indices]
    
    # 输出数据统计情况
    all_datasets = Plane_datasets + DrivAerNet_datasets
    train_data_stats = {subdir: 0 for subdir in all_datasets}
    test_data_stats = {subdir: 0 for subdir in all_datasets}
    for i in range(n_train):
        train_data_stats[names_array[i].split('-')[1]] += 1
    for i in range(-n_test,0):
        test_data_stats[names_array[i].split('-')[1]] += 1
    print("Training data statistics:")
    print("-" * 40)
    assert(sum(train_data_stats.values()) == n_train)
    for dataset in sorted(all_datasets):
        count = train_data_stats[dataset]
        if count > 0:
            percentage = count / n_train * 100
            print(f"  {dataset:15s}: {count:3d} ({percentage:5.1f}%)")
    
    print("Test data statistics:")
    print("-" * 40)
    assert(sum(test_data_stats.values()) == n_test)
    for dataset in sorted(all_datasets):
        count = test_data_stats[dataset]
        if count > 0:
            percentage = count / n_test * 100
            print(f"  {dataset:15s}: {count:3d} ({percentage:5.1f}%)")
    
    return data, names_array

