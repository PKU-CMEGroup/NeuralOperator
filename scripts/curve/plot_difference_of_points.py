from matplotlib import pyplot as plt
import numpy as np
import tqdm

def plot_batch(points_set, name, save_path=None):
    plt.figure(figsize=(10, 8))

    for ind in tqdm.tqdm(range(len(points_set))):
        points = points_set[ind]
        n = len(points)
        
        # 每隔10个点采样
        sample_indices = np.arange(0, n, 20)
        sampled_points = points[sample_indices]
        
        # 向量化计算所有差值（会占用大量内存）
        # 这里使用广播计算：sampled_points[:, np.newaxis] - points
        diffs = sampled_points[:, np.newaxis, :] - points[np.newaxis, :, :]
        diffs = diffs.reshape(-1, 2)

        plt.scatter(diffs[:, 0], diffs[:, 1], s=1, alpha=0.1)
    
    plt.title('Difference Vectors of '+name)
    plt.xlabel('X difference')
    plt.ylabel('Y difference')
    plt.grid(True, alpha=0.3)

    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

if __name__ == "__main__":
    
    name = "modified_dp_laplace_panel"
    data_name = "pcno_curve_data_1_1_5_2d_" + name
    data = np.load("../../data/curve/"+data_name+".npz")
    points_set = data["nodes"]

    for id in range(10):
        plot_batch(points_set[id*1000:(id+1)*1000], name, save_path=name+"_"+str(id)+".png")