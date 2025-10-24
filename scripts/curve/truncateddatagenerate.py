import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from tqdm import tqdm
np.random.seed(0)

def generate_2Dnormal_data(ndata,npoints,freqx,freqf):
    theta = np.linspace(0,2*np.pi,npoints,endpoint=False).reshape((-1,1))
    freq = max(freqx,freqf)
    ThetaTrig = np.zeros((npoints,2*freq))
    ThetaTrigdifferential = np.zeros((npoints,2*freq))
    for i in range(freq):
        ThetaTrig[:,2*i] = np.cos((i+1)*theta).reshape((-1,))
        ThetaTrig[:,2*i+1] = np.sin((i+1)*theta).reshape((-1,))
        ThetaTrigdifferential[:,2*i] = -(i+1)*np.sin((i+1)*theta).reshape((-1,))
        ThetaTrigdifferential[:,2*i+1] = (i+1)*np.cos((i+1)*theta).reshape((-1,))
    points = np.zeros((ndata,npoints,2))
    fin = np.zeros((ndata,npoints,1))
    fout = np.zeros((ndata,npoints,1))
    normal = np.zeros((ndata,npoints,2))
    elems = np.stack((np.arange(npoints), np.roll(np.arange(npoints), -1)), axis=1)
    for i in tqdm(range(ndata)):
        a = 2 * np.random.random((2*freqx,1)) - np.ones((2*freqx,1))
        # a = a/np.linalg.norm(a,ord=1) * np.random.uniform(0.1,0.5)
        r = np.ones((npoints,1)) + 0.5 * np.tanh(ThetaTrig[:,:2*freqx].dot(a))
        r_differential = 0.5 * (ThetaTrigdifferential[:,:2*freqx].dot(a)) * (1 - np.tanh(ThetaTrig[:,:2*freqx].dot(a))**2)
        normal[i,:,:] = np.concatenate((r * np.cos(theta) + r_differential * np.sin(theta), r * np.sin(theta) - r_differential * np.cos(theta)), axis=1)
        normal[i,:,:] = normal[i,:,:]/np.linalg.norm(normal[i,:,:], axis=1, keepdims=True)
        points[i,:,:] = np.stack((r * np.cos(theta), r * np.sin(theta)), axis=1).reshape((npoints,2))
        b = 2 * np.random.random((2*freqf,1)) - np.ones((2*freqf,1))
        fin[i,:,:] = np.random.random()*np.ones((npoints,1)) + 0.5 * np.tanh(ThetaTrig[:,:2*freqf].dot(b))
        fout[i,:,:] = integral_operator(points[i,:,:], elems, nebla_truncated_log_normal_kernel, fin[i,:,:], points[i,:,:], num_quad_points=3).reshape((npoints,1))
    return points,fin,fout,normal

def integral_operator(ypoints, elems, kernel, fin, xpoints, num_quad_points=3):
    """
    Compute fout(x) = ∫ fin(y) * kernel(x , y) dy for each node x in 'xpoints'
    
    Parameters:
    - ypoints: ndarray of shape (my,) : coordinates of nodes on the boundary 
    - elems: ndarray of shape (melems, 2) : indices of nodes forming elements, the outward is on the right
    - kernel: function with vectorization R^dx * R^dy -> R^{dx \times dy} : kernel(x , y)
    - fin: array of shape (my) : input function values on each node y
    - xpoints: ndarray of shape (mx,) : coordinates of nodes for computing fout
    - num_quad_points: int : number of Gaussian quadrature points
    
    Returns:
    - fout: ndarray of shape (mx,) : result of integral at each node x
    """
    my, mx = len(ypoints), len(xpoints)
    fout = np.zeros(mx)
    
    # Quadrature points and weights on reference element [-1, 1]
    quad_points, quad_weights = np.polynomial.legendre.leggauss(num_quad_points)
        
    
    # Loop over all elements
    for elem in elems:
        p0, p1 = ypoints[elem[0]], ypoints[elem[1]]
        J = (p1 - p0) / 2  # Jacobian of the affine map
        normal = np.array([J[1], -J[0]]) / np.linalg.norm(J)  # outward unit normal, pointing to the right 
        f0, f1 = fin[elem[0]], fin[elem[1]]

        # compute the contribution from this element to all points
        y = (p0 + p1)/2 + np.outer(quad_points, J)
        f = (f0 + f1)/2 + (f1 - f0) / 2 * quad_points
        K = kernel(xpoints, y, normal)
        fout += K.dot( (quad_weights * f) * np.linalg.norm(J) )
    
    return fout

def plot_2D_data(points):
    plt.figure()
    plt.plot(np.concatenate([points[:,0],[points[0,0]]]), np.concatenate([points[:,1],[points[0,1]]]), '-o', markersize=0.1)
    plt.axis('equal')
    plt.show()

def plot_color_coded_curve(x, y, z, cmap='viridis', linewidth=3, figsize=(10, 6)):
    """
    现代化版本的曲线颜色绘图
    参数:
        x, y: 曲线坐标
        z: 用于着色的数值
        cmap: 颜色映射名称
        linewidth: 线宽
    """
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    x = np.concatenate([x,[x[0]]])
    y = np.concatenate([y,[y[0]]])
    z = np.concatenate([z,[z[0]]])
    # 创建线段
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # 创建LineCollection（使用新的colormaps API）
    lc = LineCollection(segments, cmap=plt.colormaps[cmap], linewidth=linewidth)
    lc.set_array(z)  # 这里使用参数z而不是values
    
    # 添加到图形
    ax.add_collection(lc)
    
    # 自动调整坐标轴范围
    ax.autoscale()
    
    # 添加颜色条
    cbar = fig.colorbar(lc, ax=ax)
    cbar.set_label('Value')
    
    # 标签和标题
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_title(f'Color-coded Curve (cmap: {cmap})')
    ax.grid(True, alpha=0.3)
    
    # 保持纵横比
    ax.set_aspect('equal', adjustable='datalim')
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax


def log_kernel(x, y, n): # 这里的n其实是法向量，只是在这个核函数中没用到
    # ln(||x - y||)
    diff = x[:, np.newaxis, :] - y[np.newaxis, :, :]  # shape (mx, my, 2)
    distance_matrix = np.linalg.norm(diff, axis=2)    # shape (mx, my)

    return np.log(distance_matrix)
    
def nebla_log_normal_kernel(x, y, n):
    # (x - y) · n / ||y - x||^2
    diff = - x[:, np.newaxis, :] + y[np.newaxis, :, :]  # shape (mx, my, 2)
    distance_matrix = np.linalg.norm(diff, axis=2)    # shape (mx, my)
    dot_product = np.einsum('ijk,k->ij', diff, n)      # shape (mx, my)

    return dot_product / (distance_matrix**2)  

def nebla_truncated_log_normal_kernel(x, y, n):
    # (y-x) · n / max(||y - x||^2, r^2)
    diff = - x[:, np.newaxis, :] + y[np.newaxis, :, :]  # shape (mx, my, 2)
    distance_matrix = np.linalg.norm(diff, axis=2)    # shape (mx, my)
    dot_product = np.einsum('ijk,k->ij', diff, n)      # shape (mx, my)
    r = 0.3  # 截断距离
    # 计算分母，当距离小于r时使用r^2，否则使用实际距离的平方
    denominator = np.maximum(distance_matrix**2, r**2)
    
    return dot_product / denominator

points, fin, fout, normal = generate_2Dnormal_data(10000,1000,3,3)

m = points[0].shape[0]
elem = np.stack((np.ones(m,int),np.arange(m), np.roll(np.arange(m), -1)), axis=1)

elems = [elem] * len(points)

features = [np.concatenate((fin[i], fout[i]), axis=-1) for i in range(len(fin))]

np.savez('data/2D_nebla0.3truncatedlog_data_3_3_10000.npz', nodes_list = points, elems_list = elems, features_list = features, normal_list = normal) 