import numpy as np
from tqdm import tqdm


def count_self_intersections(curve_nodes):
    """
    使用向量化加速自相交检测
    curve_nodes: (N, 2)
    """
    N = curve_nodes.shape[0]
    if N < 4: return 0 # 三角形不可能自相交
    
    # 1. 准备线段数据
    # P1: (N, 2), P2: (N, 2)
    # segment i connects P1[i] and P2[i]
    P1 = curve_nodes
    P2 = np.roll(curve_nodes, -1, axis=0)
    
    # 2. 构建 AABB (N, 4) -> (xmin, ymin, xmax, ymax)
    # min/max along the segment
    seg_min = np.minimum(P1, P2)
    seg_max = np.maximum(P1, P2)
    
    # 3. 利用广播比较所有线段对的 AABB
    # 我们需要比较 segment i 和 segment j
    # 构造 (N, 1, 4) 和 (1, N, 4) 进行广播
    # intersect_x: (N, N) boolean
    # max(xmin_i, xmin_j) < min(xmax_i, xmax_j)
    
    # 为了节省内存，可以分步做
    # AABB overlap condition:
    # (seg_min_i.x < seg_max_j.x) & (seg_min_j.x < seg_max_i.x) & ...
    
    # Expand dims
    s_min_i = seg_min[:, np.newaxis, :] # (N, 1, 2)
    s_max_i = seg_max[:, np.newaxis, :]
    s_min_j = seg_min[np.newaxis, :, :] # (1, N, 2)
    s_max_j = seg_max[np.newaxis, :, :]
    
    # AABB Overlap Mask (N, N)
    # overlap_x = np.maximum(s_min_i[..., 0], s_min_j[..., 0]) < np.minimum(s_max_i[..., 0], s_max_j[..., 0])
    # overlap_y = np.maximum(s_min_i[..., 1], s_min_j[..., 1]) < np.minimum(s_max_i[..., 1], s_max_j[..., 1])
    # aabb_mask = overlap_x & overlap_y
    
    # 优化：直接比较
    # i.min < j.max AND j.min < i.max
    aabb_mask = (s_min_i[..., 0] < s_max_j[..., 0]) & (s_min_j[..., 0] < s_max_i[..., 0]) & \
                (s_min_i[..., 1] < s_max_j[..., 1]) & (s_min_j[..., 1] < s_max_i[..., 1])
                
    # 4. 排除相邻线段和自身
    # 排除 i == j, i == j+1, i == j-1
    # 以及首尾相接: (0, N-1)
    
    # 创建一个 mask 矩阵，只保留我们需要检查的对 (i, j) 其中 j > i + 1
    # 并且排除 (0, N-1)
    
    # 我们可以使用 triu 来只取上三角，且 k=2 (排除对角线和次对角线)
    valid_mask = np.triu(np.ones((N, N), dtype=bool), k=2)
    
    # 特殊处理首尾 (0, N-1)
    valid_mask[0, N-1] = False
    
    # 最终需要检查的候选对
    candidates_mask = aabb_mask & valid_mask
    
    # 获取候选索引
    # indices: (K, 2) where K is number of candidate pairs
    ii, jj = np.where(candidates_mask)
    
    if len(ii) == 0:
        return 0
        
    # 5. 精确检测 (Vectorized Cross Product)
    # 取出对应的点
    # segment i: p1=P1[ii], p2=P2[ii]
    # segment j: p3=P1[jj], p4=P2[jj]
    
    p1 = P1[ii]
    p2 = P2[ii]
    p3 = P1[jj]
    p4 = P2[jj]
    
    # 向量叉乘函数
    def cross(a, b):
        return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        
    v12 = p2 - p1
    v13 = p3 - p1
    v14 = p4 - p1
    
    v34 = p4 - p3
    v31 = p1 - p3
    v32 = p2 - p3
    
    cp1 = cross(v12, v13)
    cp2 = cross(v12, v14)
    cp3 = cross(v34, v31)
    cp4 = cross(v34, v32)
    
    # 判断相交
    # 严格内部相交: (cp1 * cp2 < 0) & (cp3 * cp4 < 0)
    # 使用 epsilon 处理浮点误差
    EPS = 1e-10
    intersect = (cp1 * cp2 < -EPS) & (cp3 * cp4 < -EPS)
    
    return np.sum(intersect)

def assess_curve_quality(nodes):
    """
    评估曲线质量
    
    Args:
        nodes: (B, N, 2) numpy array, B个样本，每个样本N个点
        
    Returns:
        report: dict, 包含各项指标的统计
        scores: (B,) float, 综合评分 (0-100, 越高越好) - 这是一个示例性的评分
    """
    if not isinstance(nodes, np.ndarray):
        nodes = np.array(nodes)
        
    if len(nodes.shape) == 2:
        nodes = nodes[np.newaxis, :, :] # (1, N, 2)
        
    B, N, _ = nodes.shape
    print(f"Starting quality assessment for {B} curves with {N} nodes each...")
    
    # 1. 计算边长 (Edge Lengths)
    # ------------------------------------------------
    print("1. Calculating Edge Lengths...")
    next_nodes = np.roll(nodes, -1, axis=1)
    diffs = next_nodes - nodes
    edge_lengths = np.linalg.norm(diffs, axis=2) # (B, N)
    
    min_edge_lens = np.min(edge_lengths, axis=1)
    max_edge_lens = np.max(edge_lengths, axis=1)
    mean_edge_lens = np.mean(edge_lengths, axis=1)
    std_edge_lens = np.std(edge_lengths, axis=1)
    
    # 变异系数 (Coefficient of Variation)，衡量点分布均匀程度
    cv_edge_lens = std_edge_lens / (mean_edge_lens + 1e-8)
    print(f"   - Samples with large cv: {np.sum(cv_edge_lens > 1)}")

    # 极短边检测 (比如小于平均边长的 10%)
    short_edge_threshold_ratio = 0.1
    short_edges_count = np.sum(edge_lengths < (mean_edge_lens[:, None] * short_edge_threshold_ratio), axis=1)
    
    # 极长边检测
    long_edge_threshold_ratio = 10.0
    long_edges_count = np.sum(edge_lengths > (mean_edge_lens[:, None] * long_edge_threshold_ratio), axis=1)
    print(f"   - Average Mean Edge Length: {np.mean(mean_edge_lens):.6f}")
    print(f"   - Total short edges: {np.sum(short_edges_count)}")
    print(f"   - Samples with short edges (<10% mean): {np.sum(short_edges_count > 0)}")
    print(f"   - Total long edges: {np.sum(long_edges_count)}")
    print(f"   - Samples with long edges (>10x mean): {np.sum(long_edges_count > 0)}")

    # 2. 计算转角 (Turning Angles)
    # ------------------------------------------------
    print("2. Calculating Turning Angles...")
    # 向量归一化
    norms = edge_lengths[:, :, None]
    norms[norms == 0] = 1e-8 # 防止除零
    tangent_vectors = diffs / norms
    
    # 计算相邻向量的点积
    next_tangents = np.roll(tangent_vectors, -1, axis=1)
    dot_products = np.sum(tangent_vectors * next_tangents, axis=2)
    dot_products = np.clip(dot_products, -1.0, 1.0)
    
    # 角度 (弧度)
    # 0 表示直线，pi (180度) 表示折返
    turning_angles = np.arccos(dot_products) 
    turning_angles_deg = np.rad2deg(turning_angles)
    
    max_turning_angles = np.max(turning_angles_deg, axis=1)
    mean_turning_angles = np.mean(turning_angles_deg, axis=1)
    
    # 尖角检测
    sharp_angle_threshold = 30
    sharp_angles_count = np.sum(turning_angles_deg > sharp_angle_threshold, axis=1)
    
    # 折返检测 (比如大于 170 度)
    reversal_threshold = 170
    reversals_count = np.sum(turning_angles_deg > reversal_threshold, axis=1)

    print(f"   - Average Max Turning Angle: {np.mean(max_turning_angles):.2f} deg")
    print(f"   - Samples with sharp angles (>30): {np.sum(sharp_angles_count > 0)}")
    print(f"   - Samples with reversals (>170): {np.sum(reversals_count > 0)}")

    # 3. 自相交检测 (Self-Intersections)
    # ------------------------------------------------
    print("3. Calculating Self-Intersections (this might take a while)...")
    # 由于计算量大，对每个样本单独循环
    intersection_counts = []
    for i in tqdm(range(B)):
        # 如果点数太多，可能需要跳过或采样，这里假设 N 适中
        cnt = count_self_intersections(nodes[i])
        intersection_counts.append(cnt)

    print("")
    intersection_counts = np.array(intersection_counts)
    print(f"   - Samples with intersections: {np.sum(intersection_counts > 0)}")

    # 4. 综合评分与报告
    # ------------------------------------------------
    print("4. Generating Severity Scores...")

    severity_score = np.zeros(B)
    
    # 自相交非常严重
    severity_score += intersection_counts * 1000

    # 折返很严重
    severity_score += reversals_count * 500

    # 尖角中等严重
    severity_score += sharp_angles_count * 100

    # 极短边轻微严重 (可能导致数值不稳定)
    severity_score += short_edges_count * 5
    severity_score += long_edges_count * 5
    
    # 分布极不均匀
    cv_score = np.sum((cv_edge_lens > 1).astype(float) * (cv_edge_lens-1))*50
    severity_score += cv_score/B
    print(f"   - Average cv score: {cv_score/B:.3f}")
    print(f"   - Average Severity Score: {np.mean(severity_score):.3f}")
    


if __name__ == "__main__":

    data_path = "../../data/curve/"
    data_file_path = data_path+"/pcno_curve_data_1_1_5_2d_grad_log_panel2.npz"
    print("Loading data from ", data_file_path, flush = True)
    data = np.load(data_file_path)
    
    assess_curve_quality(data['nodes'])
