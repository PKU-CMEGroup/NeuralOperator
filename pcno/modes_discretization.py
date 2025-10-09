
import math
import numpy as np

from itertools import product



def _is_primitive(m_tuple):
    g = 0
    for c in m_tuple:
        g = math.gcd(g, abs(c))
        if g == 1:
            return True
    return g == 1  # 到这里一般是 False（全 0 已排除）

def _sign_representative(m_tuple):
    """
    返回是否保留该向量：规则=找到最后一个非零分量，必须 >0。
    全零不应调用这里。
    """
    for i in range(len(m_tuple) - 1, -1, -1):
        if m_tuple[i] != 0:
            return m_tuple[i] > 0
    return False

def discrete_half_ball_modes(
    n: int,
    k: int,
    scale: float = 0,
    directions_radius: int | None = None,
    use_primitive: bool = True,
    uniform_2d: bool = True,
    area_uniform: bool = False,
    num_dirs_2d: int | None = None,
    balance_rt: bool = True,
    min_dir_fraction: float = 0.1
) -> np.ndarray:
    """
    生成半球(半圆)内按径向分层的离散点集。

    新增:
      balance_rt: 使径向与切向点密度更均衡。方法：内层减少方向数量，
                  令每层方向数 ~ 基础方向数 * r^(n-1) (2D 即 ~ r)。
      min_dir_fraction: 最少方向比例 (避免内层方向过少)

    参数:
      n: 维度
      k: 径向层数 (r = 1..k)
      directions_radius: 构造方向用的半径(仅在整数格方向模式下使用)
      use_primitive: 是否只取本原方向(整数格模式下)
      uniform_2d: n=2 时是否使用均匀角度方向代替整数格方向
      area_uniform: 若为 True，径向层采用等面积/等体积分布 r_i = (i/k)^(1/n)
      num_dirs_2d: n=2 且 uniform_2d=True 时指定方向数；默认 max(8, 4*k)
      balance_rt: 是否平衡径向与切向密度
      min_dir_fraction: balance_rt=True 时每层方向下限比例

    返回:
      np.ndarray, shape = (N, n)
    """
    if n < 1 or k < 1:
        raise ValueError("n >= 1 且 k >= 1")

    # 基础方向集 (最大集合)
    if n == 2 and uniform_2d:
        M = num_dirs_2d if num_dirs_2d is not None else max(8, 4 * k)
        angles = (np.arange(M) + 0.5) / M * math.pi  # (0, π)
        base_dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    else:
        R = directions_radius if directions_radius is not None else k
        dirs_raw = []
        rng = range(-R, R + 1)
        for m in product(rng, repeat=n):
            if all(c == 0 for c in m):
                continue
            if sum(c * c for c in m) > R * R:
                continue
            if not _sign_representative(m):
                continue
            if use_primitive and (not _is_primitive(m)):
                continue
            v = np.array(m, dtype=float)
            v /= np.linalg.norm(v)
            dirs_raw.append(v)

        uniq_dirs = []
        seen = set()
        for d in dirs_raw:
            key = tuple(np.round(d, 12))
            if key not in seen:
                seen.add(key)
                uniq_dirs.append(d)
        base_dirs = np.vstack(uniq_dirs)

    # 径向层半径
    if area_uniform:
        rs = (np.arange(1, k + 1) / k) ** (1.0 / n)
    else:
        rs = np.arange(1, k + 1) / k

    points = [np.zeros(n, dtype=float)]

    if not balance_rt:
        # 原方案: 每层使用全部方向
        for r in rs:
            points.append(base_dirs * r)
    else:
        M_full = len(base_dirs)
        min_frac = min(1.0, max(0.0, min_dir_fraction))
        for r in rs:
            # 目标方向数：与半径尺度匹配，n 维表面积 ~ r^{n-1}
            frac = r ** (n - 1)
            frac = max(frac, min_frac)
            m_layer = max(1, int(round(M_full * frac)))
            if m_layer >= M_full:
                sel_dirs = base_dirs
            else:
                # 均匀抽取（保持角度均匀，而非随机）
                idx = (np.linspace(0, M_full - 1, m_layer)).astype(int)
                sel_dirs = base_dirs[idx]
            points.append(sel_dirs * r)

    modes = np.vstack(points)

    if scale != 0:
        norms = np.linalg.norm(modes, axis=1, keepdims=True)
        modes = modes * (norms ** scale)

    return modes

def nonlinear_scale(modes, scale=1.0):
    if scale>0:
        norms = np.linalg.norm(modes, axis=1)
        max_norm = norms.max()
        scaled_norms = (norms / max_norm) ** scale
        modes_scaled = modes * scaled_norms[:, np.newaxis]
        return modes_scaled
    return modes

if __name__ == "__main__":
    import os
    import torch
    import sys

    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))

    import numpy as np
    from timeit import default_timer
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from pcno.pcno import compute_Fourier_modes
    
    k_max = 20
    print(f'kmax = {k_max}')
    ndim = 2
    scale = 0
    modes = discrete_half_ball_modes(ndim, k_max, scale)*k_max*2*np.pi/5
    modes_ref = compute_Fourier_modes(ndim, [k_max,k_max], [5,5])
    print(modes.shape, modes_ref.shape)
    # print(modes, modes_ref)
    if ndim ==3:
        import pyvista as pv
        plotter = pv.Plotter()
        plotter.add_points(modes, color='blue', point_size=10, render_points_as_spheres=True)
        plotter.add_points(modes_ref, color='red', point_size=5, render_points_as_spheres=True)
        plotter.show()

    elif ndim ==2:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6,6))
        plt.scatter(modes_ref[:,0], modes_ref[:,1], color='red', s=10)
        plt.scatter(modes[:,0], modes[:,1])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True)
        plt.show()