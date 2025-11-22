import numpy as np
from typing import List, Tuple, Union
from scipy.special import comb, erf

from .geometry import PanelGeometry 
from typing import List, Tuple


def compute_kernel(point: np.ndarray, geometry: PanelGeometry, kernel_args: Tuple):
    """
    计算给定点对几何体所有面板的势函数的影响系数。
    
    Args:
        point: 计算影响的目标点坐标，形状为(2,)
        geometry: 面板几何体对象（顺时针方向）
        kernel_args: 核类型以及参数
            "laplace_single_layer"   : k(x,y) = ln(|x-y|)/2pi
            "laplace_double_layer"   : k(x,y) = (x-y)ny /(2pi|x-y|^2)   其中ny是外法向
            "smoothed_laplace_double_layer"   : k(x,y) = (x-y)ny /(2pi(|x-y|^2 + epsilon^2))   其中ny是外法向, 参数eps
            "clipped_laplace_double_layer"    : k(x,y) = (x-y)ny /(2pi(|x-y|^2))  or 0 当|(x-y)ny| < eps，  其中ny是外法向, 参数eps
            "polynomial"             : k(x,y) = (xy + c)^p              参数 p, c
            "gaussian"               : k(x,y) = exp(-|x-y|^2/(2σ^2))    参数 σ
            
        
    Returns:
        - kernel: 对每一个panel p, 计算 k(x,y)

    """

    middle_points = geometry.panel_midpoints
    panel_cosines = geometry.panel_cosines
    panel_sines = geometry.panel_sines
    normals = np.column_stack((-panel_sines, panel_cosines))
    diff = point - middle_points

    
    kernel_type = kernel_args[0]
    if kernel_type == "laplace_single_layer": 
        # k(x,y) = ln(|x-y|)/2pi
        kernel = np.log(np.linalg.norm(diff, axis=1)) / (2*np.pi)
        kernel = np.nan_to_num(kernel, nan=0.0, posinf=0.0, neginf=0.0)
    
    elif kernel_type == "laplace_double_layer": 
        # k(x,y) = (x-y)ny /(2pi|x-y|^2)   其中ny是外法向
        kernel = np.sum(diff * normals, axis=1) / (2*np.pi * np.linalg.norm(diff, axis=1)**2)
        kernel = np.nan_to_num(kernel, nan=0.0, posinf=0.0, neginf=0.0) # TODO this is delta function
    elif kernel_type == "smoothed_laplace_double_layer": 
        # k(x,y) = (x-y)ny /(2pi(|x-y|^2 + eps^2))   其中ny是外法向
        eps = kernel_args[1]
        kernel = np.sum(diff * normals, axis=1) / (2*np.pi * (np.linalg.norm(diff, axis=1)**2 + eps**2))
    elif kernel_type == "clipped_laplace_double_layer": 
        # k(x,y) = (x-y)ny /(2pi|x-y|^2)  or 0 当 |(x-y)| < eps, 其中ny是外法向
        eps = kernel_args[1]
        kernel = np.sum(diff * normals, axis=1) / (2*np.pi * np.linalg.norm(diff, axis=1)**2)
        kernel[np.sum(diff * diff, axis=1) < eps**2] = 0.0
        kernel = np.nan_to_num(kernel, nan=0.0, posinf=0.0, neginf=0.0) # TODO this is delta function
    
    elif kernel_type == "polynomial": 
        # k(x,y) = (xy + c)^p 
        p, c = kernel_args[1], kernel_args[2] 
        kernel = (middle_points @ point + c)** p
        
    elif kernel_type == "gaussian":
        # k(x,y) = exp(-|x-y|^2/(2σ^2))
        sigma = kernel_args[1]
        kernel = np.exp(-np.sum(diff * diff, axis=1) / (2*sigma**2))

    else:
        raise ValueError(f"错误: kernel_type {kernel_type} 没有定义") 

    return kernel


def compute_kernel_influence_coefficients(point: np.ndarray, geometry: PanelGeometry, kernel_args: Tuple):
    """
    计算给定点对几何体所有面板的势函数的影响系数。
    
    Args:
        point: 计算影响的目标点坐标，形状为(2,)
        geometry: 面板几何体对象（顺时针方向）
        kernel_args: 核类型以及参数
            "laplace_single_layer"   : k(x,y) = ln(|x-y|)/2pi
            "laplace_double_layer"   : k(x,y) = (x-y)ny /(2pi|x-y|^2)   其中ny是外法向
            "smoothed_laplace_double_layer"   : k(x,y) = (x-y)ny /(2pi(|x-y|^2 + epsilon^2))   其中ny是外法向, 参数eps
            "clipped_laplace_double_layer"    : k(x,y) = (x-y)ny /(2pi(|x-y|^2))  or 0 当|(x-y)ny| < eps，  其中ny是外法向, 参数eps
            "polynomial"             : k(x,y) = (xy + c)^p              参数 p, c
            "gaussian"               : k(x,y) = exp(-|x-y|^2/(2σ^2))    参数 σ
            
        
    Returns:
        - phi: 对每一个panel p, 计算 \int_p k(x,y) dy

    """
        
    x0, y0 = point
    vertices = geometry.vertices
    panel_lengths = geometry.panel_lengths
    panel_cosines = geometry.panel_cosines
    panel_sines = geometry.panel_sines

    # 计算点到所有顶点的距离
    r_lengths = np.sqrt((x0 - vertices[:,0])**2+(y0 - vertices[:,1])**2)
    
    if np.min(r_lengths) <= 1.0e-10:
        raise ValueError(f"错误: point 和面板顶点重合 (距离={np.min(r_lengths)})， 网格有交叉")
    
    # 转换到面板局部坐标系
    x0_stars = panel_cosines*(x0 - vertices[:-1,0]) + panel_sines*(y0 - vertices[:-1,1])
    y0_stars = -panel_sines*(x0 - vertices[:-1,0]) + panel_cosines*(y0 - vertices[:-1,1])
    
    # 当 y0_stars = 0 且 x0_stars = l/2 时，点与面板共线我，们有
    collinear_mask = np.isclose(np.abs(y0_stars), 0.0, atol=1e-10, rtol=1e-10) & np.isclose(np.abs(x0_stars - panel_lengths/2.0), 0.0, atol=1e-10, rtol=1e-10)
    
    kernel_type = kernel_args[0]
    if kernel_type == "laplace_single_layer": 
        # k(x,y) = ln(|x-y|)/2pi
        phi = ( (panel_lengths  - x0_stars)*np.log(r_lengths[1:])  + x0_stars*np.log(r_lengths[0:-1]) - panel_lengths) /(2*np.pi)
        phi[~collinear_mask] += ((y0_stars * (np.arctan((panel_lengths  - x0_stars) / y0_stars) + np.arctan(x0_stars / y0_stars)) ) /(2*np.pi))[~collinear_mask]
    elif kernel_type == "laplace_double_layer": 
        # k(x,y) = (x-y)ny /(2pi|x-y|^2)   其中ny是外法向
        phi = ((np.arctan((panel_lengths  - x0_stars) / y0_stars) + np.arctan(x0_stars / y0_stars))  /(2*np.pi))
        phi[collinear_mask] = 1/2.0
    elif kernel_type == "smoothed_laplace_double_layer": 
        # k(x,y) = (x-y)ny /(2pi(|x-y|^2 + eps^2))   其中ny是外法向
        eps = kernel_args[1]
        y0_stars_eps = np.sqrt(y0_stars**2 + eps**2)
        phi = y0_stars/y0_stars_eps * (np.arctan((panel_lengths  - x0_stars) / y0_stars_eps) + np.arctan(x0_stars / y0_stars_eps))  /(2*np.pi)
        phi[collinear_mask] = 1/2.0
    elif kernel_type == "clipped_laplace_double_layer": 
        # k(x,y) = (x-y)ny /(2pi(|x-y|^2))                   其中ny是外法向
        #        = 0                         |x-y| < eps 其中ny是外法向
        eps = kernel_args[1]
        phi = ((np.arctan((panel_lengths  - x0_stars) / y0_stars) + np.arctan(x0_stars / y0_stars))  /(2*np.pi))
        phi[np.sqrt(y0_stars**2 + (x0_stars - panel_lengths/2)**2) < eps] = 0.0 
        # TODO this is an approximation
        phi[collinear_mask] = 1/2.0
    elif kernel_type == "polynomial": 
        # k(x,y) = (xy + c)^p = = (x (y_s + t(y_e - y_s) + c)^p = (a + bt + c)^p   t in (0,1)
        p, c = kernel_args[1], kernel_args[2] 
        a = x0 * vertices[:-1,0] + y0 * vertices[:-1,1]
        b = x0 * (vertices[1:,0] - vertices[:-1,0]) + y0 * (vertices[1:,1] - vertices[:-1,1])
        phi = 1/(p+1) * panel_lengths / b * ((a+b+c)**(p+1) - (a+c)**(p+1))
    elif kernel_type == "gaussian":
        # k(x,y) = exp(-|x-y|^2/(2σ^2))
        sigma = kernel_args[1]
        phi = sigma * np.sqrt(np.pi/2) * np.exp(-y0_stars**2/(2*sigma**2)) * (erf((panel_lengths  - x0_stars)/(np.sqrt(2) * sigma)) -  erf((- x0_stars)/(np.sqrt(2) * sigma)))
    else:
        raise ValueError(f"错误: kernel_type {kernel_type} 没有定义")
    
        

    return phi


def compute_all_edge_kernel_influence_coefficients(velocity_geometry, source_geometry, kernel_args: Tuple):
    """
    计算所有源面板对所有速度控制点的影响系数矩阵。
    这个函数只是用于后处理和测试，在 panel method 里并不会使用
    
    Args:
        velocity_geometry: 速度控制点所在的面板几何体
        source_geometry: 源项所在的面板几何体
        kernel_args: kernel的参数
        
    Returns:
        - influence_coefficients: 影响系数矩阵，形状为(n_edges, n_panels)
    """
    n_edges = velocity_geometry.n_panels
    n_panels = source_geometry.n_panels

    influence_coefficients = np.zeros((n_edges, n_panels))

    for i in range(n_edges):
        point = velocity_geometry.panel_midpoints[i]
        cosine = velocity_geometry.panel_cosines[i]
        sine = velocity_geometry.panel_sines[i]
        influence_coefficients[i,:] = compute_kernel_influence_coefficients(point, source_geometry, kernel_args)
    
    return influence_coefficients


def compute_kernel_integral(geometries: List[PanelGeometry], strengths: List[np.ndarray], kernel_args: Tuple):
    """
    计算源 strengths(x)在曲线 geometries 上的积分 
    phi(x) = \int sigma(x) k(x,y) dy
    这个函数只是用于后处理和测试，在 panel method 里并不会使用
    
    Args:
        geometries: 面板几何体列表
        strengths: 面板中心源强度
        kernel_args: kernel的参数
        
    Returns:
        - phi(x): 势函数      phi(x) = \int sigma(x) k(x,y) dy, k(x,y)=ln(|x-y|)/2pi
    """
    
    n_geometries = len(geometries)
    influence_coefficients = [[None] * n_geometries for _ in range(n_geometries)]
    
    for i, vel_geo in enumerate(geometries):
        for j, src_geo in enumerate(geometries):
            influence_coefficients[i][j] = compute_all_edge_kernel_influence_coefficients(vel_geo, src_geo, kernel_args)

    # 根据源强度求积分
    solutions = [np.zeros(geometry.n_panels) for geometry in geometries]
    for i, vel_geo in enumerate(geometries):  

        # 各项贡献
        for j in range(n_geometries):
            solutions[i]  += influence_coefficients[i][j]@strengths[j] 
            
    return solutions 
   




def compute_all_edge_kernel_influence_coefficients_approximation(velocity_geometry, source_geometry, kernel_args: Tuple):
    """
    计算所有源面板对所有速度控制点的影响系数矩阵。
    这个函数只是用于后处理和测试，在 panel method 里并不会使用
    
    Args:
        velocity_geometry: 速度控制点所在的面板几何体
        source_geometry: 源项所在的面板几何体
        kernel_args: kernel的参数
        
    Returns:
        - influence_coefficients: 影响系数矩阵，形状为(n_edges, n_panels)
    """
    n_edges = velocity_geometry.n_panels
    n_panels = source_geometry.n_panels

    influence_coefficients = np.zeros((n_edges, n_panels))

    for i in range(n_edges):
        point = velocity_geometry.panel_midpoints[i]
        cosine = velocity_geometry.panel_cosines[i]
        sine = velocity_geometry.panel_sines[i]
        # influence_coefficients[i,:] = compute_kernel_influence_coefficients(point, source_geometry, kernel_args)
        influence_coefficients[i,:] = compute_kernel(point, source_geometry, kernel_args) * source_geometry.panel_lengths
    
    return influence_coefficients

def compute_kernel_approximation(geometries: List[PanelGeometry], strengths: List[np.ndarray], kernel_args: Tuple):
    """
    计算源 strengths(x)在曲线 geometries 上的积分 
    phi(x) = \int sigma(x) k(x,y) dy
    这个函数只是用于后处理和测试，在 panel method 里并不会使用
    
    Args:
        geometries: 面板几何体列表
        strengths: 面板中心源强度
        kernel_args: kernel的参数
        
    Returns:
        - phi(x): 势函数      phi(x) = \int sigma(x) k(x,y) dy, k(x,y)=ln(|x-y|)/2pi
    """
    
    n_geometries = len(geometries)
    influence_coefficients = [[None] * n_geometries for _ in range(n_geometries)]
    
    for i, vel_geo in enumerate(geometries):
        for j, src_geo in enumerate(geometries):
            influence_coefficients[i][j] = compute_all_edge_kernel_influence_coefficients_approximation(vel_geo, src_geo, kernel_args)

    # 根据源强度求积分
    solutions = [np.zeros(geometry.n_panels) for geometry in geometries]
    for i, vel_geo in enumerate(geometries):  

        # 各项贡献
        for j in range(n_geometries):
            solutions[i]  += influence_coefficients[i][j]@strengths[j] 
            
    return solutions 
   

